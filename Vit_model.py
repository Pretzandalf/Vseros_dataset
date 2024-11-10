from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import os
import wandb
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torcheval.metrics.functional import multiclass_f1_score
import pandas as pd
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VITdataset(Dataset):
    def __init__(self, images_paths, images_names, images_indxes, trainable):
        self.images_paths = images_paths
        self.images_names = images_names
        self.images_indxes = images_indxes
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        image_mean, image_std = self.processor.image_mean, self.processor.image_std
        normalize = v2.Normalize(mean=image_mean, std=image_std)

        if trainable == True:
            self.transform = v2.Compose([
                v2.Resize((self.processor.size["height"], self.processor.size["width"])),
                v2.RandomHorizontalFlip(0.4),
                v2.RandomVerticalFlip(0.1),
                v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
                v2.RandomApply(transforms=[v2.ColorJitter(brightness=.3, hue=.1)], p=0.3),
                v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
                v2.ToTensor(),
                normalize
            ])
        elif trainable == False:
            self.transform = v2.Compose([
                v2.Resize((self.processor.size["height"], self.processor.size["width"])),
                v2.ToTensor(),
                normalize
            ])

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_paths, self.images_names[idx][0]))
        image = self.transform(image)
        return image, self.images_indxes[idx]

    def __len__(self):
        return len(self.images_indxes)



class Head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.gelu1 = nn.GELU()

        self.linear2 = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu1(x)

        x = self.linear2(x)

        return x

wandb.login(key = 'bb574f54db03f89674e1dba7770189c8f56e5a26')
wandb.init(project='vit-image-classification')

class VIT():
    def __init__(self,):
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.model.classifier = Head(self.model.classifier.in_features, 10)
        self.model = self.model.to(device)

    def train(self, dataset, validation,  epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion = nn.CrossEntropyLoss()
        lambda_lr = lambda epoch: 0.4 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        for epoch in range(epochs):
            self.model.train()
            for images, targets in tqdm(dataset, desc='Training', colour="cyan"):
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                model_output = self.model(images).logits
                loss = criterion(model_output, targets)
                loss.backward()
                optimizer.step()

                wandb.log({"loss": loss})
            scheduler.step()

            self.model.eval()
            F1_sum = []
            with torch.no_grad():
                for images, targets in tqdm(validation, desc='Validation', colour="green"):
                    images = images.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    model_output = self.model(images).logits
                    pred_class = torch.argmax(model_output, dim=1)

                    F1_metric = multiclass_f1_score(pred_class, targets, num_classes=10)
                    F1_sum.append(F1_metric.item())

            F1_sum = sum(F1_sum)/len(F1_sum)
            wandb.log({"F1_metric": F1_sum})

            wandb.log({"epoch": epoch + 1})

        PATH = '/home/pret/PycharmProjects/pythonNetWork/NTO_image_classification/Models/Vit_5.pt'
        torch.save(self.model.state_dict(), PATH)

    def predict(self, path_to_images):
        names = os.listdir(path_to_images)

        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        image_mean, image_std = processor.image_mean, processor.image_std
        normalize = v2.Normalize(mean=image_mean, std=image_std)

        transform = v2.Compose([
            v2.Resize((processor.size["height"], processor.size["width"])),
            v2.ToTensor(),
            normalize
        ])
        model_result = []
        for name in tqdm(names, desc='Prediction', colour="red"):
            image = Image.open(os.path.join(path_to_images, name))
            image = transform(image)
            image = torch.unsqueeze(image, 0)
            image = image.to(device)
            pred_class = torch.argmax(self.model(image).logits).item()
            model_result.append(pred_class)

        output = pd.DataFrame({
            'image_name': names,
            'predicted_class': model_result
        })

        output.to_csv('/home/pret/PycharmProjects/pythonNetWork/NTO_image_classification/Submission/submission.csv', index=False)

path = '/home/pret/PycharmProjects/pythonNetWork/NTO_image_classification/Dataset/train'

data_idx = pd.read_csv('/home/pret/PycharmProjects/pythonNetWork/NTO_image_classification/Dataset/train.csv')
img_to_class = {row['image_name']: row['class_id'] for _, row in data_idx.iterrows()}
class_to_name = {row['class_id']: row['unified_class'] for _, row in data_idx.iterrows()}
images_names = data_idx['image_name'].values
imagest_target = [img_to_class[name] for name in images_names]
X_train, X_test, y_train, y_test = train_test_split(images_names.reshape(-1,1), imagest_target, test_size=0.2, stratify=imagest_target, random_state=42)


train = VITdataset(path, X_train, y_train, trainable = True)
val = VITdataset(path, X_test, y_test, trainable = False)

train = DataLoader(train, batch_size=32, shuffle=True)
val = DataLoader(val, batch_size=16, shuffle=False)

epochs = 5

Vit = VIT()
Vit.train(train, val, epochs)

pred_path = '/home/pret/PycharmProjects/pythonNetWork/NTO_image_classification/Dataset/test'
Vit.predict(pred_path)