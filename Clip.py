import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import pandas as pd
import timm
import typing as tp
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from CLIPDataset import CLIPDataset
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from ImageEncoder import ImageEncoder
from ProjectionHead import ProjectionHead
from TextEncoder import TextEncoder


class CLIP(nn.Module):
    def __init__(self, image_embedding=2048, text_embedding=768, temp=1.0, batch_size=32):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projections = ProjectionHead(image_embedding)
        self.text_projections = ProjectionHead(text_embedding)
        self.temp = temp
        self.normalize = LayerNorm(256)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        """
        :batch: dict of images and text
        Here is what you should do:
        1) extract image and text features from batch
        2) project features into projection space (small latent space)
        3) compute cosine similarity with temperature this will be your logits
        4) compute "true" logits (eg. cosine similarity between images and images, text and text)
        5) create targets by averaging similarities from step above (do not forget about temperature)
        6) compute mean loss (see paper)
        7) return loss

        Overall: read paper.
        """

        batch_size = batch[0].size()[0]

        Bert_out = self.text_encoder(batch[0].to(device), batch[1].to(device))
        Resnet_out = self.image_encoder(batch[2].transpose(1, 3).to(device))

        text_embeding = self.text_projections(Bert_out)
        image_embeding = self.image_projections(Resnet_out)

        similarity = torch.matmul(text_embeding, image_embeding.T) * torch.exp(torch.tensor(self.temp))

        labels = torch.arange(batch_size).to(device)

        img_loss = self.cross_entropy_loss(similarity, labels)
        tex_loss = self.cross_entropy_loss(similarity.T, labels)

        loss = (img_loss + tex_loss) / 2

        return loss


def CE(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss

PATH_TO_IMAGES = '/home/pret/PycharmProjects/pythonNetWork/Datasets/NTO_clip_classification/translated_train.csv'
def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{PATH_TO_IMAGES}")
    dataframe["id"] = np.array(list(dataframe.index))
    max_id = dataframe["id"].max() + 1
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)

    return train_dataframe, valid_dataframe


def collate_fn(batch):
    images_tensors = torch.tensor(batch[0]['image']).unsqueeze(0)
    text_tensors = torch.tensor(batch[0]['input_ids']).unsqueeze(0)
    mask_tensors = torch.tensor(batch[0]['attention_mask']).unsqueeze(0)

    for i in range(1, len(batch)):
        images_tensors = torch.cat((images_tensors, batch[i]['image'].unsqueeze(0)), dim=0)
        text_tensors = torch.cat((text_tensors, batch[i]['input_ids'].unsqueeze(0)), dim=0)
        mask_tensors = torch.cat((mask_tensors, batch[i]['attention_mask'].unsqueeze(0)), dim=0)

    out = [text_tensors, mask_tensors, images_tensors]

    return out


def build_loaders(dataframe, tokenizer, mode):
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=1,
        shuffle=True if mode == "train" else False,
        collate_fn=collate_fn
    )
    return dataloader, dataset


class AvgMeter:
    def __init__(self, name="CrossEntropyLoss"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

    def __format__(self, formatspec):
        text = f"{self.name}: {format(self.avg, formatspec)}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    for batch in tqdm(train_loader, desc="Training", total=len(train_loader)):
        # batch = {key: value.to(device) for key, value in batch.items() if key != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        loss_meter.update(loss.item(), batch[2].size()[0])
    return loss_meter



@torch.no_grad()
def validate(model, validation_loader):
    loss_meter = AvgMeter()
    for batch in tqdm(validation_loader, desc="Validating", total=len(validation_loader)):
        #       batch = {key: value.to(device) for key, value in batch.items() if key != "caption"}
        loss = model(batch)
        loss_meter.update(loss.item(), batch[2].size()[0])
    return loss_meter


import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1


def procedure():
    train_df, validation_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader, _ = build_loaders(train_df, tokenizer, mode="train")
    val_loader, _ = build_loaders(validation_df, tokenizer, mode="valid")
    model = CLIP().to(device)
    params = [{"params": model.image_encoder.parameters()},
              {"params": model.text_encoder.parameters()},
              {"params": itertools.chain(model.image_projections.parameters(),
                                         model.text_projections.parameters())}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.8)
    step = "epoch"
    for epoch in range(EPOCH):
        print(f"Epoch: {epoch}. Train and Validation in progress...")
        model.train()
        train_loss = train(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        val_loss = validate(model, val_loader)

        lr_scheduler.step(val_loss.avg)
        print(f"Epoch: {epoch},", end="\n")
        print(f"Train loss: {train_loss:0.3f}", end="\n")
        print(f"Validation loss: {val_loss:0.3f}")
    return model


if __name__ == '__main__':

    model = procedure()

    import matplotlib.pyplot as plt


    @torch.inference_mode()
    def get_image_embeddings(valid_df, model):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        valid_loader, _ = build_loaders(valid_df, tokenizer, mode="valid")
        valid_image_embeddings = []
        for batch in tqdm(valid_loader, desc="Getting embeddings", total=len(valid_loader)):
#            batch = {key: value.to(device) for key, value in batch.items() if key != "caption"}
            image_features = model.image_encoder(batch[2].permute(0, 3, 1, 2).to(device)).to(device)
            image_embeddings = model.image_projections(image_features)
            valid_image_embeddings.append(image_embeddings)
        return torch.cat(valid_image_embeddings)


    @torch.inference_mode()
    def find_match(model, image_embeddings, text, image_filenames, num_examples=4):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        text_encoded = tokenizer([text])
        batch = {key: torch.tensor(value).to(device) for key, value in text_encoded.items()}

        text_features = model.text_encoder(batch["input_ids"], batch["attention_mask"])
        text_embeddings = model.text_projections(text_features)

        norm_image_embeddings = nn.functional.normalize(image_embeddings, p=2, dim=-1)
        norm_text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=-1)

        similarity = norm_text_embeddings @ norm_image_embeddings.T

        ans, ans_index = torch.topk(similarity.squeeze(0), num_examples * 5)
        match = [image_filenames[index] for index in ans_index[::5]]
        fig, ax = plt.subplots(int(num_examples / 2), int(num_examples / 2), figsize=(10, 10))
        for m, a in zip(match, ax.flatten()):
            PATH_TO_IMAGES = "C/home/pret/PycharmProjects/pythonNetWork/Datasets/NTO_clip_classification/train/"
            image = Image.open(f"{PATH_TO_IMAGES}" + f"/{m}")
            image = image.convert("RGB")
            a.imshow(image)
            a.axis("off")
        plt.show()


    PATH_TO_IMAGES = '/home/pret/PycharmProjects/pythonNetWork/Datasets/NTO_clip_classification/translated_train.csv'
    _, valid_df = make_train_valid_dfs()

    if __name__ == '__main__':
        image_embeddings = get_image_embeddings(valid_df, model)
        find_match(model, image_embeddings, "dogs", valid_df["image"].values)
