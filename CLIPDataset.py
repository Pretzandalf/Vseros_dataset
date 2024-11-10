import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import typing as tp
import cv2


class CLIPDataset(Dataset):
    def __init__(self, image_filenames, captions, tokenizer):
        """
        :image_path -- path to images
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        :tokenizer -- LM Tokenizer 
        """
        self.max_tokenizer_length = 200
        self.truncation = True
        self.padding = True
        self.image_path = "/home/pret/PycharmProjects/pythonNetWork/Datasets/NTO_clip_classification/train/"
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.tokenizer = tokenizer
        self.encoded_captions = self.tokenizer(self.captions, padding=True, add_special_tokens=True)
        #self.encoded_captions = [self.tokenizer(cap) for cap in self.captions]
        self.transforms = T.Resize([224, 244]) # This should do.

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[torch.Tensor, str]]:

        """
        This one should return dict(keys=['image', 'caption'], value=[Image, Caption])
        """

        item = {
            key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()
        }

        item['image'] = torch.from_numpy(cv2.resize(cv2.imread(self.image_path + self.image_filenames[idx], cv2.IMREAD_UNCHANGED), (244, 244)))
        #item['image'] = self.transforms(torch.from_numpy(cv2.imread(self.image_path + index)))

        item['caption'] = self.captions[idx]


        return item


    def __len__(self):
        return len(self.captions)