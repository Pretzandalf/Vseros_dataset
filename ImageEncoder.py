import torch.nn as nn
import timm 
import torchvision


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector.
    """

    def __init__(self, model_name="resnet50", pretrained=True, trainable=False ):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained = pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(2048, 2048)

    def forward(self, x):
        x = x.float()
        return self.model(x)
