import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()

        self.model = DistilBertModel.from_pretrained(model_name)

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask, output_hidden_states=True)
        return out[0][:,self.target_token_idx,:]