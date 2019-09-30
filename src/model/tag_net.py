import torch.nn as nn
import torch.nn.functional as F
import torch


class TagNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embed(x)
