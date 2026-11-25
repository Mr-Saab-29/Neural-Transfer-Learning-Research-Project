"""Small full-volume 3D ViT: 64 non-overlapping 16-cubed patches."""
import math
import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, width=48, heads=3):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)).softmax(dim=-1)
        mixed = (attention @ v).transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.projection(mixed)
        return x + self.mlp(self.norm2(x))


class TinyViT3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv3d(1, 48, kernel_size=16, stride=16)
        self.position = nn.Parameter(torch.empty(1, 64, 48))
        nn.init.normal_(self.position, std=.02)
        self.blocks = nn.Sequential(AttentionBlock(), AttentionBlock())
        self.norm = nn.LayerNorm(48)
        self.head = nn.Linear(48, 1)

    def forward(self, image):
        if image.ndim != 5 or tuple(image.shape[1:]) != (1, 64, 64, 64):
            raise ValueError('Expected B x 1 x 64 x 64 x 64 input')
        tokens = self.patch(image).flatten(2).transpose(1, 2) + self.position
        return self.head(self.norm(self.blocks(tokens)).mean(dim=1)).flatten()
