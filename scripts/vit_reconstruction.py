"""Masked voxel reconstruction using the existing ViT backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from tiny_vit3d import TinyViT3D


def unpatchify(patches):
    if patches.ndim != 3 or tuple(patches.shape[1:]) != (64, 4096):
        raise ValueError('Expected B x 64 x 4096 reconstructed patches')
    return patches.reshape(-1,4,4,4,16,16,16).permute(0,1,4,2,5,3,6).reshape(-1,1,64,64,64)


def foreground_mask(target, generator):
    coarse = (torch.rand((target.shape[0],1,8,8,8),generator=generator)<.2).float()
    hidden = F.interpolate(coarse,size=(64,64,64),mode='nearest').bool() & (target!=0)
    if not hidden.any():
        raise ValueError('No masked foreground')
    return hidden


class ReconstructionViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=TinyViT3D()
        self.backbone.head=nn.Identity()
        self.decoder=nn.Linear(48,4096)

    def forward(self,image):
        tokens=self.backbone.patch(image).flatten(2).transpose(1,2)+self.backbone.position
        features=self.backbone.norm(self.backbone.blocks(tokens))
        return unpatchify(self.decoder(features))


def transfer_backbone(model,state):
    expected={k for k in model.state_dict() if not k.startswith('head.')}
    if set(state)!=expected or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Invalid or incompatible pretrained backbone')
    full=model.state_dict()
    full.update(state)
    model.load_state_dict(full,strict=True)
