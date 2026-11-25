"""Synthetic reconstruction layout, masking and backbone-transfer checks."""
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tiny_vit3d import TinyViT3D
from vit_reconstruction import ReconstructionViT,unpatchify,foreground_mask,transfer_backbone
from train_wholebrain import step


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_patch_layout_roundtrip(self):
        volume=torch.arange(64**3,dtype=torch.float32).reshape(1,1,64,64,64)
        patches=volume[:,0].unfold(1,16,16).unfold(2,16,16).unfold(3,16,16).contiguous().reshape(1,64,4096)
        self.assertTrue(torch.equal(volume,unpatchify(patches)))
        with self.assertRaises(ValueError): unpatchify(torch.zeros(1,63,4096))

    def test_mask_is_reproducible_and_excludes_background(self):
        target=torch.ones(1,1,64,64,64); target[:,:,:16]=0
        first=foreground_mask(target,torch.Generator().manual_seed(42))
        second=foreground_mask(target,torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(first,second))
        self.assertFalse(first[:,:,:16].any())
        self.assertTrue(first.any())
        with self.assertRaises(ValueError): foreground_mask(torch.zeros_like(target),torch.Generator().manual_seed(42))

    def test_gradient_and_transfer_preserve_seeded_head(self):
        torch.manual_seed(42)
        model=ReconstructionViT()
        target=torch.randn(1,1,64,64,64)
        mask=foreground_mask(target,torch.Generator().manual_seed(43))
        pred=model(target.masked_fill(mask,0))
        self.assertEqual(pred.shape,target.shape)
        before=model.backbone.patch.weight.detach().clone()
        step((pred[mask]-target[mask]).square().mean(),model,torch.optim.Adam(model.parameters(),lr=.001))
        self.assertFalse(torch.equal(before,model.backbone.patch.weight))
        torch.manual_seed(7); random=TinyViT3D()
        torch.manual_seed(7); transferred=TinyViT3D()
        transfer_backbone(transferred,model.backbone.state_dict())
        for k,v in random.head.state_dict().items():
            self.assertTrue(torch.equal(v,transferred.head.state_dict()[k]))
        for k,v in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(v,transferred.state_dict()[k]))
        with self.assertRaises(ValueError): transfer_backbone(transferred,{})


if __name__=='__main__': unittest.main()
