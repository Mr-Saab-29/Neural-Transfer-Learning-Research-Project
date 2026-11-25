"""Synthetic ViT coverage, gradients, checkpoints and deterministic updates."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tiny_vit3d import TinyViT3D
from train_wholebrain import step
from followup_common import save_torch


class TinyViTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shape_gradients_and_position(self):
        torch.manual_seed(42)
        model=TinyViT3D()
        image=torch.randn(2,1,64,64,64)
        self.assertEqual(tuple(model(image).shape),(2,))
        model(image).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.position.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):
            model(torch.zeros(1,1,32,32,32))

    def test_patches_cover_both_volume_corners(self):
        model=TinyViT3D()
        with torch.no_grad():
            model.patch.weight.fill_(1)
            model.patch.bias.zero_()
        image=torch.zeros(1,1,64,64,64)
        image[0,0,0,0,0]=1
        image[0,0,-1,-1,-1]=2
        patches=model.patch(image)
        self.assertEqual(tuple(patches.shape),(1,48,4,4,4))
        self.assertEqual(patches[0,0,0,0,0].item(),1)
        self.assertEqual(patches[0,0,-1,-1,-1].item(),2)

    def test_optimizer_resume_matches_next_update(self):
        torch.manual_seed(7)
        model=TinyViT3D()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        image=torch.randn(1,1,64,64,64)
        optimizer.zero_grad()
        step(model(image).square().mean(),model,optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt'
            save_torch(path,dict(weights=model.state_dict(),optimizer=optimizer.state_dict()))
            state=torch.load(path,map_location='cpu',weights_only=True)
            restored=TinyViT3D()
            restored.load_state_dict(state['weights'])
            opt2=torch.optim.Adam(restored.parameters(),lr=.001)
            opt2.load_state_dict(state['optimizer'])
            self.assertTrue(torch.equal(model(image),restored(image)))
            for network,opt in ((model,optimizer),(restored,opt2)):
                opt.zero_grad()
                step(network(image).square().mean(),network,opt)
            self.assertTrue(torch.equal(model(image),restored(image)))


if __name__=='__main__':
    unittest.main()
