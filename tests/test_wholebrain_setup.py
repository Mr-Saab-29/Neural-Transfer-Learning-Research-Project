"""Synthetic checks for the new CNN, input coverage, and resumable checkpoints."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import followup_common as common
import train_wholebrain as training


class WholeBrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shapes_gradients_and_transfer(self):
        torch.manual_seed(42)
        image = torch.randn(1, 1, 64, 64, 64)
        model = common.WholeBrainCNN()
        prediction = model(image)
        self.assertEqual(tuple(prediction.shape), (1,))
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        reconstruction = common.ReconstructionCNN()
        self.assertEqual(tuple(reconstruction(image).shape), tuple(image.shape))
        head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
        model.encoder.load_state_dict(reconstruction.encoder.state_dict())
        self.assertTrue(all(torch.equal(v, model.head.state_dict()[k]) for k, v in head_before.items()))

    def test_full_coverage_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = torch.zeros(1, 128, 128, 128)
            image[0, 0, 0, 0] = 8
            image[0, -1, -1, -1] = 16
            torch.save(image, root / 'tensor.pt')
            (root / 'metadata.json').write_text('{}')
            row = dict(tensor_path='tensor.pt', tensor_sha256=common.digest(root / 'tensor.pt'),
                       metadata_sha256=common.digest(root / 'metadata.json'))
            with patch.object(common, 'ROOT', root):
                actual = common.whole_image(row)
                self.assertEqual(tuple(actual.shape), (1, 64, 64, 64))
                self.assertEqual(actual[0, 0, 0, 0].item(), 1)
                self.assertEqual(actual[0, -1, -1, -1].item(), 2)
                (root / 'metadata.json').write_text('changed')
                with self.assertRaises(ValueError):
                    common.whole_image(row)

    def test_resume_rejects_different_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            torch.manual_seed(7)
            model = common.WholeBrainCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            image = torch.randn(1, 1, 64, 64, 64)
            optimizer.zero_grad()
            training.step(model(image).square().mean(), model, optimizer)
            training.checkpoint(run, {'config': 'a'}, 1, [{'epoch': 1, 'mae': 3.}], model, optimizer)
            restored = common.WholeBrainCNN()
            opt2 = torch.optim.Adam(restored.parameters(), lr=.001)
            epoch, history = training.resume(run, {'config': 'a'}, restored, opt2)
            self.assertEqual(epoch, 1)
            self.assertEqual(history[0]['mae'], 3.)
            self.assertTrue(torch.equal(model(image), restored(image)))
            # Optimizer state is restored too: next update must remain identical.
            for network, opt in ((model, optimizer), (restored, opt2)):
                opt.zero_grad()
                training.step(network(image).square().mean(), network, opt)
            self.assertTrue(torch.equal(model(image), restored(image)))
            with self.assertRaises(ValueError):
                training.resume(run, {'config': 'b'}, restored, opt2)

    def test_train_target_scaling(self):
        values, mean, std = training.targets([{'IST_fluid': '10'}, {'IST_fluid': '20'}])
        self.assertEqual(mean, 15)
        self.assertEqual(std, 5)
        with self.assertRaises(ValueError):
            training.targets([{'IST_fluid': '10'}, {'IST_fluid': '10'}])

    def test_unapproved_or_unexplained_flag_is_rejected(self):
        row = dict(participant_id='synthetic')
        with self.assertRaises(ValueError):
            common.approved_artifact(row, dict(decision='pending', status='pass'), {})
        with self.assertRaises(ValueError):
            common.approved_artifact(row, dict(decision='accept', status='failed'), {})
        with self.assertRaises(ValueError):
            common.approved_artifact(row, dict(decision='accept', status='review', notes=''), {})

    def test_masked_reconstruction_gradient(self):
        torch.manual_seed(42)
        model = common.ReconstructionCNN()
        target = torch.randn(1, 1, 64, 64, 64)
        coarse = (torch.rand(1, 1, 8, 8, 8) < .2).float()
        mask = torch.nn.functional.interpolate(coarse, size=(64, 64, 64), mode='nearest').bool()
        self.assertTrue(mask.any())
        output = model(target.masked_fill(mask, 0))
        loss = (output[mask] - target[mask]).square().mean()
        before = next(model.encoder.parameters()).detach().clone()
        training.step(loss, model, torch.optim.Adam(model.parameters(), lr=.001))
        self.assertFalse(torch.equal(before, next(model.encoder.parameters())))


if __name__ == '__main__':
    unittest.main()
