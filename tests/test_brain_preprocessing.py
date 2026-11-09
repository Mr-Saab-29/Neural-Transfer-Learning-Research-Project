import csv
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from src.data.brain_preprocessing import preprocess_brain, review_reasons
import preprocess_brain_pilot as batch
from visualize_brain_qc import render_scan


def fixture(folder, name='raw.nii.gz', shape=(50, 60, 70)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:15, 20:34, 40:58] = 1
    data = np.indices(shape).sum(axis=0).astype(np.float32) + 1
    data[~mask.astype(bool)] = 500  # Strong extracranial signal must be excluded.
    affine = np.diag([2., 2., 2., 1.])
    raw = folder / name
    raw.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), raw)
    mask_path = folder / 'mask.nii.gz'
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    return raw, mask_path


class BrainTests(unittest.TestCase):
    def test_centering_normalization_and_affine(self):
        with tempfile.TemporaryDirectory() as name:
            raw, mask = fixture(Path(name))
            tensor, centered_mask, _, qc = preprocess_brain(raw, mask)
            brain = centered_mask.get_fdata() > 0
            coords = np.array(np.nonzero(brain))
            np.testing.assert_allclose((coords.min(1) + coords.max(1)) / 2, [63.5]*3)
            self.assertEqual(tuple(tensor.shape), (1, 128, 128, 128))
            self.assertEqual(qc['brain_retention'], 1)
            self.assertAlmostEqual(tensor[0].numpy()[brain].mean(), 0, places=5)
            self.assertAlmostEqual(tensor[0].numpy()[brain].std(), 1, places=5)
            self.assertTrue((tensor[0].numpy()[~brain] == 0).all())
            np.testing.assert_allclose(nib.affines.apply_affine(centered_mask.affine, coords.min(1)), [10, 40, 80])
            _, _, _, small_qc = preprocess_brain(raw, mask, target_shape=(8, 8, 8))
            self.assertLess(small_qc['brain_retention'], 1)
            self.assertIn('brain_crop_loss', review_reasons(small_qc, 'PT015_NKI'))

    def test_invalid_masks(self):
        with tempfile.TemporaryDirectory() as name:
            raw, path = fixture(Path(name))
            original = nib.load(path)
            data = original.get_fdata()
            for values, affine in [(data, np.eye(4)), (data * .5, original.affine),
                                   (np.zeros_like(data), original.affine),
                                   (np.ones_like(data), original.affine),
                                   (np.full_like(data, np.nan), original.affine)]:
                nib.save(nib.Nifti1Image(values, affine), path)
                with self.assertRaises(ValueError):
                    preprocess_brain(raw, path)

    def test_reoriented_mask(self):
        with tempfile.TemporaryDirectory() as name:
            raw, mask = fixture(Path(name))
            tensor, _, _, _ = preprocess_brain(raw, mask)
            for path in (raw, mask):
                img = nib.load(path)
                nib.save(img.as_reoriented(np.array([[0, -1], [1, 1], [2, 1]])), path)
            flipped, _, _, qc = preprocess_brain(raw, mask)
            np.testing.assert_allclose(tensor.numpy(), flipped.numpy(), atol=1e-5)
            self.assertEqual(qc['orientation'], 'RAS')

    def test_transaction_resume_and_render(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            relative = 'PT005_IXI/sub_test/ses_1/t1.nii.gz'
            raw, mask = fixture(root / 'data/raw/PT005_IXI/sub_test/ses_1', 't1.nii.gz')
            original = raw.read_bytes()
            logs = root / 'logs'
            logs.mkdir()
            row = {'hf_path': relative}
            def fake_extract(raw_path, work, identity, timeout):
                (work / 'native_mask.nii.gz').write_bytes(mask.read_bytes())
                (work / 'synthstrip.log').write_text('SYNTHETIC TEST ONLY')
            with patch.object(batch, 'ROOT', root), patch.object(batch, 'extract', side_effect=fake_extract) as extractor:
                folder, qc = batch.process_scan(row, 'test-image', 'test-code', 30, logs, 'test')
                batch.process_scan(row, 'test-image', 'test-code', 30, logs, 'test')
                self.assertEqual(extractor.call_count, 1)
                self.assertEqual(raw.read_bytes(), original)
                render_scan(folder, root / 'test.png', 'SYNTHETIC TEST — not MRI experiment')
                self.assertTrue((root / 'test.png').is_file())
                with self.assertRaises(ValueError):
                    batch.process_scan(row, 'changed-image', 'test-code', 30, logs, 'test')
                (folder / 'tensor.pt').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    batch.process_scan(row, 'test-image', 'test-code', 30, logs, 'test')
                self.assertEqual((folder / 'tensor.pt').read_bytes(), b'corrupt')

    def test_batch_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for relative in ('src/data/brain_preprocessing.py', 'src/data/preprocessing.py',
                             'scripts/preprocess_pilot.py', 'scripts/build_pilot_manifest.py'):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            (root / 'data/manifests').mkdir(parents=True)
            relative = 'PT015_NKI/good/ses_1/t1.nii.gz'
            _, mask = fixture(root / 'data/raw/PT015_NKI/good/ses_1', 't1.nii.gz')
            bad = {'hf_path': 'PT015_NKI/missing/ses_1/t1.nii.gz',
                   'dataset': 'PT015_NKI', 'participant_id': 'missing'}
            good = {'hf_path': relative, 'dataset': 'PT015_NKI', 'participant_id': 'good'}
            def fake_extract(raw, work, identity, timeout):
                (work / 'native_mask.nii.gz').write_bytes(mask.read_bytes())
                (work / 'synthstrip.log').write_text('SYNTHETIC TEST ONLY')
            with patch.object(batch, 'ROOT', root), patch.object(batch, 'extract', side_effect=fake_extract):
                self.assertEqual(batch.run([bad, good], 'test-image', 30), 1)
            with (root / 'data/manifests/brain_preprocessing_qc.csv').open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual([r['status'] for r in records], ['failed', 'review'])
            self.assertIn('brain_volume_outside_review_range', records[1]['reasons'])

    def test_failed_extraction_not_published(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            relative = 'PT015_NKI/sub_test/ses_1/t1.nii.gz'
            fixture(root / 'data/raw/PT015_NKI/sub_test/ses_1', 't1.nii.gz')
            logs = root / 'logs'
            logs.mkdir()
            def broken(raw, work, identity, timeout):
                (work / 'synthstrip.log').write_text('simulated failure')
                raise RuntimeError('simulated extraction failure')
            with patch.object(batch, 'ROOT', root), patch.object(batch, 'extract', side_effect=broken):
                with self.assertRaises(RuntimeError):
                    batch.process_scan({'hf_path': relative}, 'test', 'test', 30, logs, 'test')
            self.assertFalse((root / 'data/processed/brain_v1' / (relative + '.brain')).exists())
            self.assertEqual(len(list(logs.glob('*.log'))), 1)


if __name__ == '__main__':
    unittest.main()
