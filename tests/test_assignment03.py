import argparse
import csv
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from build_pilot_manifest import DATASETS, select, safe_path
import preprocess_pilot as batch
from src.data.preprocessing import preprocess_mri


class PilotTests(unittest.TestCase):
    def test_selection(self):
        with (ROOT / 'data/manifests/pretraining_cohort.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        expected = select(rows)
        random.Random(4).shuffle(rows)
        self.assertEqual(expected, select(rows))
        self.assertEqual(len({(r['dataset'], r['participant_id']) for r in expected}), 45)
        for dataset in DATASETS:
            self.assertEqual(sum(r['dataset'] == dataset for r in expected), 15)
        with self.assertRaises(ValueError):
            safe_path('../escape')

    def test_retention_and_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.nii.gz'
            data = np.indices((10, 10, 10)).sum(axis=0).astype(np.float32) + 1
            nib.save(nib.Nifti1Image(data, np.diag([2., 2., 2., 1.])), path)
            tensor, qc = preprocess_mri(path, target_shape=(6, 6, 6), return_qc=True)
            self.assertAlmostEqual(qc['foreground_retention'], 216 / 1000)
            self.assertEqual(qc['orientation'], 'RAS')
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertIsInstance(preprocess_mri(path), torch.Tensor)
            nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32), np.eye(4)), path)
            with self.assertRaises(ValueError):
                preprocess_mri(path)

    def test_batch_failure_resume_and_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / 'src/data/preprocessing.py'
            module.parent.mkdir(parents=True)
            module.write_bytes((ROOT / 'src/data/preprocessing.py').read_bytes())
            manifests = root / 'data/manifests'
            manifests.mkdir(parents=True)
            rows = [dict(dataset='PT005_IXI', participant_id=s, hf_path=f'PT005_IXI/{s}/ses_1/t1.nii.gz')
                    for s in ('bad', 'good')]
            for row in rows:
                path = root / 'data/raw' / row['hf_path']
                path.parent.mkdir(parents=True)
                if row['participant_id'] == 'bad':
                    path.write_bytes(b'broken')
                else:
                    data = np.indices((16, 16, 16)).sum(axis=0).astype(np.float32)
                    nib.save(nib.Nifti1Image(data, np.diag([2., 2., 2., 1.])), path)
            args = argparse.Namespace(revision='a'*40, offline=True, min_retention=.95)
            with patch.object(batch, 'ROOT', root), patch.object(batch, 'hf_hub_download', side_effect=AssertionError('No network')):
                with self.assertRaises(SystemExit):
                    batch.run(rows, args, manifests)
                output = root / 'data/processed/PT005_IXI/good/ses_1/t1.pt'
                original = output.read_bytes()
                batch.run(rows[1:], args, manifests)
                self.assertEqual(original, output.read_bytes())
                self.assertEqual((root / 'data/raw' / rows[0]['hf_path']).read_bytes(), b'broken')
                output.write_bytes(b'corrupt')
                with self.assertRaises(SystemExit):
                    batch.run(rows[1:], args, manifests)
                self.assertEqual(output.read_bytes(), b'corrupt')
            with (manifests / 'preprocessing_qc.csv').open() as stream:
                statuses = [r['status'] for r in csv.DictReader(stream)]
            self.assertEqual(statuses, ['failed', 'pass', 'pass', 'failed'])


if __name__ == '__main__':
    unittest.main()
