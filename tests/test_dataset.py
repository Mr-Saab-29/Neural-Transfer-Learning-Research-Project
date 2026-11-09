import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from src.data.dataset import MRIDataset, make_loader, sha256
from build_training_manifest import assign_splits, build, write_new, FIELDS


class DatasetTests(unittest.TestCase):
    def test_split_determinism_and_subject_separation(self):
        rows = [{'dataset': dataset, 'participant_id': str(i)} for dataset in ('A', 'B', 'C') for i in range(15)]
        output = assign_splits(rows, 42)
        self.assertEqual(output, assign_splits(list(reversed(rows)), 42))
        groups = [{(r['dataset'], r['participant_id']) for r in output if r['split'] == s}
                  for s in ('train', 'val', 'test')]
        self.assertEqual([len(g) for g in groups], [33, 6, 6])
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_lazy_loading_and_rejections(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            tensor = root / 'test.pt'
            torch.save(torch.ones(1, 128, 128, 128), tensor)
            rows = [dict(dataset='A', participant_id=str(i), hf_path=f'A/{i}',
                         tensor_path='test.pt', tensor_sha256=sha256(tensor),
                         metadata_sha256='test', decision='accept', qc_status='pass',
                         split=split, split_seed='42') for i, split in enumerate(('train', 'val', 'test'))]
            manifest = root / 'manifest.csv'
            write_new(manifest, rows, FIELDS)
            batch = next(iter(make_loader(manifest, 'train', root=root)))
            self.assertEqual(tuple(batch['image'].shape), (1, 1, 128, 128, 128))
            self.assertNotIn('target', batch)
            dataset = MRIDataset(manifest, 'test', root)
            tensor.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                dataset[0]
            rows[1]['participant_id'] = rows[0]['participant_id']
            other = root / 'duplicate.csv'
            write_new(other, rows, FIELDS)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                MRIDataset(other, 'train', root)

    def test_accepted_manifest_and_stale_approval(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pilot, decisions, latest = [], [], {}
            for i in range(3):
                hf_path = f'A/{i}'
                folder = root / str(i)
                folder.mkdir()
                (folder / 'tensor.pt').write_bytes(b'fixture-artifact')
                metadata = folder / 'metadata.json'
                metadata.write_text(json.dumps({'provenance': {'hf_path': hf_path},
                    'sha256': {'tensor.pt': sha256(folder / 'tensor.pt')}}))
                pilot.append(dict(dataset='A', participant_id=str(i), hf_path=hf_path))
                decisions.append(dict(hf_path=hf_path, decision='accept',
                                      metadata_sha256=sha256(metadata), notes='visual review'))
                latest[hf_path] = dict(status='pass', output_dir=str(i))
            rows = build(pilot, latest, decisions, root, 42)
            self.assertEqual({r['split'] for r in rows}, {'train', 'val', 'test'})
            decisions[0]['metadata_sha256'] = 'old'
            with self.assertRaisesRegex(ValueError, 'stale'):
                build(pilot, latest, decisions, root, 42)

    def test_qc_gating(self):
        pilot = [dict(dataset='A', participant_id='1', hf_path='A/1')]
        pending = [dict(hf_path='A/1', decision='pending', metadata_sha256='', notes='')]
        with self.assertRaisesRegex(ValueError, 'No visually accepted'):
            build(pilot, {}, pending, ROOT, 42)
        pending[0]['decision'] = 'accept'
        with self.assertRaisesRegex(ValueError, 'no successful'):
            build(pilot, {'A/1': {'status': 'failed'}}, pending, ROOT, 42)
        with self.assertRaisesRegex(ValueError, 'written review reason'):
            build(pilot, {'A/1': {'status': 'review'}}, pending, ROOT, 42)


if __name__ == '__main__':
    unittest.main()
