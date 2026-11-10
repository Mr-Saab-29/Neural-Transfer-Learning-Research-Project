"""Lazy, label-free pilot loading from a frozen, visually accepted manifest."""
import csv
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'val', 'test')


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def project_path(root, value):
    relative = Path(value)
    path = root / relative
    if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe project path: {value}')
    return path


class MRIDataset(Dataset):
    def __init__(self, manifest, split, root=ROOT):
        if split not in SPLITS:
            raise ValueError(f'Unknown split: {split}')
        self.root = Path(root)
        with Path(manifest).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        seen = set()
        for row in rows:
            subject = (row['dataset'], row['participant_id'])
            if subject in seen:
                raise ValueError(f'Duplicate subject or split leakage: {subject}')
            seen.add(subject)
            if row['split'] not in SPLITS or row['decision'] != 'accept':
                raise ValueError('Manifest must contain only accepted scans and valid splits')
            if len(row['tensor_sha256']) != 64:
                raise ValueError('Missing tensor checksum')
            project_path(self.root, row['tensor_path'])
        self.rows = [row for row in rows if row['split'] == split]
        if not self.rows:
            raise ValueError(f'No accepted scans in {split}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = project_path(self.root, row['tensor_path'])
        if sha256(path) != row['tensor_sha256']:
            raise ValueError(f'Tensor checksum mismatch: {path}')
        image = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(image, torch.Tensor) or image.dtype != torch.float32
                or tuple(image.shape) != (1, 128, 128, 128)
                or not torch.isfinite(image).all()):
            raise ValueError(f'Invalid MRI tensor: {path}')
        return {'image': image, 'dataset': row['dataset'],
                'participant_id': row['participant_id'], 'hf_path': row['hf_path']}


def make_loader(manifest, split, batch_size=1, seed=42, root=ROOT):
    """Single-process CPU loading for the 8 GB Mac; no augmentation or labels."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    return DataLoader(MRIDataset(manifest, split, root), batch_size=batch_size,
                      shuffle=split == 'train', num_workers=0, pin_memory=False,
                      drop_last=False, generator=torch.Generator().manual_seed(seed))
