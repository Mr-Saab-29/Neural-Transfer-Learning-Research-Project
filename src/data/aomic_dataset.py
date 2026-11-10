"""Approved brain tensors paired with original-scale AOMIC fluid scores."""
import math

import torch

from src.data.dataset import MRIDataset, ROOT


class AOMICDataset(MRIDataset):
    def __init__(self, manifest, split, root=ROOT):
        super().__init__(manifest, split, root)
        for row in self.rows:
            if row['dataset'] != 'PT013_AIOMIC1000' or not math.isfinite(float(row['IST_fluid'])):
                raise ValueError('Expected an AOMIC scan with a finite IST_fluid score')

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['target'] = torch.tensor(float(self.rows[index]['IST_fluid']), dtype=torch.float32)
        return item
