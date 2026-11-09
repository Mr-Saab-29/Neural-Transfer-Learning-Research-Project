#!/usr/bin/env python3
"""Read one batch per split; no training or inference."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import make_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/pilot_training.csv')
    args = parser.parse_args()
    for split in ('train', 'val', 'test'):
        loader = make_loader(args.manifest, split)
        batch = next(iter(loader))
        print(split, 'subjects=', len(loader.dataset), 'batch=', tuple(batch['image'].shape),
              'dtype=', batch['image'].dtype, 'finite=', bool(batch['image'].isfinite().all()))
