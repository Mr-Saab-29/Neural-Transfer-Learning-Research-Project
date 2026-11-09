#!/usr/bin/env python3
"""Render mask overlays and masked tensors at nine positions per scan."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess_brain_pilot import ARTIFACTS, checked_path, load_rows
from preprocess_pilot import digest, validate_tensor


def render_scan(folder, output, title):
    saved = json.loads((folder / 'metadata.json').read_text())
    for name in ARTIFACTS:
        if digest(folder / name) != saved['sha256'][name]:
            raise ValueError(f'Checksum mismatch: {folder / name}')
    tensor = torch.load(folder / 'tensor.pt', map_location='cpu', weights_only=True)
    validate_tensor(tensor)
    volume = tensor[0].numpy()
    raw = nib.load(folder / 'centered_input.nii.gz').get_fdata()
    mask = nib.load(folder / 'brain_mask.nii.gz').get_fdata() > .5
    coordinates = np.array(np.nonzero(mask))
    low, high = coordinates.min(axis=1), coordinates.max(axis=1)
    vmin, vmax = np.percentile(raw[mask], [1, 99])
    cmap = plt.get_cmap('gray').with_extremes(bad='black')
    masked = np.ma.array(volume, mask=~mask)
    fig, axes = plt.subplots(3, 6, figsize=(16, 8))
    for axis, name in enumerate(('Sagittal', 'Coronal', 'Axial')):
        for j, fraction in enumerate((.25, .5, .75)):
            index = int(round(low[axis] + fraction * (high[axis] - low[axis])))
            image_slice = np.take(raw, index, axis=axis).T
            mask_slice = np.take(mask, index, axis=axis).T
            axes[axis, j * 2].imshow(image_slice, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
            if mask_slice.any() and not mask_slice.all():
                axes[axis, j * 2].contour(mask_slice, levels=[.5], colors=['lime'], linewidths=.6)
            axes[axis, j * 2 + 1].imshow(np.take(masked, index, axis=axis).T, origin='lower',
                                       cmap=cmap, vmin=-2, vmax=3)
            axes[axis, j * 2].set_title(f'{name} {index}: input + mask', fontsize=9)
            axes[axis, j * 2 + 1].set_title('Masked / normalized', fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title + '\n25%, 50%, 75% of brain bounding box along each axis; no template registration')
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    rows = load_rows(args.all)
    qc_path = ROOT / 'data/manifests/brain_preprocessing_qc.csv'
    if not qc_path.exists():
        parser.exit(2, 'No brain preprocessing QC yet. Run preprocess_brain_pilot.py first.\n')
    with qc_path.open() as stream:
        latest = {r['hf_path']: r for r in csv.DictReader(stream)}
    output = ROOT / 'results/assignment03/brain_v1'
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    links = ['# Brain-mask visual review', '', 'Review each image for missing brain, retained skull/neck, and coverage.', '']
    for row in rows:
        record = latest.get(row['hf_path'])
        if record is None:
            counts['not_attempted'] += 1
            print(row['hf_path'], 'not attempted')
            continue
        counts[record['status']] += 1
        if record['status'] == 'failed':
            print(row['hf_path'], 'failed:', record['error'])
            links.append(f"- {row['hf_path']}: failed (see QC CSV)")
            continue
        folder = checked_path(ROOT, record['output_dir'])
        name = row['dataset'] + '_' + digest(folder / 'metadata.json')[:16] + '.png'
        render_scan(folder, output / name,
                    f"{row['dataset']} / {row['participant_id']} — {record['status']} {record['reasons']}")
        links.append(f"- [{row['hf_path']}]({name}) — {record['status']} {record['reasons']}")
    suffix = 'all' if args.all else 'nine'
    (output / f'index_{suffix}.md').write_text('\n'.join(links) + '\n')
    print('Latest status:', dict(counts))
    print('Visual review:', output / f'index_{suffix}.md')


if __name__ == '__main__':
    main()
