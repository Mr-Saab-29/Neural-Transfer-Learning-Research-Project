#!/usr/bin/env python3
"""Plot low/median/high foreground examples per dataset without changing data."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocessing import standardize_geometry, center_crop_or_pad


def main():
    with (ROOT / 'data/manifests/pilot_cohort.csv').open() as stream:
        pilot = {row['hf_path'] for row in csv.DictReader(stream)}
    latest = {}
    with (ROOT / 'data/manifests/preprocessing_qc.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['hf_path'] in pilot:
                latest[row['hf_path']] = row
    selected = []
    for dataset in sorted({row['dataset'] for row in latest.values()}):
        rows = sorted((row for row in latest.values()
                       if row['dataset'] == dataset and row['status'] in ('pass', 'review')),
                      key=lambda row: (float(row['foreground_fraction']), row['hf_path']))
        if not rows:
            continue
        for label, index in [('low', 0), ('median', len(rows)//2), ('high', len(rows)-1)]:
            selected.append((label, rows[index]))
    if not selected:
        raise SystemExit('No successful scans available.')
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3 * len(selected)), squeeze=False)
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad('black')
    for i, (label, row) in enumerate(selected):
        volume = torch.load(ROOT / row['output'], map_location='cpu', weights_only=True)[0].numpy()
        # Recover the pre-normalization mask; normalized zero is not background.
        img = standardize_geometry(ROOT / 'data/raw' / row['hf_path'])
        mask = center_crop_or_pad(img.get_fdata(dtype=np.float32) != 0).astype(bool)
        display = np.ma.array(volume, mask=~mask)
        for j, (axis, name) in enumerate([(0, 'Sagittal'), (1, 'Coronal'), (2, 'Axial')]):
            slice_image = np.take(display, volume.shape[axis] // 2, axis=axis).T
            axes[i, j].imshow(slice_image, origin='lower', cmap=cmap, vmin=-1, vmax=3,
                              interpolation='nearest')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(name + ' — center slice')
        axes[i, 0].set_ylabel(f"{row['dataset']}\n{row['participant_id']} ({label})\n"
                              f"foreground {float(row['foreground_fraction']):.1%}", fontsize=9)
    fig.suptitle('Pilot QC: low / median / high foreground fraction per dataset\n'
                 'Processed tensors; fixed Z-score window [−1, 3]; original zero background masked black', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    output = ROOT / 'results/assignment03'
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'pilot_qc_montage.png', dpi=130)
    plt.close(fig)
    with (output / 'visual_qc_selection.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['selection', 'dataset', 'participant_id', 'hf_path', 'foreground_fraction'])
        writer.writeheader()
        for label, row in selected:
            writer.writerow({'selection': label, **{k: row[k] for k in writer.fieldnames if k != 'selection'}})
    print(output / 'pilot_qc_montage.png')


if __name__ == '__main__':
    main()
