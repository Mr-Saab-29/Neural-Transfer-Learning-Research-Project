#!/usr/bin/env python3
"""Inspect native boundary contacts, before centering or resampling."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.brain_preprocessing import validate_mask
from src.data.dataset import project_path, sha256


def main():
    with (ROOT / 'data/manifests/pilot_cohort.csv').open() as stream:
        pilot = {r['hf_path'] for r in csv.DictReader(stream)}
    with (ROOT / 'data/manifests/brain_preprocessing_qc.csv').open() as stream:
        latest = {r['hf_path']: r for r in csv.DictReader(stream)}
    output = ROOT / 'results/assignment03/native_boundaries'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    links = ['# Native boundary review', '',
             'Top row: untouched native intensities. Bottom row: same slices with green mask outlines.',
             'Column 1 is the actual image edge; later columns move inward. Slice indices are zero-based.',
             'Nonzero signal at an edge is evidence to inspect, not proof of missing brain.', '']
    for path, row in sorted(latest.items()):
        if path not in pilot or row['status'] != 'review' or 'mask_touches_native_boundary' not in row['reasons']:
            continue
        raw_path = project_path(ROOT / 'data/raw', path)
        folder = project_path(ROOT, row['output_dir'])
        metadata = json.loads((folder / 'metadata.json').read_text())
        mask_path = folder / 'native_mask.nii.gz'
        if sha256(raw_path) != metadata['provenance']['raw_sha256'] or sha256(mask_path) != metadata['sha256']['native_mask.nii.gz']:
            raise ValueError(f'Input checksum mismatch: {path}')
        img = nib.load(raw_path)
        mask = validate_mask(img, nib.load(mask_path))
        values = img.get_fdata(dtype=np.float32)
        vmin, vmax = np.percentile(values[mask & (values != 0)], [1, 99])
        orientation = nib.aff2axcodes(img.affine)
        for axis in range(3):
            for edge in (0, img.shape[axis] - 1):
                boundary = np.take(mask, edge, axis=axis)
                if not boundary.any():
                    continue
                offsets = [0, 1, 2, 4, 8]
                indices = [min(img.shape[axis]-1, d) if edge == 0 else max(0, edge-d) for d in offsets]
                fig, axes = plt.subplots(2, len(indices), figsize=(15, 6))
                for j, index in enumerate(indices):
                    sl = np.take(values, index, axis=axis).T
                    ms = np.take(mask, index, axis=axis).T
                    for i in range(2):
                        axes[i, j].imshow(sl, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                        axes[i, j].set_xticks([])
                        axes[i, j].set_yticks([])
                    if ms.any() and not ms.all():
                        axes[1, j].contour(ms, levels=[.5], colors=['lime'], linewidths=.7)
                    axes[0, j].set_title(f'Native slice {index}' + (' — EDGE' if j == 0 else ''))
                axes[0, 0].set_ylabel('Original image')
                axes[1, 0].set_ylabel('Original + mask')
                signal = int((boundary & (np.take(values, edge, axis=axis) != 0)).sum())
                title = f"{row['dataset']} / {row['participant_id']} | native axis {axis} (+{orientation[axis]}) | edge {edge}"
                fig.suptitle(title + f'\nMask voxels on edge: {int(boundary.sum())}; with nonzero input signal: {signal}')
                fig.tight_layout(rect=(0, 0, 1, .90))
                name = f"{row['dataset']}_{row['participant_id']}_axis{axis}_edge{edge}.png"
                fig.savefig(output / name, dpi=120)
                plt.close(fig)
                records.append({'hf_path': path, 'axis': axis, 'edge_index': edge,
                                'mask_voxels_on_edge': int(boundary.sum()),
                                'nonzero_input_inside_edge_mask': signal, 'image': name})
                links.append(f'- [{title}]({name})')
    with (output / 'boundary_contacts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['hf_path', 'axis', 'edge_index', 'mask_voxels_on_edge', 'nonzero_input_inside_edge_mask', 'image'])
        writer.writeheader()
        writer.writerows(records)
    (output / 'index.md').write_text('\n'.join(links) + '\n')
    print(f'{len(records)} boundary views from {len({r["hf_path"] for r in records})} scans: {output / "index.md"}')
    for row in records:
        print(row['hf_path'], 'axis', row['axis'], 'edge', row['edge_index'],
              'mask', row['mask_voxels_on_edge'], 'nonzero input', row['nonzero_input_inside_edge_mask'])


if __name__ == '__main__':
    main()
