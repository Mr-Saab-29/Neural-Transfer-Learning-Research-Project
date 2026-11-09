#!/usr/bin/env python3
"""CPU smoke test: reconstruct hidden MRI voxels and round-trip a checkpoint."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import make_loader

SEED = 42
SOURCES = {'PT005_IXI', 'PT015_NKI'}


def new_model():
    return nn.Sequential(nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
                         nn.Conv3d(8, 1, 3, padding=1))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    masks = torch.Generator().manual_seed(SEED + 1)
    manifests = ROOT / 'data/manifests'
    with (manifests / 'pretraining_without_aomic_v1.csv').open() as stream:
        allowed = {(r['dataset'], r['participant_id'], r['hf_path']) for r in csv.DictReader(stream)}
    with (manifests / 'pilot_training.csv').open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [r for r in reader if r['dataset'] in SOURCES and r['decision'] == 'accept']
    if Counter(r['dataset'] for r in rows) != {s: 15 for s in SOURCES}:
        raise ValueError('Expected 15 approved subjects per pretraining source')
    for row in rows:
        key = row['dataset'], row['participant_id'], row['hf_path']
        expected = Path('data/processed/brain_v1') / (row['hf_path'] + '.brain') / 'tensor.pt'
        if key not in allowed or Path(row['tensor_path']) != expected:
            raise ValueError(f'Unexpected pretraining record: {key}')
    out = ROOT / 'results/smoke_3d'
    out.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix='run-', dir=out))
    manifest = run / 'pilot_sources.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    train = make_loader(manifest, 'train', seed=SEED)
    probe = next(iter(make_loader(manifest, 'val')))['image'][:, :, 48:80, 48:80, 48:80].contiguous()
    model = new_model()
    initial = model[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    steps = []
    for batch in train:
        # A central 32³ patch keeps the real-data check inexpensive on CPU.
        target = batch['image'][:, :, 48:80, 48:80, 48:80].contiguous()
        hidden = (torch.rand(target.shape, generator=masks) < .2) & (target != 0)
        if not hidden.any():
            raise ValueError('Patch has no masked signal')
        optimizer.zero_grad()
        prediction = model(target.masked_fill(hidden, 0))
        loss = (prediction[hidden] - target[hidden]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite reconstruction loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Missing or nonfinite gradient')
        optimizer.step()
        steps.append({'scan': batch['hf_path'][0], 'loss': loss.item()})
    change = (model[0].weight.detach() - initial).abs().max().item()
    if not change > 0 or any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Model weights failed the update check')
    checkpoint = run / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    restored = new_model()
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    with torch.inference_mode():
        reference = model.eval()(probe)
        replay = restored.eval()(probe)
        if not torch.isfinite(reference).all() or not torch.equal(reference, replay):
            raise ValueError('Checkpoint output differs on the fixed validation patch')
    report = dict(seed=SEED, split_counts=dict(Counter(r['split'] for r in rows)),
                  updates=len(steps), steps=steps, max_first_layer_weight_change=change,
                  checkpoint_max_output_difference=(reference - replay).abs().max().item())
    (run / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(steps)} CPU updates, weight change {change:.6g}, exact checkpoint match.\n{run}')


if __name__ == '__main__':
    main()
