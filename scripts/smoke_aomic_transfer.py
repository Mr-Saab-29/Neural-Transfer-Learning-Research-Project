#!/usr/bin/env python3
"""Transfer the reconstruction encoder and check supervised training on approved AOMIC scans."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke_train_3d import new_model
from src.data.aomic_dataset import AOMICDataset
from src.data.dataset import sha256


def join_rows(pilot, downstream):
    labels = {r['hf_path']: r for r in downstream}
    if len(labels) != len(downstream) or len({r['source_participant_id'] for r in downstream}) != len(labels):
        raise ValueError('Duplicate scan or original subject in downstream allocations')
    joined = []
    for scan in pilot:
        if scan['dataset'] != 'PT013_AIOMIC1000' or scan['decision'] != 'accept':
            continue
        label = labels[scan['hf_path']]
        if (label['dataset'], label['participant_id'], label['pretraining_allowed']) != (
                scan['dataset'], scan['participant_id'], 'False'):
            raise ValueError('Inconsistent frozen AOMIC mapping')
        joined.append(scan | {key: label[key] for key in ('source_participant_id', 'IST_fluid', 'split', 'split_seed')})
    if not joined:
        raise ValueError('No approved AOMIC scans matched')
    return joined


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def regressor():
    # Same encoder architecture and patch scale as the reconstruction smoke test.
    return nn.Sequential(*list(new_model().children())[:2], nn.AdaptiveAvgPool3d(2),
                         nn.Flatten(), nn.Linear(64, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path, help='model.pt from smoke_train_3d.py')
    checkpoint = parser.parse_args().checkpoint
    torch.manual_seed(42)
    torch.set_num_threads(1)
    manifests = ROOT / 'data/manifests'
    rows = join_rows(read_rows(manifests / 'pilot_training.csv'),
                     read_rows(manifests / 'aomic_downstream_splits_v1.csv'))
    pretrain_rows = read_rows(checkpoint.parent / 'pilot_sources.csv')
    if {r['dataset'] for r in pretrain_rows} != {'PT005_IXI', 'PT015_NKI'}:
        raise ValueError('Expected an IXI/NKI source checkpoint')
    out = ROOT / 'results/smoke_aomic'
    out.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix='run-', dir=out))
    manifest = run / 'approved_aomic.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fit_head(manifest, run, checkpoint)


def fit_head(manifest, run, checkpoint=None, seed=42):
    """An absent checkpoint keeps the identically seeded source encoder at initialization."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    train = AOMICDataset(manifest, 'train')
    scores = torch.tensor([float(row['IST_fluid']) for row in train.rows])
    mean, std = scores.mean().item(), scores.std(unbiased=False).item()
    if std < 1e-6:
        raise ValueError('Training scores have insufficient variance')
    source = new_model()
    if checkpoint is not None:
        source.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model = regressor()
    model[0].load_state_dict(source[0].state_dict())
    model[0].requires_grad_(False)
    head_before = model[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(model[-1].parameters(), lr=.001)
    loader = DataLoader(train, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(seed))
    losses = []
    for batch in loader:
        patch = batch['image'][:, :, 48:80, 48:80, 48:80].contiguous()
        if not losses:
            probe = patch.clone()
        pred = model(patch).squeeze(1)
        target = (batch['target'] - mean) / std
        loss = nn.functional.mse_loss(pred, target)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite regression loss')
        optimizer.zero_grad()
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model[-1].parameters()):
            raise ValueError('Nonfinite regression gradient')
        optimizer.step()
        losses.append(loss.item())
    if torch.equal(head_before, model[-1].weight) or not all(torch.equal(a,b) for a,b in zip(source[0].parameters(),model[0].parameters())):
        raise RuntimeError('Head update or frozen encoder check failed')
    saved = run / 'regressor.pt'
    torch.save(dict(weights=model.state_dict(), target_mean=mean, target_std=std), saved)
    state = torch.load(saved, map_location='cpu', weights_only=True)
    restored = regressor()
    restored.load_state_dict(state['weights'])
    with torch.inference_mode():
        reference = model.eval()(probe) * std + mean
        replay = restored.eval()(probe) * state['target_std'] + state['target_mean']
        if not torch.isfinite(reference).all() or not torch.equal(reference, replay):
            raise RuntimeError('Restored prediction mismatch')
    counts = Counter(row['split'] for row in read_rows(manifest))
    report = dict(seed=seed, split_counts={s: counts[s] for s in ('train','val','test')},
                  encoder_checkpoint=str(checkpoint.resolve()) if checkpoint else None,
                  encoder_checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
                  updates=len(losses), training_losses=losses, target='IST_fluid',
                  normalization_fit='approved training subjects only', checkpoint_max_difference=0.0)
    (run / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"PASS: {len(losses)} supervised updates; frozen encoder verified; exact checkpoint match.\n{run}")
    return saved


if __name__ == '__main__':
    main()
