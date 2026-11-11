#!/usr/bin/env python3
"""CPU whole-volume checks, SSL, and fixed-schedule paired follow-up CNN training."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from followup_common import (OUT, PROTOCOL, WholeBrainCNN, ReconstructionCNN, whole_image,
    load_config, read, csv_bytes, digest, lock, atomic_write, save_torch, json_write)


def finite_model(model):
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite model parameters')


def step(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Missing or nonfinite gradient')
    optimizer.step()
    finite_model(model)


def prepared_rows(fold, size, role):
    approved = {r['hf_path']: r for r in read(OUT / 'ready/approved_cohort.csv')}
    name = f'fold_{fold}_train_{size}.csv' if role == 'train' else f'fold_{fold}_evaluation.csv'
    rows = read(PROTOCOL / name)
    output = []
    for row in rows:
        artifact = approved[row['hf_path']]
        if any(row[k] != artifact[k] for k in ('participant_id', 'source_participant_id', 'IST_fluid')):
            raise ValueError('Fold identity/target differs from approved cohort')
        output.append(dict(artifact, fold=str(fold), role=role))
    return output


def tensors(rows):
    return torch.stack([whole_image(r) for r in rows])


def targets(rows):
    values = torch.tensor([float(r['IST_fluid']) for r in rows])
    mean, std = values.mean().item(), values.std(unbiased=False).item()
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 1e-6:
        raise ValueError('Training targets must have finite positive variance')
    return values, mean, std


def curve(history, run, metric):
    atomic_write(run / 'history.csv', csv_bytes(history))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['epoch'] for r in history], [r[metric] for r in history])
    ax.set(xlabel='Epoch', ylabel=metric, title='Training only; no outer-fold checkpoint selection')
    fig.tight_layout()
    fig.savefig(run / 'training_curve.png', dpi=150)
    plt.close(fig)


def predictions(model, images):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x.unsqueeze(0)) for x in images])


def resume(run, fingerprint, model, optimizer):
    path = run / 'last.pt'
    if not path.exists():
        return 0, []
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['fingerprint'] != fingerprint:
        raise ValueError('Run inputs differ from checkpoint; existing run preserved')
    model.load_state_dict(saved['weights'])
    optimizer.load_state_dict(saved['optimizer'])
    finite_model(model)
    return saved['epoch'], saved['history']


def checkpoint(run, fingerprint, epoch, history, model, optimizer):
    save_torch(run / 'last.pt', dict(fingerprint=fingerprint, epoch=epoch, history=history,
               weights=model.state_dict(), optimizer=optimizer.state_dict()))


def check(config, config_hash):
    # Read every selected tensor, catching stale approvals before costly training.
    for name in ('approved_cohort.csv', 'ssl_training.csv'):
        for row in read(OUT / 'ready' / name):
            image = whole_image(row)
            if image.shape != (1, 64, 64, 64) or not torch.isfinite(image).all():
                raise ValueError('Invalid whole-brain input')
    model = WholeBrainCNN()
    probe = tensors(prepared_rows(0, 15, 'train')[:1])
    output = model(probe)
    step(output.square().mean(), model, torch.optim.Adam(model.parameters(), lr=.001))
    report = dict(config_sha256=config_hash, loader_pass=True,
                  shape=list(probe.shape), parameters=sum(p.numel() for p in model.parameters()))
    json_write(OUT / 'loader_check.json', report)
    print(json.dumps(report, indent=2))


def require_check(config_hash):
    report = json.loads((OUT / 'loader_check.json').read_text())
    if report.get('config_sha256') != config_hash or not report.get('loader_pass'):
        raise ValueError('Run check for this configuration first')


def overfit(config_hash):
    require_check(config_hash)
    rows = prepared_rows(0, 15, 'train')[:4]
    images = tensors(rows)
    values, mean, std = targets(rows)
    y = (values - mean) / std
    torch.manual_seed(42)
    model = WholeBrainCNN()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=.5, max_iter=10,
                                  history_size=20, line_search_fn='strong_wolfe')
    run = OUT / 'overfit'
    run.mkdir(exist_ok=True)
    history = []
    for epoch in range(101):
        if epoch:
            def closure():
                optimizer.zero_grad()
                loss = F.mse_loss(model(images), y)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite overfit loss')
                loss.backward()
                return loss
            optimizer.step(closure)
            finite_model(model)
        pred = predictions(model, images)
        mae = ((pred * std + mean) - values).abs().mean().item()
        history.append(dict(epoch=epoch, mse=F.mse_loss(pred, y).item(), mae=mae))
        print('Overfit:', history[-1], flush=True)
        if mae < 1:
            break
    save_torch(run / 'model.pt', dict(weights=model.state_dict(), target_mean=mean, target_std=std))
    restored = WholeBrainCNN()
    restored.load_state_dict(torch.load(run / 'model.pt', weights_only=True, map_location='cpu')['weights'])
    if not torch.equal(pred, predictions(restored, images)):
        raise ValueError('Overfit checkpoint round-trip mismatch')
    curve(history, run, 'mae')
    report = dict(config_sha256=config_hash, training_subjects=4, final_mae=mae,
                  overfit_pass=mae < 1, criterion='Training MAE below 1 IST_fluid point',
                  subjects=[r['source_participant_id'] for r in rows])
    json_write(run / 'report.json', report)
    print(json.dumps(report, indent=2))


def require_overfit(config_hash):
    require_check(config_hash)
    report = json.loads((OUT / 'overfit/report.json').read_text())
    if report['config_sha256'] != config_hash or not report['overfit_pass']:
        raise ValueError('A passing four-subject training-only overfit check is required')


def pretrain(config, config_hash, max_epochs):
    require_overfit(config_hash)
    settings = config['pretraining']
    rows = read(OUT / 'ready/ssl_training.csv')
    if any(r['dataset'] not in ('PT005_IXI', 'PT015_NKI') or r['split'] != 'train' for r in rows):
        raise ValueError('SSL must use only approved IXI/NKI training subjects')
    images = tensors(rows)
    torch.manual_seed(settings['seed'])
    model = ReconstructionCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    run = OUT / 'pretraining'
    run.mkdir(exist_ok=True)
    fingerprint = dict(config_sha256=config_hash, task='masked_reconstruction')
    start, history = resume(run, fingerprint, model, optimizer)
    stop = min(settings['epochs'], start + max_epochs)
    for epoch in range(start + 1, stop + 1):
        rng = torch.Generator().manual_seed(settings['seed'] + epoch)
        losses = []
        for index in torch.randperm(len(images), generator=rng):
            target = images[index].unsqueeze(0)
            coarse = (torch.rand((1, 1, 8, 8, 8), generator=rng) < .2).float()
            mask = F.interpolate(coarse, size=(64, 64, 64), mode='nearest').bool() & (target != 0)
            if not mask.any():
                raise ValueError('No masked foreground')
            optimizer.zero_grad()
            predicted = model(target.masked_fill(mask, 0))
            loss = (predicted[mask] - target[mask]).square().mean()
            step(loss, model, optimizer)
            losses.append(loss.item())
        history.append(dict(epoch=epoch, masked_mse=sum(losses) / len(losses)))
        checkpoint(run, fingerprint, epoch, history, model, optimizer)
        print('Pretraining:', history[-1], flush=True)
    if history:
        curve(history, run, 'masked_mse')
    if stop == settings['epochs']:
        path = run / 'encoder.pt'
        if not path.exists():
            save_torch(path, dict(config_sha256=config_hash, architecture=config['architecture'],
                                 encoder=model.encoder.state_dict(), epochs=stop))
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['config_sha256'] != config_hash or any(
                not torch.equal(value, state['encoder'][key]) for key, value in model.encoder.state_dict().items()):
            raise ValueError('Existing encoder differs from completed SSL run')
        json_write(run / 'report.json', dict(config_sha256=config_hash, epochs=stop,
                   subjects=len(rows), checkpoint_sha256=digest(path)))
        print('Pretraining complete:', path)
    else:
        print(f'Paused at epoch {stop}; rerun pretrain to resume.')


def pretrained_state(config_hash):
    path = OUT / 'pretraining/encoder.pt'
    report = json.loads((OUT / 'pretraining/report.json').read_text())
    if report['config_sha256'] != config_hash or digest(path) != report['checkpoint_sha256']:
        raise ValueError('Pretraining checkpoint changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['config_sha256'] != config_hash:
        raise ValueError('Pretraining config mismatch')
    return state, digest(path)


def summarize(config_hash):
    """Pool out-of-fold subjects; never average unequal fold sizes silently."""
    _, ssl_hash = pretrained_state(config_hash)
    summaries, combined = [], []
    for size in (15, 30, 60):
        for seed in (42, 43, 44):
            paired = {}
            for kind in ('random', 'pretrained'):
                records = []
                for fold in range(3):
                    run = OUT / 'runs' / f'fold{fold}-n{size}-seed{seed}-{kind}'
                    report = json.loads((run / 'report.json').read_text())
                    expected = dict(config_sha256=config_hash, pretraining_sha256=ssl_hash,
                                    fold=fold, size=size, seed=seed, kind=kind)
                    if report['fingerprint'] != expected or any(
                            digest(run / name) != value for name, value in report['files_sha256'].items()):
                        raise ValueError('Completed run provenance mismatch')
                    fold_rows = read(run / 'predictions.csv')
                    expected_ids = {r['source_participant_id'] for r in prepared_rows(fold, size, 'evaluation')}
                    if len(fold_rows) != 30 or {r['source_participant_id'] for r in fold_rows} != expected_ids:
                        raise ValueError('Missing or duplicate out-of-fold predictions')
                    records.extend(fold_rows)
                if len(records) != 90 or len({r['source_participant_id'] for r in records}) != 90:
                    raise ValueError('Expected exactly 90 unique out-of-fold subjects')
                paired[kind] = {r['source_participant_id']: r for r in records}
                summaries.append(dict(size=size, seed=seed, kind=kind, subjects=90,
                    model_mae=sum(abs(float(r['model']) - float(r['observed'])) for r in records) / 90,
                    baseline_mae=sum(abs(float(r['training_mean']) - float(r['observed'])) for r in records) / 90))
            for sid, random in paired['random'].items():
                pre = paired['pretrained'][sid]
                if any(random[k] != pre[k] for k in ('observed', 'training_mean')):
                    raise ValueError('Paired targets or baselines disagree')
                combined.append(dict(size=size, seed=seed, source_participant_id=sid,
                    observed=random['observed'], training_mean=random['training_mean'],
                    random=random['model'], pretrained=pre['model']))
    report = dict(config_sha256=config_hash, pretraining_sha256=ssl_hash,
                  units='IST_fluid points', per_seed=summaries,
                  mean_over_seeds=[dict(size=n, kind=k, mae=sum(
                      r['model_mae'] for r in summaries if r['size'] == n and r['kind'] == k) / 3)
                      for n in (15, 30, 60) for k in ('random', 'pretrained')])
    atomic_write(OUT / 'out_of_fold_predictions.csv', csv_bytes(combined))
    json_write(OUT / 'comparison.json', report)
    print(json.dumps(report, indent=2))


def train_one(config, config_hash, fold, size, seed, kind, max_epochs):
    require_overfit(config_hash)
    state, ssl_hash = pretrained_state(config_hash)
    rows = prepared_rows(fold, size, 'train')
    images = tensors(rows)
    values, mean, std = targets(rows)
    y = (values - mean) / std
    torch.manual_seed(seed)
    model = WholeBrainCNN()
    if kind == 'pretrained':
        model.encoder.load_state_dict(state['encoder'])
    # Identical seeded head and sample order; all encoder layers train in both arms.
    settings = config['supervised']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    run = OUT / 'runs' / f'fold{fold}-n{size}-seed{seed}-{kind}'
    run.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(config_sha256=config_hash, pretraining_sha256=ssl_hash,
                       fold=fold, size=size, seed=seed, kind=kind)
    if (run / 'report.json').exists():
        report = json.loads((run / 'report.json').read_text())
        if report['fingerprint'] != fingerprint or any(digest(run / k) != v for k, v in report['files_sha256'].items()):
            raise ValueError('Completed run changed')
        print('Already complete:', run.name)
        return
    start, history = resume(run, fingerprint, model, optimizer)
    stop = min(settings['epochs'], start + max_epochs)
    for epoch in range(start + 1, stop + 1):
        model.train()
        rng = torch.Generator().manual_seed(seed + epoch)
        for index in torch.randperm(len(images), generator=rng):
            optimizer.zero_grad()
            loss = F.mse_loss(model(images[index:index + 1]), y[index:index + 1])
            step(loss, model, optimizer)
        predicted = predictions(model, images)
        history.append(dict(epoch=epoch, mse=F.mse_loss(predicted, y).item(),
                            mae=(predicted * std + mean - values).abs().mean().item()))
        checkpoint(run, fingerprint, epoch, history, model, optimizer)
        print(run.name, history[-1], flush=True)
    if history:
        curve(history, run, 'mae')
    if stop < settings['epochs']:
        print(f'Paused at epoch {stop}; evaluation remains pending.')
        return
    # Evaluation is accessed only after the fixed training schedule is complete.
    evaluation = prepared_rows(fold, size, 'evaluation')
    if {r['source_participant_id'] for r in rows} & {r['source_participant_id'] for r in evaluation}:
        raise ValueError('Subject leakage')
    eval_images = tensors(evaluation)
    checkpoint_path = run / 'regressor.pt'
    save_torch(checkpoint_path, dict(weights=model.state_dict(), target_mean=mean, target_std=std,
                                   fingerprint=fingerprint))
    restored = WholeBrainCNN()
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location='cpu')['weights'])
    pred = predictions(model, eval_images)
    if not torch.equal(pred, predictions(restored, eval_images)):
        raise ValueError('Checkpoint prediction round-trip mismatch')
    pred = pred * std + mean
    results = [dict(source_participant_id=r['source_participant_id'], observed=float(r['IST_fluid']),
                    model=float(p), training_mean=mean) for r, p in zip(evaluation, pred)]
    if not all(math.isfinite(r['model']) for r in results):
        raise ValueError('Nonfinite evaluation prediction')
    atomic_write(run / 'predictions.csv', csv_bytes(results))
    atomic_write(run / 'training_manifest.csv', csv_bytes(rows))
    report = dict(fingerprint=fingerprint, epochs=stop, training_subjects=len(rows), evaluation_subjects=len(results),
                  target_mean=mean, target_std=std, units='IST_fluid points',
                  mae={k: sum(abs(r[k] - r['observed']) for r in results) / len(results) for k in ('model', 'training_mean')},
                  files_sha256={p: digest(run / p) for p in ('regressor.pt', 'predictions.csv', 'training_manifest.csv', 'history.csv')})
    json_write(run / 'report.json', report)
    print('Complete:', run, report['mae'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'overfit', 'pretrain', 'train', 'summarize'))
    parser.add_argument('--fold', type=int, choices=range(3), default=0)
    parser.add_argument('--size', type=int, choices=(15, 30, 60), default=15)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--kind', choices=('random', 'pretrained', 'both'), default='both')
    parser.add_argument('--all', action='store_true', help='Train all 54 fold/size/seed/model runs')
    parser.add_argument('--max-epochs', type=int, default=100000, help='Additional epochs per run this invocation; does not change the frozen schedule')
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error('--max-epochs must be positive')
    config, config_hash = load_config()
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    with lock(OUT / '.training.lock'):
        if args.stage == 'check':
            check(config, config_hash)
        elif args.stage == 'overfit':
            overfit(config_hash)
        elif args.stage == 'pretrain':
            pretrain(config, config_hash, args.max_epochs)
        elif args.stage == 'summarize':
            summarize(config_hash)
        else:
            jobs = [(f, n, s, k) for f in range(3) for n in (15, 30, 60) for s in (42, 43, 44)
                    for k in ('random', 'pretrained')] if args.all else [
                        (args.fold, args.size, args.seed, k) for k in
                        (('random', 'pretrained') if args.kind == 'both' else (args.kind,))]
            for fold, size, seed, kind in jobs:
                train_one(config, config_hash, fold, size, seed, kind, args.max_epochs)


if __name__ == '__main__':
    main()
