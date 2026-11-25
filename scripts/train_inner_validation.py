#!/usr/bin/env python3
"""Separate 48-fit/12-inner-validation experiment with matched epoch-50 control."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F
from followup_common import (ROOT, WholeBrainCNN, load_config, read, csv_bytes, digest,
                             lock, atomic_write, save_torch, json_write)
from train_wholebrain import (prepared_rows, pretrained_state, require_overfit, tensors,
                             targets, predictions, step)

OUT = ROOT / 'results/aomic_inner_validation_v1'


def split_inner(rows, fold):
    if len(rows) != 60 or len({r['source_participant_id'] for r in rows}) != 60:
        raise ValueError('Expected 60 unique outer-training subjects')
    ranked = sorted(rows, key=lambda r: (hashlib.sha256(
        f"inner-v1-42|{fold}|{r['source_participant_id']}".encode()).hexdigest(), r['source_participant_id']))
    return ranked[12:], ranked[:12]


def update_selection(state, mae, epoch, weights, patience=10, min_delta=.1):
    """Freeze the selected checkpoint permanently when patience expires."""
    if not math.isfinite(mae):
        raise ValueError('Nonfinite inner validation MAE')
    if state['stop_epoch'] is not None:
        return
    if state['best_mae'] is None or mae < state['best_mae'] - min_delta:
        state.update(best_mae=mae, best_epoch=epoch, best_weights=copy.deepcopy(weights), bad_epochs=0)
    else:
        state['bad_epochs'] += 1
    if state['bad_epochs'] >= patience:
        state['stop_epoch'] = epoch


def freeze():
    config, config_hash = load_config()
    require_overfit(config_hash)
    _, ssl_hash = pretrained_state(config_hash)
    if config['supervised']['epochs'] != 50:
        raise ValueError('Expected original 50-epoch setup')
    files = {}
    for fold in range(3):
        fitting, inner = split_inner(prepared_rows(fold, 60, 'train'), fold)
        outer = prepared_rows(fold, 60, 'evaluation')
        groups = [{r['source_participant_id'] for r in rows} for rows in (fitting, inner, outer)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError('Subject leakage')
        for role, rows in (('fit', fitting), ('inner', inner), ('outer', outer)):
            files[f'fold{fold}_{role}.csv'] = csv_bytes([dict(r, role=role) for r in rows])
    plan = dict(parent_config_sha256=config_hash, pretraining_sha256=ssl_hash,
        script_sha256=digest(Path(__file__)), max_epochs=50, patience=10, min_delta_points=.1,
        fitting_subjects=48, inner_subjects=12, outer_subjects=30, seeds=[42, 43, 44],
        optimizer='Adam', learning_rate=.001, batch_size=1, augmentation='none', threads=2,
        selection='First trained epoch initializes best; improvement must exceed 0.1 MAE points. Ties keep earlier checkpoint. After 10 non-improving epochs, freeze best checkpoint. If patience never expires, select best through epoch 50.',
        control='Same 48 fitting subjects, initialization and trajectory; continue to epoch 50 for fixed control. Outer scoring happens only after both policies are fixed.',
        normalization='Mean and population SD fitted on the 48 fitting subjects only; baseline is their mean.',
        primary='Early-selected minus fixed-50 out-of-fold MAE, separately for random and pretrained initialization; paired secondary comparison of pretrained versus random under each policy.',
        interpretation='Follow-up development study after viewing prior results, not independent confirmation. Twelve validation labels count within the 60-label budget. No refit on all 60 subjects.',
        files_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    files['plan.json'] = (json.dumps(plan, indent=2) + '\n').encode()
    destination = OUT / 'frozen'
    if destination.exists():
        if any((destination / name).read_bytes() != data for name, data in files.items()):
            raise ValueError('Existing plan differs; preserved')
        print('Existing plan verified:', destination)
        return
    with tempfile.TemporaryDirectory(dir=OUT, prefix='.freeze-') as directory:
        work = Path(directory)
        for name, data in files.items():
            (work / name).write_bytes(data)
        work.rename(destination)
    print('Frozen:', destination)


def load_plan():
    _, parent_hash = load_config()
    _, ssl_hash = pretrained_state(parent_hash)
    plan = json.loads((OUT / 'frozen/plan.json').read_text())
    if (plan['parent_config_sha256'] != parent_hash or plan['pretraining_sha256'] != ssl_hash
            or plan['script_sha256'] != digest(Path(__file__))):
        raise ValueError('Frozen implementation or parent experiment changed')
    for name, checksum in plan['files_sha256'].items():
        if digest(OUT / 'frozen' / name) != checksum:
            raise ValueError('Frozen subject manifest changed')
    return plan, digest(OUT / 'frozen/plan.json')


def train(plan, plan_hash, fold, seed, kind, max_epochs):
    run = OUT / 'runs' / f'fold{fold}-seed{seed}-{kind}'
    run.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(plan_sha256=plan_hash, fold=fold, seed=seed, kind=kind)
    if (run / 'report.json').exists():
        validate_report(run, fingerprint)
        print('Already complete:', run.name)
        return
    fitting = read(OUT / f'frozen/fold{fold}_fit.csv')
    inner = read(OUT / f'frozen/fold{fold}_inner.csv')
    x, vx = tensors(fitting), tensors(inner)
    y, mean, std = targets(fitting)
    vy = torch.tensor([float(r['IST_fluid']) for r in inner])
    z = (y - mean) / std
    torch.manual_seed(seed)
    model = WholeBrainCNN()
    if kind == 'pretrained':
        state, _ = pretrained_state(plan['parent_config_sha256'])
        model.encoder.load_state_dict(state['encoder'])
    optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'])
    selection = dict(best_mae=None, best_epoch=None, best_weights=None, bad_epochs=0, stop_epoch=None)
    start, history = 0, []
    if (run / 'last.pt').exists():
        saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Checkpoint inputs changed')
        model.load_state_dict(saved['weights'])
        optimizer.load_state_dict(saved['optimizer'])
        start, history, selection = saved['epoch'], saved['history'], saved['selection']
    end = min(plan['max_epochs'], start + max_epochs)
    for epoch in range(start + 1, end + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        for index in torch.randperm(len(x), generator=generator):
            optimizer.zero_grad()
            step(F.mse_loss(model(x[index:index + 1]), z[index:index + 1]), model, optimizer)
        train_mae = (predictions(model, x) * std + mean - y).abs().mean().item()
        # Inner labels stop being queried once the early policy stops. The
        # remaining trajectory is only the matched fixed-epoch control.
        inner_mae = None
        if selection['stop_epoch'] is None:
            inner_mae = (predictions(model, vx) * std + mean - vy).abs().mean().item()
            update_selection(selection, inner_mae, epoch, model.state_dict(), plan['patience'], plan['min_delta_points'])
        history.append(dict(epoch=epoch, training_mae=train_mae, inner_mae=inner_mae))
        save_torch(run / 'last.pt', dict(fingerprint=fingerprint, epoch=epoch,
                   weights=model.state_dict(), optimizer=optimizer.state_dict(), selection=selection, history=history))
        atomic_write(run / 'history.csv', csv_bytes(history))
        print(run.name, history[-1], 'selected epoch', selection['best_epoch'], flush=True)
    if end < plan['max_epochs']:
        print('Partial run saved; repeat command to resume. Outer evaluation pending.')
        return
    # No outer labels or images are used until both checkpoint policies are fixed.
    outer = read(OUT / f'frozen/fold{fold}_outer.csv')
    ox = tensors(outer)
    results = [dict(source_participant_id=r['source_participant_id'], observed=float(r['IST_fluid']),
                    training_mean=mean) for r in outer]
    for policy, weights in (('fixed50', model.state_dict()), ('early', selection['best_weights'])):
        chosen = WholeBrainCNN()
        chosen.load_state_dict(weights)
        pred = predictions(chosen, ox) * std + mean
        if not torch.isfinite(pred).all():
            raise ValueError('Nonfinite prediction')
        save_torch(run / f'{policy}.pt', dict(weights=weights, target_mean=mean, target_std=std,
                   fingerprint=fingerprint, epoch=50 if policy == 'fixed50' else selection['best_epoch']))
        restored = WholeBrainCNN()
        restored.load_state_dict(torch.load(run / f'{policy}.pt', map_location='cpu', weights_only=True)['weights'])
        if not torch.equal(pred, predictions(restored, ox) * std + mean):
            raise ValueError('Checkpoint round-trip mismatch')
        for row, value in zip(results, pred):
            row[policy] = value.item()
    atomic_write(run / 'predictions.csv', csv_bytes(results))
    report = dict(fingerprint=fingerprint, fitting_subjects=48, inner_subjects=12, outer_subjects=30,
        selected_epoch=selection['best_epoch'], stop_epoch=selection['stop_epoch'],
        selection_end_epoch=selection['stop_epoch'] or 50, best_inner_mae=selection['best_mae'],
        target_mean=mean, target_std=std,
        mae={k: sum(abs(r[k]-r['observed']) for r in results)/30 for k in ('early', 'fixed50', 'training_mean')},
        files_sha256={name: digest(run / name) for name in ('early.pt', 'fixed50.pt', 'predictions.csv', 'history.csv')})
    json_write(run / 'report.json', report)
    print(json.dumps(report, indent=2))


def validate_report(run, fingerprint):
    report = json.loads((run / 'report.json').read_text())
    if report['fingerprint'] != fingerprint or any(digest(run / k) != v for k, v in report['files_sha256'].items()):
        raise ValueError('Completed run changed')
    return report


def summarize(plan_hash):
    per_seed, combined, selections = [], [], []
    for seed in (42, 43, 44):
        for kind in ('random', 'pretrained'):
            rows = []
            for fold in range(3):
                run = OUT / 'runs' / f'fold{fold}-seed{seed}-{kind}'
                report = validate_report(run, dict(plan_sha256=plan_hash, fold=fold, seed=seed, kind=kind))
                part = read(run / 'predictions.csv')
                expected = {r['source_participant_id'] for r in read(OUT / f'frozen/fold{fold}_outer.csv')}
                if len(part) != 30 or {r['source_participant_id'] for r in part} != expected:
                    raise ValueError('Invalid outer predictions')
                rows.extend(part)
                combined.extend(dict(r, fold=fold, seed=seed, kind=kind) for r in part)
                selections.append(dict(fold=fold, seed=seed, kind=kind, selected_epoch=report['selected_epoch'], stop_epoch=report['stop_epoch']))
            if len({r['source_participant_id'] for r in rows}) != 90:
                raise ValueError('Duplicate out-of-fold subject')
            mae = {k: sum(abs(float(r[k])-float(r['observed'])) for r in rows)/90 for k in ('early', 'fixed50', 'training_mean')}
            per_seed.append(dict(seed=seed, kind=kind, subjects=90, **mae,
                                 early_minus_fixed50=mae['early']-mae['fixed50']))
    report = dict(plan_sha256=plan_hash, units='IST_fluid points', per_seed=per_seed,
        mean_over_seeds=[dict(kind=k, **{metric: sum(r[metric] for r in per_seed if r['kind']==k)/3
             for metric in ('early', 'fixed50', 'training_mean', 'early_minus_fixed50')}) for k in ('random', 'pretrained')],
        selections=selections, interpretation='Development comparison on the same 48 fitting subjects; 12 additional labels used for selection. No significance claim.')
    atomic_write(OUT / 'out_of_fold_predictions.csv', csv_bytes(combined))
    json_write(OUT / 'comparison.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'train', 'summarize'))
    parser.add_argument('--fold', type=int, choices=range(3), default=0)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--kind', choices=('random', 'pretrained', 'both'), default='both')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=50)
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error('--max-epochs must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with lock(OUT / '.run.lock'):
        if args.stage == 'freeze':
            freeze()
            return
        plan, plan_hash = load_plan()
        if args.stage == 'summarize':
            summarize(plan_hash)
            return
        jobs = [(f, s, k) for f in range(3) for s in (42, 43, 44) for k in ('random', 'pretrained')] if args.all else [
            (args.fold, args.seed, k) for k in (('random', 'pretrained') if args.kind == 'both' else (args.kind,))]
        for fold, seed, kind in jobs:
            train(plan, plan_hash, fold, seed, kind, args.max_epochs)


if __name__ == '__main__':
    main()
