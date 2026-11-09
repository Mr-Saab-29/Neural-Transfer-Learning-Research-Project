#!/usr/bin/env python3
"""Multi-epoch frozen-encoder comparison and training-only memorization diagnostic."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from tempfile import mkdtemp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn

from aomic_validation_batch import CHECKPOINT, OUT, ROOT, csv_bytes, digest, read
from expanded_aomic_comparison import approved_training_rows, cohorts, SEEDS
from evaluate_aomic_validation import accepted_rows, evaluate
from smoke_aomic_transfer import regressor
from smoke_train_3d import new_model
from src.data.aomic_dataset import AOMICDataset
from src.data.dataset import project_path


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in zip(axes, ('mse', 'mae'),
                                   ('Training MSE (standardized scores)', 'Training MAE (IST_fluid points)')):
        axis.plot([r['epoch'] for r in history], [r[metric] for r in history])
        axis.set_xlabel('Epoch')
        axis.set_title(title)
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(rows, run, source_path, seed, epochs, overfit=False, overfit_solver='adam'):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    manifest = run / 'approved_aomic.csv'
    manifest.write_bytes(csv_bytes(rows))
    dataset = AOMICDataset(manifest, 'train')
    patches, scores = [], []
    for index, row in enumerate(dataset.rows):
        metadata = project_path(ROOT, row['tensor_path']).parent / 'metadata.json'
        if digest(metadata) != row['metadata_sha256']:
            raise ValueError('Training approval checksum mismatch')
        item = dataset[index]
        patches.append(item['image'][:, 48:80, 48:80, 48:80].clone())
        scores.append(item['target'])
    images = torch.stack(patches)
    targets = torch.stack(scores)
    target_mean = targets.mean().item()
    target_std = targets.std(unbiased=False).item()
    if target_std < 1e-6:
        raise ValueError('Selected training targets have insufficient variance')
    targets_z = (targets - target_mean) / target_std
    source = new_model()
    if source_path is not None:
        source.load_state_dict(torch.load(source_path, map_location='cpu', weights_only=True))
    model = regressor()
    model[0].load_state_dict(source[0].state_dict())
    model[0].requires_grad_(overfit)
    initial_head = model[-1].weight.detach().clone()
    if overfit:
        inputs = images
        predictor = model
    else:
        # The frozen encoder is deterministic. Cache its features once per run.
        with torch.no_grad():
            inputs = torch.cat([model[:-1](image.unsqueeze(0)) for image in images])
        predictor = model[-1]
    parameters = [p for p in model.parameters() if p.requires_grad]
    use_lbfgs = overfit and overfit_solver == 'lbfgs'
    if use_lbfgs:
        optimizer = torch.optim.LBFGS(parameters, lr=.5, max_iter=10,
                                     history_size=20, line_search_fn='strong_wolfe')
    else:
        optimizer = torch.optim.Adam(parameters, lr=.001)
    generator = torch.Generator().manual_seed(seed)
    history = []
    with (run / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'mse', 'mae'])
        writer.writeheader()
        for epoch in range(epochs + 1):
            if epoch:
                predictor.train()
                if use_lbfgs:
                    def closure():
                        optimizer.zero_grad()
                        loss = nn.functional.mse_loss(predictor(inputs).squeeze(1), targets_z)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite L-BFGS loss')
                        loss.backward()
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                            raise ValueError('Missing or nonfinite L-BFGS gradient')
                        return loss
                    optimizer.step(closure)
                    order = []
                else:
                    order = torch.randperm(len(dataset), generator=generator).tolist()
                for index in order:
                    prediction = predictor(inputs[index:index+1]).squeeze(1)
                    loss = nn.functional.mse_loss(prediction, targets_z[index:index+1])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    optimizer.zero_grad()
                    loss.backward()
                    for parameter in model.parameters():
                        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                            raise ValueError('Missing or nonfinite gradient')
                    optimizer.step()
            predictor.eval()
            with torch.no_grad():
                predictions = torch.cat([predictor(x.unsqueeze(0)).flatten() for x in inputs])
                mse = (predictions - targets_z).square().mean().item()
                mae = ((predictions - targets_z) * target_std).abs().mean().item()
            if not torch.isfinite(predictions).all() or not all(map(math.isfinite, (mse, mae))):
                raise ValueError('Nonfinite epoch predictions or metrics')
            record = dict(epoch=epoch, mse=mse, mae=mae)
            history.append(record)
            writer.writerow(record)
            stream.flush()
            if epoch % 10 == 0 or epoch == epochs:
                print(f'{run.name} epoch {epoch}/{epochs}: training MSE={mse:.5f}, MAE={mae:.3f}', flush=True)
            if overfit and epoch > 0 and mae < 1.0:
                print(f'Overfit criterion reached at epoch {epoch}: MAE={mae:.3f}', flush=True)
                break
    if torch.equal(initial_head, model[-1].weight):
        raise ValueError('Regression head did not update')
    if not overfit and any(not torch.equal(a, b) for a, b in zip(source[0].parameters(), model[0].parameters())):
        raise ValueError('Frozen encoder changed')
    checkpoint = run / 'regressor.pt'
    torch.save(dict(weights=model.state_dict(), target_mean=target_mean, target_std=target_std), checkpoint)
    restored = regressor()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    restored.load_state_dict(saved['weights'])
    with torch.no_grad():
        expected = model.eval()(images[:1]) * target_std + target_mean
        actual = restored.eval()(images[:1]) * saved['target_std'] + saved['target_mean']
        if not torch.equal(expected, actual):
            raise ValueError('Checkpoint round-trip prediction mismatch')
    plot_history(history, run / 'training_curves.png')
    prediction_rows = [
        dict(participant_id=row['participant_id'], observed=float(targets[i]),
             predicted=float(predictions[i] * target_std + target_mean))
        for i, row in enumerate(dataset.rows)]
    (run / 'training_predictions.csv').write_bytes(csv_bytes(prediction_rows))
    summary = dict(seed=seed, epochs=history[-1]['epoch'], requested_epochs=epochs,
                   training_subjects=len(rows), batch_size=len(rows) if use_lbfgs else 1,
                   optimizer='LBFGS' if use_lbfgs else 'Adam',
                   learning_rate=.5 if use_lbfgs else .001,
                   encoder_trainable=overfit, patch_shape=[32,32,32],
                   initial=history[0], final=history[-1], checkpoint_max_difference=0.,
                   target_mean=target_mean, target_std=target_std,
                   manifest_sha256=digest(manifest),
                   source_checkpoint_sha256=digest(source_path) if source_path else None)
    if overfit:
        summary['overfit_pass'] = history[-1]['mae'] < 1.0
        summary['criterion'] = 'Final training MAE below 1 IST_fluid point'
    (run / 'training_report.json').write_text(json.dumps(summary, indent=2)+'\n')
    return checkpoint, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('overfit', 'compare'))
    parser.add_argument('--epochs', type=int, help='Default 300 for overfit, 50 for comparison')
    parser.add_argument('--overfit-solver', choices=('adam', 'lbfgs'), default='adam',
                        help='L-BFGS uses all four subjects per optimizer step, up to 10 inner iterations')
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (300 if args.mode == 'overfit' else 50)
    if epochs < 1:
        parser.error('--epochs must be positive')
    if args.mode != 'overfit' and args.overfit_solver != 'adam':
        parser.error('--overfit-solver applies only to overfit mode')
    original, selected, labels = cohorts()
    rows = approved_training_rows(original, selected, labels)
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Only frozen training subjects may enter optimization')
    source_report = json.loads((CHECKPOINT.parent/'report.json').read_text())
    source = Path(source_report['encoder_checkpoint'])
    if digest(source) != source_report['encoder_checkpoint_sha256']:
        raise ValueError('Pretrained encoder checkpoint changed')
    output = ROOT / 'results/aomic_epochs'
    output.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix=args.mode+'-', dir=output))
    if args.mode == 'overfit':
        # IDs determine membership, independently of targets.
        rows.sort(key=lambda r: hashlib.sha256(('overfit-v1|'+r['source_participant_id']).encode()).digest())
        if len(rows) < 4:
            raise ValueError('At least four approved training subjects are required')
        _, summary = train(rows[:4], run, source, 42, epochs, overfit=True,
                           overfit_solver=args.overfit_solver)
        print(json.dumps(summary, indent=2), '\nSaved:', run)
        return
    validation = read(OUT/'selection.csv')
    reviews = read(OUT/'review.csv')
    if len(accepted_rows(validation, reviews)) != 12:
        raise ValueError('Expected the same 12 approved validation subjects')
    if {r['source_participant_id'] for r in rows} & {r['source_participant_id'] for r in validation}:
        raise ValueError('Training/validation overlap')
    reference = json.loads((OUT/'eval-1g7xjyxg/report.json').read_text())
    hashes = {name: digest(OUT/(name+'.csv')) for name in ('selection','review')}
    if any(reference[k+'_sha256'] != value for k, value in hashes.items()):
        raise ValueError('Validation batch differs from the pilot comparison')
    results = []
    for seed in SEEDS:
        result = dict(seed=seed)
        for mode, encoder in [('pretrained',source), ('random_encoder',None)]:
            branch = run / f'{seed}-{mode}'
            branch.mkdir()
            checkpoint, _ = train(rows, branch, encoder, seed, epochs)
            scores, _ = evaluate(checkpoint, branch)
            if any(scores[k+'_sha256'] != value for k,value in hashes.items()):
                raise ValueError('Validation changed during this comparison')
            result[mode] = scores['model_mae']
            if 'training_mean' in result and result['training_mean'] != scores['baseline_mae']:
                raise ValueError('Paired runs have different baselines')
            result['training_mean'] = scores['baseline_mae']
        results.append(result)
    summary = dict(epochs=epochs, training_subjects=len(rows), validation_subjects=12,
                   units='IST_fluid points', per_seed=results,
                   mean_mae={k: mean(r[k] for r in results) for k in ('pretrained','random_encoder','training_mean')},
                   sd_mae={k: stdev(r[k] for r in results) for k in ('pretrained','random_encoder')},
                   pretrained_beats_baseline_seeds=sum(r['pretrained'] < r['training_mean'] for r in results),
                   pretrained_beats_random_seeds=sum(r['pretrained'] < r['random_encoder'] for r in results))
    (run/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),'\nSaved:',run)


if __name__ == '__main__':
    main()
