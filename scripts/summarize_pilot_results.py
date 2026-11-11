#!/usr/bin/env python3
"""Create figures and exploratory paired bootstrap intervals from saved results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('training_mean', 'random_encoder', 'pretrained')
LABELS = ('Training mean', 'Random encoder', 'Pretrained encoder')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def paired_bootstrap(differences, repetitions=10000, seed=42):
    """Rows are subjects; columns are paired error differences between methods."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2 or len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError('Need finite paired differences for at least two subjects')
    if repetitions < 1000:
        raise ValueError('Use at least 1000 bootstrap repetitions')
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, differences.shape[1]))
    for start in range(0, repetitions, 256):
        stop = min(start + 256, repetitions)
        indices = rng.integers(len(differences), size=(stop - start, len(differences)))
        draws[start:stop] = differences[indices].mean(axis=1)
    return differences.mean(axis=0), np.quantile(draws, [.025, .975], axis=0)


def validate_predictions(rows, report, names):
    if len(rows) != report['accepted'] or len(rows) < 2:
        raise ValueError('Prediction count does not match accepted subjects')
    for key in ('participant_id', 'source_participant_id'):
        ids = [row[key] for row in rows]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'Missing or duplicate {key}')
    observed = np.array([float(row['observed']) for row in rows])
    predictions = np.array([[float(row[name]) for name in names] for row in rows])
    if not np.isfinite(observed).all() or not np.isfinite(predictions).all():
        raise ValueError('Non-finite scores or predictions')
    errors = np.abs(predictions - observed[:, None])
    for i, name in enumerate(names):
        if not np.isclose(errors[:, i].mean(), report['mae'][name], rtol=0, atol=1e-5):
            raise ValueError(f'Predictions disagree with reported MAE: {name}')
    return observed, predictions, errors


def save_figure(fig, output, name):
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{extension}', dpi=180)
    plt.close(fig)


def summarize(final, output, repetitions, seed):
    if output.exists():
        raise FileExistsError(f'Choose a new --output; already exists: {output}')
    inputs = {}

    def track(path):
        inputs[str(path)] = digest(path)
        return path

    report = json.loads(track(final / 'report.json').read_text())
    plan_path = track(final / 'plan_snapshot.json')
    plan = json.loads(plan_path.read_text())
    review_path = track(final / 'review_snapshot.csv')
    if digest(plan_path) != report['inputs']['plan_sha256'] or digest(review_path) != report['inputs']['review_sha256']:
        raise ValueError('Final snapshots do not match report hashes')
    comparison_dir = ROOT / plan['comparison']
    comparison_path = track(comparison_dir / 'comparison.json')
    if digest(comparison_path) != plan['comparison_sha256']:
        raise ValueError('Comparison differs from frozen plan')
    comparison = json.loads(comparison_path.read_text())
    seeds = sorted({model['seed'] for model in plan['models']})
    names = ['training_mean'] + [f'{kind}_{s}' for kind in KINDS[1:] for s in seeds]
    if not seeds or set(names[1:]) != {m['name'] for m in plan['models']}:
        raise ValueError('Expected paired pretrained/random models for each seed')
    rows = read_csv(track(final / 'predictions.csv'))
    observed, predictions, errors = validate_predictions(rows, report, names)
    reviews = read_csv(review_path)
    accepted_ids = {Path(r['hf_path']).parent.parent.name for r in reviews if r['decision'] == 'accept'}
    if accepted_ids != {r['participant_id'] for r in rows}:
        raise ValueError('Predictions do not match accepted review subjects')
    if len(reviews) != report['selected'] or sum(r['decision'] == 'exclude' for r in reviews) != report['excluded']:
        raise ValueError('Review counts disagree with report')
    # Average absolute errors, NOT predictions: this is not ensemble evaluation.
    grouped = np.column_stack([errors[:, 0]] + [
        errors[:, [names.index(f'{kind}_{s}') for s in seeds]].mean(axis=1)
        for kind in KINDS[1:]])
    contrasts = ('pretrained_minus_training_mean', 'random_encoder_minus_training_mean', 'pretrained_minus_random_encoder')
    differences = np.column_stack((grouped[:, 2] - grouped[:, 0], grouped[:, 1] - grouped[:, 0], grouped[:, 2] - grouped[:, 1]))
    estimates, intervals = paired_bootstrap(differences, repetitions, seed)
    uncertainty = {name: dict(mae_difference=float(estimates[i]), ci95_low=float(intervals[0, i]), ci95_high=float(intervals[1, i]))
                   for i, name in enumerate(contrasts)}
    histories = {}
    for s in seeds:
        for kind in KINDS[1:]:
            history = read_csv(track(comparison_dir / f'{s}-{kind}' / 'history.csv'))
            values = np.array([[float(r[k]) for k in ('epoch', 'mse', 'mae')] for r in history])
            if (values.shape != (comparison['epochs'] + 1, 3) or not np.isfinite(values).all()
                    or not np.array_equal(values[:, 0], np.arange(comparison['epochs'] + 1))):
                raise ValueError(f'Invalid/incomplete history: {s}-{kind}')
            histories[s, kind] = values
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.pilot-summary-', dir=output.parent))
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for (s, kind), values in histories.items():
            for ax, column in zip(axes, (1, 2)):
                ax.plot(values[:, 0], values[:, column], label=f'{kind}, seed {s}')
                ax.set_xlabel('Epoch')
                ax.grid(alpha=.2)
        axes[0].set_ylabel('Training MSE (standardized target)')
        axes[1].set_ylabel('Training MAE (IST_fluid points)')
        axes[1].legend(fontsize=7)
        save_figure(fig, temporary, 'training_curves')
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(3)
        ax.bar(x - .18, [comparison['mean_mae'][k] for k in KINDS], .36, label=f"Validation (n={comparison['validation_subjects']})")
        ax.bar(x + .18, grouped.mean(axis=0), .36, label=f'Test (n={len(rows)})')
        ax.set_xticks(x, LABELS)
        ax.set_ylabel('MAE (IST_fluid points)')
        ax.set_title('CNN values are mean MAE across fixed seeds')
        ax.legend()
        save_figure(fig, temporary, 'validation_test_mae')
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
        low = min(observed.min(), predictions.min()) - 5
        high = max(observed.max(), predictions.max()) + 5
        for ax, kind, label in zip(axes, KINDS, LABELS):
            columns = ['training_mean'] if kind == 'training_mean' else [f'{kind}_{s}' for s in seeds]
            for name in columns:
                ax.scatter(observed, predictions[:, names.index(name)], s=12, alpha=.45, label=name)
            ax.plot([low, high], [low, high], '--', color='gray')
            ax.set(xlim=(low, high), ylim=(low, high), title=label, xlabel='Observed IST_fluid')
            ax.legend(fontsize=6)
        axes[0].set_ylabel('Predicted IST_fluid')
        save_figure(fig, temporary, 'test_predictions')
        fig, ax = plt.subplots(figsize=(9, 4))
        for i in range(len(contrasts)):
            ax.plot(intervals[:, i], [i, i], color='tab:blue')
            ax.scatter(estimates[i], i, color='tab:blue')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_yticks(range(len(contrasts)), [c.replace('_', ' ') for c in contrasts])
        ax.set_xlabel('MAE difference (points); negative favors first method')
        ax.set_title('Paired subject bootstrap: exploratory 95% percentile intervals')
        save_figure(fig, temporary, 'paired_mae_differences')
        result = dict(test_subjects=len(rows), seeds=seeds, bootstrap_repetitions=repetitions,
                      bootstrap_seed=seed, units=report['units'], comparisons=uncertainty,
                      test_mean_mae=dict(zip(KINDS, grouped.mean(axis=0).tolist())),
                      input_sha256=inputs, script_sha256=digest(Path(__file__)))
        (temporary / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        with (temporary / 'paired_bootstrap.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['comparison', 'mae_difference', 'ci95_low', 'ci95_high'])
            writer.writeheader()
            writer.writerows(dict(comparison=name, **values) for name, values in uncertainty.items())
        table = '\n'.join(f"| {label} | {comparison['mean_mae'][kind]:.4f} | {grouped[:, i].mean():.4f} |" for i, (kind, label) in enumerate(zip(KINDS, LABELS)))
        ci_table = '\n'.join(f"| {name.replace('_', ' ')} | {v['mae_difference']:.4f} | [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}] |" for name, v in uncertainty.items())
        exclusions = '\n'.join(f"- `{item['hf_path']}`: {item['reason']}" for item in report['exclusions']) or '- None.'
        (temporary / 'pilot_summary.md').write_text(f'''# MRI transfer-learning pilot

## Question and design

Does MRI pretraining improve AOMIC IST_fluid prediction over random frozen CNN
features and the training-set mean? This comparison used {comparison['training_subjects']}
labeled training subjects, {comparison['validation_subjects']} validation subjects,
{comparison['epochs']} epochs, and seeds {seeds}. IXI/NKI supplied pilot pretraining;
AOMIC supplied downstream labels. The shallow CNN used central 32×32×32 patches,
a frozen encoder, and a trained regression head. Preprocessing included SynthStrip,
RAS orientation, 2 mm resampling, brain-centered crop/pad to 128³, masked
normalization, and visual QC. No transformer was evaluated.
The frozen primary model was `{plan['primary']}`.

## Results

Of {report['selected']} selected test subjects, {report['accepted']} were accepted
and {report['excluded']} excluded before scoring. All models used the same subjects.
CNN entries average MAEs across fixed seeds, not ensemble predictions.

| Method | Validation MAE | Test MAE |
|---|---:|---:|
{table}

## Exploratory uncertainty

| Contrast (first minus second) | MAE difference | 95% percentile interval |
|---|---:|---:|
{ci_table}

Intervals use {repetitions} paired bootstrap resamples of subjects, seed {seed}.
All predictions for a subject remain paired. Absolute errors are averaged across
fixed seeds within subject before computing contrasts. Negative favors the first
method. These post-evaluation intervals are exploratory, unadjusted for multiple
comparisons, and conditional on fitted models and the QC-accepted cohort. They do
not capture training-sample or retraining uncertainty. Resampling assumes independent
subjects; family/site dependence can affect coverage. Intervals crossing zero do
not establish equivalence. Across-seed SD measures a different source of variation.

## Exclusions

{exclusions}

## Interpretation and limitations

Interpret the effect estimates and intervals together; a small numerical advantage
alone is not evidence of reliable benefit. The recorded pilot's pretrained model
had worse MAE than both comparators in every seed. This limits the conclusion to
no observed pretraining benefit under the tested settings. Small training and
validation samples, limited pretraining, a frozen shallow CNN, and central patches
limit broader conclusions. The four-subject overfit check demonstrates fitting
capacity, not generalization. Higher test than validation error does not identify
its cause. These results do not establish clinical utility.

## Next study and provenance

Define a new evaluation protocol before further development. Investigate broader
brain coverage, fine-tuning, and larger labeled subsets before a transformer
comparison. This viewed test set cannot be described as untouched for changes
informed by these results. Input hashes, script hash, and bootstrap settings are
in `summary.json`; figures are PNG/PDF. This summary uses saved outputs only.
''')
        if any(digest(Path(path)) != value for path, value in inputs.items()):
            raise ValueError('An input changed during summarization')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', type=Path, default=ROOT / 'results/aomic_final_test_v1/final')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pilot_summary_v1')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(summarize(args.final.resolve(), args.output.resolve(), args.bootstrap, args.seed))


if __name__ == '__main__':
    main()
