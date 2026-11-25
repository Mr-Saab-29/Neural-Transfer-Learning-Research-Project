#!/usr/bin/env python3
"""Read-only follow-up analysis; conditional resampling is not CV inference."""
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
SIZES = (15, 30, 60)
SEEDS = (42, 43, 44)
KINDS = ('random', 'pretrained')
CAVEAT = ('Conditional fold-stratified subject resampling of fixed out-of-fold predictions. '
          'Overlapping training sets induce dependence this resampling does not capture. '
          'Percentile ranges are descriptive sensitivity ranges, not calibrated confidence '
          'intervals for cross-validation or retraining performance. No significance or '
          'equivalence claim; seeds and folds are not independent replicates.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sensitivity(differences, folds, repetitions=10000, seed=42):
    differences = np.asarray(differences, dtype=float)
    folds = np.asarray(folds)
    if differences.ndim != 2 or len(differences) != len(folds) or not np.isfinite(differences).all():
        raise ValueError('Invalid paired subject differences')
    groups = [np.flatnonzero(folds == f) for f in range(3)]
    if any(len(g) != 30 for g in groups) or repetitions < 1000:
        raise ValueError('Expected 3 folds of 30 subjects and at least 1000 resamples')
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, differences.shape[1]))
    for start in range(0, repetitions, 256):
        count = min(256, repetitions - start)
        indices = np.concatenate([rng.choice(g, size=(count, len(g))) for g in groups], axis=1)
        samples[start:start + count] = differences[indices].mean(axis=1)
    return np.quantile(samples, [.025, .975], axis=0)


def spread(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    std = float(predicted.std())
    target_std = float(observed.std())
    return dict(prediction_sd=std, observed_sd=target_std,
                sd_ratio=std / target_std if target_std > 0 else None,
                pearson_r=float(np.corrcoef(observed, predicted)[0, 1])
                if std > 1e-12 and target_std > 1e-12 else None,
                bias=float((predicted - observed).mean()))


def analyze(source, output, repetitions, seed):
    if output.exists():
        raise FileExistsError(f'Choose a new --output; existing analysis preserved: {output}')
    hashes = {}

    def track(path):
        hashes[str(path)] = digest(path)
        return path

    def load(path):
        return json.loads(track(path).read_text())

    config_path = source / 'ready/config.json'
    config = load(config_path)
    summary = load(source / 'comparison.json')
    if summary['config_sha256'] != digest(config_path):
        raise ValueError('Configuration hash mismatch')
    if digest(track(source / 'pretraining/encoder.pt')) != summary['pretraining_sha256']:
        raise ValueError('Pretraining checkpoint mismatch')
    for name, checksum in config['files_sha256'].items():
        if digest(track(source / 'ready' / name)) != checksum:
            raise ValueError('Approved setup changed')
    cohort = read(source / 'ready/approved_cohort.csv')
    by_id = {r['source_participant_id']: r for r in cohort}
    ids = sorted(by_id)
    if len(cohort) != len(ids) or len(ids) != 90:
        raise ValueError('Expected 90 unique approved subjects')
    folds = np.array([int(by_id[s]['fold']) for s in ids])
    observed = np.array([float(by_id[s]['IST_fluid']) for s in ids])
    if not np.isfinite(observed).all() or any(np.sum(folds == f) != 30 for f in range(3)):
        raise ValueError('Invalid targets or fold allocation')
    combined = read(track(source / 'out_of_fold_predictions.csv'))
    index = {(int(r['size']), int(r['seed']), r['source_participant_id']): r for r in combined}
    expected = {(n, s, sid) for n in SIZES for s in SEEDS for sid in ids}
    if len(index) != len(combined) or set(index) != expected:
        raise ValueError('Missing or duplicate combined predictions')
    histories, diagnostics, fold_differences, learning = {}, [], [], []
    errors = {}
    predictions = {}
    for n in SIZES:
        for s in SEEDS:
            records = [index[n, s, sid] for sid in ids]
            if not np.array_equal([float(r['observed']) for r in records], observed):
                raise ValueError('Observed targets disagree with approved cohort')
            baseline = np.array([float(r['training_mean']) for r in records])
            errors[n, s, 'baseline'] = np.abs(baseline - observed)
            for kind in KINDS:
                pred = np.array([float(r[kind]) for r in records])
                if not np.isfinite(pred).all() or not np.isfinite(baseline).all():
                    raise ValueError('Nonfinite predictions')
                predictions[n, s, kind] = pred
                errors[n, s, kind] = np.abs(pred - observed)
                entries = [r for r in summary['per_seed'] if (r['size'], r['seed'], r['kind']) == (n, s, kind)]
                if len(entries) != 1 or not np.isclose(entries[0]['model_mae'], errors[n, s, kind].mean(), atol=1e-5, rtol=0):
                    raise ValueError('Summary MAE mismatch')
                if not np.isclose(entries[0]['baseline_mae'], errors[n, s, 'baseline'].mean(), atol=1e-5, rtol=0):
                    raise ValueError('Baseline summary MAE mismatch')
                for fold in range(3):
                    run = source / 'runs' / f'fold{fold}-n{n}-seed{s}-{kind}'
                    report = load(run / 'report.json')
                    fingerprint = dict(config_sha256=summary['config_sha256'], pretraining_sha256=summary['pretraining_sha256'],
                                       fold=fold, size=n, seed=s, kind=kind)
                    if report['fingerprint'] != fingerprint or report['epochs'] != config['supervised']['epochs']:
                        raise ValueError('Run configuration mismatch')
                    for name, checksum in report['files_sha256'].items():
                        if digest(track(run / name)) != checksum:
                            raise ValueError(f'Changed run artifact: {run / name}')
                    saved = read(run / 'predictions.csv')
                    expected_ids = {sid for sid in ids if int(by_id[sid]['fold']) == fold}
                    if len(saved) != 30 or {r['source_participant_id'] for r in saved} != expected_ids:
                        raise ValueError('Invalid evaluation subjects')
                    for row in saved:
                        original = index[n, s, row['source_participant_id']]
                        if any(float(row[a]) != float(original[b]) for a, b in
                               (('model', kind), ('observed', 'observed'), ('training_mean', 'training_mean'))):
                            raise ValueError('Combined predictions disagree with run')
                    history = read(run / 'history.csv')
                    values = np.array([[float(r[k]) for k in ('epoch', 'mse', 'mae')] for r in history])
                    if values.shape != (report['epochs'], 3) or not np.isfinite(values).all() or not np.array_equal(values[:, 0], np.arange(1, report['epochs'] + 1)):
                        raise ValueError('Incomplete or invalid history')
                    histories[n, s, kind, fold] = values
                    mask = folds == fold
                    test_mae = float(errors[n, s, kind][mask].mean())
                    diagnostics.append(dict(size=n, seed=s, kind=kind, fold=fold,
                        first_epoch_training_mae=float(values[0, 2]), final_training_mae=float(values[-1, 2]),
                        evaluation_mae=test_mae, generalization_gap=test_mae - float(values[-1, 2]),
                        **spread(observed[mask], pred[mask])))
            for fold in range(3):
                mask = folds == fold
                fold_differences.append(dict(size=n, seed=s, fold=fold,
                    pretrained_minus_random=float((errors[n, s, 'pretrained'] - errors[n, s, 'random'])[mask].mean())))
        learning.append(dict(size=n, **{k: float(np.mean([errors[n, s, k].mean() for s in SEEDS])) for k in (*KINDS, 'baseline')}))
    contrast_names, columns = [], []
    for n in SIZES:
        for a, b in (('pretrained', 'random'), ('pretrained', 'baseline'), ('random', 'baseline')):
            contrast_names.append((n, f'{a}_minus_{b}'))
            columns.append(np.mean([errors[n, s, a] - errors[n, s, b] for s in SEEDS], axis=0))
    differences = np.column_stack(columns)
    ranges = sensitivity(differences, folds, repetitions, seed)
    contrasts = [dict(size=n, contrast=name, mean_difference=float(differences[:, i].mean()),
                     sensitivity_low=float(ranges[0, i]), sensitivity_high=float(ranges[1, i]))
                 for i, (n, name) in enumerate(contrast_names)]
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.analysis-', dir=output.parent))
    try:
        def save(fig, name):
            fig.tight_layout()
            for extension in ('png', 'pdf'):
                fig.savefig(work / f'{name}.{extension}', dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for kind, color in zip(KINDS, ('tab:blue', 'tab:orange')):
            for s in SEEDS:
                ax.plot(SIZES, [errors[n, s, kind].mean() for n in SIZES], color=color, alpha=.3, marker='.')
            ax.plot(SIZES, [r[kind] for r in learning], color=color, marker='o', label=kind)
        ax.plot(SIZES, [r['baseline'] for r in learning], '--', color='gray', label='Training mean')
        ax.set(xlabel='Labeled training subjects per fold', ylabel='Out-of-fold MAE (IST_fluid points)', xticks=SIZES)
        ax.legend(); save(fig, 'learning_curves')
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
        for i, kind in enumerate(KINDS):
            for j, n in enumerate(SIZES):
                ax = axes[i, j]
                curves = [histories[n, s, kind, f] for s in SEEDS for f in range(3)]
                for values in curves:
                    ax.plot(values[:, 0], values[:, 2], alpha=.25)
                ax.plot(curves[0][:, 0], np.mean([v[:, 2] for v in curves], axis=0), color='black', label='Mean training')
                ax.axhline(next(r[kind] for r in learning if r['size'] == n), color='red', linestyle='--', label='Final OOF MAE')
                ax.set(title=f'{kind}, n={n}', xlabel='Epoch', ylabel='MAE')
        axes[0, 0].legend(fontsize=7); save(fig, 'training_generalization')
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
        low = min(float(observed.min()), min(float(v.min()) for v in predictions.values())) - 5
        high = max(float(observed.max()), max(float(v.max()) for v in predictions.values())) + 5
        for i, kind in enumerate(KINDS):
            for j, n in enumerate(SIZES):
                ax = axes[i, j]
                for s in SEEDS:
                    ax.scatter(observed, predictions[n, s, kind], s=9, alpha=.35, label=f'Seed {s}')
                ax.plot([low, high], [low, high], '--', color='gray')
                ax.set(title=f'{kind}, n={n}', xlabel='Observed', ylabel='Predicted', xlim=(low, high), ylim=(low, high))
        axes[0, 0].legend(fontsize=7); save(fig, 'prediction_spread')
        fig, ax = plt.subplots(figsize=(8, 4))
        for fold in range(3):
            vals = [r for r in fold_differences if r['fold'] == fold]
            ax.scatter([SIZES.index(r['size']) + (fold - 1) * .15 for r in vals],
                       [r['pretrained_minus_random'] for r in vals], label=f'Fold {fold}', alpha=.8)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(range(3), SIZES)
        ax.set(xlabel='Training subjects per fold', ylabel='Pretrained minus random MAE', title='Paired fold/seed differences; dependent observations')
        ax.legend(); save(fig, 'paired_fold_differences')
        write_csv(work / 'run_diagnostics.csv', diagnostics)
        write_csv(work / 'paired_fold_differences.csv', fold_differences)
        write_csv(work / 'conditional_sensitivity.csv', contrasts)
        table = '\n'.join(f"| {r['size']} | {r['baseline']:.3f} | {r['random']:.3f} | {r['pretrained']:.3f} | {r['pretrained']-r['random']:+.3f} |" for r in learning)
        primary = next(r for r in contrasts if r['size'] == 60 and r['contrast'] == 'pretrained_minus_random')
        n60 = [r['final_training_mae'] for r in diagnostics if r['size'] == 60]
        ratios = [r['sd_ratio'] for r in diagnostics if r['size'] == 60 and r['sd_ratio'] is not None]
        report_text = f'''# Whole-brain transfer-learning follow-up

## Design

Development cross-validation on 90 QC-accepted AOMIC subjects from the original
training allocation. Three folds hold out 30 subjects each, with nested 15/30/60
training subsets. Both CNN arms fine-tune all layers for 50 fixed epochs, with
seeds 42/43/44, the same architecture and paired sample order. Input covers the full
approved volume at 64³ (effective 4 mm); CNN has {config['model_parameters']} parameters.
Pretraining used masked reconstruction for 20 epochs on {config['pretraining']['subjects']}
approved IXI/NKI training subjects. Target scaling and baseline use training labels only.
The original validation/test allocations are excluded. These are development results
after viewing the earlier pilot test, not independent confirmatory evidence.

## Learning-curve results

Values average MAE across seeds, not predictions before computing MAE.

| Labels/fold | Mean baseline | Random CNN | Pretrained CNN | Pretrained − random |
|---|---:|---:|---:|---:|
{table}

The prespecified primary contrast at 60 labels is {primary['mean_difference']:+.3f}
points (positive favors random initialization). Under these tested settings,
pretraining did not demonstrate improved label efficiency. This does not establish
that transfer learning generally fails or that methods are equivalent.

## Fitting and prediction diagnostics

At 60 labels, final training MAE across 18 runs ranges from {min(n60):.3f} to
{max(n60):.3f}. Compare these with the out-of-fold MAEs above: the large gap indicates
training-target fitting with limited held-out generalization, not an inability to
fit the training targets. Within-fold prediction SD / observed SD ranges from
{min(ratios):.3f} to {max(ratios):.3f} at 60 labels. Inspect individual values in
`run_diagnostics.csv` and scatterplots before characterizing predictions as nearly
constant. SD ratios and correlations are descriptive; neither establishes utility.

## Paired uncertainty and sensitivity

{CAVEAT}

The primary contrast's conditional 95% percentile sensitivity range is
[{primary['sensitivity_low']:+.3f}, {primary['sensitivity_high']:+.3f}] points.
The analysis uses {repetitions} resamples, RNG seed {seed}, preserving 30 subjects
per fold and all seed/method/size predictions for each sampled subject. Differences
average absolute errors across seeds; they do not treat 270 predictions as 270
independent subjects. All nine ranges are exploratory and unadjusted. Three folds
are insufficient for a reliable fold-cluster bootstrap. We report fold/seed
contrasts transparently and do not compute a naive t-test or significance claim.
A stronger uncertainty design would require repeated refitting with a prespecified
resampling design or genuinely independent evaluation data.

## Relationship to the completed pilot and next decisions

The earlier frozen central-patch pilot had test MAEs of 20.714 (mean baseline),
20.692 (random), and 20.819 (pretrained), on 123 accepted test subjects. Those subjects
and evaluation design differ: comparing these MAEs directly cannot establish an
improvement from whole-brain coverage. Both studies show no observed pretraining
advantage under their respective settings.

Limitations include 22 pretraining subjects, one pretraining seed/checkpoint,
90 development subjects, a single fold allocation, coarse spatial resolution,
and a fixed architecture/objective/schedule. The learning curves are not consistently
improving with more labels. Do not pick the best seed or sample size using these
results and describe it as a prespecified winner. Finalize these findings before
introducing a new architecture; any new experiment needs an explicit evaluation plan.

## Provenance

`analysis.json` records input hashes, settings, and diagnostics. PNG/PDF figures
and CSV tables accompany this report. No models were trained or scored anew.
'''
        (work / 'research_summary.md').write_text(report_text)
        result = dict(units=summary['units'], learning_curve=learning, paired_contrasts=contrasts,
                      uncertainty_caveat=CAVEAT, bootstrap_repetitions=repetitions, bootstrap_seed=seed,
                      primary_contrast=primary, input_sha256=hashes, script_sha256=digest(Path(__file__)))
        (work / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        if any(digest(Path(p)) != value for p, value in hashes.items()):
            raise ValueError('Inputs changed during analysis')
        work.rename(output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print('Saved analysis:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/aomic_followup_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/aomic_followup_analysis_v1')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    analyze(args.source.resolve(), args.output.resolve(), args.bootstrap, args.seed)


if __name__ == '__main__':
    main()
