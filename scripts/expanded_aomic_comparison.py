"""Prepare additional training scans, then compare paired frozen CNN encoders."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from tempfile import mkdtemp

from aomic_validation_batch import (CHECKPOINT, FROZEN, KEYS, OUT, ROOT, csv_bytes,
                                    digest, folder_for, prepare, publish, read)
from evaluate_aomic_validation import accepted_rows, evaluate
from smoke_aomic_transfer import fit_head

BATCH = ROOT / 'results/aomic_train_expansion_v1'
SEEDS = (42, 43, 44)


def cohorts():
    original = read(CHECKPOINT.parent / 'approved_aomic.csv')
    frozen = [r for r in read(FROZEN) if r['split'] == 'train']
    for key in ('participant_id', 'source_participant_id', 'hf_path'):
        if len({r[key] for r in frozen}) != len(frozen):
            raise ValueError(f'Duplicate training identity: {key}')
    labels = {r['hf_path']: r for r in frozen}
    for row in original:
        label = labels[row['hf_path']]
        if any(row[k] != label[k] for k in (*KEYS, 'IST_fluid')) or row['decision'] != 'accept':
            raise ValueError('Original training manifest disagrees with frozen allocation or QC')
    ids = {r['source_participant_id'] for r in original}
    if len(ids) != 15 or len(original) != 15:
        raise ValueError('Expected 15 original training subjects')
    pool = [{k: r[k] for k in KEYS} for r in frozen if r['source_participant_id'] not in ids]
    pool.sort(key=lambda r: hashlib.sha256(('train-expand-v1|' + r['source_participant_id']).encode()).digest())
    selected = pool[:15]
    if len(selected) != 15 or any(r['dataset'] != 'PT013_AIOMIC1000' for r in selected):
        raise ValueError('Expected 15 additional AOMIC training subjects')
    BATCH.mkdir(parents=True, exist_ok=True)
    selection = BATCH / 'selection.csv'
    data = csv_bytes(selected)
    if not selection.exists():
        publish(selection, lambda tmp: tmp.write_bytes(data))
    if selection.read_bytes() != data:
        raise ValueError('Frozen training selection differs')
    return original, selected, labels


def approved_training_rows(original, selected, labels):
    review = BATCH / 'review.csv'
    if not review.exists():
        raise SystemExit('Run prepare and complete visual review first.')
    rows = list(original)
    for row, qc in accepted_rows(selected, read(review)):
        folder = folder_for(row)
        meta = folder / 'metadata.json'
        if digest(meta) != qc['metadata_sha256']:
            raise ValueError('Training approval refers to an older output')
        saved = json.loads(meta.read_text())
        if saved['provenance']['hf_path'] != row['hf_path']:
            raise ValueError('Training tensor subject mismatch')
        label = labels[row['hf_path']]
        rows.append(dict(row, tensor_path=str((folder/'tensor.pt').relative_to(ROOT)),
                         tensor_sha256=saved['sha256']['tensor.pt'], metadata_sha256=qc['metadata_sha256'],
                         decision='accept', qc_status=qc['status'], IST_fluid=label['IST_fluid'],
                         split_seed=label['split_seed']))
    return rows


def compare(original, selected, labels):
    rows = approved_training_rows(original, selected, labels)
    reference = json.loads((OUT/'eval-1g7xjyxg/report.json').read_text())
    hashes = {name: digest(OUT / (name + '.csv')) for name in ('selection','review')}
    if any(value != reference[name + '_sha256'] for name,value in hashes.items()):
        raise ValueError('Validation selection or approvals changed since the pilot')
    source_report = json.loads((CHECKPOINT.parent/'report.json').read_text())
    source = Path(source_report['encoder_checkpoint'])
    if digest(source) != source_report['encoder_checkpoint_sha256']:
        raise ValueError('Pretrained encoder differs from the pilot checkpoint')
    run = Path(mkdtemp(prefix='comparison-', dir=BATCH))
    manifest = csv_bytes(rows)
    results = []
    for seed in SEEDS:
        metrics = {'seed': seed}
        for mode, encoder in [('pretrained', source), ('random_encoder', None)]:
            branch = run / f'{seed}-{mode}'
            branch.mkdir()
            path = branch / 'approved_aomic.csv'
            path.write_bytes(manifest)
            model = fit_head(path, branch, encoder, seed=seed)
            score, _ = evaluate(model, branch)
            if score['visually_accepted'] != 12 or any(score[k+'_sha256'] != v for k,v in hashes.items()):
                raise ValueError('Validation batch changed during comparison')
            if 'training_mean' in metrics and metrics['training_mean'] != score['baseline_mae']:
                raise ValueError('Baseline differs within a paired comparison')
            metrics[mode] = score['model_mae']
            metrics['training_mean'] = score['baseline_mae']
        metrics['random_minus_pretrained'] = metrics['random_encoder'] - metrics['pretrained']
        metrics['baseline_minus_pretrained'] = metrics['training_mean'] - metrics['pretrained']
        results.append(metrics)
    summary = dict(training_subjects=len(rows), added_subjects=len(rows)-len(original),
                   validation_subjects=12, units='IST_fluid points', per_seed=results,
                   mean_over_seeds={k: mean(r[k] for r in results) for k in results[0] if k != 'seed'},
                   sd_over_seeds={k: stdev(r[k] for r in results) for k in ('pretrained','random_encoder')},
                   pretrained_beats_baseline_seeds=sum(r['baseline_minus_pretrained'] > 0 for r in results))
    report = run / 'comparison.json'
    report.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), '\nSaved:', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'compare'))
    stage = parser.parse_args().stage
    original, selected, labels = cohorts()
    if stage == 'prepare':
        raise SystemExit(prepare(selected, BATCH))
    compare(original, selected, labels)
