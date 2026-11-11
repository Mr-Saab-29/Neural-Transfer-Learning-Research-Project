#!/usr/bin/env python3
"""Freeze score-independent follow-up development folds; never run training or QC."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SELECTION_SEED = 'aomic-followup-v1-42'
SIZES = (15, 30, 60)
FIELDS = ('dataset', 'participant_id', 'source_participant_id', 'hf_path',
          'IST_fluid', 'original_split', 'fold', 'role', 'qc_status')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rank(rows, purpose):
    return sorted(rows, key=lambda r: (sha(
        f"{SELECTION_SEED}|{purpose}|{r['source_participant_id']}".encode()),
        r['source_participant_id']))


def select_cohort(rows):
    if not rows:
        raise ValueError('Empty source manifest')
    for key in ('participant_id', 'source_participant_id', 'hf_path'):
        values = [r[key] for r in rows]
        if any(not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f'Missing or duplicate {key}')
    for r in rows:
        if r['dataset'] != 'PT013_AIOMIC1000' or r['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unexpected dataset or split')
        if r['pretraining_allowed'].lower() != 'false':
            raise ValueError('AOMIC must remain excluded from pretraining')
        if not math.isfinite(float(r['IST_fluid'])):
            raise ValueError('Nonfinite target')
    training = [r for r in rows if r['split'] == 'train']
    if len(training) < 90:
        raise ValueError('Need at least 90 subjects in the original training allocation')
    return rank(training, 'cohort')[:90]


def csv_bytes(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def record(row, fold, role):
    return {key: row[key] for key in FIELDS[:5]} | dict(
        original_split='train', fold=fold, role=role, qc_status='requires_image_qc')


def build_files(rows, source_hash):
    cohort = select_cohort(rows)
    files = {'cohort.csv': csv_bytes([record(r, i % 3, 'development') for i, r in enumerate(cohort)])}
    for fold in range(3):
        evaluation = [r for i, r in enumerate(cohort) if i % 3 == fold]
        training = rank([r for i, r in enumerate(cohort) if i % 3 != fold], f'train-fold-{fold}')
        files[f'fold_{fold}_evaluation.csv'] = csv_bytes([record(r, fold, 'evaluation') for r in evaluation])
        for size in SIZES:
            files[f'fold_{fold}_train_{size}.csv'] = csv_bytes([record(r, fold, 'train') for r in training[:size]])
    protocol = dict(
        version='aomic-followup-v1', status='selection_and_evaluation_design_frozen',
        source_manifest_sha256=source_hash, selection_seed=SELECTION_SEED,
        source_split_counts=dict(sorted(Counter(r['split'] for r in rows).items())),
        cohort_subjects=90, folds=3, evaluation_subjects_per_fold=30,
        training_sizes=list(SIZES), training_seeds=[42, 43, 44],
        methods=['training_mean', 'random_cnn_finetuned', 'pretrained_cnn_finetuned'],
        primary_metric='MAE in IST_fluid points',
        primary_contrast='pretrained_cnn_finetuned minus random_cnn_finetuned at 60 labels',
        secondary_contrasts='15/30-label comparisons and training-mean comparisons',
        analysis='For each size and seed, pool the 90 out-of-fold predictions; then average MAE across fixed seeds. Keep subject predictions paired for contrasts. Do not treat folds or seeds as independent subjects.',
        interpretation='Follow-up development cross-validation after pilot test exposure; not untouched confirmation. Some original training subjects may have appeared in the pilot.',
        selection='SHA-256 ranking of subject IDs only; targets and QC outcomes do not affect selection. Round-robin allocation across three folds; independently ranked nested training subsets within each fold.',
        qc_policy='All 90 subjects require image-based QC approval before training. Reuse existing artifacts only after verifying preprocessing compatibility and approval provenance. No silent exclusion or replacement. If a subject cannot be approved, stop and document a revised version before fitting any model.',
        leakage_controls=[
            'Original validation and test subjects are excluded from all follow-up folds.',
            'Exclude all AOMIC subjects from self-supervised pretraining; use approved IXI/NKI only.',
            'Fit target normalization and training-mean baseline on each fold/size training subset only.',
            'Use identical subjects, preprocessing, architecture, and supervised training budget across CNN methods.',
            'Never select epochs, checkpoints, or hyperparameters on outer evaluation folds. Use a fixed schedule frozen before fitting, or a separately specified inner split drawn only from training subjects.',
            'Each subject has one out-of-fold prediction per method/size/seed. Evaluation labels are for scoring only.',
        ],
        training_configuration_status='Not yet frozen: these selection manifests are not training-ready.',
        training_configuration_required=[
            'Whole-brain input geometry, preprocessing version and approved artifact hashes',
            'CNN architecture and parameter count, optimizer, learning rate, batch size, epochs and checkpoint rule',
            'Augmentations and random seeds; equal supervised budget across initialization methods',
            'IXI/NKI pretraining subjects, objective, schedule, and checkpoint hash',
            'Successful loader and training-only overfit checks',
        ],
        files_sha256={name: sha(data) for name, data in sorted(files.items())},
    )
    files['protocol.json'] = (json.dumps(protocol, indent=2) + '\n').encode()
    return files


def publish(files, output):
    if output.exists():
        if {p.name for p in output.iterdir()} != set(files) or any(
                not (output / name).is_file() or (output / name).read_bytes() != data
                for name, data in files.items()):
            raise ValueError(f'Frozen output differs; use a new version directory: {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.followup-', dir=output.parent))
    try:
        for name, data in files.items():
            (temporary / name).write_bytes(data)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/manifests/aomic_downstream_splits_v1.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/manifests/aomic_followup_v1')
    args = parser.parse_args()
    data = args.source.read_bytes()
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    files = build_files(rows, sha(data))
    publish(files, args.output)
    print(f'Frozen development protocol: {args.output.resolve()}')
    print('90 subjects; 3 folds; 30 evaluation subjects/fold; nested 15/30/60 training subjects.')
    print('QC and a separate frozen training configuration are required before fitting.')


if __name__ == '__main__':
    main()
