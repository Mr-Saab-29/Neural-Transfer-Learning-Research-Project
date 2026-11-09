#!/usr/bin/env python3
"""Select reproducible subjects, independent of input CSV ordering."""
import argparse
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('PT005_IXI', 'PT013_AIOMIC1000', 'PT015_NKI')


def safe_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe relative path: {value!r}')
    return path


def select(rows, seed='assignment03'):
    result = []
    for dataset in DATASETS:
        subjects = {}
        for row in rows:
            if row['dataset'] != dataset or row['file_exists'].lower() != 'true':
                continue
            if row['filename'] not in ('t1.nii', 't1.nii.gz'):
                continue
            if not row['participant_id']:
                raise ValueError('Empty participant ID')
            path = safe_path(row['hf_path'])
            if path.parts[0] != dataset:
                raise ValueError('Dataset/path mismatch')
            subjects.setdefault(row['participant_id'], []).append(row)
        if len(subjects) < 15:
            raise ValueError(f'{dataset}: only {len(subjects)} eligible subjects')
        ranked = sorted(subjects, key=lambda sid: (
            hashlib.sha256(f'{seed}|{dataset}|{sid}'.encode()).hexdigest(), sid))
        for sid in ranked[:15]:
            result.append(min(subjects[sid], key=lambda r: (
                r['baseline_priority'].lower() != 'true', r['session_id'], r['hf_path'])))
    if len({r['hf_path'] for r in result}) != 45:
        raise ValueError('Duplicate scan paths in pilot')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/manifests/pretraining_cohort.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/manifests/pilot_cohort.csv')
    parser.add_argument('--seed', default='assignment03')
    args = parser.parse_args()
    with args.input.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = select(list(reader), args.seed)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_bytes() != content:
            raise SystemExit('Output already exists with different contents; choose another --output.')
    else:
        with args.output.open('xb') as stream:
            stream.write(content)
    print(f'{args.output}: 45 subjects, 15 per dataset; seed={args.seed}')


if __name__ == '__main__':
    main()
