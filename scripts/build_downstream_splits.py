#!/usr/bin/env python3
"""Freeze AOMIC cognitive splits before image QC; reserve AOMIC from pretraining."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 'aomic-fluid-v1-42'


def rank_subjects(rows):
    seen = set()
    eligible = []
    for row in rows:
        sid = row['source_participant_id']
        if sid in seen:
            raise ValueError('Duplicate source subject')
        seen.add(sid)
        if row['label_status'] != 'matched':
            continue
        if not math.isfinite(float(row['IST_fluid'])):
            raise ValueError('Nonfinite target')
        eligible.append(row)
    return sorted(eligible, key=lambda r: (hashlib.sha256(
        f"{SEED}|{r['source_participant_id']}".encode()).hexdigest(), r['source_participant_id']))


def save_new(path, rows, fields):
    import io
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Frozen output differs: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(data)


def main():
    manifests = ROOT / 'data/manifests'
    source = manifests / 'aomic_label_candidates.csv'
    with source.open() as stream:
        rows = rank_subjects(list(csv.DictReader(stream)))
    n_test = n_val = round(len(rows) * .15)
    if min(n_test, n_val, len(rows)-n_test-n_val) < 1:
        raise ValueError('Too few subjects')
    output = []
    for index, row in enumerate(rows):
        split = 'test' if index < n_test else 'val' if index < n_test+n_val else 'train'
        output.append({k: row[k] for k in ('dataset', 'participant_id', 'source_participant_id', 'hf_path', 'IST_fluid')} | {
            'split': split, 'split_seed': SEED,
            'eligibility': 'requires_image_qc', 'pretraining_allowed': 'False'})
    save_new(manifests / 'aomic_downstream_splits_v1.csv', output, list(output[0]))
    with (manifests / 'pretraining_cohort.csv').open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        pretraining = [r for r in reader if r['dataset'] in ('PT005_IXI', 'PT015_NKI')]
    save_new(manifests / 'pretraining_without_aomic_v1.csv', pretraining, fields)
    from collections import Counter
    print('Downstream split counts:', dict(Counter(r['split'] for r in output)))
    print('Pretraining metadata candidates:', len(pretraining), '(IXI/NKI only; still require QC)')
    print('Label source SHA256:', hashlib.sha256(source.read_bytes()).hexdigest())
    assert len({r['source_participant_id'] for r in output}) == len(output)
    assert not ({r['hf_path'] for r in output} & {r['hf_path'] for r in pretraining})
    assert rank_subjects(list(reversed(rows))) == rows
    print('Verified unique subjects, stable ordering, and no scan-path overlap.')


if __name__ == '__main__':
    main()
