#!/usr/bin/env python3
"""Create a review checklist, then freeze visually accepted subjects and splits."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import project_path, sha256

DECISION_FIELDS = ['hf_path', 'decision', 'metadata_sha256', 'notes']
FIELDS = ['dataset', 'participant_id', 'hf_path', 'tensor_path', 'tensor_sha256',
          'metadata_sha256', 'decision', 'qc_status', 'split', 'split_seed']


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_new(path, rows, fields):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'{path} already exists; choose another output instead of overwriting')
        return
    with path.open('xb') as stream:
        stream.write(content)


def assign_splits(rows, seed):
    result = []
    for dataset in sorted({r['dataset'] for r in rows}):
        group = sorted((r for r in rows if r['dataset'] == dataset), key=lambda r: (
            hashlib.sha256(f"{seed}|{dataset}|{r['participant_id']}".encode()).hexdigest(), r['participant_id']))
        if len(group) < 3:
            raise ValueError(f'{dataset}: need at least 3 accepted subjects for three splits')
        held_out = max(1, round(len(group) * .15))
        for i, row in enumerate(group):
            split = 'test' if i < held_out else 'val' if i < 2 * held_out else 'train'
            result.append({**row, 'split': split, 'split_seed': str(seed)})
    return result


def build(pilot, latest, decisions, root, seed):
    if len({r['hf_path'] for r in decisions}) != len(decisions):
        raise ValueError('Duplicate QC decisions')
    decision_map = {r['hf_path']: r for r in decisions}
    if set(decision_map) != {r['hf_path'] for r in pilot}:
        raise ValueError('QC checklist must contain exactly the pilot scan paths')
    subjects = set()
    accepted = []
    for row in pilot:
        subject = row['dataset'], row['participant_id']
        if subject in subjects:
            raise ValueError('Pilot contains multiple scans for a subject')
        subjects.add(subject)
        decision = decision_map[row['hf_path']]
        if decision['decision'] not in ('pending', 'accept', 'exclude'):
            raise ValueError('Decision must be pending, accept or exclude')
        if decision['decision'] != 'accept':
            continue
        qc = latest.get(row['hf_path'])
        if not qc or qc['status'] not in ('pass', 'review'):
            raise ValueError(f"Accepted scan has no successful latest QC: {row['hf_path']}")
        if qc['status'] == 'review' and not decision['notes'].strip():
            raise ValueError('Accepting a flagged scan requires a written review reason')
        folder = project_path(root, qc['output_dir'])
        metadata = folder / 'metadata.json'
        if sha256(metadata) != decision['metadata_sha256']:
            raise ValueError(f"QC approval is stale or missing: {row['hf_path']}")
        saved = json.loads(metadata.read_text())
        if saved['provenance']['hf_path'] != row['hf_path']:
            raise ValueError('Output metadata belongs to a different scan')
        for name, checksum in saved['sha256'].items():
            if sha256(project_path(folder, name)) != checksum:
                raise ValueError(f'Artifact checksum mismatch: {folder / name}')
        tensor_path = folder / 'tensor.pt'
        accepted.append({k: row[k] for k in ('dataset', 'participant_id', 'hf_path')} | {
            'tensor_path': str(tensor_path.relative_to(root)),
            'tensor_sha256': saved['sha256']['tensor.pt'], 'metadata_sha256': sha256(metadata),
            'decision': 'accept', 'qc_status': qc['status']})
    if not accepted:
        raise ValueError('No visually accepted scans; complete the checklist first')
    return assign_splits(accepted, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-review', action='store_true', help='Create a pending checklist; does not approve scans')
    parser.add_argument('--decisions', type=Path, default=ROOT / 'data/manifests/brain_qc_decisions.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/manifests/pilot_training.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    pilot = read_csv(ROOT / 'data/manifests/pilot_cohort.csv')
    latest = {r['hf_path']: r for r in read_csv(ROOT / 'data/manifests/brain_preprocessing_qc.csv')}
    try:
        if args.init_review:
            rows = []
            for row in pilot:
                qc = latest.get(row['hf_path'], {})
                checksum = ''
                if qc.get('status') in ('pass', 'review'):
                    checksum = sha256(project_path(ROOT, qc['output_dir']) / 'metadata.json')
                rows.append({'hf_path': row['hf_path'], 'decision': 'pending',
                             'metadata_sha256': checksum, 'notes': ''})
            write_new(args.decisions, rows, DECISION_FIELDS)
            print('Pending review checklist:', args.decisions)
            return
        if (ROOT / 'data/manifests/.brain_preprocessing.lock').exists():
            raise ValueError('Wait for active preprocessing to finish before freezing splits')
        rows = build(pilot, latest, read_csv(args.decisions), ROOT, args.seed)
        write_new(args.output, rows, FIELDS)
        print('Frozen pilot manifest:', args.output)
        print('Split counts:', dict(Counter(r['split'] for r in rows)))
        print('No prediction targets included; this is a label-free pilot manifest.')
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
