#!/usr/bin/env python3
"""Audit source-ID joins and cognitive targets; does not approve scans or assign splits."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
SCORES = ('IST_fluid', 'IST_memory', 'IST_crystallised', 'IST_intelligence_total')
FIELDS = ['dataset', 'participant_id', 'session_id', 'hf_path', 'original_path',
          'source_participant_id', 'in_pilot', 'label_status', *SCORES, 'age', 'sex']


def numeric(value):
    if value.strip().lower() in ('', 'n/a', 'na', 'nan'):
        return ''
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite score')
    return number


def source_id(path):
    parts = PurePosixPath(path).parts
    if len(parts) != 3 or parts[1] != 'anat' or not re.fullmatch(r'sub-\d{4}', parts[0]):
        raise ValueError(f'Unexpected AOMIC path: {path}')
    if not re.fullmatch(re.escape(parts[0]) + r'_run-\d+_T1w\.nii\.gz', parts[-1]):
        raise ValueError(f'Subject/filename mismatch: {path}')
    return parts[0]


def join_labels(cohort, participants, pilot):
    lookup = {}
    for row in participants:
        sid = row['participant_id']
        if sid in lookup:
            raise ValueError(f'Duplicate source participant: {sid}')
        lookup[sid] = row
    output, seen_source, seen_fomo = [], set(), set()
    for row in cohort:
        if row['dataset'] != 'PT013_AIOMIC1000':
            continue
        sid = source_id(row['original_path'])
        if sid in seen_source or row['participant_id'] in seen_fomo:
            raise ValueError('Non-unique source/FOMO mapping; resolve before using labels')
        seen_source.add(sid)
        seen_fomo.add(row['participant_id'])
        record = {k: row[k] for k in ('dataset', 'participant_id', 'session_id', 'hf_path', 'original_path')}
        record.update(source_participant_id=sid, in_pilot=row['hf_path'] in pilot)
        source = lookup.get(sid)
        if source is None:
            record.update(label_status='unmatched', **{k: '' for k in (*SCORES, 'age', 'sex')})
        else:
            record.update({k: numeric(source[k]) for k in SCORES})
            record.update(age=numeric(source['age']), sex=source['sex'])
            record['label_status'] = 'matched' if record['IST_fluid'] != '' else 'missing_target'
        output.append(record)
    return sorted(output, key=lambda r: r['source_participant_id'])


def read_csv(path, delimiter=','):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def save_new(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing output differs: {path}; version it explicitly')
        return
    with path.open('xb') as stream:
        stream.write(data)


def main():
    metadata = ROOT / 'data/metadata/aomic_id1000'
    provenance = json.loads((metadata / 'provenance.json').read_text())
    for filename, info in provenance['files'].items():
        if hashlib.sha256((metadata / filename).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Source metadata checksum mismatch: {filename}')
    definitions = json.loads((metadata / 'participants.json').read_text())
    for score in SCORES:
        if score not in definitions:
            raise ValueError(f'Missing target definition: {score}')
    cohort_path = ROOT / 'data/manifests/pretraining_cohort.csv'
    pilot_path = ROOT / 'data/manifests/pilot_cohort.csv'
    pilot = {r['hf_path'] for r in read_csv(pilot_path)}
    participants = read_csv(metadata / 'participants.tsv', '\t')
    rows = join_labels(read_csv(cohort_path), participants, pilot)
    if not rows:
        raise ValueError('No AOMIC subjects in cohort')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    save_new(ROOT / 'data/manifests/aomic_label_candidates.csv', buffer.getvalue().encode())
    summary = {
        'source_dataset': 'ds003097', 'source_commit': provenance['commit'],
        'cohort_sha256': hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        'pilot_sha256': hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        'candidate_target': 'IST_fluid', 'target_definition': definitions['IST_fluid'],
        'source_participants': len(participants), 'aomic_cohort_subjects': len(rows),
        'matched_ids': sum(r['label_status'] != 'unmatched' for r in rows),
        'unmatched_ids': sum(r['label_status'] == 'unmatched' for r in rows),
        'available_scores': {score: sum(r[score] != '' for r in rows) for score in SCORES},
        'pilot_aomic_subjects': sum(r['in_pilot'] for r in rows),
        'pilot_available_target': sum(r['in_pilot'] and r['label_status'] == 'matched' for r in rows),
        'missing_target_subjects': [r['source_participant_id'] for r in rows if r['label_status'] == 'missing_target'],
        'status': 'metadata candidates only; image QC and downstream splits still required',
    }
    result = ROOT / 'results/assignment03/aomic_label_audit.json'
    save_new(result, (json.dumps(summary, indent=2) + '\n').encode())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
