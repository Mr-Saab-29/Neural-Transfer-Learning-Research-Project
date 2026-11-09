#!/usr/bin/env python3
"""Select 12 frozen validation subjects, or prepare them for score-blind review."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess_pilot import REPO, hf_hub_download, publish, validate_raw, digest
import preprocess_brain_pilot as brain
from visualize_brain_qc import render_scan
from src.data.dataset import project_path

OUT = ROOT / 'results/aomic_validation_v1'
FROZEN = ROOT / 'data/manifests/aomic_downstream_splits_v1.csv'
REVISION = 'bf2bb12bd5cfeaed65e4a003e72d55fb8322c28a'
CHECKPOINT = ROOT / 'results/smoke_aomic/run-iehu9t3e/regressor.pt'
KEYS = ('hf_path', 'dataset', 'source_participant_id', 'participant_id', 'split')


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def select():
    # Discard scores before ordering. No replacements after QC.
    rows = [{k: r[k] for k in KEYS} for r in read(FROZEN) if r['split'] == 'val']
    if len(rows) != 124 or any(r['dataset'] != 'PT013_AIOMIC1000' for r in rows):
        raise ValueError('Unexpected frozen validation cohort')
    if any(len({r[k] for r in rows}) != len(rows) for k in ('hf_path','source_participant_id','participant_id')):
        raise ValueError('Duplicate validation identity')
    rows.sort(key=lambda r: hashlib.sha256(('validation-v1|' + r['source_participant_id']).encode()).digest())
    rows = rows[:12]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'selection.csv'
    data = csv_bytes(rows)
    if not path.exists():
        publish(path, lambda tmp: tmp.write_bytes(data))
    if path.read_bytes() != data:
        raise ValueError('Existing selection conflicts with frozen split')
    return rows


def folder_for(row):
    return project_path(ROOT / 'data/processed/brain_v1', row['hf_path'] + '.brain')


def prepare(rows, output=OUT):
    identity = brain.image_id(brain.IMAGE)
    code_hash = hashlib.sha256(b''.join(p.read_bytes() for p in (
        Path(brain.__file__), ROOT/'src/data/brain_preprocessing.py', ROOT/'src/data/preprocessing.py',
        ROOT/'scripts/preprocess_pilot.py', ROOT/'scripts/build_pilot_manifest.py'))).hexdigest()
    training = next(r for r in read(CHECKPOINT.parent/'approved_aomic.csv') if r['split']=='train')
    provenance = json.loads((project_path(ROOT, training['tensor_path']).parent/'metadata.json').read_text())['provenance']
    if (identity, code_hash) != (provenance['synthstrip_image_id'], provenance['pipeline_sha256']):
        raise ValueError('Validation preprocessing must match the training pipeline and image')
    qc_file = output/'review.csv'
    previous = {r['hf_path']:r for r in read(qc_file)} if qc_file.exists() else {}
    logs = output/'logs'; logs.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    results = []
    for row in rows:
        record = dict(hf_path=row['hf_path'], status='failed', reasons='', metadata_sha256='', decision='pending', notes='', error='')
        try:
            raw = project_path(ROOT/'data/raw', row['hf_path'])
            if not raw.exists():
                cached = Path(hf_hub_download(repo_id=REPO, repo_type='dataset', revision=REVISION, filename=row['hf_path']))
                validate_raw(cached)
                publish(raw, lambda tmp: shutil.copyfile(cached,tmp))
            folder,qc = brain.process_scan(row,identity,code_hash,1800,logs,run_id)
            reasons = brain.review_reasons(qc,row['dataset'])
            render_scan(folder, output/(row['participant_id']+'.png'), row['participant_id']+' | '+(', '.join(reasons) or 'automated pass'))
            record.update(status='review' if reasons else 'pass',reasons=';'.join(reasons),metadata_sha256=digest(folder/'metadata.json'))
            prior = previous.get(row['hf_path'], {})
            if prior.get('metadata_sha256') == record['metadata_sha256']:
                record.update(decision=prior['decision'],notes=prior['notes'])
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        results.append(record)
        print(row['participant_id'],record['status'], record['error'],flush=True)
    with tempfile.NamedTemporaryFile(dir=output,delete=False) as stream:
        stream.write(csv_bytes(results))
    Path(stream.name).replace(qc_file)
    print('Review each successful PNG, then edit decision and notes in',qc_file)
    return int(any(r['status']=='failed' for r in results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Download, extract, and render 12 scans sequentially')
    args = parser.parse_args()
    rows = select()
    print('Selected',len(rows),'frozen validation subjects:', OUT/'selection.csv')
    if args.prepare:
        raise SystemExit(prepare(rows))
