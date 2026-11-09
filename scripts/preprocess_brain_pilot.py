#!/usr/bin/env python3
"""SynthStrip + brain-centered pilot. Defaults to nine representative scans."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nibabel as nib
import torch
from build_pilot_manifest import safe_path
from preprocess_pilot import digest, validate_raw, validate_tensor
from src.data.brain_preprocessing import preprocess_brain, review_reasons

IMAGE = 'freesurfer/synthstrip:1.8'
ARTIFACTS = ('tensor.pt', 'native_mask.nii.gz', 'brain_mask.nii.gz', 'centered_input.nii.gz', 'synthstrip.log')
FIELDS = ('run_id', 'dataset', 'participant_id', 'hf_path', 'status', 'reasons',
          'output_dir', 'brain_retention', 'brain_volume_ml', 'brain_mean', 'brain_std',
          'native_foreground_retention', 'error')


def checked_path(base, relative):
    path = base / str(safe_path(relative))
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError('Path escapes data directory')
    return path


def load_rows(all_scans=False):
    with (ROOT / 'data/manifests/pilot_cohort.csv').open() as stream:
        pilot = list(csv.DictReader(stream))
    if len({r['hf_path'] for r in pilot}) != len(pilot):
        raise ValueError('Duplicate pilot paths')
    if all_scans:
        return pilot
    with (ROOT / 'results/assignment03/visual_qc_selection.csv').open() as stream:
        selection = list(csv.DictReader(stream))
    selected = {r['hf_path'] for r in selection}
    rows = [r for r in pilot if r['hf_path'] in selected]
    if len(selected) != 9 or len(rows) != 9 or sorted(Counter(r['dataset'] for r in rows).values()) != [3, 3, 3]:
        raise ValueError('Expected nine pilot scans, three per dataset, in visual_qc_selection.csv')
    return rows


def image_id(image):
    if not shutil.which('docker'):
        raise RuntimeError('Install/start Docker Desktop; see BRAIN_PREPROCESSING.md')
    result = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'Start Docker Desktop, then run: docker pull --platform linux/amd64 {image}\n{result.stderr.strip()}')
    identity = result.stdout.strip()
    if not identity.startswith('sha256:'):
        raise RuntimeError('Could not identify the installed SynthStrip image')
    return identity


def extract(raw, work, identity, timeout):
    name = 'brain-pilot-' + uuid.uuid4().hex
    command = ['docker', 'run', '--rm', '--pull=never', '--platform', 'linux/amd64',
               '--name', name, '--network', 'none', '--cpus', '2',
               '--user', f'{os.getuid()}:{os.getgid()}',
               '-v', f'{raw.parent}:/input:ro', '-v', f'{work}:/output', identity,
               '-i', '/input/' + raw.name, '-m', '/output/native_mask.nii.gz', '-t', '2']
    try:
        with (work / 'synthstrip.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'SynthStrip exited {result.returncode}; see per-scan log')
    finally:
        # Also stop the container if the CLI times out or receives Ctrl-C.
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)


def process_scan(row, identity, code_hash, timeout, logs, run_id):
    relative = str(safe_path(row['hf_path']))
    raw = checked_path(ROOT / 'data/raw', relative)
    destination = checked_path(ROOT / 'data/processed/brain_v1', relative + '.brain')
    validate_raw(raw)
    provenance = {'raw_sha256': digest(raw), 'pipeline_sha256': code_hash,
                  'synthstrip_image_id': identity, 'hf_path': relative,
                  'extraction_args': ['-t', '2'], 'spacing_mm': [2, 2, 2], 'shape': [128, 128, 128]}
    if destination.exists():
        saved = json.loads((destination / 'metadata.json').read_text())
        if saved['provenance'] != provenance:
            raise ValueError('Existing output provenance mismatch; preserved for inspection')
        for filename in ARTIFACTS:
            if digest(destination / filename) != saved['sha256'][filename]:
                raise ValueError(f'Existing output checksum mismatch: {filename}')
        validate_tensor(torch.load(destination / 'tensor.pt', map_location='cpu', weights_only=True))
        return destination, saved['qc']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.brain-work-', dir=destination.parent) as name:
        work = Path(name)
        try:
            extract(raw.resolve(), work.resolve(), identity, timeout)
            tensor, mask, centered, qc = preprocess_brain(raw, work / 'native_mask.nii.gz')
            validate_tensor(tensor)
            torch.save(tensor, work / 'tensor.pt')
            nib.save(mask, work / 'brain_mask.nii.gz')
            nib.save(centered, work / 'centered_input.nii.gz')
            saved = {'provenance': provenance, 'qc': qc,
                     'sha256': {filename: digest(work / filename) for filename in ARTIFACTS}}
            (work / 'metadata.json').write_text(json.dumps(saved, indent=2))
            # Entire scan is published together only after successful validation.
            work.rename(destination)
        finally:
            log = work / 'synthstrip.log'
            if log.exists():
                key = hashlib.sha256(relative.encode()).hexdigest()[:16]
                shutil.copyfile(log, logs / f'{run_id}-{key}.log')
    return destination, qc


def run(rows, identity, timeout):
    torch.set_num_threads(1)
    code_hash = hashlib.sha256(b''.join(path.read_bytes() for path in (
        Path(__file__), ROOT / 'src/data/brain_preprocessing.py',
        ROOT / 'src/data/preprocessing.py', ROOT / 'scripts/preprocess_pilot.py',
        ROOT / 'scripts/build_pilot_manifest.py'))).hexdigest()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    manifests = ROOT / 'data/manifests'
    logs = ROOT / 'results/assignment03/brain_v1/logs'
    logs.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (manifests / 'brain_preprocessing_qc.csv').open('a', newline='') as stream, (manifests / 'brain_preprocessing_failures.jsonl').open('a') as failures:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        for row in rows:
            record = {k: row[k] for k in ('dataset', 'participant_id', 'hf_path')}
            record['run_id'] = run_id
            try:
                output, qc = process_scan(row, identity, code_hash, timeout, logs, run_id)
                reasons = review_reasons(qc, row['dataset'])
                record.update({k: qc[k] for k in FIELDS if k in qc})
                record.update(output_dir=str(output.relative_to(ROOT)),
                              status='review' if reasons else 'pass', reasons=';'.join(reasons))
            except Exception as error:
                record.update(status='failed', error=f'{type(error).__name__}: {error}')
                failures.write(json.dumps({**record, 'traceback': traceback.format_exc()}) + '\n')
                failures.flush()
                os.fsync(failures.fileno())
            writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
            counts[record['status']] += 1
            print(f"{row['hf_path']}: {record['status']} {record.get('reasons', '')} {record.get('error', '')}", flush=True)
    print(f'Run {run_id}: {dict(counts)}')
    return 1 if counts['failed'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Process all 45 after reviewing the nine-scan outputs')
    parser.add_argument('--image', default=IMAGE, help='Installed SynthStrip Docker image; resolved to immutable ID')
    parser.add_argument('--timeout', type=int, default=1800, help='Per-scan extraction timeout in seconds')
    parser.add_argument('--check', action='store_true', help='Check inputs and Docker image without running extraction')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    try:
        rows = load_rows(args.all)
        for row in rows:
            if not checked_path(ROOT / 'data/raw', row['hf_path']).is_file():
                raise ValueError(f"Missing raw scan: {row['hf_path']}; run baseline downloads first")
        identity = image_id(args.image)
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(2, str(error) + '\n')
    print(f'{len(rows)} scans; SynthStrip {identity}')
    if args.check:
        return
    lock = ROOT / 'data/manifests/.brain_preprocessing.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        parser.exit(2, f'Run lock exists: {lock}. Remove only if no brain-processing run is active.\n')
    os.close(fd)
    try:
        status = run(rows, identity, args.timeout)
    finally:
        lock.unlink()
    raise SystemExit(status)


if __name__ == '__main__':
    main()
