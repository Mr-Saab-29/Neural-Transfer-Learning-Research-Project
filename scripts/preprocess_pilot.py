#!/usr/bin/env python3
"""Sequential pilot processing; failures are logged and do not stop other scans."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_pilot_manifest import safe_path
import nibabel as nib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from src.data.preprocessing import preprocess_mri

REPO = 'FOMO-MRI/FOMO45K'
FIELDS = ['run_id', 'dataset', 'participant_id', 'hf_path', 'status', 'output',
          'foreground_retention', 'foreground_fraction', 'foreground_mean',
          'foreground_std', 'foreground_before_crop', 'foreground_after_crop',
          'orientation', 'spacing_mm', 'resampled_shape', 'tensor_shape',
          'error_type', 'error']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, write):
    """Atomic, no-overwrite publication on the destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='partial-', suffix='.tmp')
    os.close(fd)
    temporary = Path(name)
    try:
        write(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_raw(path):
    img = nib.load(str(path))
    if len(img.shape) != 3 or not np.isfinite(img.affine).all():
        raise ValueError('Invalid 3D NIfTI geometry')
    if abs(np.linalg.det(img.affine[:3, :3])) < 1e-12:
        raise ValueError('Singular NIfTI affine')
    if not np.isfinite(img.get_fdata(dtype=np.float32)).all():
        raise ValueError('Raw NIfTI contains nonfinite intensities')


def validate_tensor(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (1, 128, 128, 128)
            or not torch.isfinite(tensor).all()):
        raise ValueError('Invalid output tensor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/pilot_cohort.csv')
    parser.add_argument('--offline', action='store_true', help='Use raw files/cache only')
    parser.add_argument('--revision', default='bf2bb12bd5cfeaed65e4a003e72d55fb8322c28a',
                        help='Pinned FOMO snapshot recorded in the existing notebook')
    parser.add_argument('--min-retention', type=float, default=0.95)
    args = parser.parse_args()
    if not 0 <= args.min_retention <= 1:
        parser.error('--min-retention must lie in [0, 1]')
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        parser.error('--revision must be a full lowercase commit SHA')
    with args.manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    paths = [str(safe_path(row['hf_path'])) for row in rows]
    if len(set(paths)) != len(paths):
        parser.error('Duplicate scan paths')
    manifests = ROOT / 'data/manifests'
    manifests.mkdir(parents=True, exist_ok=True)
    lock = manifests / '.preprocessing.lock'
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        parser.error(f'Another run may be active: {lock}. Remove only after confirming no run is active.')
    os.close(lock_fd)
    try:
        run(rows, args, manifests)
    finally:
        lock.unlink()


def run(rows, args, manifests):
    torch.set_num_threads(1)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    pipeline_hash = hashlib.sha256((ROOT / 'src/data/preprocessing.py').read_bytes()
                                  + Path(__file__).read_bytes()).hexdigest()
    failed = review = 0
    qc_path = manifests / 'preprocessing_qc.csv'
    with qc_path.open('a', newline='') as stream, (manifests / 'preprocessing_failures.jsonl').open('a') as errors:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        for row in rows:
            record = {k: row[k] for k in ('dataset', 'participant_id', 'hf_path')}
            record['run_id'] = run_id
            try:
                relative = safe_path(row['hf_path'])
                raw = ROOT / 'data/raw' / str(relative)
                output = ROOT / 'data/processed' / str(relative.with_suffix('').with_suffix('.pt'))
                for path, base in ((raw, ROOT / 'data/raw'), (output, ROOT / 'data/processed')):
                    if not path.resolve().is_relative_to(base.resolve()):
                        raise ValueError('Path escapes data directory through symlink')
                record['output'] = str(output.relative_to(ROOT))
                if not raw.exists():
                    for attempt in range(3):
                        try:
                            cached = hf_hub_download(repo_id=REPO, repo_type='dataset',
                                filename=str(relative), revision=args.revision,
                                local_files_only=args.offline)
                            validate_raw(Path(cached))
                            publish(raw, lambda tmp: shutil.copyfile(cached, tmp))
                            break
                        except Exception:
                            if args.offline or attempt == 2:
                                raise
                            time.sleep(2 ** attempt)
                validate_raw(raw)
                provenance = {'raw_sha256': digest(raw), 'pipeline_sha256': pipeline_hash,
                              'repo': REPO, 'requested_revision': args.revision,
                              'hf_path': str(relative)}
                sidecar = output.with_suffix('.json')
                if output.exists() or sidecar.exists():
                    saved = json.loads(sidecar.read_text())
                    if saved['provenance'] != provenance or saved['tensor_sha256'] != digest(output):
                        raise ValueError('Existing output provenance/hash mismatch; preserve it and inspect manually')
                    validate_tensor(torch.load(output, map_location='cpu', weights_only=True))
                    qc = saved['qc']
                else:
                    tensor, qc = preprocess_mri(raw, return_qc=True)
                    validate_tensor(tensor)
                    publish(output, lambda tmp: torch.save(tensor, tmp))
                    saved = {'provenance': provenance, 'qc': qc, 'tensor_sha256': digest(output)}
                    publish(sidecar, lambda tmp: tmp.write_text(json.dumps(saved, indent=2)))
                record.update(qc)
                record['status'] = 'review' if qc['foreground_retention'] < args.min_retention else 'pass'
                review += record['status'] == 'review'
            except Exception as exc:
                failed += 1
                record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                errors.write(json.dumps({**record, 'traceback': traceback.format_exc()}) + '\n')
                errors.flush()
                os.fsync(errors.fileno())
            writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{row['hf_path']}: {record['status']}", flush=True)
    print(f'Run {run_id}: {len(rows)-failed-review} pass, {review} review, {failed} failed. QC: {qc_path}')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
