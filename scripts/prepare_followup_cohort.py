#!/usr/bin/env python3
"""Resumable score-blind preparation, visual-QC gate, and training setup freeze."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import final_aomic_test as preparation
from followup_common import (ROOT, OUT, PROTOCOL, load_protocol, code_hashes,
    approved_artifact, whole_image, WholeBrainCNN, ARCHITECTURE, read, csv_bytes,
    digest, lock, pipeline_hash, json_write)
from preprocess_pilot import publish
from aomic_validation_batch import REVISION


def preparation_plan():
    path = OUT / 'preparation_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['protocol_sha256'] != digest(PROTOCOL / 'protocol.json'):
            raise ValueError('Preparation protocol changed')
        return plan
    # Use the completed pilot's immutable image and pipeline, allowing cache reuse.
    pilot = json.loads((ROOT / 'results/aomic_final_test_v1/final/plan_snapshot.json').read_text())
    if pilot['preprocessing']['pipeline_sha256'] != pipeline_hash():
        raise ValueError('Pilot preprocessing differs from current code')
    plan = dict(preprocessing=pilot['preprocessing'], source_revision=REVISION,
                protocol_sha256=digest(PROTOCOL / 'protocol.json'))
    publish(path, lambda tmp: tmp.write_text(json.dumps(plan, indent=2) + '\n'))
    return plan


def freeze(rows, plan):
    ready = OUT / 'ready'
    if ready.exists():
        raise ValueError('Training setup already frozen; use train_wholebrain.py check')
    if plan['preprocessing']['pipeline_sha256'] != pipeline_hash():
        raise ValueError('Preprocessing changed during preparation')
    review_hash = digest(OUT / 'review.csv')
    reviews = read(OUT / 'review.csv')
    lookup = {r['hf_path']: r for r in reviews}
    if len(lookup) != len(reviews) or set(lookup) != {r['hf_path'] for r in rows}:
        raise ValueError('Review must cover exactly the 90 selected subjects')
    approved = [approved_artifact(r, lookup[r['hf_path']], plan['preprocessing']) for r in rows]
    # Keep the pilot SSL allocation: train subjects only, no pilot val/test or AOMIC.
    pilot = ROOT / 'data/manifests/pilot_training.csv'
    allowed_path = ROOT / 'data/manifests/pretraining_without_aomic_v1.csv'
    allowed = {(r['dataset'], r['participant_id'], r['hf_path']) for r in read(allowed_path)}
    sources = [r for r in read(pilot) if r['dataset'] in ('PT005_IXI', 'PT015_NKI') and r['split'] == 'train']
    if not sources or len({(r['dataset'], r['participant_id']) for r in sources}) != len(sources):
        raise ValueError('Missing or duplicate SSL training subjects')
    ssl = []
    for row in sources:
        if (row['dataset'], row['participant_id'], row['hf_path']) not in allowed:
            raise ValueError('SSL subject not in permitted IXI/NKI pool')
        review = dict(status=row['qc_status'], decision=row['decision'],
                      metadata_sha256=row['metadata_sha256'], notes='Previously accepted in pilot_training.csv')
        ssl.append(approved_artifact(row, review, plan['preprocessing']))
    # Verify tensor validity as well as checksums before publishing the setup.
    for row in approved + ssl:
        whole_image(row)
    config = dict(architecture=ARCHITECTURE, input_shape=[1, 64, 64, 64],
        input_transform='2x2x2 average pooling of full approved 128-cubed tensor; no cropping',
        supervised=dict(epochs=50, batch_size=1, optimizer='Adam', learning_rate=.001,
                        weight_decay=0., augmentation='none', encoder_trainable=True,
                        checkpoint_rule='final epoch only; no outer-fold selection'),
        pretraining=dict(epochs=20, seed=42, learning_rate=.001, batch_size=1,
                         objective='20% random 8-cubed block masking; MSE on masked nonzero voxels',
                         subjects=len(ssl), sources=dict(Counter(r['dataset'] for r in ssl))),
        device='cpu', threads=2, training_seeds=[42, 43, 44], training_sizes=[15, 30, 60],
        model_parameters=sum(p.numel() for p in WholeBrainCNN().parameters()),
        protocol_sha256=digest(PROTOCOL / 'protocol.json'), review_sha256=review_hash,
        preprocessing=plan['preprocessing'], pipeline_sha256=pipeline_hash(),
        source_manifest_hashes={str(p.relative_to(ROOT)): digest(p) for p in (pilot, allowed_path)},
        code_sha256=code_hashes())
    files = {'approved_cohort.csv': csv_bytes(approved), 'ssl_training.csv': csv_bytes(ssl),
             'review_snapshot.csv': (OUT / 'review.csv').read_bytes()}
    import hashlib
    if hashlib.sha256(files['review_snapshot.csv']).hexdigest() != review_hash:
        raise ValueError('QC changed during freeze; retry after closing review.csv')
    config['files_sha256'] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    with tempfile.TemporaryDirectory(prefix='.freeze-', dir=OUT) as name:
        work = Path(name)
        for filename, data in files.items():
            (work / filename).write_bytes(data)
        json_write(work / 'config.json', config)
        work.rename(ready)
    print('Frozen training setup:', ready / 'config.json')
    print('SSL training subjects:', len(ssl), '| Downstream subjects:', len(approved))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'status', 'freeze'))
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    _, rows = load_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    with lock(OUT / '.prepare.lock'):
        if args.stage == 'status':
            reviews = read(OUT / 'review.csv') if (OUT / 'review.csv').exists() else []
            print('Selected:', len(rows), '| Processing:', dict(Counter(r['status'] for r in reviews)))
            print('Decisions:', dict(Counter(r['decision'] for r in reviews)))
            print('Unlisted/not attempted:', len(rows) - len(reviews) + sum(r['status'] == 'not_attempted' for r in reviews))
            return
        plan = preparation_plan()
        if args.stage == 'freeze':
            freeze(rows, plan)
            return
        if (OUT / 'ready').exists():
            raise ValueError('Setup is frozen; preparation may not change it')
        # Reuse the established transaction/retry implementation without editing
        # frozen pilot code. This CLI owns the module in its own process.
        preparation.OUT = OUT
        with lock(ROOT / 'data/manifests/.brain_preprocessing.lock'):
            preparation.prepare(plan, rows, args.limit, args.retry_failed)


if __name__ == '__main__':
    main()
