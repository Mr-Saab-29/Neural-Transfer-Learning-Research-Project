#!/usr/bin/env python3
"""Freeze, prepare, and evaluate the final AOMIC test cohort."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import tempfile
import time

import torch

from aomic_validation_batch import (FROZEN, KEYS, ROOT, REVISION, REPO, csv_bytes,
                                    digest, folder_for, read, hf_hub_download)
import preprocess_brain_pilot as brain
from preprocess_pilot import publish, validate_raw, validate_tensor
from evaluate_aomic_validation import accepted_rows
from smoke_aomic_transfer import regressor
from src.data.dataset import project_path
from visualize_brain_qc import render_scan

OUT = ROOT / 'results/aomic_final_test_v1'


def pipeline_files():
    return [Path(brain.__file__), ROOT/'src/data/brain_preprocessing.py',
            ROOT/'src/data/preprocessing.py', ROOT/'scripts/preprocess_pilot.py',
            ROOT/'scripts/build_pilot_manifest.py']


def pipeline_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in pipeline_files())).hexdigest()


def atomic_write(path, data):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.writing-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def lock(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f'Run lock exists: {path}. Remove only after confirming no run is active.')
    os.close(fd)
    try:
        yield
    finally:
        path.unlink()


def freeze(comparison):
    if (OUT/'plan.json').exists():
        raise ValueError('Test plan already frozen. Use prepare or evaluate.')
    comparison = comparison.resolve()
    summary = json.loads((comparison/'comparison.json').read_text())
    if summary['epochs'] != 50 or summary['training_subjects'] != 30:
        raise ValueError('Expected the completed 50-epoch, 30-subject comparison')
    frozen = read(FROZEN)
    for key in ('hf_path', 'source_participant_id', 'participant_id'):
        if len({r[key] for r in frozen}) != len(frozen):
            raise ValueError('Duplicate frozen subject or scan')
    labels = {r['hf_path']: r for r in frozen}
    selected = sorted([{k:r[k] for k in KEYS} for r in frozen if r['split']=='test'],
                      key=lambda r:r['source_participant_id'])
    if len(selected) != 124 or any(r['dataset']!='PT013_AIOMIC1000' for r in selected):
        raise ValueError('Expected 124 frozen AOMIC test subjects')
    models, training_hash, shared_norm = [], None, None
    preprocessing = None
    for seed in (42,43,44):
        for kind in ('pretrained','random_encoder'):
            branch = comparison/f'{seed}-{kind}'
            manifest = branch/'approved_aomic.csv'
            rows = read(manifest)
            if len(rows)!=30 or len({r['source_participant_id'] for r in rows})!=30:
                raise ValueError('Expected 30 unique training subjects')
            for row in rows:
                expected = labels[row['hf_path']]
                if row['split']!='train' or any(row[k]!=expected[k] for k in (*KEYS,'IST_fluid')):
                    raise ValueError('Checkpoint training rows disagree with frozen allocations')
                if row['decision']!='accept':
                    raise ValueError('Unapproved training scan')
            if training_hash is not None and digest(manifest)!=training_hash:
                raise ValueError('Models used different training manifests')
            training_hash = digest(manifest)
            report = json.loads((branch/'training_report.json').read_text())
            if report['epochs']!=50 or report['encoder_trainable'] or report['seed']!=seed:
                raise ValueError('Unexpected checkpoint training configuration')
            if report['manifest_sha256']!=training_hash:
                raise ValueError('Training manifest hash disagrees with report')
            checkpoint = branch/'regressor.pt'
            state = torch.load(checkpoint,map_location='cpu',weights_only=True)
            scores = torch.tensor([float(r['IST_fluid']) for r in rows])
            norm = (scores.mean().item(),scores.std(unbiased=False).item())
            if norm!=(state['target_mean'],state['target_std']) or not all(map(math.isfinite,norm)) or norm[1]<=0:
                raise ValueError('Checkpoint target scaling differs from training scores')
            if shared_norm is not None and norm!=shared_norm:
                raise ValueError('Different normalization across models')
            shared_norm = norm
            model = regressor()
            model.load_state_dict(state['weights'])
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite checkpoint weights')
            models.append(dict(name=f'{kind}_{seed}',kind=kind,seed=seed,
                               checkpoint=str(checkpoint.relative_to(ROOT)),sha256=digest(checkpoint)))
            for row in rows:
                metadata = project_path(ROOT,row['tensor_path']).parent/'metadata.json'
                if digest(metadata)!=row['metadata_sha256']:
                    raise ValueError('Training metadata changed since approval')
                provenance = json.loads(metadata.read_text())['provenance']
                current = {k:provenance[k] for k in ('synthstrip_image_id','pipeline_sha256')}
                if preprocessing is not None and current!=preprocessing:
                    raise ValueError('Mixed training preprocessing versions')
                preprocessing = current
    if preprocessing['pipeline_sha256']!=pipeline_hash():
        raise ValueError('Processing code differs from training pipeline')
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), primary='training_mean',
                metric='MAE',units='IST_fluid points', target='IST_fluid',
                comparison=str(comparison.relative_to(ROOT)),comparison_sha256=digest(comparison/'comparison.json'),
                frozen_splits_sha256=digest(FROZEN), training_manifest_sha256=training_hash,
                target_mean=shared_norm[0],target_std=shared_norm[1],models=models,preprocessing=preprocessing,
                selection_sha256=hashlib.sha256(csv_bytes(selected)).hexdigest(),
                source_revision=REVISION,patch=[48,80],selected=124)
    plan['evaluation_code_sha256'] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (Path(__file__).resolve(), ROOT/'scripts/smoke_aomic_transfer.py',
                     ROOT/'scripts/smoke_train_3d.py', ROOT/'scripts/evaluate_aomic_validation.py')}
    if not (OUT/'selection.csv').exists():
        publish(OUT/'selection.csv',lambda tmp:tmp.write_bytes(csv_bytes(selected)))
    if digest(OUT/'selection.csv')!=plan['selection_sha256']:
        raise ValueError('Existing selection conflicts with plan')
    publish(OUT/'plan.json',lambda tmp:tmp.write_text(json.dumps(plan,indent=2)+'\n'))
    print('Frozen test plan:',OUT/'plan.json')


def load_plan():
    plan = json.loads((OUT/'plan.json').read_text())
    for path, checksum in plan['evaluation_code_sha256'].items():
        if digest(project_path(ROOT,path))!=checksum:
            raise ValueError('Evaluation implementation changed after freezing the plan')
    if digest(FROZEN)!=plan['frozen_splits_sha256'] or digest(OUT/'selection.csv')!=plan['selection_sha256']:
        raise ValueError('Frozen split or test selection changed')
    for model in plan['models']:
        if digest(project_path(ROOT,model['checkpoint']))!=model['sha256']:
            raise ValueError('Frozen checkpoint changed')
    return plan,read(OUT/'selection.csv')


def prepare(plan, selected, limit, retry_failed):
    if (OUT/'final').exists():
        raise ValueError('Final evaluation already saved')
    identity = brain.image_id(plan['preprocessing']['synthstrip_image_id'])
    if identity!=plan['preprocessing']['synthstrip_image_id'] or pipeline_hash()!=plan['preprocessing']['pipeline_sha256']:
        raise ValueError('Preprocessing differs from the frozen plan')
    review = OUT/'review.csv'
    if review.exists():
        records = read(review)
        if len(records)!=len(selected) or {r['hf_path'] for r in records}!={r['hf_path'] for r in selected}:
            raise ValueError('Review checklist does not match selected subjects')
    else:
        records = [dict(hf_path=r['hf_path'],status='not_attempted',reasons='',metadata_sha256='',
                        decision='pending',notes='',error='') for r in selected]
    by_path = {r['hf_path']:r for r in records}
    logs = OUT/'logs'; logs.mkdir(exist_ok=True)
    atomic_write(review,csv_bytes(records))
    count = 0
    for row in selected:
        record = by_path[row['hf_path']]
        if record['status'] in ('pass','review') or record['decision']=='exclude':
            continue
        if record['status']=='failed' and not retry_failed:
            continue
        if count>=limit:
            break
        count+=1
        record.update(status='failed',reasons='',metadata_sha256='',decision='pending',notes='',error='')
        try:
            raw = project_path(ROOT/'data/raw',row['hf_path'])
            if not raw.exists():
                for attempt in range(3):
                    try:
                        cached = Path(hf_hub_download(repo_id=REPO,repo_type='dataset',
                            revision=plan['source_revision'],filename=row['hf_path']))
                        validate_raw(cached)
                        publish(raw,lambda tmp:shutil.copyfile(cached,tmp))
                        break
                    except Exception:
                        if attempt==2:
                            raise
                        time.sleep(2**attempt)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            folder,qc = brain.process_scan(row,identity,pipeline_hash(),1800,logs,stamp)
            reasons = brain.review_reasons(qc,row['dataset'])
            render_scan(folder,OUT/(row['participant_id']+'.png'),row['participant_id']+' | '+(', '.join(reasons) or 'automated pass'))
            record.update(status='review' if reasons else 'pass',reasons=';'.join(reasons),metadata_sha256=digest(folder/'metadata.json'))
        except Exception as error:
            record['error']=f'{type(error).__name__}: {error}'
        atomic_write(review,csv_bytes(records))
        print(row['participant_id'],record['status'],record['error'],flush=True)
    print('Preparation status:',dict(Counter(r['status'] for r in records)))
    print('Inspect PNGs and record accept/exclude in',review)


def evaluate_final(plan, selected):
    review = OUT/'review.csv'
    records = read(review)
    accepted = accepted_rows(selected,records)
    if len(accepted)<2:
        raise ValueError('At least two accepted test scans are required')
    # Resolve all QC before reading held-out scores or making predictions.
    review_hash = digest(review)
    fingerprint = dict(plan_sha256=digest(OUT/'plan.json'),review_sha256=review_hash)
    final = OUT/'final'
    if final.exists():
        report = json.loads((final/'report.json').read_text())
        if report['inputs']!=fingerprint:
            raise ValueError('Final evaluation exists with different inputs')
        print(json.dumps(report,indent=2))
        return
    states = []
    torch.set_num_threads(1)
    for item in plan['models']:
        state = torch.load(project_path(ROOT,item['checkpoint']),map_location='cpu',weights_only=True)
        if (state['target_mean'],state['target_std'])!=(plan['target_mean'],plan['target_std']):
            raise ValueError('Checkpoint scaling differs from plan')
        model = regressor().eval()
        model.load_state_dict(state['weights'])
        states.append((item['name'],model))
    frozen = {r['hf_path']:r for r in read(FROZEN) if r['split']=='test'}
    predictions = []
    with torch.inference_mode():
        for row,qc in accepted:
            folder = folder_for(row)
            metadata = folder/'metadata.json'
            if digest(metadata)!=qc['metadata_sha256']:
                raise ValueError('Visual approval is stale')
            saved = json.loads(metadata.read_text())
            provenance = saved['provenance']
            if provenance['hf_path']!=row['hf_path'] or any(provenance[k]!=v for k,v in plan['preprocessing'].items()):
                raise ValueError('Test processing differs from frozen protocol')
            for name in brain.ARTIFACTS:
                if digest(folder/name)!=saved['sha256'][name]:
                    raise ValueError(f'Changed test artifact: {name}')
            image = torch.load(folder/'tensor.pt',map_location='cpu',weights_only=True)
            validate_tensor(image)
            label = frozen[row['hf_path']]
            if any(label[k]!=row[k] for k in KEYS):
                raise ValueError('Test label identity mismatch')
            observed = float(label[plan['target']])
            result = dict(participant_id=row['participant_id'],source_participant_id=row['source_participant_id'],
                          observed=observed,training_mean=plan['target_mean'])
            low,high = plan['patch']
            patch = image.unsqueeze(0)[:,:,low:high,low:high,low:high].contiguous()
            for name,model in states:
                result[name]=(model(patch)*plan['target_std']+plan['target_mean']).item()
            if not all(math.isfinite(result[k]) for k in ('observed','training_mean',*(name for name,_ in states))):
                raise ValueError('Nonfinite test score or prediction')
            predictions.append(result)
    if digest(review)!=review_hash:
        raise ValueError('QC changed during evaluation')
    load_plan()
    names = ['training_mean']+[name for name,_ in states]
    maes = {name:mean(abs(r[name]-r['observed']) for r in predictions) for name in names}
    grouped = {}
    for kind in ('pretrained','random_encoder'):
        values = [maes[m['name']] for m in plan['models'] if m['kind']==kind]
        grouped[kind]=dict(mean_mae=mean(values),sd_across_seeds=stdev(values))
    report = dict(inputs=fingerprint,selected=len(selected),accepted=len(accepted),
                  excluded=len(selected)-len(accepted),automated_status=dict(Counter(r['status'] for r in records)),
                  primary_model=plan['primary'],primary_mae=maes[plan['primary']],
                  units=plan['units'],mae=maes,seed_summary=grouped,
                  exclusions=[dict(hf_path=r['hf_path'],status=r['status'],reason=r['notes']) for r in records if r['decision']=='exclude'])
    with tempfile.TemporaryDirectory(prefix='.evaluation-',dir=OUT) as directory:
        temporary = Path(directory)
        (temporary/'predictions.csv').write_bytes(csv_bytes(predictions))
        (temporary/'review_snapshot.csv').write_bytes(review.read_bytes())
        (temporary/'plan_snapshot.json').write_bytes((OUT/'plan.json').read_bytes())
        (temporary/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        temporary.rename(final)
    print(json.dumps(report,indent=2),'\nSaved:',final)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','prepare','evaluate','status'))
    parser.add_argument('--comparison',type=Path)
    parser.add_argument('--limit',type=int,default=10,help='Maximum attempted scans this invocation')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if args.limit<1 or (args.stage=='freeze' and args.comparison is None):
        parser.error('Use a positive --limit and supply --comparison for freeze')
    OUT.mkdir(parents=True,exist_ok=True)
    with lock(OUT/'.run.lock'):
        if args.stage=='freeze':
            freeze(args.comparison)
            return
        plan,selected=load_plan()
        if args.stage=='prepare':
            with lock(ROOT/'data/manifests/.brain_preprocessing.lock'):
                prepare(plan,selected,args.limit,args.retry_failed)
        elif args.stage=='evaluate':
            evaluate_final(plan,selected)
        else:
            path=OUT/'review.csv'
            print('Selected:',len(selected))
            if path.exists():
                rows=read(path)
                print('Processing:',dict(Counter(r['status'] for r in rows)))
                print('Visual decisions:',dict(Counter(r['decision'] for r in rows)))
            else:
                print('Preparation pending')


if __name__=='__main__':
    main()
