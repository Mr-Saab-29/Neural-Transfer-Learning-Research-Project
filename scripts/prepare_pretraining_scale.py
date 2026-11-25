#!/usr/bin/env python3
"""Freeze nested 22/60/120 IXI/NKI cohorts and prepare them in QC-gated stages."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import final_aomic_test as preparation
from followup_common import (ROOT, read, csv_bytes, digest, lock, atomic_write,
    folder_for, approved_artifact, whole_image, load_config)
from aomic_validation_batch import REVISION
import preprocess_brain_pilot as brain

OUT=ROOT/'results/pretraining_scale_v1'
POOL=ROOT/'data/manifests/pretraining_without_aomic_v1.csv'
BASE=ROOT/'results/aomic_followup_v1/ready/ssl_training.csv'
PILOT=ROOT/'data/manifests/pilot_training.csv'
SOURCES=('PT005_IXI','PT015_NKI')
FIELDS=('dataset','participant_id','hf_path')


def select_nested(pool,base,heldout):
    if Counter(r['dataset'] for r in base)!={s:11 for s in SOURCES}:
        raise ValueError('Expected original 11 IXI + 11 NKI subjects')
    keys=[(r['dataset'],r['participant_id']) for r in base]
    if len(set(keys))!=22: raise ValueError('Duplicate original subject')
    excluded={(r['dataset'],r['participant_id']) for r in heldout}
    if set(keys)&excluded: raise ValueError('Original SSL subjects overlap preserved pilot holdouts')
    available={}
    for row in pool:
        if row['dataset'] not in SOURCES: raise ValueError('Only IXI/NKI permitted')
        key=row['dataset'],row['participant_id']
        available.setdefault(key,[]).append(row)
    for row in base:
        if row['hf_path'] not in {r['hf_path'] for r in available.get((row['dataset'],row['participant_id']),[])}:
            raise ValueError('Original scan missing from candidate pool')
    cohorts={22:[{k:r[k] for k in FIELDS} for r in base],60:[],120:[]}
    for source in SOURCES:
        initial=[{k:r[k] for k in FIELDS} for r in base if r['dataset']==source]
        initial.sort(key=lambda r:r['participant_id'])
        candidates=[]
        for key,scans in available.items():
            if key[0]!=source or key in keys or key in excluded: continue
            chosen=min(scans,key=lambda r:(r.get('baseline_priority','').lower()!='true',r.get('session_id',''),r['hf_path']))
            candidates.append({k:chosen[k] for k in FIELDS})
        candidates.sort(key=lambda r:(hashlib.sha256(
            f"ssl-scale-v1-42|{source}|{r['participant_id']}".encode()).hexdigest(),r['participant_id']))
        if len(candidates)<49: raise ValueError('Insufficient unique candidates')
        for size in (60,120): cohorts[size].extend(initial+candidates[:size//2-11])
    for size,rows in cohorts.items():
        if len({r['participant_id'] for r in rows})!=size or len({r['hf_path'] for r in rows})!=size:
            raise ValueError('Nonunique subject IDs or paths; unsafe for existing PNG filenames')
    return cohorts


def freeze_selection():
    config,parent_hash=load_config()
    base=read(BASE)
    if any(r['decision']!='accept' or r['split']!='train' for r in base):
        raise ValueError('Original SSL training approval missing')
    heldout=[r for r in read(PILOT) if r['dataset'] in SOURCES and r['split']!='train']
    cohorts=select_nested(read(POOL),base,heldout)
    files={f'cohort_{n}.csv':csv_bytes(rows) for n,rows in cohorts.items()}
    plan=dict(parent_config_sha256=parent_hash,source_revision=REVISION,
        preprocessing=config['preprocessing'],selection_seed='ssl-scale-v1-42',
        sizes=[22,60,120],balance='Equal IXI and NKI; original 22 retained exactly',
        selection='Subject-ID hashing; one baseline-priority then lexicographic scan per new subject. Original pilot non-training IXI/NKI subjects remain excluded.',
        qc='All selected subjects must be accepted. No automatic replacement/exclusion. A failure to approve requires a documented cohort revision before training.',
        staging='Prepare and approve 60 first; then 120. Progression is based on preparation capacity, not favorable downstream scores.',
        intended_experiment='CNN only; unchanged AOMIC 48/12/30 splits. Intended 20 SSL epochs: 440/1200/2400 batch-one updates at 22/60/120 subjects. Data size and compute both increase; not a matched-update causal comparison.',
        training_status='Selection and preprocessing only; expanded-cohort training configuration must be frozen separately. Do not modify the completed experiment to substitute these manifests.',
        code_sha256=digest(Path(__file__)),
        inputs_sha256={str(p.relative_to(ROOT)):digest(p) for p in (POOL,BASE,PILOT)},
        files_sha256={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    files['plan.json']=(json.dumps(plan,indent=2)+'\n').encode()
    destination=OUT/'frozen'
    if destination.exists():
        if any((destination/name).read_bytes()!=data for name,data in files.items()):
            raise ValueError('Existing selection differs; preserved')
    else:
        with tempfile.TemporaryDirectory(dir=OUT,prefix='.selection-') as directory:
            temporary=Path(directory)
            for name,data in files.items(): (temporary/name).write_bytes(data)
            temporary.rename(destination)
    print('Frozen 22/60/120 nested selections:',destination)


def load_plan():
    config,parent_hash=load_config()
    plan=json.loads((OUT/'frozen/plan.json').read_text())
    if plan['parent_config_sha256']!=parent_hash or plan['code_sha256']!=digest(Path(__file__)):
        raise ValueError('Frozen dependencies changed')
    for name,value in plan['inputs_sha256'].items():
        if digest(ROOT/name)!=value: raise ValueError('Selection source changed')
    for name,value in plan['files_sha256'].items():
        if digest(OUT/'frozen'/name)!=value: raise ValueError('Frozen cohort changed')
    if plan['preprocessing']!=config['preprocessing']: raise ValueError('Preprocessing mismatch')
    return plan


def import_approvals(plan,selected,stage):
    path=stage/'review.csv'
    records=read(path) if path.exists() else [dict(hf_path=r['hf_path'],status='not_attempted',reasons='',
              metadata_sha256='',decision='pending',notes='',error='') for r in selected]
    if len(records)!=len(selected) or {r['hf_path'] for r in records}!={r['hf_path'] for r in selected}:
        raise ValueError('Checklist differs from selection')
    sources=[ROOT/'data/manifests/brain_qc_decisions.csv']
    if stage.name=='size120': sources.append(OUT/'size60/ready/review_snapshot.csv')
    previous={}
    for source in sources:
        if source.exists():
            for r in read(source):
                if r['decision']=='accept': previous[r['hf_path']]=(r,source)
    # Original accepted training manifest is an approval source even when the
    # earlier checklist is unavailable. Full artifact provenance is checked below.
    for row in read(BASE):
        previous.setdefault(row['hf_path'],(dict(row,notes='Original accepted SSL training manifest'),BASE))
    by_path={r['hf_path']:r for r in selected}
    imported=0
    for record in records:
        if record['decision']!='pending' or record['hf_path'] not in previous: continue
        prior,source=previous[record['hf_path']]
        row=by_path[record['hf_path']]
        try:
            folder=folder_for(row)
            saved=json.loads((folder/'metadata.json').read_text())
            reasons=brain.review_reasons(saved['qc'],row['dataset'])
            notes=prior.get('notes','').strip()
            if reasons and not notes: continue
            candidate=dict(record,status='review' if reasons else 'pass',reasons=';'.join(reasons),
                metadata_sha256=prior['metadata_sha256'],decision='accept',
                notes=f"Reused verified approval from {source.relative_to(ROOT)}. {notes}",error='')
            approved_artifact(row,candidate,plan['preprocessing'])
        except (ValueError,OSError,KeyError) as error:
            print('Approval not reused:',row['participant_id'],str(error),flush=True)
            continue
        record.update(candidate); imported+=1
    atomic_write(path,csv_bytes(records))
    print('Verified approvals imported:',imported)


def freeze_approved(plan,selected,stage):
    if (stage/'ready').exists(): raise ValueError('Approved manifest already frozen')
    review=stage/'review.csv'; checksum=digest(review)
    records=read(review); lookup={r['hf_path']:r for r in records}
    if len(records)!=len(selected) or len(lookup)!=len(records) or set(lookup)!={r['hf_path'] for r in selected}:
        raise ValueError('Review must cover exact cohort')
    approved=[]
    for row in selected:
        artifact=approved_artifact(row,lookup[row['hf_path']],plan['preprocessing'])
        whole_image(artifact)
        approved.append(dict(artifact,split='train',qc_status=lookup[row['hf_path']]['status']))
    if digest(review)!=checksum: raise ValueError('Review changed while freezing')
    with tempfile.TemporaryDirectory(dir=stage,prefix='.ready-') as directory:
        work=Path(directory)
        (work/'ssl_training.csv').write_bytes(csv_bytes(approved))
        (work/'review_snapshot.csv').write_bytes(review.read_bytes())
        (work/'report.json').write_text(json.dumps(dict(subjects=len(approved),
            datasets=dict(Counter(r['dataset'] for r in approved)),plan_sha256=digest(OUT/'frozen/plan.json'),
            manifest_sha256=digest(work/'ssl_training.csv'),review_sha256=checksum),indent=2)+'\n')
        work.rename(stage/'ready')
    print('Approved SSL manifest:',stage/'ready/ssl_training.csv')


def require_sixty_ready():
    ready=OUT/'size60/ready'
    report=json.loads((ready/'report.json').read_text())
    if report['subjects']!=60 or report['plan_sha256']!=digest(OUT/'frozen/plan.json') or digest(ready/'ssl_training.csv')!=report['manifest_sha256'] or digest(ready/'review_snapshot.csv')!=report['review_sha256']:
        raise ValueError('The approved 60-subject stage changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('select','prepare','status','approve'))
    parser.add_argument('--size',type=int,choices=(60,120),default=60)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if args.limit<1: parser.error('--limit must be positive')
    OUT.mkdir(parents=True,exist_ok=True)
    with lock(OUT/'.run.lock'):
        if args.action=='select': freeze_selection(); return
        plan=load_plan()
        selected=read(OUT/f'frozen/cohort_{args.size}.csv')
        stage=OUT/f'size{args.size}'; stage.mkdir(exist_ok=True)
        if args.action=='status':
            records=read(stage/'review.csv') if (stage/'review.csv').exists() else []
            print('Selected:',len(selected),'Processing:',dict(Counter(r['status'] for r in records)),
                  'Decisions:',dict(Counter(r['decision'] for r in records)),
                  'Unlisted:',len(selected)-len(records))
            return
        if args.size==120: require_sixty_ready()
        if args.action=='approve': freeze_approved(plan,selected,stage); return
        if (stage/'ready').exists(): raise ValueError('Stage already approved; preparation locked')
        with lock(ROOT/'data/manifests/.brain_preprocessing.lock'):
            import_approvals(plan,selected,stage)
            preparation.OUT=stage
            preparation.prepare(plan,selected,args.limit,args.retry_failed)


if __name__=='__main__': main()
