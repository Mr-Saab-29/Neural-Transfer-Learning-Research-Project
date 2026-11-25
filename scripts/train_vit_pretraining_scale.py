#!/usr/bin/env python3
"""ViT pretraining-scale experiment, preserving all completed experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F
import train_inner_validation as inner
import prepare_pretraining_scale as cohorts
from followup_common import (ROOT, OUT as FOLLOWUP,
    read, csv_bytes, digest, lock, atomic_write, save_torch, json_write)
from train_wholebrain import tensors, targets, predictions, step

import train_pretrained_vit as original_vit
import train_tiny_vit as random_vit
from train_pretraining_scale import validate_sources
from tiny_vit3d import TinyViT3D
from vit_reconstruction import ReconstructionViT, foreground_mask, transfer_backbone

EXPERIMENT=ROOT/'results/vit_pretraining_scale_v1'
OUT=EXPERIMENT/'size60'


def validate_run(run,fingerprint):
    report=json.loads((run/'report.json').read_text())
    if report['fingerprint']!=fingerprint or any(digest(run/name)!=value for name,value in report['files_sha256'].items()):
        raise ValueError('Completed run changed')
    return report


def freeze(size):
    parent,parent_hash=random_vit.load_plan()
    _,original_hash=original_vit.load_plan()
    cohorts.load_plan()
    ready=cohorts.OUT/f'size{size}/ready'
    report=json.loads((ready/'report.json').read_text())
    if report['subjects']!=size or report['plan_sha256']!=digest(cohorts.OUT/'frozen/plan.json'):
        raise ValueError('Approved stage does not match frozen selection')
    for name,key in (('ssl_training.csv','manifest_sha256'),('review_snapshot.csv','review_sha256')):
        if digest(ready/name)!=report[key]: raise ValueError('Approved stage changed')
    source=ready/'ssl_training.csv'
    validate_sources(read(source),read(FOLLOWUP/'ready/ssl_training.csv'),
                     read(cohorts.OUT/f'frozen/cohort_{size}.csv'),size)
    # Explicitly verify 60 is nested in 120, including the same scan per subject.
    sixty=read(cohorts.OUT/'size60/ready/ssl_training.csv')
    if size==120:
        validate_sources(read(source),sixty,read(cohorts.OUT/'frozen/cohort_120.csv'),120)
    references={}
    _,original_ssl_hash=original_vit.backbone(original_hash)
    for fold in range(3):
        for seed in parent['seeds']:
            for module,fingerprint in (
                (random_vit,dict(plan_sha256=parent_hash,fold=fold,seed=seed)),
                (original_vit,dict(plan_sha256=original_hash,pretraining_sha256=original_ssl_hash,fold=fold,seed=seed))):
                run=module.OUT/'runs'/f'fold{fold}-seed{seed}'
                validate_run(run,fingerprint)
                for name in ('report.json','predictions.csv'):
                    path=run/name; references[str(path.relative_to(ROOT))]=digest(path)
    files={name:(random_vit.OUT/'frozen'/name).read_bytes() for name in parent['files_sha256']}
    files['ssl_training.csv']=source.read_bytes()
    files['review_snapshot.csv']=(ready/'review_snapshot.csv').read_bytes()
    plan=dict(size=size,parent_plan_sha256=parent_hash,original_plan_sha256=original_hash,source_manifest_sha256=digest(source),
        selection_plan_sha256=digest(cohorts.OUT/'frozen/plan.json'),
        pretraining=dict(epochs=20,seed=42,learning_rate=.001,subjects=size,batch_size=1,
            updates=20*size,objective='20% random 8-cubed block masking; MSE on masked nonzero voxels; final epoch checkpoint'),
        supervised={k:parent[k] for k in ('max_epochs','patience','min_delta_points','learning_rate','batch_size','threads','seeds')},
        architecture=parent['model'],parameter_count=parent['parameter_count'],
        primary='Pretrained-120 minus pretrained-22 early-selected ViT MAE; 60 is an intermediate scale. Also compare each against random ViT.',
        limitations='Development study after earlier results were viewed; single SSL seed. Fixed 20 epochs means 440/1200/2400 updates for 22/60/120 subjects, so data size and compute change together. No significance or causal data-only claim.',
        files_sha256={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        reference_sha256=references,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            (Path(__file__).resolve(),ROOT/'scripts/train_pretraining_scale.py',ROOT/'scripts/vit_reconstruction.py')})
    files['plan.json']=(json.dumps(plan,indent=2)+'\n').encode()
    destination=OUT/'frozen'
    if destination.exists():
        if any((destination/name).read_bytes()!=data for name,data in files.items()):
            raise ValueError('Frozen training plan differs; preserved')
    else:
        with tempfile.TemporaryDirectory(dir=OUT,prefix='.freeze-') as directory:
            work=Path(directory)
            for name,data in files.items(): (work/name).write_bytes(data)
            work.rename(destination)
    print('Frozen:',destination)


def load_plan():
    _,parent_hash=random_vit.load_plan()
    _,original_hash=original_vit.load_plan()
    cohorts.load_plan()
    path=OUT/'frozen/plan.json'; plan=json.loads(path.read_text())
    if plan['parent_plan_sha256']!=parent_hash or plan['original_plan_sha256']!=original_hash:
        raise ValueError('Frozen implementation changed')
    if digest(cohorts.OUT/'frozen/plan.json')!=plan['selection_plan_sha256']:
        raise ValueError('Cohort selection changed')
    if digest(cohorts.OUT/f"size{plan['size']}/ready/ssl_training.csv")!=plan['source_manifest_sha256']:
        raise ValueError('Approved source manifest changed')
    for name,value in plan['code_sha256'].items():
        if digest(ROOT/name)!=value: raise ValueError(f'Frozen code changed: {name}')
    for name,value in plan['reference_sha256'].items():
        if digest(ROOT/name)!=value: raise ValueError('ViT reference changed')
    for name,value in plan['files_sha256'].items():
        if digest(OUT/'frozen'/name)!=value: raise ValueError('Frozen manifest changed')
    return plan,digest(path)


def check(plan_hash):
    rows=read(OUT/'frozen/ssl_training.csv')
    images=tensors(rows)
    torch.manual_seed(42)
    model=ReconstructionViT()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    target=images[:1]
    mask=foreground_mask(target,torch.Generator().manual_seed(43))
    before=model.backbone.patch.weight.detach().clone()
    optimizer.zero_grad()
    loss=(model(target.masked_fill(mask,0))[mask]-target[mask]).square().mean()
    step(loss,model,optimizer)
    if torch.equal(before,model.backbone.patch.weight): raise ValueError('Backbone did not update')
    regressor=TinyViT3D()
    head={k:v.clone() for k,v in regressor.head.state_dict().items()}
    transfer_backbone(regressor,model.backbone.state_dict())
    if any(not torch.equal(v,regressor.head.state_dict()[k]) for k,v in head.items()):
        raise ValueError('Transfer changed regression head')
    json_write(OUT/'check.json',dict(plan_sha256=plan_hash,passed=True,subjects=len(rows),masked_loss=loss.item()))
    print('All SSL tensors verified; reconstruction gradient and transfer checks passed.')


def pretrain(plan,plan_hash,max_epochs):
    checked=json.loads((OUT/'check.json').read_text())
    if checked['plan_sha256']!=plan_hash or not checked['passed']: raise ValueError('Run check first')
    settings=plan['pretraining']
    images=tensors(read(OUT/'frozen/ssl_training.csv'))
    torch.manual_seed(settings['seed'])
    model=ReconstructionViT()
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    run=OUT/'pretraining'; run.mkdir(exist_ok=True)
    epoch,history=0,[]
    if (run/'last.pt').exists():
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if saved['plan_sha256']!=plan_hash: raise ValueError('SSL resume mismatch')
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer'])
        epoch,history=saved['epoch'],saved['history']
    for epoch in range(epoch+1,min(settings['epochs'],epoch+max_epochs)+1):
        generator=torch.Generator().manual_seed(settings['seed']+epoch)
        losses=[]
        for index in torch.randperm(len(images),generator=generator):
            target=images[index:index+1]
            hidden=foreground_mask(target,generator)
            optimizer.zero_grad()
            loss=(model(target.masked_fill(hidden,0))[hidden]-target[hidden]).square().mean()
            step(loss,model,optimizer); losses.append(loss.item())
        history.append(dict(epoch=epoch,masked_mse=sum(losses)/len(losses)))
        save_torch(run/'last.pt',dict(plan_sha256=plan_hash,epoch=epoch,history=history,
                   weights=model.state_dict(),optimizer=optimizer.state_dict()))
        print('SSL:',history[-1],flush=True)
    if history: atomic_write(run/'history.csv',csv_bytes(history))
    if epoch==settings['epochs']:
        path=run/'backbone.pt'
        if not path.exists():
            save_torch(path,dict(plan_sha256=plan_hash,epochs=epoch,backbone=model.backbone.state_dict()))
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['plan_sha256']!=plan_hash or any(not torch.equal(v,saved['backbone'][k]) for k,v in model.backbone.state_dict().items()):
            raise ValueError('Existing backbone conflicts with completed SSL run')
        probe=TinyViT3D(); transfer_backbone(probe,saved['backbone'])
        if not torch.isfinite(predictions(probe,images[:1])).all(): raise ValueError('Nonfinite transferred output')
        json_write(run/'report.json',dict(plan_sha256=plan_hash,epochs=epoch,subjects=len(images),
                   updates=epoch*len(images),checkpoint_sha256=digest(path),history_sha256=digest(run/'history.csv')))
        print('SSL complete:',path)
    else: print('Partial SSL run saved; repeat pretrain to resume.')


def backbone(plan_hash):
    path=OUT/'pretraining/backbone.pt'
    report=json.loads((path.parent/'report.json').read_text())
    if report['plan_sha256']!=plan_hash or report['epochs']!=20 or digest(path)!=report['checkpoint_sha256']:
        raise ValueError('Pretraining is incomplete or changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['plan_sha256']!=plan_hash: raise ValueError('Backbone plan mismatch')
    return state['backbone'],digest(path)


def train(plan,plan_hash,fold,seed,max_epochs):
    weights,ssl_hash=backbone(plan_hash)
    run=OUT/'runs'/f'fold{fold}-seed{seed}'; run.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(plan_sha256=plan_hash,pretraining_sha256=ssl_hash,fold=fold,seed=seed)
    if (run/'report.json').exists():
        random_vit.validate_run(run,fingerprint); print('Already complete:',run.name); return
    fitting=read(OUT/f'frozen/fold{fold}_fit.csv'); inner_rows=read(OUT/f'frozen/fold{fold}_inner.csv')
    x,vx=tensors(fitting),tensors(inner_rows)
    y,mean,std=targets(fitting); z=(y-mean)/std
    vy=torch.tensor([float(r['IST_fluid']) for r in inner_rows])
    torch.manual_seed(seed)
    model=TinyViT3D(); transfer_backbone(model,weights)
    settings=plan['supervised']
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    selection=dict(best_mae=None,best_epoch=None,best_weights=None,bad_epochs=0,stop_epoch=None)
    epoch,history,elapsed=0,[],0.
    if (run/'last.pt').exists():
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if saved['fingerprint']!=fingerprint: raise ValueError('Fine-tuning resume mismatch')
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer'])
        epoch,history,selection,elapsed=saved['epoch'],saved['history'],saved['selection'],saved['seconds']
    end=epoch if selection['stop_epoch'] is not None else min(settings['max_epochs'],epoch+max_epochs)
    for epoch in range(epoch+1,end+1):
        tick=time.perf_counter(); model.train()
        generator=torch.Generator().manual_seed(seed+epoch)
        for index in torch.randperm(len(x),generator=generator):
            optimizer.zero_grad()
            step(F.mse_loss(model(x[index:index+1]),z[index:index+1]),model,optimizer)
        train_mae=(predictions(model,x)*std+mean-y).abs().mean().item()
        val_mae=(predictions(model,vx)*std+mean-vy).abs().mean().item()
        inner.update_selection(selection,val_mae,epoch,model.state_dict(),settings['patience'],settings['min_delta_points'])
        elapsed+=time.perf_counter()-tick
        history.append(dict(epoch=epoch,training_mae=train_mae,inner_mae=val_mae))
        save_torch(run/'last.pt',dict(fingerprint=fingerprint,epoch=epoch,history=history,selection=selection,
                   weights=model.state_dict(),optimizer=optimizer.state_dict(),seconds=elapsed))
        print(run.name,history[-1],flush=True)
        if selection['stop_epoch'] is not None: break
    if history: atomic_write(run/'history.csv',csv_bytes(history))
    if epoch<settings['max_epochs'] and selection['stop_epoch'] is None:
        print('Partial run saved; outer evaluation pending.'); return
    model.load_state_dict(selection['best_weights'])
    outer=read(OUT/f'frozen/fold{fold}_outer.csv'); ox=tensors(outer)
    pred=predictions(model,ox)*std+mean
    if not torch.isfinite(pred).all(): raise ValueError('Nonfinite outer prediction')
    save_torch(run/'selected.pt',dict(weights=model.state_dict(),fingerprint=fingerprint,target_mean=mean,target_std=std))
    restored=TinyViT3D()
    restored.load_state_dict(torch.load(run/'selected.pt',map_location='cpu',weights_only=True)['weights'])
    if not torch.equal(pred,predictions(restored,ox)*std+mean): raise ValueError('Checkpoint round-trip mismatch')
    output=[dict(source_participant_id=r['source_participant_id'],observed=float(r['IST_fluid']),
                 pretrained_vit=p.item(),training_mean=mean) for r,p in zip(outer,pred)]
    atomic_write(run/'predictions.csv',csv_bytes(output))
    json_write(run/'report.json',dict(fingerprint=fingerprint,selected_epoch=selection['best_epoch'],stop_epoch=selection['stop_epoch'],
        epochs_executed=epoch,seconds=elapsed,best_inner_mae=selection['best_mae'],
        mae=sum(abs(r['pretrained_vit']-r['observed']) for r in output)/30,
        files_sha256={name:digest(run/name) for name in ('selected.pt','history.csv','predictions.csv')}))
    print('Complete:',run)


def summarize():
    global OUT
    paired={}; plans={}; checkpoints={}
    for size in (60,120):
        OUT=EXPERIMENT/f'size{size}'
        plan,plan_hash=load_plan(); plans[str(size)]=plan_hash
        _,ssl_hash=backbone(plan_hash); checkpoints[str(size)]=ssl_hash
        for seed in plan['supervised']['seeds']:
            for fold in range(3):
                run=OUT/'runs'/f'fold{fold}-seed{seed}'
                validate_run(run,dict(plan_sha256=plan_hash,pretraining_sha256=ssl_hash,fold=fold,seed=seed))
                part=read(run/'predictions.csv')
                expected={r['source_participant_id'] for r in read(OUT/f'frozen/fold{fold}_outer.csv')}
                if len(part)!=30 or {r['source_participant_id'] for r in part}!=expected:
                    raise ValueError('Unpaired or duplicate outer predictions')
                references={}
                _,original_hash=original_vit.load_plan()
                _,original_ssl_hash=original_vit.backbone(original_hash)
                for kind in ('random','pretrained'):
                    original=(random_vit.OUT if kind=='random' else original_vit.OUT)/'runs'/f'fold{fold}-seed{seed}'
                    fingerprint=dict(plan_sha256=plan['parent_plan_sha256'] if kind=='random' else original_hash,fold=fold,seed=seed)
                    if kind=='pretrained': fingerprint['pretraining_sha256']=original_ssl_hash
                    validate_run(original,fingerprint)
                    records=read(original/'predictions.csv')
                    if len(records)!=30 or {r['source_participant_id'] for r in records}!=expected:
                        raise ValueError('Reference cohort mismatch')
                    references[kind]={r['source_participant_id']:r for r in records}
                for row in part:
                    sid=row['source_participant_id']; key=(seed,sid)
                    for kind in references:
                        ref=references[kind][sid]
                        if any(float(row[k])!=float(ref[k]) for k in ('observed','training_mean')):
                            raise ValueError('Targets or baseline differ across arms')
                    record=dict(seed=seed,fold=fold,source_participant_id=sid,
                        observed=float(row['observed']),training_mean=float(row['training_mean']),
                        random=float(references['random'][sid]['vit']),pretrained22=float(references['pretrained'][sid]['pretrained_vit']))
                    if key in paired:
                        if any(paired[key][k]!=v for k,v in record.items()): raise ValueError('Pairing changed across sizes')
                    else: paired[key]=record
                    paired[key][f'pretrained{size}']=float(row['pretrained_vit'])
    per_seed=[]
    methods=('training_mean','random','pretrained22','pretrained60','pretrained120')
    import math
    for seed in (42,43,44):
        rows=[r for (s,_),r in paired.items() if s==seed]
        if len(rows)!=90 or any(not math.isfinite(r[k]) for r in rows for k in (*methods,'observed')):
            raise ValueError('Expected 90 complete finite paired predictions per seed')
        mae={k:sum(abs(r[k]-r['observed']) for r in rows)/90 for k in methods}
        per_seed.append(dict(seed=seed,subjects=90,**mae,
            scale60_minus22=mae['pretrained60']-mae['pretrained22'],
            scale120_minus22=mae['pretrained120']-mae['pretrained22'],
            scale120_minus60=mae['pretrained120']-mae['pretrained60']))
    metrics=(*methods,'scale60_minus22','scale120_minus22','scale120_minus60')
    report=dict(architecture='TinyViT3D',units='IST_fluid points',plan_sha256=plans,pretraining_sha256=checkpoints,
        ssl_updates={'22':440,'60':1200,'120':2400},per_seed=per_seed,
        mean_over_seeds={k:sum(r[k] for r in per_seed)/3 for k in metrics},limitations=plan['limitations'])
    atomic_write(EXPERIMENT/'out_of_fold_predictions.csv',csv_bytes([paired[k] for k in sorted(paired)]))
    json_write(EXPERIMENT/'comparison.json',report)
    print(json.dumps(report,indent=2))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','check','pretrain','train','summarize'))
    parser.add_argument('--size',type=int,choices=(60,120),default=60)
    parser.add_argument('--fold',type=int,choices=range(3),default=0)
    parser.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    parser.add_argument('--all',action='store_true',help='Run all nine downstream fold/seed jobs for the selected size')
    parser.add_argument('--max-epochs',type=int,default=50)
    args=parser.parse_args()
    if args.max_epochs<1: parser.error('--max-epochs must be positive')
    EXPERIMENT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.manual_seed(42)
    with lock(EXPERIMENT/'.run.lock'):
        if args.stage=='freeze':
            for size in (60,120):
                OUT=EXPERIMENT/f'size{size}'; OUT.mkdir(exist_ok=True); freeze(size)
            return
        if args.stage=='summarize': summarize(); return
        OUT=EXPERIMENT/f'size{args.size}'
        plan,plan_hash=load_plan()
        if args.stage=='check': check(plan_hash)
        elif args.stage=='pretrain': pretrain(plan,plan_hash,args.max_epochs)
        else:
            jobs=[(f,s) for f in range(3) for s in plan['supervised']['seeds']] if args.all else [(args.fold,args.seed)]
            for fold,seed in jobs: train(plan,plan_hash,fold,seed,args.max_epochs)


if __name__=='__main__': main()
