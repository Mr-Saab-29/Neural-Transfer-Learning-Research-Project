#!/usr/bin/env python3
"""Random ViT versus saved random CNN under the same inner-validation design."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F
import train_inner_validation as cnn
from followup_common import ROOT, WholeBrainCNN, read, csv_bytes, digest, lock, atomic_write, save_torch, json_write
from train_wholebrain import tensors, targets, predictions, step, finite_model
from tiny_vit3d import TinyViT3D

OUT = ROOT / 'results/aomic_tiny_vit_v1'


def freeze():
    parent, parent_hash = cnn.load_plan()
    files = {name: (cnn.OUT / 'frozen' / name).read_bytes() for name in parent['files_sha256']}
    references = {}
    for fold in range(3):
        for seed in parent['seeds']:
            run = cnn.OUT / 'runs' / f'fold{fold}-seed{seed}-random'
            cnn.validate_report(run, dict(plan_sha256=parent_hash, fold=fold, seed=seed, kind='random'))
            for filename in ('report.json', 'predictions.csv'):
                path = run / filename
                references[str(path.relative_to(ROOT))] = digest(path)
    import hashlib
    plan = dict(parent_plan_sha256=parent_hash,
        model='TinyViT3D: 16^3 patches, 64 tokens, width 48, 2 blocks, 3 heads, MLP ratio 2, learned positions, mean pooling',
        initialization='random only; no transformer pretraining',
        parameter_count=sum(p.numel() for p in TinyViT3D().parameters()),
        cnn_parameter_count=sum(p.numel() for p in WholeBrainCNN().parameters()),
        max_epochs=50, patience=10, min_delta_points=.1, learning_rate=.001,
        optimizer='Adam', batch_size=1, threads=2, augmentation='none', dropout=0., seeds=[42,43,44],
        normalization='Fit mean and population SD on the same 48 fitting subjects only',
        primary='ViT minus saved early-selected random CNN MAE; pool 90 out-of-fold subjects per seed, then average MAE across seeds',
        limitations='Development comparison after CNN result exposure. Different parameter counts, coarse patching, same un-tuned optimizer schedule; not a parameter-matched or optimally tuned architecture comparison. No significance claim.',
        timing='ViT training/inner-selection wall time recorded across invocations. Original CNN timing unavailable; do not claim speedup from these timings.',
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__).resolve(), ROOT/'scripts/tiny_vit3d.py')},
        cnn_reference_sha256=references,
        files_sha256={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    files['plan.json'] = (json.dumps(plan, indent=2)+'\n').encode()
    destination=OUT/'frozen'
    if destination.exists():
        if any((destination/name).read_bytes()!=data for name,data in files.items()):
            raise ValueError('Frozen transformer plan differs; preserved')
    else:
        with tempfile.TemporaryDirectory(dir=OUT,prefix='.freeze-') as directory:
            work=Path(directory)
            for name,data in files.items():
                (work/name).write_bytes(data)
            work.rename(destination)
    print('Frozen:', destination, '\nParameters:',plan['parameter_count'], '| CNN:',plan['cnn_parameter_count'])


def load_plan():
    _, parent_hash=cnn.load_plan()
    path=OUT/'frozen/plan.json'
    plan=json.loads(path.read_text())
    if plan['parent_plan_sha256']!=parent_hash:
        raise ValueError('Parent plan changed')
    for mapping in ('code_sha256','cnn_reference_sha256'):
        for name,checksum in plan[mapping].items():
            if digest(ROOT/name)!=checksum:
                raise ValueError(f'Frozen dependency changed: {name}')
    for name,checksum in plan['files_sha256'].items():
        if digest(OUT/'frozen'/name)!=checksum:
            raise ValueError('Frozen split changed')
    return plan,digest(path)


def rows(fold, role):
    return read(OUT/f'frozen/fold{fold}_{role}.csv')


def check(plan_hash):
    for fold in range(3):
        groups=[rows(fold,role) for role in ('fit','inner','outer')]
        ids=[{r['source_participant_id'] for r in group} for group in groups]
        if [len(g) for g in groups]!=[48,12,30] or [len(g) for g in ids]!=[48,12,30] or len(set.union(*ids))!=90:
            raise ValueError('Invalid or overlapping subject partitions')
    x=tensors(rows(0,'fit')[:1])
    model=TinyViT3D()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    step(model(x).square().mean(),model,optimizer)
    json_write(OUT/'loader_check.json',dict(plan_sha256=plan_hash,passed=True,shape=list(x.shape)))
    print('Loader, shape and gradient check passed.')


def overfit(plan_hash):
    checked=json.loads((OUT/'loader_check.json').read_text())
    if checked['plan_sha256']!=plan_hash or not checked['passed']:
        raise ValueError('Run check first')
    subset=rows(0,'fit')[:4]
    x=tensors(subset)
    y,mean,std=targets(subset)
    z=(y-mean)/std
    torch.manual_seed(42)
    model=TinyViT3D()
    optimizer=torch.optim.LBFGS(model.parameters(),lr=.5,max_iter=10,history_size=20,line_search_fn='strong_wolfe')
    history=[]
    for epoch in range(101):
        if epoch:
            def closure():
                optimizer.zero_grad()
                loss=F.mse_loss(model(x),z)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite diagnostic loss')
                loss.backward()
                return loss
            optimizer.step(closure)
            finite_model(model)
        mae=(predictions(model,x)*std+mean-y).abs().mean().item()
        history.append(dict(epoch=epoch,mae=mae))
        print('Overfit:',history[-1],flush=True)
        if mae<1:
            break
    save_torch(OUT/'overfit.pt',dict(weights=model.state_dict(),target_mean=mean,target_std=std))
    restored=TinyViT3D()
    restored.load_state_dict(torch.load(OUT/'overfit.pt',weights_only=True,map_location='cpu')['weights'])
    if not torch.equal(predictions(model,x),predictions(restored,x)):
        raise ValueError('Diagnostic checkpoint round-trip mismatch')
    atomic_write(OUT/'overfit_history.csv',csv_bytes(history))
    json_write(OUT/'overfit_report.json',dict(plan_sha256=plan_hash,overfit_pass=mae<1,final_mae=mae,
        checkpoint_sha256=digest(OUT/'overfit.pt'),subjects=[r['source_participant_id'] for r in subset]))
    print('Overfit pass:',mae<1)


def validate_run(run,fingerprint):
    report=json.loads((run/'report.json').read_text())
    if report['fingerprint']!=fingerprint or any(digest(run/k)!=v for k,v in report['files_sha256'].items()):
        raise ValueError('Completed transformer run changed')
    return report


def train(plan,plan_hash,fold,seed,max_epochs):
    diagnostic=json.loads((OUT/'overfit_report.json').read_text())
    if diagnostic['plan_sha256']!=plan_hash or not diagnostic['overfit_pass'] or digest(OUT/'overfit.pt')!=diagnostic['checkpoint_sha256']:
        raise ValueError('A passing transformer overfit check is required')
    run=OUT/'runs'/f'fold{fold}-seed{seed}'
    run.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(plan_sha256=plan_hash,fold=fold,seed=seed)
    if (run/'report.json').exists():
        validate_run(run,fingerprint)
        print('Already complete:',run.name)
        return
    fitting,inner=rows(fold,'fit'),rows(fold,'inner')
    x,vx=tensors(fitting),tensors(inner)
    y,mean,std=targets(fitting)
    vy=torch.tensor([float(r['IST_fluid']) for r in inner])
    z=(y-mean)/std
    torch.manual_seed(seed)
    model=TinyViT3D()
    optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'])
    selection=dict(best_mae=None,best_epoch=None,best_weights=None,bad_epochs=0,stop_epoch=None)
    start,history,elapsed=0,[],0.
    if (run/'last.pt').exists():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint:
            raise ValueError('Resume input mismatch')
        model.load_state_dict(state['weights']); optimizer.load_state_dict(state['optimizer'])
        start,history,selection,elapsed=state['epoch'],state['history'],state['selection'],state['elapsed_seconds']
    epoch=start
    end=start if selection['stop_epoch'] is not None else min(plan['max_epochs'],start+max_epochs)
    for epoch in range(start+1,end+1):
        tick=time.perf_counter()
        model.train()
        generator=torch.Generator().manual_seed(seed+epoch)
        for index in torch.randperm(len(x),generator=generator):
            optimizer.zero_grad()
            step(F.mse_loss(model(x[index:index+1]),z[index:index+1]),model,optimizer)
        training_mae=(predictions(model,x)*std+mean-y).abs().mean().item()
        inner_mae=(predictions(model,vx)*std+mean-vy).abs().mean().item()
        cnn.update_selection(selection,inner_mae,epoch,model.state_dict(),plan['patience'],plan['min_delta_points'])
        elapsed+=time.perf_counter()-tick
        history.append(dict(epoch=epoch,training_mae=training_mae,inner_mae=inner_mae))
        save_torch(run/'last.pt',dict(fingerprint=fingerprint,epoch=epoch,history=history,selection=selection,
            weights=model.state_dict(),optimizer=optimizer.state_dict(),elapsed_seconds=elapsed))
        atomic_write(run/'history.csv',csv_bytes(history))
        print(run.name,history[-1],flush=True)
        if selection['stop_epoch'] is not None:
            break
    # Repair the derived CSV if interrupted after the atomic checkpoint write.
    if history:
        atomic_write(run/'history.csv',csv_bytes(history))
    if selection['stop_epoch'] is None and epoch<50:
        print('Saved partial run; outer evaluation pending.')
        return
    model.load_state_dict(selection['best_weights'])
    outer=rows(fold,'outer')
    ox=tensors(outer)
    pred=predictions(model,ox)*std+mean
    if not torch.isfinite(pred).all():
        raise ValueError('Nonfinite outer prediction')
    save_torch(run/'selected.pt',dict(weights=model.state_dict(),target_mean=mean,target_std=std,
        fingerprint=fingerprint,selected_epoch=selection['best_epoch']))
    restored=TinyViT3D()
    restored.load_state_dict(torch.load(run/'selected.pt',map_location='cpu',weights_only=True)['weights'])
    if not torch.equal(pred,predictions(restored,ox)*std+mean):
        raise ValueError('Checkpoint round-trip mismatch')
    output=[dict(source_participant_id=r['source_participant_id'],observed=float(r['IST_fluid']),
                 vit=p.item(),training_mean=mean) for r,p in zip(outer,pred)]
    atomic_write(run/'predictions.csv',csv_bytes(output))
    json_write(run/'report.json',dict(fingerprint=fingerprint,selected_epoch=selection['best_epoch'],
        stop_epoch=selection['stop_epoch'],epochs_executed=epoch,best_inner_mae=selection['best_mae'],
        training_selection_seconds=elapsed,target_mean=mean,target_std=std,
        mae=sum(abs(r['vit']-r['observed']) for r in output)/30,
        files_sha256={name:digest(run/name) for name in ('selected.pt','predictions.csv','history.csv')}))
    print('Complete:',run)


def summarize(plan,plan_hash):
    combined,per_seed,runtime=[],[],[]
    for seed in plan['seeds']:
        predictions_by_seed=[]
        for fold in range(3):
            run=OUT/'runs'/f'fold{fold}-seed{seed}'
            report=validate_run(run,dict(plan_sha256=plan_hash,fold=fold,seed=seed))
            cnn_run=cnn.OUT/'runs'/f'fold{fold}-seed{seed}-random'
            cnn.validate_report(cnn_run,dict(plan_sha256=plan['parent_plan_sha256'],fold=fold,seed=seed,kind='random'))
            reference={r['source_participant_id']:r for r in read(cnn_run/'predictions.csv')}
            part=read(run/'predictions.csv')
            expected={r['source_participant_id'] for r in rows(fold,'outer')}
            if len(part)!=30 or {r['source_participant_id'] for r in part}!=expected or set(reference)!=expected:
                raise ValueError('Unpaired or duplicate outer predictions')
            for row in part:
                ref=reference[row['source_participant_id']]
                if any(float(row[k])!=float(ref[k]) for k in ('observed','training_mean')):
                    raise ValueError('Target or baseline mismatch between architectures')
                predictions_by_seed.append(dict(row,cnn=ref['early'],fold=fold,seed=seed))
            runtime.append(dict(fold=fold,seed=seed,selected_epoch=report['selected_epoch'],
                                epochs_executed=report['epochs_executed'],seconds=report['training_selection_seconds']))
        if len({r['source_participant_id'] for r in predictions_by_seed})!=90:
            raise ValueError('Expected 90 unique out-of-fold subjects')
        mae={k:sum(abs(float(r[k])-float(r['observed'])) for r in predictions_by_seed)/90 for k in ('vit','cnn','training_mean')}
        per_seed.append(dict(seed=seed,subjects=90,**mae,vit_minus_cnn=mae['vit']-mae['cnn']))
        combined.extend(predictions_by_seed)
    atomic_write(OUT/'out_of_fold_predictions.csv',csv_bytes(combined))
    result=dict(plan_sha256=plan_hash,units='IST_fluid points',per_seed=per_seed,
        mean_over_seeds={k:sum(r[k] for r in per_seed)/3 for k in ('vit','cnn','training_mean','vit_minus_cnn')},
        vit_parameters=plan['parameter_count'],cnn_parameters=plan['cnn_parameter_count'],runtime=runtime,
        limitations=plan['limitations'],timing_caveat=plan['timing'])
    json_write(OUT/'comparison.json',result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','check','overfit','train','summarize'))
    parser.add_argument('--fold',type=int,choices=range(3),default=0)
    parser.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--max-epochs',type=int,default=50)
    args=parser.parse_args()
    if args.max_epochs<1:
        parser.error('--max-epochs must be positive')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.manual_seed(42)
    with lock(OUT/'.run.lock'):
        if args.stage=='freeze':
            freeze(); return
        plan,plan_hash=load_plan()
        if args.stage=='check': check(plan_hash)
        elif args.stage=='overfit': overfit(plan_hash)
        elif args.stage=='summarize': summarize(plan,plan_hash)
        else:
            jobs=[(f,s) for f in range(3) for s in plan['seeds']] if args.all else [(args.fold,args.seed)]
            for fold,seed in jobs:
                train(plan,plan_hash,fold,seed,args.max_epochs)


if __name__=='__main__':
    main()
