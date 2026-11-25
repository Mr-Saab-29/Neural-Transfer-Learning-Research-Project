#!/usr/bin/env python3
"""Frozen masked-reconstruction ViT pretraining and paired downstream comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F
import train_tiny_vit as random_vit
import train_inner_validation as inner
from followup_common import (ROOT, OUT as FOLLOWUP, read, csv_bytes, digest, lock,
                             atomic_write, save_torch, json_write, load_config)
from train_wholebrain import tensors, targets, predictions, step
from tiny_vit3d import TinyViT3D
from vit_reconstruction import ReconstructionViT, foreground_mask, transfer_backbone

OUT=ROOT/'results/aomic_pretrained_vit_v1'


def freeze():
    parent,parent_hash=random_vit.load_plan()
    config,_=load_config()
    source=FOLLOWUP/'ready/ssl_training.csv'
    ssl=read(source)
    if len(ssl)!=22 or len({(r['dataset'],r['participant_id']) for r in ssl})!=22:
        raise ValueError('Expected the same 22 unique approved SSL training subjects')
    if any(r['dataset'] not in ('PT005_IXI','PT015_NKI') or r['split']!='train' or r['decision']!='accept' for r in ssl):
        raise ValueError('Only approved IXI/NKI training scans may enter pretraining')
    diagnostic=json.loads((random_vit.OUT/'overfit_report.json').read_text())
    if diagnostic['plan_sha256']!=parent_hash or not diagnostic['overfit_pass']:
        raise ValueError('Original ViT overfit diagnostic must pass')
    references={}
    for fold in range(3):
        for seed in parent['seeds']:
            run=random_vit.OUT/'runs'/f'fold{fold}-seed{seed}'
            random_vit.validate_run(run,dict(plan_sha256=parent_hash,fold=fold,seed=seed))
            for name in ('report.json','predictions.csv'):
                path=run/name
                references[str(path.relative_to(ROOT))]=digest(path)
    files={name:(random_vit.OUT/'frozen'/name).read_bytes() for name in parent['files_sha256']}
    files['ssl_training.csv']=source.read_bytes()
    plan=dict(parent_plan_sha256=parent_hash,source_ssl_sha256=digest(source),
        pretraining=dict(epochs=20,seed=42,learning_rate=.001,subjects=22,batch_size=1,
                        mask='20% random 8-cubed blocks; loss on masked nonzero voxels',
                        objective='MSE reconstruction, linear token-to-16-cubed-patch decoder',
                        checkpoint_rule='final epoch; no downstream selection'),
        supervised={k:parent[k] for k in ('max_epochs','patience','min_delta_points','learning_rate','batch_size','threads','seeds')},
        transfer='patch embedding, positions, attention blocks and final norm; fresh seed-matched regression head',
        primary='Pretrained minus random ViT out-of-fold MAE; same 48/12/30 splits and three seeds',
        limitations='Development comparison after random ViT results were viewed. One 22-subject pretraining cohort and one SSL seed. No independent confirmation or significance claim.',
        files_sha256={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        random_reference_sha256=references,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__).resolve(),ROOT/'scripts/vit_reconstruction.py')})
    files['plan.json']=(json.dumps(plan,indent=2)+'\n').encode()
    destination=OUT/'frozen'
    if destination.exists():
        if any((destination/name).read_bytes()!=data for name,data in files.items()):
            raise ValueError('Existing plan differs; preserved')
    else:
        with tempfile.TemporaryDirectory(dir=OUT,prefix='.freeze-') as directory:
            work=Path(directory)
            for name,data in files.items(): (work/name).write_bytes(data)
            work.rename(destination)
    print('Frozen:',destination)


def load_plan():
    _,parent_hash=random_vit.load_plan()
    path=OUT/'frozen/plan.json'
    plan=json.loads(path.read_text())
    if plan['parent_plan_sha256']!=parent_hash or digest(FOLLOWUP/'ready/ssl_training.csv')!=plan['source_ssl_sha256']:
        raise ValueError('Parent experiment or pretraining cohort changed')
    for mapping in ('code_sha256','random_reference_sha256'):
        for name,value in plan[mapping].items():
            if digest(ROOT/name)!=value: raise ValueError(f'Frozen dependency changed: {name}')
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
                   checkpoint_sha256=digest(path),history_sha256=digest(run/'history.csv')))
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


def summarize(plan,plan_hash):
    _,ssl_hash=backbone(plan_hash)
    combined,per_seed=[],[]
    for seed in plan['supervised']['seeds']:
        results=[]
        for fold in range(3):
            run=OUT/'runs'/f'fold{fold}-seed{seed}'
            random_vit.validate_run(run,dict(plan_sha256=plan_hash,pretraining_sha256=ssl_hash,fold=fold,seed=seed))
            original=random_vit.OUT/'runs'/f'fold{fold}-seed{seed}'
            random_vit.validate_run(original,dict(plan_sha256=plan['parent_plan_sha256'],fold=fold,seed=seed))
            reference={r['source_participant_id']:r for r in read(original/'predictions.csv')}
            part=read(run/'predictions.csv')
            expected={r['source_participant_id'] for r in read(OUT/f'frozen/fold{fold}_outer.csv')}
            if len(part)!=30 or {r['source_participant_id'] for r in part}!=expected or set(reference)!=expected:
                raise ValueError('Unpaired outer predictions')
            for row in part:
                ref=reference[row['source_participant_id']]
                if any(float(row[k])!=float(ref[k]) for k in ('observed','training_mean')):
                    raise ValueError('Target or baseline mismatch')
                results.append(dict(row,random_vit=ref['vit'],fold=fold,seed=seed))
        if len({r['source_participant_id'] for r in results})!=90: raise ValueError('Expected 90 unique subjects')
        mae={k:sum(abs(float(r[k])-float(r['observed'])) for r in results)/90 for k in ('pretrained_vit','random_vit','training_mean')}
        per_seed.append(dict(seed=seed,subjects=90,**mae,pretrained_minus_random=mae['pretrained_vit']-mae['random_vit']))
        combined.extend(results)
    result=dict(plan_sha256=plan_hash,pretraining_sha256=ssl_hash,units='IST_fluid points',per_seed=per_seed,
        mean_over_seeds={k:sum(r[k] for r in per_seed)/3 for k in ('pretrained_vit','random_vit','training_mean','pretrained_minus_random')},
        limitations=plan['limitations'])
    atomic_write(OUT/'out_of_fold_predictions.csv',csv_bytes(combined)); json_write(OUT/'comparison.json',result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','check','pretrain','train','summarize'))
    parser.add_argument('--fold',type=int,choices=range(3),default=0)
    parser.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--max-epochs',type=int,default=50)
    args=parser.parse_args()
    if args.max_epochs<1: parser.error('--max-epochs must be positive')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.manual_seed(42)
    with lock(OUT/'.run.lock'):
        if args.stage=='freeze': freeze(); return
        plan,plan_hash=load_plan()
        if args.stage=='check': check(plan_hash)
        elif args.stage=='pretrain': pretrain(plan,plan_hash,args.max_epochs)
        elif args.stage=='summarize': summarize(plan,plan_hash)
        else:
            jobs=[(f,s) for f in range(3) for s in plan['supervised']['seeds']] if args.all else [(args.fold,args.seed)]
            for fold,seed in jobs: train(plan,plan_hash,fold,seed,args.max_epochs)


if __name__=='__main__': main()
