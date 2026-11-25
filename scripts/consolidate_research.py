#!/usr/bin/env python3
"""Consolidate saved reports into tables/figures; no fitting or model evaluation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'pilot':'aomic_final_test_v1/final/report.json',
    'labels':'aomic_followup_v1/comparison.json',
    'selection':'aomic_inner_validation_v1/comparison.json',
    'vit':'aomic_tiny_vit_v1/comparison.json',
    'vit_ssl':'aomic_pretrained_vit_v1/comparison.json',
    'scale':'cnn_pretraining_scale_v1/comparison.json',
    'vit_scale':'vit_pretraining_scale_v1/comparison.json',
}


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline='') as stream: return list(csv.DictReader(stream))


def close(a,b,label):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(float(a)-float(b))>1e-5:
        raise ValueError(f'Inconsistent/nonfinite result: {label}')


def validate_reports(reports):
    for name,r in reports.items():
        if r['units']!='IST_fluid points': raise ValueError(f'Unexpected units: {name}')
    for name in ('vit','vit_ssl','scale','vit_scale'):
        report=reports[name]
        if sorted(r['seed'] for r in report['per_seed'])!=[42,43,44]:
            raise ValueError(f'Missing or duplicate seeds: {name}')
        if any(r['subjects']!=90 for r in report['per_seed']): raise ValueError('Unexpected cohort count')
        for metric,value in report['mean_over_seeds'].items():
            close(value,sum(r[metric] for r in report['per_seed'])/3,f'{name}/{metric}')
    for r in reports['selection']['mean_over_seeds']:
        records=[v for v in reports['selection']['per_seed'] if v['kind']==r['kind']]
        if sorted(v['seed'] for v in records)!=[42,43,44] or any(v['subjects']!=90 for v in records):
            raise ValueError('Incomplete selection comparison')
        for metric in ('early','fixed50','training_mean','early_minus_fixed50'):
            close(r[metric],sum(v[metric] for v in records)/3,metric)
    for r in reports['labels']['mean_over_seeds']:
        records=[v for v in reports['labels']['per_seed'] if (v['size'],v['kind'])==(r['size'],r['kind'])]
        if sorted(v['seed'] for v in records)!=[42,43,44]: raise ValueError('Incomplete label curve')
        close(r['mae'],sum(v['model_mae'] for v in records)/3,'label curve')
    selected={r['kind']:r for r in reports['selection']['mean_over_seeds']}
    scale=reports['scale']['mean_over_seeds']
    close(scale['random'],selected['random']['early'],'random CNN reuse')
    close(scale['pretrained22'],selected['pretrained']['early'],'22 CNN reuse')
    close(reports['vit']['mean_over_seeds']['cnn'],scale['random'],'CNN architecture reference')
    close(reports['vit']['mean_over_seeds']['vit'],reports['vit_ssl']['mean_over_seeds']['random_vit'],'ViT reference')
    for name in ('vit','vit_ssl','vit_scale'):
        close(reports[name]['mean_over_seeds']['training_mean'],scale['training_mean'],'baseline reuse')
    for seed in (42,43,44):
        expanded=next(r for r in reports['vit_scale']['per_seed'] if r['seed']==seed)
        original=next(r for r in reports['vit_ssl']['per_seed'] if r['seed']==seed)
        for new,old in (('random','random_vit'),('pretrained22','pretrained_vit'),('training_mean','training_mean')):
            close(expanded[new],original[old],f'ViT seed {seed} reference {new}')
    for name in ('scale','vit_scale'):
        for row in reports[name]['per_seed']:
            for larger,smaller in ((60,22),(120,22),(120,60)):
                close(row[f'scale{larger}_minus{smaller}'],row[f'pretrained{larger}']-row[f'pretrained{smaller}'],f'{name} scale contrast')
    pilot=reports['pilot']
    if (pilot['selected'],pilot['accepted'],pilot['excluded'])!=(124,123,1): raise ValueError('Unexpected pilot cohort')
    for kind in ('pretrained','random_encoder'):
        close(pilot['seed_summary'][kind]['mean_mae'],sum(pilot['mae'][f'{kind}_{s}'] for s in (42,43,44))/3,'pilot mean')


def consolidate(output):
    if output.exists(): raise FileExistsError(f'Output exists; choose a new --output: {output}')
    hashes={}
    def track(path):
        hashes[str(path.relative_to(ROOT))]=digest(path)
        return path
    reports={name:json.loads(track(ROOT/'results'/path).read_text()) for name,path in SOURCES.items()}
    validate_reports(reports)
    # Verify the cohort boundary explicitly; repeated folds are not new subjects.
    development=read(track(ROOT/'results/aomic_followup_v1/ready/approved_cohort.csv'))
    test=read(track(ROOT/'results/aomic_final_test_v1/final/predictions.csv'))
    allocated=read(track(ROOT/'data/manifests/aomic_downstream_splits_v1.csv'))
    dev_ids={r['source_participant_id'] for r in development}; test_ids={r['source_participant_id'] for r in test}
    if len(development)!=90 or len(dev_ids)!=90 or len(test)!=123 or len(test_ids)!=123 or dev_ids & test_ids:
        raise ValueError('Cohort uniqueness/separation failed')
    if not dev_ids<={r['source_participant_id'] for r in allocated if r['split']=='train'}:
        raise ValueError('Development subjects outside original training allocation')
    if not test_ids<={r['source_participant_id'] for r in allocated if r['split']=='test'}:
        raise ValueError('Pilot test identity mismatch')
    # Check pooled paired predictions reproduce scale and architecture summaries.
    paired_sources={}
    for key,path,metrics in (
        ('vit_scale','vit_pretraining_scale_v1/out_of_fold_predictions.csv',('random','pretrained22','pretrained60','pretrained120','training_mean')),
        ('scale','cnn_pretraining_scale_v1/out_of_fold_predictions.csv',('random','pretrained22','pretrained60','pretrained120','training_mean')),
        ('vit_ssl','aomic_pretrained_vit_v1/out_of_fold_predictions.csv',('pretrained_vit','random_vit','training_mean'))):
        rows=read(track(ROOT/'results'/path))
        paired_sources[key]={(int(r['seed']),r['source_participant_id']):r for r in rows}
        if len(rows)!=270: raise ValueError('Expected 90 subjects x 3 seeds, not 270 independent people')
        for seed in (42,43,44):
            part=[r for r in rows if int(r['seed'])==seed]
            if len(part)!=90 or {r['source_participant_id'] for r in part}!=dev_ids: raise ValueError('Incomplete paired predictions')
            saved=next(r for r in reports[key]['per_seed'] if r['seed']==seed)
            labels={r['source_participant_id']:float(r['IST_fluid']) for r in development}
            for row in part: close(row['observed'],labels[row['source_participant_id']],'observed target')
            for metric in metrics:
                close(sum(abs(float(r[metric])-float(r['observed'])) for r in part)/90,saved[metric],f'{key}/{seed}/{metric}')
    for identity,row in paired_sources['vit_scale'].items():
        original=paired_sources['vit_ssl'][identity]
        cnn=paired_sources['scale'][identity]
        for reference in (original,cnn):
            if int(row['fold'])!=int(reference['fold']): raise ValueError('Outer fold differs across architectures')
            close(row['training_mean'],reference['training_mean'],'paired baseline')
        for new,old in (('random','random_vit'),('pretrained22','pretrained_vit')):
            close(row[new],original[old],'paired ViT reference prediction')
    table=[]
    def add(study,evaluation,fit,inner,ssl,method,mae):
        table.append(dict(study=study,evaluation_cohort=evaluation,fitting_subjects_per_fold=fit,
            inner_validation_subjects=inner,ssl_subjects=ssl,method=method,mean_mae=mae,units='IST_fluid points'))
    p=reports['pilot']
    add('pilot','123 original test subjects',30,12,'not_applicable','training mean',p['primary_mae'])
    for kind in ('random_encoder','pretrained'):
        add('pilot','123 original test subjects',30,12,22 if kind=='pretrained' else 0,kind,p['seed_summary'][kind]['mean_mae'])
    for r in reports['labels']['mean_over_seeds']:
        add('fixed50_label_curve','90 development OOF subjects',r['size'],0,22 if r['kind']=='pretrained' else 0,r['kind'],r['mae'])
    for n in (15,30,60):
        values=[r['baseline_mae'] for r in reports['labels']['per_seed'] if r['size']==n]
        if len(values)!=6: raise ValueError('Incomplete label-budget baseline')
        for value in values: close(value,values[0],'label-budget baseline')
        add('fixed50_label_curve','90 development OOF subjects',n,0,0,'training mean',values[0])
    for r in reports['selection']['mean_over_seeds']:
        for policy in ('early','fixed50'):
            add('matched_checkpoint_selection','90 development OOF subjects',48,12,22 if r['kind']=='pretrained' else 0,r['kind']+'_'+policy,r[policy])
    scale=reports['scale']['mean_over_seeds']; vit=reports['vit_ssl']['mean_over_seeds']
    common=[('Random CNN',0,scale['random']),('CNN SSL 22',22,scale['pretrained22']),
            ('CNN SSL 60',60,scale['pretrained60']),('CNN SSL 120',120,scale['pretrained120']),
            ('Random ViT',0,vit['random_vit']),('ViT SSL 22',22,vit['pretrained_vit']),
            ('ViT SSL 60',60,reports['vit_scale']['mean_over_seeds']['pretrained60']),
            ('ViT SSL 120',120,reports['vit_scale']['mean_over_seeds']['pretrained120']),('Training mean',0,scale['training_mean'])]
    for name,n,value in common: add('common_inner_selection','90 development OOF subjects',48,12,n,name,value)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.consolidation-',dir=output.parent))
    try:
        def save(fig,name):
            fig.tight_layout()
            for extension in ('png','pdf'): fig.savefig(temporary/f'{name}.{extension}',dpi=200)
            plt.close(fig)
        with (temporary/'results_table.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        fig,ax=plt.subplots(figsize=(8,6))
        ax.barh([r[0] for r in common],[r[2] for r in common],color=['tab:blue']*4+['tab:orange']*4+['gray'])
        ax.invert_yaxis(); ax.set(xlabel='Mean MAE (IST_fluid points); lower is better',title='Development only: common 48 / 12 / 30 design')
        for i,r in enumerate(common): ax.text(r[2]+.1,i,f'{r[2]:.3f}',va='center')
        ax.set_xlim(0,24); save(fig,'common_design_comparison')
        fig,axes=plt.subplots(1,3,figsize=(16,4),sharey=True)
        for axis,key,title in ((axes[0],'scale','CNN'),(axes[1],'vit_scale','ViT')):
            means=reports[key]['mean_over_seeds']
            for row in reports[key]['per_seed']:
                axis.plot((22,60,120),[row[f'pretrained{n}'] for n in (22,60,120)],alpha=.4,marker='.',label=f"Seed {row['seed']}")
            axis.plot((22,60,120),[means[f'pretrained{n}'] for n in (22,60,120)],color='black',marker='o',label='Mean')
            axis.axhline(means['random'],linestyle='--',label=f'Random {title} mean')
            axis.axhline(means['training_mean'],linestyle=':',color='gray',label='Training mean')
            axis.set(xlabel='SSL subjects (20 epochs; updates increase)',ylabel='OOF MAE',title=f'{title} pretraining scale',xticks=(22,60,120))
            axis.legend(fontsize=7)
        for i,kind in enumerate(('random','pretrained')):
            for r in reports['selection']['per_seed']:
                if r['kind']==kind: axes[2].plot((i-.15,i+.15),(r['fixed50'],r['early']),marker='o',alpha=.6)
        axes[2].set_xticks((-.15,.15,.85,1.15),('Random\nfixed50','Random\nselected','SSL22\nfixed50','SSL22\nselected'))
        axes[2].set(ylabel='OOF MAE',title='Matched checkpoint policies; lines pair seeds')
        save(fig,'scaling_and_selection')
        fig,ax=plt.subplots(figsize=(7,4))
        for kind,color in (('random','tab:blue'),('pretrained','tab:orange')):
            for seed in (42,43,44):
                values=[next(r['model_mae'] for r in reports['labels']['per_seed']
                             if (r['size'],r['seed'],r['kind'])==(n,seed,kind)) for n in (15,30,60)]
                ax.plot((15,30,60),values,color=color,alpha=.3,marker='.')
            values=[next(r['mae'] for r in reports['labels']['mean_over_seeds']
                         if (r['size'],r['kind'])==(n,kind)) for n in (15,30,60)]
            ax.plot((15,30,60),values,color=color,marker='o',label=kind+' CNN')
        ax.plot((15,30,60),[next(r['baseline_mae'] for r in reports['labels']['per_seed'] if r['size']==n)
                           for n in (15,30,60)],'--',color='gray',label='Training mean')
        ax.set(xlabel='Fitting subjects per fold',ylabel='OOF MAE',title='Separate fixed-50-epoch labeled-data experiment',xticks=(15,30,60))
        ax.legend(); save(fig,'labeled_data_curve')
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(('Training mean','Random encoder','Pretrained encoder'),(p['primary_mae'],p['seed_summary']['random_encoder']['mean_mae'],p['seed_summary']['pretrained']['mean_mae']))
        ax.set(ylabel='Test MAE (IST_fluid points)',title='Separate original pilot test: n=123'); save(fig,'original_pilot_test')
        report=ROOT/'results/research_consolidation_v1/research_report.md'
        if report.exists(): shutil.copyfile(track(report),temporary/'research_report.md')
        if any(digest(ROOT/name)!=value for name,value in hashes.items()): raise ValueError('Source changed during consolidation')
        audit=dict(source_sha256=hashes,script_sha256=digest(Path(__file__)),
            checks=['Reported seed means reconciled','Repeated reference means reconciled','123-test/90-development identities disjoint and in original allocations','CNN and ViT scaling paired predictions reproduce reported MAEs'],
            limits='This is a saved-output consistency audit, not a full training rerun, raw-data audit, or statistical significance test.',
            evaluation_note='Do not pool pilot test and development scores. Development model selection is not independent confirmation.')
        (temporary/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True); raise
    print('Saved:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/research_consolidation_v1/package')
    args=parser.parse_args(); consolidate(args.output.resolve())
