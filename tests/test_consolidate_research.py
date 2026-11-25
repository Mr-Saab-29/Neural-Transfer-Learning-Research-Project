"""Synthetic consistency checks for consolidation; no MRI or training."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from consolidate_research import close,validate_reports


def reports():
    def summary(values):
        return dict(units='IST_fluid points',mean_over_seeds=values,
                    per_seed=[dict(seed=s,subjects=90,**values) for s in (42,43,44)])
    output=dict(scale=summary(dict(random=1.,pretrained22=2.,pretrained60=3.,pretrained120=4.,training_mean=5.)),
                vit=summary(dict(cnn=1.,vit=6.,training_mean=5.)),
                vit_ssl=summary(dict(random_vit=6.,pretrained_vit=7.,training_mean=5.)))
    output['vit_scale']=summary(dict(random=6.,pretrained22=7.,pretrained60=8.,pretrained120=9.,training_mean=5.))
    for name in ('scale','vit_scale'):
        for row in [output[name]['mean_over_seeds'],*output[name]['per_seed']]:
            for larger,smaller in ((60,22),(120,22),(120,60)):
                row[f'scale{larger}_minus{smaller}']=row[f'pretrained{larger}']-row[f'pretrained{smaller}']
    selected=[dict(kind=k,early=v,fixed50=v+1,training_mean=5.,early_minus_fixed50=-1.) for k,v in (('random',1.),('pretrained',2.))]
    output['selection']=dict(units='IST_fluid points',mean_over_seeds=selected,
                            per_seed=[dict(r,seed=s,subjects=90) for r in selected for s in (42,43,44)])
    output['labels']=dict(units='IST_fluid points',mean_over_seeds=[dict(size=15,kind='random',mae=1.)],
                         per_seed=[dict(size=15,kind='random',model_mae=1.,seed=s) for s in (42,43,44)])
    output['pilot']=dict(units='IST_fluid points',selected=124,accepted=123,excluded=1,
        seed_summary={k:dict(mean_mae=1.) for k in ('random_encoder','pretrained')},
        mae={f'{k}_{s}':1. for k in ('random_encoder','pretrained') for s in (42,43,44)})
    return output


class ConsolidationTests(unittest.TestCase):
    def test_consistent_reports(self): validate_reports(reports())

    def test_wrong_seed_average_rejected(self):
        data=reports(); data['scale']['mean_over_seeds']['random']=2.
        with self.assertRaises(ValueError): validate_reports(data)

    def test_consistent_but_changed_reference_rejected(self):
        data=reports()
        data['vit']['mean_over_seeds']['cnn']=9.
        for row in data['vit']['per_seed']: row['cnn']=9.
        with self.assertRaises(ValueError): validate_reports(data)

    def test_duplicate_seeds_rejected(self):
        data=reports(); data['vit']['per_seed'][0]['seed']=43
        with self.assertRaises(ValueError): validate_reports(data)

    def test_vit_scale_reference_change_rejected(self):
        data=reports()
        data['vit_scale']['per_seed'][0]['random']+=1
        data['vit_scale']['per_seed'][1]['random']-=1
        with self.assertRaises(ValueError): validate_reports(data)

    def test_incorrect_scale_contrast_rejected(self):
        data=reports()
        data['vit_scale']['mean_over_seeds']['scale120_minus22']=99.
        for row in data['vit_scale']['per_seed']: row['scale120_minus22']=99.
        with self.assertRaises(ValueError): validate_reports(data)

    def test_nonfinite_metrics_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError): close(value,1.,'metric')


if __name__=='__main__': unittest.main()
