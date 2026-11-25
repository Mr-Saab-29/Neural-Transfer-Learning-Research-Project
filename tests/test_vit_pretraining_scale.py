"""Synthetic summary checks; no MRI loading or training required."""
from contextlib import ExitStack
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_vit_pretraining_scale as scale


class ViTScaleTests(unittest.TestCase):
    def summarize(self, fault=None):
        def rows(path):
            path=Path(path)
            if path.name.endswith('_outer.csv'):
                fold=int(path.name[4])
            else:
                fold=int(path.parent.name[4])
            output=[]
            for i in range(30):
                row=dict(source_participant_id=str(fold*30+i),observed='10',training_mean='12')
                if 'aomic_tiny_vit_v1' in str(path): row['vit']='13'
                elif 'aomic_pretrained_vit_v1' in str(path): row['pretrained_vit']='14'
                else: row['pretrained_vit']='11' if 'size60' in str(path) else '10.5'
                output.append(row)
            if fault=='duplicate' and 'size120' in str(path) and path.name=='predictions.csv':
                output[-1]=dict(output[0])
            if fault=='target' and 'size120' in str(path) and path.name=='predictions.csv':
                output[0]['observed']='99'
            if fault=='nonfinite' and 'size120' in str(path) and path.name=='predictions.csv':
                output[0]['pretrained_vit']='nan'
            return output
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            out=Path(directory)
            stack.enter_context(patch.object(scale,'EXPERIMENT',out))
            stack.enter_context(patch.object(scale,'OUT',out/'size60'))
            stack.enter_context(patch.object(scale,'read',side_effect=rows))
            stack.enter_context(patch.object(scale,'load_plan',return_value=(dict(
                supervised=dict(seeds=[42,43,44]),parent_plan_sha256='random',limitations='synthetic'),'plan')))
            stack.enter_context(patch.object(scale,'backbone',return_value=({},'checkpoint')))
            stack.enter_context(patch.object(scale.original_vit,'load_plan',return_value=({},'original')))
            stack.enter_context(patch.object(scale.original_vit,'backbone',return_value=({},'old-checkpoint')))
            stack.enter_context(patch.object(scale,'validate_run'))
            stack.enter_context(patch('builtins.print'))
            scale.summarize()
            return json.loads((out/'comparison.json').read_text())

    def test_paired_means_and_scale_contrasts(self):
        report=self.summarize()
        means=report['mean_over_seeds']
        self.assertEqual(means['random'],3)
        self.assertEqual(means['pretrained22'],4)
        self.assertEqual(means['pretrained60'],1)
        self.assertEqual(means['pretrained120'],.5)
        self.assertEqual(means['scale120_minus22'],-3.5)
        self.assertEqual(means['scale120_minus60'],-.5)
        self.assertEqual([r['subjects'] for r in report['per_seed']],[90]*3)

    def test_rejects_duplicate_targets_and_nonfinite_predictions(self):
        for fault in ('duplicate','target','nonfinite'):
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.summarize(fault)


if __name__=='__main__': unittest.main()
