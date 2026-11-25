"""Synthetic cohort and approval-reuse checks; no downloads or MRI processing."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_pretraining_scale as scale


def fixtures():
    pool=[dict(dataset=source,participant_id=f'{source}_{i}',hf_path=f'{source}/{source}_{i}/ses_1/t1.nii.gz',
               baseline_priority='True',session_id='ses_1') for source in scale.SOURCES for i in range(100)]
    base=[r for r in pool if int(r['participant_id'].rsplit('_',1)[1])<11]
    heldout=[r for r in pool if int(r['participant_id'].rsplit('_',1)[1])==11]
    return pool,base,heldout


class ScaleTests(unittest.TestCase):
    def test_nested_balanced_deterministic_and_heldout_safe(self):
        pool,base,heldout=fixtures()
        cohorts=scale.select_nested(pool,base,heldout)
        other=scale.select_nested(list(reversed(pool)),base,heldout)
        self.assertEqual(cohorts,other)
        previous=set()
        for size in (22,60,120):
            rows=cohorts[size]
            paths={r['hf_path'] for r in rows}
            self.assertEqual(len(paths),size)
            self.assertTrue(previous<=paths)
            self.assertFalse(paths & {r['hf_path'] for r in heldout})
            for source in scale.SOURCES:
                self.assertEqual(sum(r['dataset']==source for r in rows),size//2)
            previous=paths
        self.assertEqual({r['hf_path'] for r in cohorts[22]},{r['hf_path'] for r in base})

    def test_multiple_sessions_do_not_duplicate_subject(self):
        pool,base,heldout=fixtures()
        duplicate=dict(pool[50],session_id='ses_2',baseline_priority='False',hf_path=pool[50]['hf_path'].replace('ses_1','ses_2'))
        self.assertEqual(scale.select_nested(pool,base,heldout),scale.select_nested(pool+[duplicate],base,heldout))

    def test_wrong_sources_or_original_cohort_rejected(self):
        pool,base,heldout=fixtures()
        with self.assertRaises(ValueError): scale.select_nested(pool,base[:-1],heldout)
        with self.assertRaises(ValueError): scale.select_nested(pool+[dict(pool[0],dataset='AOMIC')],base,heldout)
        with self.assertRaises(ValueError): scale.select_nested(pool,base,[base[0]])

    def test_stale_approval_is_not_carried_forward(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); stage=root/'size60'; stage.mkdir()
            folder=root/'scan'; folder.mkdir()
            (folder/'metadata.json').write_text(json.dumps(dict(qc={})))
            selected=[dict(dataset='PT005_IXI',participant_id='subject',hf_path='scan')]
            prior=dict(selected[0],decision='accept',metadata_sha256='old')
            original_read=scale.read
            def fake_read(path):
                if path==scale.BASE: return [prior]
                return original_read(path)
            with patch.object(scale,'ROOT',root),patch.object(scale,'BASE',root/'base.csv'), \
                 patch.object(scale,'read',side_effect=fake_read),patch.object(scale,'folder_for',return_value=folder), \
                 patch.object(scale.brain,'review_reasons',return_value=[]), \
                 patch.object(scale,'approved_artifact',side_effect=ValueError('stale')):
                scale.import_approvals(dict(preprocessing={}),selected,stage)
            record=original_read(stage/'review.csv')[0]
            self.assertEqual(record['decision'],'pending')
            self.assertEqual(record['status'],'not_attempted')


if __name__=='__main__': unittest.main()
