"""Synthetic protocol checks. No downloads, MRI processing, or scoring runs."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import final_aomic_test as final
from evaluate_aomic_validation import accepted_rows


class FinalProtocolTests(unittest.TestCase):
    def test_full_checklist_required(self):
        selected = [dict(hf_path='a',participant_id='a'),dict(hf_path='b',participant_id='b')]
        reviewed = [dict(hf_path='a',status='pass',decision='accept',metadata_sha256='hash',notes='')]
        with self.assertRaises(ValueError):
            accepted_rows(selected,reviewed)
        reviewed.append(dict(hf_path='b',status='not_attempted',decision='exclude',notes='skip'))
        with self.assertRaises(ValueError):
            accepted_rows(selected,reviewed)
        reviewed[1].update(status='failed',notes='download failed after retries')
        self.assertEqual(len(accepted_rows(selected,reviewed)),1)
        reviewed[1].update(decision='accept',metadata_sha256='hash')
        with self.assertRaises(ValueError):
            accepted_rows(selected,reviewed)

    def test_preparation_limit_persistence_and_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'results'; output.mkdir()
            selected=[dict(hf_path=f'A/sub_{i}/t1.nii.gz',participant_id=f'sub_{i}',dataset='A') for i in range(3)]
            for row in selected:
                path=root/'data/raw'/row['hf_path']
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'synthetic invalid scan')
            plan=dict(preprocessing=dict(synthstrip_image_id='test',pipeline_sha256='code'))
            with patch.object(final,'ROOT',root),patch.object(final,'OUT',output), \
                 patch.object(final.brain,'image_id',return_value='test'), \
                 patch.object(final,'pipeline_hash',return_value='code'), \
                 patch.object(final.brain,'process_scan',side_effect=RuntimeError('synthetic failure')) as process:
                final.prepare(plan,selected,1,False)
                records=final.read(output/'review.csv')
                self.assertEqual([r['status'] for r in records],['failed','not_attempted','not_attempted'])
                final.prepare(plan,selected,1,False)
                records=final.read(output/'review.csv')
                self.assertEqual([r['status'] for r in records],['failed','failed','not_attempted'])
                final.prepare(plan,selected,1,True)
                self.assertEqual(process.call_args.args[0]['hf_path'],selected[0]['hf_path'])

    def test_lock_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with final.lock(path/'lock'):
                with self.assertRaises(RuntimeError):
                    with final.lock(path/'lock'):
                        pass
            self.assertFalse((path/'lock').exists())
            final.atomic_write(path/'progress.csv',b'first')
            final.atomic_write(path/'progress.csv',b'second')
            self.assertEqual((path/'progress.csv').read_bytes(),b'second')


if __name__=='__main__':
    unittest.main()
