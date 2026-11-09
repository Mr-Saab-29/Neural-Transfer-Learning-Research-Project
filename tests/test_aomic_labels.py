from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_aomic_labels import join_labels, source_id, SCORES


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(dataset='PT013_AIOMIC1000', participant_id='sub_9999',
                        session_id='ses_1', hf_path='example',
                        original_path='sub-0031/anat/sub-0031_run-3_T1w.nii.gz')
        self.source = dict(participant_id='sub-0031', age='25', sex='female',
                           **{key: '85' for key in SCORES})

    def test_original_id_not_fomo_id(self):
        result = join_labels([self.row], [self.source], {'example'})[0]
        self.assertEqual(result['source_participant_id'], 'sub-0031')
        self.assertEqual(result['IST_fluid'], 85.)
        self.assertTrue(result['in_pilot'])

    def test_missing_and_unmatched(self):
        self.source['IST_fluid'] = 'n/a'
        result = join_labels([self.row], [self.source], set())[0]
        self.assertEqual(result['label_status'], 'missing_target')
        self.assertEqual(result['IST_fluid'], '')
        self.assertEqual(join_labels([self.row], [], set())[0]['label_status'], 'unmatched')

    def test_duplicate_and_wrong_path(self):
        with self.assertRaises(ValueError):
            join_labels([self.row], [self.source, self.source], set())
        with self.assertRaises(ValueError):
            join_labels([self.row, self.row], [self.source], set())
        with self.assertRaises(ValueError):
            source_id('sub-0031/anat/sub-0032_run-3_T1w.nii.gz')


if __name__ == '__main__':
    unittest.main()
