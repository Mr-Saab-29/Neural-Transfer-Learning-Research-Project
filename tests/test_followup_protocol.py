"""Synthetic selection and leakage checks; no source MRI files required."""
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_followup_protocol import build_files, publish


def subjects():
    return [dict(dataset='PT013_AIOMIC1000', participant_id=f'sub_{i}',
                 source_participant_id=f'sub-{i:04}', hf_path=f'A/sub_{i}/ses_1/t1.nii.gz',
                 IST_fluid=str(i + 1), split='train' if i < 100 else 'test' if i < 110 else 'val',
                 pretraining_allowed='False') for i in range(120)]


def ids(data):
    return {r['source_participant_id'] for r in csv.DictReader(io.StringIO(data.decode()))}


class ProtocolTests(unittest.TestCase):
    def test_nested_disjoint_complete_and_stable(self):
        rows = subjects()
        files = build_files(rows, 'source')
        self.assertEqual(files, build_files(list(reversed(rows)), 'source'))
        cohort = ids(files['cohort.csv'])
        self.assertEqual(len(cohort), 90)
        self.assertTrue(cohort <= {r['source_participant_id'] for r in rows if r['split'] == 'train'})
        evaluated = set()
        for fold in range(3):
            evaluation = ids(files[f'fold_{fold}_evaluation.csv'])
            self.assertEqual(len(evaluation), 30)
            self.assertFalse(evaluated & evaluation)
            evaluated |= evaluation
            previous = set()
            for size in (15, 30, 60):
                training = ids(files[f'fold_{fold}_train_{size}.csv'])
                self.assertEqual(len(training), size)
                self.assertTrue(previous <= training)
                self.assertFalse(training & evaluation)
                previous = training
            self.assertEqual(previous | evaluation, cohort)
        self.assertEqual(evaluated, cohort)
        changed = [dict(r, IST_fluid='999') for r in rows]
        other = build_files(changed, 'other')
        for name in files:
            if name.endswith('.csv'):
                self.assertEqual(ids(files[name]), ids(other[name]))

    def test_invalid_subjects(self):
        rows = subjects()
        with self.assertRaises(ValueError):
            build_files(rows + [rows[0]], 'x')
        with self.assertRaises(ValueError):
            build_files(rows[:89], 'x')
        rows[0]['IST_fluid'] = 'nan'
        with self.assertRaises(ValueError):
            build_files(rows, 'x')

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'frozen'
            files = build_files(subjects(), 'x')
            publish(files, output)
            publish(files, output)
            (output / 'cohort.csv').write_text('changed')
            with self.assertRaises(ValueError):
                publish(files, output)
            self.assertEqual((output / 'cohort.csv').read_text(), 'changed')


if __name__ == '__main__':
    unittest.main()
