"""Synthetic summary checks; no MRI data, model loading, or training."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_pilot_results import paired_bootstrap, validate_predictions


class SummaryTests(unittest.TestCase):
    def test_constant_paired_advantage(self):
        estimate, interval = paired_bootstrap(np.tile([-2., 0., 2.], (8, 1)), 1000)
        np.testing.assert_array_equal(estimate, [-2., 0., 2.])
        np.testing.assert_array_equal(interval, [[-2., 0., 2.], [-2., 0., 2.]])

    def test_reproducible_and_paired(self):
        differences = np.column_stack((np.arange(12.), -np.arange(12.)))
        first = paired_bootstrap(differences, 1000, 7)
        second = paired_bootstrap(differences, 1000, 7)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(first[1][0, 0], -first[1][1, 1])
        self.assertAlmostEqual(first[1][1, 0], -first[1][0, 1])

    def test_prediction_integrity(self):
        rows = [dict(participant_id='a', source_participant_id='x', observed='0', model='1'),
                dict(participant_id='b', source_participant_id='y', observed='4', model='1')]
        report = dict(accepted=2, mae=dict(model=2.))
        validate_predictions(rows, report, ['model'])
        with self.assertRaises(ValueError):
            validate_predictions(rows, dict(accepted=2, mae=dict(model=9)), ['model'])
        with self.assertRaises(ValueError):
            validate_predictions([rows[0], rows[0]], report, ['model'])
        rows[1]['model'] = 'nan'
        with self.assertRaises(ValueError):
            validate_predictions(rows, report, ['model'])

    def test_bad_bootstrap_inputs(self):
        for values in ([[1.]], [[1.], [float('nan')]]):
            with self.assertRaises(ValueError):
                paired_bootstrap(values)
        with self.assertRaises(ValueError):
            paired_bootstrap([[1.], [2.]], 10)


if __name__ == '__main__':
    unittest.main()
