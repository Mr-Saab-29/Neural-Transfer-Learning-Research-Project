"""Synthetic numerical checks; no MRI data or training."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_followup_results import sensitivity, spread


class AnalysisTests(unittest.TestCase):
    def test_constant_and_paired_ranges(self):
        folds = np.repeat(np.arange(3), 30)
        values = np.column_stack((np.arange(90.), -np.arange(90.), np.ones(90) * 2))
        first = sensitivity(values, folds, 1000, 42)
        np.testing.assert_array_equal(first, sensitivity(values, folds, 1000, 42))
        np.testing.assert_allclose(first[:, 2], [2, 2])
        np.testing.assert_allclose(first[:, 0], -first[::-1, 1])

    def test_fold_stratification_preserves_constant_fold_means(self):
        folds = np.repeat(np.arange(3), 30)
        values = (folds * 10.).reshape(-1, 1)
        np.testing.assert_allclose(sensitivity(values, folds, 1000), [[10], [10]])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            sensitivity(np.ones((89, 1)), np.repeat(np.arange(3), 30))
        with self.assertRaises(ValueError):
            sensitivity(np.full((90, 1), np.nan), np.repeat(np.arange(3), 30))
        with self.assertRaises(ValueError):
            sensitivity(np.ones((90, 1)), np.zeros(90))

    def test_constant_predictions_are_not_perfect_correlation(self):
        result = spread(np.arange(30.), np.ones(30))
        self.assertEqual(result['sd_ratio'], 0)
        self.assertIsNone(result['pearson_r'])
        result = spread(np.arange(30.), np.arange(30.))
        self.assertAlmostEqual(result['pearson_r'], 1.)
        self.assertEqual(result['sd_ratio'], 1.)
        self.assertEqual(result['bias'], 0.)


if __name__ == '__main__':
    unittest.main()
