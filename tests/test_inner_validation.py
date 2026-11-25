"""Synthetic inner split and stopping-policy checks; no MRI or training runs."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_inner_validation import split_inner, update_selection


class InnerSelectionTests(unittest.TestCase):
    def test_score_independent_stable_partition(self):
        rows = [dict(source_participant_id=f'sub-{i}', IST_fluid=i) for i in range(60)]
        fit, inner = split_inner(rows, 0)
        self.assertEqual((len(fit), len(inner)), (48, 12))
        self.assertEqual((fit, inner), split_inner(list(reversed(rows)), 0))
        self.assertFalse({r['source_participant_id'] for r in fit} & {r['source_participant_id'] for r in inner})
        changed = [dict(r, IST_fluid=999) for r in rows]
        fit2, inner2 = split_inner(changed, 0)
        for a, b in ((fit, fit2), (inner, inner2)):
            self.assertEqual([r['source_participant_id'] for r in a], [r['source_participant_id'] for r in b])
        with self.assertRaises(ValueError):
            split_inner(rows[:-1] + [rows[0]], 0)

    def test_stop_freezes_checkpoint_and_ties_keep_earlier(self):
        state = dict(best_mae=None, best_epoch=None, best_weights=None, bad_epochs=0, stop_epoch=None)
        weights = {'weight': torch.tensor([1.])}
        update_selection(state, 10., 1, weights, patience=2)
        weights['weight'].fill_(9.)
        self.assertEqual(state['best_weights']['weight'].item(), 1.)
        update_selection(state, 9.95, 2, weights, patience=2)
        update_selection(state, 10., 3, weights, patience=2)
        self.assertEqual(state['stop_epoch'], 3)
        update_selection(state, 1., 4, weights, patience=2)
        self.assertEqual(state['best_epoch'], 1)

    def test_improvement_resets_patience_and_rejects_nan(self):
        state = dict(best_mae=None, best_epoch=None, best_weights=None, bad_epochs=0, stop_epoch=None)
        for epoch, value in enumerate((10., 10., 9., 9.), 1):
            update_selection(state, value, epoch, {}, patience=2)
        self.assertEqual(state['best_epoch'], 3)
        self.assertEqual(state['bad_epochs'], 1)
        self.assertIsNone(state['stop_epoch'])
        with self.assertRaises(ValueError):
            update_selection(state, float('nan'), 5, {})


if __name__ == '__main__':
    unittest.main()
