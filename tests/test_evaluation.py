"""
Unit tests for business_entity_resolution/src/evaluation.py.
"""

import unittest
import numpy as np
import pandas as pd
from business_entity_resolution.src.evaluation import (
    f05,
    calculate_f05,
    evaluate_single_entity,
    evaluate_predictions,
    evaluate_predictions_detailed,
    find_optimal_threshold,
    apply_decision_rule,
)


class TestEvaluation(unittest.TestCase):
    def test_f05_mathematical_formula(self):
        """Test F0.5 calculation against known analytical values."""
        # F0.5 = 1.25 * P * R / (0.25 * P + R)
        # When P=1.0, R=1.0 -> 1.0
        self.assertAlmostEqual(f05(1.0, 1.0), 1.0)
        # When P=0.0 or R=0.0 -> 0.0
        self.assertEqual(f05(0.0, 1.0), 0.0)
        self.assertEqual(f05(1.0, 0.0), 0.0)
        self.assertEqual(f05(0.0, 0.0), 0.0)

        # Official README example: P = 2/3, R = 1.0 -> F0.5 = 5/7 = 0.7142857
        p = 2.0 / 3.0
        r = 1.0
        expected = (1.25 * p * r) / (0.25 * p + r)
        self.assertAlmostEqual(f05(p, r), expected)
        self.assertAlmostEqual(calculate_f05(p, r), 5.0 / 7.0, places=5)

    def test_singleton_scoring_rules(self):
        """
        Verify singleton handling per competition rules:
        - True empty, Pred empty -> 1.0 (correct singleton)
        - True empty, Pred non-empty -> 0.0 (false merge)
        - True non-empty, Pred empty -> 0.0 (missed match)
        """
        # True singleton, predicted empty
        score, p, r = evaluate_single_entity(set(), set())
        self.assertEqual(score, 1.0)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)

        # True singleton, predicted false merge
        score, p, r = evaluate_single_entity(set(), {"S2-99"})
        self.assertEqual(score, 0.0)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)

        # True matches, predicted empty
        score, p, r = evaluate_single_entity({"S2-1"}, set())
        self.assertEqual(score, 0.0)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)

    def test_macro_averaging(self):
        """Verify macro averaging across diverse entities."""
        # 3 entities:
        # S1: P=2/3, R=1.0 -> F0.5 = 5/7 = 0.7142857
        # S2: Singleton, predicted empty -> F0.5 = 1.0
        # S3: True {A}, Pred {B} -> F0.5 = 0.0
        gt = {
            "S1-1": {"S2-1", "S3-1"},
            "S1-2": set(),
            "S1-3": {"S2-2"},
        }
        pred = {
            "S1-1": {"S2-1", "S2-99", "S3-1"},
            "S1-2": set(),
            "S1-3": {"S2-999"},
        }

        expected_macro = ((5.0 / 7.0) + 1.0 + 0.0) / 3.0
        res = evaluate_predictions_detailed(gt, pred)
        self.assertAlmostEqual(res["macro_f05"], expected_macro, places=5)
        self.assertEqual(res["singletons"], 1)
        self.assertEqual(res["singleton_accuracy"], 1.0)
        self.assertAlmostEqual(evaluate_predictions(gt, pred), expected_macro, places=5)

    def test_find_optimal_threshold(self):
        """Verify that threshold sweeping finds the optimal operating point."""
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
        }
        s1_ids = ["S1-1", "S1-1", "S1-2"]
        cand_ids = ["S2-1", "S2-distractor", "S2-noise"]
        probs = [0.85, 0.60, 0.40]

        # If threshold is 0.70:
        # S1-1 predicts {S2-1} -> F0.5 = 1.0
        # S1-2 predicts set() -> F0.5 = 1.0
        # Macro F0.5 = 1.0!
        res = find_optimal_threshold(
            s1_ids, cand_ids, probs, gt, threshold_range=(0.30, 0.90, 0.05)
        )
        self.assertAlmostEqual(res["best_f05"], 1.0)
        self.assertGreaterEqual(res["best_threshold"], 0.61)
        self.assertLessEqual(res["best_threshold"], 0.85)

    def test_apply_decision_rule(self):
        """Verify apply_decision_rule produces exact submission columns and handles singletons."""
        scored_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "score": 0.85},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-20", "score": 0.92},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-30", "score": 0.35},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-40", "score": 0.20},
        ])
        all_s1 = ["S1-1", "S1-2", "S1-3"]  # S1-3 has no candidates at all

        df_out = apply_decision_rule(scored_df, all_s1, threshold=0.70)
        self.assertEqual(list(df_out.columns), ["source1_entity_id", "matched_entity_ids"])
        self.assertEqual(len(df_out), 3)

        out_map = dict(zip(df_out["source1_entity_id"], df_out["matched_entity_ids"]))
        # S1-1 has S2-20 (0.92) and S2-10 (0.85) in descending order
        self.assertEqual(out_map["S1-1"], "S2-20,S2-10")
        # S1-2 had score 0.20 < 0.70 -> empty
        self.assertEqual(out_map["S1-2"], "")
        # S1-3 had zero candidates -> empty
        self.assertEqual(out_map["S1-3"], "")


if __name__ == "__main__":
    unittest.main()
