"""
Unit tests for business_entity_resolution/src/model.py.
"""

import os
import sys
import tempfile
import unittest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import numpy as np
from business_entity_resolution.src.model import (
    EntityMatcher,
    train_model,
    predict_scores,
    decide_matches,
)


class TestModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 400
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        # Synthetic binary relationship
        self.y = ((self.X[:, 0] + self.X[:, 1]) > 0.3).astype(np.int32)
        self.X_val = np.random.randn(100, self.n_features).astype(np.float32)
        self.y_val = ((self.X_val[:, 0] + self.X_val[:, 1]) > 0.3).astype(np.int32)

    def test_fit_and_predict_lightgbm(self):
        """Test EntityMatcher with LightGBM backend."""
        matcher = EntityMatcher(algorithm="lightgbm", n_estimators=25)
        matcher.fit(self.X, self.y, X_val=self.X_val, y_val=self.y_val)
        probs = matcher.predict_proba(self.X_val)
        self.assertEqual(len(probs), 100)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        self.assertIsNotNone(matcher.feature_importances_)
        self.assertEqual(len(matcher.feature_importances_), self.n_features)

    def test_fit_and_predict_hist_gb_fallback(self):
        """Test EntityMatcher with HistGradientBoosting fallback backend."""
        matcher = EntityMatcher(algorithm="hist_gb", n_estimators=25)
        matcher.fit(self.X, self.y)
        probs = matcher.predict_proba(self.X_val)
        self.assertEqual(len(probs), 100)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_predict_in_batches(self):
        """Test predict_in_batches returns identical results to predict_proba."""
        matcher = EntityMatcher(algorithm="auto", n_estimators=20)
        matcher.fit(self.X, self.y)
        single_pass = matcher.predict_proba(self.X)
        batched = matcher.predict_in_batches(self.X, batch_size=50)
        np.testing.assert_allclose(single_pass, batched, atol=1e-6)

    def test_decide_matches(self):
        """Test decide_matches threshold and top-k logic."""
        cand_ids = ["S2-1", "S2-2", "S3-1", "S3-2"]
        scores = [0.85, 0.45, 0.92, 0.71]

        # Threshold 0.70 without max_matches -> S3-1 (0.92), S2-1 (0.85), S3-2 (0.71)
        res = decide_matches(cand_ids, scores, threshold=0.70)
        self.assertEqual(res, ["S3-1", "S2-1", "S3-2"])

        # Threshold 0.70 with max_matches=2 -> Top 2: S3-1, S2-1
        res_top2 = decide_matches(cand_ids, scores, threshold=0.70, max_matches=2)
        self.assertEqual(res_top2, ["S3-1", "S2-1"])

        # Threshold 0.95 -> Empty (singleton / no match)
        res_none = decide_matches(cand_ids, scores, threshold=0.95)
        self.assertEqual(res_none, [])

        # Empty candidates
        self.assertEqual(decide_matches([], [], threshold=0.5), [])

    def test_save_and_load(self):
        """Test model artifact serialization and reloading."""
        matcher = EntityMatcher(algorithm="auto", n_estimators=15)
        matcher.fit(self.X, self.y)
        orig_preds = matcher.predict_proba(self.X_val[:10])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            matcher.save(model_path)
            self.assertTrue(os.path.exists(model_path))

            loaded = EntityMatcher.load(model_path)
            loaded_preds = loaded.predict_proba(self.X_val[:10])
            np.testing.assert_allclose(orig_preds, loaded_preds, atol=1e-6)

    def test_convenience_functions(self):
        """Test train_model and predict_scores functions."""
        m = train_model(self.X, self.y, n_estimators=10)
        scores = predict_scores(m, self.X_val[:5])
        self.assertEqual(len(scores), 5)


if __name__ == "__main__":
    unittest.main()
