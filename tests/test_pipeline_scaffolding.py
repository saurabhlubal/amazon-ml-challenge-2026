"""
Unit tests for pipeline utilities, output formatting, interface compliance,
baseline modules, and synthetic end-to-end validation.
"""

import os
import sys
import shutil
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import (
    DELIM,
    MATCHING_HEADER,
    CANDIDATE_HEADER,
    format_id_list,
    parse_id_list,
    stream_tsv_records,
    validate_submission_files,
)
from scripts.run_pipeline import run_pipeline

import business_entity_resolution.src.normalization as norm
import business_entity_resolution.src.blocking as block
import business_entity_resolution.src.features as feat
import business_entity_resolution.src.model as mdl
import business_entity_resolution.src.evaluation as evl


class TestPipelineScaffolding(unittest.TestCase):

    def test_format_id_list(self):
        self.assertEqual(format_id_list([]), "")
        self.assertEqual(format_id_list(["S2-001"]), "S2-001")
        self.assertEqual(format_id_list(["S3-002", "S2-001", "S3-002"]), "S2-001,S3-002")
        self.assertEqual(format_id_list(["  S2-100 ", "S2-100"]), "S2-100")

    def test_parse_id_list(self):
        self.assertEqual(parse_id_list(""), [])
        self.assertEqual(parse_id_list("   "), [])
        self.assertEqual(parse_id_list("S2-001,S3-002"), ["S2-001", "S3-002"])

    def test_interface_imports(self):
        """Verify that all required shared interfaces exist and have expected callable names."""
        self.assertTrue(callable(getattr(norm, "normalize_record", None)))
        self.assertTrue(callable(getattr(block, "generate_candidates", None)))
        self.assertTrue(callable(getattr(feat, "build_features", None)))
        self.assertTrue(callable(getattr(mdl, "train_model", None)))
        self.assertTrue(callable(getattr(mdl, "predict_scores", None)))
        self.assertTrue(callable(getattr(mdl, "decide_matches", None)))
        self.assertTrue(callable(getattr(evl, "f05", None)))
        self.assertTrue(callable(getattr(evl, "evaluate_predictions", None)))

    def test_normalization(self):
        """Verify normalization preserves unicode and handles abbreviations."""
        # 1. Plus preservation in B+ Retail Inc
        rec1 = {
            "entity_id": "S1-1",
            "business_name": "B+ Retail Inc",
            "business_address": "1712 Montebello Avenue, Phoenix, AZ",
            "country": "US"
        }
        res1 = norm.normalize_record(rec1)
        self.assertIn("b+", res1["business_name"])
        self.assertIn("ave", res1["business_address"])
        self.assertEqual(res1["country"], "US")
        self.assertIn("1712", res1["address_numbers"])

        # 2. Devanagari Hindi text preservation
        rec2 = {
            "entity_id": "S2-1",
            "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "business_address": "KH NO. -570/13, NEW DELHI",
            "country": "India"
        }
        res2 = norm.normalize_record(rec2)
        self.assertIn("राम", res2["business_name"])
        self.assertEqual(res2["country"], "INDIA")

        # 3. French accented characters preservation
        rec3 = {
            "entity_id": "S3-1",
            "business_name": "LLC Moncada Léarning Center",
            "business_address": "10 Rue de Lyon",
            "country": "France"
        }
        res3 = norm.normalize_record(rec3)
        self.assertIn("léarning", res3["business_name"])
        self.assertEqual(res3["country"], "FRANCE")

    def test_blocking_and_features(self):
        """Test candidate generation and feature extraction between records."""
        cands = [
            {"entity_id": "S2-001", "business_name": "Acme Corporation", "business_address": "123 Main St", "country": "US"},
            {"entity_id": "S3-001", "business_name": "Acme Corp", "business_address": "123 Main Street", "country": "US"},
            {"entity_id": "S2-002", "business_name": "Global Logistics", "business_address": "999 Port Rd", "country": "US"},
        ]
        indexes = block.build_blocking_indexes(cands)

        s1 = {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"}
        found = block.generate_candidates(s1, indexes)
        self.assertIn("S2-001", found)
        self.assertIn("S3-001", found)
        self.assertNotIn("S2-002", found)

        # Feature extraction
        f_pos = feat.build_features(s1, cands[0])
        self.assertGreater(f_pos["name_jaccard"], 0.5)
        self.assertEqual(f_pos["country_match"], 1.0)

        f_neg = feat.build_features(s1, cands[2])
        self.assertEqual(f_neg["name_exact_norm"], 0.0)

    def test_model_training_and_decision(self):
        """Test model training on pair features, score prediction, and decision threshold."""
        X_train = [
            {"name_exact_norm": 1.0, "name_jaccard": 1.0, "addr_exact_norm": 1.0, "country_match": 1.0},
            {"name_exact_norm": 1.0, "name_jaccard": 0.8, "addr_exact_norm": 0.5, "country_match": 1.0},
            {"name_exact_norm": 0.0, "name_jaccard": 0.1, "addr_exact_norm": 0.0, "country_match": 0.0},
            {"name_exact_norm": 0.0, "name_jaccard": 0.0, "addr_exact_norm": 0.0, "country_match": 1.0},
        ]
        y_train = [1, 1, 0, 0]

        model = mdl.train_model(X_train, y_train)
        self.assertIsNotNone(model)

        scores = mdl.predict_scores(model, X_train)
        self.assertEqual(len(scores), 4)
        self.assertGreater(scores[0], scores[2])

        matches = mdl.decide_matches(["C1", "C2", "C3", "C4"], scores, threshold=0.5)
        self.assertIn("C1", matches)
        self.assertNotIn("C3", matches)

    def test_evaluation_macro_f05(self):
        """Test macro-averaged F0.5 under various conditions including singletons."""
        # 1. Formula test
        self.assertEqual(evl.f05(0.0, 0.0), 0.0)
        self.assertAlmostEqual(evl.f05(2.0 / 3.0, 1.0), 0.7142857, places=4)

        # 2. Macro evaluation test
        # S1-1: True match [S2-1, S3-1], Pred [S2-1, S3-1] -> Prec=1.0, Rec=1.0 -> F0.5=1.0
        # S1-2: Singleton True [], Pred [] -> F0.5=1.0
        # S1-3: Singleton True [], Pred [S2-2] -> F0.5=0.0
        # S1-4: True match [S2-3], Pred [] -> F0.5=0.0
        gt = {
            "S1-1": {"S2-1", "S3-1"},
            "S1-2": set(),
            "S1-3": set(),
            "S1-4": {"S2-3"}
        }
        pred = {
            "S1-1": {"S2-1", "S3-1"},
            "S1-2": set(),
            "S1-3": {"S2-2"},
            "S1-4": set()
        }
        macro_score = evl.evaluate_predictions(gt, pred)
        # Expected: (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.50
        self.assertAlmostEqual(macro_score, 0.50, places=4)

    def test_synthetic_end_to_end_validation(self):
        """Create a mini synthetic test environment, run the pipeline, and run the official validator."""
        temp_dir = tempfile.mkdtemp()
        try:
            test_dir = os.path.join(temp_dir, "test")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(test_dir, exist_ok=True)

            # Create synthetic test_source1.tsv
            with open(os.path.join(test_dir, "test_source1.tsv"), "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("S1-001\tAcme Corp\t123 Main St\tUS\n")
                f.write("S1-002\tApex Store\t456 Market Rd\tIndia\n")
                f.write("S1-003\tCafé Paris\t10 Rue de Lyon\tFrance\n")

            # Create synthetic test_source2.tsv
            with open(os.path.join(test_dir, "test_source2.tsv"), "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("S2-001\tAcme Corp\t123 Main St\tUS\n")
                f.write("S2-002\tOther Store\t789 Elm St\tUS\n")

            # Create synthetic test_source3.tsv
            with open(os.path.join(test_dir, "test_source3.tsv"), "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("S3-001\tAcme Corp\t123 Main St\tUS\n")
                f.write("S3-002\tCafé Paris\t10 Rue de Lyon\tFrance\n")

            # Run pipeline in mock mode
            exit_code = run_pipeline(
                test_dir=test_dir,
                output_dir=output_dir,
                threshold=0.5,
                use_mock=True,
                validate=True,
                check_ids=True,
            )
            self.assertEqual(exit_code, 0, "Pipeline with validator should exit with 0 (PASS)")

            # Check that output files exist
            matching_path = os.path.join(output_dir, "matching_results.tsv")
            candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
            self.assertTrue(os.path.isfile(matching_path))
            self.assertTrue(os.path.isfile(candidate_path))

            # Inspect lines in matching_results.tsv
            with open(matching_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 4)  # Header + 3 entities
                self.assertEqual(lines[0], MATCHING_HEADER)
                self.assertIn("S1-001\tS2-001,S3-001\n", lines)
                self.assertIn("S1-002\t\n", lines)
                self.assertIn("S1-003\tS3-002\n", lines)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_synthetic_end_to_end_validation_with_real_modules(self):
        """Test full pipeline with REAL baseline modules (use_mock=False) and validate with official validator."""
        temp_dir = tempfile.mkdtemp()
        try:
            test_dir = os.path.join(temp_dir, "test")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(test_dir, exist_ok=True)

            with open(os.path.join(test_dir, "test_source1.tsv"), "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("S1-001\tAcme Corp\t123 Main St\tUS\n")
                f.write("S1-002\tApex Store\t456 Market Rd\tIndia\n")
                f.write("S1-003\tCafé Paris\t10 Rue de Lyon\tFrance\n")

            with open(os.path.join(test_dir, "test_source2.tsv"), "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("S2-001\tAcme Corporation\t123 Main Street\tUS\n")
                f.write("S2-002\tUnrelated Store\t999 Elm St\tUS\n")

            with open(os.path.join(test_dir, "test_source3.tsv"), "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("S3-001\tAcme Corp\t123 Main St\tUS\n")
                f.write("S3-002\tCafé Paris\t10 Rue de Lyon\tFrance\n")

            exit_code = run_pipeline(
                test_dir=test_dir,
                output_dir=output_dir,
                threshold=0.5,
                use_mock=False,
                validate=True,
                check_ids=True,
            )
            self.assertEqual(exit_code, 0, "Real module pipeline should exit with 0 (PASS)")

            matching_path = os.path.join(output_dir, "matching_results.tsv")
            with open(matching_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 4)
                self.assertEqual(lines[0], MATCHING_HEADER)
                self.assertIn("S1-001\tS2-001,S3-001\n", lines)
                self.assertIn("S1-002\t\n", lines)
                self.assertIn("S1-003\tS3-002\n", lines)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
