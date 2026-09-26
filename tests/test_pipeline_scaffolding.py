"""
Unit tests for pipeline utilities, output formatting, interface compliance,
and synthetic end-to-end validation.
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


class TestPipelineScaffolding(unittest.TestCase):

    def test_format_id_list(self):
        self.assertEqual(format_id_list([]), "")
        self.assertEqual(format_id_list(["S2-001"]), "S2-001")
        # Duplicates and order
        self.assertEqual(format_id_list(["S3-002", "S2-001", "S3-002"]), "S2-001,S3-002")
        # Whitespace handling
        self.assertEqual(format_id_list(["  S2-100 ", "S2-100"]), "S2-100")

    def test_parse_id_list(self):
        self.assertEqual(parse_id_list(""), [])
        self.assertEqual(parse_id_list("   "), [])
        self.assertEqual(parse_id_list("S2-001,S3-002"), ["S2-001", "S3-002"])

    def test_interface_imports(self):
        """Verify that all required shared interfaces exist and have expected callable names."""
        import business_entity_resolution.src.normalization as norm
        import business_entity_resolution.src.blocking as block
        import business_entity_resolution.src.features as feat
        import business_entity_resolution.src.model as mdl
        import business_entity_resolution.src.evaluation as evl

        self.assertTrue(callable(getattr(norm, "normalize_record", None)))
        self.assertTrue(callable(getattr(block, "generate_candidates", None)))
        self.assertTrue(callable(getattr(feat, "build_features", None)))
        self.assertTrue(callable(getattr(mdl, "train_model", None)))
        self.assertTrue(callable(getattr(mdl, "predict_scores", None)))
        self.assertTrue(callable(getattr(mdl, "decide_matches", None)))
        self.assertTrue(callable(getattr(evl, "f05", None)))
        self.assertTrue(callable(getattr(evl, "evaluate_predictions", None)))

    def test_evaluation_f05(self):
        """Test F0.5 formula from evaluation.py."""
        from business_entity_resolution.src.evaluation import f05
        self.assertEqual(f05(0.0, 0.0), 0.0)
        # Precision = 2/3, Recall = 1.0 -> F0.5 ≈ 0.7142857
        score = f05(2.0 / 3.0, 1.0)
        self.assertAlmostEqual(score, 0.7142857, places=4)

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
                # S1-001 matches S2-001 and S3-001
                self.assertIn("S1-001\tS2-001,S3-001\n", lines)
                # S1-002 has no match (singleton)
                self.assertIn("S1-002\t\n", lines)
                # S1-003 matches S3-002
                self.assertIn("S1-003\tS3-002\n", lines)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
