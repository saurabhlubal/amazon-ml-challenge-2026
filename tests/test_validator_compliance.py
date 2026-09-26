"""
End-to-end test verifying strict compliance with student_resource/utils/validate_submission.py.
"""

import os
import subprocess
import sys
import tempfile
import unittest
import pandas as pd
from business_entity_resolution.src.evaluation import apply_decision_rule


class TestValidatorCompliance(unittest.TestCase):
    def test_validate_submission_passes(self):
        """
        Create synthetic test set and generated outputs, then run the official
        validate_submission.py script to guarantee exit code 0 (PASS).
        """
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        validator_script = os.path.join(workspace_root, "student_resource", "utils", "validate_submission.py")
        self.assertTrue(os.path.exists(validator_script), f"Validator not found at {validator_script}")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "dataset", "test")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # 1. Create test source files
            test_s1 = pd.DataFrame([
                {"entity_id": "S1-0001", "business_name": "Alpha Corp", "business_address": "1 Main St", "country": "US"},
                {"entity_id": "S1-0002", "business_name": "Beta LLC", "business_address": "2 Elm St", "country": "US"},
                {"entity_id": "S1-0003", "business_name": "Gamma SARL", "business_address": "3 Rue Paris", "country": "France"},
                {"entity_id": "S1-0004", "business_name": "Singleton Shop", "business_address": "4 Pine Rd", "country": "India"},
            ])
            test_s2 = pd.DataFrame([
                {"entity_id": "S2-0010", "business_name": "Alpha Corporation", "business_address": "1 Main Street", "country": "US"},
                {"entity_id": "S2-0020", "business_name": "Beta Co", "business_address": "2 Elm Street", "country": "US"},
            ])
            test_s3 = pd.DataFrame([
                {"entity_id": "S3-0030", "business_name": "Gamma Societe", "business_address": "3 Rue de Paris", "country": "France"},
            ])

            test_s1.to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)
            test_s2.to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)
            test_s3.to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)

            # 2. Create candidate pairs
            cand_pairs_df = pd.DataFrame([
                {"source1_entity_id": "S1-0001", "candidate_entity_ids": "S2-0010"},
                {"source1_entity_id": "S1-0002", "candidate_entity_ids": "S2-0020"},
                {"source1_entity_id": "S1-0003", "candidate_entity_ids": "S3-0030"},
                {"source1_entity_id": "S1-0004", "candidate_entity_ids": ""},
            ])
            cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
            cand_pairs_df.to_csv(cand_path, sep="\t", index=False)

            # 3. Create scored candidates and apply decision rule
            scored_candidates = pd.DataFrame([
                {"source1_entity_id": "S1-0001", "candidate_entity_id": "S2-0010", "score": 0.95},
                {"source1_entity_id": "S1-0002", "candidate_entity_id": "S2-0020", "score": 0.88},
                {"source1_entity_id": "S1-0003", "candidate_entity_id": "S3-0030", "score": 0.91},
            ])
            all_s1 = list(test_s1["entity_id"])
            matching_df = apply_decision_rule(scored_candidates, all_s1, threshold=0.70)
            match_path = os.path.join(output_dir, "matching_results.tsv")
            matching_df.to_csv(match_path, sep="\t", index=False)

            # 4. Run validate_submission.py with --check-ids
            cmd = [
                sys.executable,
                validator_script,
                "--matching", match_path,
                "--candidate", cand_path,
                "--test-dir", test_dir,
                "--check-ids",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)

            self.assertEqual(proc.returncode, 0, f"Validator failed with stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            self.assertIn("PASS — no blocking issues found", proc.stdout)


if __name__ == "__main__":
    unittest.main()
