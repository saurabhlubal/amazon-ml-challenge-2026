"""
Submission verification and analysis script for Amazon ML Challenge 2026.
Wraps student_resource/utils/validate_submission.py and performs additional
distribution, singleton, and consistency checks.
"""

import os
import sys
import argparse
from typing import Dict, Set

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import (
    DELIM,
    validate_submission_files,
)


def analyze_submission_files(matching_path: str, candidate_path: str, test_dir: str):
    print("=" * 70)
    print("SUBMISSION FILE DIAGNOSTICS & SUMMARY")
    print("=" * 70)

    # 1. Matching Results Analysis
    if not os.path.isfile(matching_path):
        print(f"ERROR: {matching_path} does not exist.")
        return

    s1_rows = 0
    empty_matches = 0
    total_matches = 0
    s2_matches = 0
    s3_matches = 0
    match_distribution = {}

    with open(matching_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(DELIM)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(DELIM)
            s1_id = parts[0]
            ids_str = parts[1] if len(parts) > 1 else ""
            ids = [i.strip() for i in ids_str.split(",") if i.strip()]

            s1_rows += 1
            n_m = len(ids)
            match_distribution[n_m] = match_distribution.get(n_m, 0) + 1

            if n_m == 0:
                empty_matches += 1
            else:
                total_matches += n_m
                for mid in ids:
                    if mid.startswith("S2-"):
                        s2_matches += 1
                    elif mid.startswith("S3-"):
                        s3_matches += 1

    print(f"Matching Results: {matching_path}")
    print(f"  Total Source1 entities : {s1_rows:,}")
    print(f"  Singletons (0 matches) : {empty_matches:,} ({empty_matches / s1_rows * 100:.2f}%)")
    print(f"  Total matches          : {total_matches:,} (avg {total_matches / s1_rows:.2f}/S1)")
    print(f"  Source 2 matches       : {s2_matches:,} ({s2_matches / total_matches * 100 if total_matches else 0:.1f}%)")
    print(f"  Source 3 matches       : {s3_matches:,} ({s3_matches / total_matches * 100 if total_matches else 0:.1f}%)")

    print("\nMatch count distribution:")
    for k in sorted(match_distribution.keys())[:12]:
        print(f"  {k} matches: {match_distribution[k]:,} ({match_distribution[k] / s1_rows * 100:.2f}%)")

    # 2. Candidate Pairs Analysis
    if os.path.isfile(candidate_path):
        cand_rows = 0
        total_candidates = 0
        cand_distribution = {}
        with open(candidate_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(DELIM)
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split(DELIM)
                ids_str = parts[1] if len(parts) > 1 else ""
                ids = [i.strip() for i in ids_str.split(",") if i.strip()]
                cand_rows += 1
                n_c = len(ids)
                total_candidates += n_c

        print("\nCandidate Pairs: " + candidate_path)
        print(f"  Total Source1 entities : {cand_rows:,}")
        print(f"  Total candidates       : {total_candidates:,} (avg {total_candidates / cand_rows:.2f}/S1)")
        reduction_ratio = 1.0 - (total_matches / total_candidates) if total_candidates > 0 else 0.0
        print(f"  Filtering retention    : {total_matches / total_candidates * 100 if total_candidates else 0:.2f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Verify and analyze submission files")
    parser.add_argument("--matching", default=os.path.join(PROJECT_ROOT, "output", "matching_results.tsv"))
    parser.add_argument("--candidate", default=os.path.join(PROJECT_ROOT, "output", "candidate_pairs.tsv"))
    parser.add_argument("--test-dir", default=os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test"))
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()

    # 1. Run official validator
    code = validate_submission_files(args.matching, args.candidate, args.test_dir, args.check_ids)

    # 2. Print detailed distribution diagnostics if valid or readable
    analyze_submission_files(args.matching, args.candidate, args.test_dir)

    sys.exit(code)


if __name__ == "__main__":
    main()
