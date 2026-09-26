"""
Merge partitioned submission files and run official competition validator.
"""

import os
import sys
import glob
import subprocess
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import CANDIDATE_HEADER, MATCHING_HEADER


def merge_shards(
    shard_dir: str,
    output_dir: str,
    test_dir: Optional[str] = None,
    validate: bool = True,
) -> bool:
    """
    Concatenate all partition outputs into final submission files and validate.
    """
    os.makedirs(output_dir, exist_ok=True)
    final_cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    final_match_path = os.path.join(output_dir, "matching_results.tsv")

    cand_shards = sorted(glob.glob(os.path.join(shard_dir, "*candidate*.tsv")))
    match_shards = sorted(glob.glob(os.path.join(shard_dir, "*matching*.tsv")))

    # Exclude final outputs if shard_dir == output_dir
    cand_shards = [s for s in cand_shards if os.path.abspath(s) != os.path.abspath(final_cand_path)]
    match_shards = [s for s in match_shards if os.path.abspath(s) != os.path.abspath(final_match_path)]

    print(f"Merging {len(cand_shards)} candidate shards and {len(match_shards)} matching shards...")

    total_cand_rows = 0
    with open(final_cand_path, "w", encoding="utf-8", newline="") as f_out:
        f_out.write(CANDIDATE_HEADER)
        for s_path in cand_shards:
            with open(s_path, "r", encoding="utf-8") as f_in:
                next(f_in, None)  # Skip header
                for line in f_in:
                    f_out.write(line)
                    total_cand_rows += 1

    total_match_rows = 0
    with open(final_match_path, "w", encoding="utf-8", newline="") as f_out:
        f_out.write(MATCHING_HEADER)
        for s_path in match_shards:
            with open(s_path, "r", encoding="utf-8") as f_in:
                next(f_in, None)  # Skip header
                for line in f_in:
                    f_out.write(line)
                    total_match_rows += 1

    print(f"Merged candidate_pairs.tsv: {total_cand_rows:,} rows ({os.path.getsize(final_cand_path) / (1024**2):.1f} MB)")
    print(f"Merged matching_results.tsv: {total_match_rows:,} rows ({os.path.getsize(final_match_path) / (1024**2):.1f} MB)")

    if validate and test_dir:
        val_script = os.path.join(PROJECT_ROOT, "student_resource", "utils", "validate_submission.py")
        if os.path.isfile(val_script):
            print("\nRunning official submission validator...")
            cmd = [
                sys.executable,
                val_script,
                "--matching", final_match_path,
                "--candidate", final_cand_path,
                "--test-dir", test_dir,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
            return res.returncode == 0

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge shard outputs into final submission files.")
    parser.add_argument("--shard-dir", required=True, help="Directory containing shard TSVs")
    parser.add_argument("--output-dir", default="output", help="Final output directory")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Test dataset directory")
    parser.add_argument("--skip-validation", action="store_true", help="Skip official validator")

    args = parser.parse_args()
    merge_shards(args.shard_dir, args.output_dir, args.test_dir, validate=not args.skip_validation)
