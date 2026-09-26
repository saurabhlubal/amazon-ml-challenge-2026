"""
Comprehensive Benchmark Script for Normalization and Blocking on Challenge Data.

Measures:
- True-match blocking recall against official ground truth
- Candidate distribution (Mean, Median, P90, P95, P99, Max)
- Index build and query runtime (S1 entities per second)
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Set

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath("."))

from business_entity_resolution.src.blocking import (
    build_blocking_indexes,
    evaluate_blocking,
    stream_source_file,
)


def run_benchmark(
    dataset_dir: str = "student_resource/dataset",
    sample_s1: int = 10000,
    background_s2_s3: int = 100000,
    max_candidates: int = 60,
):
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — BLOCKING BENCHMARK (Teammate 1)")
    print("=" * 70)
    print(f"Sample S1 Entities:     {sample_s1:,}")
    print(f"Background Pool/Source: {background_s2_s3:,}")
    print(f"Max Candidates Cap:     {max_candidates}")
    print("-" * 70)

    train_dir = os.path.join(dataset_dir, "train")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    # 1. Load Ground Truth
    print("[1/4] Loading ground truth sample...")
    gt_map: Dict[str, Set[str]] = {}
    target_ids: Set[str] = set()

    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= sample_s1:
                break
            p = line.strip().split("\t")
            matches = [m.strip() for m in p[1].split(",") if m.strip()] if len(p) > 1 and p[1] else []
            gt_map[p[0]] = set(matches)
            for m in matches:
                target_ids.add(m)

    total_targets = sum(len(m) for m in gt_map.values())
    print(f"      Loaded {len(gt_map):,} S1 queries with {total_targets:,} true match targets ({len(target_ids):,} unique).")

    # 2. Load S1 Records
    print("[2/4] Loading corresponding Source 1 records...")
    s1_records: Dict[str, Dict[str, str]] = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if p[0] in gt_map:
                s1_records[p[0]] = {
                    "entity_id": p[0],
                    "business_name": p[1],
                    "business_address": p[2] if len(p) > 2 else "",
                    "country": p[3] if len(p) > 3 else "UNKNOWN",
                }
                if len(s1_records) == len(gt_map):
                    break

    # 3. Load Candidate Pool (True targets + random background distractors)
    print("[3/4] Streaming candidate pool (Source 2 and Source 3)...")
    candidate_records: Dict[str, Dict[str, str]] = {}

    def load_candidates_from_file(filepath: str, max_bg: int):
        bg_count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.strip().split("\t")
                sid = p[0]
                if sid in target_ids:
                    candidate_records[sid] = {
                        "entity_id": sid,
                        "business_name": p[1],
                        "business_address": p[2] if len(p) > 2 else "",
                        "country": p[3] if len(p) > 3 else "UNKNOWN",
                    }
                elif bg_count < max_bg:
                    candidate_records[sid] = {
                        "entity_id": sid,
                        "business_name": p[1],
                        "business_address": p[2] if len(p) > 2 else "",
                        "country": p[3] if len(p) > 3 else "UNKNOWN",
                    }
                    bg_count += 1

    load_candidates_from_file(s2_path, background_s2_s3)
    load_candidates_from_file(s3_path, background_s2_s3)
    print(f"      Candidate pool assembled: {len(candidate_records):,} total entities.")

    # 4. Build Multi-Index
    print("[4/4] Building multi-blocking indexes...")
    t0 = time.time()
    indexes = build_blocking_indexes(candidate_records.values(), verbose=False)
    index_time = time.time() - t0
    print(f"      Indexes built in {index_time:.2f} seconds.")

    # 5. Evaluate Blocking
    print("\nRunning blocking evaluation on query set...")
    metrics = evaluate_blocking(
        s1_records=s1_records,
        ground_truth=gt_map,
        indexes=indexes,
        max_candidates=max_candidates,
    )

    print("\n" + "=" * 70)
    print("OFFICIAL BLOCKING EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Source-1 Entities:       {len(s1_records):,}")
    print(f"Candidate Records in Pool:     {len(candidate_records):,}")
    print(f"Total True Match Pairs:        {metrics['total_true_matches']:,}")
    print(f"Found True Match Pairs:        {metrics['found_true_matches']:,}")
    print(f"Blocking Recall:               {metrics['blocking_recall']:.4%} ({metrics['found_true_matches']}/{metrics['total_true_matches']})")
    print(f"Average Candidates / S1:       {metrics['average_candidates']:.2f}")
    print(f"Median Candidates / S1:        {metrics['median_candidates']:.1f}")
    print(f"P90 Candidates / S1:           {metrics['p90_candidates']:.1f}")
    print(f"P95 Candidates / S1:           {metrics['p95_candidates']:.1f}")
    print(f"P99 Candidates / S1:           {metrics['p99_candidates']:.1f}")
    print(f"Maximum Candidates / S1:       {metrics['max_candidates']}")
    print(f"Query Execution Time:          {metrics['runtime_sec']:.2f} seconds")
    print(f"Throughput:                    {metrics['s1_per_sec']:.1f} S1 records/second")
    print(f"Index Build Time:              {index_time:.2f} seconds")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blocking Benchmark")
    parser.add_argument("--sample-s1", type=int, default=10000)
    parser.add_argument("--background", type=int, default=100000)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--dataset-dir", type=str, default="student_resource/dataset")
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset_dir,
        sample_s1=args.sample_s1,
        background_s2_s3=args.background,
        max_candidates=args.max_candidates,
    )
