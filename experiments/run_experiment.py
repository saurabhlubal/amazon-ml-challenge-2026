"""
Comparative Blocking Experiment Runner for Business Entity Resolution.
Directly compares:
  A. Baseline Blocker
  B. New Blocker (Shingles, Compact Names, PIN/Postal, Street Number, State)
  C. Combined Blocker
on the exact same 5,000 Source1 records against ground truth.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Set, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.blocking import build_blocking_indexes, generate_candidates
from business_entity_resolution.src.evaluation import to_id_set


def run_blocking_comparison(
    n_s1_sample: int = 5000,
    background_s2_s3_records: int = 50000,
):
    print("=" * 70)
    print("COMPARATIVE BLOCKING STRATEGY EXPERIMENT")
    print("=" * 70)

    dataset_train_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
    s1_file = os.path.join(dataset_train_dir, "train_source1.tsv")
    s2_file = os.path.join(dataset_train_dir, "train_source2.tsv")
    s3_file = os.path.join(dataset_train_dir, "train_source3.tsv")
    gt_file = os.path.join(dataset_train_dir, "train_ground_truth.tsv")

    # 1. Load Ground Truth for S1 sample
    print(f"Loading first {n_s1_sample:,} Source 1 records...")
    s1_records: List[Dict[str, str]] = []
    target_s1_ids: Set[str] = set()
    for rec in stream_tsv_records(s1_file, max_records=n_s1_sample):
        s1_records.append(rec)
        target_s1_ids.add(rec["entity_id"])

    print(f"Loading ground truth mappings for {len(target_s1_ids):,} entities...")
    gt_map: Dict[str, Set[str]] = {}
    needed_target_cands: Set[str] = set()
    total_true_positives = 0

    for rec in stream_tsv_records(gt_file):
        s1_id = rec.get("source1_entity_id", "")
        if s1_id in target_s1_ids:
            matches_str = rec.get("matched_entity_ids", "")
            matches = to_id_set(matches_str)
            gt_map[s1_id] = matches
            needed_target_cands.update(matches)
            total_true_positives += len(matches)
            if len(gt_map) >= len(target_s1_ids):
                break

    for s1_id in target_s1_ids:
        if s1_id not in gt_map:
            gt_map[s1_id] = set()

    n_singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    print(f"Ground Truth loaded:")
    print(f"  Total S1 entities   : {len(target_s1_ids):,}")
    print(f"  True matched pairs  : {total_true_positives:,} (avg {total_true_positives / len(target_s1_ids):.2f}/S1)")
    print(f"  True singletons     : {n_singletons:,} ({n_singletons / len(target_s1_ids) * 100:.1f}%)")
    print(f"  Target S2/S3 IDs    : {len(needed_target_cands):,}")

    # 2. Build candidate pool: needed target candidates + background sample
    print(f"\nBuilding candidate pool (True targets + {background_s2_s3_records:,} background records)...")
    candidate_pool: List[Dict[str, str]] = []
    found_targets: Set[str] = set()
    bg_count = 0

    for path in (s2_file, s3_file):
        for rec in stream_tsv_records(path):
            eid = rec.get("entity_id", "")
            if eid in needed_target_cands:
                candidate_pool.append(rec)
                found_targets.add(eid)
            elif bg_count < background_s2_s3_records:
                candidate_pool.append(rec)
                bg_count += 1

    print(f"Candidate pool built: {len(candidate_pool):,} records total (targets captured: {len(found_targets):,}/{len(needed_target_cands):,})")

    # 3. Evaluate each strategy
    strategies = [
        ("A. Baseline Blocker", "baseline"),
        ("B. New Blocker Only", "new"),
        ("C. Combined Blocker", "combined"),
    ]

    results = []

    for label, strat in strategies:
        print(f"\nEvaluating: {label}...")
        t_build = time.time()
        indexes = build_blocking_indexes(candidate_pool, max_bucket_size=500, strategy=strat, store_records=False)
        t_build_elapsed = time.time() - t_build

        t_gen = time.time()
        candidate_counts = []
        captured = 0

        for s1_rec in s1_records:
            s1_id = s1_rec["entity_id"]
            cands = generate_candidates(s1_rec, indexes)
            candidate_counts.append(len(cands))
            true_m = gt_map.get(s1_id, set())
            captured += len(cands & true_m)

        t_gen_elapsed = time.time() - t_gen
        total_cands = sum(candidate_counts)
        avg_cands = np.mean(candidate_counts)
        p95_cands = float(np.percentile(candidate_counts, 95))
        max_cands = max(candidate_counts)
        recall = captured / total_true_positives if total_true_positives > 0 else 0.0

        res = {
            "label": label,
            "strategy": strat,
            "num_keys": len(indexes["index"]),
            "build_time": t_build_elapsed,
            "gen_time": t_gen_elapsed,
            "total_cands": total_cands,
            "avg_cands": avg_cands,
            "p95_cands": p95_cands,
            "max_cands": max_cands,
            "captured": captured,
            "recall": recall,
        }
        results.append(res)

    print("\n" + "=" * 95)
    print(f"{'Configuration':<25} {'Recall':<10} {'Avg Cands':<12} {'P95 Cands':<12} {'Max Cands':<12} {'Total Cands':<14} {'Gen Time'}")
    print("=" * 95)
    for r in results:
        print(f"{r['label']:<25} {r['recall']:<10.4f} {r['avg_cands']:<12.2f} {r['p95_cands']:<12.1f} {r['max_cands']:<12} {r['total_cands']:<14,} {r['gen_time']:.2f}s")
    print("=" * 95)

    return results


if __name__ == "__main__":
    run_blocking_comparison(n_s1_sample=5000, background_s2_s3_records=50000)
