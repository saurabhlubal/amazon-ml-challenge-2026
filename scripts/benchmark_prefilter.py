"""
Benchmark Script for Cheap Candidate Prefilter Configurations.
Evaluates:
- 'none' (Baseline / Raw Blocker output)
- 'country_only'
- 'conservative'
- 'moderate'
- 'selective'

Measures on the exact same dataset:
1. Candidate Recall against Ground Truth
2. Candidate Distribution (Avg, Median, P95, Max)
3. Zero-candidate S1 count
4. Macro F0.5
5. Precision
6. Match Recall
7. Throughput (S1/sec)
8. Peak RAM (MB)
9. Total Runtime
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import gc
import json
import numpy as np
import psutil
from typing import Dict, List, Set, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.blocking import get_blocking_keys
from business_entity_resolution.src.prefilter import prefilter_pair, filter_candidates
from business_entity_resolution.src.model import FastLogisticRegression
from business_entity_resolution.src.evaluation import f05, evaluate_predictions, to_id_set
from sagemaker.vectorized_matcher import CompactCand, CompactInvertedIndex, extract_features_vectorized


def run_prefilter_benchmarks(
    n_s1: int = 5000,
    background_cands: int = 150000,
    model_path: str = "business_entity_resolution/src/trained_model.json",
):
    print("=" * 85)
    print("AMAZON ML CHALLENGE 2026 — CANDIDATE PREFILTER BENCHMARK SUITE")
    print(f"Sample: {n_s1:,} S1 queries | Background Pool: {background_cands:,} candidates")
    print("=" * 85)

    dataset_train_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
    s1_file = os.path.join(dataset_train_dir, "train_source1.tsv")
    s2_file = os.path.join(dataset_train_dir, "train_source2.tsv")
    s3_file = os.path.join(dataset_train_dir, "train_source3.tsv")
    gt_file = os.path.join(dataset_train_dir, "train_ground_truth.tsv")

    # Load model
    full_model_path = os.path.join(PROJECT_ROOT, model_path)
    with open(full_model_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    model = FastLogisticRegression()
    model.weights = list(m_data["weights"])
    model.bias = float(m_data["bias"])
    weights = np.array(model.weights, dtype=np.float32)
    bias = float(model.bias)
    threshold = float(m_data.get("optimal_threshold", 0.50))
    logit_thresh = np.log(threshold / (1.0 - threshold)) if (0.0 < threshold < 1.0) else 0.0

    # 1. Load S1 queries and Ground Truth
    print(f"\n[1/3] Loading {n_s1:,} Source 1 entities and Ground Truth...")
    s1_raw_list = list(stream_tsv_records(s1_file, max_records=n_s1))
    s1_norms = [normalize_record(r) for r in s1_raw_list]
    target_s1_ids = {r["entity_id"] for r in s1_raw_list}

    gt_map: Dict[str, Set[str]] = {}
    needed_true_cands: Set[str] = set()
    total_true_pairs = 0

    for rec in stream_tsv_records(gt_file):
        sid = rec.get("source1_entity_id", "")
        if sid in target_s1_ids:
            matches = to_id_set(rec.get("matched_entity_ids", ""))
            gt_map[sid] = matches
            needed_true_cands.update(matches)
            total_true_pairs += len(matches)

    for sid in target_s1_ids:
        if sid not in gt_map:
            gt_map[sid] = set()

    print(f"      Loaded {len(s1_norms):,} S1 queries with {total_true_pairs:,} true positive pairs.")

    # 2. Build Candidate Index (True positives + Background records)
    print(f"\n[2/3] Building Inverted Index (True targets + {background_cands:,} background records)...")
    t0_idx = time.time()
    index = CompactInvertedIndex(max_bucket_size=300)

    bg_count = 0
    half_bg = background_cands // 2
    for p, limit in ((s2_file, half_bg), (s3_file, half_bg)):
        for rec in stream_tsv_records(p):
            eid = rec.get("entity_id", "")
            if eid in needed_true_cands:
                index.add_record(rec, strategy="combined")
            elif bg_count < background_cands:
                index.add_record(rec, strategy="combined")
                bg_count += 1

    print(f"      Index built: {len(index):,} candidates in {time.time() - t0_idx:.2f}s.")

    # 3. Test Configurations
    configs = ["none", "country_only", "conservative", "moderate", "selective"]
    results_summary = []

    print("\n[3/3] Evaluating Prefilter Configurations...")
    print("-" * 85)

    proc = psutil.Process()

    for cfg in configs:
        gc.collect()
        ram_before = proc.memory_info().rss / (1024 * 1024)
        t_start = time.time()

        captured_cands = 0
        cand_counts = []
        zero_cand_s1 = 0
        pred_map: Dict[str, Set[str]] = {}

        for s1 in s1_norms:
            sid = s1["entity_id"]
            cand_idxs = index.get_candidate_indices(s1, strategy="combined")
            if not cand_idxs:
                cand_counts.append(0)
                zero_cand_s1 += 1
                pred_map[sid] = set()
                continue

            cands_raw = [index.cand_records[i] for i in cand_idxs]

            # Apply candidate prefilter
            if cfg == "none":
                filtered_cands = cands_raw
            else:
                filtered_cands = [c for c in cands_raw if prefilter_pair(s1, c, config=cfg)]

            n_surv = len(filtered_cands)
            cand_counts.append(n_surv)
            if n_surv == 0:
                zero_cand_s1 += 1
                pred_map[sid] = set()
                continue

            # Candidate recall tracking
            surviving_eids = {c.entity_id for c in filtered_cands}
            true_for_s1 = gt_map.get(sid, set())
            if true_for_s1:
                captured_cands += len(surviving_eids & true_for_s1)

            # Feature extraction and model scoring
            X = extract_features_vectorized(s1, filtered_cands)
            logits = np.dot(X, weights) + bias
            pred_mask = logits >= logit_thresh

            matched_eids = set()
            for i, matched in enumerate(pred_mask):
                if matched:
                    matched_eids.add(filtered_cands[i].entity_id)
            pred_map[sid] = matched_eids

        elapsed = time.time() - t_start
        ram_after = proc.memory_info().rss / (1024 * 1024)
        peak_ram = max(ram_before, ram_after)

        # Compute metrics
        cand_recall = captured_cands / max(total_true_pairs, 1)
        avg_cands = np.mean(cand_counts)
        med_cands = np.median(cand_counts)
        p95_cands = np.percentile(cand_counts, 95)
        max_cands = np.max(cand_counts)
        s1_per_sec = len(s1_norms) / max(elapsed, 0.001)

        macro_f05 = evaluate_predictions(gt_map, pred_map)
        total_pred = sum(len(p) for p in pred_map.values())
        total_correct = sum(len(pred_map.get(sid, set()) & g) for sid, g in gt_map.items())
        precision = total_correct / max(total_pred, 1)
        match_recall = total_correct / max(total_true_pairs, 1)

        cfg_summary = {
            "config": cfg,
            "cand_recall": cand_recall,
            "avg_cands": avg_cands,
            "median_cands": med_cands,
            "p95_cands": p95_cands,
            "max_cands": int(max_cands),
            "zero_cand_s1": zero_cand_s1,
            "macro_f05": macro_f05,
            "precision": precision,
            "match_recall": match_recall,
            "s1_per_sec": s1_per_sec,
            "peak_ram_mb": peak_ram,
            "total_time_s": elapsed,
        }
        results_summary.append(cfg_summary)

        print(f"Config: {cfg.upper():<14} | Cand Recall: {cand_recall:7.2%} | Avg Cands: {avg_cands:5.1f} (P95: {p95_cands:4.0f})")
        print(f"  F0.5: {macro_f05:.4f} | Prec: {precision:.4f} | Rec: {match_recall:.4f} | Rate: {s1_per_sec:6.1f} S1/s | Time: {elapsed:.2f}s")
        print("-" * 85)

    # Print Table
    print("\n" + "=" * 105)
    print(f"{'Configuration':<14} | {'CandRec':<8} | {'AvgCand':<7} | {'P95':<5} | {'F0.5':<7} | {'Prec':<7} | {'MatchRec':<8} | {'S1/sec':<7} | {'Time(s)':<7}")
    print("=" * 105)
    for r in results_summary:
        print(f"{r['config']:<14} | {r['cand_recall']:7.2%} | {r['avg_cands']:7.1f} | {r['p95_cands']:5.0f} | {r['macro_f05']:7.4f} | {r['precision']:7.4f} | {r['match_recall']:8.4f} | {r['s1_per_sec']:7.1f} | {r['total_time_s']:7.2f}")
    print("=" * 105)

    out_json = os.path.join(PROJECT_ROOT, "output", "benchmark_prefilter_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved benchmark results to {out_json}")


if __name__ == "__main__":
    run_prefilter_benchmarks(n_s1=5000, background_cands=150000)
