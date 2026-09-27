"""
Benchmark Matrix for Four Configurations (Amazon ML Challenge 2026 Phase 4).

Compares on the exact same representative training subset:
A. Current validated pipeline (No prefilter, NumPy matcher)
B. DuckDB/Polars optimization only (No prefilter, Polars matcher)
C. Cheap prefilter only (Conservative prefilter + Cap 80, NumPy matcher)
D. Cheap prefilter + DuckDB/Polars (Conservative prefilter + Cap 80, Polars matcher)

Measures all 12 required metrics:
1. Candidate recall
2. Overall ground-truth recall
3. Macro F0.5
4. Precision
5. Average candidates/S1
6. Median candidates/S1
7. P95 candidates/S1
8. Maximum candidates/S1
9. Zero-candidate S1 count
10. S1/sec throughput
11. Peak RAM (MB)
12. Total runtime (s)
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
from business_entity_resolution.src.prefilter import prefilter_pair
from business_entity_resolution.src.model import FastLogisticRegression
from business_entity_resolution.src.evaluation import f05, evaluate_predictions, to_id_set
from sagemaker.vectorized_matcher import CompactInvertedIndex, CompactCand, extract_features_vectorized
from sagemaker.duckdb_matcher import process_s1_batch_polars


def process_s1_batch_numpy(
    s1_batch: List[Dict[str, Any]],
    index: CompactInvertedIndex,
    model: FastLogisticRegression,
    threshold: float,
    blocking_strategy: str = "combined",
    prefilter_config: str = "none",
    max_candidates: Any = None,
) -> List[Tuple[str, List[str], List[str]]]:
    """Flexible NumPy batch processor supporting optional prefilter and capping."""
    weights = np.array(model.weights, dtype=np.float32)
    bias = float(model.bias)
    logit_thresh = np.log(threshold / (1.0 - threshold)) if (0.0 < threshold < 1.0) else 0.0

    results = []

    for s1_rec in s1_batch:
        s1_norm = normalize_record(s1_rec) if "name_tokens_set" not in s1_rec else s1_rec
        s1_id = s1_norm["entity_id"]

        cand_indices = index.get_candidate_indices(s1_norm, strategy=blocking_strategy)
        if not cand_indices:
            results.append((s1_id, [], []))
            continue

        cand_records = [index.cand_records[ci] for ci in cand_indices]

        # Prefilter
        if prefilter_config != "none":
            filtered = [c for c in cand_records if prefilter_pair(s1_norm, c, config=prefilter_config)]
        else:
            filtered = cand_records

        # Capping
        if max_candidates and len(filtered) > max_candidates:
            s1_nt = s1_norm.get("name_tokens_set", set())
            s1_at = s1_norm.get("address_tokens_set", set())
            s1_name = s1_norm.get("business_name", "")

            def cheap_score(c):
                score = 0
                if s1_name == c.business_name:
                    score += 10
                score += len(s1_nt & c.name_tokens_set) * 3
                score += len(s1_at & c.address_tokens_set) * 2
                return score

            filtered.sort(key=cheap_score, reverse=True)
            filtered = filtered[:max_candidates]

        if not filtered:
            results.append((s1_id, [], []))
            continue

        cand_eids = sorted([c.entity_id for c in filtered])

        # Feature extraction and scoring
        X = extract_features_vectorized(s1_norm, filtered)
        logits = np.dot(X, weights) + bias
        pred_mask = logits >= logit_thresh

        matched_eids = [filtered[i].entity_id for i, m in enumerate(pred_mask) if m]
        results.append((s1_id, cand_eids, sorted(matched_eids)))

    return results


def run_benchmark_matrix(
    n_s1: int = 5000,
    background_cands: int = 150000,
    model_path: str = "business_entity_resolution/src/trained_model.json",
):
    print("=" * 95)
    print("AMAZON ML CHALLENGE 2026 — PHASE 4 BENCHMARK MATRIX")
    print(f"Dataset Sample: {n_s1:,} Source-1 queries | Candidate Pool: {background_cands:,} candidates")
    print("=" * 95)

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
    threshold = float(m_data.get("optimal_threshold", 0.50))

    # 1. Load S1 entities and Ground Truth
    print(f"\n[Step 1] Loading {n_s1:,} S1 queries and ground truth mappings...")
    s1_records = list(stream_tsv_records(s1_file, max_records=n_s1))
    target_s1_ids = {r["entity_id"] for r in s1_records}

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

    print(f"  Loaded {len(s1_records):,} S1 queries with {total_true_pairs:,} true positive match pairs.")

    # 2. Build Candidate Inverted Index
    print(f"\n[Step 2] Building Candidate Index ({background_cands:,} background records + true targets)...")
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

    idx_time = time.time() - t0_idx
    print(f"  Candidate Index built: {len(index):,} candidates in {idx_time:.2f}s.")

    # 3. Define Configurations
    matrix_configs = [
        {
            "id": "A",
            "name": "Current Validated Pipeline",
            "prefilter": "none",
            "capping": None,
            "engine": "numpy",
        },
        {
            "id": "B",
            "name": "DuckDB/Polars Optimization Only",
            "prefilter": "none",
            "capping": None,
            "engine": "polars",
        },
        {
            "id": "C",
            "name": "Cheap Prefilter Only",
            "prefilter": "conservative",
            "capping": 80,
            "engine": "numpy",
        },
        {
            "id": "D",
            "name": "Cheap Prefilter + DuckDB/Polars",
            "prefilter": "conservative",
            "capping": 80,
            "engine": "polars",
        },
    ]

    benchmark_results = []
    proc = psutil.Process()
    batch_size = 1000

    print("\n[Step 3] Executing Benchmark Matrix across Configurations A, B, C, D...")
    print("-" * 95)

    for cfg in matrix_configs:
        gc.collect()
        ram_start = proc.memory_info().rss / (1024 * 1024)
        t_start = time.time()

        all_results = []
        for b_i in range(0, len(s1_records), batch_size):
            batch = s1_records[b_i : b_i + batch_size]
            if cfg["engine"] == "numpy":
                res = process_s1_batch_numpy(
                    batch,
                    index,
                    model,
                    threshold,
                    blocking_strategy="combined",
                    prefilter_config=cfg["prefilter"],
                    max_candidates=cfg["capping"],
                )
            else:
                res = process_s1_batch_polars(
                    batch,
                    index,
                    model,
                    threshold,
                    blocking_strategy="combined",
                    prefilter_config=cfg["prefilter"],
                    max_candidates=cfg["capping"],
                )
            all_results.extend(res)

        elapsed = time.time() - t_start
        ram_end = proc.memory_info().rss / (1024 * 1024)
        peak_ram = max(ram_start, ram_end)

        # Metrics computation
        cand_counts = [len(c_eids) for _, c_eids, _ in all_results]
        zero_cand_count = sum(1 for c in cand_counts if c == 0)

        # Candidate recall
        captured_cands = 0
        pred_map: Dict[str, Set[str]] = {}
        for sid, c_eids, m_eids in all_results:
            true_for_s1 = gt_map.get(sid, set())
            if true_for_s1:
                captured_cands += len(set(c_eids) & true_for_s1)
            pred_map[sid] = set(m_eids)

        cand_recall = captured_cands / max(total_true_pairs, 1)
        macro_f05 = evaluate_predictions(gt_map, pred_map)

        tot_pred = sum(len(p) for p in pred_map.values())
        tot_corr = sum(len(pred_map.get(sid, set()) & g) for sid, g in gt_map.items())
        precision = tot_corr / max(tot_pred, 1)
        overall_recall = tot_corr / max(total_true_pairs, 1)

        throughput = len(s1_records) / max(elapsed, 0.001)

        row = {
            "config_id": cfg["id"],
            "config_name": cfg["name"],
            "engine": cfg["engine"],
            "prefilter": cfg["prefilter"],
            "capping": cfg["capping"],
            "cand_recall": cand_recall,
            "overall_recall": overall_recall,
            "macro_f05": macro_f05,
            "precision": precision,
            "avg_cands": float(np.mean(cand_counts)),
            "med_cands": float(np.median(cand_counts)),
            "p95_cands": float(np.percentile(cand_counts, 95)),
            "max_cands": int(np.max(cand_counts)),
            "zero_cand_s1": zero_cand_count,
            "throughput_s1_sec": throughput,
            "peak_ram_mb": peak_ram,
            "runtime_sec": elapsed,
        }
        benchmark_results.append(row)

        print(f"[{cfg['id']}] {cfg['name']}")
        print(f"    Cand Recall : {cand_recall:7.2%} | Match Recall: {overall_recall:7.2%}")
        print(f"    Macro F0.5  : {macro_f05:.4f}  | Precision   : {precision:.4f}")
        print(f"    Candidates  : Avg={row['avg_cands']:.1f}, Med={row['med_cands']:.1f}, P95={row['p95_cands']:.0f}, Max={row['max_cands']}")
        print(f"    Throughput  : {throughput:6.1f} S1/sec (Total: {elapsed:.2f}s, Peak RAM: {peak_ram:.1f} MB)")
        print("-" * 95)

    # Save results
    out_file = os.path.join(PROJECT_ROOT, "output", "benchmark_matrix_phase4.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    # Print Final Markdown Table
    print("\n" + "=" * 125)
    print("FINAL BENCHMARK MATRIX (PHASE 4 DECISION POINT)")
    print("=" * 125)
    header = f"{'Config':<4} | {'Approach':<35} | {'CandRec':<8} | {'F0.5':<7} | {'Prec':<7} | {'AvgCand':<7} | {'P95':<5} | {'S1/s':<7} | {'Time(s)':<7}"
    print(header)
    print("-" * 125)
    for r in benchmark_results:
        print(f"{r['config_id']:<4} | {r['config_name']:<35} | {r['cand_recall']:7.2%} | {r['macro_f05']:7.4f} | {r['precision']:7.4f} | {r['avg_cands']:7.1f} | {r['p95_cands']:5.0f} | {r['throughput_s1_sec']:7.1f} | {r['runtime_sec']:7.2f}")
    print("=" * 125)
    print(f"Results saved to {out_file}\n")


if __name__ == "__main__":
    run_benchmark_matrix(n_s1=5000, background_cands=150000)
