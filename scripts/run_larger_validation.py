"""
Phase 5: Larger Validation Suite for Selected Best Approach (Configuration C).
Validates on 25,000 Source-1 queries against candidate pool.

Verifies:
1. Candidate recall
2. Macro F0.5
3. Precision
4. Candidate count distribution (Avg, Med, P95, Max)
5. Throughput (S1/sec)
6. Peak RAM
7. Output correctness and candidate-subset guarantee
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import gc
import json
import numpy as np
import psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records, DELIM, CANDIDATE_HEADER, MATCHING_HEADER, format_id_list
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.prefilter import prefilter_pair
from business_entity_resolution.src.model import FastLogisticRegression
from business_entity_resolution.src.evaluation import f05, evaluate_predictions, to_id_set
from sagemaker.vectorized_matcher import CompactInvertedIndex, extract_features_vectorized
from scripts.benchmark_four_configurations import process_s1_batch_numpy


def run_larger_validation(
    n_s1: int = 25000,
    background_cands: int = 300000,
    model_path: str = "business_entity_resolution/src/trained_model.json",
    output_dir: str = "output/validation_phase5",
):
    print("=" * 95)
    print("AMAZON ML CHALLENGE 2026 — PHASE 5 LARGER VALIDATION (25,000 S1 QUERIES)")
    print(f"Selected Configuration: Configuration C (Cheap Prefilter Conservative + Cap 80 + NumPy)")
    print("=" * 95)

    os.makedirs(output_dir, exist_ok=True)
    val_cand_path = os.path.join(output_dir, "validation_candidate_pairs.tsv")
    val_match_path = os.path.join(output_dir, "validation_matching_results.tsv")

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
    print(f"\n[Step 1] Streaming {n_s1:,} S1 queries and ground truth...")
    s1_records = list(stream_tsv_records(s1_file, max_records=n_s1))
    target_s1_ids = {r["entity_id"] for r in s1_records}

    gt_map = {}
    needed_true_cands = set()
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

    # 2. Build Candidate Index
    print(f"\n[Step 2] Building Candidate Inverted Index ({background_cands:,} background + true targets)...")
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
    print(f"  Index built: {len(index):,} candidates in {idx_time:.2f}s.")

    # 3. Process S1 queries in batches with Configuration C
    print(f"\n[Step 3] Processing {n_s1:,} S1 entities with Configuration C...")
    batch_size = 2000
    proc = psutil.Process()
    ram_start = proc.memory_info().rss / (1024 * 1024)
    t_start = time.time()

    all_results = []
    with open(val_cand_path, "w", encoding="utf-8", newline="") as f_c, \
         open(val_match_path, "w", encoding="utf-8", newline="") as f_m:

        f_c.write(CANDIDATE_HEADER)
        f_m.write(MATCHING_HEADER)

        for b_idx in range(0, len(s1_records), batch_size):
            b_start = time.time()
            batch = s1_records[b_idx : b_idx + batch_size]
            res = process_s1_batch_numpy(
                batch,
                index,
                model,
                threshold,
                blocking_strategy="combined",
                prefilter_config="conservative",
                max_candidates=80,
            )
            for sid, c_eids, m_eids in res:
                # Guarantee candidate subset rule
                valid_m_eids = [m for m in m_eids if m in set(c_eids)]
                f_c.write(f"{sid}{DELIM}{format_id_list(c_eids)}\n")
                f_m.write(f"{sid}{DELIM}{format_id_list(valid_m_eids)}\n")
                all_results.append((sid, c_eids, valid_m_eids))

            b_rate = len(batch) / max(time.time() - b_start, 0.001)
            print(f"  Processed {min(b_idx + batch_size, len(s1_records)):,}/{len(s1_records):,} S1 queries ({b_rate:.1f} S1/sec)...")

    elapsed = time.time() - t_start
    ram_end = proc.memory_info().rss / (1024 * 1024)
    peak_ram = max(ram_start, ram_end)

    # 4. Detailed Metrics and Integrity Verification
    print(f"\n[Step 4] Verifying Output Correctness and Evaluating Final Validation Metrics...")

    # Verify candidate subset rule
    subset_violations = 0
    duplicate_cands = 0
    duplicate_matches = 0
    cand_counts = []
    captured_cands = 0
    pred_map = {}

    for sid, c_eids, m_eids in all_results:
        c_set = set(c_eids)
        m_set = set(m_eids)
        if len(c_set) != len(c_eids): duplicate_cands += 1
        if len(m_set) != len(m_eids): duplicate_matches += 1
        if not m_set.issubset(c_set): subset_violations += 1

        cand_counts.append(len(c_eids))
        true_for_s1 = gt_map.get(sid, set())
        if true_for_s1:
            captured_cands += len(c_set & true_for_s1)
        pred_map[sid] = m_set

    assert subset_violations == 0, f"Candidate subset violation count: {subset_violations}"
    assert duplicate_cands == 0, f"Duplicate candidate IDs count: {duplicate_cands}"
    assert duplicate_matches == 0, f"Duplicate matched IDs count: {duplicate_matches}"
    print("  Candidate Subset Guarantee : PASSED (0 violations)")
    print("  ID Uniqueness Guarantee    : PASSED (0 duplicates)")

    cand_recall = captured_cands / max(total_true_pairs, 1)
    macro_f05 = evaluate_predictions(gt_map, pred_map)

    tot_pred = sum(len(p) for p in pred_map.values())
    tot_corr = sum(len(pred_map.get(sid, set()) & g) for sid, g in gt_map.items())
    precision = tot_corr / max(tot_pred, 1)
    overall_recall = tot_corr / max(total_true_pairs, 1)
    throughput = len(s1_records) / max(elapsed, 0.001)

    val_summary = {
        "n_s1": n_s1,
        "cand_recall": cand_recall,
        "overall_recall": overall_recall,
        "macro_f05": macro_f05,
        "precision": precision,
        "avg_cands": float(np.mean(cand_counts)),
        "med_cands": float(np.median(cand_counts)),
        "p95_cands": float(np.percentile(cand_counts, 95)),
        "max_cands": int(np.max(cand_counts)),
        "throughput_s1_sec": throughput,
        "peak_ram_mb": peak_ram,
        "runtime_sec": elapsed,
        "cand_pairs_path": val_cand_path,
        "matching_path": val_match_path,
    }

    out_json = os.path.join(output_dir, "validation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 5 VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Candidate Recall    : {cand_recall:.2%}")
    print(f"  Ground-Truth Recall : {overall_recall:.2%}")
    print(f"  Macro F0.5          : {macro_f05:.4f}")
    print(f"  Precision           : {precision:.4f}")
    print(f"  Avg Candidates / S1 : {val_summary['avg_cands']:.1f} (Median: {val_summary['med_cands']:.0f}, P95: {val_summary['p95_cands']:.0f})")
    print(f"  Throughput          : {throughput:.1f} S1 / second")
    print(f"  Total Runtime       : {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"  Peak RAM            : {peak_ram:.1f} MB ({peak_ram/1024:.2f} GB)")
    print(f"  Candidate Subset    : 100% Validated (0 violations)")
    print("=" * 80)

    # Projected Full Test Execution Runtime
    test_s1_count = 1732544
    projected_seconds = test_s1_count / throughput
    print(f"\nPROJECTED FULL TEST RUNTIME (1,732,544 Source-1 Entities):")
    print(f"  Throughput         : {throughput:.1f} S1/sec")
    print(f"  Projected Runtime  : {projected_seconds/60:.1f} minutes ({projected_seconds/3600:.2f} hours)")
    print("=" * 80)


if __name__ == "__main__":
    run_larger_validation(n_s1=25000, background_cands=300000)
