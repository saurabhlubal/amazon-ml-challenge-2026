"""
Experiment Runner for Business Entity Resolution Baseline.
Evaluates multi-key blocking, feature extraction, model training,
threshold tuning, and exact Macro F0.5 evaluation on a real training subset.
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
from business_entity_resolution.src.features import build_features
from business_entity_resolution.src.model import train_model, predict_scores, decide_matches
from business_entity_resolution.src.evaluation import f05, evaluate_predictions, to_id_set


def run_baseline_experiment(
    n_s1_sample: int = 5000,
    background_s2_s3_records: int = 50000,
    train_ratio: float = 0.7,
):
    print("=" * 70)
    print("RUNNING BASELINE ENTITY RESOLUTION EXPERIMENT")
    print("=" * 70)
    t_start = time.time()

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

    # Fill in any missing S1 as empty set (singletons)
    for s1_id in target_s1_ids:
        if s1_id not in gt_map:
            gt_map[s1_id] = set()

    n_singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    print(f"Target Ground Truth loaded:")
    print(f"  Total S1 entities   : {len(target_s1_ids):,}")
    print(f"  True matched pairs  : {total_true_positives:,} (avg {total_true_positives / len(target_s1_ids):.2f}/S1)")
    print(f"  True singletons     : {n_singletons:,} ({n_singletons / len(target_s1_ids) * 100:.1f}%)")
    print(f"  Target S2/S3 IDs    : {len(needed_target_cands):,}")

    # 2. Build candidate pool: needed target candidates + background sample
    print(f"\nBuilding candidate pool (True match IDs + {background_s2_s3_records:,} background records)...")
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

    # 3. Build Blocking Index
    t_block_build = time.time()
    print("Building multi-key inverted index on candidate pool...")
    indexes = build_blocking_indexes(candidate_pool, max_bucket_size=500, store_records=True)
    print(f"Index built in {time.time() - t_block_build:.2f}s ({len(indexes['index']):,} unique blocking keys).")

    # 4. Generate candidates for all S1 records & Measure Blocking Performance
    print("\nRunning candidate generation (blocking) on Source 1 records...")
    t_cand_gen = time.time()
    s1_candidates: Dict[str, Set[str]] = {}
    candidate_counts: List[int] = []
    captured_true_matches = 0

    for s1_rec in s1_records:
        s1_id = s1_rec["entity_id"]
        cands = generate_candidates(s1_rec, indexes)
        s1_candidates[s1_id] = cands
        candidate_counts.append(len(cands))

        true_matches = gt_map.get(s1_id, set())
        captured_true_matches += len(cands & true_matches)

    t_cand_gen_elapsed = time.time() - t_cand_gen
    total_cands = sum(candidate_counts)
    avg_cands = np.mean(candidate_counts)
    p95_cands = float(np.percentile(candidate_counts, 95))
    max_cands = max(candidate_counts)
    blocking_recall = (captured_true_matches / total_true_positives) if total_true_positives > 0 else 0.0

    print("-" * 70)
    print("BLOCKING (CANDIDATE GENERATION) METRICS")
    print("-" * 70)
    print(f"Total candidate pairs generated : {total_cands:,}")
    print(f"Average candidates / S1        : {avg_cands:.2f}")
    print(f"P95 candidates / S1            : {p95_cands:.1f}")
    print(f"Max candidates / S1            : {max_cands}")
    print(f"Candidate Recall (Ceiling)     : {blocking_recall:.4f} ({captured_true_matches:,}/{total_true_positives:,} true matches captured)")
    print(f"Blocking runtime               : {t_cand_gen_elapsed:.2f}s ({len(s1_records) / t_cand_gen_elapsed:.1f} S1/sec)")

    # 5. Split into Train & Validation sets
    n_train = int(len(s1_records) * train_ratio)
    train_s1 = s1_records[:n_train]
    val_s1 = s1_records[n_train:]

    print(f"\nDataset Split: Train={len(train_s1):,} S1, Validation={len(val_s1):,} S1")

    # 6. Feature extraction for Training Pairs
    print("Building training pairs and features...")
    t_feat = time.time()
    records_store = indexes["records"]

    X_train: List[Dict[str, float]] = []
    y_train: List[int] = []
    pos_pairs = 0
    neg_pairs = 0

    for s1_rec in train_s1:
        s1_id = s1_rec["entity_id"]
        true_matches = gt_map.get(s1_id, set())
        candidates = s1_candidates.get(s1_id, set())

        # Positives
        for mid in true_matches:
            cand_rec = records_store.get(mid)
            if cand_rec:
                X_train.append(build_features(s1_rec, cand_rec))
                y_train.append(1)
                pos_pairs += 1

        # Hard negatives from candidates (cap negatives per entity to 5 to avoid extreme class imbalance)
        negs = list(candidates - true_matches)[:5]
        for nid in negs:
            cand_rec = records_store.get(nid)
            if cand_rec:
                X_train.append(build_features(s1_rec, cand_rec))
                y_train.append(0)
                neg_pairs += 1

    print(f"Training pairs built in {time.time() - t_feat:.2f}s:")
    print(f"  Positive pairs : {pos_pairs:,}")
    print(f"  Negative pairs : {neg_pairs:,}")
    print(f"  Total pairs    : {len(X_train):,}")

    # 7. Model Training
    t_train = time.time()
    print("Training baseline matching model...")
    model = train_model(X_train, y_train)
    t_train_elapsed = time.time() - t_train
    print(f"Model trained in {t_train_elapsed:.2f}s ({type(model).__name__}).")

    # 8. Validation & Threshold Tuning
    print("\nEvaluating on Validation Set across decision thresholds...")
    val_gt = {rec["entity_id"]: gt_map[rec["entity_id"]] for rec in val_s1}

    # Pre-compute candidate features and scores for validation
    val_candidate_data = []
    for s1_rec in val_s1:
        s1_id = s1_rec["entity_id"]
        cand_list = sorted(list(s1_candidates.get(s1_id, set())))
        if not cand_list:
            val_candidate_data.append((s1_id, [], []))
            continue
        pairs_X = [build_features(s1_rec, records_store.get(cid, {"entity_id": cid})) for cid in cand_list]
        scores = predict_scores(model, pairs_X)
        val_candidate_data.append((s1_id, cand_list, scores))

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    best_thresh = 0.5
    best_macro_f05 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    print("-" * 70)
    print(f"{'Threshold':<10} {'Macro F0.5':<12} {'Precision':<12} {'Recall':<12} {'Total Matches'}")
    print("-" * 70)

    for thresh in thresholds:
        val_predictions: Dict[str, Set[str]] = {}
        total_pred_matches = 0
        all_tp = 0
        all_fp = 0
        all_fn = 0

        for s1_id, cand_list, scores in val_candidate_data:
            matches = decide_matches(cand_list, scores, threshold=thresh)
            val_predictions[s1_id] = set(matches)
            total_pred_matches += len(matches)

            true_set = val_gt.get(s1_id, set())
            pred_set = set(matches)
            tp = len(true_set & pred_set)
            all_tp += tp
            all_fp += (len(pred_set) - tp)
            all_fn += (len(true_set) - tp)

        macro_score = evaluate_predictions(val_gt, val_predictions)
        overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        overall_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0

        marker = " *" if macro_score > best_macro_f05 else ""
        print(f"{thresh:<10.2f} {macro_score:<12.4f} {overall_prec:<12.4f} {overall_rec:<12.4f} {total_pred_matches:<12,}{marker}")

        if macro_score > best_macro_f05:
            best_macro_f05 = macro_score
            best_thresh = thresh
            best_prec = overall_prec
            best_rec = overall_rec

    print("-" * 70)
    print(f"Optimal Threshold        : {best_thresh}")
    print(f"Best Macro F0.5          : {best_macro_f05:.4f}")
    print(f"Precision at Best        : {best_prec:.4f}")
    print(f"Recall at Best           : {best_rec:.4f}")
    print(f"Total Experiment Runtime : {time.time() - t_start:.2f} seconds")
    print("=" * 70)

    return {
        "n_s1_records": len(s1_records),
        "total_true_positives": total_true_positives,
        "pos_pairs": pos_pairs,
        "neg_pairs": neg_pairs,
        "total_cands": total_cands,
        "avg_cands": avg_cands,
        "p95_cands": p95_cands,
        "blocking_recall": blocking_recall,
        "optimal_threshold": best_thresh,
        "macro_f05": best_macro_f05,
        "precision": best_prec,
        "recall": best_rec,
        "model_name": type(model).__name__,
        "runtime": time.time() - t_start
    }


if __name__ == "__main__":
    run_baseline_experiment(n_s1_sample=5000, background_s2_s3_records=50000)
