"""
Model training and threshold selection script for Business Entity Resolution.
Trains FastLogisticRegression using the improved combined blocker on 5,000 S1 sample.
Tests thresholds [0.45, 0.50, 0.55, 0.60] and saves the optimal model and threshold to disk.
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, Set, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.blocking import build_blocking_indexes, generate_candidates
from business_entity_resolution.src.features import build_features
from business_entity_resolution.src.model import (
    train_model,
    predict_scores,
    decide_matches,
    FEATURE_NAMES,
    FastLogisticRegression,
)
from business_entity_resolution.src.evaluation import f05, evaluate_predictions, to_id_set


def train_and_select_threshold(n_s1_sample: int = 5000, background_records: int = 50000):
    print("=" * 70)
    print("TRAINING MODEL & SELECTING OPTIMAL THRESHOLD (COMBINED BLOCKER)")
    print("=" * 70)

    dataset_train_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
    s1_file = os.path.join(dataset_train_dir, "train_source1.tsv")
    s2_file = os.path.join(dataset_train_dir, "train_source2.tsv")
    s3_file = os.path.join(dataset_train_dir, "train_source3.tsv")
    gt_file = os.path.join(dataset_train_dir, "train_ground_truth.tsv")

    # 1. Load S1 records and Ground Truth
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

    # 2. Build candidate pool
    print(f"Building candidate pool (True targets + {background_records:,} background records)...")
    candidate_pool: List[Dict[str, str]] = []
    bg_count = 0
    for path in (s2_file, s3_file):
        for rec in stream_tsv_records(path):
            eid = rec.get("entity_id", "")
            if eid in needed_target_cands:
                candidate_pool.append(rec)
            elif bg_count < background_records:
                candidate_pool.append(rec)
                bg_count += 1

    print(f"Candidate pool built: {len(candidate_pool):,} records.")

    # 3. Build Combined Blocking Index
    print("Building Combined multi-key blocking index...")
    indexes = build_blocking_indexes(candidate_pool, max_bucket_size=500, strategy="combined", store_records=True)

    # 4. Generate candidates
    print("Generating candidates for S1 sample...")
    s1_candidates: Dict[str, Set[str]] = {}
    captured = 0
    for s1_rec in s1_records:
        s1_id = s1_rec["entity_id"]
        cands = generate_candidates(s1_rec, indexes)
        s1_candidates[s1_id] = cands
        captured += len(cands & gt_map[s1_id])

    blocking_recall = captured / total_true_positives
    print(f"Candidate Recall: {blocking_recall:.4f} ({captured:,}/{total_true_positives:,} true matches captured)")

    # 5. Split Train (70%) and Validation (30%)
    n_train = int(len(s1_records) * 0.7)
    train_s1 = s1_records[:n_train]
    val_s1 = s1_records[n_train:]
    records_store = indexes["records"]

    print(f"\nBuilding training pair features (Train={len(train_s1):,} S1)...")
    X_train: List[Dict[str, float]] = []
    y_train: List[int] = []

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

        # Hard negatives (subsampled up to 5 per entity)
        negs = list(candidates - true_matches)[:5]
        for nid in negs:
            cand_rec = records_store.get(nid)
            if cand_rec:
                X_train.append(build_features(s1_rec, cand_rec))
                y_train.append(0)

    print(f"Training pairs: {len(X_train):,} ({sum(y_train):,} positives, {len(y_train) - sum(y_train):,} negatives)")

    # 6. Train model
    print("Fitting FastLogisticRegression model...")
    t_train = time.time()
    model = train_model(X_train, y_train)
    print(f"Model trained in {time.time() - t_train:.2f}s.")

    # 7. Evaluate requested thresholds: [0.45, 0.50, 0.55, 0.60]
    val_gt = {rec["entity_id"]: gt_map[rec["entity_id"]] for rec in val_s1}
    val_pairs_data = []

    print(f"Scoring validation pairs (Val={len(val_s1):,} S1)...")
    for s1_rec in val_s1:
        s1_id = s1_rec["entity_id"]
        cands = sorted(list(s1_candidates.get(s1_id, set())))
        if not cands:
            val_pairs_data.append((s1_id, [], []))
            continue
        X_val = [build_features(s1_rec, records_store.get(cid, {"entity_id": cid})) for cid in cands]
        scores = predict_scores(model, X_val)
        val_pairs_data.append((s1_id, cands, scores))

    target_thresholds = [0.45, 0.50, 0.55, 0.60]
    best_thresh = 0.50
    best_f05 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    print("-" * 70)
    print(f"{'Threshold':<12} {'Macro F0.5':<14} {'Precision':<14} {'Recall':<14} {'Predicted Matches'}")
    print("-" * 70)

    for thresh in target_thresholds:
        val_pred = {}
        total_pred = 0
        all_tp = all_fp = all_fn = 0

        for s1_id, cands, scores in val_pairs_data:
            matches = decide_matches(cands, scores, threshold=thresh)
            val_pred[s1_id] = set(matches)
            total_pred += len(matches)

            true_set = val_gt.get(s1_id, set())
            pred_set = set(matches)
            tp = len(true_set & pred_set)
            all_tp += tp
            all_fp += (len(pred_set) - tp)
            all_fn += (len(true_set) - tp)

        macro_score = evaluate_predictions(val_gt, val_pred)
        prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0

        marker = " *" if macro_score > best_f05 else ""
        print(f"{thresh:<12.2f} {macro_score:<14.4f} {prec:<14.4f} {rec:<14.4f} {total_pred:<14,}{marker}")

        if macro_score > best_f05:
            best_f05 = macro_score
            best_thresh = thresh
            best_prec = prec
            best_rec = rec

    print("-" * 70)
    print(f"Optimal Threshold Selected : {best_thresh}")
    print(f"Macro F0.5 at Optimal      : {best_f05:.4f}")
    print(f"Precision at Optimal       : {best_prec:.4f}")
    print(f"Recall at Optimal          : {best_rec:.4f}")

    # 8. Save model weights and configuration
    weights = model.coef_[0].tolist() if hasattr(model, "coef_") else model.weights.tolist()
    bias = float(model.intercept_[0]) if hasattr(model, "intercept_") else float(model.bias)

    model_config = {
        "model_type": "FastLogisticRegression",
        "weights": weights,
        "bias": bias,
        "feature_names": FEATURE_NAMES,
        "optimal_threshold": best_thresh,
        "macro_f05": best_f05,
        "precision": best_prec,
        "recall": best_rec,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    model_save_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")
    with open(model_save_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    print(f"\nModel artifact saved to: {model_save_path}")
    print("=" * 70)
    return model_config


if __name__ == "__main__":
    train_and_select_threshold(n_s1_sample=5000, background_records=50000)
