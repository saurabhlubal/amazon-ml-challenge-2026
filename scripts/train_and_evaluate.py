#!/usr/bin/env python3
"""
End-to-end training, threshold optimization, and Macro F0.5 validation script.

Samples positive pairs, hard negatives (lexical/blocking distractors), and
no-match singletons from competition data; extracts pairwise features; trains
the EntityMatcher; sweeps probability thresholds to maximize Macro F0.5; and
logs comprehensive benchmark metrics.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys
import time
import zipfile
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add workspace root to sys.path so business_entity_resolution is importable
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from business_entity_resolution.src.features import (
    FEATURE_NAMES,
    build_features,
    clean_text,
    get_tokens,
)
from business_entity_resolution.src.model import EntityMatcher, decide_matches
from business_entity_resolution.src.evaluation import (
    evaluate_predictions_detailed,
    find_optimal_threshold,
)


def find_dataset_source(explicit_path: Optional[str] = None) -> Tuple[str, bool]:
    """
    Locate training data directory or zip archive.

    Returns
    -------
    tuple (path, is_zip)
    """
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path, zipfile.is_zipfile(explicit_path)

    # Check local student_resource/dataset/train
    local_dir = os.path.join(WORKSPACE_ROOT, "student_resource", "dataset", "train")
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "train_ground_truth.tsv")):
        return local_dir, False

    # Check downloads for zip file
    downloads_dir = os.path.expanduser("~/Downloads")
    zip_matches = glob.glob(os.path.join(downloads_dir, "*student_resource*.zip"))
    for zp in zip_matches:
        if os.path.isfile(zp) and os.path.getsize(zp) > 100_000_000:
            return zp, True

    raise FileNotFoundError(
        "Could not find training dataset. Please provide path to student_resource/dataset "
        "or student_resource.zip via --data-path."
    )


def stream_tsv_chunks(
    data_source: str,
    is_zip: bool,
    subpath: str,
    chunksize: int = 250_000,
) -> Generator[pd.DataFrame, None, None]:
    """Yield chunks of DataFrame safely keeping handles open."""
    if is_zip:
        with zipfile.ZipFile(data_source, "r") as z:
            with z.open(subpath, "r") as f:
                reader = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)
                for chunk in reader:
                    yield chunk
    else:
        p = os.path.join(data_source, os.path.basename(subpath))
        with open(p, "r", encoding="utf-8") as f:
            reader = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)
            for chunk in reader:
                yield chunk


def load_dataset_slice(
    data_source: str,
    is_zip: bool,
    num_s1: int = 1500,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
    """
    Load a stratified slice of S1 records, candidate records, and ground truth.
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading ground truth from {data_source} (zip={is_zip})...")

    # Read the top slice of ground truth for fast and reliable sampling
    gt_slice_rows = max(10_000, num_s1 * 10)
    if is_zip:
        with zipfile.ZipFile(data_source, "r") as z:
            with z.open("student_resource/dataset/train/train_ground_truth.tsv", "r") as f:
                gt_df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, nrows=gt_slice_rows)
    else:
        gt_path = os.path.join(data_source, "train_ground_truth.tsv")
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, nrows=gt_slice_rows)

    # Stratified sample: include both matched records and singletons
    singletons = gt_df[gt_df["matched_entity_ids"].str.strip() == ""]
    multi_matched = gt_df[gt_df["matched_entity_ids"].str.strip() != ""]

    singleton_frac = 0.08  # ~8% singletons, matching competition distribution
    num_single = int(num_s1 * singleton_frac)
    num_multi = num_s1 - num_single

    sample_single = singletons.sample(n=min(num_single, len(singletons)), random_state=seed)
    sample_multi = multi_matched.sample(n=min(num_multi, len(multi_matched)), random_state=seed)
    sample_gt = pd.concat([sample_multi, sample_single]).sample(frac=1.0, random_state=seed)

    target_s1_ids = set(sample_gt["source1_entity_id"])
    target_cands: Set[str] = set()
    ground_truth: Dict[str, Set[str]] = {}

    for _, row in sample_gt.iterrows():
        s1 = row["source1_entity_id"]
        m_str = str(row["matched_entity_ids"]).strip()
        if m_str:
            cands = {c.strip() for c in m_str.split(",") if c.strip()}
            ground_truth[s1] = cands
            target_cands.update(cands)
        else:
            ground_truth[s1] = set()

    print(f"Selected {len(target_s1_ids):,} S1 entities ({len(sample_single)} singletons).")
    print(f"Total true candidate matches required: {len(target_cands):,}")

    # Now load record dictionaries
    s1_dict: Dict[str, Dict[str, Any]] = {}
    cand_dict: Dict[str, Dict[str, Any]] = {}

    # 1. Load S1 records
    print("Reading Source 1 records...")
    t0 = time.time()
    for chunk in stream_tsv_chunks(data_source, is_zip, "student_resource/dataset/train/train_source1.tsv", chunksize=250_000):
        match_chunk = chunk[chunk["entity_id"].isin(target_s1_ids)]
        for _, r in match_chunk.iterrows():
            s1_dict[r["entity_id"]] = {
                "entity_id": r["entity_id"],
                "business_name": r["business_name"],
                "business_address": r["business_address"],
                "country": r["country"],
            }
        if len(s1_dict) >= len(target_s1_ids):
            break
    print(f"Loaded {len(s1_dict):,} S1 records in {time.time()-t0:.2f}s.")

    # 2. Load S2 and S3 records
    for src_name, subpath in [
        ("Source 2", "student_resource/dataset/train/train_source2.tsv"),
        ("Source 3", "student_resource/dataset/train/train_source3.tsv"),
    ]:
        print(f"Reading {src_name} records...")
        t_src = time.time()
        prefix = "S2-" if "2" in src_name else "S3-"
        needed_src_ids = {cid for cid in target_cands if cid.startswith(prefix)}
        extra_pool_needed = 2500  # Extra pool for negative mining

        for chunk in stream_tsv_chunks(data_source, is_zip, subpath, chunksize=500_000):
            # True matches
            match_chunk = chunk[chunk["entity_id"].isin(needed_src_ids)]
            for _, r in match_chunk.iterrows():
                cand_dict[r["entity_id"]] = {
                    "entity_id": r["entity_id"],
                    "business_name": r["business_name"],
                    "business_address": r["business_address"],
                    "country": r["country"],
                }
                needed_src_ids.discard(r["entity_id"])

            # Background pool of candidates for hard/random negative mining
            if extra_pool_needed > 0:
                sample_pool = chunk.head(min(extra_pool_needed, len(chunk)))
                for _, r in sample_pool.iterrows():
                    cand_dict[r["entity_id"]] = {
                        "entity_id": r["entity_id"],
                        "business_name": r["business_name"],
                        "business_address": r["business_address"],
                        "country": r["country"],
                    }
                extra_pool_needed -= len(sample_pool)

            if len(needed_src_ids) == 0 and extra_pool_needed <= 0:
                break

        print(f"Loaded {src_name} in {time.time()-t_src:.2f}s (remaining missing: {len(needed_src_ids)}).")

    print(f"Total candidate records ready: {len(cand_dict):,}")
    return s1_dict, cand_dict, ground_truth


def build_inverted_index(cand_dict: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build token -> candidate_ids index for hard negative retrieval."""
    index: Dict[str, List[str]] = collections.defaultdict(list)
    common_stops = {"inc", "ltd", "corp", "llc", "co", "pvt", "limited", "the", "and", "of"}

    for cid, rec in cand_dict.items():
        name_clean = clean_text(rec.get("business_name"))
        tokens = [t for t in get_tokens(name_clean) if len(t) >= 3 and t not in common_stops]
        for t in set(tokens):
            index[t].append(cid)

    return index


def generate_pairs_with_negatives(
    s1_ids: List[str],
    s1_dict: Dict[str, Dict[str, Any]],
    cand_dict: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    token_index: Dict[str, List[str]],
    negatives_per_positive: int = 3,
    singletons_negatives: int = 4,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str, int]], Dict[str, List[str]]]:
    """
    Construct labeled candidate pairs with positive pairs, hard token-collision
    negatives, country-matched negatives, and singleton distractor pairs.
    """
    rng = random.Random(seed)
    pairs: List[Tuple[str, str, int]] = []
    candidates_by_s1: Dict[str, List[str]] = collections.defaultdict(list)
    all_cand_ids = list(cand_dict.keys())

    for s1_id in s1_ids:
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue

        true_matches = ground_truth.get(s1_id, set())
        # 1. Positives
        for cid in true_matches:
            if cid in cand_dict:
                pairs.append((s1_id, cid, 1))
                candidates_by_s1[s1_id].append(cid)

        # 2. Hard Negatives mining (token collision but not true match)
        s1_name_clean = clean_text(s1_rec.get("business_name"))
        tokens = [t for t in get_tokens(s1_name_clean) if len(t) >= 3]

        hard_candidates: Set[str] = set()
        for t in tokens:
            matching_cids = token_index.get(t, [])
            for mcid in matching_cids:
                if mcid not in true_matches:
                    hard_candidates.add(mcid)
                if len(hard_candidates) >= 15:
                    break
            if len(hard_candidates) >= 15:
                break

        # Number of negatives to add
        target_num_negs = len(true_matches) * negatives_per_positive if true_matches else singletons_negatives
        selected_negs: List[str] = []

        if hard_candidates:
            selected_negs.extend(rng.sample(list(hard_candidates), min(len(hard_candidates), target_num_negs)))

        # Fill remaining with random negatives from candidate pool
        if len(selected_negs) < target_num_negs:
            needed = target_num_negs - len(selected_negs)
            sampled = rng.sample(all_cand_ids, min(needed * 2, len(all_cand_ids)))
            for rand_cid in sampled:
                if rand_cid not in true_matches and rand_cid not in selected_negs:
                    selected_negs.append(rand_cid)
                if len(selected_negs) >= target_num_negs:
                    break

        for neg_cid in selected_negs:
            pairs.append((s1_id, neg_cid, 0))
            candidates_by_s1[s1_id].append(neg_cid)

    rng.shuffle(pairs)
    return pairs, candidates_by_s1


def run_experiment(
    num_s1: int = 1500,
    algorithm: str = "auto",
    data_path: Optional[str] = None,
    output_dir: str = "experiments/results",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run complete ML matching experiment and return detailed report.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_total_time = time.time()

    # Step 1: Locate and load data
    data_source, is_zip = find_dataset_source(data_path)
    s1_dict, cand_dict, ground_truth = load_dataset_slice(data_source, is_zip, num_s1=num_s1, seed=seed)

    # Step 2: Train/Validation split on S1 entities (prevents entity data leakage)
    all_s1_list = sorted(list(s1_dict.keys()))
    rng = random.Random(seed)
    rng.shuffle(all_s1_list)

    val_fraction = 0.30
    val_size = int(len(all_s1_list) * val_fraction)
    val_s1_ids = all_s1_list[:val_size]
    train_s1_ids = all_s1_list[val_size:]

    print(f"\nEntity Split: {len(train_s1_ids):,} Train S1, {len(val_s1_ids):,} Validation S1.")

    # Step 3: Candidate pairs generation & Hard negative mining
    print("Building inverted token index for hard negative mining...")
    token_index = build_inverted_index(cand_dict)

    print("Generating train candidate pairs (positives + hard negatives)...")
    train_pairs, _ = generate_pairs_with_negatives(
        train_s1_ids, s1_dict, cand_dict, ground_truth, token_index,
        negatives_per_positive=3, singletons_negatives=4, seed=seed,
    )
    pos_train = sum(1 for _, _, y in train_pairs if y == 1)
    neg_train = len(train_pairs) - pos_train
    print(f"Train Pairs: {len(train_pairs):,} total ({pos_train:,} positive, {neg_train:,} negative).")

    print("Generating validation candidate pairs...")
    val_pairs, _ = generate_pairs_with_negatives(
        val_s1_ids, s1_dict, cand_dict, ground_truth, token_index,
        negatives_per_positive=4, singletons_negatives=4, seed=seed + 1,
    )
    pos_val = sum(1 for _, _, y in val_pairs if y == 1)
    neg_val = len(val_pairs) - pos_val
    print(f"Validation Pairs: {len(val_pairs):,} total ({pos_val:,} positive, {neg_val:,} negative).")

    # Step 4: Pairwise Feature Extraction
    print(f"\nExtracting {len(FEATURE_NAMES)} pairwise features for Train pairs...")
    t_feat_start = time.time()
    train_feat_records = [
        build_features(s1_dict[s1], cand_dict[cid]) for s1, cid, _ in train_pairs
    ]
    X_train = pd.DataFrame(train_feat_records, columns=FEATURE_NAMES).fillna(0.0)
    y_train = np.array([y for _, _, y in train_pairs], dtype=np.int32)
    feat_train_time = time.time() - t_feat_start
    print(f"Train feature extraction completed in {feat_train_time:.2f}s ({len(train_pairs)/feat_train_time:.0f} pairs/sec).")

    print(f"Extracting pairwise features for Validation pairs...")
    t_val_feat_start = time.time()
    val_feat_records = [
        build_features(s1_dict[s1], cand_dict[cid]) for s1, cid, _ in val_pairs
    ]
    X_val = pd.DataFrame(val_feat_records, columns=FEATURE_NAMES).fillna(0.0)
    y_val = np.array([y for _, _, y in val_pairs], dtype=np.int32)
    feat_val_time = time.time() - t_val_feat_start
    print(f"Validation feature extraction completed in {feat_val_time:.2f}s.")

    # Step 5: Train Model
    print(f"\nTraining EntityMatcher (algorithm={algorithm})...")
    t_train_start = time.time()
    matcher = EntityMatcher(
        algorithm=algorithm,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=seed,
    )
    matcher.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=25)
    train_time = time.time() - t_train_start
    print(f"Model trained successfully in {train_time:.2f}s (backend: {matcher.backend}).")

    # Step 6: Inference on Validation
    print("Running inference on validation pairs...")
    t_inf_start = time.time()
    val_probs = matcher.predict_proba(X_val)
    inference_time = time.time() - t_inf_start
    print(f"Inference completed in {inference_time:.3f}s ({len(val_pairs)/inference_time:.0f} pairs/sec).")

    # Step 7: Threshold Sweep for Macro F0.5 Optimization
    print("\nSweeping decision thresholds [0.20 -> 0.95] for Macro F0.5...")
    val_gt = {s1: ground_truth[s1] for s1 in val_s1_ids}
    val_s1_col = [s1 for s1, _, _ in val_pairs]
    val_cand_col = [cid for _, cid, _ in val_pairs]

    sweep_res = find_optimal_threshold(
        val_s1_col,
        val_cand_col,
        val_probs,
        val_gt,
        threshold_range=(0.20, 0.95, 0.02),
    )

    best_thresh = float(sweep_res["best_threshold"])
    best_f05 = float(sweep_res["best_f05"])
    best_prec = float(sweep_res["best_precision"])
    best_rec = float(sweep_res["best_recall"])

    print("=" * 60)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"  Optimal Probability Threshold : {best_thresh:.2f}")
    print(f"  Validation Macro F0.5        : {best_f05:.4f}")
    print(f"  Macro Precision               : {best_prec:.4f}")
    print(f"  Macro Recall                  : {best_rec:.4f}")

    # Baseline comparison at default 0.50 threshold
    pred_05 = {}
    for s1 in val_s1_ids:
        cands = [
            cid for (s, cid, _), p in zip(val_pairs, val_probs)
            if s == s1 and p >= 0.50
        ]
        pred_05[s1] = set(cands)
    metrics_05 = evaluate_predictions_detailed(val_gt, pred_05)
    print(f"\n  Baseline (threshold=0.50)    : F0.5={metrics_05['macro_f05']:.4f} "
          f"(Precision={metrics_05['macro_precision']:.4f}, Recall={metrics_05['macro_recall']:.4f})")
    print(f"  Gain from threshold tuning   : +{best_f05 - metrics_05['macro_f05']:.4f} F0.5")

    # Save model artifact
    model_save_path = os.path.join(output_dir, "entity_matcher.pkl")
    matcher.save(model_save_path)
    print(f"\nModel artifact saved to: {model_save_path}")

    # Top feature importances if LightGBM
    top_features = []
    if matcher.backend == "lightgbm" and hasattr(matcher.model, "feature_importances_"):
        importances = matcher.model.feature_importances_
        feature_ranking = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Most Predictive Matching Features:")
        for rank, (fname, imp) in enumerate(feature_ranking[:10], 1):
            print(f"  {rank:2d}. {fname:<30}: {imp}")
            top_features.append({"feature": fname, "importance": int(imp)})

    total_time = time.time() - start_total_time

    # Step 8: Build Experiment Summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_backend": matcher.backend,
        "features_count": len(FEATURE_NAMES),
        "train_s1_count": len(train_s1_ids),
        "val_s1_count": len(val_s1_ids),
        "train_pairs_count": len(train_pairs),
        "val_pairs_count": len(val_pairs),
        "sampling_strategy": "True positives + token-collision hard negatives + random negatives + singletons",
        "best_threshold": best_thresh,
        "validation_macro_f05": best_f05,
        "macro_precision": best_prec,
        "macro_recall": best_rec,
        "baseline_f05_at_05": metrics_05["macro_f05"],
        "runtime_seconds": {
            "feature_extraction_train": round(feat_train_time, 2),
            "feature_extraction_val": round(feat_val_time, 2),
            "training": round(train_time, 2),
            "inference": round(inference_time, 3),
            "total": round(total_time, 2),
        },
        "top_features": top_features,
    }

    # Save experiment results json
    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Append to experiment log markdown
    log_path = os.path.join(WORKSPACE_ROOT, "experiments", "experiment_log.md")
    log_entry = (
        f"| E01 | {summary['timestamp']} | `feature/ml-matching` | LightGBM | "
        f"{len(FEATURE_NAMES)} signals | Thresh={best_thresh:.2f} | "
        f"**{best_f05:.4f}** | Prec={best_prec:.4f}, Rec={best_rec:.4f} | "
        f"Pos + Hard Negatives + Singleton Distractors | Keep |\n"
    )

    if not os.path.exists(log_path):
        header = (
            "# Amazon ML Challenge 2026 — Experiment Log\n\n"
            "| Exp ID | Date/Time | Branch | Model | Features | Threshold | Local Macro F0.5 | Precision/Recall | Sampling Strategy | Status |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n"
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header + log_entry)
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    print(f"\nExperiment logged to: {log_path}")
    print(f"Total experiment time: {total_time:.2f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate entity matcher with threshold tuning.")
    parser.add_argument("--num-s1", type=int, default=1500, help="Number of S1 entities to sample (default: 1500)")
    parser.add_argument("--algorithm", type=str, default="auto", choices=["auto", "lightgbm", "hist_gb", "random_forest"])
    parser.add_argument("--data-path", type=str, default=None, help="Path to student_resource/dataset or zip archive")
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    args = parser.parse_args()

    run_experiment(
        num_s1=args.num_s1,
        algorithm=args.algorithm,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
