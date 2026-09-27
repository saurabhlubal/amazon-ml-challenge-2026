import os
import sys
import json
import time
import numpy as np

PROJECT_ROOT = r"g:\Amazon ML 2026\amazon-ml-challenge-2026"
sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.prefilter import prefilter_pair
from business_entity_resolution.src.evaluation import f05, evaluate_predictions, to_id_set
from sagemaker.vectorized_matcher import CompactInvertedIndex, extract_features_vectorized

def run_diagnostic():
    train_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # Load 2,000 US S1 records
    print("Loading 2,000 US S1 records from train...")
    s1_records = []
    s1_ids = set()
    for rec in stream_tsv_records(s1_path):
        if rec.get("country", "").upper() == "US":
            s1_records.append(rec)
            s1_ids.add(rec["entity_id"])
            if len(s1_records) >= 2000:
                break

    print(f"Loaded {len(s1_records)} S1 records.")

    # Load ground truth for these S1
    gt_map = {}
    for rec in stream_tsv_records(gt_path):
        sid = rec.get("source1_entity_id", "")
        if sid in s1_ids:
            gt_map[sid] = to_id_set(rec.get("matched_entity_ids", ""))
    for sid in s1_ids:
        if sid not in gt_map:
            gt_map[sid] = set()

    # Load model
    model_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    weights = np.array(m_data["weights"], dtype=np.float32)
    bias = float(m_data["bias"])
    threshold = float(m_data.get("optimal_threshold", 0.50))
    logit_thresh = np.log(threshold / (1.0 - threshold)) if (0.0 < threshold < 1.0) else 0.0

    # Build index with 500,000 US records from S2 and S3 (REAL uncurated candidate pool!)
    print("Indexing 500,000 US candidates from S2 and S3 (uncurated)...")
    index = CompactInvertedIndex(max_bucket_size=300)
    count = 0
    for path in (s2_path, s3_path):
        for rec in stream_tsv_records(path):
            if rec.get("country", "").upper() == "US":
                index.add_record(rec, strategy="combined")
                count += 1
                if count >= 500000:
                    break
        if count >= 500000:
            break
    print(f"Index built with {len(index):,} candidates.")

    # Evaluate current pipeline logic
    s1_norm_list = [normalize_record(r) for r in s1_records]
    preds = {}
    cand_counts = []
    match_counts = []

    for s1_norm in s1_norm_list:
        s1_id = s1_norm["entity_id"]
        cand_indices = index.get_candidate_indices(s1_norm, strategy="combined")
        if not cand_indices:
            preds[s1_id] = set()
            cand_counts.append(0)
            match_counts.append(0)
            continue

        cand_records = [index.cand_records[ci] for ci in cand_indices]
        filtered = [c for c in cand_records if prefilter_pair(s1_norm, c, config="conservative")]

        if len(filtered) > 80:
            s1_nt = s1_norm["name_tokens_set"]
            s1_at = s1_norm["address_tokens_set"]
            s1_name = s1_norm["business_name"]
            def cs(c):
                return (10 if s1_name == c.business_name else 0) + len(s1_nt & c.name_tokens_set) * 3 + len(s1_at & c.address_tokens_set) * 2
            filtered.sort(key=cs, reverse=True)
            filtered = filtered[:80]

        cand_eids = sorted([c.entity_id for c in filtered])
        cand_counts.append(len(cand_eids))

        X = extract_features_vectorized(s1_norm, filtered)
        logits = np.dot(X, weights) + bias
        pred_mask = logits >= logit_thresh
        matched_eids = set([filtered[i].entity_id for i, m in enumerate(pred_mask) if m])
        preds[s1_id] = matched_eids
        match_counts.append(len(matched_eids))

    score = evaluate_predictions(gt_map, preds)
    avg_c = np.mean(cand_counts)
    avg_m = np.mean(match_counts)
    print(f"\n--- CURRENT PIPELINE PERFORMANCE ON REAL UNCURATED POOL ---")
    print(f"Macro F0.5 Score: {score:.4f}")
    print(f"Avg Candidates per S1: {avg_c:.1f}")
    print(f"Avg Matches per S1: {avg_m:.1f}")

    # Now let's test what happens with higher logit thresholds or top-K caps!
    for test_thresh in [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98]:
        lt = np.log(test_thresh / (1.0 - test_thresh))
        test_preds = {}
        t_match_counts = []
        for s1_norm in s1_norm_list:
            s1_id = s1_norm["entity_id"]
            cand_indices = index.get_candidate_indices(s1_norm, strategy="combined")
            if not cand_indices:
                test_preds[s1_id] = set()
                t_match_counts.append(0)
                continue
            cand_records = [index.cand_records[ci] for ci in cand_indices]
            filtered = [c for c in cand_records if prefilter_pair(s1_norm, c, config="conservative")]
            if len(filtered) > 80:
                s1_nt = s1_norm["name_tokens_set"]
                s1_at = s1_norm["address_tokens_set"]
                s1_name = s1_norm["business_name"]
                def cs(c):
                    return (10 if s1_name == c.business_name else 0) + len(s1_nt & c.name_tokens_set) * 3 + len(s1_at & c.address_tokens_set) * 2
                filtered.sort(key=cs, reverse=True)
                filtered = filtered[:80]
            X = extract_features_vectorized(s1_norm, filtered)
            logits = np.dot(X, weights) + bias
            mask = logits >= lt
            m_eids = set([filtered[i].entity_id for i, m in enumerate(mask) if m])
            test_preds[s1_id] = m_eids
            t_match_counts.append(len(m_eids))
        t_score = evaluate_predictions(gt_map, test_preds)
        print(f"Threshold {test_thresh:.2f} (logit={lt:+.2f}) -> Macro F0.5: {t_score:.4f} | Avg Matches: {np.mean(t_match_counts):.2f}")

if __name__ == "__main__":
    run_diagnostic()
