import os
import sys
import json
import numpy as np

PROJECT_ROOT = r"g:\Amazon ML 2026\amazon-ml-challenge-2026"
sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.prefilter import prefilter_pair
from business_entity_resolution.src.evaluation import f05, evaluate_predictions, to_id_set
from sagemaker.vectorized_matcher import CompactInvertedIndex, extract_features_vectorized

def test_calibrated_scoring():
    train_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # Load 2,000 US S1 records
    print("Loading 2,000 US S1 records...")
    s1_records = []
    s1_ids = set()
    for rec in stream_tsv_records(s1_path):
        if rec.get("country", "").upper() == "US":
            s1_records.append(rec)
            s1_ids.add(rec["entity_id"])
            if len(s1_records) >= 2000:
                break

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

    # Build index with 500,000 US candidates + ALL needed true targets
    print("Building candidate index with 500,000 background + true targets...")
    needed_true = set()
    for tids in gt_map.values():
        needed_true.update(tids)

    index = CompactInvertedIndex(max_bucket_size=300)
    bg_count = 0
    for path in (s2_path, s3_path):
        for rec in stream_tsv_records(path):
            eid = rec.get("entity_id", "")
            if eid in needed_true:
                index.add_record(rec, strategy="combined")
            elif bg_count < 500000:
                index.add_record(rec, strategy="combined")
                bg_count += 1

    print(f"Index built with {len(index):,} candidates.")

    # Normalize S1 queries
    s1_norm_list = [normalize_record(r) for r in s1_records]

    # Test grid of configurations:
    # Top-K in [3, 4, 5, None]
    # Logit threshold in [-0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 1.1]
    # Name consistency filter (name_sim > 0.15)
    print("\n--- GRID SEARCH CALIBRATION ---")
    for min_name_check in [False, True]:
        for top_k in [3, 4, None]:
            for logit_t in [-0.2, 0.0, 0.3, 0.5, 0.7, 0.9]:
                preds = {}
                m_counts = []
                for s1_norm in s1_norm_list:
                    s1_id = s1_norm["entity_id"]
                    cand_indices = index.get_candidate_indices(s1_norm, strategy="combined")
                    if not cand_indices:
                        preds[s1_id] = set()
                        m_counts.append(0)
                        continue

                    cand_records = [index.cand_records[ci] for ci in cand_indices]
                    filtered = [c for c in cand_records if prefilter_pair(s1_norm, c, config="conservative")]
                    if not filtered:
                        preds[s1_id] = set()
                        m_counts.append(0)
                        continue

                    # Vectorized feature extraction
                    X = extract_features_vectorized(s1_norm, filtered)
                    logits = np.dot(X, weights) + bias

                    scored_cands = []
                    for i, logit_val in enumerate(logits):
                        if logit_val < logit_t:
                            continue
                        # If min_name_check: check name features (features 0,1,2,3,4,5)
                        if min_name_check:
                            max_name_f = max(X[i, 0], X[i, 1], X[i, 2], X[i, 3], X[i, 4], X[i, 5])
                            if max_name_f < 0.20:
                                continue
                            max_addr_f = max(X[i, 8], X[i, 9], X[i, 10])
                            if max_addr_f < 0.15 and not (X[i, 1] > 0.9 and X[i, 7] > 0.9):
                                continue

                        scored_cands.append((logit_val, filtered[i].entity_id))

                    # Sort by score descending
                    scored_cands.sort(key=lambda x: x[0], reverse=True)
                    if top_k is not None:
                        scored_cands = scored_cands[:top_k]

                    matched_set = {cid for _, cid in scored_cands}
                    preds[s1_id] = matched_set
                    m_counts.append(len(matched_set))

                score = evaluate_predictions(gt_map, preds)
                avg_m = np.mean(m_counts)
                print(f"NameCheck={str(min_name_check):5s} | TopK={str(top_k):4s} | Logit={logit_t:+.2f} => F0.5: {score:.4f} | AvgMatches: {avg_m:.2f}")

if __name__ == "__main__":
    test_calibrated_scoring()
