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

def fast_grid_search():
    train_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("Loading 2,000 US S1 records...", flush=True)
    s1_records = []
    s1_ids = set()
    for rec in stream_tsv_records(s1_path):
        if rec.get("country", "").upper() == "US":
            s1_records.append(rec)
            s1_ids.add(rec["entity_id"])
            if len(s1_records) >= 2000:
                break

    gt_map = {}
    needed_true = set()
    for rec in stream_tsv_records(gt_path):
        sid = rec.get("source1_entity_id", "")
        if sid in s1_ids:
            m = to_id_set(rec.get("matched_entity_ids", ""))
            gt_map[sid] = m
            needed_true.update(m)
    for sid in s1_ids:
        if sid not in gt_map:
            gt_map[sid] = set()

    # Load model
    model_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    weights = np.array(m_data["weights"], dtype=np.float32)
    bias = float(m_data["bias"])

    # Build index with 300,000 candidates + true targets
    print(f"Building candidate index (300k background + {len(needed_true):,} true targets)...", flush=True)
    index = CompactInvertedIndex(max_bucket_size=300)
    bg_count = 0
    for path in (s2_path, s3_path):
        for rec in stream_tsv_records(path):
            eid = rec.get("entity_id", "")
            if eid in needed_true:
                index.add_record(rec, strategy="combined")
            elif bg_count < 300000:
                index.add_record(rec, strategy="combined")
                bg_count += 1

    print(f"Index built: {len(index):,} candidates.", flush=True)

    # 1. Extract candidates and compute logits ONCE
    print("Extracting features and caching candidate logits...", flush=True)
    s1_norm_list = [normalize_record(r) for r in s1_records]
    cached_data = [] # List of (s1_id, cand_eids, logits, name_max_sims, addr_max_sims)

    t0 = time.time()
    for s1_norm in s1_norm_list:
        s1_id = s1_norm["entity_id"]
        cand_indices = index.get_candidate_indices(s1_norm, strategy="combined")
        if not cand_indices:
            cached_data.append((s1_id, [], np.array([]), np.array([]), np.array([])))
            continue

        cand_records = [index.cand_records[ci] for ci in cand_indices]
        filtered = [c for c in cand_records if prefilter_pair(s1_norm, c, config="conservative")]
        if not filtered:
            cached_data.append((s1_id, [], np.array([]), np.array([]), np.array([])))
            continue

        X = extract_features_vectorized(s1_norm, filtered)
        logits = np.dot(X, weights) + bias
        cand_eids = [c.entity_id for c in filtered]

        # max name sim = max(raw_exact, norm_exact, sig_match, jaccard, containment, ngram)
        name_max = np.max(X[:, :6], axis=1)
        addr_max = np.max(X[:, 8:11], axis=1)

        cached_data.append((s1_id, cand_eids, logits, name_max, addr_max))

    print(f"Features and logits cached in {time.time()-t0:.2f}s.", flush=True)

    # 2. Fast sweep over configurations in memory
    print("\n" + "=" * 80, flush=True)
    print("GRID SEARCH RESULTS ON REAL CANDIDATE POOL", flush=True)
    print("=" * 80, flush=True)

    best_score = -1.0
    best_config = None

    for min_name in [False, True]:
        for top_k in [None, 5, 4, 3, 2]:
            for logit_t in [-0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
                preds = {}
                m_counts = []
                for s1_id, cand_eids, logits, name_max, addr_max in cached_data:
                    if len(cand_eids) == 0:
                        preds[s1_id] = set()
                        m_counts.append(0)
                        continue

                    # Filter mask
                    mask = logits >= logit_t
                    if min_name:
                        # Require minimum name overlap and address overlap
                        mask = mask & (name_max >= 0.20) & (addr_max >= 0.15)

                    idxs = np.where(mask)[0]
                    if len(idxs) == 0:
                        preds[s1_id] = set()
                        m_counts.append(0)
                        continue

                    # Sort by logit descending
                    sorted_order = np.argsort(-logits[idxs])
                    if top_k is not None:
                        sorted_order = sorted_order[:top_k]

                    selected_eids = {cand_eids[idxs[k]] for k in sorted_order}
                    preds[s1_id] = selected_eids
                    m_counts.append(len(selected_eids))

                score = evaluate_predictions(gt_map, preds)
                avg_m = np.mean(m_counts)

                if score > best_score:
                    best_score = score
                    best_config = (min_name, top_k, logit_t, avg_m)

                # Print top results
                if score >= 0.70 or logit_t in [0.0, 0.5, 0.8]:
                    print(f"MinFilter={str(min_name):5s} | TopK={str(top_k):4s} | Logit={logit_t:+.2f} => F0.5: {score:.4f} | AvgMatches: {avg_m:.2f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"BEST CONFIGURATION: MinFilter={best_config[0]}, TopK={best_config[1]}, Logit={best_config[2]:+.2f} => Macro F0.5: {best_score:.4f} (AvgMatches: {best_config[3]:.2f})", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    fast_grid_search()
