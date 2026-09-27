import os
import sys
import json
import numpy as np

PROJECT_ROOT = r"g:\Amazon ML 2026\amazon-ml-challenge-2026"
sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.features import build_features
from business_entity_resolution.src.evaluation import to_id_set

def analyze_true_vs_false():
    train_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # Load 500 S1 records and their GT
    s1_records = {}
    for r in stream_tsv_records(s1_path, max_records=500):
        s1_records[r["entity_id"]] = r

    gt_map = {}
    needed = set()
    for r in stream_tsv_records(gt_path):
        sid = r.get("source1_entity_id", "")
        if sid in s1_records:
            matches = to_id_set(r.get("matched_entity_ids", ""))
            gt_map[sid] = matches
            needed.update(matches)

    # Load needed S2/S3 records
    cand_records = {}
    for p in (s2_path, s3_path):
        for r in stream_tsv_records(p):
            eid = r.get("entity_id", "")
            if eid in needed:
                cand_records[eid] = r
            if len(cand_records) >= len(needed):
                break

    # Load model
    model_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    weights = np.array(m_data["weights"], dtype=np.float32)
    bias = float(m_data["bias"])
    feature_names = m_data["feature_names"]

    # Calculate logits for TRUE matches
    true_logits = []
    for sid, true_eids in gt_map.items():
        s1_rec = s1_records[sid]
        for tid in true_eids:
            if tid in cand_records:
                c_rec = cand_records[tid]
                feats = build_features(s1_rec, c_rec)
                x = np.array([feats[fn] for fn in feature_names], dtype=np.float32)
                l = np.dot(x, weights) + bias
                true_logits.append(l)

    true_logits = np.array(true_logits)
    print(f"TRUE MATCHES LOGITS (N={len(true_logits)}):")
    print(f"  Min: {np.min(true_logits):.2f}")
    print(f"  P10: {np.percentile(true_logits, 10):.2f}")
    print(f"  Median: {np.median(true_logits):.2f}")
    print(f"  Mean: {np.mean(true_logits):.2f}")
    print(f"  P90: {np.percentile(true_logits, 90):.2f}")
    print(f"  Max: {np.max(true_logits):.2f}")
    print(f"  % with logit >= 0.0: {(true_logits >= 0.0).mean()*100:.1f}%")
    print(f"  % with logit >= 1.0: {(true_logits >= 1.0).mean()*100:.1f}%")
    print(f"  % with logit >= 1.5: {(true_logits >= 1.5).mean()*100:.1f}%")

if __name__ == "__main__":
    run_diagnostic = analyze_true_vs_false()
