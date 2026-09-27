import os
import sys
import numpy as np

PROJECT_ROOT = r"g:\Amazon ML 2026\amazon-ml-challenge-2026"
sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.features import build_features
from business_entity_resolution.src.evaluation import f05, evaluate_predictions, to_id_set

def test_filtering_on_train():
    train_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("Loading 5,000 S1 records...")
    s1_records = {}
    for r in stream_tsv_records(s1_path, max_records=5000):
        s1_records[r["entity_id"]] = normalize_record(r)

    print("Loading ground truth...")
    gt_map = {}
    needed = set()
    for r in stream_tsv_records(gt_path):
        sid = r.get("source1_entity_id", "")
        if sid in s1_records:
            m = to_id_set(r.get("matched_entity_ids", ""))
            gt_map[sid] = m
            needed.update(m)
    for sid in s1_records:
        if sid not in gt_map:
            gt_map[sid] = set()

    print(f"Entities: {len(s1_records)}, True singletons: {sum(1 for m in gt_map.values() if not m)}")

    # Load needed true candidates + some distractors
    c_records = {}
    for p in (s2_path, s3_path):
        for r in stream_tsv_records(p):
            eid = r.get("entity_id", "")
            if eid in needed:
                c_records[eid] = normalize_record(r)
            if len(c_records) >= len(needed):
                break

    # Test pairwise scoring function
    def score_pair(s1, c):
        # 1. Street number conflict check
        s1_num = s1.get("street_number", "")
        c_num = c.get("street_number", "")
        if s1_num and c_num and s1_num != c_num:
            return -999.0  # Conflicting street numbers!

        # 2. Name features
        s1_name = s1.get("business_name", "")
        c_name = c.get("business_name", "")
        s1_toks = s1.get("name_tokens_set", set())
        c_toks = c.get("name_tokens_set", set())

        name_exact = (s1_name == c_name and bool(s1_name))
        name_sig = (s1.get("name_signature") == c.get("name_signature") and bool(s1.get("name_signature")))
        
        inter_len = len(s1_toks & c_toks)
        union_len = len(s1_toks | c_toks)
        name_jaccard = inter_len / max(union_len, 1)
        min_toks = min(len(s1_toks), len(c_toks))
        name_containment = inter_len / max(min_toks, 1) if min_toks > 0 else 0.0

        # Char shingles
        s1_shingles = set(s1.get("char_shingles", []))
        c_shingles = set(c.get("char_shingles", []))
        inter_sh = len(s1_shingles & c_shingles)
        union_sh = len(s1_shingles | c_shingles)
        name_ngram_jaccard = inter_sh / max(union_sh, 1)

        # 3. Address features
        s1_addr_toks = s1.get("address_tokens_set", set())
        c_addr_toks = c.get("address_tokens_set", set())
        addr_inter = len(s1_addr_toks & c_addr_toks)
        addr_union = len(s1_addr_toks | c_addr_toks)
        addr_jaccard = addr_inter / max(addr_union, 1)
        addr_min = min(len(s1_addr_toks), len(c_addr_toks))
        addr_containment = addr_inter / max(addr_min, 1) if addr_min > 0 else 0.0

        s1_nums = s1.get("address_numbers", set())
        c_nums = c.get("address_numbers", set())
        num_inter = len(s1_nums & c_nums)
        num_union = len(s1_nums | c_nums)
        num_jaccard = num_inter / max(num_union, 1)

        # 4. Mandatory minimum consistency checks
        # In business ER, two records CANNOT be the same business if name has zero overlap AND address has zero overlap
        max_name_sim = max(name_jaccard, name_containment, name_ngram_jaccard, 1.0 if name_exact else 0.0)
        max_addr_sim = max(addr_jaccard, addr_containment, 1.0 if (s1.get("business_address") == c.get("business_address") and s1.get("business_address")) else 0.0)

        # If name has no relation at all (e.g. '#fdrationdeveloppement' vs 'Team Ecole')
        if max_name_sim < 0.20:
            return -999.0

        # If address has no relation at all (different cities/streets)
        if max_addr_sim < 0.20 and not (name_exact and not s1.get("business_address")):
            return -999.0

        # Composite score
        score = (
            (2.5 if name_exact else 0.0)
            + (1.5 if name_sig else 0.0)
            + 2.0 * name_ngram_jaccard
            + 1.5 * name_jaccard
            + 1.0 * name_containment
            + 2.5 * addr_containment
            + 1.5 * addr_jaccard
            + 1.0 * num_jaccard
        )
        return score

    print("Evaluating scoring function on true positive pairs...")
    scores = []
    for sid, tids in gt_map.items():
        s1 = s1_records[sid]
        for tid in tids:
            if tid in c_records:
                c = c_records[tid]
                sc = score_pair(s1, c)
                scores.append(sc)

    scores = np.array(scores)
    print(f"True positive scores: Min={scores.min():.2f}, P1={np.percentile(scores, 1):.2f}, P5={np.percentile(scores, 5):.2f}, Median={np.median(scores):.2f}, Mean={np.mean(scores):.2f}")
    print(f"Percentage of true matches passing score >= 2.0: {(scores >= 2.0).mean()*100:.1f}%")
    print(f"Percentage of true matches passing score >= 2.5: {(scores >= 2.5).mean()*100:.1f}%")
    print(f"Percentage of true matches passing score >= 3.0: {(scores >= 3.0).mean()*100:.1f}%")

if __name__ == "__main__":
    test_filtering_on_train()
