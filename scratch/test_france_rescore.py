import os
import sys
import json
import time
import numpy as np

PROJECT_ROOT = r"g:\Amazon ML 2026\amazon-ml-challenge-2026"
sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from sagemaker.vectorized_matcher import CompactCand, extract_features_vectorized

def test_france_rescore():
    print("=" * 70)
    print("TESTING RE-SCORING ON FRANCE")
    print("=" * 70)

    test_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    match_path = os.path.join(PROJECT_ROOT, "output", "matching_results.tsv")

    # Load model
    model_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    weights = np.array(m_data["weights"], dtype=np.float32)
    bias = float(m_data["bias"])

    # 1. Load France S1 records and their current matches
    print("Reading France S1 records and current matches...", flush=True)
    france_s1 = {}
    france_matches = {}
    needed_cands = set()

    with open(s1_path, "r", encoding="utf-8") as f1, open(match_path, "r", encoding="utf-8") as fm:
        next(f1); next(fm)
        for l1, lm in zip(f1, fm):
            p1 = l1.strip().split("\t")
            if len(p1) >= 4 and p1[3].upper() == "FRANCE":
                sid = p1[0]
                france_s1[sid] = normalize_record({
                    "entity_id": sid,
                    "business_name": p1[1],
                    "business_address": p1[2],
                    "country": "FRANCE",
                })
                pm = lm.strip().split("\t")
                mids = pm[1].split(",") if len(pm) > 1 and pm[1] else []
                france_matches[sid] = mids
                needed_cands.update(mids)

    print(f"Loaded {len(france_s1):,} France S1 entities.", flush=True)
    print(f"Current France matches: {sum(len(m) for m in france_matches.values()):,} total ({sum(len(m) for m in france_matches.values())/len(france_s1):.2f} avg).", flush=True)
    print(f"Needed candidate records: {len(needed_cands):,}.", flush=True)

    # 2. Load only needed candidate records
    print("Loading candidate records from test_source2 and test_source3...", flush=True)
    t0_load = time.time()
    cand_store = {}
    for p in (s2_path, s3_path):
        with open(p, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                eid, _, rest = line.partition("\t")
                if eid in needed_cands:
                    parts = line.strip().split("\t")
                    bname = parts[1] if len(parts) > 1 else ""
                    baddr = parts[2] if len(parts) > 2 else ""
                    c_norm = normalize_record({
                        "entity_id": eid,
                        "business_name": bname,
                        "business_address": baddr,
                        "country": "FRANCE",
                    })
                    cand_store[eid] = CompactCand(
                        entity_id=eid,
                        raw_business_name=c_norm.get("raw_business_name", ""),
                        business_name=c_norm.get("business_name", ""),
                        name_signature=c_norm.get("name_signature", ""),
                        business_address=c_norm.get("business_address", ""),
                        country="FRANCE",
                        compact_name=c_norm.get("compact_name", ""),
                        name_tokens_set=c_norm.get("name_tokens_set", set()),
                        address_tokens_set=c_norm.get("address_tokens_set", set()),
                        address_numbers=c_norm.get("address_numbers", set()),
                    )
                    if len(cand_store) >= len(needed_cands):
                        break
        if len(cand_store) >= len(needed_cands):
            break

    print(f"Loaded {len(cand_store):,} candidate records in {time.time()-t0_load:.2f}s.", flush=True)

    # 3. Re-score and filter each France S1 entity
    print("Re-scoring France matches with calibrated constraints...", flush=True)
    t0_score = time.time()
    new_match_counts = []
    sample_outputs = []

    for sid, s1_norm in france_s1.items():
        curr_mids = france_matches.get(sid, [])
        if not curr_mids:
            new_match_counts.append(0)
            continue

        cands = [cand_store[cid] for cid in curr_mids if cid in cand_store]
        if not cands:
            new_match_counts.append(0)
            continue

        X = extract_features_vectorized(s1_norm, cands)
        logits = np.dot(X, weights) + bias

        scored = []
        for i, logit_val in enumerate(logits):
            if logit_val < 0.20:
                continue

            # Minimum name consistency check
            max_name = max(X[i, 0], X[i, 1], X[i, 2], X[i, 3], X[i, 4], X[i, 5])
            if max_name < 0.20:
                continue

            # Minimum address consistency check (must share address tokens unless name is exact)
            max_addr = max(X[i, 8], X[i, 9], X[i, 10])
            if max_addr < 0.15 and not (X[i, 1] > 0.9 and X[i, 7] > 0.9):
                continue

            scored.append((logit_val, cands[i].entity_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = [cid for _, cid in scored[:4]]
        new_match_counts.append(len(top_matches))

        if len(sample_outputs) < 5 and len(curr_mids) > 10:
            sample_outputs.append((sid, s1_norm["business_name"], s1_norm["business_address"], curr_mids, top_matches))

    new_match_counts = np.array(new_match_counts)
    print(f"\n--- FRANCE RE-SCORING RESULTS ---")
    print(f"Entities: {len(new_match_counts):,}")
    print(f"Old avg matches: {sum(len(m) for m in france_matches.values())/len(france_s1):.2f}")
    print(f"NEW avg matches: {np.mean(new_match_counts):.2f}")
    print(f"NEW max matches: {np.max(new_match_counts)}")
    print(f"Singletons (0 matches): {(new_match_counts == 0).sum():,} ({(new_match_counts == 0).mean()*100:.2f}%)")
    print(f"Entities with 1-4 matches: {(new_match_counts > 0).sum():,} ({(new_match_counts > 0).mean()*100:.2f}%)")
    print(f"Re-scoring time: {time.time()-t0_score:.2f}s ({len(france_s1)/(time.time()-t0_score):.1f} S1/sec)")

    print("\nSample entities before & after:")
    for sid, name, addr, old_m, new_m in sample_outputs:
        print(f"  S1: {sid} | '{name}' | '{addr}'")
        print(f"    Old ({len(old_m)} matches): {old_m[:6]} ...")
        print(f"    New ({len(new_m)} matches): {new_m}")

if __name__ == "__main__":
    test_france_rescore()
