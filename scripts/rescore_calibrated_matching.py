"""
Calibrated Re-scoring and Re-ranking for Production Matching TSV.
Eliminates multi-chunk false positive accumulation, enforces name/address consistency,
caps matches to Top-4, and restores singleton precision.

Executes country-by-country under 800 MB RAM.
"""

import os
import sys
import gc
import time
import json
import psutil
import numpy as np
from typing import Dict, List, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records, DELIM, MATCHING_HEADER
from business_entity_resolution.src.normalization import normalize_record
from sagemaker.vectorized_matcher import CompactCand, extract_features_vectorized


def rescore_country_partition(
    country: str,
    s1_dict: Dict[str, Dict],
    input_match_file: str,
    output_match_file: str,
    cand_sources: List[str],
    weights: np.ndarray,
    bias: float,
    logit_threshold: float = 0.20,
    top_k: int = 4,
):
    print(f"\n{'='*80}")
    print(f"RE-SCORING PARTITION: {country}")
    print(f"  Input Matches   : {input_match_file}")
    print(f"  Output Matches  : {output_match_file}")
    print(f"  Logit Threshold : {logit_threshold:+.2f}")
    print(f"  Top-K Cap       : {top_k}")
    print(f"{'='*80}")

    t0_country = time.time()
    proc = psutil.Process()

    # 1. Read existing match candidates for this country
    print(f"[{country} 1/4] Reading existing candidate matches...", flush=True)
    entity_matches: Dict[str, List[str]] = {}
    needed_cands: Set[str] = set()

    with open(input_match_file, "r", encoding="utf-8") as f:
        for line in f:
            sid, _, rest = line.rstrip("\r\n").partition("\t")
            if sid and sid in s1_dict:
                mids = [m for m in rest.split(",") if m] if rest else []
                entity_matches[sid] = mids
                needed_cands.update(mids)

    print(f"  {country} S1 Entities : {len(s1_dict):,}", flush=True)
    print(f"  Current Matches     : {sum(len(m) for m in entity_matches.values()):,} total ({sum(len(m) for m in entity_matches.values())/max(len(s1_dict),1):.2f} avg)", flush=True)
    print(f"  Unique Cand IDs     : {len(needed_cands):,}", flush=True)

    # 2. Stream candidate sources and load ONLY the needed candidate records
    print(f"[{country} 2/4] Loading and normalizing {len(needed_cands):,} candidate records...", flush=True)
    t0_load = time.time()
    cand_store: Dict[str, CompactCand] = {}

    for src_path in cand_sources:
        if not os.path.isfile(src_path):
            continue
        with open(src_path, "r", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                eid, _, rest = line.partition("\t")
                if eid in needed_cands:
                    parts = line.rstrip("\r\n").split("\t")
                    bname = parts[1] if len(parts) > 1 else ""
                    baddr = parts[2] if len(parts) > 2 else ""
                    c_norm = normalize_record({
                        "entity_id": eid,
                        "business_name": bname,
                        "business_address": baddr,
                        "country": country,
                    })
                    cand_store[eid] = CompactCand(
                        entity_id=eid,
                        raw_business_name=c_norm.get("raw_business_name", ""),
                        business_name=c_norm.get("business_name", ""),
                        name_signature=c_norm.get("name_signature", ""),
                        business_address=c_norm.get("business_address", ""),
                        country=country,
                        compact_name=c_norm.get("compact_name", ""),
                        name_tokens_set=c_norm.get("name_tokens_set", set()),
                        address_tokens_set=c_norm.get("address_tokens_set", set()),
                        address_numbers=c_norm.get("address_numbers", set()),
                    )
                    if len(cand_store) >= len(needed_cands):
                        break
        if len(cand_store) >= len(needed_cands):
            break

    load_time = time.time() - t0_load
    ram_mb = proc.memory_info().rss / 1024 / 1024
    print(f"  Loaded {len(cand_store):,} candidate records in {load_time:.2f}s (RAM: {ram_mb:.1f} MB)", flush=True)

    # 3. Vectorized Re-scoring and Filtering
    print(f"[{country} 3/4] Re-scoring and applying calibrated constraints...", flush=True)
    t0_score = time.time()
    calibrated_results: Dict[str, List[str]] = {}
    new_match_counts = []

    for sid, s1_norm in s1_dict.items():
        curr_mids = entity_matches.get(sid, [])
        if not curr_mids:
            calibrated_results[sid] = []
            new_match_counts.append(0)
            continue

        cands = [cand_store[cid] for cid in curr_mids if cid in cand_store]
        if not cands:
            calibrated_results[sid] = []
            new_match_counts.append(0)
            continue

        # Extract features and compute logits
        X = extract_features_vectorized(s1_norm, cands)
        logits = np.dot(X, weights) + bias

        scored = []
        for i, logit_val in enumerate(logits):
            if logit_val < logit_threshold:
                continue

            # Minimum name consistency check (features 0-5)
            max_name = max(X[i, 0], X[i, 1], X[i, 2], X[i, 3], X[i, 4], X[i, 5])
            if max_name < 0.20:
                continue

            # Minimum address consistency check (features 8-10)
            # Must share address tokens unless name is an exact match and address is empty
            max_addr = max(X[i, 8], X[i, 9], X[i, 10])
            if max_addr < 0.15 and not (X[i, 1] > 0.9 and X[i, 7] > 0.9):
                continue

            scored.append((logit_val, cands[i].entity_id))

        # Sort by logit descending and keep at most Top-K
        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = [cid for _, cid in scored[:top_k]]
        calibrated_results[sid] = top_matches
        new_match_counts.append(len(top_matches))

    score_time = time.time() - t0_score
    rate = len(s1_dict) / max(score_time, 0.001)
    new_match_counts = np.array(new_match_counts)

    print(f"  Scored {len(s1_dict):,} S1 in {score_time:.2f}s ({rate:.1f} S1/sec)", flush=True)
    print(f"  Old Avg Matches     : {sum(len(m) for m in entity_matches.values())/max(len(s1_dict),1):.2f}", flush=True)
    print(f"  NEW Avg Matches     : {np.mean(new_match_counts):.2f}", flush=True)
    print(f"  NEW Max Matches     : {np.max(new_match_counts)}", flush=True)
    print(f"  Predicted Singletons: {(new_match_counts == 0).sum():,} ({(new_match_counts == 0).mean()*100:.2f}%)", flush=True)

    # 4. Write calibrated partition file
    print(f"[{country} 4/4] Writing calibrated partition to {output_match_file}...", flush=True)
    with open(output_match_file, "w", encoding="utf-8", newline="") as f_out:
        for sid in s1_dict.keys():
            matches = calibrated_results.get(sid, [])
            f_out.write(f"{sid}\t{','.join(matches)}\n")

    # Clean up memory
    del cand_store
    del entity_matches
    del calibrated_results
    gc.collect()

    print(f"[{country}] Completed in {time.time()-t0_country:.2f}s.")
    return output_match_file


def run_full_calibration(
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    model_path: str = "business_entity_resolution/src/trained_model.json",
    logit_threshold: float = 0.20,
    top_k: int = 4,
):
    total_t0 = time.time()
    print("=" * 85)
    print("AMAZON ML CHALLENGE 2026 — GLOBAL CALIBRATED RE-SCORING PIPELINE")
    print(f"Test Directory    : {test_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"Logit Threshold   : {logit_threshold:+.2f}")
    print(f"Top-K Cap         : {top_k}")
    print("=" * 85)

    tmp_dir = os.path.join(output_dir, "tmp_partitions")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    cand_sources = [s2_path, s3_path]

    # Load model
    full_model_path = os.path.join(PROJECT_ROOT, model_path)
    with open(full_model_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    weights = np.array(m_data["weights"], dtype=np.float32)
    bias = float(m_data["bias"])

    # 1. Read S1 records grouped by country, preserving global order
    print("\n[Step 1] Loading and normalizing Source 1 test records...", flush=True)
    t0_s1 = time.time()
    s1_order: List[str] = []
    s1_by_country: Dict[str, Dict[str, Dict]] = {"FRANCE": {}, "US": {}, "INDIA": {}}

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            sid = parts[0]
            bname = parts[1] if len(parts) > 1 else ""
            baddr = parts[2] if len(parts) > 2 else ""
            country = parts[3].strip().upper() if len(parts) > 3 and parts[3] else "UNKNOWN"
            if country not in s1_by_country:
                s1_by_country[country] = {}

            s1_order.append(sid)
            s1_by_country[country][sid] = normalize_record({
                "entity_id": sid,
                "business_name": bname,
                "business_address": baddr,
                "country": country,
            })

    print(f"  Loaded {len(s1_order):,} Source 1 records in {time.time()-t0_s1:.2f}s:")
    for c, d in s1_by_country.items():
        print(f"    - {c}: {len(d):,} entities")

    # 2. Process each country partition
    calibrated_partition_files = []
    for country in ["FRANCE", "US", "INDIA"]:
        if country not in s1_by_country or not s1_by_country[country]:
            continue
        in_match_file = os.path.join(tmp_dir, f"part_match_{country.lower()}.tsv")
        out_match_file = os.path.join(tmp_dir, f"part_match_calibrated_{country.lower()}.tsv")

        res_file = rescore_country_partition(
            country=country,
            s1_dict=s1_by_country[country],
            input_match_file=in_match_file,
            output_match_file=out_match_file,
            cand_sources=cand_sources,
            weights=weights,
            bias=bias,
            logit_threshold=logit_threshold,
            top_k=top_k,
        )
        calibrated_partition_files.append(res_file)

    # 3. Assemble Final matching_results.tsv in Exact Original Order
    print(f"\n[Step 3] Assembling final matching_results.tsv in exact original order...", flush=True)
    t0_assemble = time.time()
    match_lookup: Dict[str, str] = {}

    for part_f in calibrated_partition_files:
        with open(part_f, "r", encoding="utf-8") as f:
            for line in f:
                sid, _, matches = line.rstrip("\r\n").partition("\t")
                if sid:
                    match_lookup[sid] = matches

    final_match_path = os.path.join(output_dir, "matching_results.tsv")
    final_cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    total_matches = 0
    with open(final_match_path, "w", encoding="utf-8", newline="") as f_out:
        f_out.write(MATCHING_HEADER)
        for sid in s1_order:
            matches_str = match_lookup.get(sid, "")
            f_out.write(f"{sid}{DELIM}{matches_str}\n")
            if matches_str:
                total_matches += len(matches_str.split(","))

    assemble_time = time.time() - t0_assemble
    avg_matches = total_matches / len(s1_order)
    size_mb = os.path.getsize(final_match_path) / (1024 * 1024)

    print(f"  Successfully assembled {len(s1_order):,} entities in {assemble_time:.2f}s:")
    print(f"    File: {final_match_path} ({size_mb:.1f} MB)")
    print(f"    Total Matches: {total_matches:,}")
    print(f"    Avg Matches/S1: {avg_matches:.2f}")

    # 4. Validate with Official Challenge Validator
    print(f"\n[Step 4] Running Official validate_submission.py...", flush=True)
    validator_path = os.path.join(PROJECT_ROOT, "student_resource", "utils", "validate_submission.py")
    cmd = f'python "{validator_path}" --matching "{final_match_path}" --candidate "{final_cand_path}" --test-dir "{test_dir}"'
    val_ret = os.system(cmd)
    if val_ret == 0:
        print(">>> OFFICIAL VALIDATOR RESULT: PASS (100% compliant)")
    else:
        print(f">>> OFFICIAL VALIDATOR RESULT: FAILED (Exit Code {val_ret})")

    # 5. Package Submission Zip
    print(f"\n[Step 5] Packaging Final Submission Zip...", flush=True)
    packager_path = os.path.join(PROJECT_ROOT, "scripts", "package_submission.py")
    pkg_ret = os.system(f'python "{packager_path}"')
    if pkg_ret == 0:
        print(">>> SUBMISSION ZIP PACKAGE GENERATED SUCCESSFULLY")

    print("\n" + "=" * 85)
    print(f"CALIBRATION PIPELINE COMPLETE IN {time.time()-total_t0:.2f}s.")
    print(f"Final matching_results.tsv is ready for upload!")
    print("=" * 85)


if __name__ == "__main__":
    run_full_calibration(
        test_dir="student_resource/dataset/test",
        output_dir="output",
        model_path="business_entity_resolution/src/trained_model.json",
        logit_threshold=0.20,
        top_k=4,
    )
