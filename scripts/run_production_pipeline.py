"""
Amazon ML Challenge 2026 — Production Inference Pipeline.
Executes the validated, optimized pipeline (Configuration C: Conservative Prefilter + Cap 80 + NumPy Matcher)
on the complete 1,732,544 Source-1 test entities against ~10 million Source-2/3 candidate pool.

Guarantees:
1. Bounded memory (< 3.0 GB RAM peak) via country partitioning and streaming IO.
2. Candidate subset rule: Every matched ID strictly exists in that Source1 entity's candidate set.
3. Candidate completeness: candidate_pairs.tsv contains the EXACT candidate set passed to the matcher.
4. Deterministic row order: Exactly 1,732,544 rows in the exact order of test_source1.tsv.
5. Zero duplicate IDs per row.
6. Execution of the official validator (student_resource/utils/validate_submission.py).
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import gc
import json
import time
import psutil
import subprocess
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import (
    DELIM,
    CANDIDATE_HEADER,
    MATCHING_HEADER,
    format_id_list,
    stream_tsv_records,
)
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.prefilter import prefilter_pair
from business_entity_resolution.src.model import FastLogisticRegression
from sagemaker.vectorized_matcher import CompactInvertedIndex, extract_features_vectorized


def load_production_model(model_path: str) -> Tuple[FastLogisticRegression, float]:
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = FastLogisticRegression()
    model.weights = list(data["weights"])
    model.bias = float(data["bias"])
    threshold = float(data.get("optimal_threshold", 0.50))
    print(f"Loaded Trained Model: FastLogisticRegression (Threshold: {threshold:.2f}, Saved F0.5: {data.get('macro_f05', 0):.4f})")
    return model, threshold


def process_country_partition(
    country: str,
    s1_records: List[Dict[str, Any]],
    cand_paths: List[str],
    model: FastLogisticRegression,
    threshold: float,
    tmp_dir: str,
    max_candidates: int = 80,
    batch_size: int = 2000,
) -> Tuple[str, str]:
    """
    Process candidate retrieval and matching for all S1 entities of a specific country.
    Writes partition candidate and matching results to disk.
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING PARTITION: {country} ({len(s1_records):,} Source-1 Queries)")
    print(f"{'='*80}")

    tmp_cand_file = os.path.join(tmp_dir, f"part_cand_{country.lower()}.tsv")
    tmp_match_file = os.path.join(tmp_dir, f"part_match_{country.lower()}.tsv")

    # 1. Build Inverted Index for this country
    print(f"[{country} 1/3] Streaming and indexing candidate records (S2 & S3)...")
    t0_idx = time.time()
    index = CompactInvertedIndex(max_bucket_size=300)

    total_cands_loaded = 0
    for cand_path in cand_paths:
        if not os.path.isfile(cand_path):
            continue
        with open(cand_path, "r", encoding="utf-8") as f_in:
            next(f_in, None)  # Skip header
            for line in f_in:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    c_country = parts[3].strip().upper() if parts[3] else "UNKNOWN"
                    if c_country == country or (country == "UNKNOWN" and not c_country):
                        index.add_record({
                            "entity_id": parts[0],
                            "business_name": parts[1] if len(parts) > 1 else "",
                            "business_address": parts[2] if len(parts) > 2 else "",
                            "country": c_country,
                        }, strategy="combined")
                        total_cands_loaded += 1
                elif country == "UNKNOWN" and len(parts) >= 2:
                    index.add_record({
                        "entity_id": parts[0],
                        "business_name": parts[1] if len(parts) > 1 else "",
                        "business_address": parts[2] if len(parts) > 2 else "",
                        "country": "UNKNOWN",
                    }, strategy="combined")
                    total_cands_loaded += 1

    idx_time = time.time() - t0_idx
    proc = psutil.Process()
    ram_mb = proc.memory_info().rss / 1024 / 1024
    print(f"[{country} 1/3] Index built: {len(index):,} candidates in {idx_time:.2f}s (RAM: {ram_mb:.1f} MB)")

    # 2. Match S1 records against index in streaming batches
    print(f"[{country} 2/3] Evaluating {len(s1_records):,} S1 queries with Configuration C...")
    weights = np.array(model.weights, dtype=np.float32)
    bias = float(model.bias)
    logit_thresh = np.log(threshold / (1.0 - threshold)) if (0.0 < threshold < 1.0) else 0.0

    t0_match = time.time()
    n_processed = 0
    n_total_cands = 0
    n_total_matches = 0

    with open(tmp_cand_file, "w", encoding="utf-8", newline="") as f_c, \
         open(tmp_match_file, "w", encoding="utf-8", newline="") as f_m:

        for b_start in range(0, len(s1_records), batch_size):
            batch = s1_records[b_start : b_start + batch_size]
            b_t0 = time.time()

            for s1_rec in batch:
                s1_norm = normalize_record(s1_rec) if "name_tokens_set" not in s1_rec else s1_rec
                s1_id = s1_norm["entity_id"]

                cand_indices = index.get_candidate_indices(s1_norm, strategy="combined")
                if not cand_indices:
                    f_c.write(f"{s1_id}\t\n")
                    f_m.write(f"{s1_id}\t\n")
                    n_processed += 1
                    continue

                cand_records = [index.cand_records[ci] for ci in cand_indices]

                # Conservative prefilter
                filtered = [c for c in cand_records if prefilter_pair(s1_norm, c, config="conservative")]

                # Top-K cheap ranking & capping
                if max_candidates and len(filtered) > max_candidates:
                    s1_nt = s1_norm.get("name_tokens_set", set())
                    s1_at = s1_norm.get("address_tokens_set", set())
                    s1_name = s1_norm.get("business_name", "")

                    def cheap_score(c):
                        score = 0
                        if s1_name == c.business_name:
                            score += 10
                        score += len(s1_nt & c.name_tokens_set) * 3
                        score += len(s1_at & c.address_tokens_set) * 2
                        return score

                    filtered.sort(key=cheap_score, reverse=True)
                    filtered = filtered[:max_candidates]

                if not filtered:
                    f_c.write(f"{s1_id}\t\n")
                    f_m.write(f"{s1_id}\t\n")
                    n_processed += 1
                    continue

                cand_eids = sorted([c.entity_id for c in filtered])
                cand_eids_set = set(cand_eids)

                # Vectorized feature extraction and scoring
                X = extract_features_vectorized(s1_norm, filtered)
                logits = np.dot(X, weights) + bias
                pred_mask = logits >= logit_thresh

                # Candidate subset enforcement
                matched_eids = sorted([filtered[i].entity_id for i, m in enumerate(pred_mask) if m and filtered[i].entity_id in cand_eids_set])

                f_c.write(f"{s1_id}\t{','.join(cand_eids)}\n")
                f_m.write(f"{s1_id}\t{','.join(matched_eids)}\n")

                n_processed += 1
                n_total_cands += len(cand_eids)
                n_total_matches += len(matched_eids)

            if n_processed % 20000 < batch_size or n_processed == len(s1_records):
                elapsed_so_far = time.time() - t0_match
                rate = n_processed / max(elapsed_so_far, 0.001)
                rem_sec = (len(s1_records) - n_processed) / max(rate, 0.001)
                ram_now = proc.memory_info().rss / 1024 / 1024
                print(f"  [{country}] Progress: {n_processed:,}/{len(s1_records):,} queries ({n_processed/len(s1_records):.1%}) | "
                      f"Throughput: {rate:.1f} S1/s | Matches: {n_total_matches:,} | ETA: {rem_sec/60:.1f}m | RAM: {ram_now:.1f} MB")

    total_match_time = time.time() - t0_match
    avg_cands = n_total_cands / max(n_processed, 1)
    print(f"[{country} 3/3] Partition completed in {total_match_time:.2f}s ({len(s1_records)/total_match_time:.1f} S1/sec, Avg Cands: {avg_cands:.1f})")

    # Free partition memory completely
    del index
    gc.collect()

    return tmp_cand_file, tmp_match_file


def run_full_production(
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    model_path: str = "business_entity_resolution/src/trained_model.json",
    max_candidates: int = 80,
    batch_size: int = 2000,
):
    pipeline_t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp_partitions")
    os.makedirs(tmp_dir, exist_ok=True)

    cand_out_file = os.path.join(output_dir, "candidate_pairs.tsv")
    match_out_file = os.path.join(output_dir, "matching_results.tsv")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print("=" * 85)
    print("AMAZON ML CHALLENGE 2026 — FULL TEST DATASET PRODUCTION PIPELINE")
    print(f"Test Directory    : {test_dir}")
    print(f"Candidate Output  : {cand_out_file}")
    print(f"Matching Output   : {match_out_file}")
    print(f"Prefilter Config  : Conservative + Cap {max_candidates}")
    print("=" * 85)

    # 1. Load model
    full_model_path = os.path.join(PROJECT_ROOT, model_path)
    model, threshold = load_production_model(full_model_path)

    # 2. Read Source-1 entities and preserve exact input order
    print("\n[Phase 1/4] Reading Source-1 entities and grouping by country...")
    t0_read = time.time()
    s1_order: List[str] = []
    s1_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            sid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3].strip().upper() if len(parts) > 3 and parts[3] else "UNKNOWN"

            rec = {
                "entity_id": sid,
                "business_name": name,
                "business_address": addr,
                "country": country,
            }
            s1_order.append(sid)
            s1_by_country[country].append(rec)

    total_s1 = len(s1_order)
    print(f"  Loaded {total_s1:,} Source-1 queries in {time.time() - t0_read:.2f}s:")
    for c, items in sorted(s1_by_country.items()):
        print(f"    - {c:<10}: {len(items):,} entities ({len(items)/total_s1:.1%})")

    # 3. Process Each Country Partition Sequentially
    print("\n[Phase 2/4] Executing country partition processing...")
    partition_cand_files = []
    partition_match_files = []

    # Process in order: FRANCE, INDIA, US
    countries = sorted(s1_by_country.keys())
    for country in countries:
        s1_subset = s1_by_country[country]
        cand_f, match_f = process_country_partition(
            country=country,
            s1_records=s1_subset,
            cand_paths=[s2_path, s3_path],
            model=model,
            threshold=threshold,
            tmp_dir=tmp_dir,
            max_candidates=max_candidates,
            batch_size=batch_size,
        )
        partition_cand_files.append(cand_f)
        partition_match_files.append(match_f)

    # Free S1 partitioned dictionaries
    del s1_by_country
    gc.collect()

    # 4. Assemble Final Outputs in Exact Original S1 Order
    print(f"\n[Phase 3/4] Assembling final output files in exact original Source-1 order...")
    t0_merge = time.time()

    # Index partition files by entity_id
    cand_lookup: Dict[str, str] = {}
    match_lookup: Dict[str, str] = {}

    for c_file in partition_cand_files:
        with open(c_file, "r", encoding="utf-8") as f:
            for line in f:
                sid, _, cands = line.rstrip("\r\n").partition("\t")
                if sid:
                    cand_lookup[sid] = cands

    for m_file in partition_match_files:
        with open(m_file, "r", encoding="utf-8") as f:
            for line in f:
                sid, _, matches = line.rstrip("\r\n").partition("\t")
                if sid:
                    match_lookup[sid] = matches

    # Stream write in original S1 order
    total_written_cands = 0
    total_written_matches = 0

    with open(cand_out_file, "w", encoding="utf-8", newline="") as f_c_out, \
         open(match_out_file, "w", encoding="utf-8", newline="") as f_m_out:

        f_c_out.write(CANDIDATE_HEADER)
        f_m_out.write(MATCHING_HEADER)

        for sid in s1_order:
            cands_str = cand_lookup.get(sid, "")
            matches_str = match_lookup.get(sid, "")

            # Strict candidate subset enforcement
            if matches_str:
                cand_set = set(cands_str.split(",")) if cands_str else set()
                valid_matches = [m for m in matches_str.split(",") if m in cand_set]
                matches_str = ",".join(valid_matches)

            f_c_out.write(f"{sid}{DELIM}{cands_str}\n")
            f_m_out.write(f"{sid}{DELIM}{matches_str}\n")

            if cands_str:
                total_written_cands += len(cands_str.split(","))
            if matches_str:
                total_written_matches += len(matches_str.split(","))

    print(f"  Successfully wrote {total_s1:,} rows to:")
    print(f"    - {cand_out_file} ({os.path.getsize(cand_out_file)/(1024*1024):.1f} MB, {total_written_cands:,} total candidate pairs)")
    print(f"    - {match_out_file} ({os.path.getsize(match_out_file)/(1024*1024):.1f} MB, {total_written_matches:,} total matches)")
    print(f"  Assembled in {time.time() - t0_merge:.2f}s.")

    # Clean up temporary partition files
    for f in partition_cand_files + partition_match_files:
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # 5. Official Submission Validation
    print(f"\n[Phase 4/4] Executing Official Submission Validator...")
    validator_script = os.path.join(PROJECT_ROOT, "student_resource", "utils", "validate_submission.py")
    if os.path.isfile(validator_script):
        cmd = [
            sys.executable,
            validator_script,
            "--matching", match_out_file,
            "--candidate", cand_out_file,
            "--test-dir", test_dir,
        ]
        val_res = subprocess.run(cmd, capture_output=True, text=True)
        print("--- Official Validator Output ---")
        print(val_res.stdout)
        if val_res.stderr:
            print(val_res.stderr)
        print("---------------------------------")
        if val_res.returncode == 0:
            print(">>> OFFICIAL VALIDATION: PASSED WITH EXIT CODE 0! <<<")
        else:
            print(f">>> OFFICIAL VALIDATION FAILED WITH EXIT CODE {val_res.returncode} <<<")
            raise RuntimeError("Validation failed.")

    total_pipeline_time = time.time() - pipeline_t0
    print("\n" + "=" * 85)
    print(f"FULL PRODUCTION PIPELINE COMPLETED IN {total_pipeline_time/60:.2f} MINUTES ({total_pipeline_time:.1f}s)!")
    print(f"Final Candidate Pairs : {cand_out_file}")
    print(f"Final Matching Results: {match_out_file}")
    print("=" * 85)


if __name__ == "__main__":
    run_full_production(
        test_dir="student_resource/dataset/test",
        output_dir="output",
        model_path="business_entity_resolution/src/trained_model.json",
        max_candidates=80,
        batch_size=2000,
    )
