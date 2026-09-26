"""
Benchmark script for the optimized SageMaker entity resolution pipeline.
Evaluates 10,000 test Source 1 records covering the entire optimized path:
compact in-memory inverted index, cheap pre-filtering, vectorized feature extraction,
batch FastLogisticRegression scoring, and validation.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import (
    DELIM,
    CANDIDATE_HEADER,
    MATCHING_HEADER,
    format_id_list,
    stream_tsv_records,
    validate_submission_files,
)
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.model import FastLogisticRegression
from sagemaker.vectorized_matcher import CompactInvertedIndex, process_s1_batch_vectorized


def run_benchmark(
    n_s1: int = 10000,
    n_cands: int = 300000,
    batch_size: int = 2000,
    output_dir: str = "output",
):
    print("=" * 75)
    print("AMAZON ML CHALLENGE 2026 — OPTIMIZED PIPELINE BENCHMARK (10k S1)")
    print("=" * 75)

    test_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    model_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")

    os.makedirs(output_dir, exist_ok=True)
    bench_cand_out = os.path.join(output_dir, "benchmark_candidate_pairs.tsv")
    bench_match_out = os.path.join(output_dir, "benchmark_matching_results.tsv")

    # 1. Load trained model
    with open(model_path, "r", encoding="utf-8") as f:
        mdata = json.load(f)
    model = FastLogisticRegression()
    model.weights = list(mdata["weights"])
    model.bias = float(mdata["bias"])
    threshold = float(mdata.get("optimal_threshold", 0.50))
    print(f"Loaded Model: FastLogisticRegression (Threshold: {threshold:.2f}, Saved Macro F0.5: {mdata.get('macro_f05', 0):.4f})")

    # 2. Ingest candidate pool into CompactInvertedIndex
    print(f"\n[Step 1] Ingesting {n_cands:,} Source 2/3 candidate records into compact in-memory index...")
    t_idx_start = time.time()
    index = CompactInvertedIndex(max_bucket_size=300)

    half_cands = n_cands // 2
    for p, limit in ((s2_path, half_cands), (s3_path, half_cands)):
        for rec in stream_tsv_records(p, max_records=limit):
            index.add_record(rec, strategy="combined")

    idx_time = time.time() - t_idx_start
    print(f"Index built in {idx_time:.2f}s ({len(index):,} candidate records, {len(index.key_to_cand_idxs):,} unique keys, {len(index)/idx_time:.0f} rec/s)")

    # 3. Stream and process 10k S1 records
    print(f"\n[Step 2] Processing {n_s1:,} test Source 1 records in batches of {batch_size}...")
    t_match_start = time.time()

    cand_counts = []
    total_matches = 0
    total_singletons = 0
    s1_processed = 0

    with open(bench_cand_out, "w", encoding="utf-8", newline="") as f_cand, \
         open(bench_match_out, "w", encoding="utf-8", newline="") as f_match:

        f_cand.write(CANDIDATE_HEADER)
        f_match.write(MATCHING_HEADER)

        s1_batch = []
        for raw_s1 in stream_tsv_records(s1_path, max_records=n_s1):
            s1_batch.append(raw_s1)

            if len(s1_batch) >= batch_size:
                results = process_s1_batch_vectorized(
                    s1_batch,
                    index,
                    model,
                    threshold,
                    blocking_strategy="combined",
                )
                for s1_id, c_eids, m_eids in results:
                    f_cand.write(f"{s1_id}{DELIM}{format_id_list(c_eids)}\n")
                    f_match.write(f"{s1_id}{DELIM}{format_id_list(m_eids)}\n")
                    cand_counts.append(len(c_eids))
                    if m_eids:
                        total_matches += len(m_eids)
                    else:
                        total_singletons += 1
                    s1_processed += 1

                s1_batch.clear()
                elapsed = time.time() - t_match_start
                rate = s1_processed / max(elapsed, 0.001)
                print(f"  Processed {s1_processed:,}/{n_s1:,} S1 records ({rate:.1f} S1/sec)...")

        if s1_batch:
            results = process_s1_batch_vectorized(
                s1_batch,
                index,
                model,
                threshold,
                blocking_strategy="combined",
            )
            for s1_id, c_eids, m_eids in results:
                f_cand.write(f"{s1_id}{DELIM}{format_id_list(c_eids)}\n")
                f_match.write(f"{s1_id}{DELIM}{format_id_list(m_eids)}\n")
                cand_counts.append(len(c_eids))
                if m_eids:
                    total_matches += len(m_eids)
                else:
                    total_singletons += 1
                s1_processed += 1
            s1_batch.clear()

    match_time = time.time() - t_match_start
    throughput = s1_processed / max(match_time, 0.001)

    print("\n" + "=" * 75)
    print("BENCHMARK RESULTS & METRICS")
    print("=" * 75)
    print(f"Total S1 Processed      : {s1_processed:,}")
    print(f"Execution Time          : {match_time:.2f} seconds")
    print(f"Single-Core Throughput  : {throughput:.1f} S1 records/second")
    print(f"Total Candidates Found  : {sum(cand_counts):,}")
    print(f"Average Candidates/S1   : {np.mean(cand_counts):.1f}")
    print(f"Median Candidates/S1    : {np.median(cand_counts):.1f}")
    print(f"P95 Candidates/S1       : {np.percentile(cand_counts, 95):.1f}")
    print(f"Total Matches Found     : {total_matches:,} ({total_matches/s1_processed:.2f} matches/S1)")
    print(f"Singleton (No-match) S1 : {total_singletons:,} ({total_singletons/s1_processed*100:.1f}%)")

    # 4. Extrapolations for full test set (1,732,544 records)
    FULL_S1_COUNT = 1732544
    print("\n" + "=" * 75)
    print("ESTIMATED RUNTIME FOR FULL 1,732,544 TEST S1 DATASET")
    print("=" * 75)
    core_configs = [
        (1, "1 vCPU (Local Core)"),
        (4, "4 vCPUs (Local / ml.c5.xlarge)"),
        (16, "16 vCPUs (ml.c5.4xlarge)"),
        (36, "36 vCPUs (ml.c5.9xlarge)"),
        (72, "72 vCPUs (ml.c5.18xlarge / 4x ml.c5.4xlarge cluster)"),
    ]

    for cores, desc in core_configs:
        est_sec = FULL_S1_COUNT / (throughput * cores)
        if est_sec < 60:
            time_str = f"{est_sec:.1f} seconds"
        elif est_sec < 3600:
            time_str = f"{est_sec / 60:.1f} minutes"
        else:
            time_str = f"{est_sec / 3600:.2f} hours ({est_sec / 60:.1f} min)"
        print(f"  * {desc:<45}: {time_str}")

    print("=" * 75)

    # 5. Format validation
    print("\n[Step 3] Validating submission formatting rules...")
    errors = []
    
    # Check matching file format
    s1_in_match = set()
    match_pairs = {}
    with open(bench_match_out, "r", encoding="utf-8") as f:
        header = f.readline()
        if header != MATCHING_HEADER:
            errors.append(f"Invalid matching header: {repr(header)}")
        for line_no, line in enumerate(f, 2):
            parts = line.split(DELIM)
            if len(parts) != 2:
                errors.append(f"Matching line {line_no} does not have 2 columns")
                continue
            s1_id, m_str = parts[0], parts[1].strip()
            if s1_id in s1_in_match:
                errors.append(f"Duplicate S1 ID in matching: {s1_id}")
            s1_in_match.add(s1_id)
            match_pairs[s1_id] = set(m_str.split(",")) if m_str else set()

    # Check candidate file format
    s1_in_cand = set()
    with open(bench_cand_out, "r", encoding="utf-8") as f:
        header = f.readline()
        if header != CANDIDATE_HEADER:
            errors.append(f"Invalid candidate header: {repr(header)}")
        for line_no, line in enumerate(f, 2):
            parts = line.split(DELIM)
            if len(parts) != 2:
                errors.append(f"Candidate line {line_no} does not have 2 columns")
                continue
            s1_id, c_str = parts[0], parts[1].strip()
            if s1_id in s1_in_cand:
                errors.append(f"Duplicate S1 ID in candidates: {s1_id}")
            s1_in_cand.add(s1_id)
            c_set = set(c_str.split(",")) if c_str else set()
            # Check candidate subset rule
            m_set = match_pairs.get(s1_id, set())
            if not m_set.issubset(c_set):
                errors.append(f"S1 {s1_id}: Matched IDs {m_set - c_set} not in candidates!")

    is_valid = len(errors) == 0
    print(f"Submission Format Valid: {is_valid}")
    if not is_valid:
        for err in errors[:5]:
            print(f"  [Error] {err}")
    else:
        print(f"  All {len(s1_in_match):,} records verified against official competition rules (PASS)")
    print(f"Candidate file size : {os.path.getsize(bench_cand_out) / 1024:.1f} KB")
    print(f"Matching file size  : {os.path.getsize(bench_match_out) / 1024:.1f} KB")

    return {
        "s1_count": s1_processed,
        "throughput": throughput,
        "match_time": match_time,
        "mean_cands": float(np.mean(cand_counts)),
        "p95_cands": float(np.percentile(cand_counts, 95)),
        "matches": total_matches,
        "singletons": total_singletons,
        "is_valid": is_valid,
    }


if __name__ == "__main__":
    run_benchmark(n_s1=10000, n_cands=300000, batch_size=2000)
