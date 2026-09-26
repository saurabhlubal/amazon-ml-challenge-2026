"""
Benchmark and Validation Script for Memory-Bounded Pipeline.
Evaluates 10,000 Source 1 records against candidate partitions.
Measures:
1. Peak RAM (via psutil process telemetry)
2. S1 throughput (records/sec)
3. Candidate recall parity against monolithic benchmark
4. Projected full-run runtime on 2 x ml.t3.xlarge
5. Official submission validator execution (student_resource/utils/validate_submission.py)
"""

import os
import sys
import time
import json
import shutil
import threading
import subprocess
import numpy as np
import psutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "sagemaker" in sys.modules:
    sagemaker_pkg = sys.modules["sagemaker"]
    if hasattr(sagemaker_pkg, "__path__") and CURRENT_DIR not in sagemaker_pkg.__path__:
        sagemaker_pkg.__path__.insert(0, CURRENT_DIR)

from scripts.pipeline_utils import (
    DELIM,
    CANDIDATE_HEADER,
    MATCHING_HEADER,
    format_id_list,
    stream_tsv_records,
)
from business_entity_resolution.src.model import FastLogisticRegression
from sagemaker.vectorized_matcher import (
    CompactInvertedIndex,
    process_s1_batch_vectorized,
    merge_chunk_outputs,
)
from sagemaker.entrypoint import load_model, partition_candidates_to_disk


class MemoryTracker:
    """Background memory monitor using psutil RSS."""
    def __init__(self, interval_sec: float = 0.05):
        self.interval_sec = interval_sec
        self.peak_rss_mb = 0.0
        self.running = False
        self._thread = None
        self._proc = psutil.Process()

    def _monitor(self):
        while self.running:
            try:
                rss = self._proc.memory_info().rss / 1024 / 1024
                if rss > self.peak_rss_mb:
                    self.peak_rss_mb = rss
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self):
        self.peak_rss_mb = self._proc.memory_info().rss / 1024 / 1024
        self.running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        final_rss = self._proc.memory_info().rss / 1024 / 1024
        if final_rss > self.peak_rss_mb:
            self.peak_rss_mb = final_rss
        return self.peak_rss_mb


def run_benchmark(
    n_s1: int = 10000,
    n_cands: int = 300000,
    cand_chunk_size: int = 100000,
    batch_size: int = 2000,
    bench_dir: str = "output/benchmark_bounded",
):
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — MEMORY-BOUNDED PIPELINE BENCHMARK")
    print(f"Configuration: {n_s1:,} S1 records | {n_cands:,} Candidate pool | Chunk size: {cand_chunk_size:,}")
    print("=" * 80)

    test_dir = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    model_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")

    shutil.rmtree(bench_dir, ignore_errors=True)
    os.makedirs(bench_dir, exist_ok=True)

    # 1. Load trained model
    model, threshold = load_model(model_path)
    print(f"Loaded Model: FastLogisticRegression (Threshold: {threshold:.2f})")

    # 2. Extract representative S1 records
    print(f"\n[Step 1] Loading {n_s1:,} representative Source 1 records...")
    s1_records = list(stream_tsv_records(s1_path, max_records=n_s1))
    print(f"  Loaded {len(s1_records):,} S1 records.")

    # -------------------------------------------------------------
    # BASELINE: Monolithic Index (Existing Benchmark)
    # -------------------------------------------------------------
    print(f"\n[Step 2] Running Baseline Monolithic Index ({n_cands:,} candidates in single index)...")
    mono_mem = MemoryTracker()
    mono_mem.start()
    t_mono_idx_start = time.time()
    mono_index = CompactInvertedIndex(max_bucket_size=300)

    half_cands = n_cands // 2
    for p, limit in ((s2_path, half_cands), (s3_path, half_cands)):
        for rec in stream_tsv_records(p, max_records=limit):
            mono_index.add_record(rec, strategy="combined")

    mono_idx_time = time.time() - t_mono_idx_start
    print(f"  Monolithic Index built in {mono_idx_time:.2f}s ({len(mono_index):,} records, {len(mono_index.key_to_cand_idxs):,} keys)")

    t_mono_match_start = time.time()
    mono_results = process_s1_batch_vectorized(
        s1_records,
        mono_index,
        model,
        threshold,
        blocking_strategy="combined",
    )
    mono_match_time = time.time() - t_mono_match_start
    mono_peak_ram = mono_mem.stop()
    mono_rate = len(s1_records) / max(mono_match_time, 0.001)

    baseline_cands = {sid: set(c) for sid, c, _ in mono_results}
    baseline_matches = {sid: set(m) for sid, _, m in mono_results}
    total_baseline_cands = sum(len(c) for c in baseline_cands.values())
    total_baseline_matches = sum(len(m) for m in baseline_matches.values())

    print(f"  Monolithic Baseline Results:")
    print(f"    Match Time           : {mono_match_time:.2f}s ({mono_rate:.1f} S1/s)")
    print(f"    Peak RAM             : {mono_peak_ram:.1f} MB ({mono_peak_ram / 1024:.2f} GB)")
    print(f"    Total Candidates     : {total_baseline_cands:,} (avg {total_baseline_cands / len(s1_records):.1f}/S1)")
    print(f"    Total Matches Found  : {total_baseline_matches:,}")

    # Free monolithic index
    del mono_index
    del mono_results
    import gc
    gc.collect()

    # -------------------------------------------------------------
    # NEW APPROACH: Bounded Sequential Partition Pipeline
    # -------------------------------------------------------------
    print(f"\n[Step 3] Running New Memory-Bounded Pipeline (Cand chunk size: {cand_chunk_size:,})...")
    bounded_mem = MemoryTracker()
    bounded_mem.start()
    t_bounded_start = time.time()

    # Partition candidates to bounded disk shards
    cand_shard_dir = os.path.join(bench_dir, "cand_shards")
    cand_files = partition_candidates_to_disk(
        cand_paths=[s2_path, s3_path],
        cand_shard_dir=cand_shard_dir,
        cand_chunk_size=cand_chunk_size,
        max_cand_records=n_cands,
    )
    num_chunks = len(cand_files)
    print(f"  Partitioned candidate pool into {num_chunks} disk chunks of ~{cand_chunk_size:,} records.")

    tmp_cand_files = []
    tmp_match_files = []
    chunk_times = []

    for k, c_path in enumerate(cand_files):
        t_c_start = time.time()
        idx = CompactInvertedIndex(max_bucket_size=300)
        for rec in stream_tsv_records(c_path):
            idx.add_record(rec, strategy="combined")

        tmp_c = os.path.join(bench_dir, f"tmp_cand_chunk_{k}.tsv")
        tmp_m = os.path.join(bench_dir, f"tmp_match_chunk_{k}.tsv")
        tmp_cand_files.append(tmp_c)
        tmp_match_files.append(tmp_m)

        with open(tmp_c, "w", encoding="utf-8", newline="") as fc, \
             open(tmp_m, "w", encoding="utf-8", newline="") as fm:
            for b_idx in range(0, len(s1_records), batch_size):
                batch = s1_records[b_idx : b_idx + batch_size]
                res = process_s1_batch_vectorized(batch, idx, model, threshold, "combined")
                for sid, c_eids, m_eids in res:
                    fc.write(f"{sid}\t{','.join(c_eids)}\n")
                    fm.write(f"{sid}\t{','.join(m_eids)}\n")

        del idx
        gc.collect()
        c_elapsed = time.time() - t_c_start
        chunk_times.append(c_elapsed)
        print(f"    Chunk {k+1}/{num_chunks}: Indexed and evaluated 10k S1 in {c_elapsed:.2f}s")

    # Streaming K-way merge
    final_cand_path = os.path.join(bench_dir, "benchmark_candidate_pairs.tsv")
    final_match_path = os.path.join(bench_dir, "benchmark_matching_results.tsv")

    t_mrg_start = time.time()
    tot_s1, tot_cands, tot_matches = merge_chunk_outputs(
        tmp_cand_files,
        tmp_match_files,
        final_cand_path,
        final_match_path,
    )
    mrg_time = time.time() - t_mrg_start
    total_bounded_time = time.time() - t_bounded_start
    bounded_peak_ram = bounded_mem.stop()

    bounded_throughput = tot_s1 / max(total_bounded_time, 0.001)

    print(f"\n  Bounded Pipeline Complete:")
    print(f"    Total Runtime        : {total_bounded_time:.2f}s (Merge: {mrg_time:.2f}s)")
    print(f"    Overall Throughput   : {bounded_throughput:.1f} S1 records/second")
    print(f"    Peak RAM (psutil)    : {bounded_peak_ram:.1f} MB ({bounded_peak_ram / 1024:.2f} GB)")
    print(f"    Total Candidates     : {tot_cands:,} (avg {tot_cands / tot_s1:.1f}/S1)")
    print(f"    Total Matches Found  : {tot_matches:,} (avg {tot_matches / tot_s1:.2f}/S1)")

    # -------------------------------------------------------------
    # 4. RECALL & PARITY VERIFICATION
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CANDIDATE RECALL & PARITY ANALYSIS")
    print("=" * 80)

    # Read bounded outputs back
    bounded_cands = {}
    with open(final_cand_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split(DELIM)
            bounded_cands[parts[0]] = set(parts[1].strip().split(",")) if parts[1].strip() else set()

    bounded_matches = {}
    with open(final_match_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split(DELIM)
            bounded_matches[parts[0]] = set(parts[1].strip().split(",")) if parts[1].strip() else set()

    # Candidate recall against monolithic benchmark:
    found_cands_count = 0
    for sid, base_c in baseline_cands.items():
        found_cands_count += len(base_c.intersection(bounded_cands.get(sid, set())))
    cand_recall = found_cands_count / max(1, total_baseline_cands)

    found_matches_count = 0
    for sid, base_m in baseline_matches.items():
        found_matches_count += len(base_m.intersection(bounded_matches.get(sid, set())))
    match_recall = found_matches_count / max(1, total_baseline_matches)

    subset_verified = all(
        bounded_matches[sid].issubset(bounded_cands[sid])
        for sid in bounded_matches
    )

    print(f"Candidate Recall vs Baseline Monolithic Benchmark: {cand_recall * 100:.2f}% ({found_cands_count:,}/{total_baseline_cands:,})")
    print(f"Match Recall vs Baseline Monolithic Benchmark    : {match_recall * 100:.2f}% ({found_matches_count:,}/{total_baseline_matches:,})")
    print(f"Strict Candidate-Subset Verification             : {'PASS (100% Valid)' if subset_verified else 'FAIL'}")

    # -------------------------------------------------------------
    # 5. FULL TEST RUN PROJECTIONS (1,732,544 S1 on 2 x ml.t3.xlarge)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PROJECTED FULL TEST RUNTIME (1,732,544 S1 ON 2 x ml.t3.xlarge)")
    print("=" * 80)
    FULL_S1 = 1732544
    S1_PER_WORKER = FULL_S1 // 2  # 866,272 S1 per node

    # Worker runtime is S1_PER_WORKER / throughput
    proj_sec_1_node = FULL_S1 / bounded_throughput
    proj_sec_2_nodes = S1_PER_WORKER / bounded_throughput

    print(f"S1 Throughput (Single Instance): {bounded_throughput:.1f} records/second")
    print(f"Projected Runtime (1 Instance) : {proj_sec_1_node / 60:.1f} minutes ({proj_sec_1_node / 3600:.2f} hours)")
    print(f"Projected Runtime (2 Instances): {proj_sec_2_nodes / 60:.1f} minutes ({proj_sec_2_nodes / 3600:.2f} hours)")
    print(f"Peak RAM on 16 GB Instance     : {bounded_peak_ram:.1f} MB (Safety headroom: {16384 - bounded_peak_ram:.1f} MB free)")

    # -------------------------------------------------------------
    # 6. OFFICIAL COMPETITION VALIDATOR
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OFFICIAL SUBMISSION VALIDATOR EXECUTION")
    print("=" * 80)

    # Prepare temporary test directory containing test_source1.tsv with the 10k S1 records
    test_sub_dir = os.path.join(bench_dir, "test_subset")
    os.makedirs(test_sub_dir, exist_ok=True)
    sub_s1_path = os.path.join(test_sub_dir, "test_source1.tsv")

    with open(s1_path, "r", encoding="utf-8") as f_in, open(sub_s1_path, "w", encoding="utf-8", newline="") as f_out:
        header = f_in.readline()
        f_out.write(header)
        for i, line in enumerate(f_in):
            if i >= n_s1:
                break
            f_out.write(line)

    validator_py = os.path.join(PROJECT_ROOT, "student_resource", "utils", "validate_submission.py")
    cmd = [
        sys.executable,
        validator_py,
        "--matching", final_match_path,
        "--candidate", final_cand_path,
        "--test-dir", test_sub_dir,
    ]

    print(f"Executing official validator:")
    print(f"  Command: {' '.join(cmd)}")
    v_res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n--- Validator Output ---")
    print(v_res.stdout)
    if v_res.stderr:
        print("--- Validator Errors ---")
        print(v_res.stderr)

    validator_passed = (v_res.returncode == 0)
    print(f"Official Validator Exit Code: {v_res.returncode} -> {'PASS' if validator_passed else 'FAIL'}")

    return {
        "n_s1": n_s1,
        "n_cands": n_cands,
        "cand_chunk_size": cand_chunk_size,
        "peak_ram_mb": bounded_peak_ram,
        "throughput": bounded_throughput,
        "cand_recall": cand_recall,
        "match_recall": match_recall,
        "subset_verified": subset_verified,
        "proj_min_2x_t3xlarge": proj_sec_2_nodes / 60,
        "validator_passed": validator_passed,
    }


if __name__ == "__main__":
    run_benchmark(n_s1=10000, n_cands=300000, cand_chunk_size=100000, batch_size=2000)
