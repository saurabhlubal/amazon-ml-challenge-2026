"""
SageMaker Processing Container Entrypoint for Amazon ML Challenge 2026.
Executes vectorized, high-throughput candidate generation and matching on assigned S1 shard(s).
Runs with bounded memory and multi-core CPU utilization.
"""

import os
import sys
import shutil
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# If 'sagemaker' package is in sys.modules (from installed AWS SageMaker SDK),
# extend its __path__ so 'sagemaker.vectorized_matcher' can also be resolved
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

try:
    from vectorized_matcher import (
        CompactInvertedIndex,
        process_s1_batch_vectorized,
        merge_chunk_outputs,
    )
except ImportError:
    from sagemaker.vectorized_matcher import (
        CompactInvertedIndex,
        process_s1_batch_vectorized,
        merge_chunk_outputs,
    )


def load_model(model_path: str):
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = FastLogisticRegression()
    model.weights = list(data["weights"])
    model.bias = float(data["bias"])
    threshold = float(data.get("optimal_threshold", 0.50))
    return model, threshold


def partition_candidates_to_disk(
    cand_paths: List[str],
    cand_shard_dir: str,
    cand_chunk_size: int = 1250000,
    max_cand_records: Optional[int] = None,
    country_filter: Optional[str] = None,
) -> List[str]:
    """
    Split Source 2 & Source 3 into bounded disk-backed chunk files.
    Each file has at most cand_chunk_size records with a standard TSV header.
    Runs with near-zero memory footprint and O(1) streaming IO.
    """
    os.makedirs(cand_shard_dir, exist_ok=True)
    chunk_files = []

    chunk_idx = 0
    records_in_chunk = 0
    total_indexed = 0
    current_fp = None

    per_source_limit = (max_cand_records // max(1, len(cand_paths))) if max_cand_records else None

    def open_next_chunk():
        nonlocal chunk_idx, current_fp, records_in_chunk
        if current_fp:
            current_fp.close()
        p = os.path.join(cand_shard_dir, f"cand_chunk_{chunk_idx:03d}.tsv")
        chunk_files.append(p)
        current_fp = open(p, "w", encoding="utf-8", newline="")
        current_fp.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        chunk_idx += 1
        records_in_chunk = 0

    for cand_path in cand_paths:
        if not os.path.isfile(cand_path):
            continue
        source_count = 0
        with open(cand_path, "r", encoding="utf-8") as f_in:
            next(f_in, None)  # Skip header
            for line in f_in:
                if not line.strip():
                    continue
                if country_filter:
                    parts = line.split("\t")
                    if len(parts) >= 4:
                        c_country = parts[3].strip().upper()
                        if c_country and c_country != country_filter:
                            continue

                if current_fp is None or records_in_chunk >= cand_chunk_size:
                    open_next_chunk()

                current_fp.write(line)
                records_in_chunk += 1
                total_indexed += 1
                source_count += 1

                if per_source_limit and source_count >= per_source_limit:
                    break

    if current_fp:
        current_fp.close()

    return chunk_files


def run_shard_processing(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    model_path: str,
    output_dir: str,
    shard_id: int = 0,
    total_shards: int = 1,
    batch_size: int = 2000,
    cand_chunk_size: int = 1250000,
    max_s1_records: Optional[int] = None,
    max_cand_records: Optional[int] = None,
    country_filter: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    cand_out = os.path.join(output_dir, f"candidate_pairs_part_{shard_id:03d}.tsv")
    match_out = os.path.join(output_dir, f"matching_results_part_{shard_id:03d}.tsv")

    print(f"\n[Shard {shard_id}/{total_shards}] Initializing memory-bounded pipeline...")
    print(f"  Source 1 Input : {s1_path}")
    print(f"  Candidate Out  : {cand_out}")
    print(f"  Matching Out   : {match_out}")
    print(f"  Cand Chunk Size: {cand_chunk_size:,} records")
    if country_filter:
        print(f"  Country Filter : {country_filter}")

    # 1. Load model
    model, threshold = load_model(model_path)
    print(f"  Loaded model from {model_path} (Threshold: {threshold:.2f})")

    # 2. Partition candidates to bounded disk shards
    print("\n[Partitioning] Splitting candidate records into bounded disk shards...")
    t_part_start = time.time()
    cand_shard_dir = os.path.join(output_dir, f"cand_shards_w{shard_id}")
    cand_files = partition_candidates_to_disk(
        cand_paths=[s2_path, s3_path],
        cand_shard_dir=cand_shard_dir,
        cand_chunk_size=cand_chunk_size,
        max_cand_records=max_cand_records,
        country_filter=country_filter,
    )
    num_cand_chunks = len(cand_files)
    print(f"[Partitioning] Created {num_cand_chunks} candidate chunk(s) in {time.time() - t_part_start:.2f}s")

    # 3. Stream S1 against each candidate chunk sequentially
    tmp_cand_files = []
    tmp_match_files = []
    t_match_total_start = time.time()

    for k, cand_file in enumerate(cand_files):
        print(f"\n[Chunk {k+1}/{num_cand_chunks}] Building index from {os.path.basename(cand_file)}...")
        t_idx_start = time.time()
        index = CompactInvertedIndex(max_bucket_size=300)
        for rec in stream_tsv_records(cand_file):
            index.add_record(rec, strategy="combined")
        idx_time = time.time() - t_idx_start
        print(f"  Index built: {len(index):,} candidates, {len(index.key_to_cand_idxs):,} keys in {idx_time:.2f}s")

        tmp_cand = os.path.join(output_dir, f"tmp_cand_w{shard_id}_c{k}.tsv")
        tmp_match = os.path.join(output_dir, f"tmp_match_w{shard_id}_c{k}.tsv")
        tmp_cand_files.append(tmp_cand)
        tmp_match_files.append(tmp_match)

        t_chunk_match = time.time()
        chunk_s1 = 0
        chunk_cands = 0
        chunk_matches = 0

        with open(tmp_cand, "w", encoding="utf-8", newline="") as f_c, \
             open(tmp_match, "w", encoding="utf-8", newline="") as f_m:

            s1_batch = []
            row_idx = 0

            for raw_rec in stream_tsv_records(s1_path, max_records=max_s1_records):
                if total_shards > 1 and (row_idx % total_shards) != shard_id:
                    row_idx += 1
                    continue
                row_idx += 1

                if country_filter:
                    s1_country = raw_rec.get("country", "").strip().upper()
                    if s1_country and s1_country != country_filter:
                        continue

                s1_batch.append(raw_rec)

                if len(s1_batch) >= batch_size:
                    results = process_s1_batch_vectorized(
                        s1_batch,
                        index,
                        model,
                        threshold,
                        blocking_strategy="combined",
                    )
                    for s1_id, c_eids, m_eids in results:
                        f_c.write(f"{s1_id}\t{','.join(c_eids)}\n")
                        f_m.write(f"{s1_id}\t{','.join(m_eids)}\n")
                        chunk_s1 += 1
                        chunk_cands += len(c_eids)
                        chunk_matches += len(m_eids)

                    s1_batch.clear()
                    if max_s1_records and chunk_s1 >= max_s1_records:
                        break

            if s1_batch:
                results = process_s1_batch_vectorized(
                    s1_batch,
                    index,
                    model,
                    threshold,
                    blocking_strategy="combined",
                )
                for s1_id, c_eids, m_eids in results:
                    f_c.write(f"{s1_id}\t{','.join(c_eids)}\n")
                    f_m.write(f"{s1_id}\t{','.join(m_eids)}\n")
                    chunk_s1 += 1
                    chunk_cands += len(c_eids)
                    chunk_matches += len(m_eids)
                s1_batch.clear()

        # Free index memory immediately
        del index
        import gc
        gc.collect()

        # Delete chunk candidate TSV to reclaim disk space
        if os.path.isfile(cand_file):
            try:
                os.remove(cand_file)
            except OSError:
                pass

        elapsed_c = time.time() - t_chunk_match
        rate_c = chunk_s1 / max(elapsed_c, 0.001)
        print(f"  [Chunk {k+1}/{num_cand_chunks}] Evaluated {chunk_s1:,} S1 records in {elapsed_c:.2f}s ({rate_c:.1f} rec/s, matches: {chunk_matches:,})")

    # Clean up cand_shard_dir
    try:
        shutil.rmtree(cand_shard_dir, ignore_errors=True)
    except Exception:
        pass

    # 4. Deterministic Streaming K-Way Merge
    print(f"\n[Merging] Streaming merge of {num_cand_chunks} chunk outputs into final submission shards...")
    t_merge_start = time.time()
    total_s1, total_candidates, total_matches = merge_chunk_outputs(
        tmp_cand_files,
        tmp_match_files,
        cand_out,
        match_out,
    )
    print(f"[Merging] Complete in {time.time() - t_merge_start:.2f}s")

    # 5. Remove temporary partial chunk files
    for p in tmp_cand_files + tmp_match_files:
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass

    total_time = time.time() - t_match_total_start
    overall_rate = total_s1 / max(total_time, 0.001)
    print(f"\n[Shard {shard_id}] COMPLETE: {total_s1:,} S1 records in {total_time:.2f}s ({overall_rate:.1f} rec/s)")
    print(f"  Total Candidates Written: {total_candidates:,} (avg {total_candidates/max(1, total_s1):.1f}/S1)")
    print(f"  Total Matches Found     : {total_matches:,} (avg {total_matches/max(1, total_s1):.2f}/S1)")
    print(f"  Candidate Output Size   : {os.path.getsize(cand_out) / 1024 / 1024:.2f} MB")
    print(f"  Matching Output Size    : {os.path.getsize(match_out) / 1024 / 1024:.2f} MB")
    return total_s1, total_candidates, total_matches



def resolve_sagemaker_cluster_config(shard_id: int, total_shards: int) -> Tuple[int, int]:
    """
    Auto-detect shard index from SageMaker cluster configuration if running multi-instance.
    """
    config_path = "/opt/ml/config/resourceconfig.json"
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                res_cfg = json.load(f)
            hosts = sorted(res_cfg.get("hosts", []))
            cur_host = res_cfg.get("current_host")
            if cur_host in hosts and len(hosts) > 1:
                auto_shard = hosts.index(cur_host)
                auto_total = len(hosts)
                print(f"[SageMaker Cluster Discovery] Detected node {cur_host} -> Shard {auto_shard}/{auto_total}")
                return auto_shard, auto_total
        except Exception as e:
            print(f"[Warning] Failed to read {config_path}: {e}")
    return shard_id, total_shards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker Worker Entrypoint for Entity Resolution")
    parser.add_argument("--s1-path", default="/opt/ml/processing/input/test_source1.tsv")
    parser.add_argument("--s2-path", default="/opt/ml/processing/input/test_source2.tsv")
    parser.add_argument("--s3-path", default="/opt/ml/processing/input/test_source3.tsv")
    parser.add_argument("--model-path", default="/opt/ml/processing/model/trained_model.json")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--cand-chunk-size", type=int, default=1250000)
    parser.add_argument("--max-s1-records", type=int, default=None)
    parser.add_argument("--max-cand-records", type=int, default=None)
    parser.add_argument("--country-filter", type=str, default=None)

    args = parser.parse_args()

    # Auto-detect cluster topology if running inside SageMaker multi-instance job
    resolved_shard_id, resolved_total_shards = resolve_sagemaker_cluster_config(args.shard_id, args.total_shards)

    # Fallback to local paths ONLY if container input paths don't exist
    if not os.path.exists(args.s1_path):
        args.s1_path = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test", "test_source1.tsv")
    if not os.path.exists(args.s2_path):
        args.s2_path = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test", "test_source2.tsv")
    if not os.path.exists(args.s3_path):
        args.s3_path = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test", "test_source3.tsv")
    if not os.path.exists(args.model_path):
        args.model_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")
    
    # Only fallback to local output directory if /opt/ml/processing does NOT exist (local dev mode)
    if args.output_dir.startswith("/opt/ml") and not os.path.exists("/opt/ml/processing"):
        args.output_dir = os.path.join(PROJECT_ROOT, "output")

    run_shard_processing(
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        shard_id=resolved_shard_id,
        total_shards=resolved_total_shards,
        batch_size=args.batch_size,
        cand_chunk_size=args.cand_chunk_size,
        max_s1_records=args.max_s1_records,
        max_cand_records=args.max_cand_records,
        country_filter=args.country_filter,
    )

