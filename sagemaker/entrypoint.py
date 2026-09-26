"""
SageMaker Processing Container Entrypoint for Amazon ML Challenge 2026.
Executes vectorized, high-throughput candidate generation and matching on assigned S1 shard(s).
Runs with bounded memory and multi-core CPU utilization.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional

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
from business_entity_resolution.src.model import FastLogisticRegression
from sagemaker.vectorized_matcher import CompactInvertedIndex, process_s1_batch_vectorized


def load_model(model_path: str):
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = FastLogisticRegression()
    model.weights = list(data["weights"])
    model.bias = float(data["bias"])
    threshold = float(data.get("optimal_threshold", 0.50))
    return model, threshold


def run_shard_processing(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    model_path: str,
    output_dir: str,
    shard_id: int = 0,
    total_shards: int = 1,
    batch_size: int = 2000,
    max_s1_records: Optional[int] = None,
    max_cand_records: Optional[int] = None,
    country_filter: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    cand_out = os.path.join(output_dir, f"candidate_pairs_part_{shard_id:03d}.tsv")
    match_out = os.path.join(output_dir, f"matching_results_part_{shard_id:03d}.tsv")

    print(f"\n[Shard {shard_id}/{total_shards}] Initializing pipeline...")
    print(f"  Source 1 Input : {s1_path}")
    print(f"  Candidate Out  : {cand_out}")
    print(f"  Matching Out   : {match_out}")
    if country_filter:
        print(f"  Country Filter : {country_filter}")

    # 1. Load model
    model, threshold = load_model(model_path)
    print(f"  Loaded model from {model_path} (Threshold: {threshold:.2f})")

    # 2. Build in-memory index
    print("\n[Indexing] Ingesting Source 2 & 3 candidate records into compact in-memory index...")
    t_idx_start = time.time()
    index = CompactInvertedIndex(max_bucket_size=300)

    total_cands = 0
    for cand_path in (s2_path, s3_path):
        source_name = os.path.basename(cand_path)
        print(f"  Streaming {source_name}...")
        t_src = time.time()
        cnt = 0
        for rec in stream_tsv_records(cand_path, max_records=max_cand_records):
            if country_filter:
                c_country = rec.get("country", "").strip().upper()
                if c_country and c_country != country_filter:
                    continue

            index.add_record(rec, strategy="combined")
            cnt += 1
            total_cands += 1
            if max_cand_records and total_cands >= max_cand_records:
                break
        print(f"  {source_name}: indexed {cnt:,} records in {time.time() - t_src:.2f}s")
        if max_cand_records and total_cands >= max_cand_records:
            break

    print(f"[Indexing] Complete: {len(index):,} candidate records, {len(index.key_to_cand_idxs):,} keys ({time.time() - t_idx_start:.2f}s)")

    # 3. Stream and process assigned S1 shard
    print(f"\n[Matching] Processing Source 1 records (Batch size: {batch_size})...")
    t_match_start = time.time()

    total_s1 = 0
    total_matches = 0
    total_candidates = 0

    with open(cand_out, "w", encoding="utf-8", newline="") as f_cand, \
         open(match_out, "w", encoding="utf-8", newline="") as f_match:

        f_cand.write(CANDIDATE_HEADER)
        f_match.write(MATCHING_HEADER)

        s1_batch = []
        row_idx = 0

        for raw_rec in stream_tsv_records(s1_path, max_records=max_s1_records):
            # If input file is un-partitioned, shard by round-robin
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
                    f_cand.write(f"{s1_id}{DELIM}{format_id_list(c_eids)}\n")
                    f_match.write(f"{s1_id}{DELIM}{format_id_list(m_eids)}\n")
                    total_s1 += 1
                    total_candidates += len(c_eids)
                    total_matches += len(m_eids)

                s1_batch.clear()

                if total_s1 % 5000 == 0:
                    rate = total_s1 / (time.time() - t_match_start)
                    print(f"  [Shard {shard_id}] Processed {total_s1:,} S1 records ({rate:.1f} rec/s, matches: {total_matches:,})")

                if max_s1_records and total_s1 >= max_s1_records:
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
                f_cand.write(f"{s1_id}{DELIM}{format_id_list(c_eids)}\n")
                f_match.write(f"{s1_id}{DELIM}{format_id_list(m_eids)}\n")
                total_s1 += 1
                total_candidates += len(c_eids)
                total_matches += len(m_eids)
            s1_batch.clear()

    total_time = time.time() - t_match_start
    overall_rate = total_s1 / max(total_time, 0.001)
    print(f"\n[Shard {shard_id}] Complete: {total_s1:,} S1 records in {total_time:.2f}s ({overall_rate:.1f} rec/s)")
    print(f"  Total Candidates Written: {total_candidates:,} (avg {total_candidates/max(1, total_s1):.1f}/S1)")
    print(f"  Total Matches Found     : {total_matches:,} (avg {total_matches/max(1, total_s1):.2f}/S1)")
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
        max_s1_records=args.max_s1_records,
        max_cand_records=args.max_cand_records,
        country_filter=args.country_filter,
    )
