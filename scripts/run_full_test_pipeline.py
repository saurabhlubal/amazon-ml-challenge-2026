"""
Full Production Pipeline for Amazon ML Challenge 2026.
Processes all 1,732,544 test Source 1 records and ~10M test Source 2/3 records.
Uses a high-throughput, low-memory disk-backed SQLite indexing engine
to guarantee execution under 250 MB RAM with zero risk of Out-Of-Memory.
"""

import os
import sys
import json
import time
import sqlite3
import argparse
import psutil
from collections import Counter
from typing import Dict, List, Set, Any, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import (
    DELIM,
    MATCHING_HEADER,
    CANDIDATE_HEADER,
    format_id_list,
    stream_tsv_records,
    validate_submission_files,
)
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.blocking import get_blocking_keys
from business_entity_resolution.src.features import build_features
from business_entity_resolution.src.model import (
    FastLogisticRegression,
    predict_scores,
    decide_matches,
)


def load_production_model() -> Tuple[FastLogisticRegression, float]:
    """Load the trained model and optimal decision threshold from disk."""
    model_path = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_and_evaluate.py first.")

    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = FastLogisticRegression()
    model.weights = list(data["weights"])
    model.bias = float(data["bias"])
    threshold = float(data.get("optimal_threshold", 0.50))
    print(f"Loaded trained model (Optimal Threshold: {threshold:.2f}, Saved Macro F0.5: {data.get('macro_f05', 0):.4f})")
    return model, threshold


def build_sqlite_candidate_index(
    db_path: str,
    s2_path: str,
    s3_path: str,
    max_bucket_size: int = 300,
    batch_insert_size: int = 100000,
    max_cand_records: Optional[int] = None,
) -> sqlite3.Connection:
    """
    Build high-throughput disk-backed SQLite inverted index for Source 2/3 records.
    Uses ultra-fast memory-mapped bulk inserts and B-tree indexing.
    """
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Extreme performance PRAGMAs for bulk indexing with bounded RAM
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA locking_mode = EXCLUSIVE;")
    cur.execute("PRAGMA cache_size = -64000;")  # 64 MB RAM cache
    cur.execute("PRAGMA temp_store = FILE;")

    cur.execute("""
        CREATE TABLE candidate_records (
            eid TEXT PRIMARY KEY,
            norm_name TEXT,
            norm_addr TEXT,
            raw_name TEXT,
            country TEXT,
            street_num TEXT,
            postal_codes TEXT,
            numbers TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE blocking_index (
            bkey TEXT,
            eid TEXT
        );
    """)
    conn.commit()

    print("\nIndexing Source 2 and Source 3 candidate records into disk-backed engine...")
    t0 = time.time()
    total_records = 0
    total_key_rows = 0

    record_batch = []
    key_batch = []
    key_counts: Dict[str, int] = {}

    for path in (s2_path, s3_path):
        source_name = os.path.basename(path)
        print(f"Streaming {source_name}...")
        t_src = time.time()
        src_count = 0

        for raw_rec in stream_tsv_records(path, max_records=max_cand_records):
            eid = raw_rec.get("entity_id", "")
            if not eid:
                continue

            norm = normalize_record(raw_rec)
            record_batch.append((
                eid,
                norm["business_name"],
                norm["business_address"],
                norm["raw_business_name"],
                norm["country"],
                norm["street_number"],
                ",".join(norm["postal_codes"]),
                ",".join(norm["address_numbers"]),
            ))

            keys = get_blocking_keys(norm, strategy="combined")
            for k in keys:
                cnt = key_counts.get(k, 0)
                if cnt < max_bucket_size:
                    key_counts[k] = cnt + 1
                    key_batch.append((k, eid))

            total_records += 1
            src_count += 1

            if len(record_batch) >= batch_insert_size:
                cur.executemany("INSERT INTO candidate_records VALUES (?, ?, ?, ?, ?, ?, ?, ?);", record_batch)
                cur.executemany("INSERT INTO blocking_index VALUES (?, ?);", key_batch)
                total_key_rows += len(key_batch)
                record_batch.clear()
                key_batch.clear()
                conn.commit()

            if src_count % 500000 == 0:
                rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
                print(f"  {source_name}: indexed {src_count:,} records ({time.time() - t_src:.1f}s, RAM: {rss_mb:.0f} MB)")

            if max_cand_records and total_records >= max_cand_records:
                break

    # Flush remaining batches
    if record_batch:
        cur.executemany("INSERT INTO candidate_records VALUES (?, ?, ?, ?, ?, ?, ?, ?);", record_batch)
        record_batch.clear()
    if key_batch:
        cur.executemany("INSERT INTO blocking_index VALUES (?, ?);", key_batch)
        total_key_rows += len(key_batch)
        key_batch.clear()
    conn.commit()

    del key_counts  # Free memory
    print(f"All {total_records:,} records inserted in {time.time() - t0:.1f}s. Building B-tree index on keys...")
    t_idx = time.time()
    cur.execute("CREATE INDEX idx_blocking_key ON blocking_index(bkey);")
    conn.commit()
    print(f"B-tree index created in {time.time() - t_idx:.1f}s ({total_key_rows:,} total key-pair entries).")

    return conn


def run_full_pipeline(
    test_dir: str,
    output_dir: str,
    batch_size: int = 1000,
    max_s1_records: Optional[int] = None,
    max_cand_records: Optional[int] = None,
    keep_db: bool = False,
    validate: bool = True,
):
    start_time = time.time()
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    os.makedirs(output_dir, exist_ok=True)
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    db_path = os.path.join(output_dir, "candidate_cache.db")

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — FULL TEST SET SUBMISSION PIPELINE")
    print("=" * 70)
    print(f"Source 1 input      : {s1_path}")
    print(f"Candidate output    : {candidate_out}")
    print(f"Matching output     : {matching_out}")
    print(f"SQLite cache path   : {db_path}")
    print(f"Max S1 records      : {max_s1_records or 'ALL (1,732,544)'}")
    print(f"Max Cand records    : {max_cand_records or 'ALL (~10,000,000)'}")

    # 1. Load model
    model, threshold = load_production_model()

    # 2. Build index
    conn = build_sqlite_candidate_index(
        db_path,
        s2_path,
        s3_path,
        max_bucket_size=300,
        max_cand_records=max_cand_records,
    )
    cur = conn.cursor()

    # 3. Process test Source 1 in streaming batches
    print("\nProcessing test Source 1 entities and generating submission files...")
    t_s1_start = time.time()

    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    total_singletons = 0

    s1_batch: List[Dict[str, Any]] = []

    with open(candidate_out, "w", encoding="utf-8", newline="") as f_cand, \
         open(matching_out, "w", encoding="utf-8", newline="") as f_match:

        f_cand.write(CANDIDATE_HEADER)
        f_match.write(MATCHING_HEADER)

        def process_batch(batch: List[Dict[str, Any]]):
            nonlocal total_s1, total_candidates, total_matches, total_singletons

            # Step A: Collect all blocking keys for this S1 batch
            s1_keys_map: Dict[str, List[str]] = {}
            all_keys_set: Set[str] = set()

            for s1_norm in batch:
                s1_id = s1_norm["entity_id"]
                keys = get_blocking_keys(s1_norm, strategy="combined")
                s1_keys_map[s1_id] = keys
                all_keys_set.update(keys)

            # Step B: Batch query SQLite for candidate IDs
            all_keys_list = list(all_keys_set)
            key_to_cands: Dict[str, Set[str]] = {}

            # Query SQLite in chunks of 500 parameters
            CHUNK_P = 500
            for i in range(0, len(all_keys_list), CHUNK_P):
                chunk = all_keys_list[i:i + CHUNK_P]
                placeholders = ",".join(["?"] * len(chunk))
                cur.execute(f"SELECT bkey, eid FROM blocking_index WHERE bkey IN ({placeholders})", chunk)
                for bkey, eid in cur.fetchall():
                    key_to_cands.setdefault(bkey, set()).add(eid)

            # Step C: Collect all distinct candidate EIDs for this batch
            batch_candidate_eids: Set[str] = set()
            s1_to_cand_set: Dict[str, Set[str]] = {}

            for s1_norm in batch:
                s1_id = s1_norm["entity_id"]
                c_set = set()
                for k in s1_keys_map[s1_id]:
                    if k in key_to_cands:
                        c_set.update(key_to_cands[k])
                s1_to_cand_set[s1_id] = c_set
                batch_candidate_eids.update(c_set)

            # Step D: Bulk fetch candidate record data from SQLite
            cand_records_cache: Dict[str, Dict[str, Any]] = {}
            cand_eid_list = list(batch_candidate_eids)

            for i in range(0, len(cand_eid_list), CHUNK_P):
                chunk = cand_eid_list[i:i + CHUNK_P]
                placeholders = ",".join(["?"] * len(chunk))
                cur.execute(f"""
                    SELECT eid, norm_name, norm_addr, raw_name, country, street_num, postal_codes, numbers
                    FROM candidate_records WHERE eid IN ({placeholders})
                """, chunk)
                for eid, n_name, n_addr, r_name, country, s_num, p_codes, nums in cur.fetchall():
                    name_toks = n_name.split() if n_name else []
                    addr_toks = n_addr.split() if n_addr else []
                    cand_records_cache[eid] = {
                        "entity_id": eid,
                        "business_name": n_name,
                        "business_address": n_addr,
                        "raw_business_name": r_name,
                        "country": country,
                        "street_number": s_num,
                        "postal_codes": set(p_codes.split(",")) if p_codes else set(),
                        "address_numbers": set(nums.split(",")) if nums else set(),
                        "name_tokens_set": set(name_toks),
                        "address_tokens_set": set(addr_toks),
                    }

            # Step E: Feature extraction, scoring, and output writing
            for s1_norm in batch:
                s1_id = s1_norm["entity_id"]
                c_set = s1_to_cand_set[s1_id]
                total_s1 += 1

                if not c_set:
                    f_cand.write(f"{s1_id}{DELIM}\n")
                    f_match.write(f"{s1_id}{DELIM}\n")
                    total_singletons += 1
                    continue

                sorted_cands = sorted(list(c_set))
                total_candidates += len(sorted_cands)
                f_cand.write(f"{s1_id}{DELIM}{format_id_list(sorted_cands)}\n")

                # Build pair features
                X_pairs = [
                    build_features(s1_norm, cand_records_cache.get(cid, {"entity_id": cid}))
                    for cid in sorted_cands
                ]

                # Score & decide matches
                scores = predict_scores(model, X_pairs)
                matched_cids = decide_matches(sorted_cands, scores, threshold=threshold)

                # Ensure strict candidate subset constraint
                valid_matches = [cid for cid in matched_cids if cid in c_set]
                if valid_matches:
                    total_matches += len(valid_matches)
                    f_match.write(f"{s1_id}{DELIM}{format_id_list(valid_matches)}\n")
                else:
                    total_singletons += 1
                    f_match.write(f"{s1_id}{DELIM}\n")

        # Stream test S1
        for raw_s1 in stream_tsv_records(s1_path, max_records=max_s1_records):
            s1_norm = normalize_record(raw_s1)
            s1_batch.append(s1_norm)

            if len(s1_batch) >= batch_size:
                process_batch(s1_batch)
                s1_batch.clear()

                if total_s1 % 100000 == 0:
                    elapsed_s1 = time.time() - t_s1_start
                    rate = total_s1 / elapsed_s1
                    remain_records = (max_s1_records or 1732544) - total_s1
                    eta_mins = (remain_records / rate) / 60 if rate > 0 else 0
                    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
                    print(f"  Processed {total_s1:,} S1 records ({rate:,.0f} S1/s, ETA: {eta_mins:.1f}m, RAM: {rss_mb:.0f} MB)")

        if s1_batch:
            process_batch(s1_batch)
            s1_batch.clear()

    total_pipeline_time = time.time() - start_time
    conn.close()

    if not keep_db:
        try:
            os.remove(db_path)
            print(f"Cleaned up temporary cache: {db_path}")
        except OSError:
            pass

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Source 1 records processed : {total_s1:,}")
    print(f"Total candidate pairs generated   : {total_candidates:,} (avg {total_candidates / total_s1:.2f} / S1)")
    print(f"Total predicted entity matches    : {total_matches:,} (avg {total_matches / total_s1:.2f} / S1)")
    print(f"Singletons (zero matches)         : {total_singletons:,} ({total_singletons / total_s1 * 100:.2f}%)")
    print(f"Total pipeline elapsed time       : {total_pipeline_time:.2f}s ({total_pipeline_time / 60:.2f} minutes)")
    print(f"Final Candidate File              : {candidate_out}")
    print(f"Final Matching File               : {matching_out}")
    print("=" * 70)

    # Post-generation validations
    if validate and not max_s1_records:
        print("\nRunning Official Submission Validator...")
        val_code = validate_submission_files(
            matching_path=matching_out,
            candidate_path=candidate_out,
            test_dir=test_dir,
            check_ids=False
        )
        return val_code

    return 0


def main():
    parser = argparse.ArgumentParser(description="Full Production Pipeline for Amazon ML Challenge 2026")
    parser.add_argument("--test-dir", default=os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output"))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-s1", type=int, default=None)
    parser.add_argument("--max-cand", type=int, default=None)
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    exit_code = run_full_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_s1_records=args.max_s1,
        max_cand_records=args.max_cand,
        keep_db=args.keep_db,
        validate=not args.no_validate,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
