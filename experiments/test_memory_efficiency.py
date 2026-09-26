"""
Memory benchmark to verify memory footprint of compact candidate tuples and indexing.
"""

import os
import sys
import time
import psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import stream_tsv_records
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.blocking import get_blocking_keys


def benchmark_compact_indexing(max_test_records: int = 500000):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)

    s2_file = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train", "train_source2.tsv")
    print(f"Reading and indexing {max_test_records:,} records...")

    t0 = time.time()
    # Compact store: eid -> (norm_name, norm_addr, name_sig, street_num, postal_codes_tuple, numbers_tuple, country)
    store = {}
    index = {}

    count = 0
    for rec in stream_tsv_records(s2_file, max_records=max_test_records):
        eid = rec["entity_id"]
        norm = normalize_record(rec)

        # Store compact tuple instead of full dict
        store[eid] = (
            norm["business_name"],
            norm["business_address"],
            norm["name_signature"],
            norm["street_number"],
            tuple(norm["postal_codes"]),
            tuple(norm["address_numbers"]),
            norm["country"],
            norm["raw_business_name"],
        )

        keys = get_blocking_keys(norm, strategy="combined")
        for k in keys:
            bucket = index.setdefault(k, [])
            if len(bucket) < 300:
                bucket.append(eid)

        count += 1
        if count % 100000 == 0:
            print(f"  Processed {count:,} records... ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    mem_after = process.memory_info().rss / (1024 ** 2)
    mem_used = mem_after - mem_before

    print(f"\nResults for {count:,} records:")
    print(f"  Indexing time : {elapsed:.2f} seconds ({count / elapsed:,.0f} records/sec)")
    print(f"  Memory delta  : {mem_used:.1f} MB ({mem_used / (count / 100000):.1f} MB per 100k records)")
    print(f"  Unique keys   : {len(index):,}")
    print(f"  Extrapolated 5M records memory: ~{mem_used * 10 / 1024:.2f} GB")


if __name__ == "__main__":
    benchmark_compact_indexing(max_test_records=200000)
