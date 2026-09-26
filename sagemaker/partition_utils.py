"""
Data partitioning and sharding utilities for distributed SageMaker processing.
Enables independent parallel execution across multiple SageMaker instances and CPU worker pools.
"""

import os
import csv
from typing import List, Generator, Dict, Any, Optional


def partition_tsv_file(
    input_file: str,
    output_dir: str,
    num_shards: int = 16,
    prefix: str = "shard",
) -> List[str]:
    """
    Partition a large TSV file into N independent shards with the original header preserved.
    Uses constant memory streaming.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_paths = [os.path.join(output_dir, f"{prefix}_{i:03d}.tsv") for i in range(num_shards)]
    shard_files = [open(p, "w", encoding="utf-8", newline="") for p in shard_paths]

    try:
        with open(input_file, "r", encoding="utf-8", newline="") as f_in:
            reader = csv.reader(f_in, delimiter="\t")
            header = next(reader)
            header_line = "\t".join(header) + "\n"

            # Write header to each shard
            for f_out in shard_files:
                f_out.write(header_line)

            # Round-robin or chunk distribution
            row_idx = 0
            for row in reader:
                shard_idx = row_idx % num_shards
                shard_files[shard_idx].write("\t".join(row) + "\n")
                row_idx += 1

    finally:
        for f_out in shard_files:
            f_out.close()

    print(f"Partitioned {row_idx:,} records from {input_file} into {num_shards} shards in {output_dir}")
    return shard_paths


def get_shard_record_count(file_path: str) -> int:
    """Safely count records in a TSV excluding header."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header
        for _ in f:
            count += 1
    return count
