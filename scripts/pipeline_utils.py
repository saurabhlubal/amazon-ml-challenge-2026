"""
Utility functions for data streaming, formatting, and submission validation.
Designed for high throughput, minimal memory usage, and strict adherence to
the Amazon ML Challenge 2026 submission specifications.
"""

import os
import csv
import sys
import subprocess
from typing import Iterable, Set, Dict, Iterator, Optional, List


DELIM = "\t"
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"


def format_id_list(entity_ids: Iterable[str]) -> str:
    """
    Format a collection of entity IDs into a clean, comma-separated string.
    Removes duplicates, trims whitespace, and sorts deterministically.
    """
    if not entity_ids:
        return ""
    clean_ids = sorted({str(eid).strip() for eid in entity_ids if str(eid).strip()})
    return ",".join(clean_ids)


def parse_id_list(id_str: str) -> List[str]:
    """
    Parse a comma-separated ID list string into a list of cleaned IDs.
    """
    if not id_str or not id_str.strip():
        return []
    return [item.strip() for item in id_str.split(",") if item.strip()]


def stream_tsv_records(
    file_path: str,
    max_records: Optional[int] = None
) -> Iterator[Dict[str, str]]:
    """
    Stream records from a TSV file one row at a time with minimal memory footprint.
    Yields dictionary for each record:
      {"entity_id": ..., "business_name": ..., "business_address": ..., "country": ...}
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=DELIM)
        try:
            header = next(reader)
        except StopIteration:
            return

        header = [col.strip().lower() for col in header]
        count = 0
        for row in reader:
            if not row:
                continue
            record = {}
            for i, col in enumerate(header):
                record[col] = row[i] if i < len(row) else ""
            yield record
            count += 1
            if max_records is not None and count >= max_records:
                break


def validate_submission_files(
    matching_path: str,
    candidate_path: Optional[str],
    test_dir: str,
    check_ids: bool = False
) -> int:
    """
    Execute student_resource/utils/validate_submission.py against the generated outputs.
    Returns the exit code (0 for PASS, 1 for FAIL).
    """
    validator_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "student_resource",
        "utils",
        "validate_submission.py"
    )
    if not os.path.isfile(validator_script):
        print(f"ERROR: Validator script not found at {validator_script}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_path,
        "--test-dir", test_dir
    ]
    if candidate_path:
        cmd.extend(["--candidate", candidate_path])
    if check_ids:
        cmd.append("--check-ids")

    print(f"Running validator: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode
