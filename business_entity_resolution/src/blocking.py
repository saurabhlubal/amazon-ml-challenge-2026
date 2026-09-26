"""
Candidate generation (blocking) module for Business Entity Resolution.
Builds multi-key inverted indexes and generates high-recall candidate sets
while pruning excessively large buckets.
"""

from typing import Dict, Any, Set, List, Iterable, Optional
from business_entity_resolution.src.normalization import normalize_record


# Common high-frequency words to exclude from single-token blocking keys
COMMON_STOP_WORDS = {
    "corp", "inc", "co", "ltd", "pvt", "llc", "and", "the", "for", "with",
    "services", "service", "solutions", "solution", "group", "holdings",
    "enterprises", "enterprise", "industries", "industry", "company",
    "store", "shop", "center", "centre", "market", "mart", "supermarket",
    "st", "rd", "ave", "dr", "ln", "ct", "hwy", "blvd", "apt", "ste",
    "north", "south", "east", "west", "main", "new", "city",
}

DEFAULT_MAX_BUCKET_SIZE = 500


def get_blocking_keys(record: Dict[str, Any]) -> List[str]:
    """Generate multiple high-recall blocking keys for a record."""
    norm_rec = record if "name_tokens" in record else normalize_record(record)

    keys = []
    norm_name = norm_rec.get("business_name", "")
    norm_addr = norm_rec.get("business_address", "")
    country = norm_rec.get("country", "")
    name_sig = norm_rec.get("name_signature", "")
    name_tokens = norm_rec.get("name_tokens", [])

    # 1. Exact normalized name
    if norm_name:
        keys.append(f"name:{norm_name}")
        # 2. Name + Country
        if country:
            keys.append(f"nc:{norm_name}|{country}")

    # 3. Order-invariant name signature
    if name_sig and name_sig != norm_name:
        keys.append(f"sig:{name_sig}")

    # 4. Exact normalized address (if substantial)
    if norm_addr and len(norm_addr) >= 5:
        keys.append(f"addr:{norm_addr}")
        if country:
            keys.append(f"ac:{norm_addr}|{country}")

    # 5. Distinctive name tokens (length >= 4 and not in stop words)
    for tok in set(name_tokens):
        if len(tok) >= 4 and tok not in COMMON_STOP_WORDS:
            # Pair token with country or length to keep specificity high
            if country:
                keys.append(f"tok_c:{tok}|{country}")
            else:
                keys.append(f"tok:{tok}")

    return keys


def build_blocking_indexes(
    records_iter: Iterable[Dict[str, Any]],
    max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE,
    store_records: bool = True
) -> Dict[str, Any]:
    """
    Build inverted indexes over candidate records (Source 2 and/or Source 3).

    Parameters
    ----------
    records_iter : iterable of dict
        Stream or collection of candidate records.
    max_bucket_size : int
        Maximum number of candidates allowed per key bucket.
    store_records : bool
        Whether to store normalized candidate records for feature building.

    Returns
    -------
    dict
        Inverted index structure containing key -> list of entity_ids.
    """
    index: Dict[str, List[str]] = {}
    record_store: Dict[str, Dict[str, Any]] = {}

    for raw_rec in records_iter:
        eid = raw_rec.get("entity_id", "")
        if not eid:
            continue

        norm_rec = normalize_record(raw_rec) if "name_tokens" not in raw_rec else raw_rec
        if store_records:
            record_store[eid] = norm_rec

        keys = get_blocking_keys(norm_rec)
        for k in keys:
            bucket = index.setdefault(k, [])
            if len(bucket) < max_bucket_size:
                bucket.append(eid)

    # Prune overflowing buckets to prevent high false-positive explosion
    clean_index = {k: v for k, v in index.items() if len(v) <= max_bucket_size}

    return {
        "index": clean_index,
        "records": record_store,
        "max_bucket_size": max_bucket_size,
    }


def generate_candidates(source1_record: Dict[str, Any], indexes: Dict[str, Any]) -> Set[str]:
    """
    Generate candidate Source2/Source3 entity IDs for one Source1 record.

    Parameters
    ----------
    source1_record : dict
        Source1 entity (raw or normalized).
    indexes : dict
        Pre-built blocking indexes.

    Returns
    -------
    set[str]
        Set of candidate entity IDs.
    """
    inv_index = indexes.get("index", {})
    if not inv_index and "name_index" in indexes:
        # Fallback compatibility with simple index structures
        inv_index = indexes.get("name_index", {})

    keys = get_blocking_keys(source1_record)
    candidates: Set[str] = set()

    for k in keys:
        matches = inv_index.get(k)
        if matches:
            candidates.update(matches)

    # Filter out self matches or non-S2/S3 IDs if any
    clean_candidates = {cid for cid in candidates if cid.startswith(("S2-", "S3-"))}
    return clean_candidates