"""
Candidate generation (blocking) module for Business Entity Resolution.
Builds multi-key inverted indexes and generates high-recall candidate sets
while pruning excessively large buckets.
Supports baseline, new (shingle, compact, postal/PIN, street-addr, state), and combined strategies.
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
    "north", "south", "east", "west", "main", "new", "city", "road", "street",
    "fl", "bldg", "ext", "opp", "nr", "cir", "po", "lane", "court",
}

DEFAULT_MAX_BUCKET_SIZE = 500


def get_blocking_keys(record: Dict[str, Any], strategy: str = "combined") -> List[str]:
    """
    Generate multiple high-recall blocking keys for a record.

    Parameters
    ----------
    record : dict
        Raw or normalized entity record.
    strategy : str
        'baseline', 'new', or 'combined'.
    """
    norm_rec = record if "name_tokens" in record else normalize_record(record)

    keys = []
    norm_name = norm_rec.get("business_name", "")
    norm_addr = norm_rec.get("business_address", "")
    country = norm_rec.get("country", "")
    name_sig = norm_rec.get("name_signature", "")
    name_tokens = norm_rec.get("name_tokens", [])
    addr_tokens = norm_rec.get("address_tokens", [])
    compact_name = norm_rec.get("compact_name", "")
    street_num = norm_rec.get("street_number", "")
    postal_codes = norm_rec.get("postal_codes", set())
    state = norm_rec.get("state", "")
    shingles = norm_rec.get("char_shingles", [])

    include_baseline = strategy in ("baseline", "combined")
    include_new = strategy in ("new", "combined")

    # --- BASELINE KEYS ---
    if include_baseline:
        # 1. Exact normalized name
        if norm_name:
            keys.append(f"name:{norm_name}")
            if country:
                keys.append(f"nc:{norm_name}|{country}")

        # 2. Order-invariant name signature
        if name_sig and name_sig != norm_name:
            keys.append(f"sig:{name_sig}")

        # 3. Exact normalized address (if substantial)
        if norm_addr and len(norm_addr) >= 5:
            keys.append(f"addr:{norm_addr}")
            if country:
                keys.append(f"ac:{norm_addr}|{country}")

        # 4. Distinctive name tokens (length >= 4 and not in stop words)
        for tok in set(name_tokens):
            if len(tok) >= 4 and tok not in COMMON_STOP_WORDS:
                if country:
                    keys.append(f"tok_c:{tok}|{country}")
                else:
                    keys.append(f"tok:{tok}")

    # --- NEW HIGH-RECALL KEYS ---
    if include_new:
        # 5. Compact Name (handles domains like .com, handles @, and spaces/legal suffix differences)
        if compact_name and len(compact_name) >= 3 and compact_name != norm_name:
            keys.append(f"cmp:{compact_name}")
            if country:
                keys.append(f"cmp_c:{compact_name}|{country}")

        # 6. Character 5-gram prefix of compact name (captures typos, prefixes, suffixes)
        if compact_name and len(compact_name) >= 6:
            prefix5 = compact_name[:5]
            if country:
                keys.append(f"pref5:{prefix5}|{country}")

        # 7. Street Number + Primary Name Token (allows len >= 2 for short names like b+, 3m, hp)
        if street_num and name_tokens:
            for n_tok in name_tokens[:3]:
                if len(n_tok) >= 2 and n_tok not in COMMON_STOP_WORDS:
                    keys.append(f"num_name:{street_num}|{n_tok[:5]}")
                    if country:
                        keys.append(f"num_name_c:{street_num}|{n_tok[:5]}|{country}")
                    break

        # 8. Street Number + Distinctive Street Word (highly selective address key)
        if street_num and addr_tokens:
            for a_tok in addr_tokens:
                if len(a_tok) >= 4 and a_tok not in COMMON_STOP_WORDS and not a_tok.isdigit():
                    if country:
                        keys.append(f"num_addr:{street_num}|{a_tok}|{country}")
                    else:
                        keys.append(f"num_addr:{street_num}|{a_tok}")
                    break

        # 9. Postal / PIN code + Primary Name Token
        if postal_codes and name_tokens:
            for pin in list(postal_codes)[:2]:
                for n_tok in name_tokens[:2]:
                    if len(n_tok) >= 2 and n_tok not in COMMON_STOP_WORDS:
                        keys.append(f"pin_tok:{pin}|{n_tok[:4]}")
                        break

        # 10. State / Region + Distinctive Name Token
        if state and name_tokens:
            for n_tok in name_tokens:
                if len(n_tok) >= 3 and n_tok not in COMMON_STOP_WORDS:
                    keys.append(f"st_tok:{state}|{n_tok}")
                    break

        # 11. Shingle pairs (first and last 3-grams of compact name)
        if shingles and len(shingles) >= 2 and len(compact_name) >= 6:
            shing_key = f"sh_fl:{shingles[0]}_{shingles[-1]}"
            if country:
                keys.append(f"{shing_key}|{country}")

    return keys


def build_blocking_indexes(
    records_iter: Iterable[Dict[str, Any]],
    max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE,
    strategy: str = "combined",
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
    strategy : str
        Blocking key strategy: 'baseline', 'new', or 'combined'.
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

        keys = get_blocking_keys(norm_rec, strategy=strategy)
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
        "strategy": strategy,
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
        inv_index = indexes.get("name_index", {})

    strategy = indexes.get("strategy", "combined")
    keys = get_blocking_keys(source1_record, strategy=strategy)
    candidates: Set[str] = set()

    for k in keys:
        matches = inv_index.get(k)
        if matches:
            candidates.update(matches)

    # Filter out self matches or non-S2/S3 IDs if any
    clean_candidates = {cid for cid in candidates if cid.startswith(("S2-", "S3-"))}
    return clean_candidates