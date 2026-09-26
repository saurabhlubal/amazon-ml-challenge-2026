"""
Candidate Generation and Multi-Index Blocking for Business Entity Resolution.

Builds multiple high-recall, low-overhead blocking indexes partitioned strictly by country:
1. Exact Core Name Index: Canonical business name without legal suffixes.
2. Sorted Core Tokens Index: Word-order invariant representation.
3. Compact / Domain Root Index: Compacted alphanumeric strings and URL domains.
4. Informative Name Token Inverted Index: Low-frequency distinctive name tokens.
5. Address Number + Locality Token Index: Street number combined with distinctive city/locality tokens.
6. Address Two-Number Index: Multi-number address signatures (e.g. 141 and 71).

Combines candidate generators with priority scoring and top-K bounded capping to ensure
maximum true-match recall while bounding candidate sets for downstream ML matching.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from business_entity_resolution.src.normalization import (
    normalize_record,
    clean_text,
    clean_digit,
)

# Frequency thresholds to prevent candidate explosion from common words
MAX_TOKEN_DOCS = 120
DEFAULT_MAX_CANDIDATES = 60


def build_blocking_indexes(
    candidate_records: Iterable[Dict[str, Any]],
    max_token_docs: int = MAX_TOKEN_DOCS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Build multi-layer blocking indexes from candidate entities (Source 2 and Source 3).

    Parameters
    ----------
    candidate_records : iterable of dict
        Candidate records with entity_id, business_name, business_address, country.
    max_token_docs : int
        Maximum document frequency for inverted index name tokens.
    verbose : bool
        Whether to log progress.

    Returns
    -------
    dict
        Dictionary containing pre-built inverted indexes and configurations.
    """
    t0 = time.time()

    index_exact_core: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_sorted_core: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_compact: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_token: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_addr_num_word: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    index_addr_two_num: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    token_freq: Dict[Tuple[str, str], int] = defaultdict(int)

    count = 0
    for rec in candidate_records:
        norm = normalize_record(rec)
        cid = norm["entity_id"]
        c = norm["country"]

        # 1. Exact core name
        if norm["core_name"]:
            index_exact_core[(c, norm["core_name"])].append(cid)

        # 2. Sorted core tokens
        if norm["sorted_core"]:
            index_sorted_core[(c, norm["sorted_core"])].append(cid)

        # 3. Compact forms and domain root
        for comp in norm["compact_forms"]:
            index_compact[(c, comp)].append(cid)

        # 4. Informative name tokens
        for t in norm["core_tokens"]:
            if len(t) >= 3:
                token_freq[(c, t)] += 1
                index_token[(c, t)].append(cid)

        # 5. Address: house/building numbers combined with locality tokens
        for d in norm["digits"]:
            for at in norm["significant_address_tokens"]:
                index_addr_num_word[(c, d, at)].append(cid)

        # 6. Address: two digits
        if norm["first_digit"] and norm["second_digit"]:
            index_addr_two_num[(c, norm["first_digit"], norm["second_digit"])].append(cid)

        count += 1
        if verbose and count % 500000 == 0:
            print(f"Indexed {count:,} candidate records in {time.time() - t0:.1f}s...")

    # Prune inverted name token index to remove high-frequency terms
    pruned_token_index: Dict[Tuple[str, str], List[str]] = {}
    for key, cands in index_token.items():
        if len(cands) <= max_token_docs:
            pruned_token_index[key] = cands

    build_time = time.time() - t0
    if verbose:
        print(f"Blocking indexes built for {count:,} records in {build_time:.2f}s.")

    return {
        "exact_core": dict(index_exact_core),
        "sorted_core": dict(index_sorted_core),
        "compact": dict(index_compact),
        "token": pruned_token_index,
        "addr_num_word": dict(index_addr_num_word),
        "addr_two_num": dict(index_addr_two_num),
        "token_freq": dict(token_freq),
        "total_records": count,
        "build_time_sec": build_time,
    }


def generate_candidates(
    source1_record: Dict[str, Any],
    indexes: Dict[str, Any],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Set[str]:
    """
    Generate candidate Source2/Source3 entity IDs for one Source1 record.

    Parameters
    ----------
    source1_record : dict
        Source1 entity containing entity_id, business_name, business_address, country.
    indexes : dict
        Pre-built blocking indexes from `build_blocking_indexes`.
    max_candidates : int
        Maximum number of candidates to return (prioritized by signal strength).

    Returns
    -------
    set[str]
        Set of candidate entity IDs.
    """
    p1 = normalize_record(source1_record)
    c = p1["country"]

    index_exact_core = indexes["exact_core"]
    index_sorted_core = indexes["sorted_core"]
    index_compact = indexes["compact"]
    index_token = indexes["token"]
    index_addr_num_word = indexes["addr_num_word"]
    index_addr_two_num = indexes["addr_two_num"]

    cand_scores: Dict[str, float] = defaultdict(float)

    # 1. Exact core name match (Weight: 5.0)
    if p1["core_name"]:
        for cid in index_exact_core.get((c, p1["core_name"]), ()):
            cand_scores[cid] += 5.0

    # 2. Sorted core tokens match (Weight: 4.0)
    if p1["sorted_core"]:
        for cid in index_sorted_core.get((c, p1["sorted_core"]), ()):
            cand_scores[cid] += 4.0

    # 3. Compact / domain name match (Weight: 3.5)
    for comp in p1["compact_forms"]:
        for cid in index_compact.get((c, comp), ()):
            cand_scores[cid] += 3.5

    # 4. Address: house number + locality token match (Weight: 3.0)
    for d in p1["digits"]:
        for at in p1["significant_address_tokens"]:
            for cid in index_addr_num_word.get((c, d, at), ()):
                cand_scores[cid] += 3.0

    # 5. Address: two digits match (Weight: 2.0)
    if p1["first_digit"] and p1["second_digit"]:
        for cid in index_addr_two_num.get((c, p1["first_digit"], p1["second_digit"]), ()):
            cand_scores[cid] += 2.0

    # 6. Informative name tokens match (Weight: 1.0)
    for t in p1["core_tokens"]:
        if len(t) >= 4:
            for cid in index_token.get((c, t), ()):
                cand_scores[cid] += 1.0

    # If within budget, return all candidates
    if len(cand_scores) <= max_candidates:
        return set(cand_scores.keys())

    # Otherwise, sort by prioritized score and retain top-K
    sorted_candidates = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
    return {x[0] for x in sorted_candidates[:max_candidates]}


def batch_generate_candidates(
    s1_records: Iterable[Dict[str, Any]],
    indexes: Dict[str, Any],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Dict[str, Set[str]]:
    """
    Generate candidates for a collection of Source1 records.

    Returns
    -------
    dict
        s1_entity_id -> set of candidate entity IDs.
    """
    results: Dict[str, Set[str]] = {}
    for s1_rec in s1_records:
        sid = s1_rec["entity_id"]
        results[sid] = generate_candidates(s1_rec, indexes, max_candidates=max_candidates)
    return results


def evaluate_blocking(
    s1_records: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    indexes: Dict[str, Any],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Dict[str, Any]:
    """
    Benchmark blocking performance against true ground truth matches.

    Parameters
    ----------
    s1_records : dict
        s1_entity_id -> S1 raw record dict.
    ground_truth : dict
        s1_entity_id -> set of true matching S2/S3 entity IDs.
    indexes : dict
        Pre-built blocking indexes.
    max_candidates : int
        Candidate capping threshold.

    Returns
    -------
    dict
        Comprehensive metrics dictionary:
        - blocking_recall: float
        - total_true_matches: int
        - found_true_matches: int
        - average_candidates: float
        - median_candidates: float
        - p90_candidates: float
        - p95_candidates: float
        - p99_candidates: float
        - max_candidates: int
        - runtime_sec: float
        - s1_per_sec: float
    """
    t0 = time.time()
    total_true = 0
    found_true = 0
    candidate_counts: List[int] = []

    for sid, s1_rec in s1_records.items():
        true_set = ground_truth.get(sid, set())
        total_true += len(true_set)

        cands = generate_candidates(s1_rec, indexes, max_candidates=max_candidates)
        candidate_counts.append(len(cands))

        if true_set:
            found_true += len(true_set & cands)

    elapsed = time.time() - t0
    arr = np.array(candidate_counts) if candidate_counts else np.zeros(1)
    recall = (found_true / total_true) if total_true > 0 else 0.0

    return {
        "blocking_recall": float(recall),
        "total_true_matches": total_true,
        "found_true_matches": found_true,
        "average_candidates": float(arr.mean()),
        "median_candidates": float(np.median(arr)),
        "p90_candidates": float(np.percentile(arr, 90)),
        "p95_candidates": float(np.percentile(arr, 95)),
        "p99_candidates": float(np.percentile(arr, 99)),
        "max_candidates": int(arr.max()),
        "runtime_sec": float(elapsed),
        "s1_per_sec": float(len(s1_records) / elapsed) if elapsed > 0 else 0.0,
    }


def stream_source_file(filepath: str) -> Iterable[Dict[str, str]]:
    """Yield records line-by-line from a source TSV without loading entire file into memory."""
    with open(filepath, "r", encoding="utf-8") as f:
        header = next(f, None)
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) >= 4:
                yield {
                    "entity_id": parts[0],
                    "business_name": parts[1],
                    "business_address": parts[2],
                    "country": parts[3],
                }
            elif len(parts) == 3:
                yield {
                    "entity_id": parts[0],
                    "business_name": parts[1],
                    "business_address": parts[2],
                    "country": "UNKNOWN",
                }
            elif len(parts) == 2:
                yield {
                    "entity_id": parts[0],
                    "business_name": parts[1],
                    "business_address": "",
                    "country": "UNKNOWN",
                }
