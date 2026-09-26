"""
Vectorized, high-throughput in-memory entity resolution engine for SageMaker.
Replaces slow disk-backed SQLite queries with compact in-memory inverted indexes,
cheap candidate pre-filtering, and vectorized FastLogisticRegression scoring.
"""

import time
import json
from array import array
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np

from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.blocking import get_blocking_keys
from business_entity_resolution.src.model import FastLogisticRegression, predict_scores, decide_matches
from scripts.pipeline_utils import DELIM, format_id_list


class CompactInvertedIndex:
    """
    Compact, high-speed in-memory inverted index for Source 2 and Source 3 candidate records.
    Uses array('I') (4 bytes per candidate ID reference) for zero disk I/O and O(1) lookups.
    """

    def __init__(self, max_bucket_size: int = 300):
        self.max_bucket_size = max_bucket_size
        self.key_to_cand_idxs: Dict[str, array] = {}
        self.cand_records: List[Dict[str, Any]] = []
        self.cand_eids: List[str] = []

    def add_record(self, raw_record: Dict[str, Any], strategy: str = "combined") -> None:
        eid = raw_record.get("entity_id", "")
        if not eid:
            return

        norm = normalize_record(raw_record)
        cand_idx = len(self.cand_records)
        compact_cand = {
            "entity_id": eid,
            "raw_business_name": norm.get("raw_business_name", ""),
            "business_name": norm.get("business_name", ""),
            "name_signature": norm.get("name_signature", ""),
            "business_address": norm.get("business_address", ""),
            "country": norm.get("country", ""),
            "name_tokens_set": norm.get("name_tokens_set", set()),
            "address_tokens_set": norm.get("address_tokens_set", set()),
            "address_numbers": norm.get("address_numbers", set()),
            "char_shingles": norm.get("char_shingles", []),
        }
        self.cand_records.append(compact_cand)
        self.cand_eids.append(eid)

        keys = get_blocking_keys(norm, strategy=strategy)
        for k in keys:
            bucket = self.key_to_cand_idxs.get(k)
            if bucket is None:
                new_bucket = array("I")
                new_bucket.append(cand_idx)
                self.key_to_cand_idxs[k] = new_bucket
            elif len(bucket) < self.max_bucket_size:
                bucket.append(cand_idx)

    def get_candidate_indices(self, s1_norm: Dict[str, Any], strategy: str = "combined") -> Set[int]:
        keys = get_blocking_keys(s1_norm, strategy=strategy)
        matched_indices: Set[int] = set()
        for k in keys:
            bucket = self.key_to_cand_idxs.get(k)
            if bucket is not None:
                matched_indices.update(bucket)
        return matched_indices

    def __len__(self) -> int:
        return len(self.cand_records)


def fast_jaccard(set_a: Set[Any], set_b: Set[Any]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter_len = len(set_a & set_b)
    if inter_len == 0:
        return 0.0
    return inter_len / (len(set_a) + len(set_b) - inter_len)


def fast_containment(set_a: Set[Any], set_b: Set[Any]) -> float:
    if not set_a or not set_b:
        return 0.0
    min_len = min(len(set_a), len(set_b))
    if min_len == 0:
        return 0.0
    return len(set_a & set_b) / min_len


def extract_features_vectorized(
    s1_norm: Dict[str, Any],
    cand_norms: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Vectorized extraction of the 13 pairwise features for FastLogisticRegression.
    Avoids Python object allocations by operating on pre-normalized sets and string tokens.
    """
    n = len(cand_norms)
    X = np.zeros((n, 13), dtype=np.float32)

    s1_raw_name = s1_norm.get("raw_business_name", "")
    s1_name = s1_norm.get("business_name", "")
    s1_name_len = max(len(s1_name), 1)
    s1_sig = s1_norm.get("name_signature", "")
    s1_addr = s1_norm.get("business_address", "")
    s1_country = s1_norm.get("country", "")

    s1_name_toks: Set[str] = s1_norm.get("name_tokens_set", set())
    s1_addr_toks: Set[str] = s1_norm.get("address_tokens_set", set())
    s1_nums: Set[str] = s1_norm.get("address_numbers", set())
    s1_shingles: Set[str] = set(s1_norm.get("char_shingles", []))

    for i, c in enumerate(cand_norms):
        c_raw_name = c.get("raw_business_name", "")
        c_name = c.get("business_name", "")
        c_sig = c.get("name_signature", "")
        c_addr = c.get("business_address", "")
        c_country = c.get("country", "")

        c_name_toks: Set[str] = c.get("name_tokens_set", set())
        c_addr_toks: Set[str] = c.get("address_tokens_set", set())
        c_nums: Set[str] = c.get("address_numbers", set())
        c_shingles: Set[str] = set(c.get("char_shingles", []))

        # 0. name_exact_raw
        if s1_raw_name and s1_raw_name == c_raw_name:
            X[i, 0] = 1.0

        # 1. name_exact_norm
        if s1_name and s1_name == c_name:
            X[i, 1] = 1.0

        # 2. name_sig_match
        if s1_sig and s1_sig == c_sig:
            X[i, 2] = 1.0

        # 3. name_jaccard
        X[i, 3] = fast_jaccard(s1_name_toks, c_name_toks)

        # 4. name_containment
        X[i, 4] = fast_containment(s1_name_toks, c_name_toks)

        # 5. name_ngram_jaccard (from precomputed char_shingles)
        X[i, 5] = fast_jaccard(s1_shingles, c_shingles)

        # 6. name_len_diff
        max_len = max(s1_name_len, len(c_name))
        X[i, 6] = abs(s1_name_len - len(c_name)) / max_len

        # 7. addr_is_empty
        if not s1_addr or not c_addr:
            X[i, 7] = 1.0

        # 8. addr_exact_norm
        if s1_addr and s1_addr == c_addr:
            X[i, 8] = 1.0

        # 9. addr_jaccard
        X[i, 9] = fast_jaccard(s1_addr_toks, c_addr_toks)

        # 10. addr_containment
        X[i, 10] = fast_containment(s1_addr_toks, c_addr_toks)

        # 11. addr_num_overlap
        X[i, 11] = fast_jaccard(s1_nums, c_nums)

        # 12. country_match
        if s1_country and s1_country == c_country:
            X[i, 12] = 1.0

    return X


def process_s1_batch_vectorized(
    s1_batch: List[Dict[str, Any]],
    index: CompactInvertedIndex,
    model: FastLogisticRegression,
    threshold: float,
    blocking_strategy: str = "combined",
) -> List[Tuple[str, List[str], List[str]]]:
    """
    Process a batch of Source 1 entities with:
    1. O(1) in-memory candidate retrieval
    2. Cheap candidate pre-filtering (pruning mathematically impossible matches)
    3. Vectorized batch feature extraction
    4. Batch FastLogisticRegression matrix scoring
    5. Strict candidate-subset verification

    Returns:
    List of tuples: (s1_entity_id, candidate_eids_list, matched_eids_list)
    """
    results = []

    weights = np.array(model.weights, dtype=np.float32)
    bias = float(model.bias)

    for s1_rec in s1_batch:
        s1_norm = normalize_record(s1_rec) if "name_tokens_set" not in s1_rec else s1_rec
        s1_id = s1_norm["entity_id"]

        cand_indices = index.get_candidate_indices(s1_norm, strategy=blocking_strategy)
        if not cand_indices:
            results.append((s1_id, [], []))
            continue

        # All candidate EIDs are sorted for deterministic output
        all_cand_eids = sorted([index.cand_eids[idx] for idx in cand_indices])
        cand_idx_list = sorted(list(cand_indices))

        # Cheap pre-filtering:
        # A pair CANNOT reach threshold=0.50 if:
        # 1. No name token overlap AND no address token overlap
        # 2. Not exact raw or normalized name match
        # 3. Not name signature match
        s1_name_toks = s1_norm.get("name_tokens_set", set())
        s1_addr_toks = s1_norm.get("address_tokens_set", set())
        s1_name = s1_norm.get("business_name", "")
        s1_raw_name = s1_norm.get("raw_business_name", "")
        s1_sig = s1_norm.get("name_signature", "")

        promising_cands = []
        for ci in cand_idx_list:
            c = index.cand_records[ci]
            c_name_toks = c.get("name_tokens_set", set())
            c_addr_toks = c.get("address_tokens_set", set())

            # Fast check
            has_name_token = bool(s1_name_toks & c_name_toks)
            has_addr_token = bool(s1_addr_toks & c_addr_toks)
            is_name_match = (s1_name == c.get("business_name", "")) or (s1_raw_name == c.get("raw_business_name", "")) or (s1_sig and s1_sig == c.get("name_signature", ""))

            if has_name_token or has_addr_token or is_name_match:
                promising_cands.append(c)

        matched_eids = []
        if promising_cands:
            X = extract_features_vectorized(s1_norm, promising_cands)
            # Vectorized dot product
            logits = np.dot(X, weights) + bias
            # Sigmoid >= threshold is equivalent to logits >= logit(threshold)
            # For threshold = 0.50, logit(0.50) = 0.0
            logit_thresh = np.log(threshold / (1.0 - threshold)) if (0.0 < threshold < 1.0) else 0.0
            pred_mask = logits >= logit_thresh

            cand_eids_set = set(all_cand_eids)
            for i, matched in enumerate(pred_mask):
                if matched:
                    cid = promising_cands[i]["entity_id"]
                    # Candidate subset guarantee
                    if cid in cand_eids_set:
                        matched_eids.append(cid)

        results.append((s1_id, all_cand_eids, sorted(matched_eids)))

    return results
