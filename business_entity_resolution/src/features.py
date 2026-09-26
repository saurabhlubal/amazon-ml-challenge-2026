"""
Feature engineering module for Business Entity Resolution.
Extracts fast, lightweight numeric similarity features between Source1 and candidate pairs.
"""

from typing import Dict, Any, Set
from business_entity_resolution.src.normalization import normalize_record


def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 0.0
    return len(set_a & set_b) / union_len


def containment_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute containment (overlap divided by minimum set size)."""
    if not set_a or not set_b:
        return 0.0
    min_len = min(len(set_a), len(set_b))
    if min_len == 0:
        return 0.0
    return len(set_a & set_b) / min_len


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if not text:
        return set()
    cleaned = f"#{text}#"
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i:i + n] for i in range(len(cleaned) - n + 1)}


def build_features(source1_record: Dict[str, Any], candidate_record: Dict[str, Any]) -> Dict[str, float]:
    """
    Build comparison features for an S1 / candidate pair.

    Parameters
    ----------
    source1_record : dict
        Source1 entity (raw or normalized).
    candidate_record : dict
        Source2 or Source3 candidate entity (raw or normalized).

    Returns
    -------
    dict
        Feature name -> numeric float value.
    """
    s1 = source1_record if "name_tokens_set" in source1_record else normalize_record(source1_record)
    c2 = candidate_record if "name_tokens_set" in candidate_record else normalize_record(candidate_record)

    s1_name = s1.get("business_name", "")
    c2_name = c2.get("business_name", "")
    s1_addr = s1.get("business_address", "")
    c2_addr = c2.get("business_address", "")
    s1_country = s1.get("country", "")
    c2_country = c2.get("country", "")

    s1_name_tokens: Set[str] = s1.get("name_tokens_set", set())
    c2_name_tokens: Set[str] = c2.get("name_tokens_set", set())
    s1_addr_tokens: Set[str] = s1.get("address_tokens_set", set())
    c2_addr_tokens: Set[str] = c2.get("address_tokens_set", set())

    # 1. Name features
    name_exact_raw = 1.0 if s1.get("raw_business_name") == c2.get("raw_business_name") and s1_name else 0.0
    name_exact_norm = 1.0 if s1_name == c2_name and s1_name else 0.0
    name_sig_match = 1.0 if s1.get("name_signature") == c2.get("name_signature") and s1.get("name_signature") else 0.0

    name_jaccard = jaccard_similarity(s1_name_tokens, c2_name_tokens)
    name_containment = containment_similarity(s1_name_tokens, c2_name_tokens)

    # Character n-grams for typo / phonetic similarity
    s1_ngrams = get_char_ngrams(s1_name, 3)
    c2_ngrams = get_char_ngrams(c2_name, 3)
    name_ngram_jaccard = jaccard_similarity(s1_ngrams, c2_ngrams)

    max_name_len = max(len(s1_name), len(c2_name), 1)
    name_len_diff = abs(len(s1_name) - len(c2_name)) / max_name_len

    # 2. Address features
    addr_is_empty = 1.0 if not s1_addr or not c2_addr else 0.0
    addr_exact_norm = 1.0 if s1_addr == c2_addr and s1_addr else 0.0
    addr_jaccard = jaccard_similarity(s1_addr_tokens, c2_addr_tokens)
    addr_containment = containment_similarity(s1_addr_tokens, c2_addr_tokens)

    # Number overlap in addresses (e.g. street numbers, pin codes)
    s1_nums: Set[str] = s1.get("address_numbers", set())
    c2_nums: Set[str] = c2.get("address_numbers", set())
    addr_num_overlap = jaccard_similarity(s1_nums, c2_nums)

    # 3. Country match
    country_match = 1.0 if s1_country == c2_country and s1_country else 0.0

    return {
        "name_exact_raw": name_exact_raw,
        "name_exact_norm": name_exact_norm,
        "name_sig_match": name_sig_match,
        "name_jaccard": name_jaccard,
        "name_containment": name_containment,
        "name_ngram_jaccard": name_ngram_jaccard,
        "name_len_diff": name_len_diff,
        "addr_is_empty": addr_is_empty,
        "addr_exact_norm": addr_exact_norm,
        "addr_jaccard": addr_jaccard,
        "addr_containment": addr_containment,
        "addr_num_overlap": addr_num_overlap,
        "country_match": country_match,
    }