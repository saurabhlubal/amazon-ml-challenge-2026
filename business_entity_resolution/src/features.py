"""
Pairwise feature extraction for Business Entity Resolution.

Extracts similarity, lexical, numeric, and categorical comparison signals
between a Source 1 query entity and candidate entities (Source 2 / Source 3).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

# Rapidfuzz with fallback to standard library difflib
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

# Compiled regular expressions for speed
RE_ALPHANUM = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\b\d+\b")
RE_TOKEN = re.compile(r"\b\w+\b", re.UNICODE)


def clean_text(text: Optional[str]) -> str:
    """Standardize string: lowercase, remove punctuation, collapse whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Remove punctuation, lowercase, collapse whitespace
    text = RE_ALPHANUM.sub(" ", text.lower())
    return RE_WHITESPACE.sub(" ", text).strip()


def get_tokens(text: str) -> List[str]:
    """Tokenize clean text into list of word tokens."""
    return RE_TOKEN.findall(text)


def get_token_set(text: str) -> Set[str]:
    """Return set of word tokens."""
    return set(get_tokens(text))


def get_digits(text: Optional[str]) -> List[str]:
    """Extract all standalone or embedded digit sequences."""
    if not text:
        return []
    return RE_DIGITS.findall(str(text))


def levenshtein_sim(s1: str, s2: str) -> float:
    """Normalized character similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if HAS_RAPIDFUZZ:
        return float(distance.Levenshtein.normalized_similarity(s1, s2))
    else:
        return float(difflib.SequenceMatcher(None, s1, s2).ratio())


def token_sort_sim(s1: str, s2: str) -> float:
    """Token sort ratio similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if HAS_RAPIDFUZZ:
        return float(fuzz.token_sort_ratio(s1, s2) / 100.0)
    else:
        sorted_s1 = " ".join(sorted(s1.split()))
        sorted_s2 = " ".join(sorted(s2.split()))
        return float(difflib.SequenceMatcher(None, sorted_s1, sorted_s2).ratio())


def token_set_sim(s1: str, s2: str) -> float:
    """Token set ratio similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if HAS_RAPIDFUZZ:
        return float(fuzz.token_set_ratio(s1, s2) / 100.0)
    else:
        set1 = set(s1.split())
        set2 = set(s2.split())
        inter = sorted(list(set1 & set2))
        diff1 = sorted(list(set1 - set2))
        diff2 = sorted(list(set2 - set1))
        t0 = " ".join(inter)
        t1 = " ".join(inter + diff1)
        t2 = " ".join(inter + diff2)
        r1 = difflib.SequenceMatcher(None, t0, t1).ratio() if t0 and t1 else 0.0
        r2 = difflib.SequenceMatcher(None, t0, t2).ratio() if t0 and t2 else 0.0
        r3 = difflib.SequenceMatcher(None, t1, t2).ratio() if t1 and t2 else 0.0
        return float(max(r1, r2, r3, difflib.SequenceMatcher(None, s1, s2).ratio()))


def jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    inter_len = len(tokens1 & tokens2)
    union_len = len(tokens1 | tokens2)
    return inter_len / union_len if union_len > 0 else 0.0


def dice_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Dice similarity (Sørensen-Dice) between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    inter_len = len(tokens1 & tokens2)
    total_len = len(tokens1) + len(tokens2)
    return (2.0 * inter_len) / total_len if total_len > 0 else 0.0


def overlap_coefficient(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute overlap coefficient (containment) between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    inter_len = len(tokens1 & tokens2)
    min_len = min(len(tokens1), len(tokens2))
    return inter_len / min_len if min_len > 0 else 0.0


# Feature column specification
FEATURE_NAMES: List[str] = [
    # Business Name features
    "name_levenshtein_sim",
    "name_token_sort_sim",
    "name_token_set_sim",
    "name_token_jaccard",
    "name_token_dice",
    "name_token_overlap_min",
    "name_exact_match",
    "name_prefix_match_4",
    "name_suffix_match_4",
    "name_length_diff",
    "name_length_ratio",
    "name_token_count_diff",
    "name_token_count_ratio",
    "name_is_single_token",
    
    # Address features
    "address_is_missing_s1",
    "address_is_missing_s2",
    "address_is_missing_either",
    "address_is_missing_both",
    "address_levenshtein_sim",
    "address_token_sort_sim",
    "address_token_jaccard",
    "address_token_dice",
    "address_token_overlap_min",
    "address_exact_match",
    "address_length_diff",
    "address_length_ratio",

    # Country features
    "country_exact_match",
    "country_mismatch",
    "country_is_missing",

    # Numeric & Token Agreement features
    "num_name_jaccard",
    "num_name_conflict",
    "num_address_jaccard",
    "num_address_conflict",
    "num_all_jaccard",
    "num_all_has_common",
    "num_all_conflict",
    "num_primary_match",

    # Cross-field Interactions
    "name_and_country_match",
    "name_and_address_jaccard_prod",
    "overall_composite_sim",
]


def build_features(
    source1_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
) -> Dict[str, float]:
    """
    Build comprehensive comparison features for an S1 / candidate pair.

    Parameters
    ----------
    source1_record : dict
        Source1 entity containing:
        entity_id, business_name, business_address, country
        (plus optional pre-normalized fields)
    candidate_record : dict
        Source2 or Source3 candidate entity containing:
        entity_id, business_name, business_address, country
        (plus optional pre-normalized fields)

    Returns
    -------
    dict
        Feature name -> numeric float value.
    """
    # 1. Extract and clean strings
    raw_name1 = str(source1_record.get("business_name") or "")
    raw_name2 = str(candidate_record.get("business_name") or "")
    name1 = clean_text(raw_name1)
    name2 = clean_text(raw_name2)

    raw_addr1 = str(source1_record.get("business_address") or "")
    raw_addr2 = str(candidate_record.get("business_address") or "")
    addr1 = clean_text(raw_addr1)
    addr2 = clean_text(raw_addr2)

    country1 = str(source1_record.get("country") or "").strip().upper()
    country2 = str(candidate_record.get("country") or "").strip().upper()

    # 2. Tokenize
    tokens_n1 = get_tokens(name1)
    tokens_n2 = get_tokens(name2)
    set_n1 = set(tokens_n1)
    set_n2 = set(tokens_n2)

    tokens_a1 = get_tokens(addr1)
    tokens_a2 = get_tokens(addr2)
    set_a1 = set(tokens_a1)
    set_a2 = set(tokens_a2)

    # 3. Business Name features
    len_n1, len_n2 = len(name1), len(name2)
    cnt_n1, cnt_n2 = len(tokens_n1), len(tokens_n2)

    feat_name_lev = levenshtein_sim(name1, name2)
    feat_name_sort = token_sort_sim(name1, name2)
    feat_name_set = token_set_sim(name1, name2)
    feat_name_jaccard = jaccard_similarity(set_n1, set_n2)
    feat_name_dice = dice_similarity(set_n1, set_n2)
    feat_name_overlap = overlap_coefficient(set_n1, set_n2)
    feat_name_exact = 1.0 if name1 and name1 == name2 else 0.0

    # Prefix/suffix match (min 4 chars)
    prefix_len = 4
    if len_n1 >= prefix_len and len_n2 >= prefix_len:
        feat_name_prefix = 1.0 if name1[:prefix_len] == name2[:prefix_len] else 0.0
        feat_name_suffix = 1.0 if name1[-prefix_len:] == name2[-prefix_len:] else 0.0
    else:
        feat_name_prefix = 1.0 if name1 and name1 == name2 else 0.0
        feat_name_suffix = 1.0 if name1 and name1 == name2 else 0.0

    feat_name_len_diff = float(abs(len_n1 - len_n2))
    feat_name_len_ratio = (min(len_n1, len_n2) / max(len_n1, len_n2)) if max(len_n1, len_n2) > 0 else 1.0
    feat_name_cnt_diff = float(abs(cnt_n1 - cnt_n2))
    feat_name_cnt_ratio = (min(cnt_n1, cnt_n2) / max(cnt_n1, cnt_n2)) if max(cnt_n1, cnt_n2) > 0 else 1.0
    feat_name_single_token = 1.0 if (cnt_n1 == 1 or cnt_n2 == 1) else 0.0

    # 4. Address features
    addr1_missing = 1.0 if not addr1 else 0.0
    addr2_missing = 1.0 if not addr2 else 0.0
    addr_either_missing = 1.0 if (addr1_missing or addr2_missing) else 0.0
    addr_both_missing = 1.0 if (addr1_missing and addr2_missing) else 0.0

    if addr_either_missing:
        feat_addr_lev = 0.0
        feat_addr_sort = 0.0
        feat_addr_jaccard = 0.0
        feat_addr_dice = 0.0
        feat_addr_overlap = 0.0
        feat_addr_exact = 0.0
        feat_addr_len_diff = float(abs(len(addr1) - len(addr2)))
        feat_addr_len_ratio = 0.0
    else:
        feat_addr_lev = levenshtein_sim(addr1, addr2)
        feat_addr_sort = token_sort_sim(addr1, addr2)
        feat_addr_jaccard = jaccard_similarity(set_a1, set_a2)
        feat_addr_dice = dice_similarity(set_a1, set_a2)
        feat_addr_overlap = overlap_coefficient(set_a1, set_a2)
        feat_addr_exact = 1.0 if addr1 == addr2 else 0.0
        feat_addr_len_diff = float(abs(len(addr1) - len(addr2)))
        max_l = max(len(addr1), len(addr2))
        feat_addr_len_ratio = min(len(addr1), len(addr2)) / max_l if max_l > 0 else 1.0

    # 5. Country features
    country_missing = 1.0 if (not country1 or not country2) else 0.0
    if country_missing:
        feat_country_exact = 0.0
        feat_country_mismatch = 0.0
    elif country1 == country2:
        feat_country_exact = 1.0
        feat_country_mismatch = 0.0
    else:
        feat_country_exact = 0.0
        feat_country_mismatch = 1.0

    # 6. Numeric agreement
    # Name digits
    digits_n1 = set(get_digits(raw_name1))
    digits_n2 = set(get_digits(raw_name2))
    num_name_jaccard = jaccard_similarity(digits_n1, digits_n2)
    num_name_conflict = 1.0 if (digits_n1 and digits_n2 and not (digits_n1 & digits_n2)) else 0.0

    # Address digits (house numbers, postal codes, unit numbers)
    digits_a1 = set(get_digits(raw_addr1))
    digits_a2 = set(get_digits(raw_addr2))
    num_addr_jaccard = jaccard_similarity(digits_a1, digits_a2)
    num_addr_conflict = 1.0 if (digits_a1 and digits_a2 and not (digits_a1 & digits_a2)) else 0.0

    # All digits combined
    all_dig1 = digits_n1 | digits_a1
    all_dig2 = digits_n2 | digits_a2
    num_all_jaccard = jaccard_similarity(all_dig1, all_dig2)
    num_all_has_common = 1.0 if (all_dig1 & all_dig2) else 0.0
    num_all_conflict = 1.0 if (all_dig1 and all_dig2 and not (all_dig1 & all_dig2)) else 0.0

    # Primary number match (first numeric token in address or name)
    first_num1 = (get_digits(raw_addr1) or get_digits(raw_name1) or [None])[0]
    first_num2 = (get_digits(raw_addr2) or get_digits(raw_name2) or [None])[0]
    if first_num1 and first_num2:
        num_primary_match = 1.0 if first_num1 == first_num2 else 0.0
    else:
        num_primary_match = 0.0

    # 7. Cross-field interactions
    name_and_country_match = feat_name_exact * feat_country_exact
    name_and_address_prod = feat_name_jaccard * feat_addr_jaccard
    overall_composite = (0.5 * feat_name_sort) + (0.35 * feat_addr_sort) + (0.15 * feat_country_exact)

    return {
        "name_levenshtein_sim": feat_name_lev,
        "name_token_sort_sim": feat_name_sort,
        "name_token_set_sim": feat_name_set,
        "name_token_jaccard": feat_name_jaccard,
        "name_token_dice": feat_name_dice,
        "name_token_overlap_min": feat_name_overlap,
        "name_exact_match": feat_name_exact,
        "name_prefix_match_4": feat_name_prefix,
        "name_suffix_match_4": feat_name_suffix,
        "name_length_diff": feat_name_len_diff,
        "name_length_ratio": feat_name_len_ratio,
        "name_token_count_diff": feat_name_cnt_diff,
        "name_token_count_ratio": feat_name_cnt_ratio,
        "name_is_single_token": feat_name_single_token,

        "address_is_missing_s1": addr1_missing,
        "address_is_missing_s2": addr2_missing,
        "address_is_missing_either": addr_either_missing,
        "address_is_missing_both": addr_both_missing,
        "address_levenshtein_sim": feat_addr_lev,
        "address_token_sort_sim": feat_addr_sort,
        "address_token_jaccard": feat_addr_jaccard,
        "address_token_dice": feat_addr_dice,
        "address_token_overlap_min": feat_addr_overlap,
        "address_exact_match": feat_addr_exact,
        "address_length_diff": feat_addr_len_diff,
        "address_length_ratio": feat_addr_len_ratio,

        "country_exact_match": feat_country_exact,
        "country_mismatch": feat_country_mismatch,
        "country_is_missing": country_missing,

        "num_name_jaccard": num_name_jaccard,
        "num_name_conflict": num_name_conflict,
        "num_address_jaccard": num_addr_jaccard,
        "num_address_conflict": num_addr_conflict,
        "num_all_jaccard": num_all_jaccard,
        "num_all_has_common": num_all_has_common,
        "num_all_conflict": num_all_conflict,
        "num_primary_match": num_primary_match,

        "name_and_country_match": name_and_country_match,
        "name_and_address_jaccard_prod": name_and_address_prod,
        "overall_composite_sim": overall_composite,
    }


def extract_candidate_features(
    candidate_pairs: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]],
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Batch feature extraction from an iterable of (s1_record, candidate_record) tuples.

    Parameters
    ----------
    candidate_pairs : iterable of (dict, dict)
        Stream or collection of entity record pairs.
    feature_names : list of str, optional
        Subset of features to extract. Defaults to FEATURE_NAMES.

    Returns
    -------
    pd.DataFrame
        DataFrame of features with clean float dtypes and no NaNs.
    """
    cols = feature_names or FEATURE_NAMES
    rows = []

    for s1_rec, cand_rec in candidate_pairs:
        feat_dict = build_features(s1_rec, cand_rec)
        rows.append([feat_dict[k] for k in cols])

    if not rows:
        return pd.DataFrame(columns=cols, dtype=np.float32)

    df = pd.DataFrame(rows, columns=cols, dtype=np.float32)
    return df.fillna(0.0)


def extract_features_from_df(
    pairs_df: pd.DataFrame,
    s1_dict: Dict[str, Dict[str, Any]],
    cand_dict: Dict[str, Dict[str, Any]],
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
) -> pd.DataFrame:
    """
    Extract features from a DataFrame of candidate pairs by looking up records in entity dictionaries.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Table with s1_col and cand_col columns.
    s1_dict : dict
        Mapping s1_id -> S1 record dict.
    cand_dict : dict
        Mapping cand_id -> candidate record dict.

    Returns
    -------
    pd.DataFrame
        Extracted features.
    """
    s1_ids = pairs_df[s1_col].values
    cand_ids = pairs_df[cand_col].values

    rows = []
    empty_rec: Dict[str, Any] = {"business_name": "", "business_address": "", "country": ""}

    for s1_id, cand_id in zip(s1_ids, cand_ids):
        rec1 = s1_dict.get(s1_id, empty_rec)
        rec2 = cand_dict.get(cand_id, empty_rec)
        rows.append(build_features(rec1, rec2))

    if not rows:
        return pd.DataFrame(columns=FEATURE_NAMES, dtype=np.float32)

    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=np.float32).fillna(0.0)