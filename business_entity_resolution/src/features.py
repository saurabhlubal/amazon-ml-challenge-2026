"""
Pairwise feature extraction for Business Entity Resolution.

Extracts similarity, lexical, numeric, and categorical comparison signals
between a Source 1 query entity and candidate entities (Source 2 / Source 3).
Includes parsed record caching, legal entity normalization, address component
normalization, and multilingual Unicode safety.
"""

from __future__ import annotations

import collections
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

# Rapidfuzz with fallback to standard library difflib
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

# Compiled regular expressions for speed (Unicode-aware)
RE_ALPHANUM = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\b\d+\b")
RE_TOKEN = re.compile(r"\b\w+\b", re.UNICODE)

# Common business legal entity abbreviations
LEGAL_ENTITY_MAP: Dict[str, str] = {
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "limited": "ltd",
    "ltd": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "company": "co",
    "co": "co",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "gmbh": "gmbh",
    "sa": "sa",
    "sarl": "sarl",
}

# Common address street suffixes
STREET_SUFFIX_MAP: Dict[str, str] = {
    "street": "st",
    "st": "st",
    "road": "rd",
    "rd": "rd",
    "drive": "dr",
    "dr": "dr",
    "avenue": "ave",
    "ave": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "lane": "ln",
    "ln": "ln",
    "highway": "hwy",
    "hwy": "hwy",
    "circle": "cir",
    "cir": "cir",
    "court": "ct",
    "ct": "ct",
}


import unicodedata

def strip_latin_accents(text: str) -> str:
    """Strip combining diacritics from Latin scripts (e.g. é -> e) while preserving Indic/Devanagari matras."""
    result = []
    for c in unicodedata.normalize("NFKD", text):
        if "\u0300" <= c <= "\u036f":
            continue
        result.append(c)
    return "".join(result)


def clean_text(text: Optional[str]) -> str:
    """Standardize string: lowercase, strip Latin accents, remove punctuation, collapse whitespace (Unicode-safe)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = strip_latin_accents(text)
    text = RE_ALPHANUM.sub(" ", text.lower())
    return RE_WHITESPACE.sub(" ", text).strip()


def get_tokens(text: str) -> List[str]:
    """Tokenize clean text into list of word tokens."""
    return RE_TOKEN.findall(text)


def get_digits(text: Optional[str]) -> List[str]:
    """Extract all standalone digit sequences."""
    if not text:
        return []
    return RE_DIGITS.findall(str(text))


def normalize_legal_tokens(tokens: Sequence[str]) -> List[str]:
    """Map legal entity abbreviations to standardized canonical forms."""
    return [LEGAL_ENTITY_MAP.get(t, t) for t in tokens]


def normalize_street_tokens(tokens: Sequence[str]) -> List[str]:
    """Map street/address abbreviations to standardized canonical forms."""
    return [STREET_SUFFIX_MAP.get(t, t) for t in tokens]


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


class ParsedRecord:
    """
    Cached, pre-tokenized record for rapid pairwise feature computation.
    """
    __slots__ = (
        "entity_id", "raw_name", "raw_addr", "country",
        "name", "addr", "tokens_n", "set_n",
        "norm_tokens_n", "norm_set_n", "core_set_n", "tokens_a", "set_a",
        "norm_tokens_a", "norm_set_a", "digits_n", "digits_a",
        "all_digits", "first_num", "len_name", "len_addr",
        "cnt_name", "cnt_addr", "addr_missing",
    )

    def __init__(self, record: Dict[str, Any]):
        self.entity_id = str(record.get("entity_id") or "")
        self.raw_name = str(record.get("business_name") or "")
        self.raw_addr = str(record.get("business_address") or "")
        self.country = str(record.get("country") or "").strip().upper()

        self.name = clean_text(self.raw_name)
        self.addr = clean_text(self.raw_addr)

        self.tokens_n = get_tokens(self.name)
        self.set_n = set(self.tokens_n)
        self.norm_tokens_n = normalize_legal_tokens(self.tokens_n)
        self.norm_set_n = set(self.norm_tokens_n)
        legal_words = set(LEGAL_ENTITY_MAP.keys()) | set(LEGAL_ENTITY_MAP.values())
        self.core_set_n = {t for t in self.tokens_n if t not in legal_words}

        self.tokens_a = get_tokens(self.addr)
        self.set_a = set(self.tokens_a)
        self.norm_tokens_a = normalize_street_tokens(self.tokens_a)
        self.norm_set_a = set(self.norm_tokens_a)

        self.digits_n = set(get_digits(self.raw_name))
        self.digits_a = set(get_digits(self.raw_addr))
        self.all_digits = self.digits_n | self.digits_a

        all_nums = get_digits(self.raw_addr) or get_digits(self.raw_name) or [None]
        self.first_num = all_nums[0]

        self.len_name = len(self.name)
        self.len_addr = len(self.addr)
        self.cnt_name = len(self.tokens_n)
        self.cnt_addr = len(self.tokens_a)
        self.addr_missing = 1.0 if not self.addr else 0.0


# Complete Feature Names Specification (44 features)
FEATURE_NAMES: List[str] = [
    # Business Name features
    "name_levenshtein_sim",
    "name_token_sort_sim",
    "name_token_set_sim",
    "name_token_jaccard",
    "name_token_dice",
    "name_token_overlap_min",
    "name_norm_token_jaccard",
    "name_norm_token_overlap_min",
    "name_core_token_jaccard",
    "name_core_token_dice",
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
    "address_norm_token_jaccard",
    "address_norm_token_overlap_min",
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


def build_features_from_parsed(p1: ParsedRecord, p2: ParsedRecord) -> Dict[str, float]:
    """
    Build features using pre-parsed records for ultra-fast throughput.
    """
    # 1. Name features
    feat_name_lev = levenshtein_sim(p1.name, p2.name)
    feat_name_sort = token_sort_sim(p1.name, p2.name)
    feat_name_set = token_set_sim(p1.name, p2.name)
    feat_name_jaccard = jaccard_similarity(p1.set_n, p2.set_n)
    feat_name_dice = dice_similarity(p1.set_n, p2.set_n)
    feat_name_overlap = overlap_coefficient(p1.set_n, p2.set_n)
    feat_name_norm_jaccard = jaccard_similarity(p1.norm_set_n, p2.norm_set_n)
    feat_name_norm_overlap = overlap_coefficient(p1.norm_set_n, p2.norm_set_n)
    feat_name_core_jaccard = jaccard_similarity(p1.core_set_n, p2.core_set_n)
    feat_name_core_dice = dice_similarity(p1.core_set_n, p2.core_set_n)
    feat_name_exact = 1.0 if p1.name and p1.name == p2.name else 0.0

    # Prefix/suffix match (min 4 chars)
    prefix_len = 4
    if p1.len_name >= prefix_len and p2.len_name >= prefix_len:
        feat_name_prefix = 1.0 if p1.name[:prefix_len] == p2.name[:prefix_len] else 0.0
        feat_name_suffix = 1.0 if p1.name[-prefix_len:] == p2.name[-prefix_len:] else 0.0
    else:
        feat_name_prefix = 1.0 if p1.name and p1.name == p2.name else 0.0
        feat_name_suffix = 1.0 if p1.name and p1.name == p2.name else 0.0

    max_n_len = max(p1.len_name, p2.len_name)
    feat_name_len_diff = float(abs(p1.len_name - p2.len_name))
    feat_name_len_ratio = (min(p1.len_name, p2.len_name) / max_n_len) if max_n_len > 0 else 1.0

    max_n_cnt = max(p1.cnt_name, p2.cnt_name)
    feat_name_cnt_diff = float(abs(p1.cnt_name - p2.cnt_name))
    feat_name_cnt_ratio = (min(p1.cnt_name, p2.cnt_name) / max_n_cnt) if max_n_cnt > 0 else 1.0
    feat_name_single_token = 1.0 if (p1.cnt_name == 1 or p2.cnt_name == 1) else 0.0

    # 2. Address features
    addr_either_missing = 1.0 if (p1.addr_missing or p2.addr_missing) else 0.0
    addr_both_missing = 1.0 if (p1.addr_missing and p2.addr_missing) else 0.0

    if addr_either_missing:
        feat_addr_lev = 0.0
        feat_addr_sort = 0.0
        feat_addr_jaccard = 0.0
        feat_addr_dice = 0.0
        feat_addr_overlap = 0.0
        feat_addr_norm_jaccard = 0.0
        feat_addr_norm_overlap = 0.0
        feat_addr_exact = 0.0
        feat_addr_len_diff = float(abs(p1.len_addr - p2.len_addr))
        feat_addr_len_ratio = 0.0
    else:
        feat_addr_lev = levenshtein_sim(p1.addr, p2.addr)
        feat_addr_sort = token_sort_sim(p1.addr, p2.addr)
        feat_addr_jaccard = jaccard_similarity(p1.set_a, p2.set_a)
        feat_addr_dice = dice_similarity(p1.set_a, p2.set_a)
        feat_addr_overlap = overlap_coefficient(p1.set_a, p2.set_a)
        feat_addr_norm_jaccard = jaccard_similarity(p1.norm_set_a, p2.norm_set_a)
        feat_addr_norm_overlap = overlap_coefficient(p1.norm_set_a, p2.norm_set_a)
        feat_addr_exact = 1.0 if p1.addr == p2.addr else 0.0
        feat_addr_len_diff = float(abs(p1.len_addr - p2.len_addr))
        max_a_len = max(p1.len_addr, p2.len_addr)
        feat_addr_len_ratio = min(p1.len_addr, p2.len_addr) / max_a_len if max_a_len > 0 else 1.0

    # 3. Country features
    country_missing = 1.0 if (not p1.country or not p2.country) else 0.0
    if country_missing:
        feat_country_exact = 0.0
        feat_country_mismatch = 0.0
    elif p1.country == p2.country:
        feat_country_exact = 1.0
        feat_country_mismatch = 0.0
    else:
        feat_country_exact = 0.0
        feat_country_mismatch = 1.0

    # 4. Numeric agreement
    num_name_jaccard = jaccard_similarity(p1.digits_n, p2.digits_n)
    num_name_conflict = 1.0 if (p1.digits_n and p2.digits_n and not (p1.digits_n & p2.digits_n)) else 0.0

    num_addr_jaccard = jaccard_similarity(p1.digits_a, p2.digits_a)
    num_addr_conflict = 1.0 if (p1.digits_a and p2.digits_a and not (p1.digits_a & p2.digits_a)) else 0.0

    num_all_jaccard = jaccard_similarity(p1.all_digits, p2.all_digits)
    num_all_has_common = 1.0 if (p1.all_digits & p2.all_digits) else 0.0
    num_all_conflict = 1.0 if (p1.all_digits and p2.all_digits and not (p1.all_digits & p2.all_digits)) else 0.0

    if p1.first_num is not None and p2.first_num is not None:
        num_primary_match = 1.0 if p1.first_num == p2.first_num else 0.0
    else:
        num_primary_match = 0.0

    # 5. Cross-field interactions
    name_and_country_match = feat_name_exact * feat_country_exact
    name_and_address_prod = feat_name_norm_jaccard * feat_addr_norm_jaccard
    overall_composite = (0.5 * feat_name_sort) + (0.35 * feat_addr_sort) + (0.15 * feat_country_exact)

    return {
        "name_levenshtein_sim": feat_name_lev,
        "name_token_sort_sim": feat_name_sort,
        "name_token_set_sim": feat_name_set,
        "name_token_jaccard": feat_name_jaccard,
        "name_token_dice": feat_name_dice,
        "name_token_overlap_min": feat_name_overlap,
        "name_norm_token_jaccard": feat_name_norm_jaccard,
        "name_norm_token_overlap_min": feat_name_norm_overlap,
        "name_core_token_jaccard": feat_name_core_jaccard,
        "name_core_token_dice": feat_name_core_dice,
        "name_exact_match": feat_name_exact,
        "name_prefix_match_4": feat_name_prefix,
        "name_suffix_match_4": feat_name_suffix,
        "name_length_diff": feat_name_len_diff,
        "name_length_ratio": feat_name_len_ratio,
        "name_token_count_diff": feat_name_cnt_diff,
        "name_token_count_ratio": feat_name_cnt_ratio,
        "name_is_single_token": feat_name_single_token,

        "address_is_missing_s1": p1.addr_missing,
        "address_is_missing_s2": p2.addr_missing,
        "address_is_missing_either": addr_either_missing,
        "address_is_missing_both": addr_both_missing,
        "address_levenshtein_sim": feat_addr_lev,
        "address_token_sort_sim": feat_addr_sort,
        "address_token_jaccard": feat_addr_jaccard,
        "address_token_dice": feat_addr_dice,
        "address_token_overlap_min": feat_addr_overlap,
        "address_norm_token_jaccard": feat_addr_norm_jaccard,
        "address_norm_token_overlap_min": feat_addr_norm_overlap,
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
    candidate_record : dict
        Source2 or Source3 candidate entity containing:
        entity_id, business_name, business_address, country

    Returns
    -------
    dict
        Feature name -> numeric float value.
    """
    p1 = source1_record if isinstance(source1_record, ParsedRecord) else ParsedRecord(source1_record)
    p2 = candidate_record if isinstance(candidate_record, ParsedRecord) else ParsedRecord(candidate_record)
    return build_features_from_parsed(p1, p2)


def extract_candidate_features(
    candidate_pairs: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]],
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Batch feature extraction from an iterable of (s1_record, candidate_record) tuples.

    Parameters
    ----------
    candidate_pairs : iterable of (dict, dict)
        Collection of entity record pairs.
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
    Extract features with parsed record caching for maximum efficiency.
    """
    s1_ids = pairs_df[s1_col].values
    cand_ids = pairs_df[cand_col].values

    # Pre-parse unique records
    parsed_s1: Dict[str, ParsedRecord] = {}
    parsed_cand: Dict[str, ParsedRecord] = {}
    empty_raw = {"business_name": "", "business_address": "", "country": ""}

    unique_s1 = set(s1_ids)
    unique_cand = set(cand_ids)

    for sid in unique_s1:
        raw = s1_dict.get(sid, empty_raw)
        parsed_s1[sid] = ParsedRecord(raw)

    for cid in unique_cand:
        raw = cand_dict.get(cid, empty_raw)
        parsed_cand[cid] = ParsedRecord(raw)

    rows = []
    for sid, cid in zip(s1_ids, cand_ids):
        p1 = parsed_s1[sid]
        p2 = parsed_cand[cid]
        rows.append(build_features_from_parsed(p1, p2))

    if not rows:
        return pd.DataFrame(columns=FEATURE_NAMES, dtype=np.float32)

    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=np.float32).fillna(0.0)