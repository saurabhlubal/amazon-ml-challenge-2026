"""
Candidate Prefilter Module for Business Entity Resolution.
Applies fast, inexpensive, deterministic filtering to prune impossible candidate pairs
before expensive feature extraction and scoring.

CRITICAL INVARIANT:
High recall must be preserved. A candidate removed by the prefilter can never be recovered.
All operations are cheap O(1) set intersections, string equality, or length checks.
No Levenshtein or fuzzy operations are used.
"""

from typing import Dict, Any, List, Set, Union, Optional


def is_country_compatible(c1: str, c2: str) -> bool:
    """True if countries match or if either is empty/unknown."""
    if not c1 or not c2:
        return True
    return c1 == c2


def prefilter_pair(
    s1: Union[Dict[str, Any], Any],
    c: Union[Dict[str, Any], Any],
    config: str = "conservative",
) -> bool:
    """
    Evaluate whether a candidate pair (s1, c) should survive the cheap prefilter.

    Parameters
    ----------
    s1 : dict or CompactCand
        Normalized Source 1 entity.
    c : dict or CompactCand
        Normalized Candidate entity (Source 2 or 3).
    config : str
        Prefilter strictness configuration:
        - 'none': Keep all candidates (pass-through).
        - 'country_only': Only prune country mismatches.
        - 'conservative': Country match + at least one name or address signal.
        - 'moderate': Country match + token overlap or prefix + number match.
        - 'selective': Country match + strict name or multi-token address overlap.

    Returns
    -------
    bool
        True if the candidate should be kept for feature extraction; False to discard.
    """
    if config == "none":
        return True

    # 1. Country compatibility check (0% cross-country true matches)
    s1_country = s1.get("country", "")
    c_country = c.get("country", "")
    if s1_country and c_country and s1_country != c_country:
        return False

    if config == "country_only":
        return True

    s1_name_toks: Set[str] = s1.get("name_tokens_set", set())
    c_name_toks: Set[str] = c.get("name_tokens_set", set())
    s1_addr_toks: Set[str] = s1.get("address_tokens_set", set())
    c_addr_toks: Set[str] = c.get("address_tokens_set", set())

    # Fast token overlaps
    has_name_tok = bool(s1_name_toks & c_name_toks)
    has_addr_tok = bool(s1_addr_toks & c_addr_toks)

    s1_name = s1.get("business_name", "")
    c_name = c.get("business_name", "")
    s1_sig = s1.get("name_signature", "")
    c_sig = c.get("name_signature", "")
    is_name_match = (s1_name == c_name and bool(s1_name)) or (s1_sig == c_sig and bool(s1_sig))

    if config == "conservative":
        # At least one positive signal must exist:
        # - Any name token overlap
        # - Exact / signature name match
        # - Any address token overlap
        # - Shared address numbers + compact prefix
        if has_name_tok or has_addr_tok or is_name_match:
            return True

        s1_nums: Set[str] = s1.get("address_numbers", set())
        c_nums: Set[str] = c.get("address_numbers", set())
        if s1_nums and c_nums and bool(s1_nums & c_nums):
            s1_cmp = s1.get("compact_name", "")
            c_cmp = c.get("compact_name", "")
            if s1_cmp and c_cmp and len(s1_cmp) >= 3 and len(c_cmp) >= 3 and s1_cmp[:3] == c_cmp[:3]:
                return True

        # Check raw business name equality as ultimate fallback
        s1_raw = s1.get("raw_business_name", "")
        c_raw = c.get("raw_business_name", "")
        if s1_raw and c_raw and s1_raw == c_raw:
            return True

        return False

    elif config == "moderate":
        # Moderate: Stronger name or address overlap requirement
        if is_name_match or has_name_tok:
            return True

        s1_nums = s1.get("address_numbers", set())
        c_nums = c.get("address_numbers", set())
        has_num_overlap = bool(s1_nums & c_nums) if (s1_nums and c_nums) else False

        # If no name token, address must have at least 1 token match
        if has_addr_tok:
            # If address matches, check that name is not wildly incompatible
            s1_cmp = s1.get("compact_name", "")
            c_cmp = c.get("compact_name", "")
            # Either address has 2+ tokens or number overlap, or compact names share prefix
            common_addr_count = len(s1_addr_toks & c_addr_toks)
            if common_addr_count >= 2 or has_num_overlap:
                return True
            if s1_cmp and c_cmp and s1_cmp[:2] == c_cmp[:2]:
                return True
            # Single address token is allowed if address is exact
            s1_addr = s1.get("business_address", "")
            c_addr = c.get("business_address", "")
            if s1_addr and c_addr and s1_addr == c_addr:
                return True

        return False

    elif config == "selective":
        # Selective: Requires either name token overlap, or multi-token address overlap
        if is_name_match:
            return True

        if has_name_tok:
            # Check length difference: cannot differ by > 80% without address support
            max_len = max(len(s1_name), len(c_name), 1)
            len_diff = abs(len(s1_name) - len(c_name)) / max_len
            if len_diff <= 0.80 or has_addr_tok:
                return True

        common_addr = len(s1_addr_toks & c_addr_toks)
        if common_addr >= 2:
            return True

        s1_nums = s1.get("address_numbers", set())
        c_nums = c.get("address_numbers", set())
        if (common_addr >= 1) and (s1_nums and c_nums and (s1_nums & c_nums)):
            return True

        return False

    raise ValueError(f"Unknown prefilter config: {config}")


def filter_candidates(
    s1_norm: Dict[str, Any],
    candidate_records: List[Any],
    config: str = "conservative",
) -> List[Any]:
    """Filter a list of candidate records for a given Source 1 record."""
    if config == "none":
        return candidate_records
    return [c for c in candidate_records if prefilter_pair(s1_norm, c, config=config)]
