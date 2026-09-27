"""
DuckDB and Polars Vectorized Matcher for Business Entity Resolution.
Provides vectorized feature extraction and scoring using Polars and DuckDB,
preserving exact 13-feature semantics and FastLogisticRegression decisions.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import polars as pl

from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.prefilter import prefilter_pair
from business_entity_resolution.src.model import FastLogisticRegression
from sagemaker.vectorized_matcher import CompactInvertedIndex, CompactCand


def extract_features_polars(
    pairs_data: Dict[str, List[Any]]
) -> np.ndarray:
    """
    Extract 13 features using Polars compiled SIMD Rust expressions.
    Guarantees exact mathematical parity with extract_features_vectorized.
    """
    if not pairs_data["s1_name"]:
        return np.empty((0, 13), dtype=np.float32)

    df = pl.DataFrame(pairs_data)

    inter_nt = pl.col("s1_name_toks").list.set_intersection(pl.col("c_name_toks")).list.len()
    union_nt = pl.col("s1_name_toks").list.len() + pl.col("c_name_toks").list.len() - inter_nt
    min_nt = pl.min_horizontal(pl.col("s1_name_toks").list.len(), pl.col("c_name_toks").list.len())

    inter_sh = pl.col("s1_shingles").list.set_intersection(pl.col("c_shingles")).list.len()
    union_sh = pl.col("s1_shingles").list.len() + pl.col("c_shingles").list.len() - inter_sh

    inter_at = pl.col("s1_addr_toks").list.set_intersection(pl.col("c_addr_toks")).list.len()
    union_at = pl.col("s1_addr_toks").list.len() + pl.col("c_addr_toks").list.len() - inter_at
    min_at = pl.min_horizontal(pl.col("s1_addr_toks").list.len(), pl.col("c_addr_toks").list.len())

    inter_num = pl.col("s1_nums").list.set_intersection(pl.col("c_nums")).list.len()
    union_num = pl.col("s1_nums").list.len() + pl.col("c_nums").list.len() - inter_num

    s1_len = pl.col("s1_name").str.len_chars().cast(pl.Float32)
    c_len = pl.col("c_name").str.len_chars().cast(pl.Float32)

    f_df = df.select([
        ((pl.col("s1_raw_name") == pl.col("c_raw_name")) & (pl.col("s1_raw_name") != "")).cast(pl.Float32).alias("f0"),
        ((pl.col("s1_name") == pl.col("c_name")) & (pl.col("s1_name") != "")).cast(pl.Float32).alias("f1"),
        ((pl.col("s1_sig") == pl.col("c_sig")) & (pl.col("s1_sig") != "")).cast(pl.Float32).alias("f2"),
        (inter_nt / union_nt).fill_nan(0.0).alias("f3"),
        (inter_nt / min_nt).fill_nan(0.0).alias("f4"),
        (inter_sh / union_sh).fill_nan(0.0).alias("f5"),
        ((s1_len - c_len).abs() / pl.max_horizontal(s1_len, c_len, 1.0)).alias("f6"),
        ((pl.col("s1_addr") == "") | (pl.col("c_addr") == "")).cast(pl.Float32).alias("f7"),
        ((pl.col("s1_addr") == pl.col("c_addr")) & (pl.col("s1_addr") != "")).cast(pl.Float32).alias("f8"),
        (inter_at / union_at).fill_nan(0.0).alias("f9"),
        (inter_at / min_at).fill_nan(0.0).alias("f10"),
        (inter_num / union_num).fill_nan(0.0).alias("f11"),
        ((pl.col("s1_country") == pl.col("c_country")) & (pl.col("s1_country") != "")).cast(pl.Float32).alias("f12"),
    ])

    return f_df.to_numpy()


def process_s1_batch_polars(
    s1_batch: List[Dict[str, Any]],
    index: CompactInvertedIndex,
    model: FastLogisticRegression,
    threshold: float,
    blocking_strategy: str = "combined",
    prefilter_config: str = "none",
    max_candidates: Optional[int] = None,
) -> List[Tuple[str, List[str], List[str]]]:
    """
    Process an S1 batch using Polars vectorized feature extraction.
    """
    weights = np.array(model.weights, dtype=np.float32)
    bias = float(model.bias)
    logit_thresh = np.log(threshold / (1.0 - threshold)) if (0.0 < threshold < 1.0) else 0.0

    batch_s1_norms = []
    batch_surviving_cands = []
    all_pairs_s1_idx = []
    all_pairs_cand_idx = []

    pairs_data = {
        "s1_raw_name": [],
        "c_raw_name": [],
        "s1_name": [],
        "c_name": [],
        "s1_sig": [],
        "c_sig": [],
        "s1_addr": [],
        "c_addr": [],
        "s1_country": [],
        "c_country": [],
        "s1_name_toks": [],
        "c_name_toks": [],
        "s1_addr_toks": [],
        "c_addr_toks": [],
        "s1_nums": [],
        "c_nums": [],
        "s1_shingles": [],
        "c_shingles": [],
    }

    results_scaffold = []

    for b_i, s1_rec in enumerate(s1_batch):
        s1_norm = normalize_record(s1_rec) if "name_tokens_set" not in s1_rec else s1_rec
        s1_id = s1_norm["entity_id"]
        cand_indices = index.get_candidate_indices(s1_norm, strategy=blocking_strategy)

        if not cand_indices:
            results_scaffold.append((s1_id, [], []))
            continue

        cand_records = [index.cand_records[ci] for ci in cand_indices]

        # Prefilter
        if prefilter_config != "none":
            filtered = [c for c in cand_records if prefilter_pair(s1_norm, c, config=prefilter_config)]
        else:
            filtered = cand_records

        # Capping
        if max_candidates and len(filtered) > max_candidates:
            s1_nt = s1_norm.get("name_tokens_set", set())
            s1_at = s1_norm.get("address_tokens_set", set())
            s1_name = s1_norm.get("business_name", "")

            def cheap_score(c):
                score = 0
                if s1_name == c.business_name:
                    score += 10
                score += len(s1_nt & c.name_tokens_set) * 3
                score += len(s1_at & c.address_tokens_set) * 2
                return score

            filtered.sort(key=cheap_score, reverse=True)
            filtered = filtered[:max_candidates]

        if not filtered:
            results_scaffold.append((s1_id, [], []))
            continue

        cand_eids = sorted([c.entity_id for c in filtered])
        results_scaffold.append((s1_id, cand_eids, []))
        s1_idx_in_active = len(batch_s1_norms)
        batch_s1_norms.append(s1_norm)
        batch_surviving_cands.append(filtered)

        # Append to batch dataframe columns
        s1_raw = s1_norm.get("raw_business_name", "")
        s1_name = s1_norm.get("business_name", "")
        s1_sig = s1_norm.get("name_signature", "")
        s1_addr = s1_norm.get("business_address", "")
        s1_country = s1_norm.get("country", "")
        s1_name_toks = list(s1_norm.get("name_tokens_set", set()))
        s1_addr_toks = list(s1_norm.get("address_tokens_set", set()))
        s1_nums = list(s1_norm.get("address_numbers", set()))
        s1_shingles = list(s1_norm.get("char_shingles", []))

        for c_i, c in enumerate(filtered):
            all_pairs_s1_idx.append(b_i)
            all_pairs_cand_idx.append(c.entity_id)

            pairs_data["s1_raw_name"].append(s1_raw)
            pairs_data["c_raw_name"].append(c.raw_business_name)
            pairs_data["s1_name"].append(s1_name)
            pairs_data["c_name"].append(c.business_name)
            pairs_data["s1_sig"].append(s1_sig)
            pairs_data["c_sig"].append(c.name_signature)
            pairs_data["s1_addr"].append(s1_addr)
            pairs_data["c_addr"].append(c.business_address)
            pairs_data["s1_country"].append(s1_country)
            pairs_data["c_country"].append(c.country)
            pairs_data["s1_name_toks"].append(s1_name_toks)
            pairs_data["c_name_toks"].append(list(c.name_tokens_set))
            pairs_data["s1_addr_toks"].append(s1_addr_toks)
            pairs_data["c_addr_toks"].append(list(c.address_tokens_set))
            pairs_data["s1_nums"].append(s1_nums)
            pairs_data["c_nums"].append(list(c.address_numbers))
            pairs_data["s1_shingles"].append(s1_shingles)
            pairs_data["c_shingles"].append(list(c.char_shingles))

    # If any pairs to score, run Polars vectorized extraction & BLAS dot product
    if pairs_data["s1_name"]:
        X_mat = extract_features_polars(pairs_data)
        logits = np.dot(X_mat, weights) + bias
        pred_mask = logits >= logit_thresh

        s1_matches_map = {}
        for p_idx, matched in enumerate(pred_mask):
            if matched:
                b_i = all_pairs_s1_idx[p_idx]
                cid = all_pairs_cand_idx[p_idx]
                s1_matches_map.setdefault(b_i, []).append(cid)

        # Assemble final results
        final_results = []
        for b_i, (sid, c_eids, _) in enumerate(results_scaffold):
            matches = sorted(s1_matches_map.get(b_i, []))
            final_results.append((sid, c_eids, matches))
        return final_results

    return results_scaffold
