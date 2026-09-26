"""
Competition-aligned evaluation and decision threshold optimization.

Computes Macro F0.5 across Source 1 entities with strict singleton handling,
optimizes decision thresholds to maximize the competition metric, and produces
submission-ready match predictions.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd


def f05(precision: float, recall: float) -> float:
    """
    Calculate F0.5 score from precision and recall.

    Formula
    -------
    F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)

    Parameters
    ----------
    precision : float
    recall : float

    Returns
    -------
    float
        F0.5 score in [0.0, 1.0].
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0

    denom = (0.25 * precision) + recall
    if denom <= 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


# Alias matching instruction
calculate_f05 = f05


def evaluate_single_entity(
    true_set: Set[str],
    pred_set: Set[str],
) -> Tuple[float, float, float]:
    """
    Evaluate one Source 1 entity according to the competition rules.

    Rules
    -----
    - Singletons (true_set is empty):
      - If pred_set is empty: Precision=1.0, Recall=1.0, F0.5=1.0.
      - If pred_set is non-empty: Precision=0.0, Recall=0.0, F0.5=0.0 (false merge).
    - Entities with true matches (true_set non-empty):
      - If pred_set is empty: Precision=0.0, Recall=0.0, F0.5=0.0.
      - If pred_set is non-empty:
          Precision = |true & pred| / |pred|
          Recall = |true & pred| / |true|
          F0.5 = calculate_f05(Precision, Recall)

    Returns
    -------
    tuple of (f05_score, precision, recall)
    """
    if not true_set:
        if not pred_set:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    prec = tp / float(len(pred_set))
    rec = tp / float(len(true_set))
    score = f05(prec, rec)
    return score, prec, rec


def evaluate_predictions(
    ground_truth: Dict[str, Union[Set[str], Sequence[str]]],
    predictions: Dict[str, Union[Set[str], Sequence[str]]],
) -> float:
    """
    Calculate macro-averaged F0.5 across Source1 entities.

    Parameters
    ----------
    ground_truth : dict
        S1 entity ID -> set/list of true matched IDs.
    predictions : dict
        S1 entity ID -> set/list of predicted matched IDs.

    Returns
    -------
    float
        Macro F0.5.
    """
    detailed = evaluate_predictions_detailed(ground_truth, predictions)
    return detailed["macro_f05"]


def evaluate_predictions_detailed(
    ground_truth: Dict[str, Union[Set[str], Sequence[str]]],
    predictions: Dict[str, Union[Set[str], Sequence[str]]],
) -> Dict[str, float]:
    """
    Calculate detailed evaluation metrics (Macro F0.5, Precision, Recall).

    Parameters
    ----------
    ground_truth : dict
        S1 entity ID -> set/list of true matched IDs.
    predictions : dict
        S1 entity ID -> set/list of predicted matched IDs.

    Returns
    -------
    dict
        {'macro_f05': float, 'macro_precision': float, 'macro_recall': float,
         'num_entities': int, 'singletons': int, 'singleton_accuracy': float}
    """
    if not ground_truth:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "num_entities": 0,
            "singletons": 0,
            "singleton_accuracy": 0.0,
        }

    f05_scores = []
    precisions = []
    recalls = []

    num_singletons = 0
    correct_singletons = 0

    for s1_id, t_ids in ground_truth.items():
        true_set = set(t_ids) if not isinstance(t_ids, set) else t_ids
        pred_val = predictions.get(s1_id, set())
        pred_set = set(pred_val) if not isinstance(pred_val, set) else pred_val

        score, prec, rec = evaluate_single_entity(true_set, pred_set)
        f05_scores.append(score)
        precisions.append(prec)
        recalls.append(rec)

        if not true_set:
            num_singletons += 1
            if not pred_set:
                correct_singletons += 1

    singleton_acc = (correct_singletons / num_singletons) if num_singletons > 0 else 1.0

    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "num_entities": len(ground_truth),
        "singletons": num_singletons,
        "singleton_accuracy": float(singleton_acc),
    }


def find_optimal_threshold(
    s1_ids: Sequence[str],
    candidate_ids: Sequence[str],
    pred_probs: Sequence[float],
    ground_truth: Dict[str, Union[Set[str], Sequence[str]]],
    threshold_range: Tuple[float, float, float] = (0.20, 0.95, 0.02),
    max_matches_per_s1: Optional[int] = None,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """
    Sweep probability thresholds to find the threshold maximizing Macro F0.5.

    Parameters
    ----------
    s1_ids : sequence of str
        Source 1 IDs for each scored candidate pair.
    candidate_ids : sequence of str
        Candidate IDs for each scored candidate pair.
    pred_probs : sequence of float
        Match probabilities for each candidate pair.
    ground_truth : dict
        Mapping S1 ID -> set/list of true matched IDs.
    threshold_range : tuple of (start, stop, step)
        Grid for sweeping thresholds.
    max_matches_per_s1 : int, optional
        Maximum matches to allow per S1 entity.

    Returns
    -------
    dict
        {'best_threshold': float, 'best_f05': float, 'best_precision': float,
         'best_recall': float, 'sweep_results': list}
    """
    # Group pairs by s1_id for rapid evaluation
    pairs_by_s1: Dict[str, List[Tuple[str, float]]] = {}
    for s1, cid, p in zip(s1_ids, candidate_ids, pred_probs):
        if s1 not in pairs_by_s1:
            pairs_by_s1[s1] = []
        pairs_by_s1[s1].append((cid, float(p)))

    start, stop, step = threshold_range
    thresholds = np.arange(start, stop + 1e-6, step)

    best_thresh = float(thresholds[0])
    best_res = {"macro_f05": -1.0, "macro_precision": 0.0, "macro_recall": 0.0}
    sweep_history = []

    for t in thresholds:
        t_val = float(round(t, 4))
        # Build predictions for this threshold
        predictions: Dict[str, Set[str]] = {}
        for s1 in ground_truth.keys():
            candidates = pairs_by_s1.get(s1, [])
            valid_cands = [(cid, score) for cid, score in candidates if score >= t_val]
            if max_matches_per_s1 is not None and max_matches_per_s1 > 0:
                valid_cands.sort(key=lambda x: x[1], reverse=True)
                valid_cands = valid_cands[:max_matches_per_s1]
            predictions[s1] = {cid for cid, _ in valid_cands}

        metrics = evaluate_predictions_detailed(ground_truth, predictions)
        sweep_history.append(
            {
                "threshold": t_val,
                "macro_f05": metrics["macro_f05"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
            }
        )

        if metrics["macro_f05"] > best_res["macro_f05"]:
            best_res = metrics
            best_thresh = t_val

    return {
        "best_threshold": best_thresh,
        "best_f05": best_res["macro_f05"],
        "best_precision": best_res["macro_precision"],
        "best_recall": best_res["macro_recall"],
        "sweep_history": sweep_history,
    }


def apply_decision_rule(
    scored_candidates_df: pd.DataFrame,
    all_s1_ids: Iterable[str],
    score_column: str = "score",
    threshold: float = 0.70,
    max_matches_per_s1: Optional[int] = None,
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
) -> pd.DataFrame:
    """
    Format predictions into competition-compliant DataFrame.

    Produces a DataFrame with exactly two columns:
    ['source1_entity_id', 'matched_entity_ids']
    where matched_entity_ids is a comma-separated list of matches (empty for singletons).
    Ensures every S1 ID in all_s1_ids appears exactly once.

    Parameters
    ----------
    scored_candidates_df : pd.DataFrame
        DataFrame containing s1_col, cand_col, and score_column.
    all_s1_ids : iterable of str
        The full required set of Source 1 IDs (e.g. from test_source1.tsv).
    score_column : str
        Name of score column.
    threshold : float
        Minimum score for a match.
    max_matches_per_s1 : int, optional
        Maximum number of matched IDs per S1.
    s1_col : str
    cand_col : str

    Returns
    -------
    pd.DataFrame
        ['source1_entity_id', 'matched_entity_ids']
    """
    # Filter above threshold
    filtered = scored_candidates_df[scored_candidates_df[score_column] >= threshold]

    # Group by S1
    matches_map: Dict[str, List[str]] = {}
    if not filtered.empty:
        # Sort descending within group
        sorted_df = filtered.sort_values([s1_col, score_column], ascending=[True, False])
        for s1, group in sorted_df.groupby(s1_col):
            c_list = list(group[cand_col].values)
            if max_matches_per_s1 is not None and max_matches_per_s1 > 0:
                c_list = c_list[:max_matches_per_s1]
            matches_map[s1] = c_list

    # Ensure every S1 ID is present in output exactly once
    rows = []
    for s1 in all_s1_ids:
        c_list = matches_map.get(s1, [])
        rows.append(
            {
                "source1_entity_id": s1,
                "matched_entity_ids": ",".join(c_list),
            }
        )

    return pd.DataFrame(rows)
