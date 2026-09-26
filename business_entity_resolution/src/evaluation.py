"""
Evaluation module for Business Entity Resolution.
Calculates macro-averaged F0.5 score conforming to the official challenge specification.
"""

from typing import Dict, Any, Set, Union, List


def f05(precision: float, recall: float) -> float:
    """
    Calculate F0.5 score: (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Weights precision 2x over recall.
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    denominator = 0.25 * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.25 * precision * recall) / denominator


def to_id_set(val: Any) -> Set[str]:
    """Helper to convert sets, lists, or comma-separated strings to set of cleaned IDs."""
    if val is None:
        return set()
    if isinstance(val, set):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, (list, tuple)):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, str):
        if not val.strip():
            return set()
        return {x.strip() for x in val.split(",") if x.strip()}
    return set()


def evaluate_predictions(
    ground_truth: Dict[str, Union[Set[str], List[str], str]],
    predictions: Dict[str, Union[Set[str], List[str], str]]
) -> float:
    """
    Calculate macro-averaged F0.5 across all Source1 entities in ground_truth.

    Parameters
    ----------
    ground_truth : dict
        S1 entity ID -> true matched IDs (set, list, or comma-separated string).
    predictions : dict
        S1 entity ID -> predicted matched IDs (set, list, or comma-separated string).

    Returns
    -------
    float
        Macro F0.5 score across all entities.
    """
    if not ground_truth:
        return 0.0

    total_score = 0.0
    n_entities = len(ground_truth)

    for s1_id, gt_val in ground_truth.items():
        true_ids = to_id_set(gt_val)
        pred_val = predictions.get(s1_id, set())
        pred_ids = to_id_set(pred_val)

        # Singleton evaluation (zero true matches)
        if not true_ids:
            if not pred_ids:
                total_score += 1.0  # Correctly predicted no-match
            else:
                total_score += 0.0  # False merge on singleton
            continue

        # Non-singleton entity with zero predictions
        if not pred_ids:
            total_score += 0.0  # Missed all matches
            continue

        # True matches present and predictions made
        true_positives = len(true_ids & pred_ids)
        if true_positives == 0:
            total_score += 0.0
            continue

        prec = true_positives / len(pred_ids)
        rec = true_positives / len(true_ids)
        total_score += f05(prec, rec)

    return total_score / n_entities