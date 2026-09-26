def f05(precision, recall):
    """
    Calculate F0.5 score.
    """
    if precision == 0 and recall == 0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)


def evaluate_predictions(ground_truth, predictions):
    """
    Calculate macro-averaged F0.5 across Source1 entities.

    Parameters
    ----------
    ground_truth : dict
        S1 entity ID -> set of true matched IDs.

    predictions : dict
        S1 entity ID -> set of predicted matched IDs.

    Returns
    -------
    float
        Macro F0.5.
    """
    raise NotImplementedError