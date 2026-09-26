def train_model(X, y):
    """
    Train the entity matching model.

    Parameters
    ----------
    X : array-like
        Training features.

    y : array-like
        Binary match labels.

    Returns
    -------
    object
        Trained model.
    """
    raise NotImplementedError


def predict_scores(model, X):
    """
    Predict match scores/probabilities.
    """
    raise NotImplementedError


def decide_matches(candidate_ids, scores, threshold):
    """
    Convert candidate scores into final matched IDs.

    Parameters
    ----------
    candidate_ids : list[str]
        Candidate entity IDs.

    scores : array-like
        Match scores corresponding to candidate_ids.

    threshold : float
        Minimum score required for a match.

    Returns
    -------
    list[str]
        Final matched entity IDs.
    """
    raise NotImplementedError