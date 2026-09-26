"""
Model module for Business Entity Resolution.
Implements a fast, calibrated baseline classifier supporting both sklearn
and a zero-dependency vectorized NumPy/SciPy fallback.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
from scipy.optimize import minimize
from scipy.special import expit


FEATURE_NAMES = [
    "name_exact_raw",
    "name_exact_norm",
    "name_sig_match",
    "name_jaccard",
    "name_containment",
    "name_ngram_jaccard",
    "name_len_diff",
    "addr_is_empty",
    "addr_exact_norm",
    "addr_jaccard",
    "addr_containment",
    "addr_num_overlap",
    "country_match",
]


def extract_feature_matrix(X: Union[List[Dict[str, float]], np.ndarray]) -> np.ndarray:
    """Convert a list of feature dictionaries or array into a standard 2D NumPy array."""
    if isinstance(X, np.ndarray):
        return X.astype(np.float64)

    if not X:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)

    if isinstance(X[0], dict):
        matrix = np.zeros((len(X), len(FEATURE_NAMES)), dtype=np.float64)
        for i, row in enumerate(X):
            for j, fname in enumerate(FEATURE_NAMES):
                matrix[i, j] = float(row.get(fname, 0.0))
        return matrix

    return np.array(X, dtype=np.float64)


class FastLogisticRegression:
    """Fast, robust L2-regularized Logistic Regression implemented in NumPy / SciPy."""

    def __init__(self, l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        if n_samples == 0:
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            return self

        # Objective function and gradient for L-BFGS-B
        def loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            z = X @ w + b
            # Use stable expit for sigmoid
            p = expit(z)
            eps = 1e-15
            p_clipped = np.clip(p, eps, 1.0 - eps)

            # Binary cross entropy + L2 penalty
            bce = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
            reg = 0.5 * self.l2_reg * np.sum(w ** 2)
            total_loss = bce + reg

            # Gradients
            err = (p - y) / n_samples
            grad_w = (X.T @ err) + self.l2_reg * w
            grad_b = np.sum(err)
            grad = np.append(grad_w, grad_b)
            return total_loss, grad

        init_params = np.zeros(n_features + 1)
        res = minimize(
            loss_and_grad,
            init_params,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 100}
        )

        self.weights = res.x[:-1]
        self.bias = float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        z = X @ self.weights + self.bias
        return expit(z)


def train_model(X: Union[List[Dict[str, float]], np.ndarray], y: Union[List[int], np.ndarray]) -> Any:
    """
    Train the entity matching model.

    Parameters
    ----------
    X : array-like or list of feature dicts
        Training pair comparison features.
    y : array-like
        Binary match labels (1 for true match, 0 for negative candidate).

    Returns
    -------
    object
        Trained model instance.
    """
    X_mat = extract_feature_matrix(X)
    y_arr = np.array(y, dtype=np.float64)

    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf.fit(X_mat, y_arr)
        return clf
    except ImportError:
        # Fallback to pure NumPy/SciPy L-BFGS logistic regression
        model = FastLogisticRegression(l2_reg=0.1)
        model.fit(X_mat, y_arr)
        return model


def predict_scores(model: Any, X: Union[List[Dict[str, float]], np.ndarray]) -> List[float]:
    """
    Predict match scores / probabilities for candidate pairs.

    Parameters
    ----------
    model : object
        Trained matching model.
    X : list of dicts or 2D array
        Pair features.

    Returns
    -------
    list[float]
        Predicted match probability for each candidate.
    """
    if model is None:
        # Fallback heuristic if no model is provided
        X_mat = extract_feature_matrix(X)
        if len(X_mat) == 0:
            return []
        # Heuristic weighted sum of name and country features
        scores = 0.5 * X_mat[:, 3] + 0.3 * X_mat[:, 8] + 0.2 * X_mat[:, 11]
        return scores.tolist()

    X_mat = extract_feature_matrix(X)
    if len(X_mat) == 0:
        return []

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_mat)
        if probs.ndim == 2:
            return probs[:, 1].tolist()
        return probs.tolist()

    return [0.0] * len(X_mat)


def decide_matches(
    candidate_ids: List[str],
    scores: Union[List[float], np.ndarray],
    threshold: float = 0.5
) -> List[str]:
    """
    Convert candidate scores into final matched IDs based on threshold.

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
    matches = []
    for cid, score in zip(candidate_ids, scores):
        if score >= threshold:
            matches.append(cid)
    return matches