"""
Supervised machine learning model for pairwise entity matching.

Provides a fast, precision-oriented tabular classifier (LightGBM with
HistGradientBoosting fallback) and matching decision logic.
"""

from __future__ import annotations

import os
import pickle
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try LightGBM, with graceful fallback to scikit-learn
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


class EntityMatcher:
    """
    Lightweight supervised classifier for entity match scoring.
    """

    def __init__(
        self,
        algorithm: str = "auto",
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        algorithm : str, default='auto'
            'auto', 'lightgbm', 'hist_gb', or 'random_forest'.
        n_estimators : int
            Number of boosting trees.
        learning_rate : float
            Shrinkage rate.
        max_depth : int
            Maximum tree depth.
        num_leaves : int
            Maximum tree leaves (LightGBM).
        subsample : float
            Row subsampling fraction.
        colsample_bytree : float
            Feature subsampling fraction.
        random_state : int
            Random seed.
        """
        self.algorithm = algorithm
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model = None
        self.backend = None
        self.feature_names: Optional[List[str]] = None

    def _init_backend(self):
        if self.algorithm == "lightgbm" or (self.algorithm == "auto" and HAS_LIGHTGBM):
            self.backend = "lightgbm"
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                subsample=self.subsample,
                subsample_freq=1,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
        elif self.algorithm == "random_forest":
            self.backend = "random_forest"
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            self.backend = "hist_gb"
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, Sequence[int]],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series, Sequence[int]]] = None,
        early_stopping_rounds: int = 25,
    ) -> EntityMatcher:
        """
        Fit the model on pairwise features X and binary match labels y.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y, dtype=np.int32)

        self._init_backend()

        if self.backend == "lightgbm" and X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_arr = X_val.values.astype(np.float32)
            else:
                X_val_arr = np.asarray(X_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.int32)

            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            try:
                self.model.fit(
                    X_arr,
                    y_arr,
                    eval_X=X_val_arr,
                    eval_y=y_val_arr,
                    callbacks=callbacks,
                )
            except TypeError:
                self.model.fit(
                    X_arr,
                    y_arr,
                    eval_set=[(X_val_arr, y_val_arr)],
                    callbacks=callbacks,
                )
        else:
            self.model.fit(X_arr, y_arr)

        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict probability of match (class 1) for each sample.

        Returns
        -------
        np.ndarray
            1D array of match probabilities in [0.0, 1.0].
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if len(X_arr) == 0:
            return np.array([], dtype=np.float32)

        probs = self.model.predict_proba(X_arr)
        # Class 1 probabilities
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1].astype(np.float32)
        elif probs.ndim == 2 and probs.shape[1] == 1:
            return probs[:, 0].astype(np.float32)
        return probs.astype(np.float32)

    def save(self, path: str):
        """Serialize model to file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> EntityMatcher:
        """Load serialized model from file."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ======================================================================
# Shared Upstream Interface Functions
# ======================================================================

def train_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, Sequence[int]],
    **kwargs,
) -> EntityMatcher:
    """
    Train the entity matching model.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Training features.
    y : array-like
        Binary match labels.

    Returns
    -------
    EntityMatcher
        Trained matcher model.
    """
    matcher = EntityMatcher(**kwargs)
    matcher.fit(X, y)
    return matcher


def predict_scores(
    model: Union[EntityMatcher, Any],
    X: Union[np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """
    Predict match scores / probabilities for feature matrix X.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise AttributeError("Model does not provide predict_proba method.")


def decide_matches(
    candidate_ids: Sequence[str],
    scores: Sequence[float],
    threshold: float,
    max_matches: Optional[int] = None,
) -> List[str]:
    """
    Convert candidate scores into final matched IDs for one Source 1 entity.

    Filters candidates whose score is >= threshold.
    If max_matches is specified, keeps only the top-k highest scoring matches.

    Parameters
    ----------
    candidate_ids : list[str]
        Candidate entity IDs.
    scores : array-like
        Match scores corresponding to candidate_ids.
    threshold : float
        Minimum score required for a match.
    max_matches : int, optional
        Maximum number of matched IDs to return.

    Returns
    -------
    list[str]
        Final matched entity IDs in descending score order.
    """
    if not candidate_ids or len(scores) == 0:
        return []

    # Filter by threshold
    passed = [
        (cid, float(sc))
        for cid, sc in zip(candidate_ids, scores)
        if sc >= threshold
    ]

    if not passed:
        return []

    # Sort descending by score
    passed.sort(key=lambda x: x[1], reverse=True)

    if max_matches is not None and max_matches > 0:
        passed = passed[:max_matches]

    return [cid for cid, _ in passed]