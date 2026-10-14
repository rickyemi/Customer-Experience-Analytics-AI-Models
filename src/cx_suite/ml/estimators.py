"""Custom estimators kept in an importable module so pickled models load anywhere."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from ..config import SEED
from .features import preprocessor


def _hgb(seed: int) -> Pipeline:
    return Pipeline([("prep", preprocessor(scale=False)),
                     ("clf", HistGradientBoostingClassifier(
                         learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                         min_samples_leaf=150, l2_regularization=1.0, random_state=seed))])


class TLearner:
    """Two-model uplift learner: tau(x) = P(y=1 | x, treated) - P(y=1 | x, control)."""

    def __init__(self, seed: int = SEED):
        self.m1, self.m0 = _hgb(seed), _hgb(seed)

    def fit(self, X, t, y):
        self.m1.fit(X[t == 1], y[t == 1])
        self.m0.fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]
