from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from snorkel.labeling.model import LabelModel as SnorkelLabelModel
    _SNORKEL_AVAILABLE = True
except ImportError:
    _SNORKEL_AVAILABLE = False
ABSTAIN = -1
NON_FRAUD = 0
FRAUD = 1

class LabelModel:

    def __init__(self, n_sources: int=3, cardinality: int=2, source_weights: list[float] | None=None, snorkel_kwargs: dict | None=None) -> None:
        self.n_sources = n_sources
        self.cardinality = cardinality
        self.source_weights = np.array(source_weights) if source_weights else None
        self.snorkel_kwargs = snorkel_kwargs or {}
        self._model = None
        self._fitted = False
        if _SNORKEL_AVAILABLE:
            self._model = SnorkelLabelModel(cardinality=cardinality, verbose=False, **self.snorkel_kwargs)

    def fit(self, L: np.ndarray, Y_dev: np.ndarray | None=None, n_epochs: int=500, lr: float=0.01, seed: int=42) -> 'LabelModel':
        if _SNORKEL_AVAILABLE and self._model is not None:
            self._model.fit(L_train=L, Y_dev=Y_dev, n_epochs=n_epochs, lr=lr, seed=seed, progress_bar=False)
        else:
            coverage = (L != ABSTAIN).mean(axis=0)
            self.source_weights = coverage / (coverage.sum() + 1e-09)
        self._fitted = True
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError('Call fit() before predict_proba()')
        if _SNORKEL_AVAILABLE and self._model is not None:
            probs_2d = self._model.predict_proba(L)
            return probs_2d[:, FRAUD]
        return self._weighted_vote_proba(L)

    def predict(self, L: np.ndarray, threshold: float=0.5) -> np.ndarray:
        return (self.predict_proba(L) >= threshold).astype(int)

    def _weighted_vote_proba(self, L: np.ndarray) -> np.ndarray:
        weights = self.source_weights
        if weights is None:
            weights = np.ones(L.shape[1]) / L.shape[1]
        non_abstain = (L != ABSTAIN).astype(float)
        fraud_vote = (L == FRAUD).astype(float)
        weighted_fraud = (fraud_vote * weights * non_abstain).sum(axis=1)
        weighted_total = (non_abstain * weights).sum(axis=1)
        probs = np.where(weighted_total > 0, weighted_fraud / weighted_total, 0.5)
        return probs

    @staticmethod
    def llm_to_lf_labels(llm_resolution_fraud: pd.Series, llm_error: pd.Series | None=None) -> np.ndarray:
        labels = np.where(llm_resolution_fraud == 1, FRAUD, NON_FRAUD)
        if llm_error is not None:
            labels = np.where(llm_error.notna(), ABSTAIN, labels)
        return labels

    @staticmethod
    def rule_to_lf_labels(rule_predictions: np.ndarray) -> np.ndarray:
        lf = np.where(rule_predictions == 1, FRAUD, NON_FRAUD)
        lf = np.where(rule_predictions == -1, ABSTAIN, lf)
        return lf

    @staticmethod
    def score_to_lf_labels(probs: np.ndarray, threshold: float=0.5) -> np.ndarray:
        return np.where(probs >= threshold, FRAUD, NON_FRAUD)
