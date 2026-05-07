from __future__ import annotations
import numpy as np
from antifraud_pipeline.antifraud_lib import EnsembleStacker

def stack_predictions(base_predictions: dict[str, np.ndarray], y_test: np.ndarray, method: str='weighted', weights: np.ndarray | None=None) -> tuple[np.ndarray, float, float]:
    stacker = EnsembleStacker(base_predictions, y_test)
    if method == 'simple':
        return stacker.simple_average()
    elif method == 'rank':
        return stacker.rank_average()
    else:
        return stacker.weighted_average(weights=weights)

def add_gnn_to_stack(base_predictions: dict[str, np.ndarray], gnn_probs: np.ndarray, test_mask: np.ndarray, gnn_key: str='GNN') -> dict[str, np.ndarray]:
    updated = dict(base_predictions)
    updated[gnn_key] = gnn_probs[test_mask]
    return updated
