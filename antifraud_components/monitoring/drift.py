from __future__ import annotations
import numpy as np
import pandas as pd
from antifraud_pipeline.antifraud_lib import DriftMonitor

def run_drift_check(monitor: DriftMonitor, current_data: pd.DataFrame, current_scores: np.ndarray, current_labels: np.ndarray | None=None, threshold: float=0.5, verbose: bool=True) -> dict:
    report = monitor.full_monitoring_check(current_data=current_data, current_scores=current_scores, current_labels=current_labels, threshold=threshold)
    if verbose:
        print(f"Drift recommendation: {report['recommendation']}")
        for alert in report.get('alerts', []):
            print(f'  {alert}')
    monitor.plot_drift_report(report, current_scores=current_scores)
    return report

def build_drift_monitor(reference_data: pd.DataFrame, reference_scores: np.ndarray, reference_labels: np.ndarray, feature_names: list[str]) -> DriftMonitor:
    return DriftMonitor(reference_data=reference_data, reference_scores=reference_scores, reference_labels=reference_labels, feature_names=feature_names)
