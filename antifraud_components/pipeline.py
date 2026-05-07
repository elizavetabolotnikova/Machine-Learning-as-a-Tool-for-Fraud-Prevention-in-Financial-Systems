from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from antifraud_pipeline.antifraud_lib import FeatureEngineer, DataPreparer, RuleBasedFilter, ModelTrainer, EnsembleStacker, ModelInterpreter, DriftMonitor, CombinedAntiFraudPipeline, BusinessMetricsAnalyzer, RetrainingPipeline, calibrate_and_evaluate, PRIMARY_FBETA
from antifraud_components.rules.l1_rules import fit_l1_rule_filter
from antifraud_components.monitoring.drift import build_drift_monitor

class AntiFraudSystem:

    def __init__(self, feature_engineer: FeatureEngineer | None=None, rule_filter: RuleBasedFilter | None=None, data_preparer: DataPreparer | None=None, business_analyzer: BusinessMetricsAnalyzer | None=None) -> None:
        self.fe = feature_engineer or FeatureEngineer()
        self.rules = rule_filter
        self.data_preparer = data_preparer or DataPreparer()
        self.business = business_analyzer or BusinessMetricsAnalyzer()
        self.trainer: Optional[ModelTrainer] = None
        self.stacker: Optional[EnsembleStacker] = None
        self.pipeline: Optional[CombinedAntiFraudPipeline] = None
        self.drift_monitor: Optional[DriftMonitor] = None
        self.retraining: Optional[RetrainingPipeline] = None

    @classmethod
    def from_config(cls, avg_fraud_amount: float=50000, review_cost: float=500, false_block_cost: float=2000) -> 'AntiFraudSystem':
        return cls(business_analyzer=BusinessMetricsAnalyzer(avg_fraud_amount=avg_fraud_amount, review_cost=review_cost, false_block_cost=false_block_cost))

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str='resolution_fraud', cat_column_names: list[str] | None=None, models: list[str] | None=None) -> 'AntiFraudSystem':
        print('=== AntiFraudSystem.fit() ===')
        print('\n[1] Preparing train/val/test splits...')
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_preparer.prepare(df, feature_columns, target_col)
        print('\n[2] Fitting L1 rule filter...')
        self.rules = fit_l1_rule_filter(X_train, y_train)
        self.rules.apply(X_test)
        self.rules.print_stats()
        print('\n[3] Training GBM suite...')
        self.trainer = ModelTrainer(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, feature_columns=feature_columns, cat_column_names=cat_column_names)
        for name in models or ['xgboost', 'lightgbm', 'catboost']:
            n = name.lower()
            if n in ('xgboost', 'xgb'):
                self.trainer.train_xgboost()
            elif n in ('lightgbm', 'lgb'):
                self.trainer.train_lightgbm()
            elif n in ('catboost', 'cat'):
                self.trainer.train_catboost()
        if self.trainer.predictions:
            print('\n[4] Stacking ensemble...')
            self.stacker = EnsembleStacker(self.trainer.predictions, y_test)
            self.stacker.weighted_average()
        best_model_name = max(self.trainer.results, key=lambda k: self.trainer.results[k].get('F_beta', 0))
        best_model = self.trainer.models[best_model_name]
        best_threshold = self.trainer.results[best_model_name].get('Threshold', 0.5)
        print(f"\n[5] Best model: {best_model_name} (F_β={self.trainer.results[best_model_name]['F_beta']:.4f})")
        if hasattr(best_model, 'model'):
            best_model = best_model['model']
        self.pipeline = CombinedAntiFraudPipeline(rule_filter=self.rules, ml_model=best_model, threshold=best_threshold)
        print('\n[6] Initializing drift monitor...')
        val_probs = self.trainer.predictions.get(best_model_name, np.zeros(len(y_val)))
        self.drift_monitor = build_drift_monitor(reference_data=X_val, reference_scores=val_probs if len(val_probs) == len(y_val) else np.zeros(len(y_val)), reference_labels=y_val.values, feature_names=feature_columns)
        print('\n=== Fit complete ===')
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError('Call fit() first')
        return self.pipeline.predict(X)

    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> dict:
        if self.pipeline is None:
            raise RuntimeError('Call fit() first')
        return self.pipeline.evaluate(X, y_true)

    def check_drift(self, current_data: pd.DataFrame, current_scores: np.ndarray, current_labels: np.ndarray | None=None) -> dict:
        if self.drift_monitor is None:
            raise RuntimeError('Call fit() first')
        from antifraud_components.monitoring.drift import run_drift_check
        return run_drift_check(self.drift_monitor, current_data=current_data, current_scores=current_scores, current_labels=current_labels)
