from __future__ import annotations
import numpy as np
import pandas as pd
from antifraud_pipeline.antifraud_lib import ModelTrainer

def train_gbm_suite(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, feature_columns: list[str], cat_column_names: list[str] | None=None, models: list[str] | None=None) -> ModelTrainer:
    trainer = ModelTrainer(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, feature_columns=feature_columns, cat_column_names=cat_column_names)
    to_train = models or ['xgboost', 'lightgbm', 'catboost']
    for name in to_train:
        n = name.lower()
        if n in ('xgboost', 'xgb'):
            trainer.train_xgboost()
        elif n in ('lightgbm', 'lgb', 'lgbm'):
            trainer.train_lightgbm()
        elif n in ('catboost', 'cat'):
            trainer.train_catboost()
        elif n in ('logreg', 'lr', 'logistic'):
            trainer.train_logistic_regression()
        elif n == 'all':
            trainer.train_all()
            break
        else:
            raise ValueError(f'Unknown model: {name!r}')
    return trainer
