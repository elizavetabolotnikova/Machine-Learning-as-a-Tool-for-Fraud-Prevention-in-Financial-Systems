# -*- coding: utf-8 -*-
"""Пакет ML-пайплайна антифрода (код в `antifraud_lib`, этапы в `antifraud_stages`).

CLI: ``python -m antifraud_pipeline`` из корня репозитория (где лежат ``data/``, ``outputs/``).
"""
from .antifraud_stages import (
    P,
    init_notebook,
    resolve_best_ml_model_name_for_combined,
    run_all_stages,
    stage_01_load_and_profile,
    stage_02_eda_plots,
    stage_03_feature_engineering,
    stage_04_prepare_split,
    stage_05_rules,
    stage_06_train_models,
    stage_07_compare_plots,
    stage_08_neural,
    stage_09_ensemble,
    stage_10_shap,
    stage_11_drift,
    stage_12_business,
    stage_13_calibration,
    stage_14_retraining,
    stage_15_combined,
    stage_16_report,
)

__all__ = [
    "P",
    "init_notebook",
    "resolve_best_ml_model_name_for_combined",
    "run_all_stages",
    "stage_01_load_and_profile",
    "stage_02_eda_plots",
    "stage_03_feature_engineering",
    "stage_04_prepare_split",
    "stage_05_rules",
    "stage_06_train_models",
    "stage_07_compare_plots",
    "stage_08_neural",
    "stage_09_ensemble",
    "stage_10_shap",
    "stage_11_drift",
    "stage_12_business",
    "stage_13_calibration",
    "stage_14_retraining",
    "stage_15_combined",
    "stage_16_report",
]
