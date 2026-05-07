from __future__ import annotations
import pandas as pd
from antifraud_pipeline.antifraud_lib import ModelInterpreter

def explain_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list[str], model_name: str='Model', cat_column_names: list[str] | None=None, n_background: int=500, n_explain: int=800) -> ModelInterpreter:
    interp = ModelInterpreter(model=model, X_train=X_train, X_test=X_test, y_test=y_test, feature_names=feature_names, model_name=model_name, cat_column_names=cat_column_names)
    interp.compute_shap_values(n_background=n_background, n_explain=n_explain)
    interp.plot_shap_summary()
    interp.plot_shap_bar()
    interp.plot_native_importance()
    return interp
