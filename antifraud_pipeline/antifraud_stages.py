
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

P = SimpleNamespace()


def resolve_best_ml_model_name_for_combined(trainer) -> str | None:
    """
    Имя ML-модели для комбинированного пайплайна (правила + ML): лучшая по Fβ на test
    среди моделей, у которых есть объект в `trainer.models` с `predict_proba` и скоры в `predictions`.
    Пропускаем ансамбль без единого estimators-объекта, нейросеть (лежит отдельно), Combined.
    """
    exclude = {
        "Autoencoder",
        "Combined_Pipeline",
        "BestEnsemble",
        "NN_Classifier",
    }
    ranked = sorted(
        [
            (name, float(r.get("F_beta") or 0))
            for name, r in getattr(trainer, "results", {}).items()
            if name not in exclude
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    for name, _ in ranked:
        if name not in getattr(trainer, "models", {}):
            continue
        m = trainer.models[name]
        if isinstance(m, dict) or not hasattr(m, "predict_proba"):
            continue
        if getattr(trainer, "predictions", {}).get(name) is None:
            continue
        return name
    return None


def init_notebook(project_root: Path | None = None) -> None:
    """Указать корень проекта (папка с `data/`). По умолчанию — текущая рабочая директория."""
    from . import antifraud_lib as lib

    lib.set_project_root(project_root if project_root is not None else Path.cwd())
    print("PROJECT_ROOT:", lib.PROJECT_ROOT)
    print("DATA_PATH:", lib.DATA_PATH)
    print("OUTPUT_DIR:", lib.OUTPUT_DIR)


def stage_01_load_and_profile() -> None:
    from . import antifraud_lib as lib

    print("=" * 80)
    print("РАЗДЕЛ 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ")
    print("=" * 80)
    P.df = pd.read_csv(lib.DATA_PATH)
    P.df = lib.subset_dataframe_to_original_notebook_columns(P.df)
    df = P.df
    print(f"\nРазмер (только колонки как в оригинальном ноутбуке + метаданные): {df.shape[0]} × {df.shape[1]}")
    print(f"\nfraud_flg:\n{df['fraud_flg'].value_counts()}")
    pos_rate = df["fraud_flg"].mean()
    print(f"\nДоля fraud: {pos_rate:.4%}, соотношение 1:{int((1 - pos_rate) / max(pos_rate, 1e-12))}")
    model_cols = lib.columns_present_model_features(df)
    print(f"\nТипы (только признаки модели, {len(model_cols)} полей):\n{df[model_cols].dtypes.value_counts()}")
    missing = df[model_cols].isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    md = pd.DataFrame({"count": missing, "percent": missing_pct})
    md = md[md["count"] > 0].sort_values("percent", ascending=False)
    print(f"\nПропуски по признакам модели (топ-20):\n{md.head(20)}")
    print(f"\ndescribe (только признаки модели):\n{df[model_cols].describe().round(3).T.head(25)}")


def stage_02_eda_plots() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 2. EDA — графики")
    print("=" * 80)
    lib.plot_fraud_distribution(P.df)
    lib.plot_numeric_distributions(P.df)
    P.top_corr = lib.plot_correlation_with_target(P.df)
    print("\nТоп-15 по |corr| с fraud_flg:")
    print(P.top_corr.head(15).round(4))


def stage_03_feature_engineering() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 3. ИНЖЕНЕРИЯ ПРИЗНАКОВ")
    print("=" * 80)
    P.fe = lib.FeatureEngineer()
    P.df_fe = P.fe.fit_transform(P.df)
    print(f"\nРазмер после FE: {P.df_fe.shape}")


def stage_04_prepare_split() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 4. ПОДГОТОВКА / СПЛИТ")
    print("=" * 80)
    P.preparer = lib.DataPreparer(
        target_col="fraud_flg",
        time_col="order_creation_dttm",
        feature_allowlist=lib.ORIGINAL_NOTEBOOK_MODEL_FEATURES,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = P.preparer.prepare(P.df_fe)
    P.X_train, P.y_train = X_train, y_train
    P.X_val, P.y_val = X_val, y_val
    P.X_test, P.y_test = X_test, y_test
    print(f"\nX_train {X_train.shape}, fraud {y_train.mean():.4%}")
    print(f"X_val   {X_val.shape}, fraud {y_val.mean():.4%}")
    print(f"X_test  {X_test.shape}, fraud {y_test.mean():.4%}")


def stage_05_rules() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 5. ПРАВИЛА L1")
    print("=" * 80)
    print(
        "  Только 3 правила: extreme_velocity_1h, glue_size_spike, glue_ead_spike. "
        "Отбор на train [block]: precision, lift, доля срабатываний, мин. TP; "
        f"доп. Fβ≥{lib.RULE_MIN_FBETA_ON_BLOCK} (β={lib.PRIMARY_FBETA})."
    )
    X_train, y_train = P.X_train, P.y_train
    xs = X_train
    rule_filter = lib.RuleBasedFilter()

    def qn(col: str, p: float) -> float:
        return float(xs[col].quantile(p))

    if "puid_orders_1h_without_refunds" in xs.columns and "order_loan" in xs.columns:
        th_vel = xs["puid_orders_1h_without_refunds"].quantile(0.95)
        rule_filter.add_rule_if_passes_train(
            "extreme_velocity_1h",
            lambda X, tv=th_vel: (X["puid_orders_1h_without_refunds"] > tv) & (X["order_loan"] > 2000),
            "block",
            X_train,
            y_train,
        )
    if "glue_size" in xs.columns:
        gs = qn("glue_size", 0.995)
        rule_filter.add_rule_if_passes_train(
            "glue_size_spike",
            lambda X, a=gs: X["glue_size"] >= a,
            "block",
            X_train,
            y_train,
        )
    if "glue_ead" in xs.columns:
        ge = qn("glue_ead", 0.995)
        rule_filter.add_rule_if_passes_train(
            "glue_ead_spike",
            lambda X, a=ge: X["glue_ead"] >= a,
            "block",
            X_train,
            y_train,
        )

    P.rule_filter = rule_filter
    P.mask_grey_test, P.rule_preds_test = rule_filter.apply(P.X_test)
    rule_filter.print_stats()
    rb = P.rule_preds_test == 1
    rs = P.rule_preds_test == 0
    if rb.sum() > 0:
        print(f"\n  Precision блокировок по правилам: {P.y_test[rb].mean():.4f}")
    if rs.sum() > 0:
        print(f"  Fraud среди «safe» по правилам: {P.y_test[rs].mean():.4%}")
    print(f"  Доля в серой зоне (ML): {P.mask_grey_test.mean():.2%}")


def stage_06_train_models() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 6. ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    P.trainer = lib.ModelTrainer(
        P.X_train,
        P.y_train,
        P.X_val,
        P.y_val,
        P.X_test,
        P.y_test,
        feature_columns=P.preparer.feature_columns,
        cat_column_names=P.preparer.cat_columns,
    )
    P.trainer.train_all()


def stage_08_neural() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 8. НЕЙРОСЕТИ")
    print("=" * 80)
    P.nn_model = None
    P.nn_scaler = None
    if not getattr(lib, "TORCH_AVAILABLE", False):
        print("  PyTorch недоступен — этап пропущен.")
        return
    import torch

    ae_model, ae_scaler, ae_scores, ae_auc, ae_pr_auc = lib.train_autoencoder(
        P.X_train, P.y_train, P.X_test, P.y_test, epochs=30
    )
    P.trainer.results["Autoencoder"] = {
        "ROC-AUC": ae_auc,
        "PR-AUC": ae_pr_auc,
        "F1": None,
        "F2": None,
        "Precision": None,
        "Recall": None,
    }
    P.trainer.predictions["Autoencoder"] = ae_scores

    P.nn_model, P.nn_scaler, nn_probs, _, _ = lib.train_nn_classifier(
        P.X_train, P.y_train, P.X_val, P.y_val, P.X_test, P.y_test, epochs=50
    )
    with torch.no_grad():
        val_nn = torch.sigmoid(
            P.nn_model(torch.FloatTensor(P.nn_scaler.transform(P.X_val.values))).squeeze()
        ).numpy()
    best_th_nn, _ = P.trainer.find_optimal_threshold(P.y_val, val_nn)
    nn_metrics = P.trainer.evaluate_model("NN_Classifier", P.y_test, nn_probs, threshold=best_th_nn)
    P.trainer.results["NN_Classifier"] = nn_metrics
    P.trainer.predictions["NN_Classifier"] = nn_probs
    print(
        f"  NN Test Fβ (β={lib.PRIMARY_FBETA}): {nn_metrics['F_beta']:.4f}"
    )


def stage_09_ensemble() -> None:
    from sklearn.metrics import average_precision_score

    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 9. АНСАМБЛИ")
    print("=" * 80)
    trainer = P.trainer
    ensemble_models = {}
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        if name in trainer.predictions:
            ensemble_models[name] = trainer.predictions[name]
    if getattr(lib, "TORCH_AVAILABLE", False) and "NN_Classifier" in trainer.predictions:
        ensemble_models["NN_Classifier"] = trainer.predictions["NN_Classifier"]

    P.stacker = None
    if len(ensemble_models) < 2:
        print("  Нужно ≥2 моделей для ансамбля.")
        return

    import torch

    stacker = lib.EnsembleStacker(ensemble_models, P.y_test)
    P.stacker = stacker
    print("--- Ансамблирование ---")
    avg_pred, _, avg_pr = stacker.simple_average()
    wavg_pred, _, wavg_pr = stacker.weighted_average()
    rank_pred, _, rank_pr = stacker.rank_average()
    from scipy.stats import rankdata

    val_meta = {}
    for name in ensemble_models:
        if name == "XGBoost":
            val_meta[name] = trainer.models["XGBoost"].predict_proba(P.X_val)[:, 1]
        elif name == "LightGBM":
            val_meta[name] = trainer.models["LightGBM"].predict_proba(P.X_val)[:, 1]
        elif name == "CatBoost":
            val_meta[name] = trainer.models["CatBoost"].predict_proba(P.X_val)[:, 1]
        elif name == "NN_Classifier" and P.nn_model is not None:
            P.nn_model.eval()
            with torch.no_grad():
                val_meta[name] = torch.sigmoid(
                    P.nn_model(torch.FloatTensor(P.nn_scaler.transform(P.X_val.values))).squeeze()
                ).numpy()
    names = list(ensemble_models.keys())
    X_val_meta = np.column_stack([val_meta[n] for n in names])
    X_test_meta = np.column_stack([ensemble_models[n] for n in names])
    stacked_pred, stacked_auc, stacked_pr = stacker.stacking_with_logreg(X_val_meta, P.y_val, X_test_meta, P.y_test)

    n_val, n_test = len(P.y_val), len(P.y_test)
    val_simple = X_val_meta.mean(axis=1)
    w_ap_val = np.array([average_precision_score(P.y_val, val_meta[n]) for n in names])
    w_ap_val = w_ap_val / w_ap_val.sum()
    val_wavg = X_val_meta @ w_ap_val
    wavg_pred_val = X_test_meta @ w_ap_val
    val_rank_avg = np.mean(
        [rankdata(X_val_meta[:, i]) / n_val for i in range(X_val_meta.shape[1])],
        axis=0,
    )
    rank_pred_val = np.mean(
        [rankdata(ensemble_models[n]) / n_test for n in names],
        axis=0,
    )
    val_stack = stacker.stacking_model.predict_proba(X_val_meta)[:, 1]

    fb_s = lib.fbeta_test_with_val_threshold(trainer, P.y_val, val_simple, P.y_test, avg_pred)
    fb_w = lib.fbeta_test_with_val_threshold(trainer, P.y_val, val_wavg, P.y_test, wavg_pred_val)
    fb_r = lib.fbeta_test_with_val_threshold(trainer, P.y_val, val_rank_avg, P.y_test, rank_pred_val)
    fb_st = lib.fbeta_test_with_val_threshold(trainer, P.y_val, val_stack, P.y_test, stacked_pred)

    candidates = [
        ("SimpleAvg", fb_s, avg_pred, val_simple),
        ("WeightedAvg", fb_w, wavg_pred_val, val_wavg),
        ("RankAvg", fb_r, rank_pred_val, val_rank_avg),
        ("Stacking", fb_st, stacked_pred, val_stack),
    ]
    best_name = max(candidates, key=lambda x: x[1])[0]
    print(
        f"\n  Fβ (β={lib.PRIMARY_FBETA}) на test при пороге с val: "
        f"Simple={fb_s:.4f} Weighted={fb_w:.4f} Rank={fb_r:.4f} Stacking={fb_st:.4f}"
    )
    print(f"  Лучший ансамбль (по Fβ): {best_name}")
    pred_map = {c[0]: (c[2], c[3]) for c in candidates}
    best_pred, val_for_th = pred_map[best_name]
    best_th_ens, _ = trainer.find_optimal_threshold(P.y_val, val_for_th)
    ens_m = trainer.evaluate_model("BestEnsemble", P.y_test, best_pred, threshold=best_th_ens)
    trainer.results["BestEnsemble"] = ens_m
    trainer.predictions["BestEnsemble"] = best_pred


def stage_10_shap() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 10. SHAP")
    print("=" * 80)
    trainer = P.trainer
    best_boosting = max(
        [
            (n, trainer.results[n].get("F_beta") or 0)
            for n in ["XGBoost", "LightGBM", "CatBoost"]
            if n in trainer.results
        ],
        key=lambda x: x[1],
    )
    P.best_model_name = best_boosting[0]
    P.best_model = trainer.models[P.best_model_name]
    print(
        f"Модель для SHAP: {P.best_model_name} (Fβ, β={lib.PRIMARY_FBETA}={best_boosting[1]:.4f})"
    )
    P.interpreter = lib.ModelInterpreter(
        model=P.best_model,
        X_train=P.X_train,
        X_test=P.X_test,
        y_test=P.y_test,
        feature_names=P.preparer.feature_columns,
        model_name=P.best_model_name,
        cat_column_names=P.preparer.cat_columns,
    )
    P.interpreter.compute_shap_values()
    P.interpreter.plot_shap_summary()
    P.interpreter.plot_shap_bar()
    P.interpreter.plot_native_importance()


def stage_11_drift() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 11. МОНИТОРИНГ ДРЕЙФА")
    print("=" * 80)
    trainer = P.trainer
    name = P.best_model_name
    val_scores = trainer.models[name].predict_proba(P.X_val)[:, 1]
    P.monitor = lib.DriftMonitor(
        reference_data=P.X_val,
        reference_scores=val_scores,
        reference_labels=P.y_val.values,
        feature_names=P.preparer.feature_columns,
    )
    test_scores = trainer.predictions[name]
    th = trainer.results[name]["Threshold"]
    P.drift_report = P.monitor.full_monitoring_check(
        current_data=P.X_test,
        current_scores=test_scores,
        current_labels=P.y_test.values,
        threshold=th,
    )
    print(f"Дрейф признаков: {P.drift_report['feature_drift']['n_drifted']}")
    print(f"Рекомендация: {P.drift_report['recommendation']}")
    P.monitor.plot_drift_report(P.drift_report, current_scores=test_scores)


def stage_15_combined() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 15. ПРАВИЛА + ML")
    print("=" * 80)
    trainer = P.trainer
    ml_name = resolve_best_ml_model_name_for_combined(trainer)
    if ml_name is None:
        ml_name = getattr(P, "best_model_name", None) or "XGBoost"
        print(
            f"  Не удалось выбрать модель по Fβ среди sklearn-моделей — fallback для Combined: {ml_name}"
        )
    else:
        fb = trainer.results[ml_name].get("F_beta")
        print(
            f"  Combined: ML = {ml_name} (Fβ={fb:.4f} — лучшая по Fβ на test среди доступных ML-моделей; "
            "не SHAP-only)"
        )
    if ml_name not in trainer.models:
        print(f"  Ошибка: модель '{ml_name}' нет в trainer.models — Combined пропущен.")
        return
    P.combined_ml_model_name = ml_name
    th = trainer.results[ml_name]["Threshold"]
    pipe = lib.CombinedAntiFraudPipeline(P.rule_filter, trainer.models[ml_name], threshold=th)
    P.combined_results = pipe.evaluate(P.X_test, P.y_test)
    P.trainer.results["Combined_Pipeline"] = P.combined_results


def stage_07_compare_plots() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 7. СРАВНЕНИЕ")
    print("=" * 80)
    P.metrics_df = lib.plot_comprehensive_comparison(P.trainer.results, P.trainer.predictions, P.y_test)


def stage_12_business() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 12. БИЗНЕС-МЕТРИКИ")
    print("=" * 80)
    P.biz_analyzer = lib.BusinessMetricsAnalyzer()
    preds = P.trainer.predictions[P.best_model_name]
    th = P.trainer.results[P.best_model_name]["Threshold"]
    y_pred = (preds >= th).astype(int)
    P.biz_report = P.biz_analyzer.compute_business_impact(P.y_test, y_pred)
    for k, v in P.biz_report.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")
    P.threshold_analysis = P.biz_analyzer.threshold_sensitivity_analysis(P.y_test, preds)
    P.biz_analyzer.plot_threshold_analysis(P.threshold_analysis)


def stage_13_calibration() -> None:
    from . import antifraud_lib as lib

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 13. КАЛИБРОВКА")
    print("=" * 80)
    P.calibrated_model, P.calibrated_probs = lib.calibrate_and_evaluate(
        P.trainer.models[P.best_model_name], P.X_val, P.y_val, P.X_test, P.y_test, model_name=P.best_model_name
    )


def stage_14_retraining() -> None:
    from . import antifraud_lib as lib
    import xgboost as xgb

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 14. ДООБУЧЕНИЕ (симуляция)")
    print("=" * 80)
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": P.trainer.scale_pos_weight,
        "verbosity": 0,
    }
    # Порог был 0.005 под PR-AUC; для Fβ на test приросты часто 0.001–0.003 — иначе челленджер почти никогда не проходит.
    P.retrain_pipeline = lib.RetrainingPipeline(xgb.XGBClassifier, params, min_improvement=0.001)
    P.retrain_pipeline.set_champion(
        P.trainer.models[P.best_model_name],
        {
            "F_beta": P.trainer.results[P.best_model_name].get("F_beta"),
            "PR-AUC": P.trainer.results[P.best_model_name]["PR-AUC"],
        },
    )
    ch = xgb.XGBClassifier(**params)
    ch.fit(P.X_train, P.y_train, eval_set=[(P.X_val, P.y_val)], verbose=False)
    replaced = P.retrain_pipeline.champion_challenger_test(
        ch, P.X_test, P.y_test, trainer=P.trainer, y_val=P.y_val, X_val=P.X_val
    )
    if replaced:
        new_model = P.retrain_pipeline.current_champion
        val_scores_new = new_model.predict_proba(P.X_val)[:, 1]
        test_scores_new = new_model.predict_proba(P.X_test)[:, 1]
        th_new, _ = P.trainer.find_optimal_threshold(P.y_val, val_scores_new)
        P.trainer.models["XGBoost"] = new_model
        P.trainer.predictions["XGBoost"] = test_scores_new
        P.trainer.results["XGBoost"] = P.trainer.evaluate_model(
            "XGBoost", P.y_test, test_scores_new, threshold=th_new
        )
        print(
            "  Дообученный XGBoost записан в P.trainer (models / predictions / results['XGBoost'])."
        )
        if P.best_model_name == "XGBoost":
            print(
                "  best_model_name — XGBoost: дальнейшие шаги с лучшей моделью видят обновлённые веса."
            )
        if getattr(P, "monitor", None) is not None:
            P.monitor.update_baseline(P.X_val.copy(), val_scores_new, P.y_val.values)
            P.drift_report = P.monitor.full_monitoring_check(
                current_data=P.X_test,
                current_scores=test_scores_new,
                current_labels=P.y_test.values,
                threshold=th_new,
            )
            print(
                "  Бейзлайн мониторинга дрейфа обновлён (reference = val под новым чемпионом)."
            )
            print(f"  Повторная проверка (test vs новый baseline): {P.drift_report['recommendation']}")
            P.monitor.plot_drift_report(P.drift_report, current_scores=test_scores_new)


def _final_winner_y_pred(P, trainer, lib, winner_name: str):
    """Бинарные предсказания на test для модели-победителя по Fβ (в т.ч. Combined_Pipeline)."""
    if winner_name == "Combined_Pipeline":
        rf = getattr(P, "rule_filter", None)
        ml = getattr(P, "combined_ml_model_name", None) or getattr(P, "best_model_name", None)
        if rf is None or not ml or ml not in trainer.models:
            return None
        th_ml = trainer.results[ml]["Threshold"]
        pipe = lib.CombinedAntiFraudPipeline(rf, trainer.models[ml], threshold=th_ml)
        y_pred, _, _ = pipe.predict(P.X_test)
        return np.asarray(y_pred, dtype=int)
    if winner_name not in getattr(trainer, "predictions", {}) or trainer.predictions[winner_name] is None:
        return None
    preds = np.asarray(trainer.predictions[winner_name], dtype=float).ravel()
    r = trainer.results.get(winner_name, {})
    th = r.get("Threshold")
    if th is None:
        th = 0.5
    return (preds >= float(th)).astype(int)


def stage_16_report() -> None:
    from . import antifraud_lib as lib

    def _print_metrics_block(title: str, name: str, r: dict) -> None:
        if not r:
            print(f"  {title}: {name} — метрики недоступны")
            return
        print(f"  {title}: {name}")
        fb = r.get("F_beta")
        fb_s = f"{fb:.4f}" if fb is not None else "n/a"
        line1 = (
            f"    PR-AUC={r.get('PR-AUC', 0):.4f}  ROC-AUC={r.get('ROC-AUC', 0):.4f}  "
            f"Fβ={fb_s}  F1={r.get('F1', 0) or 0:.4f}  "
            f"Precision={r.get('Precision', 0):.4f}  Recall={r.get('Recall', 0):.4f}"
        )
        print(line1)
        th = r.get("Threshold")
        fpr = r.get("FPR")
        ll = r.get("Log Loss")
        extra = []
        if th is not None:
            extra.append(f"порог={th:.4f}")
        if fpr is not None:
            extra.append(f"FPR={fpr:.4f}")
        if ll is not None:
            extra.append(f"LogLoss={ll:.4f}")
        if extra:
            print(f"    {'  '.join(extra)}")
        tp, fp, tn, fn = r.get("TP"), r.get("FP"), r.get("TN"), r.get("FN")
        if all(x is not None for x in (tp, fp, tn, fn)):
            print(f"    матрица: TP={tp} FP={fp} TN={tn} FN={fn}")

    print("\n" + "=" * 80)
    print("РАЗДЕЛ 16. ОТЧЁТ")
    print("=" * 80)
    trainer = getattr(P, "trainer", None)
    if trainer is None or not getattr(trainer, "results", None):
        print("  Нет результатов обучения (запустите этап 6+).")
        print(f"\nАртефакты: {lib.OUTPUT_DIR}")
        return

    sm = sorted(
        [(n, r.get("F_beta") or 0) for n, r in trainer.results.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\n  --- Топ моделей по Fβ на test (β={lib.PRIMARY_FBETA}, порог с val) ---")
    for i, (n, fb) in enumerate(sm[:8], 1):
        print(f"  {i}. {n}: {fb:.4f}")

    if sm:
        winner_name, _winner_fb = sm[0]
        winner_r = trainer.results.get(winner_name, {})
        print("\n  --- Итоговая модель (лучшая по Fβ на test) ---")
        _print_metrics_block("Модель", winner_name, winner_r)

    best_shap = getattr(P, "best_model_name", None)
    if best_shap and best_shap in trainer.results and best_shap != (sm[0][0] if sm else None):
        print("\n  --- Модель для SHAP / калибровки / бизнес-метрик (разделы 10–13) ---")
        _print_metrics_block("Модель", best_shap, trainer.results[best_shap])

    comb = getattr(P, "combined_results", None) or trainer.results.get("Combined_Pipeline")
    if comb:
        print("\n  --- Пайплайн «правила + ML» (Combined) ---")
        cm = getattr(P, "combined_ml_model_name", None)
        if cm:
            print(f"  (ML-часть: {cm} — выбрана по Fβ в разделе 15)")
        _print_metrics_block("Система", "Combined_Pipeline", comb)

    _comb = getattr(P, "combined_results", None) or {}
    print(f"\n  Combined ROC-AUC (как в разделе 15): {_comb.get('ROC-AUC', 0):.4f}")
    if getattr(P, "drift_report", None):
        print(f"  Мониторинг дрейфа: {P.drift_report.get('recommendation')}")

    if sm:
        final_name = sm[0][0]
        P.final_best_model_name = final_name
        y_pred_final = _final_winner_y_pred(P, trainer, lib, final_name)
        print("\n  --- Итоговые бизнес-метрики (лучшая по Fβ на test после всех этапов) ---")
        print(f"  Модель: {final_name}")
        if y_pred_final is None:
            print("  Не удалось восстановить предсказания (нет rule_filter или predictions).")
            P.final_biz_report = None
        else:
            biz = lib.BusinessMetricsAnalyzer()
            P.final_biz_report = biz.compute_business_impact(P.y_test, y_pred_final)
            for k, v in P.final_biz_report.items():
                print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")

    print(f"\n  Артефакты (графики, CSV): {lib.OUTPUT_DIR}")


def run_all_stages() -> None:
    """Как `antifraud_pipeline_full.py` — весь пайплайн подряд."""
    init_notebook()
    stage_01_load_and_profile()
    stage_02_eda_plots()
    stage_03_feature_engineering()
    stage_04_prepare_split()
    stage_05_rules()
    stage_06_train_models()
    stage_07_compare_plots()
    stage_08_neural()
    stage_09_ensemble()
    stage_10_shap()
    stage_11_drift()
    stage_12_business()
    stage_13_calibration()
    stage_14_retraining()
    stage_15_combined()
    stage_16_report()
