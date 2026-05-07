from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import os
import numpy as np
import pandas as pd
P = SimpleNamespace()

def _gnn_dir() -> Path:
    subdir = os.environ.get('GNN_SUBDIR', '').strip()
    base = Path('outputs/gnn')
    return base / subdir if subdir else base

def _display_saved_figure_in_notebook(path: str | Path) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is None:
            return
        p = Path(path)
        if p.is_file():
            display(Image(filename=str(p)))
    except Exception:
        pass

def _align_gnn_probs_to_tabular_order_ids(tab_order_ids: np.ndarray, gnn_order_ids: np.ndarray, gnn_probs_all: np.ndarray, fill_value: float=0.5) -> np.ndarray | None:
    g_oid = np.asarray(gnn_order_ids).ravel()
    pr = np.asarray(gnn_probs_all, dtype=float).ravel()
    if g_oid.shape[0] != pr.shape[0]:
        return None
    lookup: dict = {}
    for i, oid in enumerate(g_oid):
        lookup[oid] = pr[i]
    tab = np.asarray(tab_order_ids).ravel()
    out = np.empty(len(tab), dtype=float)
    for i, oid in enumerate(tab):
        out[i] = float(lookup.get(oid, fill_value))
    return out

def _register_graphsage_in_trainer(P) -> None:
    from pathlib import Path
    from antifraud_components.models.gnn import gnn_predict
    if getattr(P, 'gnn_model', None) is None or getattr(P, 'trainer', None) is None:
        return
    df_fe = getattr(P, 'df_fe', None)
    if df_fe is None or 'order_id' not in df_fe.columns:
        return
    test_oid = getattr(P, 'test_order_ids', None)
    val_oid = getattr(P, 'val_order_ids', None)
    if test_oid is None or val_oid is None:
        return
    if not hasattr(P, 'gnn_probs_all') or P.gnn_probs_all is None:
        if getattr(P, 'gnn_data', None) is not None:
            P.gnn_probs_all = gnn_predict(P.gnn_model, P.gnn_data)
        else:
            _sp = _gnn_dir() / 'gnn_scores.parquet'
            if not _sp.exists():
                print('  GraphSAGE: gnn_scores.parquet не найден — регистрация пропущена.')
                return
            import pandas as _pd_r
            _df_s = _pd_r.read_parquet(_sp)
            _oid2p = dict(zip(_df_s['order_id'].values, _df_s['gnn_fraud_prob'].values))
            _gnn_oids_raw = getattr(P, 'gnn_order_ids', None)
            _gnn_oids = np.asarray(_gnn_oids_raw) if _gnn_oids_raw is not None and len(_gnn_oids_raw) > 0 else np.array([])
            if len(_gnn_oids) == 0:
                _op = _gnn_dir() / 'gnn_order_ids.npy'
                if _op.exists():
                    _gnn_oids = np.load(_op, allow_pickle=True)
                    P.gnn_order_ids = _gnn_oids
            P.gnn_probs_all = np.array([_oid2p.get(oid, 0.0) for oid in _gnn_oids], dtype=float)
    n_p = len(P.gnn_probs_all)
    gnn_oid: np.ndarray | None = None
    for src in (getattr(P, 'gnn_order_ids', None), getattr(P.gnn_data, 'order_ids_np', None) if getattr(P, 'gnn_data', None) is not None else None):
        if src is None:
            continue
        a = np.asarray(src)
        if a.shape[0] == n_p:
            gnn_oid = a
            break
    if gnn_oid is None:
        npy_path = _gnn_dir() / 'gnn_order_ids.npy'
        if npy_path.is_file():
            a = np.load(npy_path, allow_pickle=True)
            if a.shape[0] == n_p:
                gnn_oid = a
    if gnn_oid is None:
        print('  GraphSAGE в leaderboard: не сопоставлены order_id узлов графа — пропуск.')
        return

    def _tabular_rows_without_gnn_node(tab_order_ids: np.ndarray, gnn_order_ids: np.ndarray) -> tuple[int, int]:
        gset = set(np.asarray(gnn_order_ids).ravel().tolist())
        tab = np.asarray(tab_order_ids).ravel()
        n_miss = int(sum((1 for o in tab if o not in gset)))
        return (n_miss, len(tab))
    y_tr = getattr(P, 'y_train', None)
    fraud_prior = float(np.asarray(y_tr, dtype=float).mean()) if y_tr is not None else 0.08
    te = _align_gnn_probs_to_tabular_order_ids(test_oid, gnn_oid, P.gnn_probs_all, fill_value=fraud_prior)
    va = _align_gnn_probs_to_tabular_order_ids(val_oid, gnn_oid, P.gnn_probs_all, fill_value=fraud_prior)
    if te is None or va is None or len(te) != len(P.y_test) or (len(va) != len(P.y_val)):
        print('  GraphSAGE в leaderboard: несовпадение длин val/test — пропуск.')
        return
    mv_te, nt_te = _tabular_rows_without_gnn_node(test_oid, gnn_oid)
    mv_va, nt_va = _tabular_rows_without_gnn_node(val_oid, gnn_oid)
    if mv_te > 0 or mv_va > 0:
        print(f'  GraphSAGE → табличный сплит: заказов без узла в графе (скор подставлен {fraud_prior:.3f} = fraud prior): val {mv_va:,}/{nt_va:,} ({100 * mv_va / max(nt_va, 1):.1f}%), test {mv_te:,}/{nt_te:,} ({100 * mv_te / max(nt_te, 1):.1f}%). PR-AUC/Fβ ниже §11.1 из-за неполного покрытия графа.')
    trainer = P.trainer
    th, _ = trainer.find_optimal_threshold(P.y_val, va)
    metrics = trainer.evaluate_model('GraphSAGE', P.y_test, te, threshold=th)
    trainer.results['GraphSAGE'] = metrics
    trainer.predictions['GraphSAGE'] = te
    P.gnn_val_probs_tabular = va
    P.gnn_test_probs_tabular = te
    print(f"  GraphSAGE в табличном leaderboard (val/test как у GBM): Fβ={metrics.get('F_beta', 0) or 0:.4f}  PR-AUC={metrics.get('PR-AUC', 0):.4f}")

def resolve_best_ml_model_name_for_combined(trainer) -> str | None:
    exclude = {'Autoencoder', 'Combined_Pipeline', 'BestEnsemble', 'BestEnsemble_no_NN', 'NN_Classifier', 'GraphSAGE'}
    ranked = sorted([(name, float(r.get('F_beta') or 0)) for name, r in getattr(trainer, 'results', {}).items() if name not in exclude], key=lambda x: x[1], reverse=True)
    for name, _ in ranked:
        if name not in getattr(trainer, 'models', {}):
            continue
        m = trainer.models[name]
        if isinstance(m, dict) or not hasattr(m, 'predict_proba'):
            continue
        if getattr(trainer, 'predictions', {}).get(name) is None:
            continue
        return name
    return None

def init_notebook(project_root: Path | None=None) -> None:
    from . import antifraud_lib as lib
    lib.set_project_root(project_root if project_root is not None else Path.cwd())
    print('PROJECT_ROOT:', lib.PROJECT_ROOT)
    print('DATA_PATH:', lib.DATA_PATH)
    print('OUTPUT_DIR:', lib.OUTPUT_DIR)

def stage_01_load_and_profile() -> None:
    from . import antifraud_lib as lib
    if lib.DATA_PATH.suffix == '.parquet':
        P.df = pd.read_parquet(lib.DATA_PATH)
    else:
        P.df = pd.read_csv(lib.DATA_PATH)
    if 'fraud_flg' in P.df.columns and 'resolution_fraud' not in P.df.columns:
        P.df = P.df.rename(columns={'fraud_flg': 'resolution_fraud'})
    P.df = lib.subset_dataframe_to_original_notebook_columns(P.df)
    df = P.df
    print(f'\nРазмер (только колонки как в оригинальном ноутбуке + метаданные): {df.shape[0]} × {df.shape[1]}')
    print(f"\nresolution_fraud:\n{df['resolution_fraud'].value_counts()}")
    pos_rate = df['resolution_fraud'].mean()
    print(f'\nДоля fraud: {pos_rate:.4%}, соотношение 1:{int((1 - pos_rate) / max(pos_rate, 1e-12))}')
    model_cols = lib.columns_present_model_features(df)
    print(f'\nТипы (только признаки модели, {len(model_cols)} полей):\n{df[model_cols].dtypes.value_counts()}')
    missing = df[model_cols].isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    md = pd.DataFrame({'count': missing, 'percent': missing_pct})
    md = md[md['count'] > 0].sort_values('percent', ascending=False)
    print(f'\nПропуски по признакам модели (топ-20):\n{md.head(20)}')
    print(f'\ndescribe (только признаки модели):\n{df[model_cols].describe().round(3).T.head(25)}')

def stage_02_eda_plots() -> None:
    from . import antifraud_lib as lib
    lib.plot_fraud_distribution(P.df)
    lib.plot_resolution_class_distribution(P.df)
    lib.plot_numeric_distributions(P.df)
    P.top_corr = lib.plot_correlation_with_target(P.df)
    lib.plot_eda_risk_slices(P.df)
    lib.plot_eda_correlation_heatmap(P.df)
    print('\nТоп-15 по |corr| с resolution_fraud:')
    print(P.top_corr.head(15).round(4))

def stage_03_feature_engineering() -> None:
    from . import antifraud_lib as lib
    P.fe = lib.FeatureEngineer()
    P.df_fe = P.fe.fit_transform(P.df)
    print(f'\nРазмер после FE: {P.df_fe.shape}')

def stage_04_prepare_split() -> None:
    from . import antifraud_lib as lib
    P.preparer = lib.DataPreparer(target_col='resolution_fraud', time_col='order_creation_dttm', feature_allowlist=lib.ORIGINAL_NOTEBOOK_MODEL_FEATURES)
    X_train, y_train, X_val, y_val, X_test, y_test = P.preparer.prepare(P.df_fe)
    P.X_train, P.y_train = (X_train, y_train)
    P.X_val, P.y_val = (X_val, y_val)
    P.X_test, P.y_test = (X_test, y_test)
    P.train_order_ids = getattr(P.preparer, 'train_order_ids', None)
    P.val_order_ids = getattr(P.preparer, 'val_order_ids', None)
    P.test_order_ids = getattr(P.preparer, 'test_order_ids', None)
    print(f'\nX_train {X_train.shape}, fraud {y_train.mean():.4%}')
    print(f'X_val   {X_val.shape}, fraud {y_val.mean():.4%}')
    print(f'X_test  {X_test.shape}, fraud {y_test.mean():.4%}')

def stage_05_rules() -> None:
    from . import antifraud_lib as lib
    print(f'  Только 3 правила: extreme_velocity_1h, glue_size_spike, glue_ead_spike. Отбор на train [block]: precision, lift, доля срабатываний, мин. TP; доп. Fβ≥{lib.RULE_MIN_FBETA_ON_BLOCK} (β={lib.PRIMARY_FBETA}).')
    X_train, y_train = (P.X_train, P.y_train)
    xs = X_train
    rule_filter = lib.RuleBasedFilter()

    def qn(col: str, p: float) -> float:
        return float(xs[col].quantile(p))
    if 'puid_orders_1h_without_refunds' in xs.columns and 'order_loan' in xs.columns:
        th_vel = xs['puid_orders_1h_without_refunds'].quantile(0.95)
        rule_filter.add_rule_if_passes_train('extreme_velocity_1h', lambda X, tv=th_vel: (X['puid_orders_1h_without_refunds'] > tv) & (X['order_loan'] > 2000), 'block', X_train, y_train)
    if 'glue_size' in xs.columns:
        gs = qn('glue_size', 0.995)
        rule_filter.add_rule_if_passes_train('glue_size_spike', lambda X, a=gs: X['glue_size'] >= a, 'block', X_train, y_train)
    if 'glue_ead' in xs.columns:
        ge = qn('glue_ead', 0.995)
        rule_filter.add_rule_if_passes_train('glue_ead_spike', lambda X, a=ge: X['glue_ead'] >= a, 'block', X_train, y_train)
    P.rule_filter = rule_filter
    P.mask_grey_test, P.rule_preds_test = rule_filter.apply(P.X_test)
    rule_filter.print_stats()
    rb = P.rule_preds_test == 1
    rs = P.rule_preds_test == 0
    if rb.sum() > 0:
        print(f'\n  Precision блокировок по правилам: {P.y_test[rb].mean():.4f}')
    if rs.sum() > 0:
        print(f'  Fraud среди «safe» по правилам: {P.y_test[rs].mean():.4%}')
    print(f'  Доля в серой зоне (ML): {P.mask_grey_test.mean():.2%}')

def stage_07_train_models() -> None:
    from . import antifraud_lib as lib
    P.trainer = lib.ModelTrainer(P.X_train, P.y_train, P.X_val, P.y_val, P.X_test, P.y_test, feature_columns=P.preparer.feature_columns, cat_column_names=P.preparer.cat_columns)
    P.trainer.train_all()

def stage_09_neural() -> None:
    from . import antifraud_lib as lib
    P.nn_model = None
    P.nn_scaler = None
    if not getattr(lib, 'TORCH_AVAILABLE', False):
        print('  PyTorch недоступен — этап пропущен.')
        return
    print('  Импорт torch (при первом запуске в сессии может занять 1–3 минуты)...', flush=True)
    import torch
    dev = lib.resolve_nn_torch_device()
    print(f'  PyTorch {torch.__version__}, потоков torch: {torch.get_num_threads()}, нейросети→{dev} (на Mac по умолчанию cpu вместо mps; для MPS: env ANTIFRAUD_TORCH_DEVICE=mps)', flush=True)
    ae_model, ae_scaler, ae_scores, ae_auc, ae_pr_auc = lib.train_autoencoder(P.X_train, P.y_train, P.X_test, P.y_test, epochs=30)
    P.trainer.results['Autoencoder'] = {'ROC-AUC': ae_auc, 'PR-AUC': ae_pr_auc, 'F1': None, 'F2': None, 'Precision': None, 'Recall': None}
    P.trainer.predictions['Autoencoder'] = ae_scores
    P.nn_model, P.nn_scaler, nn_probs, _, _ = lib.train_nn_classifier(P.X_train, P.y_train, P.X_val, P.y_val, P.X_test, P.y_test, epochs=50)
    with torch.no_grad():
        val_nn = torch.sigmoid(P.nn_model(torch.FloatTensor(P.nn_scaler.transform(P.X_val.values))).squeeze()).cpu().numpy()
    best_th_nn, _ = P.trainer.find_optimal_threshold(P.y_val, val_nn)
    nn_metrics = P.trainer.evaluate_model('NN_Classifier', P.y_test, nn_probs, threshold=best_th_nn)
    P.trainer.results['NN_Classifier'] = nn_metrics
    P.trainer.predictions['NN_Classifier'] = nn_probs
    print(f"  NN Test Fβ (β={lib.PRIMARY_FBETA}): {nn_metrics['F_beta']:.4f}")

def stage_10_ensemble() -> None:
    from sklearn.metrics import average_precision_score
    from . import antifraud_lib as lib
    trainer = P.trainer
    ensemble_models = {}
    for name in ['XGBoost', 'LightGBM', 'CatBoost']:
        if name in trainer.predictions:
            ensemble_models[name] = trainer.predictions[name]
    nn_ready = getattr(lib, 'TORCH_AVAILABLE', False) and 'NN_Classifier' in trainer.predictions and (getattr(P, 'nn_model', None) is not None) and (getattr(P, 'nn_scaler', None) is not None)
    if nn_ready:
        ensemble_models['NN_Classifier'] = trainer.predictions['NN_Classifier']
        print('  NN_Classifier добавлен в ансамбль.')
    elif 'NN_Classifier' in trainer.predictions:
        print('  NN_Classifier есть в predictions, но P.nn_model/nn_scaler недоступны — исключён из ансамбля.')
    gva = getattr(P, 'gnn_val_probs_tabular', None)
    if 'GraphSAGE' in trainer.predictions and trainer.predictions['GraphSAGE'] is not None and (gva is not None) and (len(gva) == len(P.y_val)):
        ensemble_models['GraphSAGE'] = trainer.predictions['GraphSAGE']
        print('  GraphSAGE добавлен в ансамбль (avg / rank / stack вместе с GBM/NN).')
    elif 'GraphSAGE' in trainer.predictions and trainer.predictions['GraphSAGE'] is not None:
        print('  GraphSAGE в predictions, но нет gnn_val_probs_tabular — не включаем в ансамбль.')
    P.stacker = None
    if len(ensemble_models) < 2:
        print('  Нужно ≥2 моделей для ансамбля.')
        return
    import torch
    from scipy.stats import rankdata
    had_nn = 'NN_Classifier' in ensemble_models

    def _run_ensemble_pipeline(em: dict, results_key: str, *, verbose: bool) -> tuple[str, dict] | None:
        if len(em) < 2:
            return None
        stacker_local = lib.EnsembleStacker(em, P.y_test)
        if results_key == 'BestEnsemble':
            P.stacker = stacker_local
        silent = not verbose
        if verbose:
            print('--- Ансамблирование ---')
        avg_te, _, _ = stacker_local.simple_average(silent=silent)
        stacker_local.weighted_average(silent=silent)
        stacker_local.rank_average(silent=silent)
        val_meta_l: dict = {}
        for name in em:
            if name == 'XGBoost':
                val_meta_l[name] = trainer.models['XGBoost'].predict_proba(P.X_val)[:, 1]
            elif name == 'LightGBM':
                val_meta_l[name] = trainer.models['LightGBM'].predict_proba(P.X_val)[:, 1]
            elif name == 'CatBoost':
                val_meta_l[name] = trainer.models['CatBoost'].predict_proba(P.X_val)[:, 1]
            elif name == 'NN_Classifier':
                P.nn_model.eval()
                with torch.no_grad():
                    val_meta_l[name] = torch.sigmoid(P.nn_model(torch.FloatTensor(P.nn_scaler.transform(P.X_val.values))).squeeze()).numpy()
            elif name == 'GraphSAGE':
                val_meta_l[name] = np.asarray(gva, dtype=float)
        names_l = list(em.keys())
        X_val_l = np.column_stack([val_meta_l[n] for n in names_l])
        X_te_l = np.column_stack([em[n] for n in names_l])
        stacked_te, _, _ = stacker_local.stacking_with_logreg(X_val_l, P.y_val, X_te_l, P.y_test, silent=silent)
        n_val, n_test = (len(P.y_val), len(P.y_test))
        val_simple_l = X_val_l.mean(axis=1)
        w_ap_l = np.array([average_precision_score(P.y_val, val_meta_l[n]) for n in names_l])
        w_ap_l = w_ap_l / w_ap_l.sum()
        val_wavg_l = X_val_l @ w_ap_l
        wavg_te_l = X_te_l @ w_ap_l
        val_rank_l = np.mean([rankdata(X_val_l[:, i]) / n_val for i in range(X_val_l.shape[1])], axis=0)
        rank_te_l = np.mean([rankdata(em[n]) / n_test for n in names_l], axis=0)
        val_stack_l = stacker_local.stacking_model.predict_proba(X_val_l)[:, 1]
        fb_s = lib.fbeta_test_with_val_threshold(trainer, P.y_val, val_simple_l, P.y_test, avg_te)
        fb_w = lib.fbeta_test_with_val_threshold(trainer, P.y_val, val_wavg_l, P.y_test, wavg_te_l)
        fb_r = lib.fbeta_test_with_val_threshold(trainer, P.y_val, val_rank_l, P.y_test, rank_te_l)
        fb_st = lib.fbeta_test_with_val_threshold(trainer, P.y_val, val_stack_l, P.y_test, stacked_te)
        candidates_l = [('SimpleAvg', fb_s, avg_te, val_simple_l), ('WeightedAvg', fb_w, wavg_te_l, val_wavg_l), ('RankAvg', fb_r, rank_te_l, val_rank_l), ('Stacking', fb_st, stacked_te, val_stack_l)]
        best_nm = max(candidates_l, key=lambda x: x[1])[0]
        if verbose:
            print(f'\n  Fβ (β={lib.PRIMARY_FBETA}) на test при пороге с val: Simple={fb_s:.4f} Weighted={fb_w:.4f} Rank={fb_r:.4f} Stacking={fb_st:.4f}')
            print(f'  Лучший ансамбль (по Fβ): {best_nm}')
        pred_map_l = {c[0]: (c[2], c[3]) for c in candidates_l}
        best_pred_l, val_for_th_l = pred_map_l[best_nm]
        if results_key == 'BestEnsemble':
            P.unified_best_ensemble_val_prob = np.asarray(val_for_th_l, dtype=float).copy()
        elif results_key == 'BestEnsemble_no_NN':
            P.unified_best_ensemble_no_nn_val_prob = np.asarray(val_for_th_l, dtype=float).copy()
        th_l, _ = trainer.find_optimal_threshold(P.y_val, val_for_th_l)
        ens_ml = trainer.evaluate_model(results_key, P.y_test, best_pred_l, threshold=th_l)
        trainer.results[results_key] = ens_ml
        trainer.predictions[results_key] = best_pred_l
        fb_out = ens_ml.get('F_beta', 0) or 0
        label_h = 'BestEnsemble' if results_key == 'BestEnsemble' else results_key
        print(f"  {label_h} ({best_nm}) — ROC-AUC: {ens_ml['ROC-AUC']:.4f} | PR-AUC: {ens_ml['PR-AUC']:.4f} | Fβ: {fb_out:.4f}")
        return (best_nm, ens_ml)
    _run_ensemble_pipeline(ensemble_models, 'BestEnsemble', verbose=True)
    if had_nn:
        em_no_nn = {k: v for k, v in ensemble_models.items() if k != 'NN_Classifier'}
        if len(em_no_nn) >= 2:
            print('\n--- То же ансамблирование без NN_Classifier (сравнение с полным составом) ---')
            out = _run_ensemble_pipeline(em_no_nn, 'BestEnsemble_no_NN', verbose=False)
            if out is not None:
                fb_full = trainer.results.get('BestEnsemble', {}).get('F_beta') or 0
                fb_nn_out = trainer.results.get('BestEnsemble_no_NN', {}).get('F_beta') or 0
                delta = float(fb_nn_out) - float(fb_full)
                print(f'  Сводка Fβ на test (порог с val): с NN — {fb_full:.4f}, без NN — {fb_nn_out:.4f} (Δ={delta:+.4f})')
    print('\n--- Итоговое сравнение всех моделей (включая NN и ансамбль) ---')
    P.metrics_df = lib.plot_comprehensive_comparison(P.trainer.results, P.trainer.predictions, P.y_test)

def stage_11_shap() -> None:
    from . import antifraud_lib as lib
    trainer = P.trainer
    best_boosting = max([(n, trainer.results[n].get('F_beta') or 0) for n in ['XGBoost', 'LightGBM', 'CatBoost'] if n in trainer.results], key=lambda x: x[1])
    P.best_model_name = best_boosting[0]
    P.best_model = trainer.models[P.best_model_name]
    print(f'Модель для SHAP: {P.best_model_name} (Fβ, β={lib.PRIMARY_FBETA}={best_boosting[1]:.4f})')
    P.interpreter = lib.ModelInterpreter(model=P.best_model, X_train=P.X_train, X_test=P.X_test, y_test=P.y_test, feature_names=P.preparer.feature_columns, model_name=P.best_model_name, cat_column_names=P.preparer.cat_columns)
    P.interpreter.compute_shap_values()
    P.interpreter.plot_shap_summary()
    P.interpreter.plot_shap_bar()
    P.interpreter.plot_native_importance()

def stage_12_drift() -> None:
    from . import antifraud_lib as lib
    trainer = P.trainer
    name = P.best_model_name
    val_scores = trainer.models[name].predict_proba(P.X_val)[:, 1]
    P.monitor = lib.DriftMonitor(reference_data=P.X_val, reference_scores=val_scores, reference_labels=P.y_val.values, feature_names=P.preparer.feature_columns)
    test_scores = trainer.predictions[name]
    th = trainer.results[name]['Threshold']
    P.drift_report = P.monitor.full_monitoring_check(current_data=P.X_test, current_scores=test_scores, current_labels=P.y_test.values, threshold=th)
    print(f"Дрейф признаков: {P.drift_report['feature_drift']['n_drifted']}")
    print(f"Рекомендация: {P.drift_report['recommendation']}")
    P.monitor.plot_drift_report(P.drift_report, current_scores=test_scores)

def stage_15_combined() -> None:
    from . import antifraud_lib as lib
    lib.refresh_unified_tabular_metrics(P, quiet=True)
    print('  Перед Combined: выполнен тихий единый пересчёт метрик (пороги с val).')
    trainer = P.trainer
    ml_name = resolve_best_ml_model_name_for_combined(trainer)
    if ml_name is None:
        ml_name = getattr(P, 'best_model_name', None) or 'XGBoost'
        print(f'  Не удалось выбрать модель по Fβ среди sklearn-моделей — fallback для Combined: {ml_name}')
    else:
        fb = trainer.results[ml_name].get('F_beta')
        print(f'  Combined: ML = {ml_name} (Fβ={fb:.4f} — лучшая по Fβ на test среди доступных ML-моделей; не SHAP-only)')
    if ml_name not in trainer.models:
        print(f"  Ошибка: модель '{ml_name}' нет в trainer.models — Combined пропущен.")
        return
    P.combined_ml_model_name = ml_name
    th = trainer.results[ml_name]['Threshold']
    pipe = lib.CombinedAntiFraudPipeline(P.rule_filter, trainer.models[ml_name], threshold=th)
    P.combined_results = pipe.evaluate(P.X_test, P.y_test)
    P.trainer.results['Combined_Pipeline'] = P.combined_results

def stage_11_compare_plots() -> None:
    from . import antifraud_lib as lib
    P.metrics_df = lib.plot_comprehensive_comparison(P.trainer.results, P.trainer.predictions, P.y_test)

def stage_16_business() -> None:
    from . import antifraud_lib as lib
    P.biz_analyzer = lib.BusinessMetricsAnalyzer()
    preds = P.trainer.predictions[P.best_model_name]
    th = P.trainer.results[P.best_model_name]['Threshold']
    y_pred = (preds >= th).astype(int)
    P.biz_report = P.biz_analyzer.compute_business_impact(P.y_test, y_pred)
    for k, v in P.biz_report.items():
        print(f'  {k}: {v:,.2f}' if isinstance(v, float) else f'  {k}: {v:,}')
    P.threshold_analysis = P.biz_analyzer.threshold_sensitivity_analysis(P.y_test, preds)
    P.biz_analyzer.plot_threshold_analysis(P.threshold_analysis)

def stage_13_calibration() -> None:
    from . import antifraud_lib as lib
    P.calibrated_model, P.calibrated_probs = lib.calibrate_and_evaluate(P.trainer.models[P.best_model_name], P.X_val, P.y_val, P.X_test, P.y_test, model_name=P.best_model_name)

def stage_14_retraining() -> None:
    from . import antifraud_lib as lib
    import xgboost as xgb
    params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'eval_metric': 'aucpr', 'early_stopping_rounds': 30, 'random_state': 42, 'n_jobs': -1, 'scale_pos_weight': P.trainer.scale_pos_weight, 'verbosity': 0}
    P.retrain_pipeline = lib.RetrainingPipeline(xgb.XGBClassifier, params, min_improvement=0.001)
    P.retrain_pipeline.set_champion(P.trainer.models[P.best_model_name], {'F_beta': P.trainer.results[P.best_model_name].get('F_beta'), 'PR-AUC': P.trainer.results[P.best_model_name]['PR-AUC']})
    ch = xgb.XGBClassifier(**params)
    ch.fit(P.X_train, P.y_train, eval_set=[(P.X_val, P.y_val)], verbose=False)
    replaced = P.retrain_pipeline.champion_challenger_test(ch, P.X_test, P.y_test, trainer=P.trainer, y_val=P.y_val, X_val=P.X_val)
    if replaced:
        new_model = P.retrain_pipeline.current_champion
        val_scores_new = new_model.predict_proba(P.X_val)[:, 1]
        test_scores_new = new_model.predict_proba(P.X_test)[:, 1]
        th_new, _ = P.trainer.find_optimal_threshold(P.y_val, val_scores_new)
        P.trainer.models['XGBoost'] = new_model
        P.trainer.predictions['XGBoost'] = test_scores_new
        P.trainer.results['XGBoost'] = P.trainer.evaluate_model('XGBoost', P.y_test, test_scores_new, threshold=th_new)
        print("  Дообученный XGBoost записан в P.trainer (models / predictions / results['XGBoost']).")
        if P.best_model_name == 'XGBoost':
            print('  best_model_name — XGBoost: дальнейшие шаги с лучшей моделью видят обновлённые веса.')
        if 'BestEnsemble' in P.trainer.results:
            print('  Предупреждение: BestEnsemble включает XGBoost-компонент со старыми скорами. Метрики ансамбля в §17 приблизительны (ΔFβ < 0.002). Для точного пересчёта: stage_10_ensemble() → stage_17_report().')
        if getattr(P, 'monitor', None) is not None:
            P.monitor.update_baseline(P.X_val.copy(), val_scores_new, P.y_val.values)
            P.drift_report = P.monitor.full_monitoring_check(current_data=P.X_test, current_scores=test_scores_new, current_labels=P.y_test.values, threshold=th_new)
            print('  Бейзлайн мониторинга дрейфа обновлён (reference = val под новым чемпионом).')
            print(f"  Повторная проверка (test vs новый baseline): {P.drift_report['recommendation']}")
            P.monitor.plot_drift_report(P.drift_report, current_scores=test_scores_new)

def _final_winner_y_pred(P, trainer, lib, winner_name: str):
    if winner_name == 'Combined_Pipeline':
        rf = getattr(P, 'rule_filter', None)
        ml = getattr(P, 'combined_ml_model_name', None) or getattr(P, 'best_model_name', None)
        if rf is None or not ml or ml not in trainer.models:
            return None
        th_ml = trainer.results[ml]['Threshold']
        pipe = lib.CombinedAntiFraudPipeline(rf, trainer.models[ml], threshold=th_ml)
        y_pred, _, _ = pipe.predict(P.X_test)
        return np.asarray(y_pred, dtype=int)
    if winner_name not in getattr(trainer, 'predictions', {}) or trainer.predictions[winner_name] is None:
        return None
    preds = np.asarray(trainer.predictions[winner_name], dtype=float).ravel()
    r = trainer.results.get(winner_name, {})
    th = r.get('Threshold')
    if th is None:
        th = 0.5
    return (preds >= float(th)).astype(int)

def stage_17_report() -> None:
    from . import antifraud_lib as lib

    def _print_metrics_block(title: str, name: str, r: dict) -> None:
        if not r:
            print(f'  {title}: {name} — метрики недоступны')
            return
        print(f'  {title}: {name}')
        fb = r.get('F_beta')
        fb_s = f'{fb:.4f}' if fb is not None else 'n/a'
        line1 = f"    PR-AUC={r.get('PR-AUC', 0):.4f}  ROC-AUC={r.get('ROC-AUC', 0):.4f}  Fβ={fb_s}  F1={r.get('F1', 0) or 0:.4f}  Precision={r.get('Precision', 0):.4f}  Recall={r.get('Recall', 0):.4f}"
        print(line1)
        th = r.get('Threshold')
        fpr = r.get('FPR')
        ll = r.get('Log Loss')
        extra = []
        if th is not None:
            extra.append(f'порог={th:.4f}')
        if fpr is not None:
            extra.append(f'FPR={fpr:.4f}')
        if ll is not None:
            extra.append(f'LogLoss={ll:.4f}')
        if extra:
            print(f"    {'  '.join(extra)}")
        tp, fp, tn, fn = (r.get('TP'), r.get('FP'), r.get('TN'), r.get('FN'))
        if all((x is not None for x in (tp, fp, tn, fn))):
            print(f'    матрица: TP={tp} FP={fp} TN={tn} FN={fn}')
    trainer = getattr(P, 'trainer', None)
    if trainer is None or not getattr(trainer, 'results', None):
        print('  Нет результатов обучения (запустите этап 6+).')
        print(f'\nАртефакты: {lib.OUTPUT_DIR}')
        return
    lib.refresh_unified_tabular_metrics(P)
    sm = sorted([(n, r.get('F_beta') or 0) for n, r in trainer.results.items()], key=lambda x: x[1], reverse=True)
    print(f'\n  --- Топ моделей по Fβ на test (β={lib.PRIMARY_FBETA}, порог с val) ---')
    for i, (n, fb) in enumerate(sm[:8], 1):
        print(f'  {i}. {n}: {fb:.4f}')
    if sm:
        winner_name, _winner_fb = sm[0]
        winner_r = trainer.results.get(winner_name, {})
        print('\n  --- Итоговая модель (лучшая по Fβ на test) ---')
        _print_metrics_block('Модель', winner_name, winner_r)
        if winner_name == 'GraphSAGE':
            print('    Примечание: GraphSAGE — графовая модель; скоры выровнены по order_id к тем же строкам val/test, что и у GBM (см. этап стекинга GNN).')
        elif winner_name == 'BestEnsemble':
            print('    Примечание: BestEnsemble может включать GraphSAGE в avg/rank/stack вместе с GBM/NN (если GNN зарегистрирован на этапе 12).')
    best_shap = getattr(P, 'best_model_name', None)
    if best_shap and best_shap in trainer.results and (best_shap != (sm[0][0] if sm else None)):
        print('\n  --- Модель для SHAP / калибровки / бизнес-метрик (разделы 10–13) ---')
        _print_metrics_block('Модель', best_shap, trainer.results[best_shap])
    comb = getattr(P, 'combined_results', None) or trainer.results.get('Combined_Pipeline')
    if comb:
        print('\n  --- Пайплайн «правила + ML» (Combined) ---')
        cm = getattr(P, 'combined_ml_model_name', None)
        if cm:
            print(f'  (ML-часть: {cm} — выбрана по Fβ в разделе 15)')
        _print_metrics_block('Система', 'Combined_Pipeline', comb)
    _comb = getattr(P, 'combined_results', None) or {}
    print(f"\n  Combined ROC-AUC (как в разделе 15): {_comb.get('ROC-AUC', 0):.4f}")
    if getattr(P, 'drift_report', None):
        print(f"  Мониторинг дрейфа: {P.drift_report.get('recommendation')}")
    if sm:
        final_name = sm[0][0]
        P.final_best_model_name = final_name
        y_pred_final = _final_winner_y_pred(P, trainer, lib, final_name)
        print('\n  --- Итоговые бизнес-метрики (лучшая по Fβ на test после всех этапов) ---')
        print(f'  Модель: {final_name}')
        if y_pred_final is None:
            print('  Не удалось восстановить предсказания (нет rule_filter или predictions).')
            P.final_biz_report = None
        else:
            biz = lib.BusinessMetricsAnalyzer()
            P.final_biz_report = biz.compute_business_impact(P.y_test, y_pred_final)
            for k, v in P.final_biz_report.items():
                print(f'  {k}: {v:,.2f}' if isinstance(v, float) else f'  {k}: {v:,}')
    _care_path = Path('outputs/gnn/care/gnn_scores.parquet')
    if _care_path.exists():
        try:
            import pandas as _pd_care
            import numpy as _np_care
            from sklearn.metrics import average_precision_score as _ap_care, roc_auc_score as _roc_care, fbeta_score as _fb_care
            _df_care = _pd_care.read_parquet(_care_path)
            _val_oids = np.asarray(getattr(P, 'val_order_ids', None) or [])
            _te_oids = np.asarray(getattr(P, 'test_order_ids', None) or [])
            _y_val_np = np.asarray(getattr(P, 'y_val', []))
            _y_te_np = np.asarray(getattr(P, 'y_test', []))
            if len(_te_oids) and len(_y_te_np) == len(_te_oids):
                _col = 'care_gnn_fraud_prob' if 'care_gnn_fraud_prob' in _df_care.columns else 'gnn_fraud_prob'
                _oid2p = dict(zip(_df_care['order_id'].astype(str).values, _df_care[_col].values))
                _p_val = _np_care.array([_oid2p.get(str(o), 0.5) for o in _val_oids], dtype=float)
                _p_te = _np_care.array([_oid2p.get(str(o), 0.5) for o in _te_oids], dtype=float)
                _best_th_c, _best_fb_v = (0.5, -1.0)
                for _th in _np_care.arange(0.01, 0.99, 0.01):
                    _fv = _fb_care(_y_val_np, (_p_val >= _th).astype(int), beta=2.0, zero_division=0)
                    if _fv > _best_fb_v:
                        _best_fb_v, _best_th_c = (_fv, float(_th))
                _fb_te_c = _fb_care(_y_te_np, (_p_te >= _best_th_c).astype(int), beta=2.0, zero_division=0)
                _prauc_c = _ap_care(_y_te_np, _p_te)
                _rocauc_c = _roc_care(_y_te_np, _p_te)
                print('\n  --- CARE-GNN (мульти-реляционная агрегация) ---')
                print(f'  PR-AUC={_prauc_c:.4f}  ROC-AUC={_rocauc_c:.4f}  Fβ={_fb_te_c:.4f}  порог={_best_th_c:.2f}')
        except Exception as _e_care:
            print(f'\n  CARE-GNN: ошибка при загрузке скоров ({_e_care})')
    print(f'\n  Артефакты (графики, CSV): {lib.OUTPUT_DIR}')

def run_all_stages() -> None:
    init_notebook()
    stage_01_load_and_profile()
    stage_02_eda_plots()
    stage_03_feature_engineering()
    stage_04_prepare_split()
    stage_05_rules()
    stage_06_llm_labeling()
    stage_07_train_models()
    stage_08_gnn()
    stage_09_neural()
    stage_10_ensemble()
    stage_11_shap()
    stage_11a_gnn_explain()
    stage_12_drift()
    stage_13_calibration()
    stage_14_retraining()
    stage_15_combined()
    stage_16_business()
    stage_17_report()

def _llm_display_results(df_labeled: 'pd.DataFrame') -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
    Path('figures').mkdir(exist_ok=True)
    if 'llm_error' in df_labeled.columns:
        _err = df_labeled['llm_error']
        ok_mask = _err.isna() | _err.astype(str).str.strip().str.lower().isin(['nan', 'none', ''])
    else:
        ok_mask = pd.Series(True, index=df_labeled.index)
    print(f'\n  Успешно: {ok_mask.sum():,}/{len(df_labeled):,}   Ошибок: {(~ok_mask).sum()}')
    if 'llm_resolution_lvl_1' in df_labeled.columns:
        print('\n  Распределение llm_resolution_lvl_1:')
        print(df_labeled.loc[ok_mask, 'llm_resolution_lvl_1'].value_counts(dropna=False).to_string())
    df_ok = df_labeled[ok_mask].copy()
    if len(df_ok) == 0:
        print('  Все строки — ошибки LLM, показываем полный датасет.')
        df_ok = df_labeled.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if 'llm_resolution_lvl_1' in df_ok.columns:
        lvl1_counts = df_ok['llm_resolution_lvl_1'].value_counts(dropna=False)
        if len(lvl1_counts) > 0:
            lvl1_counts.plot.bar(ax=axes[0], color='#3498db')
        else:
            axes[0].text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_title('Распределение llm_resolution_lvl_1')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)
    if 'llm_confidence' in df_ok.columns:
        conf_counts = df_ok['llm_confidence'].fillna('unknown').astype(str).value_counts()
        if len(conf_counts) > 0:
            conf_counts.plot.bar(ax=axes[1], color='#2ecc71')
        else:
            axes[1].text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Распределение confidence')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig('figures/llm_labeling_distribution.png', dpi=120, bbox_inches='tight')
    plt.show()

    def _row_3class(r_fraud, r_neg, r_nofr):
        if r_fraud == 1:
            return 'fraud'
        if r_neg == 1:
            return 'negative'
        if r_nofr == 1:
            return 'no_fraud'
        return None
    expert_cols = {'resolution_fraud', 'resolution_negative', 'resolution_no_fraud'}
    if expert_cols.issubset(df_ok.columns):
        df_ok = df_ok.copy()
        for _col in list(expert_cols) + ['llm_resolution_fraud', 'llm_resolution_negative', 'llm_resolution_no_fraud']:
            if _col in df_ok.columns:
                df_ok[_col] = pd.to_numeric(df_ok[_col], errors='coerce').fillna(0).astype(int)
        df_ok['expert_3class'] = df_ok.apply(lambda r: _row_3class(r.get('resolution_fraud', 0), r.get('resolution_negative', 0), r.get('resolution_no_fraud', 0)), axis=1)
        df_ok['llm_3class'] = df_ok.apply(lambda r: _row_3class(r.get('llm_resolution_fraud', 0), r.get('llm_resolution_negative', 0), r.get('llm_resolution_no_fraud', 0)), axis=1)
        cmp = df_ok.dropna(subset=['expert_3class', 'llm_3class'])
        print(f'\n  Строк для 3-классового сравнения: {len(cmp):,}')
        print(f"  Эскалаций LLM (все флаги = 0):    {df_ok['llm_3class'].isna().sum()}")
        if len(cmp) == 0:
            print('  Нет строк для сравнения — пропускаем матрицу ошибок.')
            return
        LABELS = ['fraud', 'negative', 'no_fraud']
        LABELS_RU = {'fraud': 'фрод', 'negative': 'негатив', 'no_fraud': 'без негатива'}
        print('\n' + '=' * 60)
        print('CLASSIFICATION REPORT (3 класса: fraud / negative / no_fraud)')
        print('=' * 60)
        print(classification_report(cmp['expert_3class'], cmp['llm_3class'], labels=LABELS, target_names=[LABELS_RU[l] for l in LABELS], zero_division=0))
        cm = confusion_matrix(cmp['expert_3class'], cmp['llm_3class'], labels=LABELS)
        labels_ru = [LABELS_RU[l] for l in LABELS]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=labels_ru)
        disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
        axes[0].set_title('Confusion Matrix\n(строки = эксперт, столбцы = LLM)')
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        disp_norm = ConfusionMatrixDisplay(np.round(cm_norm, 2), display_labels=labels_ru)
        disp_norm.plot(ax=axes[1], colorbar=False, cmap='Blues', values_format='.0%')
        axes[1].set_title('Recall per class\n(строки нормированы)')
        plt.tight_layout()
        plt.savefig('figures/llm_3class_confusion.png', dpi=120, bbox_inches='tight')
        plt.show()
        print('\n' + '=' * 60)
        print('BREAKDOWN ПО УВЕРЕННОСТИ (llm_confidence)')
        print('=' * 60)
        CONF_LEVELS = ['high', 'medium', 'low']
        conf_col = 'llm_confidence'
        cmp = cmp.copy()
        if conf_col in cmp.columns:
            cmp[conf_col] = cmp[conf_col].astype(str).str.lower().str.strip()
        summary_rows = []
        for conf in CONF_LEVELS:
            sub = cmp[cmp[conf_col] == conf] if conf_col in cmp.columns else pd.DataFrame()
            if sub.empty:
                print(f'\n  [{conf.upper()}] — нет данных')
                continue
            n_total = len(sub)
            n_agree = (sub['expert_3class'] == sub['llm_3class']).sum()
            accuracy = n_agree / n_total
            print(f'\n  [{conf.upper()}]  n={n_total:,}  accuracy={accuracy:.1%}')
            print(classification_report(sub['expert_3class'], sub['llm_3class'], labels=LABELS, target_names=[LABELS_RU[l] for l in LABELS], zero_division=0))
            for cls in LABELS:
                y_true_bin = (sub['expert_3class'] == cls).astype(int)
                y_pred_bin = (sub['llm_3class'] == cls).astype(int)
                summary_rows.append({'confidence': conf, 'class': LABELS_RU[cls], 'n_expert': int(y_true_bin.sum()), 'precision': precision_score(y_true_bin, y_pred_bin, zero_division=0), 'recall': recall_score(y_true_bin, y_pred_bin, zero_division=0), 'f1': f1_score(y_true_bin, y_pred_bin, zero_division=0)})
        if summary_rows:
            print('\n' + '=' * 60)
            print('СВОДНАЯ ТАБЛИЦА: precision / recall / f1 по классу × уверенности')
            print('=' * 60)
            summary_df = pd.DataFrame(summary_rows)
            pivot = summary_df.pivot_table(index='class', columns='confidence', values=['precision', 'recall', 'f1'], aggfunc='first').round(3)
            try:
                display(pivot)
            except NameError:
                print(pivot.to_string())
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
            colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}
            for ax, metric in zip(axes, ['precision', 'recall', 'f1']):
                for conf in CONF_LEVELS:
                    sub_s = summary_df[summary_df['confidence'] == conf]
                    if sub_s.empty:
                        continue
                    ax.bar(np.arange(len(LABELS)) + CONF_LEVELS.index(conf) * 0.25, [sub_s[sub_s['class'] == LABELS_RU[l]][metric].values[0] if len(sub_s[sub_s['class'] == LABELS_RU[l]]) > 0 else 0 for l in LABELS], width=0.25, label=conf, color=colors[conf], alpha=0.85)
                ax.set_xticks(np.arange(len(LABELS)) + 0.25)
                ax.set_xticklabels([LABELS_RU[l] for l in LABELS], rotation=15)
                ax.set_title(metric)
                ax.set_ylim(0, 1.05)
                ax.legend(title='confidence')
            plt.suptitle('Метрики по классу × уверенности LLM', y=1.02)
            plt.tight_layout()
            plt.savefig('figures/llm_metrics_by_confidence.png', dpi=120, bbox_inches='tight')
            plt.show()
            print('\n  Практический вывод:')
            for conf in CONF_LEVELS:
                sub_s = summary_df[summary_df['confidence'] == conf]
                if sub_s.empty:
                    continue
                n_sub = len(cmp[cmp[conf_col] == conf])
                sub_cmp = cmp[cmp[conf_col] == conf]
                acc_sub = (sub_cmp['expert_3class'] == sub_cmp['llm_3class']).mean() if n_sub > 0 else float('nan')
                fraud_rec = sub_s[sub_s['class'] == 'фрод']['recall'].values
                fraud_rec_val = fraud_rec[0] if len(fraud_rec) > 0 else float('nan')
                print(f'    {conf:6s}: n={n_sub:4d}  accuracy={acc_sub:.0%}  recall(фрод)={fraud_rec_val:.0%}')
    else:
        print('\n  Экспертные метки отсутствуют — 3-классовую валидацию пропускаем.')

def stage_06_llm_labeling() -> None:
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    out_path = Path('outputs/llm_labels.parquet')
    force_retrain = os.environ.get('FORCE_LLM_RETRAIN', '').strip().lower() in ('1', 'true', 'yes')
    if out_path.exists() and (not force_retrain):
        try:
            P.df_llm_labeled = pd.read_parquet(out_path)
            print(f'  LLM: найден готовый артефакт, разметка пропущена: {out_path}')
            print('  Принудительно перезапустить LLM-разметку: FORCE_LLM_RETRAIN=1')
            _llm_display_results(P.df_llm_labeled)
            return
        except Exception as e:
            print(f'  LLM: не удалось прочитать {out_path}, запускаю разметку заново ({e}).')
    try:
        import sys, importlib.util
        _root = str(Path(__file__).parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        import tokens as _tokens
    except Exception:
        pass
    if not os.environ.get('SOY_TOKEN'):
        print('  SOY_TOKEN не установлен — этап пропущен.')
        P.df_llm_labeled = None
        return
    from antifraud_components.labeling.eliza_client import ElizaClient
    from antifraud_components.labeling.llm_labeler import LLMLabeler
    eliza = ElizaClient.from_env(timeout=90)
    if not eliza.probe():
        print('  Рабочий эндпоинт не найден. Проверь SOY_TOKEN/VPN.')
        P.df_llm_labeled = None
        return
    print(f'  Эндпоинт: {eliza.base_url}\n  Модель:   {eliza.model_name}')
    llm_data_path = Path('data/llm_labeling.xlsx')
    if not llm_data_path.exists():
        print(f'  {llm_data_path} не найден — этап пропущен.')
        P.df_llm_labeled = None
        return
    df_to_label = pd.read_excel(llm_data_path, engine='openpyxl').dropna(axis=1, how='all')
    if df_to_label.shape[0] > 0:
        if df_to_label.iloc[0].astype(str).str.match('^(string|uint\\d*|int\\d*|double|float\\d*|bool)$').any():
            df_to_label = df_to_label.iloc[1:].reset_index(drop=True)
    print(f'  Загружено: {len(df_to_label):,} строк, {df_to_label.shape[1]} колонок')

    def _has_resolution_lvl1(df: pd.DataFrame) -> pd.Series:
        if 'resolution_lvl_1' not in df.columns:
            return pd.Series(True, index=df.index)
        s = df['resolution_lvl_1']
        t = s.astype(str).str.strip()
        return s.notna() & (t != '') & (t.str.lower() != 'nan') & (t.str.lower() != '<na>')
    if 'resolution_lvl_1' in df_to_label.columns:
        n_before = len(df_to_label)
        _m = _has_resolution_lvl1(df_to_label)
        df_to_label = df_to_label.loc[_m].copy()
        print(f'  С экспертной меткой: {len(df_to_label):,} (отфильтровано без метки: {n_before - len(df_to_label):,})')
    system_prompt = LLMLabeler._load_system_prompt(Path('docs/llm_alert_resolution_labeling_prompt.md'))
    labeler = LLMLabeler(client=eliza, system_prompt=system_prompt, max_workers=2)
    LLM_SAMPLE_SIZE = 2000
    if 'resolution_lvl_1' in df_to_label.columns:
        MIN_PER_TYPE = 5
        _strat_parts: list[pd.DataFrame] = []
        for _, grp in df_to_label.groupby('resolution_lvl_1', sort=False):
            _strat_parts.append(grp.sample(min(len(grp), MIN_PER_TYPE), random_state=42))
        strat = pd.concat(_strat_parts, axis=0) if _strat_parts else df_to_label.iloc[0:0]
        n_remaining = LLM_SAMPLE_SIZE - len(strat)
        if n_remaining > 0:
            pool_rest = df_to_label.drop(index=strat.index, errors='ignore')
            pool_rest = pool_rest.loc[_has_resolution_lvl1(pool_rest)].copy()
            k = min(n_remaining, len(pool_rest))
            rest = pool_rest.sample(k, random_state=42) if k > 0 else pool_rest.iloc[0:0]
            df_sample = pd.concat([strat, rest]).sample(frac=1, random_state=42)
        else:
            df_sample = strat.sample(frac=1, random_state=42)
        df_sample = df_sample.loc[_has_resolution_lvl1(df_sample)].copy()
        print(f'  К разметке: {len(df_sample)} строк (стратифицированная выборка по resolution_lvl_1)')
        print(df_sample['resolution_lvl_1'].value_counts(dropna=False).to_string())
    else:
        df_sample = df_to_label.head(LLM_SAMPLE_SIZE)
        print(f'  К разметке: {LLM_SAMPLE_SIZE} строк')
    df_labeled = labeler.label_dataframe(df_sample)
    P.df_llm_labeled = df_labeled
    Path('outputs').mkdir(exist_ok=True)
    df_save = df_labeled.copy()
    for _col in df_save.select_dtypes(include='object').columns:
        df_save[_col] = df_save[_col].astype(str)
    df_save.to_parquet(out_path, index=False)
    print(f'\n  Сохранено: {out_path}')
    _llm_display_results(df_labeled)

def _gnn_show_metrics(gnn_out: 'Path', scores_path: 'Path', history: dict, trained_fresh: bool) -> None:
    import matplotlib.pyplot as plt
    import pandas as _pd
    from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score, classification_report as _cr
    if not scores_path.exists():
        print('  gnn_scores.parquet не найден — запустите gnn_fraud_detection.ipynb.')
        return
    df_gnn_scores = _pd.read_parquet(scores_path)
    if trained_fresh:
        print(f'  Скоры сохранены: {scores_path}  ({len(df_gnn_scores):,} заказов)')
    else:
        print(f'  Скоры загружены из чекпойнта: {scores_path}  ({len(df_gnn_scores):,} заказов)')
    val_oids = np.asarray(P.val_order_ids).ravel()
    test_oids = np.asarray(P.test_order_ids).ravel()
    df_val = _pd.DataFrame({'order_id': val_oids, 'y': np.asarray(P.y_val).ravel()})
    df_test = _pd.DataFrame({'order_id': test_oids, 'y': np.asarray(P.y_test).ravel()})
    df_val = df_val.merge(df_gnn_scores[['order_id', 'gnn_fraud_prob']], on='order_id', how='left')
    df_test = df_test.merge(df_gnn_scores[['order_id', 'gnn_fraud_prob']], on='order_id', how='left')
    df_val['gnn_fraud_prob'] = df_val['gnn_fraud_prob'].fillna(0.5)
    df_test['gnn_fraud_prob'] = df_test['gnn_fraud_prob'].fillna(0.5)
    val_y = df_val['y'].values.astype(int)
    val_prob = df_val['gnn_fraud_prob'].values
    test_y = df_test['y'].values.astype(int)
    test_prob = df_test['gnn_fraud_prob'].values
    BETA = 2.0
    best_th, best_fb_val = (0.5, -1.0)
    for _th in np.arange(0.01, 0.99, 0.01):
        _fb = fbeta_score(val_y, (val_prob >= _th).astype(int), beta=BETA, zero_division=0)
        if _fb > best_fb_val:
            best_fb_val, best_th = (_fb, float(_th))
    test_fbeta = fbeta_score(test_y, (test_prob >= best_th).astype(int), beta=BETA, zero_division=0)
    test_prauc = average_precision_score(test_y, test_prob)
    test_rocauc = roc_auc_score(test_y, test_prob)
    print('\n' + '=' * 55)
    print(f'TEST  GraphSAGE  (порог по val Fβ = {best_th:.2f})')
    print('=' * 55)
    print(f'F_β (β={BETA}):  {test_fbeta:.4f}')
    print(f'PR-AUC:         {test_prauc:.4f}')
    print(f'ROC-AUC:        {test_rocauc:.4f}')
    print()
    print(_cr(test_y, (test_prob >= best_th).astype(int), target_names=['non-fraud', 'fraud']))
    if trained_fresh and history and (not history.get('skipped')):
        best_epoch = int(np.argmax(history['val_fbeta'])) + 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history['train_loss'], label='train loss')
        axes[0].axvline(best_epoch - 1, color='red', linestyle='--', label=f'best={best_epoch}')
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[1].plot(history['val_fbeta'])
        axes[1].axvline(best_epoch - 1, color='red', linestyle='--')
        axes[1].set_title(f'Val F_β (β={BETA})')
        axes[1].set_xlabel('Epoch')
        axes[2].plot(history['val_prauc'])
        axes[2].axvline(best_epoch - 1, color='red', linestyle='--')
        axes[2].set_title('Val PR-AUC')
        axes[2].set_xlabel('Epoch')
        plt.tight_layout()
        _fig_path = Path('figures/gnn_training_curves.png')
        plt.savefig(_fig_path, dpi=120, bbox_inches='tight')
        plt.show()
        print(f'  Кривые обучения: {_fig_path}')

def stage_08_gnn() -> None:
    from pathlib import Path
    from . import antifraud_lib as lib
    if not getattr(lib, 'TORCH_AVAILABLE', False):
        print('  PyTorch/torch_geometric недоступны — GNN пропущен.')
        P.gnn_model = None
        return
    import os
    import json as _json
    import pandas as _pd_gnn
    from antifraud_components.models.gnn import try_load_gnn_from_checkpoint, build_graph_data, plot_gnn_subgraph_preview, train_gnn
    force_retrain = os.environ.get('FORCE_GNN_RETRAIN', '').strip().lower() in ('1', 'true', 'yes')
    gnn_out = _gnn_dir()
    gnn_out.mkdir(parents=True, exist_ok=True)
    print(f'  GNN директория: {gnn_out}')
    _meta_path = gnn_out / 'gnn_meta.json'
    _scores_path = gnn_out / 'gnn_scores.parquet'
    _oids_path = gnn_out / 'gnn_order_ids.npy'
    _graph_path = gnn_out / 'gnn_graph_data.pt'
    if not force_retrain and _meta_path.exists() and _scores_path.exists():
        _meta = _json.loads(_meta_path.read_text(encoding='utf-8'))
        _saved_fc = _meta.get('feature_cols') or []
        _in_ch = _meta.get('in_channels', 0)
        _ckpt_hidden = _meta.get('hidden', 128)
        _ckpt_dropout = _meta.get('dropout', 0.3)
        _loaded = try_load_gnn_from_checkpoint(gnn_out, in_channels=_in_ch, hidden=_ckpt_hidden, dropout=_ckpt_dropout, feature_cols=_saved_fc, split_mode=_meta.get('split_mode'))
        if _loaded is None:
            import torch as _torch_diag
            _w = gnn_out / 'best_model.pt'
            if not _w.exists():
                _reason = f'нет файла весов {_w}'
            elif _meta.get('in_channels') != _in_ch:
                _reason = f"in_channels mismatch: meta={_meta.get('in_channels')} vs ожидалось {_in_ch}"
            elif _meta.get('hidden') != _ckpt_hidden:
                _reason = f"hidden mismatch: meta={_meta.get('hidden')} vs {_ckpt_hidden}"
            elif abs(float(_meta.get('dropout', 0)) - _ckpt_dropout) > 1e-06:
                _reason = f"dropout mismatch: meta={_meta.get('dropout')} vs {_ckpt_dropout}"
            else:
                _reason = 'неизвестная причина (возможно feature_cols или split_mode не совпали)'
            raise RuntimeError(f'Чекпойнт найден, но не загружен: {_reason}. Для переобучения с нуля: FORCE_GNN_RETRAIN=1')
        if _loaded is not None:
            import torch as _torch
            P.gnn_model, P.gnn_history = _loaded
            P.gnn_feature_cols = _saved_fc
            P.gnn_order_ids = np.load(_oids_path, allow_pickle=True) if _oids_path.exists() else np.array([])
            P.gnn_scaler = None
            print(f'  GNN: быстрая загрузка — граф не перестраивается')
            print(f"  best_epoch={_meta.get('best_epoch')}, best_val_Fβ={_meta.get('best_val_fbeta', 0):.4f}")
            _load_graph = os.environ.get('GNN_LOAD_GRAPH', '').strip().lower() in ('1', 'true', 'yes')
            if _load_graph and _graph_path.exists():
                try:
                    P.gnn_data = _torch.load(_graph_path, weights_only=False)
                    P.gnn_data.order_ids_np = np.asarray(P.gnn_order_ids)
                    print(f'  Граф загружен из кэша: {P.gnn_data.num_nodes:,} узлов, {P.gnn_data.num_edges:,} рёбер')
                except Exception as _eg:
                    P.gnn_data = None
                    print(f'  Граф не загружен ({_eg})')
            else:
                P.gnn_data = None
                if not _load_graph:
                    print('  Граф не загружен (для GNNExplainer: GNN_LOAD_GRAPH=1 или FORCE_GNN_RETRAIN=1)')
            try:
                _gnn_show_metrics(gnn_out, _scores_path, P.gnn_history, trained_fresh=False)
            except Exception as _e:
                print(f'  GNN оценка пропущена: {_e}')
            _register_graphsage_in_trainer(P)
            return
    GNN_ENTITY_COLS = ['card_id', 'device_id', 'cookie', 'delivery_phone', 'user_phone', 'puid', 'crypta_id', 'delivery_address_ids', 'ip', 'ip_isp', 'phone_hash']
    gnn_path = Path('data/df_gnn.parquet')
    _use_df_fe = os.environ.get('GNN_USE_DF_FE', '').strip().lower() in ('1', 'true', 'yes')
    if _use_df_fe:
        df_gnn = P.df_fe.copy()
        print(f"  GNN_USE_DF_FE=1 → используется P.df_fe (датасет ML): {df_gnn.shape}  ({df_gnn['resolution_fraud'].mean():.2%} фрода)")
        _missing_entity = [c for c in GNN_ENTITY_COLS if c not in df_gnn.columns]
        if _missing_entity and gnn_path.exists() and ('order_id' in df_gnn.columns):
            try:
                import pyarrow.parquet as _pq
                _gnn_schema_names = set(_pq.read_schema(gnn_path).names)
                _avail_entity = [c for c in _missing_entity if c in _gnn_schema_names]
                if _avail_entity:
                    _df_entity = _pd_gnn.read_parquet(gnn_path, columns=['order_id'] + _avail_entity)
                    df_gnn = df_gnn.merge(_df_entity, on='order_id', how='left')
                    print(f'  Подмержены entity-колонки из df_gnn.parquet: {_avail_entity}  → {df_gnn.shape}')
            except Exception as _ee:
                print(f'  Не удалось подмержить entity-колонки: {_ee}')
        elif _missing_entity and (not gnn_path.exists()):
            print('  df_gnn.parquet не найден — entity-колонки недоступны, рёбер не будет')
    elif gnn_path.exists():
        df_gnn = _pd_gnn.read_parquet(gnn_path)
        print(f"  Загружен GNN-датасет: {df_gnn.shape}  ({df_gnn['resolution_fraud'].mean():.2%} фрода)")
    else:
        df_gnn = P.df_fe
        print(f'  df_gnn.parquet не найден — используется P.df_fe: {df_gnn.shape}')
    entity_cols = [c for c in GNN_ENTITY_COLS if c in df_gnn.columns]
    print(f'  Entity-колонки для рёбер: {entity_cols}')
    df_fe = getattr(P, 'df_fe', None)
    if df_fe is not None and 'order_id' in df_gnn.columns and ('order_id' in df_fe.columns):
        fe_new = [c for c in df_fe.columns if c not in df_gnn.columns and c != 'order_id']
        if fe_new:
            df_gnn = df_gnn.merge(df_fe[['order_id'] + fe_new], on='order_id', how='left')
            print(f'  Слито {len(fe_new)} engineered-признаков из P.df_fe → {df_gnn.shape}')
    tro = getattr(P, 'train_order_ids', None)
    vro = getattr(P, 'val_order_ids', None)
    teo = getattr(P, 'test_order_ids', None)
    split_gnn: dict = {}
    gnn_split_mode = 'chronological'
    tabular_ok = tro is not None and vro is not None and (teo is not None) and ('order_id' in df_gnn.columns) and (len(np.asarray(tro).ravel()) > 0) and (len(np.asarray(vro).ravel()) > 0) and (len(np.asarray(teo).ravel()) > 0)
    if tabular_ok:
        split_gnn = {'train_order_ids': tro, 'val_order_ids': vro, 'test_order_ids': teo, 'y_train_labels': getattr(P, 'y_train', None), 'y_val_labels': getattr(P, 'y_val', None), 'y_test_labels': getattr(P, 'y_test', None)}
        gnn_split_mode = 'tabular_y_v2'
        print('  GNN: train/val/test по тем же order_id и меткам, что и табличные модели (этап 4).')
    else:
        print('  GNN: нет трёх непустых train/val/test order_id — внутренний сплит 65/15/20.')
    _ckpt_fc = None
    if not force_retrain and _meta_path.exists():
        _meta = _json.loads(_meta_path.read_text(encoding='utf-8'))
        _saved_fc = _meta.get('feature_cols') or []
        if _saved_fc:
            _ckpt_fc = _saved_fc
            print(f'  GNN: выравнивание по чекпойнту → {len(_saved_fc)} признаков')
    P.gnn_data, P.gnn_scaler, P.gnn_feature_cols, P.gnn_order_ids = build_graph_data(df_gnn, entity_cols=entity_cols or None, feature_cols=_ckpt_fc, id_col='order_id' if 'order_id' in df_gnn.columns else None, target_col='resolution_fraud', **split_gnn)
    print(f'  Граф: {P.gnn_data.num_nodes:,} узлов, {P.gnn_data.num_edges:,} рёбер, {len(P.gnn_feature_cols)} признаков')
    np.save(_oids_path, P.gnn_order_ids, allow_pickle=True)
    try:
        import torch as _torch
        _torch.save(P.gnn_data, _graph_path)
        print(f'  Граф сохранён в кэш: {_graph_path}')
    except Exception as _eg:
        print(f'  Граф не сохранён ({_eg})')
    P.gnn_model, P.gnn_history = train_gnn(P.gnn_data, in_channels=len(P.gnn_feature_cols), hidden=128, dropout=0.3, lr=0.001, epochs=80, early_stopping_patience=20, output_dir=str(gnn_out), feature_cols=P.gnn_feature_cols, skip_if_checkpoint=not force_retrain, split_mode=gnn_split_mode)
    prev_path = plot_gnn_subgraph_preview(P.gnn_data, output_path='figures/gnn_graph_preview.png', max_nodes=400)
    if prev_path is not None:
        _display_saved_figure_in_notebook(prev_path)
    P.gnn_data.order_ids_np = np.asarray(P.gnn_order_ids)
    trained_fresh = P.gnn_history and (not P.gnn_history.get('skipped'))
    if trained_fresh:
        from antifraud_components.models.gnn import gnn_predict
        probs_all = gnn_predict(P.gnn_model, P.gnn_data)
        df_s = _pd_gnn.DataFrame({'order_id': P.gnn_order_ids, 'gnn_fraud_prob': probs_all})
        df_s.to_parquet(_scores_path, index=False)
        print(f'  Скоры сохранены: {_scores_path}  ({len(df_s):,} заказов)')
    try:
        _gnn_show_metrics(gnn_out, _scores_path, P.gnn_history, trained_fresh=trained_fresh)
    except Exception as _e:
        print(f'  GNN оценка пропущена: {_e}')
    _register_graphsage_in_trainer(P)

def stage_10_snorkel() -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    from antifraud_components.labeling.label_model import LabelModel, ABSTAIN
    from antifraud_components.rules.l1_rules import build_l1_rule_filter_from_df
    lf_llm, df_llm = (None, None)
    try:
        df_llm = pd.read_parquet('outputs/llm_labels.parquet')
        print(f'  LLM labels: {len(df_llm):,} строк')
        lf_llm = LabelModel.llm_to_lf_labels(df_llm['llm_resolution_fraud'], df_llm['llm_error'])
        print(f'  LF (LLM):   {dict(zip(*np.unique(lf_llm, return_counts=True)))}')
    except FileNotFoundError:
        print('  outputs/llm_labels.parquet не найден — сначала LLM-разметка (stage_06_llm_labeling).')
    lf_rules = None
    if df_llm is not None:
        common = [c for c in P.X_train.columns if c in df_llm.columns]
        if common:
            rf = build_l1_rule_filter_from_df(df_llm[common])
            _, rule_preds = rf.apply(df_llm[common])
            lf_rules = LabelModel.rule_to_lf_labels(rule_preds)
            print(f'  LF (Rules): {dict(zip(*np.unique(lf_rules, return_counts=True)))}')
    lf_gnn = None
    try:
        df_gnn = pd.read_parquet(_gnn_dir() / 'gnn_scores.parquet')
        if df_llm is not None and 'order_id' in getattr(df_llm, 'columns', []) and ('order_id' in df_gnn.columns):
            merged = df_llm[['order_id']].merge(df_gnn, on='order_id', how='left')
            lf_gnn = LabelModel.score_to_lf_labels(merged['gnn_fraud_prob'].fillna(0.5).values)
            print(f'  LF (GNN):   {dict(zip(*np.unique(lf_gnn, return_counts=True)))}')
    except FileNotFoundError:
        print(f"  {_gnn_dir() / 'gnn_scores.parquet'} не найден — источник GNN пропущен.")
    sources = {n: lf for n, lf in [('LLM', lf_llm), ('Rules', lf_rules), ('GNN', lf_gnn)] if lf is not None}
    if len(sources) < 2:
        print('\n  Нужно ≥2 источников. Запусти LLM-разметку и GNN (stage_06 / stage_08).')
        P.lm_probs, P.label_model = (None, None)
        return
    n = min((len(v) for v in sources.values()))
    L = np.column_stack([v[:n] for v in sources.values()])
    print(f'\n  Матрица L: {L.shape}  источники: {list(sources.keys())}')
    print(f'  Покрытие:  {(L != ABSTAIN).mean(axis=0).round(3)}')
    Y_dev = None
    if df_llm is not None and 'resolution_fraud' in df_llm.columns:
        raw = df_llm['resolution_fraud'].values[:n]
        valid = ~np.isnan(raw.astype(float))
        if valid.sum() > 10:
            Y_dev = raw[valid].astype(int)
            print(f'  Dev-set:   {valid.sum():,} размеченных строк')
    lm = LabelModel(n_sources=len(sources))
    lm.fit(L, Y_dev=Y_dev, n_epochs=300, lr=0.01)
    P.lm_probs = lm.predict_proba(L)
    P.label_model = lm
    print(f'\n  P(fraud):  mean={P.lm_probs.mean():.3f}  std={P.lm_probs.std():.3f}')
    print(f'  > 0.5:     {(P.lm_probs > 0.5).sum():,}')
    if df_llm is not None and 'resolution_fraud' in df_llm.columns and df_llm['resolution_fraud'].notna().any():
        labeled = df_llm['resolution_fraud'].notna().values[:n]
        y_true = df_llm['resolution_fraud'].values[:n][labeled].astype(int)
        y_prob = P.lm_probs[labeled]
        y_pred = (y_prob >= 0.5).astype(int)
        print('\n  === LABEL MODEL vs EXPERT ===')
        print(f'  ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}')
        print(f'  PR-AUC:  {average_precision_score(y_true, y_prob):.4f}')
        print()
        print(classification_report(y_true, y_pred, target_names=['non-fraud', 'fraud']))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(y_prob[y_true == 0], bins=30, alpha=0.6, label='non-fraud', color='steelblue')
        axes[0].hist(y_prob[y_true == 1], bins=30, alpha=0.6, label='fraud', color='tomato')
        axes[0].set_title('P(fraud) Label Model по классам')
        axes[0].legend()
        source_aucs = []
        for i, name in enumerate(list(sources.keys())):
            lf_i = L[labeled, i]
            valid_s = lf_i != ABSTAIN
            if valid_s.sum() > 10:
                source_aucs.append((name, roc_auc_score(y_true[valid_s], lf_i[valid_s])))
        source_aucs.append(('LabelModel', roc_auc_score(y_true, y_prob)))
        _n, _a = zip(*source_aucs)
        axes[1].bar(_n, _a, color=['#3498db', '#e67e22', '#2ecc71', '#9b59b6'][:len(_n)])
        axes[1].set_title('ROC-AUC: источники vs Label Model')
        axes[1].set_ylim(0, 1)
        axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('figures/label_model_comparison.png', dpi=120, bbox_inches='tight')
        plt.show()

def stage_10_gnn_comparison() -> None:
    import matplotlib.pyplot as plt
    from antifraud_components.models.gnn import evaluate_gnn
    if getattr(P, 'gnn_model', None) is None:
        print('  GNN не обучен — сравнение пропущено.')
        return
    P.gnn_metrics = evaluate_gnn(P.gnn_model, P.gnn_data, mask_name='test_mask')
    print('\n  === GNN TEST METRICS (узлы графа в test_mask; Fβ — порог с val, как у GBM) ===')
    for k, v in P.gnn_metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    gbm_best = max(P.trainer.results, key=lambda k: P.trainer.results[k].get('F_beta', 0))
    gbm_fb = P.trainer.results[gbm_best].get('F_beta', 0)
    gnn_fb = P.gnn_metrics['F_beta']
    print(f"\n  {gbm_best} (GBM):   F_β={gbm_fb:.4f}  PR-AUC={P.trainer.results[gbm_best].get('PR-AUC', 0):.4f}")
    print(f"  GraphSAGE (GNN): F_β={gnn_fb:.4f}  PR-AUC={P.gnn_metrics['PR-AUC']:.4f}")
    print('\n  Примечание:\n  • Метрики здесь и в §12/20 должны совпадать: y берётся из табличного сплита (P.y_*),\n    а не из df_gnn.parquet (разные снэпшоты resolution_fraud).\n  • Если §11.1 расходится с §12 — значит stage_08 запущен со старым чекпойнтом без\n    переопределения y. Удалите outputs/gnn/best_model.pt и перезапустите stage_08_gnn().\n  • Итоговое сравнение всех моделей — в этапе 20 (единый пересчёт метрик).')
    hist = getattr(P, 'gnn_history', None) or {}
    tl = hist.get('train_loss') or []
    if len(tl) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(hist['train_loss'])
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[1].plot(hist['val_fbeta'])
        axes[1].set_title('Val F_β (β=2)')
        axes[1].set_xlabel('Epoch')
        axes[2].plot(hist['val_prauc'])
        axes[2].set_title('Val PR-AUC')
        axes[2].set_xlabel('Epoch')
        plt.suptitle('GNN GraphSAGE — кривые обучения', fontsize=13)
        plt.tight_layout()
        plt.savefig('figures/gnn_training_curves.png', dpi=120, bbox_inches='tight')
        plt.show()
        plt.close('all')
    else:
        print('  Кривые обучения не построены: нет истории эпох (модель из чекпойнта без gnn_history.json).')

def stage_11b_gnn_interactive(max_nodes: int=300, extra_hover_cols: list[str] | None=None) -> None:
    from antifraud_components.models.gnn import gnn_predict, plot_gnn_interactive
    if getattr(P, 'gnn_model', None) is None:
        print('  GNN не обучен — запустите stage_08_gnn().')
        return
    if not hasattr(P, 'gnn_probs_all') or P.gnn_probs_all is None:
        P.gnn_probs_all = gnn_predict(P.gnn_model, P.gnn_data)
    _hover_cols = extra_hover_cols or ['order_loan', 'card_age', 'phone_age', 'puid_age', 'glue_size', 'glue_ead']
    node_meta = None
    df_fe = getattr(P, 'df_fe', None)
    if df_fe is not None and 'order_id' in df_fe.columns:
        cols_avail = ['order_id'] + [c for c in _hover_cols if c in df_fe.columns]
        node_meta = df_fe[cols_avail].copy()
    plot_gnn_interactive(data=P.gnn_data, order_ids=getattr(P, 'gnn_order_ids', None), probs=P.gnn_probs_all, node_meta_df=node_meta, max_nodes=max_nodes, output_path=str(_gnn_dir() / 'gnn_interactive.html'), show_in_notebook=True)

def stage_12_gnn_stacking() -> None:
    from pathlib import Path
    import pandas as pd
    from antifraud_components.models.gnn import gnn_predict
    if getattr(P, 'gnn_model', None) is None:
        print('  GNN не обучен — стекинг пропущен.')
        return
    P.gnn_probs_all = gnn_predict(P.gnn_model, P.gnn_data)
    test_mask = P.gnn_data.test_mask.cpu().numpy()
    gnn_test_probs = P.gnn_probs_all[test_mask]
    print(f'  GNN test: {len(gnn_test_probs):,} заказов')
    print(f'  GBM test: {len(P.y_test):,} заказов')
    if 'order_id' in P.df_fe.columns:
        _gnn_dir().mkdir(parents=True, exist_ok=True)
        n_p = len(P.gnn_probs_all)

        def _resolve_order_ids() -> np.ndarray | None:
            for src in (getattr(P, 'gnn_order_ids', None), getattr(P.gnn_data, 'order_ids_np', None) if getattr(P, 'gnn_data', None) is not None else None):
                if src is None:
                    continue
                a = np.asarray(src)
                if a.shape[0] == n_p:
                    return a
            npy_path = _gnn_dir() / 'gnn_order_ids.npy'
            if npy_path.is_file():
                a = np.load(npy_path, allow_pickle=True)
                if a.shape[0] == n_p:
                    return a
            return None
        order_ids = _resolve_order_ids()
        if order_ids is not None:
            df_gnn_scores = pd.DataFrame({'order_id': order_ids, 'gnn_fraud_prob': P.gnn_probs_all}).drop_duplicates(subset=['order_id'], keep='first')
            df_scores = P.df_fe[['order_id']].merge(df_gnn_scores, on='order_id', how='left')
            n_na = int(df_scores['gnn_fraud_prob'].isna().sum())
            if n_na:
                print(f'  Предупреждение: для {n_na:,} заказов из df_fe нет узла в GNN-графе (gnn_fraud_prob=NaN).')
        elif n_p == len(P.df_fe):
            df_scores = P.df_fe[['order_id']].copy()
            df_scores['gnn_fraud_prob'] = P.gnn_probs_all
        else:
            _gd = _gnn_dir()
            raise ValueError(f"Не удалось сопоставить gnn_probs_all с order_id: нет P.gnn_order_ids / P.gnn_data.order_ids_np / {_gd / 'gnn_order_ids.npy'} длины {n_p:,}. Запустите stage_08_gnn() и перезагрузите модуль при обновлении кода (Kernel → Restart, затем importlib.reload(antifraud_pipeline.antifraud_stages)).")
        _scores_out = _gnn_dir() / 'gnn_scores.parquet'
        df_scores.to_parquet(_scores_out, index=False)
        print(f'  GNN-скоры сохранены: {_scores_out}')
    else:
        print('  order_id не найден — merge вручную по индексу.')
    _register_graphsage_in_trainer(P)

def stage_11a_gnn_explain() -> None:
    import importlib
    import antifraud_components.explain.gnn_explain as _gnn_explain
    importlib.reload(_gnn_explain)
    GNNExplainerWrapper = _gnn_explain.GNNExplainerWrapper
    from antifraud_components.models.gnn import gnn_predict
    if getattr(P, 'gnn_model', None) is None:
        print('  GNN не обучен — объяснимость пропущена.')
        return
    if getattr(P, 'gnn_data', None) is None:
        print('  GNN загружен из чекпойнта без графа — показываю сохранённые объяснения.')
        _explain_pngs = sorted(_gnn_dir().glob('gnn_explain_node_*.png'))
        if _explain_pngs:
            for _p in _explain_pngs:
                print(f'  {_p.name}')
                _display_saved_figure_in_notebook(_p)
        else:
            print('  Сохранённых PNG нет. Для пересчёта: FORCE_GNN_RETRAIN=1.')
        return
    n_nodes = int(P.gnn_data.num_nodes)
    n_edges = int(P.gnn_data.edge_index.shape[1])
    gnn_explainer = GNNExplainerWrapper(P.gnn_model, P.gnn_data, n_epochs=25, explain_num_hops=2, explain_max_nodes=2500)
    if n_nodes > 8000 or n_edges > 250000:
        print(f'  Граф большой ({n_nodes:,} узлов, {n_edges:,} рёбер): объяснение только на локальном подграфе (~{gnn_explainer.explain_max_nodes} узлов); важность признаков — |∂P(fraud)/∂x| (градиент), без PyG Explainer на подграфе.')
    if not hasattr(P, 'gnn_probs_all') or P.gnn_probs_all is None:
        P.gnn_probs_all = gnn_predict(P.gnn_model, P.gnn_data)
    test_mask = P.gnn_data.test_mask.cpu().numpy()
    fraud_idx = test_mask & (P.gnn_data.y.cpu().numpy() == 1)
    top_nodes = np.where(fraud_idx)[0][:5]
    print(f'  Топ-5 фродовых узлов в test: {top_nodes}')
    for node_idx in top_nodes[:2]:
        print(f'\n  --- Узел {node_idx} (fraud_prob={P.gnn_probs_all[node_idx]:.3f}) — топ признаков ---')
        for feat, score in gnn_explainer.top_features(node_idx, P.gnn_feature_cols, top_k=10):
            print(f'    {feat}: {score:.4f}')
    if len(top_nodes) > 0:
        gnn_explainer.visualize(node_idx=top_nodes[0], feature_names=P.gnn_feature_cols, output_dir=str(_gnn_dir()), top_k=15)
