# =============================================================================
# 6. Обучение моделей
# =============================================================================
print("\n" + "=" * 80)
print("РАЗДЕЛ 6. ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 80)


class ModelTrainer:
    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_columns,
        cat_columns_for_catboost,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.feature_columns = feature_columns
        self.cat_columns_for_catboost = list(cat_columns_for_catboost or [])
        self.models = {}
        self.results = {}
        self.predictions = {}
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        self.scale_pos_weight = float(n_neg / max(int(n_pos), 1))
        print(f"  scale_pos_weight = {self.scale_pos_weight:.2f}")

    def evaluate_model(self, name, y_true, y_prob, y_pred=None, threshold=0.5):
        if y_pred is None:
            y_pred = (y_prob >= threshold).astype(int)
        metrics = {
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "PR-AUC": average_precision_score(y_true, y_prob),
            "F1": f1_score(y_true, y_pred),
            "F2": fbeta_score(y_true, y_pred, beta=2),
            "F0.5": fbeta_score(y_true, y_pred, beta=0.5),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred),
            "Log Loss": log_loss(y_true, y_prob),
            "Threshold": threshold,
        }
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics.update({"TP": tp, "FP": fp, "TN": tn, "FN": fn})
        metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return metrics

    def find_optimal_threshold(self, y_true, y_prob, metric="f1", max_fpr=None, beta=1.0):
        best_score = -1.0
        best_threshold = 0.5
        for th in np.arange(0.01, 0.99, 0.01):
            y_pred = (y_prob >= th).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            if max_fpr is not None and fpr > max_fpr:
                continue
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "f2":
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            elif metric == "fbeta":
                score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(th)
        return best_threshold, best_score

    def train_logistic_regression(self):
        print("\n--- Логистическая регрессия (baseline) ---")
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        X_te_scaled = scaler.transform(self.X_test)
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, C=0.1, solver="lbfgs", random_state=42
        )
        model.fit(X_tr_scaled, self.y_train)
        y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
        y_prob_test = model.predict_proba(X_te_scaled)[:, 1]
        best_th, best_f1 = self.find_optimal_threshold(self.y_val, y_prob_val, metric="f1")
        print(f"  Оптимальный порог (val F1): {best_th:.2f}, F1={best_f1:.4f}")
        metrics = self.evaluate_model("LogReg", self.y_test, y_prob_test, threshold=best_th)
        self.models["LogReg"] = {"model": model, "scaler": scaler}
        self.results["LogReg"] = metrics
        self.predictions["LogReg"] = y_prob_test
        print(f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}, F1: {metrics['F1']:.4f}")
        return model

    def train_xgboost(self):
        print("\n--- XGBoost ---")
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            gamma=0.1,
        )
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        y_prob_val = model.predict_proba(self.X_val)[:, 1]
        y_prob_test = model.predict_proba(self.X_test)[:, 1]
        best_th, best_f1 = self.find_optimal_threshold(self.y_val, y_prob_val, metric="f1")
        print(f"  Оптимальный порог (val F1): {best_th:.2f}, F1={best_f1:.4f}")
        metrics = self.evaluate_model("XGBoost", self.y_test, y_prob_test, threshold=best_th)
        self.models["XGBoost"] = model
        self.results["XGBoost"] = metrics
        self.predictions["XGBoost"] = y_prob_test
        print(f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")
        return model

    def train_lightgbm(self):
        print("\n--- LightGBM ---")
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            metric="average_precision",
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            num_leaves=63,
            verbose=-1,
        )
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        y_prob_val = model.predict_proba(self.X_val)[:, 1]
        y_prob_test = model.predict_proba(self.X_test)[:, 1]
        best_th, best_f1 = self.find_optimal_threshold(self.y_val, y_prob_val, metric="f1")
        print(f"  Оптимальный порог (val F1): {best_th:.2f}, F1={best_f1:.4f}")
        metrics = self.evaluate_model("LightGBM", self.y_test, y_prob_test, threshold=best_th)
        self.models["LightGBM"] = model
        self.results["LightGBM"] = metrics
        self.predictions["LightGBM"] = y_prob_test
        print(f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")
        return model

    def train_catboost(self):
        print("\n--- CatBoost ---")
        cat_indices = [
            i for i, col in enumerate(self.feature_columns) if col in self.cat_columns_for_catboost
        ]
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            eval_metric="PRAUC",
            early_stopping_rounds=50,
            random_seed=42,
            verbose=False,
            l2_leaf_reg=3.0,
            border_count=254,
            thread_count=-1,
        )
        train_pool = Pool(self.X_train, self.y_train, cat_features=cat_indices)
        val_pool = Pool(self.X_val, self.y_val, cat_features=cat_indices)
        model.fit(train_pool, eval_set=val_pool)
        y_prob_val = model.predict_proba(self.X_val)[:, 1]
        y_prob_test = model.predict_proba(self.X_test)[:, 1]
        best_th, best_f1 = self.find_optimal_threshold(self.y_val, y_prob_val, metric="f1")
        print(f"  Оптимальный порог (val F1): {best_th:.2f}, F1={best_f1:.4f}")
        metrics = self.evaluate_model("CatBoost", self.y_test, y_prob_test, threshold=best_th)
        self.models["CatBoost"] = model
        self.results["CatBoost"] = metrics
        self.predictions["CatBoost"] = y_prob_test
        print(f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")
        return model

    def train_xgboost_with_smote(self):
        print("\n--- XGBoost + SMOTE ---")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        print(f"  SMOTE: {len(self.X_train)} → {len(X_resampled)} наблюдений")
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
        )
        model.fit(X_resampled, y_resampled, eval_set=[(self.X_val, self.y_val)], verbose=False)
        y_prob_val = model.predict_proba(self.X_val)[:, 1]
        y_prob_test = model.predict_proba(self.X_test)[:, 1]
        best_th, best_f1 = self.find_optimal_threshold(self.y_val, y_prob_val, metric="f1")
        metrics = self.evaluate_model("XGBoost+SMOTE", self.y_test, y_prob_test, threshold=best_th)
        self.models["XGBoost+SMOTE"] = model
        self.results["XGBoost+SMOTE"] = metrics
        self.predictions["XGBoost+SMOTE"] = y_prob_test
        print(f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")
        return model

    def train_all(self):
        self.train_logistic_regression()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_catboost()
        self.train_xgboost_with_smote()


trainer = ModelTrainer(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_columns=preparer.feature_columns,
    cat_columns_for_catboost=preparer.cat_columns,
)
trainer.train_all()
