import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    fbeta_score,
    roc_curve,
    precision_score,
    recall_score,
    log_loss,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

import shap
from scipy import stats
from scipy.spatial.distance import jensenshannon

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
sns.set_style("whitegrid")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

# Родитель пакета — корень репозитория (каталоги data/, outputs/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def set_project_root(path: Path | None = None) -> None:
    """Для ноутбука: set_project_root(Path.cwd()) если CSV в cwd."""
    global PROJECT_ROOT, DATA_PATH, OUTPUT_DIR
    if path is not None:
        PROJECT_ROOT = path.resolve()
        DATA_PATH = PROJECT_ROOT / 'data' / 'df_split_model_RF_new_features.csv'
        OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'antifraud_pipeline'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "df_split_model_RF_new_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "antifraud_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Порог на val и выбор лучшей модели: F-beta, beta>1 усиливает recall относительно precision.
PRIMARY_FBETA = 2.0
THRESHOLD_METRIC = "fbeta"
# Раздел 5 — ручные правила: отбор на train.
# block: не только Fβ «в один столбец», а качество зоны срабатывания:
#   precision среди срабатываний, lift к базовой доле фрода, потолок доли срабатываний, минимум TP.
# safe: доля фрода среди помеченных «безопасными» не выше порога.
RULE_SAFE_ZONE_MAX_FRAUD_RATE = 0.05
# Пороги для block (все условия должны выполняться одновременно):
RULE_MIN_PRECISION_ON_BLOCK = 0.10
RULE_MIN_LIFT_BLOCK = 1.15
RULE_MAX_FIRE_RATE_BLOCK = 0.50
RULE_MIN_TP_ON_BLOCK = 5
RULE_MIN_FIRE_COUNT_BLOCK = 15
# Дополнительно: изолированный Fβ не ниже этого — слабые по полноте/балансу режутся (β=PRIMARY_FBETA).
RULE_MIN_FBETA_ON_BLOCK = 0.03

# Признаки X из split_model_RF_new_features (1).ipynb — блок «needed_features» перед RandomForest.
# age_crypta_by_puid в оригинале закомментирован и не входит в модель.
# Поля, в имени которых есть «score», в модель не включаем (см. также drop в FeatureEngineer).
ORIGINAL_NOTEBOOK_MODEL_FEATURES = [
    "order_loan",
    "cookie_age",
    "card_age",
    "phone_age",
    "password_age",
    "puid_age",
    "puid_days_since_login_id_creation",
    "months_sum",
    "puid_card_age",
    "cart_item_category_fp_risk_category",
    "delivery_days",
    "glue_size",
    "glue_ead",
    "puid_card_ids_count",
    "cnt_split_paid",
    "sum_split_paid",
    "puid_months_with",
    "puid_months_with_taxiuber",
    "puid_days_since_first_transaction",
    "puid_orders_1h_without_refunds",
    "puid_orders_24h_without_refunds",
    "puid_orders_2d_without_refunds",
    "debt_on_card_id",
    "debt_on_cookie",
    "debt_on_crypta_id",
    "debt_on_delivery_phone",
    "debt_on_device_id",
    "debt_on_phone",
    "first_order_flg",
    "i_puid_karma",
    "i_browser_is_mobile",
    "is_yabank_card_owner",
    "phone_diff",
    "loan_base_limit_part",
    "loan_available_limit_part",
    "paid_sum_loan_part",
    "loan_1h_base_limit",
    "loan_24h_base_limit",
    "loan_2d_base_limit",
    "ages_diff_days",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_hour_of_day",
    "cos_hour_of_day",
]

# Служебные поля: цель, сплит, FE (в X не попадают через exclude в DataPreparer).
PIPELINE_META_COLUMNS = [
    "fraud_flg",
    "order_id",
    "order_creation_dttm",
    "merchant_id",
    "max_iso_eventtime_str",
    "split_type",
]

# Соответствие имён в CSV/YQL (с «market») → имена в пайплайне (без «market»).
_COLUMNS_RENAME_DROP_MARKET = {
    "market_glue_size": "glue_size",
    "market_glue_ead": "glue_ead",
    "puid_months_with_market": "puid_months_with",
    "npv_37_day_with_market_compensation": "npv_37_day_with_compensation",
}


def rename_columns_drop_market(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовывает колонки, в названии которых было слово market (как отдельный фрагмент имени)."""
    to_apply = {k: v for k, v in _COLUMNS_RENAME_DROP_MARKET.items() if k in df.columns}
    if to_apply:
        df = df.rename(columns=to_apply, copy=False)
    return df


def subset_dataframe_to_original_notebook_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только признаки из оригинального ноутбука и метаданные пайплайна."""
    df = rename_columns_drop_market(df)
    want = list(dict.fromkeys(ORIGINAL_NOTEBOOK_MODEL_FEATURES + PIPELINE_META_COLUMNS))
    want = [c for c in want if "score" not in c.lower()]
    present = [c for c in want if c in df.columns]
    missing = [c for c in ORIGINAL_NOTEBOOK_MODEL_FEATURES if c not in df.columns]
    if missing:
        print(
            f"  Внимание: отсутствуют {len(missing)} признаков из оригинального ноутбука "
            f"(первые 12): {missing[:12]}"
        )
    extra_dropped = len(df.columns) - len(present)
    if extra_dropped > 0:
        print(f"  Отброшено лишних колонок из CSV: {extra_dropped}")
    return df[present].copy()


def columns_present_model_features(df: pd.DataFrame) -> list[str]:
    """Имена признаков из оригинального ноутбука, которые есть в датафрейме."""
    return [c for c in ORIGINAL_NOTEBOOK_MODEL_FEATURES if c in df.columns]


def dataframe_for_eda(dfx: pd.DataFrame, target_col: str = "fraud_flg") -> pd.DataFrame:
    """
    Только модельные признаки и цель — без order_id, merchant_id, дат и прочих метаданных пайплайна.
    Используйте для EDA-графиков и корреляций.
    """
    cols = columns_present_model_features(dfx)
    if target_col in dfx.columns and target_col not in cols:
        cols = cols + [target_col]
    return dfx[cols].copy()


def _savefig(name: str) -> None:
    plt.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()


def plot_confusion_matrix_and_roc(
    y_true,
    y_prob,
    threshold: float,
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    """
    Матрица ошибок (test) и ROC-кривая с AUC для одной модели.
    Сохраняет PNG в out_dir (по умолчанию OUTPUT_DIR).
    """
    out = out_dir if out_dir is not None else OUTPUT_DIR
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Предсказано: OK", "Предсказано: fraud"],
        yticklabels=["Факт: OK", "Факт: fraud"],
    )
    axes[0].set_title(f"Матрица ошибок (test) — {model_name}\nпорог = {threshold:.3f}")
    axes[0].set_ylabel("Истина")
    axes[0].set_xlabel("Предсказание")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {auc_val:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"ROC (test) — {model_name}")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Диагностика: {model_name}", fontsize=12, y=1.02)
    plt.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    path = out / f"diagnostic_cm_roc_{safe}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()
    return path


def plot_fraud_distribution(dfx: pd.DataFrame) -> None:
    if "fraud_flg" not in dfx.columns:
        raise ValueError("plot_fraud_distribution: нет колонки fraud_flg")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28},
    )
    counts = dfx["fraud_flg"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(counts.index.astype(str), counts.values, color=colors)
    axes[0].set_title("Распределение классов (абсолютное)")
    axes[0].set_xlabel("Класс")
    axes[0].set_ylabel("Количество")
    vc = dfx["fraud_flg"].value_counts()
    pcts_arr = vc.values / vc.sum() * 100
    n_cls = len(vc)
    colors_use = (colors * ((n_cls + 1) // 2))[:n_cls]
    explode_use = tuple(0.08 if i == len(vc) - 1 else 0 for i in range(n_cls))
    legend_labels = [
        "Легитимные" if int(float(k)) == 0 else "Мошеннические" for k in vc.index
    ]
    wedges, _texts, autotexts = axes[1].pie(
        pcts_arr,
        labels=None,
        colors=colors_use,
        autopct="%1.2f%%",
        startangle=90,
        explode=explode_use,
        radius=1.05,
        pctdistance=0.72,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(11)
    axes[1].legend(
        wedges,
        legend_labels,
        title="Класс",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=11,
        frameon=True,
    )
    axes[1].set_title("Распределение классов (процентное)")
    axes[1].set_aspect("equal")
    plt.tight_layout()
    _savefig("01_class_distribution.png")


def plot_numeric_distributions(dfx: pd.DataFrame, target_col: str = "fraud_flg", n_cols: int = 4) -> None:
    dfx = dataframe_for_eda(dfx, target_col)
    numeric_cols = dfx.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col and dfx[c].nunique() > 2]
    key_features = [
        "order_loan",
        "cookie_age", "card_age", "phone_age", "puid_age",
        "puid_orders_24h_without_refunds", "debt_on_card_id",
        "loan_base_limit_part", "loan_available_limit_part",
    ]
    key_features = [f for f in key_features if f in numeric_cols]
    n_rows = (len(key_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, col in enumerate(key_features):
        ax = axes[idx]
        for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
            subset = dfx[dfx[target_col] == label][col].dropna()
            q_low, q_high = subset.quantile(0.01), subset.quantile(0.99)
            subset_clipped = subset[(subset >= q_low) & (subset <= q_high)]
            ax.hist(
                subset_clipped,
                bins=50,
                alpha=0.5,
                color=color,
                label=f'{"Fraud" if label else "Legit"}',
                density=True,
            )
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)
    for idx in range(len(key_features), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle("Распределение признаков по классам", fontsize=14, y=1.01)
    plt.tight_layout()
    _savefig("02_feature_distributions.png")


def plot_correlation_with_target(dfx: pd.DataFrame, target_col: str = "fraud_flg", top_n: int = 25):
    dfx = dataframe_for_eda(dfx, target_col)
    numeric_df = dfx.select_dtypes(include=[np.number])
    corr_with_target = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_corr = corr_with_target.head(top_n)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_corr)))
    ax.barh(range(len(top_corr)), top_corr.values, color=colors)
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index, fontsize=9)
    ax.set_xlabel("|Корреляция| с fraud_flg")
    ax.set_title(f"Топ-{top_n} признаков по абсолютной корреляции с fraud_flg")
    ax.invert_yaxis()
    plt.tight_layout()
    _savefig("03_correlation_with_target.png")
    return top_corr

class FeatureEngineer:
    def __init__(self):
        self.feature_names_created = []

    def parse_datetime_features(self, df):
        df = df.copy()
        for col in ["max_iso_eventtime_str", "order_creation_dttm"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
        if "order_creation_dttm" in df.columns and pd.api.types.is_datetime64_any_dtype(df["order_creation_dttm"]):
            dt = df["order_creation_dttm"]
            df["hour_of_day"] = dt.dt.hour
            df["day_of_week"] = dt.dt.dayofweek
            df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            df["is_night"] = ((dt.dt.hour >= 23) | (dt.dt.hour <= 5)).astype(int)
            df["is_business_hours"] = ((dt.dt.hour >= 9) & (dt.dt.hour <= 18)).astype(int)
            df["day_of_month"] = dt.dt.day
            df["is_month_start"] = (dt.dt.day <= 5).astype(int)
            df["is_month_end"] = (dt.dt.day >= 25).astype(int)
            self.feature_names_created.extend([
                "hour_of_day", "day_of_week", "is_weekend", "is_night",
                "is_business_hours", "day_of_month", "is_month_start", "is_month_end",
            ])
        return df

    def create_ratio_features(self, df):
        df = df.copy()
        if "order_loan" in df.columns and "loan_base_limit_part" in df.columns:
            df["loan_to_limit_ratio"] = df["order_loan"] / (df["loan_base_limit_part"] + 1e-8)
        debt_cols = [c for c in df.columns if c.startswith("debt_on_")]
        for col in debt_cols:
            ratio_name = f"{col}_to_loan_ratio"
            df[ratio_name] = df[col] / (df["order_loan"] + 1e-8)
            self.feature_names_created.append(ratio_name)
        if "order_loan" in df.columns and "sum_split_paid" in df.columns:
            df["current_to_historical_ratio"] = df["order_loan"] / (df["sum_split_paid"] + 1e-8)
            self.feature_names_created.append("current_to_historical_ratio")
        return df

    def create_age_features(self, df):
        df = df.copy()
        age_cols = ["cookie_age", "card_age", "phone_age", "password_age", "puid_age"]
        existing_age = [c for c in age_cols if c in df.columns]
        if len(existing_age) >= 2:
            df["min_age"] = df[existing_age].min(axis=1)
            df["max_age"] = df[existing_age].max(axis=1)
            df["mean_age"] = df[existing_age].mean(axis=1)
            df["age_range"] = df["max_age"] - df["min_age"]
            df["age_std"] = df[existing_age].std(axis=1)
            self.feature_names_created.extend([
                "min_age", "max_age", "mean_age", "age_range", "age_std",
            ])
        for col in existing_age:
            flag_name = f"{col}_is_new"
            if str(df[col].dtype) in ("float64", "int64", "Int64"):
                threshold = df[col].quantile(0.05)
                df[flag_name] = (df[col] <= threshold).astype(int)
                self.feature_names_created.append(flag_name)
        return df

    def create_velocity_features(self, df):
        df = df.copy()
        if "puid_orders_1h_without_refunds" in df.columns and "puid_orders_24h_without_refunds" in df.columns:
            df["velocity_1h_to_24h"] = (
                df["puid_orders_1h_without_refunds"] / (df["puid_orders_24h_without_refunds"] + 1e-8)
            )
            self.feature_names_created.append("velocity_1h_to_24h")
        if "puid_orders_24h_without_refunds" in df.columns and "puid_orders_2d_without_refunds" in df.columns:
            df["velocity_24h_to_2d"] = (
                df["puid_orders_24h_without_refunds"] / (df["puid_orders_2d_without_refunds"] + 1e-8)
            )
            self.feature_names_created.append("velocity_24h_to_2d")
        if "loan_1h_base_limit" in df.columns and "loan_24h_base_limit" in df.columns:
            df["loan_velocity_1h_24h"] = (
                df["loan_1h_base_limit"] / (df["loan_24h_base_limit"] + 1e-8)
            )
            self.feature_names_created.append("loan_velocity_1h_24h")
        return df

    def create_anomaly_flags(self, df):
        df = df.copy()
        if "first_order_flg" in df.columns and "order_loan" in df.columns:
            high_threshold = df["order_loan"].quantile(0.9)
            df["first_order_high_amount"] = (
                (df["first_order_flg"] == 1) & (df["order_loan"] > high_threshold)
            ).astype(int)
            self.feature_names_created.append("first_order_high_amount")
        if "phone_diff" in df.columns:
            df["phone_mismatch"] = (df["phone_diff"] != 0).astype(int)
            self.feature_names_created.append("phone_mismatch")
        debt_cols = [c for c in df.columns if c.startswith("debt_on_")]
        if len(debt_cols) >= 2:
            df["multi_debt_flag"] = ((df[debt_cols] > 0).sum(axis=1) >= 3).astype(int)
            self.feature_names_created.append("multi_debt_flag")
        if "is_night" in df.columns and "min_age" in df.columns:
            df["night_new_account"] = (
                (df.get("is_night", 0) == 1) & (df["min_age"] < df["min_age"].quantile(0.1))
            ).astype(int)
            self.feature_names_created.append("night_new_account")
        return df

    def fit_transform(self, df):
        print("  Парсинг временных признаков...")
        df = self.parse_datetime_features(df)
        print("  Создание отношений (ratio features)...")
        df = self.create_ratio_features(df)
        print("  Создание возрастных признаков...")
        df = self.create_age_features(df)
        print("  Создание velocity-признаков...")
        df = self.create_velocity_features(df)
        print("  Создание флагов аномалий...")
        df = self.create_anomaly_flags(df)
        drop_score = [c for c in df.columns if "score" in c.lower()]
        if drop_score:
            df = df.drop(columns=drop_score, errors="ignore")
            print(f"  Удалены колонки с «score» в имени: {drop_score}")
        print(f"\n  Всего создано новых признаков: {len(self.feature_names_created)}")
        return df

class DataPreparer:
    def __init__(
        self,
        target_col="fraud_flg",
        time_col="order_creation_dttm",
        test_size=0.2,
        val_size=0.15,
        feature_allowlist: list[str] | None = None,
    ):
        self.target_col = target_col
        self.time_col = time_col
        self.test_size = test_size
        self.val_size = val_size
        self.feature_allowlist = feature_allowlist
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.cat_columns = []
        self.num_columns = []

    def identify_columns(self, df):
        exclude_cols = [
            self.target_col, "order_id", "max_iso_eventtime_str",
            "order_creation_dttm", "split_type",
        ]
        self.cat_columns = []
        self.num_columns = []
        if self.feature_allowlist is not None:
            for col in self.feature_allowlist:
                if col not in df.columns or col in exclude_cols:
                    continue
                if df[col].dtype == "object" or (
                    df[col].nunique() < 20
                    and df[col].dtype in ["int64", "float64"]
                    and col.endswith("_flg")
                ):
                    if df[col].dtype == "object":
                        self.cat_columns.append(col)
                    else:
                        self.num_columns.append(col)
                else:
                    if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                        self.num_columns.append(col)
            self.feature_columns = self.num_columns + self.cat_columns
            print(f"  Числовых признаков (allowlist): {len(self.num_columns)}")
            print(f"  Категориальных признаков (allowlist): {len(self.cat_columns)}")
            print(f"  Всего признаков: {len(self.feature_columns)}")
            return

        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype == "object" or (
                df[col].nunique() < 20 and df[col].dtype in ["int64", "float64"] and col.endswith("_flg")
            ):
                if df[col].dtype == "object":
                    self.cat_columns.append(col)
                else:
                    self.num_columns.append(col)
            else:
                if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                    self.num_columns.append(col)
        self.feature_columns = self.num_columns + self.cat_columns
        print(f"  Числовых признаков: {len(self.num_columns)}")
        print(f"  Категориальных признаков: {len(self.cat_columns)}")
        print(f"  Всего признаков: {len(self.feature_columns)}")

    def encode_categoricals(self, df):
        df = df.copy()
        for col in self.cat_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].fillna("__MISSING__").astype(str)
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        return df

    def temporal_split(self, df):
        if self.time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            df_sorted = df.sort_values(self.time_col).reset_index(drop=True)
            n = len(df_sorted)
            train_end = int(n * (1 - self.test_size - self.val_size))
            val_end = int(n * (1 - self.test_size))
            df_train = df_sorted.iloc[:train_end]
            df_val = df_sorted.iloc[train_end:val_end]
            df_test = df_sorted.iloc[val_end:]
            print(f"  Хронологический сплит:")
            print(f"    Train: {len(df_train)} ({df_train[self.target_col].mean():.4%} fraud)")
            print(f"    Val:   {len(df_val)} ({df_val[self.target_col].mean():.4%} fraud)")
            print(f"    Test:  {len(df_test)} ({df_test[self.target_col].mean():.4%} fraud)")
            if pd.api.types.is_datetime64_any_dtype(df_sorted[self.time_col]):
                print(f"    Train period: {df_train[self.time_col].min()} — {df_train[self.time_col].max()}")
                print(f"    Val period:   {df_val[self.time_col].min()} — {df_val[self.time_col].max()}")
                print(f"    Test period:  {df_test[self.time_col].min()} — {df_test[self.time_col].max()}")
            return df_train, df_val, df_test
        print("  Временная колонка недоступна, используем стратифицированный сплит.")
        df_train_val, df_test = train_test_split(
            df, test_size=self.test_size, stratify=df[self.target_col], random_state=42
        )
        relative_val = self.val_size / (1 - self.test_size)
        df_train, df_val = train_test_split(
            df_train_val, test_size=relative_val,
            stratify=df_train_val[self.target_col], random_state=42,
        )
        print(f"  Стратифицированный сплит:")
        print(f"    Train: {len(df_train)} ({df_train[self.target_col].mean():.4%} fraud)")
        print(f"    Val:   {len(df_val)} ({df_val[self.target_col].mean():.4%} fraud)")
        print(f"    Test:  {len(df_test)} ({df_test[self.target_col].mean():.4%} fraud)")
        return df_train, df_val, df_test

    def handle_missing(self, X_train, X_val, X_test):
        imputer = SimpleImputer(strategy="median")
        num_cols_present = [c for c in self.num_columns if c in X_train.columns]
        X_train[num_cols_present] = imputer.fit_transform(X_train[num_cols_present])
        X_val[num_cols_present] = imputer.transform(X_val[num_cols_present])
        X_test[num_cols_present] = imputer.transform(X_test[num_cols_present])
        self.imputer = imputer
        return X_train, X_val, X_test

    def prepare(self, df):
        n0 = len(df)
        df = df.dropna(subset=[self.target_col]).copy()
        if len(df) < n0:
            print(f"  Удалено строк без целевой переменной ({self.target_col}): {n0 - len(df)}")
        print("  Идентификация столбцов...")
        self.identify_columns(df)
        print("  Кодирование категориальных переменных...")
        df = self.encode_categoricals(df)
        print("  Разделение на train/val/test...")
        df_train, df_val, df_test = self.temporal_split(df)
        feature_cols = [c for c in self.feature_columns if c in df_train.columns]
        self.feature_columns = feature_cols
        X_train = df_train[feature_cols].copy()
        y_train = df_train[self.target_col].copy()
        X_val = df_val[feature_cols].copy()
        y_val = df_val[self.target_col].copy()
        X_test = df_test[feature_cols].copy()
        y_test = df_test[self.target_col].copy()
        print("  Обработка пропусков...")
        X_train, X_val, X_test = self.handle_missing(X_train, X_val, X_test)
        for frame in [X_train, X_val, X_test]:
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            frame.fillna(0, inplace=True)

        def _sanitize_y(X, y, split_name):
            y = pd.to_numeric(y, errors="coerce")
            m = y.notna()
            if not m.all():
                print(f"  Удалено строк с нечисловым/пропущенным y ({split_name}): {(~m).sum()}")
            X = X.loc[m].reset_index(drop=True)
            y = y.loc[m].astype(np.int64).reset_index(drop=True)
            return X, y

        X_train, y_train = _sanitize_y(X_train, y_train, "train")
        X_val, y_val = _sanitize_y(X_val, y_val, "val")
        X_test, y_test = _sanitize_y(X_test, y_test, "test")
        return X_train, y_train, X_val, y_val, X_test, y_test


def _rule_condition_mask(condition_fn, X):
    mask = condition_fn(X)
    if isinstance(mask, pd.Series):
        mask = mask.values
    return np.asarray(mask, dtype=bool).ravel()


class RuleBasedFilter:
    def __init__(self):
        self.rules = []
        self.rule_stats = defaultdict(lambda: {"blocked": 0, "passed": 0, "safe": 0})

    def add_rule(self, name, condition_fn, action="block"):
        self.rules.append({"name": name, "condition": condition_fn, "action": action})

    def add_rule_if_passes_train(
        self,
        name,
        condition_fn,
        action,
        X_train,
        y_train,
        *,
        min_precision=None,
        min_lift=None,
        max_fire_rate=None,
        min_tp=None,
        min_fire_count=None,
        min_fbeta=None,
        safe_max_fraud=None,
    ):
        """
        Добавить правило только если на train оно проходит отбор.

        - block: среди срабатываний precision (доля фрода) ≥ min_precision; lift ≥ min_lift;
          доля срабатываний ≤ max_fire_rate; TP ≥ min_tp; число срабатываний ≥ min_fire_count;
          изолированный Fβ ≥ min_fbeta (доп. фильтр «не только шум»).
        - safe: среди срабатываний доля фрода ≤ safe_max_fraud.
        """
        min_precision = min_precision if min_precision is not None else RULE_MIN_PRECISION_ON_BLOCK
        min_lift = min_lift if min_lift is not None else RULE_MIN_LIFT_BLOCK
        max_fire_rate = max_fire_rate if max_fire_rate is not None else RULE_MAX_FIRE_RATE_BLOCK
        min_tp = min_tp if min_tp is not None else RULE_MIN_TP_ON_BLOCK
        min_fire_count = min_fire_count if min_fire_count is not None else RULE_MIN_FIRE_COUNT_BLOCK
        min_fbeta = min_fbeta if min_fbeta is not None else RULE_MIN_FBETA_ON_BLOCK
        safe_max_fraud = safe_max_fraud if safe_max_fraud is not None else RULE_SAFE_ZONE_MAX_FRAUD_RATE
        y_true = np.asarray(y_train).astype(int).ravel()
        if len(X_train) != len(y_true):
            print(f"  Правило '{name}': пропуск (разная длина X_train и y_train).")
            return False
        try:
            mask = _rule_condition_mask(condition_fn, X_train)
        except Exception as e:
            print(f"  Правило '{name}': пропуск (ошибка условия): {e}")
            return False
        if action == "block":
            n_fire = int(mask.sum())
            if n_fire == 0:
                print(f"  Правило '{name}' [block] отклонено: нет срабатываний на train.")
                return False
            y_fire = y_true[mask]
            tp = int(y_fire.sum())
            prec = tp / n_fire
            fire_rate = n_fire / len(y_true)
            base_rate = float(np.mean(y_true)) + 1e-12
            lift = prec / base_rate
            fb = float(
                fbeta_score(y_true, mask.astype(np.int32), beta=PRIMARY_FBETA, zero_division=0)
            )
            fail = []
            if n_fire < min_fire_count:
                fail.append(f"n={n_fire}<{min_fire_count}")
            if tp < min_tp:
                fail.append(f"TP={tp}<{min_tp}")
            if prec < min_precision:
                fail.append(f"prec={prec:.3f}<{min_precision}")
            if fire_rate > max_fire_rate:
                fail.append(f"fire={fire_rate:.1%}>{max_fire_rate:.0%}")
            if lift < min_lift:
                fail.append(f"lift={lift:.2f}<{min_lift}")
            if fb < min_fbeta:
                fail.append(f"Fβ={fb:.3f}<{min_fbeta}")
            if fail:
                print(
                    f"  Правило '{name}' [block] отклонено: {', '.join(fail)} "
                    f"(prec={prec:.3f}, lift={lift:.2f}, fire={fire_rate:.1%}, Fβ={fb:.4f})"
                )
                return False
            self.add_rule(name, condition_fn, action)
            print(
                f"  Правило '{name}' [block] добавлено: prec={prec:.3f}, lift={lift:.2f}, "
                f"fire={fire_rate:.1%}, TP={tp}, Fβ={fb:.4f}"
            )
            return True
        if action == "safe":
            if mask.sum() == 0:
                print(f"  Правило '{name}' [safe] отклонено: нет срабатываний на train.")
                return False
            fr = float(y_true[mask].mean())
            if fr <= safe_max_fraud:
                self.add_rule(name, condition_fn, action)
                print(
                    f"  Правило '{name}' [safe] добавлено: fraud в зоне safe={fr:.4%} (порог ≤ {safe_max_fraud:.2%})"
                )
                return True
            print(
                f"  Правило '{name}' [safe] отклонено: fraud в зоне safe={fr:.4%} > {safe_max_fraud:.2%}"
            )
            return False
        print(f"  Правило '{name}': неизвестное action={action!r}")
        return False

    def apply(self, X, return_details=False):
        n = len(X)
        rule_predictions = np.full(n, -1, dtype=int)
        triggered_rules = [[] for _ in range(n)]
        for rule in self.rules:
            try:
                mask = rule["condition"](X)
                if isinstance(mask, pd.Series):
                    mask = mask.values
                if rule["action"] == "block":
                    to_block = mask & (rule_predictions == -1)
                    rule_predictions[to_block] = 1
                    self.rule_stats[rule["name"]]["blocked"] += to_block.sum()
                elif rule["action"] == "safe":
                    to_safe = mask & (rule_predictions == -1)
                    rule_predictions[to_safe] = 0
                    self.rule_stats[rule["name"]]["safe"] += to_safe.sum()
                for i in np.where(mask)[0]:
                    triggered_rules[i].append(rule["name"])
            except Exception as e:
                print(f"  Ошибка в правиле '{rule['name']}': {e}")
        mask_grey = (rule_predictions == -1)
        self.rule_stats["__summary__"] = {
            "total": n,
            "blocked_by_rules": (rule_predictions == 1).sum(),
            "safe_by_rules": (rule_predictions == 0).sum(),
            "grey_zone": mask_grey.sum(),
        }
        if return_details:
            return mask_grey, rule_predictions, triggered_rules
        return mask_grey, rule_predictions

    def print_stats(self):
        summary = self.rule_stats.get("__summary__", {})
        print(f"\n  --- Статистика правил ---")
        print(f"  Всего транзакций: {summary.get('total', 0)}")
        print(f"  Заблокировано правилами: {summary.get('blocked_by_rules', 0)}")
        print(f"  Безопасно по правилам: {summary.get('safe_by_rules', 0)}")
        print(f"  Серая зона (→ ML): {summary.get('grey_zone', 0)}")
        for name, st in self.rule_stats.items():
            if name != "__summary__":
                total_triggered = st["blocked"] + st["safe"]
                if total_triggered > 0:
                    print(f"    Правило '{name}': blocked={st['blocked']}, safe={st['safe']}")


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
        cat_column_names=None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.feature_columns = feature_columns
        self.cat_column_names = list(cat_column_names or [])
        self.models = {}
        self.results = {}
        self.predictions = {}
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        self.scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"  scale_pos_weight = {self.scale_pos_weight:.2f}")

    def evaluate_model(self, name, y_true, y_prob, y_pred=None, threshold=0.5):
        if y_pred is None:
            y_pred = (y_prob >= threshold).astype(int)
        metrics = {
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "PR-AUC": average_precision_score(y_true, y_prob),
            "F1": f1_score(y_true, y_pred),
            "F2": fbeta_score(y_true, y_pred, beta=2),
            "F_beta": fbeta_score(y_true, y_pred, beta=PRIMARY_FBETA, zero_division=0),
            "F0.5": fbeta_score(y_true, y_pred, beta=0.5),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred),
            "Log Loss": log_loss(y_true, y_prob),
            "Threshold": threshold,
        }
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics["TP"] = tp
        metrics["FP"] = fp
        metrics["TN"] = tn
        metrics["FN"] = fn
        metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        return metrics

    def find_optimal_threshold(self, y_true, y_prob, metric=None, max_fpr=None, beta=None):
        if metric is None:
            metric = THRESHOLD_METRIC
        if beta is None:
            beta = PRIMARY_FBETA if metric == "fbeta" else (2.0 if metric == "f2" else 1.0)
        thresholds = np.arange(0.01, 0.99, 0.01)
        best_score = -1
        best_threshold = 0.5
        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
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
                best_threshold = th
        return best_threshold, best_score

    def _plot_after_train(self, model_name: str, y_prob_test, threshold: float) -> None:
        path = plot_confusion_matrix_and_roc(self.y_test, y_prob_test, threshold, model_name)
        print(f"  Диагностика (CM + ROC): {path}")

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
        best_th, best_sc = self.find_optimal_threshold(self.y_val, y_prob_val)
        print(
            f"  Оптимальный порог (val Fβ, β={PRIMARY_FBETA}): {best_th:.2f}, score={best_sc:.4f}"
        )
        metrics = self.evaluate_model("LogReg", self.y_test, y_prob_test, threshold=best_th)
        self.models["LogReg"] = {"model": model, "scaler": scaler}
        self.results["LogReg"] = metrics
        self.predictions["LogReg"] = y_prob_test
        print(
            f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f} | Fβ: {metrics['F_beta']:.4f}"
        )
        self._plot_after_train("LogReg", y_prob_test, best_th)
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
        best_th, best_sc = self.find_optimal_threshold(self.y_val, y_prob_val)
        print(
            f"  Оптимальный порог (val Fβ, β={PRIMARY_FBETA}): {best_th:.2f}, score={best_sc:.4f}"
        )
        metrics = self.evaluate_model("XGBoost", self.y_test, y_prob_test, threshold=best_th)
        self.models["XGBoost"] = model
        self.results["XGBoost"] = metrics
        self.predictions["XGBoost"] = y_prob_test
        print(
            f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f} | Fβ: {metrics['F_beta']:.4f}"
        )
        self._plot_after_train("XGBoost", y_prob_test, best_th)
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
        best_th, best_sc = self.find_optimal_threshold(self.y_val, y_prob_val)
        print(
            f"  Оптимальный порог (val Fβ, β={PRIMARY_FBETA}): {best_th:.2f}, score={best_sc:.4f}"
        )
        metrics = self.evaluate_model("LightGBM", self.y_test, y_prob_test, threshold=best_th)
        self.models["LightGBM"] = model
        self.results["LightGBM"] = metrics
        self.predictions["LightGBM"] = y_prob_test
        print(
            f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f} | Fβ: {metrics['F_beta']:.4f}"
        )
        self._plot_after_train("LightGBM", y_prob_test, best_th)
        return model

    def train_catboost(self):
        print("\n--- CatBoost ---")
        cat_indices = [i for i, col in enumerate(self.feature_columns) if col in self.cat_column_names]
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
        best_th, best_sc = self.find_optimal_threshold(self.y_val, y_prob_val)
        print(
            f"  Оптимальный порог (val Fβ, β={PRIMARY_FBETA}): {best_th:.2f}, score={best_sc:.4f}"
        )
        metrics = self.evaluate_model("CatBoost", self.y_test, y_prob_test, threshold=best_th)
        self.models["CatBoost"] = model
        self.results["CatBoost"] = metrics
        self.predictions["CatBoost"] = y_prob_test
        print(
            f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f} | Fβ: {metrics['F_beta']:.4f}"
        )
        self._plot_after_train("CatBoost", y_prob_test, best_th)
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
        best_th, best_sc = self.find_optimal_threshold(self.y_val, y_prob_val)
        print(
            f"  Оптимальный порог (val Fβ, β={PRIMARY_FBETA}): {best_th:.2f}, score={best_sc:.4f}"
        )
        metrics = self.evaluate_model("XGBoost+SMOTE", self.y_test, y_prob_test, threshold=best_th)
        self.models["XGBoost+SMOTE"] = model
        self.results["XGBoost+SMOTE"] = metrics
        self.predictions["XGBoost+SMOTE"] = y_prob_test
        print(
            f"  Test ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f} | Fβ: {metrics['F_beta']:.4f}"
        )
        self._plot_after_train("XGBoost+SMOTE", y_prob_test, best_th)
        return model

    def train_all(self):
        self.train_logistic_regression()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_catboost()
        self.train_xgboost_with_smote()


def fbeta_test_with_val_threshold(trainer, y_val, val_prob, y_test, test_prob):
    """F_beta на test при пороге, оптимизированном на val (как для основных моделей)."""
    th, _ = trainer.find_optimal_threshold(y_val, val_prob)
    y_pred = (test_prob >= th).astype(int)
    return fbeta_score(y_test, y_pred, beta=PRIMARY_FBETA, zero_division=0)


class EnsembleStacker:
    def __init__(self, base_model_predictions, y_true):
        self.base_predictions = base_model_predictions
        self.y_true = y_true
        self.stacking_model = None

    def simple_average(self):
        preds = np.column_stack(list(self.base_predictions.values()))
        avg_pred = preds.mean(axis=1)
        auc = roc_auc_score(self.y_true, avg_pred)
        pr_auc = average_precision_score(self.y_true, avg_pred)
        print(f"  Simple Average — ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        return avg_pred, auc, pr_auc

    def weighted_average(self, weights=None):
        names = list(self.base_predictions.keys())
        preds = np.column_stack(list(self.base_predictions.values()))
        if weights is None:
            weights = np.array(
                [average_precision_score(self.y_true, self.base_predictions[n]) for n in names]
            )
            weights = weights / weights.sum()
        print(f"  Веса: {dict(zip(names, weights.round(3)))}")
        weighted_pred = preds @ weights
        auc = roc_auc_score(self.y_true, weighted_pred)
        pr_auc = average_precision_score(self.y_true, weighted_pred)
        print(f"  Weighted Average — ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        return weighted_pred, auc, pr_auc

    def rank_average(self):
        from scipy.stats import rankdata

        preds = [rankdata(pred) / len(pred) for pred in self.base_predictions.values()]
        avg_ranks = np.mean(preds, axis=0)
        auc = roc_auc_score(self.y_true, avg_ranks)
        pr_auc = average_precision_score(self.y_true, avg_ranks)
        print(f"  Rank Average — ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        return avg_ranks, auc, pr_auc

    def stacking_with_logreg(self, X_val_preds, y_val, X_test_preds, y_test):
        print("\n  Стекинг (LogReg на мета-признаках):")
        meta_model = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
        meta_model.fit(X_val_preds, y_val)
        y_prob_stacked = meta_model.predict_proba(X_test_preds)[:, 1]
        auc = roc_auc_score(y_test, y_prob_stacked)
        pr_auc = average_precision_score(y_test, y_prob_stacked)
        print(f"  Stacking LogReg — ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        self.stacking_model = meta_model
        return y_prob_stacked, auc, pr_auc


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # noqa: A001

if TORCH_AVAILABLE:

    class FraudAutoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, encoding_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    def train_autoencoder(X_train, y_train, X_test, y_test, epochs=30, batch_size=512):
        print("\n--- Автоэнкодер (Anomaly Detection) ---")
        scaler = StandardScaler()
        X_train_normal = X_train[y_train == 0].values
        X_train_scaled = scaler.fit_transform(X_train_normal)
        X_test_scaled = scaler.transform(X_test.values)
        train_tensor = torch.FloatTensor(X_train_scaled)
        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True
        )
        input_dim = X_train_scaled.shape[1]
        model = FraudAutoencoder(input_dim, encoding_dim=32)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step(total_loss / max(len(train_loader), 1))
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test_scaled)
            reconstructed = model(test_tensor)
            recon_errors = torch.mean((test_tensor - reconstructed) ** 2, dim=1).numpy()
        auc = roc_auc_score(y_test, recon_errors)
        pr_auc = average_precision_score(y_test, recon_errors)
        print(f"  Autoencoder ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        return model, scaler, recon_errors, auc, pr_auc

    class FraudClassifierNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.network(x)

    def train_nn_classifier(X_train, y_train, X_val, y_val, X_test, y_test, epochs=50, batch_size=512, lr=1e-3):
        print("\n--- Нейросетевой классификатор ---")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train.values)
        X_va = scaler.transform(X_val.values)
        X_te = scaler.transform(X_test.values)
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_train.values))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_tensor_x = torch.FloatTensor(X_va)
        val_tensor_y = torch.FloatTensor(y_val.values)
        input_dim = X_tr.shape[1]
        model = FraudClassifierNN(input_dim)
        pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x).squeeze()
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            with torch.no_grad():
                val_logits = model(val_tensor_x).squeeze()
                val_probs = torch.sigmoid(val_logits).numpy()
                val_auc = roc_auc_score(y_val, val_probs)
            scheduler.step(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
            if patience_counter >= 10:
                print(f"    Early stopping at epoch {epoch + 1}")
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_te)
            y_prob_test = torch.sigmoid(model(test_tensor).squeeze()).numpy()
        auc = roc_auc_score(y_test, y_prob_test)
        pr_auc = average_precision_score(y_test, y_prob_test)
        print(f"  NN Classifier ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        return model, scaler, y_prob_test, auc, pr_auc


class ModelInterpreter:
    def __init__(
        self,
        model,
        X_train,
        X_test,
        y_test,
        feature_names,
        model_name="Model",
        cat_column_names: list[str] | None = None,
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.model_name = model_name
        self.cat_column_names = list(cat_column_names or [])
        self.shap_values = None
        self.explain_idx = None
        self.X_explain = None

    def compute_shap_values(self, n_background=500, n_explain=800):
        print(f"\n  Вычисление SHAP для {self.model_name}...")
        cols = list(self.feature_names)
        bg_idx = np.random.choice(len(self.X_train), min(n_background, len(self.X_train)), replace=False)
        explain_idx = np.random.choice(len(self.X_test), min(n_explain, len(self.X_test)), replace=False)
        X_bg = self.X_train.iloc[bg_idx][cols].copy()
        X_explain = self.X_test.iloc[explain_idx][cols].copy()
        try:
            # CatBoost: нужен Pool с теми же cat_features, что при обучении; иначе «Feature … is not present».
            if self.model_name == "CatBoost":
                from catboost import Pool

                cat_idx = [i for i, c in enumerate(cols) if c in self.cat_column_names]
                pool_kw = {"cat_features": cat_idx} if cat_idx else {}
                bg_pool = Pool(X_bg, **pool_kw)
                ex_pool = Pool(X_explain, **pool_kw)
                explainer = shap.TreeExplainer(self.model, bg_pool)
                self.shap_values = explainer.shap_values(ex_pool, check_additivity=False)
            else:
                # LightGBM/XGBoost: иногда не сходится additivity (округление, missing). Для отчётов SHAP допустимо.
                explainer = shap.TreeExplainer(self.model, X_bg)
                self.shap_values = explainer.shap_values(X_explain, check_additivity=False)
            self.X_explain = X_explain
            self.explain_idx = explain_idx
            print(f"  SHAP shape: {np.array(self.shap_values).shape}")
        except Exception as e:
            print(f"  SHAP TreeExplainer: {e}")

    def plot_shap_summary(self):
        if self.shap_values is None:
            return
        sv = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values
        plt.figure(figsize=(12, 10))
        shap.summary_plot(sv, self.X_explain, feature_names=self.feature_names, show=False, max_display=25)
        plt.title(f"SHAP Summary — {self.model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"04_shap_summary_{self.model_name}.png", dpi=150, bbox_inches="tight")
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
            else:
                plt.close()
        except Exception:
            plt.close()

    def plot_shap_bar(self):
        if self.shap_values is None:
            return None
        sv = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values
        mean_abs_shap = np.abs(sv).mean(axis=0)
        feature_importance = pd.Series(mean_abs_shap, index=self.feature_names).sort_values(ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(10, 8))
        feature_importance.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"SHAP — {self.model_name}")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"05_shap_bar_{self.model_name}.png", dpi=150, bbox_inches="tight")
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
            else:
                plt.close()
        except Exception:
            plt.close()
        return feature_importance

    def plot_native_importance(self):
        try:
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
            elif hasattr(self.model, "get_feature_importance"):
                importances = self.model.get_feature_importance()
            else:
                return None
            fi = pd.Series(importances, index=self.feature_names).sort_values(ascending=False).head(25)
            fig, ax = plt.subplots(figsize=(10, 8))
            fi.plot(kind="barh", ax=ax, color="darkorange")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"06_native_importance_{self.model_name}.png", dpi=150, bbox_inches="tight")
            try:
                from IPython import get_ipython
                if get_ipython() is not None:
                    plt.show()
                else:
                    plt.close()
            except Exception:
                plt.close()
            return fi
        except Exception as e:
            print(f"  Native importance: {e}")
            return None

class DriftMonitor:
    def __init__(self, reference_data, reference_scores, reference_labels, feature_names, alert_thresholds=None):
        self.reference_data = reference_data
        self.reference_scores = reference_scores
        self.reference_labels = reference_labels
        self.feature_names = feature_names
        self.drift_history = []
        # KS при больших выборках почти всегда «значим» — комбинируем с PSI.
        # Val vs test — разные сплиты: операционные метрики по умолчанию не сравниваем.
        self.alert_thresholds = alert_thresholds or {
            "psi_threshold": 0.2,
            "ks_pvalue": 0.001,
            "min_psi_with_ks": 0.05,
            "js_threshold": 0.15,
            "min_psi_score_drift": 0.08,
            "precision_drop": 0.12,
            "recall_drop": 0.12,
            "fpr_increase": 0.04,
            "critical_feature_fraction": 0.45,
        }

    def update_baseline(self, reference_data, reference_scores, reference_labels=None):
        """
        Обновить эталон (обычно после дообучения и смены модели).
        Иначе reference остаётся от старой модели и мониторинг бесконечно даёт RETRAIN.
        """
        self.reference_data = reference_data.copy() if hasattr(reference_data, "copy") else reference_data
        self.reference_scores = np.asarray(reference_scores, dtype=float).ravel()
        if reference_labels is not None:
            self.reference_labels = np.asarray(reference_labels).ravel()

    def compute_psi(self, expected, actual, n_bins=10):
        breakpoints = np.quantile(expected[~np.isnan(expected)], np.linspace(0, 1, n_bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        expected_pct = (expected_counts + 1) / (expected_counts.sum() + len(expected_counts))
        actual_pct = (actual_counts + 1) / (actual_counts.sum() + len(actual_counts))
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    def compute_feature_drift(self, current_data):
        drift_report = {}
        for col in self.feature_names:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue
            if current_data[col].dtype not in ["float64", "int64", "float32", "int32"]:
                continue
            ref_vals = self.reference_data[col].dropna().values
            cur_vals = current_data[col].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue
            ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)
            psi = self.compute_psi(ref_vals, cur_vals)
            drift_report[col] = {
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "psi": psi,
                "drifted_ks": ks_pvalue < self.alert_thresholds["ks_pvalue"],
                "drifted_psi": psi > self.alert_thresholds["psi_threshold"],
            }
        return drift_report

    def _feature_is_drifted(self, d):
        """KS один даёт ложные срабатывания на больших n; требуем ещё заметный PSI."""
        th = self.alert_thresholds
        if d["drifted_psi"]:
            return True
        if d["drifted_ks"] and d["psi"] >= th.get("min_psi_with_ks", 0.05):
            return True
        return False

    def compute_score_drift(self, current_scores):
        bins = np.linspace(0, 1, 51)
        ref_hist, _ = np.histogram(self.reference_scores, bins=bins, density=True)
        cur_hist, _ = np.histogram(current_scores, bins=bins, density=True)
        ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
        cur_hist = cur_hist / (cur_hist.sum() + 1e-10)
        js_div = jensenshannon(ref_hist, cur_hist)
        psi = self.compute_psi(self.reference_scores, current_scores)
        ks_stat, ks_pvalue = stats.ks_2samp(self.reference_scores, current_scores)
        th = self.alert_thresholds
        js_t = th["js_threshold"]
        psi_t = th["psi_threshold"]
        min_psi = th.get("min_psi_score_drift", 0.08)
        score_drifted = (js_div > js_t and psi > min_psi) or (psi > psi_t)
        return {
            "js_divergence": js_div,
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "score_drifted": score_drifted,
        }

    def compute_operational_metrics(self, y_true, y_prob, threshold=0.5):
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        }

    def full_monitoring_check(
        self,
        current_data,
        current_scores,
        current_labels=None,
        threshold=0.5,
        *,
        alert_on_operational_shift=False,
    ):
        """
        alert_on_operational_shift: если False (по умолчанию), не сравниваем precision/recall/FPR
        между reference и current — при эталоне=val и current=test это разные выборки и ложные тревоги.
        Включайте True только когда reference и current — сопоставимые периоды/потоки.
        """
        report = {"timestamp": datetime.now().isoformat(), "n_transactions": len(current_data), "alerts": []}
        feature_drift = self.compute_feature_drift(current_data)
        drifted_features = [f for f, d in feature_drift.items() if self._feature_is_drifted(d)]
        drifted_sorted = sorted(drifted_features, key=lambda f: feature_drift[f]["psi"], reverse=True)
        top_plot = drifted_sorted[:15]
        report["feature_drift"] = {
            "n_drifted": len(drifted_features),
            "drifted_features": drifted_sorted[:10],
            "details": {f: feature_drift[f] for f in top_plot},
        }
        frac_crit = self.alert_thresholds.get("critical_feature_fraction", 0.45)
        if len(self.feature_names) > 0 and len(drifted_features) > len(self.feature_names) * frac_crit:
            report["alerts"].append(f"CRITICAL: дрейф по многим признакам ({len(drifted_features)})")
        score_drift = self.compute_score_drift(current_scores)
        report["score_drift"] = score_drift
        if score_drift["score_drifted"]:
            report["alerts"].append(f"WARNING: дрейф скоров (JS={score_drift['js_divergence']:.4f})")
        if current_labels is not None and alert_on_operational_shift:
            ops = self.compute_operational_metrics(current_labels, current_scores, threshold)
            ref_ops = self.compute_operational_metrics(self.reference_labels, self.reference_scores, threshold)
            report["operational_metrics"] = ops
            report["reference_metrics"] = ref_ops
            if ref_ops["precision"] - ops["precision"] > self.alert_thresholds["precision_drop"]:
                report["alerts"].append("WARNING: падение Precision")
            if ref_ops["recall"] - ops["recall"] > self.alert_thresholds["recall_drop"]:
                report["alerts"].append("WARNING: падение Recall")
            if ops["fpr"] - ref_ops["fpr"] > self.alert_thresholds["fpr_increase"]:
                report["alerts"].append("WARNING: рост FPR")
        elif current_labels is not None:
            ops = self.compute_operational_metrics(current_labels, current_scores, threshold)
            report["operational_metrics"] = ops
            report["reference_metrics"] = self.compute_operational_metrics(
                self.reference_labels, self.reference_scores, threshold
            )
        n_alerts = len(report["alerts"])
        critical = any(a.startswith("CRITICAL") for a in report["alerts"])
        if n_alerts == 0:
            report["recommendation"] = "OK"
        elif n_alerts == 1:
            report["recommendation"] = "MONITOR_CLOSELY"
        elif n_alerts == 2 and not (critical and score_drift["score_drifted"]):
            report["recommendation"] = "MONITOR_CLOSELY"
        else:
            report["recommendation"] = "RETRAIN"
        self.drift_history.append(report)
        return report

    def plot_drift_report(self, report, current_scores=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fd = report.get("feature_drift", {}).get("details", {})
        if fd:
            names = list(fd.keys())
            psi_vals = [fd[f]["psi"] for f in names]
            th_psi = self.alert_thresholds.get("psi_threshold", 0.2)
            colors = [
                "#e74c3c" if p > th_psi else "#f39c12" if p > th_psi * 0.5 else "#2ecc71" for p in psi_vals
            ]
            axes[0].barh(names, psi_vals, color=colors)
            axes[0].invert_yaxis()
            axes[0].set_xlabel("PSI")
            axes[0].set_title("PSI по признакам (топ по дрейфу)")
            axes[0].axvline(th_psi, color="gray", linestyle="--", alpha=0.7, label=f"порог {th_psi}")
            axes[0].legend(loc="lower right", fontsize=7)
        else:
            axes[0].text(0.5, 0.5, "Нет данных", ha="center")
        bins = np.linspace(0, 1, 51)
        ref = np.asarray(self.reference_scores, dtype=float).ravel()
        axes[1].hist(ref, bins=bins, alpha=0.55, label="Reference (эталон)", density=True, color="steelblue")
        if current_scores is not None:
            cur = np.asarray(current_scores, dtype=float).ravel()
            axes[1].hist(cur, bins=bins, alpha=0.45, label="Current (текущие)", density=True, color="darkorange")
        sd = report.get("score_drift", {})
        js_s = sd.get("js_divergence")
        psi_s = sd.get("psi")
        sub = []
        if js_s is not None:
            sub.append(f"JS={js_s:.4f}")
        if psi_s is not None:
            sub.append(f"PSI={psi_s:.4f}")
        axes[1].set_title("Распределение скоров" + (f" ({', '.join(sub)})" if sub else ""))
        axes[1].set_xlabel("score")
        axes[1].legend()
        alert_text = report.get("recommendation", "") + "\n" + "\n".join(report.get("alerts", []))
        axes[2].text(0.05, 0.95, alert_text or "OK", transform=axes[2].transAxes, va="top", fontsize=9, family="monospace")
        axes[2].axis("off")
        plt.suptitle("Мониторинг дрейфа", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "08_drift_monitoring.png", dpi=150, bbox_inches="tight")
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
            else:
                plt.close()
        except Exception:
            plt.close()

class CombinedAntiFraudPipeline:
    def __init__(self, rule_filter, ml_model, threshold=0.5):
        self.rule_filter = rule_filter
        self.ml_model = ml_model
        self.threshold = threshold

    def predict(self, X):
        n = len(X)
        final_predictions = np.zeros(n, dtype=int)
        final_scores = np.zeros(n, dtype=float)
        decision_source = np.array([""] * n, dtype=object)
        mask_grey, rule_preds = self.rule_filter.apply(X)
        blocked = rule_preds == 1
        safe = rule_preds == 0
        final_predictions[blocked] = 1
        final_scores[blocked] = 1.0
        decision_source[blocked] = "rule_block"
        final_predictions[safe] = 0
        final_scores[safe] = 0.0
        decision_source[safe] = "rule_safe"
        grey_idx = np.where(mask_grey)[0]
        if len(grey_idx) > 0:
            X_grey = X.iloc[grey_idx]
            if hasattr(self.ml_model, "predict_proba"):
                ml_scores = self.ml_model.predict_proba(X_grey)[:, 1]
            else:
                ml_scores = self.ml_model.predict(X_grey)
            final_scores[grey_idx] = ml_scores
            final_predictions[grey_idx] = (ml_scores >= self.threshold).astype(int)
            decision_source[grey_idx] = np.where(ml_scores >= self.threshold, "ml_block", "ml_pass")
        return final_predictions, final_scores, decision_source

    def evaluate(self, X, y_true):
        final_preds, final_scores, sources = self.predict(X)
        print(f"\n  Распределение решений:")
        unique, counts = np.unique(sources, return_counts=True)
        for src, cnt in zip(unique, counts):
            m = sources == src
            fr = y_true[m].mean() if m.any() else 0
            print(f"    {src}: {cnt} ({cnt / len(X):.2%}), fraud rate: {fr:.4%}")
        auc = roc_auc_score(y_true, final_scores)
        pr_auc = average_precision_score(y_true, final_scores)
        f1 = f1_score(y_true, final_preds)
        f_beta = fbeta_score(y_true, final_preds, beta=PRIMARY_FBETA, zero_division=0)
        f2 = fbeta_score(y_true, final_preds, beta=2, zero_division=0)
        f05 = fbeta_score(y_true, final_preds, beta=0.5, zero_division=0)
        precision = precision_score(y_true, final_preds, zero_division=0)
        recall = recall_score(y_true, final_preds)
        cm = confusion_matrix(y_true, final_preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(
            f"\n  ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Fβ (β={PRIMARY_FBETA}): {f_beta:.4f} | F1: {f1:.4f}"
        )
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | FPR: {fpr:.4f}")
        return {
            "ROC-AUC": auc,
            "PR-AUC": pr_auc,
            "F1": f1,
            "F2": f2,
            "F_beta": f_beta,
            "F0.5": f05,
            "Precision": precision,
            "Recall": recall,
            "FPR": fpr,
            "Threshold": self.threshold,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        }

def plot_comprehensive_comparison(results, predictions, y_test):
    metrics_df = pd.DataFrame(results).T
    display_cols = ["ROC-AUC", "PR-AUC", "F_beta", "F1", "F2", "Precision", "Recall", "FPR"]
    available_cols = [c for c in display_cols if c in metrics_df.columns]
    print("\n--- Сравнительная таблица метрик ---")
    print(metrics_df[available_cols].round(4).to_string())
    metrics_df[available_cols].round(4).to_csv(OUTPUT_DIR / "model_comparison_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for name, y_prob in predictions.items():
        if y_prob is None or name == "Autoencoder":
            continue
        try:
            fpr_vals, tpr_vals, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            axes[0].plot(fpr_vals, tpr_vals, label=f"{name} (AUC={auc_val:.4f})")
        except Exception:
            pass
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC")
    axes[0].legend(fontsize=7, loc="lower right")
    for name, y_prob in predictions.items():
        if y_prob is None or name == "Autoencoder":
            continue
        try:
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            axes[1].plot(rec, prec, label=f"{name} (AP={ap:.4f})")
        except Exception:
            pass
    axes[1].axhline(y=y_test.mean(), color="k", linestyle="--", alpha=0.3)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR")
    axes[1].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()

    model_names = [n for n in results if results[n].get("ROC-AUC") is not None]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fbeta_label = f"Fβ (β={PRIMARY_FBETA})"
    for ax, metric in zip(axes, ["ROC-AUC", "PR-AUC", fbeta_label]):
        key = "F_beta" if metric == fbeta_label else metric
        vals = [results[n].get(key) or 0 for n in model_names]
        if metric in (fbeta_label, "F1"):
            vals = [v or 0 for v in vals]
        ax.bar(range(len(model_names)), vals, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names))))
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(metric)
    plt.suptitle("Сравнение моделей", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_model_comparison_bars.png", dpi=150, bbox_inches="tight")
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()

    names_with_probs = [n for n in model_names if n in predictions and predictions[n] is not None]
    top3 = sorted(
        [(n, results[n].get("F_beta", 0) or 0) for n in names_with_probs],
        key=lambda x: x[1],
        reverse=True,
    )[:3]
    if top3:
        n_cm = len(top3)
        fig, axes = plt.subplots(1, n_cm, figsize=(6 * n_cm, 5))
        if n_cm == 1:
            axes = np.array([axes])
        for idx, (name, _) in enumerate(top3):
            th = results[name].get("Threshold", 0.5) or 0.5
            y_pred = (predictions[name] >= th).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
            )
            axes[idx].set_title(f"{name}\n(th={th:.2f})")
        plt.suptitle(f"Confusion matrices (top-3 by Fβ, β={PRIMARY_FBETA})", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "11_confusion_matrices.png", dpi=150, bbox_inches="tight")
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
            else:
                plt.close()
        except Exception:
            plt.close()
    return metrics_df

class BusinessMetricsAnalyzer:
    def __init__(self, avg_fraud_amount=50000, review_cost=500, false_block_cost=2000):
        self.avg_fraud_amount = avg_fraud_amount
        self.review_cost = review_cost
        self.false_block_cost = false_block_cost

    def compute_business_impact(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fraud_prevented = tp * self.avg_fraud_amount
        fraud_missed = fn * self.avg_fraud_amount
        review_costs = (tp + fp) * self.review_cost
        false_block_costs = fp * self.false_block_cost
        total_cost = fraud_missed + review_costs + false_block_costs
        total_saved = fraud_prevented - review_costs - false_block_costs
        return {
            "fraud_prevented_count": tp,
            "fraud_prevented_amount": fraud_prevented,
            "fraud_missed_count": fn,
            "fraud_missed_amount": fraud_missed,
            "false_blocks_count": fp,
            "false_block_costs": false_block_costs,
            "review_costs": review_costs,
            "total_costs": total_cost,
            "net_savings": total_saved,
            "roi": total_saved / max(total_cost, 1),
        }

    def threshold_sensitivity_analysis(self, y_true, y_prob):
        rows = []
        for th in np.arange(0.05, 0.95, 0.05):
            y_pred = (y_prob >= th).astype(int)
            biz = self.compute_business_impact(y_true, y_pred)
            biz["threshold"] = th
            biz["precision"] = precision_score(y_true, y_pred, zero_division=0)
            biz["recall"] = recall_score(y_true, y_pred)
            biz["f1"] = f1_score(y_true, y_pred)
            biz["f_beta"] = fbeta_score(y_true, y_pred, beta=PRIMARY_FBETA, zero_division=0)
            rows.append(biz)
        return pd.DataFrame(rows)

    def plot_threshold_analysis(self, analysis_df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        th = analysis_df["threshold"]

        axes[0, 0].plot(th, analysis_df["net_savings"] / 1e6, "b-o", markersize=4)
        axes[0, 0].set_title("Чистая экономия vs порог")
        axes[0, 0].set_xlabel("Порог классификации")
        axes[0, 0].set_ylabel("Чистая экономия, млн (усл. ед.)")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(th, analysis_df["precision"], label="Precision", color="C0")
        axes[0, 1].plot(th, analysis_df["recall"], label="Recall", color="C1")
        axes[0, 1].set_title("Precision и Recall vs порог")
        axes[0, 1].set_xlabel("Порог классификации")
        axes[0, 1].set_ylabel("Precision / Recall")
        axes[0, 1].legend(loc="best")
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(th, analysis_df["fraud_prevented_count"], label="TP (предотвращено)", color="C2")
        axes[1, 0].plot(th, analysis_df["fraud_missed_count"], label="FN (пропущено)", color="C3")
        axes[1, 0].set_title("Предотвращённые и пропущенные фроды vs порог")
        axes[1, 0].set_xlabel("Порог классификации")
        axes[1, 0].set_ylabel("Количество случаев")
        axes[1, 0].legend(loc="best")
        axes[1, 0].grid(True, alpha=0.3)

        fbeta_col = "f_beta" if "f_beta" in analysis_df.columns else "f1"
        axes[1, 1].plot(th, analysis_df[fbeta_col], "g-s", markersize=4)
        axes[1, 1].set_title(f"Fβ vs порог (β={PRIMARY_FBETA})")
        axes[1, 1].set_xlabel("Порог классификации")
        axes[1, 1].set_ylabel(f"Fβ (β={PRIMARY_FBETA})")
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Чувствительность к порогу (бизнес-метрики)", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "12_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
            else:
                plt.close()
        except Exception:
            plt.close()

def _calibrated_classifier_prefit(base_model, method: str = "isotonic"):
    """
    Калибровка уже обученной модели. В sklearn >= 1.6 `cv='prefit'` убран;
    используется FrozenEstimator (см. документацию CalibratedClassifierCV).
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    except ImportError:
        return CalibratedClassifierCV(base_model, cv="prefit", method=method)


def calibrate_and_evaluate(model, X_val, y_val, X_test, y_test, model_name="Model"):
    print(f"\n  Калибровка {model_name}...")
    calibrated = _calibrated_classifier_prefit(model, method="isotonic")
    calibrated.fit(X_val, y_val)
    y_prob_raw = model.predict_proba(X_test)[:, 1]
    y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for y_prob, label, ax_idx in [(y_prob_raw, "До", 0), (y_prob_cal, "После", 1)]:
        fraction_pos, mean_predicted = calibration_curve(y_test, y_prob, n_bins=15)
        axes[ax_idx].plot(mean_predicted, fraction_pos, "s-")
        axes[ax_idx].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[ax_idx].set_title(label)
    plt.suptitle(f"Калибровка — {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"13_calibration_{model_name}.png", dpi=150, bbox_inches="tight")
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()
    print(f"  Brier raw: {brier_score_loss(y_test, y_prob_raw):.6f} | cal: {brier_score_loss(y_test, y_prob_cal):.6f}")
    return calibrated, y_prob_cal

class RetrainingPipeline:
    def __init__(self, model_class, model_params, min_improvement=0.005):
        self.model_class = model_class
        self.model_params = model_params
        self.min_improvement = min_improvement
        self.model_history = []
        self.current_champion = None
        self.champion_metrics = None

    def set_champion(self, model, metrics):
        self.current_champion = model
        self.champion_metrics = metrics
        self.model_history.append({"timestamp": datetime.now().isoformat(), "metrics": metrics, "action": "champion"})

    def champion_challenger_test(self, challenger, X_test, y_test, trainer=None, y_val=None, X_val=None):
        chall_probs = challenger.predict_proba(X_test)[:, 1]
        champ_probs = self.current_champion.predict_proba(X_test)[:, 1]
        if trainer is not None and y_val is not None and X_val is not None:
            champ_val = self.current_champion.predict_proba(X_val)[:, 1]
            chall_val = challenger.predict_proba(X_val)[:, 1]
            th_c, _ = trainer.find_optimal_threshold(y_val, champ_val)
            th_d, _ = trainer.find_optimal_threshold(y_val, chall_val)
            fb_c = fbeta_score(
                y_test, (champ_probs >= th_c).astype(int), beta=PRIMARY_FBETA, zero_division=0
            )
            fb_d = fbeta_score(
                y_test, (chall_probs >= th_d).astype(int), beta=PRIMARY_FBETA, zero_division=0
            )
            improvement = fb_d - fb_c
            print(f"  Champion Fβ (β={PRIMARY_FBETA}): {fb_c:.4f} | Challenger: {fb_d:.4f} | Δ: {improvement:+.4f}")
            if improvement >= self.min_improvement:
                self.current_champion = challenger
                self.champion_metrics = {
                    "F_beta": fb_d,
                    "PR-AUC": average_precision_score(y_test, chall_probs),
                }
                return True
            print(
                f"  Чемпион не заменён: нужен прирост Fβ ≥ min_improvement={self.min_improvement} "
                f"(сейчас {improvement:+.4f}). Иначе смена модели шумит на тесте."
            )
            return False
        champ_pr = average_precision_score(y_test, champ_probs)
        chall_pr = average_precision_score(y_test, chall_probs)
        improvement = chall_pr - champ_pr
        print(f"  Champion PR-AUC: {champ_pr:.4f} | Challenger: {chall_pr:.4f} | Δ: {improvement:+.4f}")
        if improvement >= self.min_improvement:
            self.current_champion = challenger
            self.champion_metrics = {"PR-AUC": chall_pr}
            return True
        print(
            f"  Чемпион не заменён: прирост PR-AUC ({improvement:+.4f}) < min_improvement={self.min_improvement}."
        )
        return False

