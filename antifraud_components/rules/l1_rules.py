from __future__ import annotations
import pandas as pd
from antifraud_pipeline.antifraud_lib import RuleBasedFilter, RULE_MIN_PRECISION_ON_BLOCK, RULE_MIN_LIFT_BLOCK, RULE_MAX_FIRE_RATE_BLOCK, RULE_MIN_TP_ON_BLOCK, RULE_MIN_FIRE_COUNT_BLOCK, RULE_MIN_FBETA_ON_BLOCK

def fit_l1_rule_filter(X_train: pd.DataFrame, y_train: pd.Series) -> RuleBasedFilter:
    rule_filter = RuleBasedFilter()
    xs = X_train

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
    return rule_filter

def build_l1_rule_filter_from_df(df: pd.DataFrame) -> RuleBasedFilter:
    rule_filter = RuleBasedFilter()
    if 'puid_orders_1h_without_refunds' in df.columns and 'order_loan' in df.columns:
        tv = float(df['puid_orders_1h_without_refunds'].quantile(0.95))
        rule_filter.add_rule('extreme_velocity_1h', lambda X, _tv=tv: (X['puid_orders_1h_without_refunds'] > _tv) & (X['order_loan'] > 2000), 'block')
    if 'glue_size' in df.columns:
        gs = float(df['glue_size'].quantile(0.995))
        rule_filter.add_rule('glue_size_spike', lambda X, a=gs: X['glue_size'] >= a, 'block')
    if 'glue_ead' in df.columns:
        ge = float(df['glue_ead'].quantile(0.995))
        rule_filter.add_rule('glue_ead_spike', lambda X, a=ge: X['glue_ead'] >= a, 'block')
    return rule_filter
