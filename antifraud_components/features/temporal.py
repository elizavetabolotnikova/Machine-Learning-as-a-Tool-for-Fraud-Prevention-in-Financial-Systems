import pandas as pd
from antifraud_pipeline.antifraud_lib import FeatureEngineer

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer()
    return fe.add_temporal_features(df)
