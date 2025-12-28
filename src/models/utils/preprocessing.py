# innovative_models/utils/preprocessing.py
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, LabelEncoder


class DataPreprocessor:
    """Data preprocessing utilities"""

    @staticmethod
    def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identifies numeric and categorical columns"""
        cols = [c for c in df.columns if not c.startswith("target_") and
                c not in ("snapshot_date", "user_id", "customer_id", "snapshot")]
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in cat if df[c].nunique() > 1 and df[c].nunique() <= 200]
        return num, cat

    @staticmethod
    def create_sequences(df: pd.DataFrame, context_length: int, prediction_horizon: int):
        """Creates sequences for time series"""
        sequences = []
        targets = []

        for trend_id in df['trend_id'].unique():
            trend_df = df[df['trend_id'] == trend_id].sort_values('snapshot_date')

            if len(trend_df) < context_length + prediction_horizon:
                continue

            feature_cols = [col for col in trend_df.columns
                            if not col.startswith('target_') and
                            col not in ['snapshot_date', 'trend_id', 'trend_type']]

            features = trend_df[feature_cols].values.astype(np.float32)

            target_cols = [col for col in trend_df.columns if col.startswith('target_')]
            if target_cols:
                trend_targets = trend_df[target_cols].values.astype(np.float32)
            else:
                trend_targets = features[:, -1:]

            for i in range(len(features) - context_length - prediction_horizon + 1):
                hist_seq = features[i:i + context_length]
                fut_seq = features[i + context_length:i + context_length + prediction_horizon]
                target_seq = trend_targets[i + context_length:i + context_length + prediction_horizon]

                sequences.append((hist_seq, fut_seq))
                targets.append(target_seq)

        return sequences, targets