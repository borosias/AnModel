import os

import pandas as pd
from src.models.models.context_aware import ContextAwareModel

model = ContextAwareModel.load("./src/models/production_models/context_aware_model1.pkl")

# Твои данные


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "src", "analytics", "data", "daily_features", "snapshot_2025_12_10")

def load_dataset(name: str) -> pd.DataFrame:
    path = os.path.join(SNAPSHOT_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)

df = load_dataset("daily_snapshot1")
print("Input:")
print(df)

result = model.predict(df)
out = pd.concat(
    [
        df[["user_id", 'total_events', 'unique_days', 'total_clicks', 'total_purchases', 'total_spent', 'distinct_items', 'events_last_7d', 'events_last_30d', 'purchases_last_30d', 'spent_last_30d', 'conversion_rate_30d', 'avg_order_value_30d', 'purchase_frequency', 'avg_spend_per_event', 'days_since_first', 'days_since_last', 'events_per_day', 'recency_score', 'last_event_type', 'last_region', 'last_item', 'trend_popularity_mean', 'trend_popularity_max']
],
        result
    ],
    axis=1
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(out.sort_values("purchase_proba", ascending=False))


# Проверим feature_columns
print("\nExpected features:")
print(model.feature_columns_.tolist())
