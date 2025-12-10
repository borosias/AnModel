import pandas as pd
from src.models.models.context_aware import ContextAwareModel

model = ContextAwareModel.load("./src/models/production_models/context_aware_model1.pkl")

# Твои данные
data = {
    "total_events": 2000,
    "unique_days": 216,
    "total_clicks": 347,
    "total_purchases": 308,
    "total_spent": 457636.77,
    "distinct_items": 92,
    "events_last_7d": 189,
    "events_last_30d": 275,
    "purchases_last_30d": 47,
    "spent_last_30d": 60909.95,
    "conversion_rate_30d": 0.171,
    "avg_order_value_30d": 1295.96,
    "purchase_frequency": 1.426,
    "avg_spend_per_event": 228.82,
    "days_since_first": 353,
    "days_since_last": 0,
    "events_per_day": 9.26,
    "recency_score": 1.0,
    "last_event_type": "add_to_cart",
    "last_region": "UA-30",
    "last_item": "item_2",
    "trend_popularity_mean": 19.36,
    "trend_popularity_max": 75
}

df = pd.DataFrame([data])
print("Input:")
print(df)

result = model.predict(df)
print("\nPrediction:")
print(result)

# Проверим feature_columns
print("\nExpected features:")
print(model.feature_columns_.tolist())
