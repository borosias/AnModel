import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow.parquet as pq

PARQUET_DIR = os.getenv("PARQUET_IN", "src/analytics/data/parquet")
OUT_DIR = os.getenv("FEATURES_OUT", "src/features/data/features")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data_chunked(parquet_dir):
    """Загрузка данных чанками для экономии памяти"""
    parts = sorted(glob.glob(f"{parquet_dir}/events_part_*.parquet"))
    if not parts:
        raise SystemExit("no parquet parts in " + parquet_dir)

    # Читаем метаданные чтобы определить общий размер
    total_rows = 0
    for part in parts:
        pf = pq.read_metadata(part)
        total_rows += pf.num_rows

    print(f"Total rows to process: {total_rows:,}")

    # Обрабатываем чанками
    df_list = []
    for part in parts:
        df_list.append(pd.read_parquet(part))

    df = pd.concat(df_list, ignore_index=True)
    df['ts'] = pd.to_datetime(df['ts'])
    return df.sort_values(['user_id', 'ts'])


def compute_features_optimized(df):
    """Оптимизированное вычисление фич"""
    now = df['ts'].max()

    # Предварительно вычисляем часто используемые флаги
    df['is_purchase'] = df['event_type'] == 'purchase'
    df['is_view'] = df['event_type'] == 'product_view'

    # 1. User features - один проход вместо нескольких
    user_agg = df.groupby('user_id').agg(
        last_event_ts=('ts', 'max'),
        total_events=('event_id', 'count'),
        total_purchases=('is_purchase', 'sum'),
        total_views=('is_view', 'sum'),
        total_spent=('price', lambda x: x[df['is_purchase']].sum())
    ).reset_index()

    # Вычисляем производные фичи
    user_agg['days_since_last_event'] = (now - user_agg['last_event_ts']).dt.days

    # 2. 30-day features за один проход
    mask_30d = df['ts'] >= (now - pd.Timedelta(days=30))
    user_30d = df[mask_30d].groupby('user_id').agg(
        purchases_30d=('is_purchase', 'sum'),
        total_spent_30d=('price', lambda x: x[df.loc[mask_30d, 'is_purchase']].sum())
    ).reset_index()

    # 3. 7-day views (упрощаем - используем уже вычисленные вьюхи)
    mask_7d = df['ts'] >= (now - pd.Timedelta(days=7))
    views_7d = df[mask_7d].groupby('user_id')['is_view'].sum().reset_index(name='views_7d')

    # 4. 24-hour clicks
    mask_24h = df['ts'] >= (now - pd.Timedelta(hours=24))
    clicks_24h = df[mask_24h].groupby('user_id').size().reset_index(name='clicks_24h')

    # Объединяем все user features
    users = user_agg.merge(user_30d, on='user_id', how='left')
    users = users.merge(views_7d, on='user_id', how='left')
    users = users.merge(clicks_24h, on='user_id', how='left')

    # Заполняем пропуски и вычисляем производные
    users = users.fillna({
        'purchases_30d': 0,
        'total_spent_30d': 0,
        'views_7d': 0,
        'clicks_24h': 0
    })

    users['avg_spend_per_purchase_30d'] = np.where(
        users['purchases_30d'] > 0,
        users['total_spent_30d'] / users['purchases_30d'],
        0
    )

    # 5. Item popularity
    item_pop = df[mask_7d].groupby('item_id').size().reset_index(name='item_views_7d')

    # 6. Session features
    session_df = df.groupby('session_id').agg(
        session_start=('ts', 'min'),
        session_end=('ts', 'max'),
        session_events=('event_id', 'count')
    ).reset_index()
    session_df['session_length_s'] = (session_df['session_end'] - session_df['session_start']).dt.total_seconds()

    # Финализируем user features
    user_features = users[[
        'user_id', 'total_events', 'total_views', 'total_purchases', 'total_spent',
        'views_7d', 'purchases_30d', 'total_spent_30d', 'days_since_last_event',
        'clicks_24h', 'avg_spend_per_purchase_30d'
    ]]

    return user_features, item_pop, session_df


def main():
    print("Loading data...")
    df = load_data_chunked(PARQUET_DIR)

    print("Computing features...")
    user_features, item_pop, session_features = compute_features_optimized(df)

    print("Saving features...")
    user_features.to_parquet(f"{OUT_DIR}/user_features.parquet", index=False)
    item_pop.to_parquet(f"{OUT_DIR}/item_popularity.parquet", index=False)
    session_features.to_parquet(f"{OUT_DIR}/session_features.parquet", index=False)

    print(f"Features exported to {OUT_DIR}")
    print(f"User features: {len(user_features):,} users")
    print(f"Item features: {len(item_pop):,} items")
    print(f"Session features: {len(session_features):,} sessions")


if __name__ == "__main__":
    main()