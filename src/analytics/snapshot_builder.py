import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
import sys
from pathlib import Path

# Добавляем путь для импорта существующих фич
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_events_from_parquet() -> pd.DataFrame:
    """Загружает события из существующей папки parquet"""
    parquet_dir = Path("src/analytics/data/parquet")
    parquet_files = list(parquet_dir.glob("events_part_*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")

    print(f"Loading {len(parquet_files)} parquet files...")
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    events = pd.concat(dfs, ignore_index=True)
    events['ts'] = pd.to_datetime(events['ts'])
    return events.sort_values('ts')


def create_time_based_snapshots(events: pd.DataFrame,
                                train_end: datetime,
                                val_end: datetime,
                                test_end: datetime,
                                window_back_days: int = 30,
                                window_forward_days: int = 7) -> tuple:
    """
    Создает time-based snapshots с учетом существующих фич
    """

    # Определяем даты снэпшотов (раз в неделю для эффективности)
    min_date = events['ts'].min().ceil('D')
    max_date = events['ts'].max().floor('D')

    all_snapshot_dates = pd.date_range(
        start=min_date + timedelta(days=window_back_days),
        end=max_date - timedelta(days=window_forward_days),
        freq="7D"  # Weekly snapshots
    )

    # Разделяем на train/val/test
    train_snapshots = [d for d in all_snapshot_dates if d <= train_end]
    val_snapshots = [d for d in all_snapshot_dates if train_end < d <= val_end]
    test_snapshots = [d for d in all_snapshot_dates if val_end < d <= test_end]

    print(f"Total snapshots: {len(all_snapshot_dates)}")
    print(f"Train: {len(train_snapshots)}, Val: {len(val_snapshots)}, Test: {len(test_snapshots)}")

    # Создаем снэпшоты
    train_data = _build_snapshots(events, train_snapshots, window_back_days, window_forward_days)
    val_data = _build_snapshots(events, val_snapshots, window_back_days, window_forward_days)
    test_data = _build_snapshots(events, test_snapshots, window_back_days, window_forward_days)

    return train_data, val_data, test_data


def _build_snapshots(events: pd.DataFrame, snapshot_dates: list,
                     window_back_days: int, window_forward_days: int) -> pd.DataFrame:
    """Строит снэпшоты для списка дат"""

    snapshots = []

    for i, snapshot_date in enumerate(snapshot_dates):
        if i % 5 == 0:  # Прогресс
            print(f"  Snapshot {i + 1}/{len(snapshot_dates)}: {snapshot_date.date()}")

        # Временные окна
        feature_start = snapshot_date - timedelta(days=window_back_days)
        feature_end = snapshot_date
        target_start = snapshot_date
        target_end = snapshot_date + timedelta(days=window_forward_days)

        # Данные для фичей (только исторические)
        feature_events = events[
            (events['ts'] >= feature_start) &
            (events['ts'] < feature_end)
            ]

        # Вычисляем фичи (совместимо с существующим build_features.py)
        user_features = _compute_snapshot_features(feature_events, snapshot_date)

        if user_features.empty:
            continue

        # Таргеты (будущие события)
        target_events = events[
            (events['ts'] >= target_start) &
            (events['ts'] < target_end)
            ]
        targets = _compute_targets(target_events, user_features.index)

        # Объединяем
        snapshot = user_features.join(targets, how='left').fillna(0)
        snapshot['snapshot_date'] = snapshot_date
        snapshots.append(snapshot.reset_index())

    return pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()


def _compute_snapshot_features(events: pd.DataFrame, snapshot_date: datetime) -> pd.DataFrame:
    """Вычисляет фичи на момент снэпшота (совместимо с build_features.py)"""

    if events.empty:
        return pd.DataFrame()

    # Базовые фичи как в build_features.py
    user_features = events.groupby('user_id').agg({
        'event_id': 'count',
        'ts': 'max',
        'price': 'sum'
    }).rename(columns={
        'event_id': 'total_events',
        'ts': 'last_event_ts',
        'price': 'total_spent'
    })

    # Recency (аналогично days_since_last_event из build_features)
    user_features['days_since_last_event'] = (
            snapshot_date - user_features['last_event_ts']
    ).dt.days

    # Event type counts (аналогично is_purchase, is_view)
    event_counts = events.pivot_table(
        index='user_id',
        columns='event_type',
        values='event_id',
        aggfunc='count',
        fill_value=0
    )
    event_counts = event_counts.add_prefix('count_')
    user_features = user_features.join(event_counts, how='left')

    # Purchase-specific features
    purchases = events[events['event_type'] == 'purchase']
    if not purchases.empty:
        purchase_features = purchases.groupby('user_id').agg({
            'event_id': 'count',
            'price': ['sum', 'mean', 'std']
        })
        purchase_features.columns = [
            'purchase_count',
            'total_spent_purchases',
            'avg_purchase_value',
            'std_purchase_value'
        ]
        user_features = user_features.join(purchase_features, how='left')

    # Session features (совместимо с session_features.parquet)
    session_features = events.groupby(['user_id', 'session_id']).agg({
        'ts': ['min', 'max', 'count']
    }).reset_index()
    session_features.columns = ['user_id', 'session_id', 'session_start', 'session_end', 'session_events']

    user_session_features = session_features.groupby('user_id').agg({
        'session_events': ['mean', 'max', 'sum'],
        'session_id': 'count'
    })
    user_session_features.columns = [
        'avg_session_events',
        'max_session_events',
        'total_session_events',
        'total_sessions'
    ]
    user_features = user_features.join(user_session_features, how='left')

    # Заполняем пропуски и чистим
    user_features = user_features.fillna(0)
    user_features = user_features.drop(['last_event_ts'], axis=1, errors='ignore')

    return user_features


def _compute_targets(target_events: pd.DataFrame, user_index: pd.Index) -> pd.DataFrame:
    """Вычисляет таргеты - совместимо с существующей логикой"""

    # Основной таргет: покупка в целевом окне
    purchases = target_events[target_events['event_type'] == 'purchase']

    targets = pd.DataFrame(index=user_index)
    targets['target_purchase'] = 0
    targets['target_purchase_count'] = 0
    targets['target_spent'] = 0

    if not purchases.empty:
        purchase_stats = purchases.groupby('user_id').agg({
            'event_id': 'count',
            'price': 'sum'
        }).rename(columns={
            'event_id': 'target_purchase_count',
            'price': 'target_spent'
        })

        targets.loc[purchase_stats.index, 'target_purchase'] = 1
        targets.loc[purchase_stats.index, 'target_purchase_count'] = purchase_stats['target_purchase_count']
        targets.loc[purchase_stats.index, 'target_spent'] = purchase_stats['target_spent']

    return targets


def validate_and_save(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """Валидация и сохранение снэпшотов"""

    # Создаем директорию
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n=== Validation ===")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(df) > 0:
            pos_rate = df['target_purchase'].mean() * 100
            print(f"{name}: {len(df):,} samples, {pos_rate:.2f}% positive")

    # Сохраняем
    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test.parquet", index=False)

    print(f"\n=== Saved to {output_dir} ===")
    print(f"Files: train.parquet ({len(train_df):,} rows)")
    print(f"       val.parquet ({len(val_df):,} rows)")
    print(f"       test.parquet ({len(test_df):,} rows)")


def main():
    parser = argparse.ArgumentParser(description='Build time-based snapshots compatible with existing features')
    parser.add_argument('--train-end', type=str, required=True, help='Train end date (YYYY-MM-DD)')
    parser.add_argument('--val-end', type=str, required=True, help='Validation end date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, required=True, help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--window-back', type=int, default=30, help='Feature window in days')
    parser.add_argument('--window-forward', type=int, default=7, help='Target window in days')
    parser.add_argument('--output-dir', type=str, default='data/snapshots', help='Output directory')

    args = parser.parse_args()

    print("=== Time-based Snapshot Builder ===")
    print(f"Loading existing data from src/analytics/data/parquet/")

    # Загружаем данные
    events = load_events_from_parquet()
    print(f"Loaded {len(events):,} events from {events['ts'].min()} to {events['ts'].max()}")

    # Парсим даты
    train_end = pd.to_datetime(args.train_end)
    val_end = pd.to_datetime(args.val_end)
    test_end = pd.to_datetime(args.test_end)

    # Строим снэпшоты
    print("\nBuilding snapshots...")
    train_df, val_df, test_df = create_time_based_snapshots(
        events=events,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        window_back_days=args.window_back,
        window_forward_days=args.window_forward
    )

    # Сохраняем
    validate_and_save(train_df, val_df, test_df, args.output_dir)


if __name__ == "__main__":
    main()