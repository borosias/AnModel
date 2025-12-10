import glob
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

EVENTS_DIR = "../data/parquet"
OUT_DIR = f"../data/daily_features/snapshot_{datetime.now().strftime('%Y_%m_%d')}"
TRENDS_PATH = "../trends_data/trends_master.parquet"

os.makedirs(OUT_DIR, exist_ok=True)


def _print_progress(current, total, prefix=""):
    if not total:
        return
    current = max(0, min(current, total))
    percent = (current / total) * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent:.0f}%)", end="", flush=True)
    if current >= total:
        print()


def _get_logger():
    logger = logging.getLogger("daily_snapshot_builder1")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    try:
        fh = logging.FileHandler(os.path.join(OUT_DIR, "daily_snapshot_builder1.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


def load_events(events_dir):
    logger = _get_logger()
    files = sorted(glob.glob(os.path.join(events_dir, "events_part_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {events_dir}")
    parts = []
    for i, fpath in enumerate(files, start=1):
        try:
            parts.append(pd.read_parquet(fpath))
        except Exception:
            pass
        _print_progress(i, len(files), prefix="Загрузка parquet файлов")

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date
    if 'price' in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    logger.info(f"Загружено {len(df)} событий ({df['date'].min()} - {df['date'].max()})")
    return df


def load_trends(trends_path: str = TRENDS_PATH):
    logger = _get_logger()
    if not os.path.exists(trends_path):
        logger.info("Тренды не найдены, пропускаем")
        return {}

    try:
        df = pd.read_parquet(trends_path)
    except Exception:
        return {}

    if df.empty or "date" not in df.columns or "popularity" not in df.columns:
        return {}

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

    daily = (
        df.dropna(subset=["popularity"])
        .groupby("date")["popularity"]
        .agg(trend_popularity_mean="mean", trend_popularity_max="max")
        .sort_index()
    )

    if daily.empty:
        return {}

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index).ffill()
    daily.index = daily.index.date

    logger.info(f"Загружены тренды для {len(daily)} дней")
    return daily.to_dict(orient="index")


def build_daily_snapshot(df, trends_dict=None):
    logger = _get_logger()
    if df.empty:
        return pd.DataFrame()

    # Строим снапшот на "сегодня" (последнюю доступную дату данных)
    target_date = df['date'].max()
    target_ts = pd.Timestamp(datetime.combine(target_date, datetime.max.time())).tz_localize('UTC')

    logger.info(f"Создаём снапшот за {target_date}")

    has_item = 'item_id' in df.columns
    has_region = 'region' in df.columns

    users = df['user_id'].unique()
    snapshots = []

    for idx, user_id in enumerate(users, start=1):
        # Берем всю историю пользователя
        user_events = df[df['user_id'] == user_id].sort_values('ts')
        if user_events.empty:
            continue

        # Фильтруем события до целевой даты (включительно)
        # В данном случае, так как мы берем max date, это все события, но логика универсальна
        history = user_events[user_events['date'] <= target_date]
        if history.empty:
            continue

        # --- Базовые метрики (Lifetime) ---
        total_events = len(history)
        total_clicks = (history['event_type'] == 'click').sum()
        total_purchases = (history['event_type'] == 'purchase').sum()
        total_spent = history['price'].sum() if 'price' in history.columns else 0.0

        # --- Даты и Recency ---
        first_ts = history['ts'].iloc[0]
        last_ts = history['ts'].iloc[-1]

        days_since_first = (target_ts - first_ts).days
        days_since_last = (target_ts - last_ts).days

        unique_active_days = len(history['date'].unique())
        events_per_day = total_events / max(1, unique_active_days)

        # --- ROLLING METRICS (Новое!) ---
        # Эффективный расчет скользящих окон
        hist_dates = history['date'].values
        target_date_np = np.datetime64(target_date)

        # Mask for last 7 days
        mask_7d = (hist_dates > target_date_np - np.timedelta64(7, 'D'))
        events_last_7d = np.sum(mask_7d)

        # Mask for last 30 days
        mask_30d = (hist_dates > target_date_np - np.timedelta64(30, 'D'))
        events_last_30d = np.sum(mask_30d)

        purchases_last_30d = 0
        spent_last_30d = 0.0

        if mask_30d.any():
            subset_30d = history[mask_30d]
            purchases_last_30d = (subset_30d['event_type'] == 'purchase').sum()
            if 'price' in subset_30d.columns:
                spent_last_30d = subset_30d['price'].sum()

        # --- Контекст ---
        seen_items = set(history['item_id'].dropna().astype(str)) if has_item else set()
        last_row = history.iloc[-1]
        unique_active_days = len(history['date'].unique()) # <--- Это считалось

        row = {
            "snapshot_date": target_date,
            "user_id": user_id,
            # Lifetime
            "total_events": int(total_events),
            "unique_days": int(unique_active_days),
            "total_clicks": int(total_clicks),
            "total_purchases": int(total_purchases),
            "total_spent": float(total_spent),
            "distinct_items": len(seen_items),
            # Rolling
            "events_last_7d": int(events_last_7d),
            "events_last_30d": int(events_last_30d),
            "purchases_last_30d": int(purchases_last_30d),
            "spent_last_30d": float(spent_last_30d),
            # --- НОВЫЕ DERIVED FEATURES (синхронизация с snapshot_builder1.py) ---
            "conversion_rate_30d": float(purchases_last_30d / max(1, events_last_30d)),
            "avg_order_value_30d": float(spent_last_30d / max(1, purchases_last_30d)),
            "purchase_frequency": float(total_purchases / max(1, unique_active_days)),
            "avg_spend_per_event": float(total_spent / max(1, total_events)),
            # Recency
            "days_since_first": int(days_since_first),
            "days_since_last": int(days_since_last),
            "events_per_day": float(events_per_day),
            # --- НОВОЕ: Recency score ---
            "recency_score": float(1.0 / (1.0 + days_since_last)),
            # Last Context
            "last_event_type": last_row.get('event_type'),
            "last_region": last_row.get('region') if has_region else None,
            "last_item": last_row.get('item_id') if has_item else None,
        }

        if trends_dict:
            t = trends_dict.get(target_date)
            if t: row.update(t)

        snapshots.append(row)
        _print_progress(idx, len(users), prefix="Daily Snapshot")

    result = pd.DataFrame(snapshots)
    logger.info(f"Снапшот готов: {len(result)} строк")
    return result


def main():
    logger = _get_logger()
    logger.info("Запуск daily_snapshot_builder1")

    # 1. Грузим сырые данные
    df = load_events(EVENTS_DIR)
    if df.empty:
        logger.warning("Нет данных для построения!")
        return

    # 2. Грузим тренды
    trends_dict = load_trends(TRENDS_PATH)

    # 3. Строим снапшот на последнюю дату
    snapshot = build_daily_snapshot(df, trends_dict=trends_dict)

    # 4. Сохраняем
    if not snapshot.empty:
        out_path = os.path.join(OUT_DIR, "daily_snapshot1.parquet")
        snapshot.to_parquet(out_path, index=False)
        logger.info(f"Снапшот сохранён в {out_path}")
    else:
        logger.warning("Снапшот пуст")


if __name__ == "__main__":
    main()