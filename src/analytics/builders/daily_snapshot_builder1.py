import glob
import logging
import os
from datetime import datetime

import pandas as pd

EVENTS_DIR = "../data/parquet"
OUT_DIR = f"../data/daily_features/snapshot_{datetime.now().strftime('%Y_%m_%d')}"
TRENDS_PATH = "../trends_data/trends_master.parquet"

os.makedirs(OUT_DIR, exist_ok=True)

def _print_progress(current, total, prefix=""):
    if not total:
        return
    current = max(0, min(current, total))
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "-" * (bar_len - filled)
    percent = (current / total) * 100
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
    except Exception as e:
        logger.warning(f"Не удалось создать файловый логгер: {e}")
    return logger

def load_events(events_dir):
    logger = _get_logger()
    files = sorted(glob.glob(os.path.join(events_dir, "events_part_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {events_dir}")
    parts = []
    for i, fpath in enumerate(files, start=1):
        parts.append(pd.read_parquet(fpath))
        _print_progress(i, len(files), prefix="Загрузка parquet файлов")
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date
    if 'price' in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    logger.info(f"Загружено {len(df)} событий ({df['date'].min()} - {df['date'].max()})")
    return df

def load_trends(trends_path: str = TRENDS_PATH) -> pd.DataFrame:
    logger = _get_logger()

    # 0. Проверка файла
    if not os.path.exists(trends_path):
        logger.info(f"Файл трендов не найден: {trends_path} — продолжаем без тренд‑фич")
        return {}

    try:
        df = pd.read_parquet(trends_path)
    except Exception as e:
        logger.warning(f"Не удалось прочитать тренды из {trends_path}: {e}")
        return {}

    if df.empty:
        logger.info("Файл трендов пустой — пропускаем")
        return {}

    if "date" not in df.columns or "popularity" not in df.columns:
        logger.info("Тренды отсутствуют или некорректны — пропускаем")
        return {}

    # 1. Приводим типы
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

    # 2. Агрегируем по тем датам, которые есть (как отдаёт Google Trends)
    daily = (
        df.dropna(subset=["popularity"])
        .groupby("date")["popularity"]
        .agg(
            trend_popularity_mean="mean",
            trend_popularity_max="max",
        )
        .sort_index()
    )

    if daily.empty:
        logger.info("Тренды после агрегации пустые — продолжаем без тренд‑фич")
        return {}

    # 3. Разворачиваем в сплошной дневной ряд и тянем значения вперёд
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index).ffill()

    # Индекс делаем типом date, чтобы совпадал с snapshot_date / date
    daily.index = daily.index.date

    logger.info(
        f"Загружены тренды для {len(daily)} дней "
        f"({daily.index.min()} - {daily.index.max()})"
    )

    # daily_snapshot_builder1, судя по коду, ожидает dict[date] -> {feature_name: value}
    return daily.to_dict(orient="index")

def build_daily_snapshot(df, trends_dict=None):
    logger = _get_logger()
    if df.empty:
        return pd.DataFrame()
    last_date = df['date'].max()
    logger.info(f"Создаём снапшот за {last_date}")
    has_item = 'item_id' in df.columns
    has_region = 'region' in df.columns
    users = df['user_id'].unique()
    snapshots = []

    for idx, user_id in enumerate(users, start=1):
        user_events = df[df['user_id'] == user_id].sort_values('ts')
        if user_events.empty:
            continue
        user_day = last_date
        day_events = user_events[user_events['date'] <= user_day]
        if day_events.empty:
            continue

        total_events = len(day_events)
        total_clicks = (day_events['event_type'] == 'click').sum()
        total_purchases = (day_events['event_type'] == 'purchase').sum()
        total_spent = day_events['price'].sum() if 'price' in day_events.columns else 0.0
        seen_items = set(day_events['item_id'].dropna().astype(str)) if has_item else set()
        first_ts = user_events['ts'].iloc[0]
        last_ts = day_events['ts'].iloc[-1]
        days_since_first = (pd.Timestamp(datetime.combine(user_day, datetime.max.time())).tz_localize('UTC') - first_ts).days
        days_since_last = (pd.Timestamp(datetime.combine(user_day, datetime.max.time())).tz_localize('UTC') - last_ts).days
        events_per_day = total_events / max(1, len(day_events['date'].unique()))
        last_row = day_events.iloc[-1]
        last_event_type = last_row.get('event_type', None)
        last_region = last_row['region'] if has_region else None
        last_item = last_row['item_id'] if has_item else None

        row = {
            "snapshot_date": user_day,
            "user_id": user_id,
            "total_events": total_events,
            "total_clicks": total_clicks,
            "total_purchases": total_purchases,
            "total_spent": float(total_spent),
            "distinct_items": len(seen_items),
            "days_since_first": days_since_first,
            "days_since_last": days_since_last,
            "events_per_day": events_per_day,
            "last_event_type": last_event_type,
            "last_region": last_region,
            "last_item": last_item,
        }
        if trends_dict:
            t = trends_dict.get(user_day)
            if t is not None:
                row.update(t)
        snapshots.append(row)
        _print_progress(idx, len(users), prefix="Построение daily snapshot")

    result = pd.DataFrame(snapshots)
    logger.info(f"Снапшот готов: {len(result)} строк")
    return result

def main():
    logger = _get_logger()
    logger.info("Запуск daily_snapshot_builder1")
    df = load_events(EVENTS_DIR)
    trends_dict = load_trends(TRENDS_PATH)
    snapshot = build_daily_snapshot(df, trends_dict=trends_dict)
    if not snapshot.empty:
        out_path = os.path.join(OUT_DIR,"daily_snapshot1.parquet")
        snapshot.to_parquet(out_path, index=False)
        logger.info(f"Снапшот сохранён в {out_path}")
    else:
        logger.warning("Нет данных для сохранения")

if __name__ == "__main__":
    main()
