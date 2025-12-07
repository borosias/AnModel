import os
import glob
import logging
from bisect import bisect_right
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

EVENTS_DIR = "../data/parquet"
OUT_DIR = "../data/snapshots/model1"
HORIZON_DAYS = 7
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TRENDS_PATH = "../trends_data/trends_master.parquet"  # путь к трендам

os.makedirs(OUT_DIR, exist_ok=True)


def _print_progress(current, total, prefix=""):  # простой прогресс бар без зависимостей
    """Отображает прогресс в одной строке. current начинается с 0 или 1.

    Безопасно для total == 0 (ничего не выводит).
    """
    if not total:
        return
    # нормализуем current в диапазон [0..total]
    current = max(0, min(current, total))
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "-" * (bar_len - filled)
    percent = (current / total) * 100
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent:.0f}%)", end="", flush=True)
    if current >= total:
        print()  # перенос строки по завершении


def _get_logger():
    """Создает и возвращает логгер для процесса построения снапшотов.

    Логи пишутся как в консоль, так и в файл OUT_DIR/snapshot_builder1.log
    """
    logger = logging.getLogger("snapshot_builder1")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Консоль
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Файл
    try:
        log_path = os.path.join(OUT_DIR, "snapshot_builder1.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        # Если не удалось открыть файл, продолжаем только с консолью
        logger.warning(f"Не удалось создать файловый логгер: {e}")

    return logger


def load_events(events_dir):
    """Загрузка событий"""
    logger = _get_logger()
    files = sorted(glob.glob(os.path.join(events_dir, "events_part_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {events_dir}")

    # Читаем все файлы с отображением прогресса
    parts = []
    total_files = len(files)
    for i, fpath in enumerate(files, start=1):
        try:
            parts.append(pd.read_parquet(fpath))
        except Exception as e:
            logger.error(f"Ошибка чтения {fpath}: {e}")
            raise
        _print_progress(i, total_files, prefix="Загрузка parquet файлов")
    df = pd.concat(parts, ignore_index=True)

    # Преобразования
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")

    df["date"] = df["ts"].dt.date
    if 'price' in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    logger.info(f"Загружено {len(df)} событий")
    logger.info(f"Период: {df['date'].min()} - {df['date'].max()}")

    return df

def load_trends(trends_path: str = TRENDS_PATH) -> pd.DataFrame:
    logger = _get_logger()

    if not os.path.exists(trends_path):
        logger.info(f"Файл трендов не найден: {trends_path} — продолжаем без тренд‑фич")
        return pd.DataFrame()

    try:
        df = pd.read_parquet(trends_path)
    except Exception as e:
        logger.warning(f"Не удалось прочитать тренды из {trends_path}: {e}")
        return pd.DataFrame()

    if df.empty:
        logger.info(f"Файл трендов пустой: {trends_path} — продолжаем без тренд‑фич")
        return pd.DataFrame()

    if "date" not in df.columns or "popularity" not in df.columns:
        logger.warning("В trends_master.parquet нет колонок 'date' и 'popularity' — тренды игнорируются")
        return pd.DataFrame()

    # 1. Приводим типы
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

    # 2. Агрегируем по тем датам, которые есть (недельные точки)
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
        return pd.DataFrame()

    # 3. Разворачиваем до сплошного дневного ряда и тянем значения вперёд
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index).ffill()

    # Индекс делаем типом date, чтобы совпадал со snapshot_date
    daily.index = daily.index.date

    logger.info(
        f"Загружены тренды: {len(daily)} дней "
        f"({daily.index.min()} - {daily.index.max()})"
    )

    return daily


def build_snapshots_simple(df, horizon_days=HORIZON_DAYS, trends_daily: pd.DataFrame | None = None):
    """
    Оптимизированное построение snapshots с логированием прогресса.

    Основная оптимизация:
    - Обработка по пользователям с накапливанием метрик во времени (без полного пересчёта на каждый день).
    - Быстрый расчёт целевой переменной через бинарный поиск по покупкам пользователя.
    """
    logger = _get_logger()

    # Подготовка трендов для быстрого доступа
    trends_dict = {}
    if trends_daily is not None and not trends_daily.empty:
        trends_dict = trends_daily.to_dict(orient="index")
        logger.info(f"Тренд‑фичи будут использованы для {len(trends_dict)} дней")

    # Уникальные даты для snapshots
    all_dates = sorted(df['date'].unique())
    total_days = len(all_dates)
    logger.info(f"Будет создано snapshots на {total_days} дней")

    # Предварительные признаки
    has_item = 'item_id' in df.columns
    has_region = 'region' in df.columns

    # Группировка по пользователям один раз
    users = df['user_id'].unique()
    n_users = len(users)
    logger.info(f"Пользователей к обработке: {n_users}")

    snapshots = []

    # Для каждого пользователя строим временной профиль и генерируем строки снапшотов
    for u_idx, user_id in enumerate(users, start=1):
        user_events = df[df['user_id'] == user_id].sort_values('ts')
        if user_events.empty:
            continue

        first_ts = user_events['ts'].iloc[0]

        # Агрегаты по дням для пользователя
        # Считаем события по типам за день
        clicks_mask = (user_events['event_type'] == 'click')
        purchases_mask = (user_events['event_type'] == 'purchase')

        # Подсчёты за день
        per_day_counts = (
            user_events.groupby('date')
            .agg(
                total_events=('event_type', 'size'),
                total_clicks=('event_type', lambda s: (s == 'click').sum()),
                total_purchases=('event_type', lambda s: (s == 'purchase').sum()),
                total_spent=(
                    'price',
                    lambda s: float(pd.to_numeric(s, errors='coerce').fillna(0.0).sum())
                ),
            )
            .sort_index()
        )

        # Последние значения за день (для "last_*" и last_ts)
        last_rows_per_day = (
            user_events.sort_values('ts')
            .groupby('date', as_index=True)
            .tail(1)
            .set_index('date')
        )

        # Множества товаров за день (для кумулятивного distinct_items)
        if has_item:
            items_per_day = (
                user_events.dropna(subset=['item_id'])
                .groupby('date')['item_id']
                .apply(lambda s: set(s.astype(str).tolist()))
            )
        else:
            items_per_day = pd.Series(dtype=object)

        # Даты с активностью пользователя
        user_days = list(per_day_counts.index)
        if not user_days:
            continue

        # Начинаем создавать строки со дня первого появления пользователя до последнего дня всех данных
        start_idx = np.searchsorted(all_dates, user_days[0], side='left')

        # Кумаулятивные переменные
        cum_total_events = 0
        cum_total_clicks = 0
        cum_total_purchases = 0
        cum_total_spent = 0.0
        cum_unique_days = 0
        seen_items = set()
        last_event_type = None
        last_region = None
        last_item = None
        last_ts = None

        # Указатель по дням пользователя
        pd_ptr = 0
        n_user_days = len(user_days)

        # Подготовка покупок для таргета (по датам)
        purchases_df = user_events[purchases_mask]
        if not purchases_df.empty:
            purchase_dates = purchases_df['date'].tolist()
            purchase_amounts = pd.to_numeric(purchases_df['price'], errors='coerce').fillna(0.0).astype(float).tolist()
            purchases_ord = [d.toordinal() for d in purchase_dates]
        else:
            purchases_ord = []
            purchase_amounts = []

        for d in all_dates[start_idx:]:
            # Обновляем кумулятивные величины, пока не догоним текущую дату снапшота
            while pd_ptr < n_user_days and user_days[pd_ptr] <= d:
                day = user_days[pd_ptr]
                agg = per_day_counts.loc[day]
                cum_total_events += int(agg['total_events'])
                cum_total_clicks += int(agg['total_clicks'])
                cum_total_purchases += int(agg['total_purchases'])
                cum_total_spent += float(agg['total_spent'])
                cum_unique_days += 1

                # Последние значения дня
                if day in last_rows_per_day.index:
                    lr = last_rows_per_day.loc[day]
                    last_event_type = lr.get('event_type', last_event_type)
                    if has_region:
                        last_region = lr.get('region', last_region)
                    if has_item:
                        last_item = lr.get('item_id', last_item)
                    last_ts = lr.get('ts', last_ts)

                # уникальные товары
                if has_item and day in items_per_day.index:
                    seen_items.update(items_per_day.loc[day])

                pd_ptr += 1

            # Если на текущий момент у пользователя ещё нет ни одного события — пропускаем строку
            if cum_total_events == 0:
                continue

            # Даты/времена для расчётов интервалов
            snapshot_datetime = pd.Timestamp(
                datetime.combine(d, datetime.max.time())
            ).tz_localize('UTC')

            # Фичи времени
            days_since_first = (snapshot_datetime - first_ts).days if first_ts is not None else 999
            days_since_last = (snapshot_datetime - last_ts).days if last_ts is not None else 0

            # Цель: следующая покупка в горизонте
            will_purchase = 0
            days_to_next = 999
            next_amount = 0.0
            if purchases_ord:
                s_ord = d.toordinal()
                idx = bisect_right(purchases_ord, s_ord)
                if idx < len(purchases_ord):
                    delta_days = purchases_ord[idx] - s_ord
                    if 0 < delta_days <= horizon_days:
                        will_purchase = 1
                        days_to_next = delta_days
                        next_amount = float(purchase_amounts[idx])

            row = {
                "snapshot_date": d,
                "user_id": user_id,
                "total_events": int(cum_total_events),
                "unique_days": int(cum_unique_days),
                "total_clicks": int(cum_total_clicks),
                "total_purchases": int(cum_total_purchases),
                "total_spent": float(cum_total_spent),
                "distinct_items": int(len(seen_items)) if has_item else 0,
                "days_since_first": int(days_since_first),
                "days_since_last": int(days_since_last),
                "events_per_day": float(cum_total_events / max(1, cum_unique_days)),
                "last_event_type": last_event_type,
                "last_region": last_region if has_region else None,
                "last_item": last_item if has_item else None,
                "will_purchase_next_7d": int(will_purchase),
                "days_to_next_purchase": int(days_to_next),
                "next_purchase_amount": float(next_amount),
            }

            # Добавляем тренд‑фичи для текущей даты, если доступны
            if trends_dict:
                t = trends_dict.get(d)
                if t is not None:
                    row.update(t)

            snapshots.append(row)

        # Прогресс по пользователям
        _print_progress(u_idx, n_users, prefix="Построение snapshots по пользователям")

    # Формируем DataFrame из всех строк
    if snapshots:
        result = pd.DataFrame(snapshots)
        logger.info(f"\nСоздано {len(result)} snapshots")
        pos = int(result['will_purchase_next_7d'].sum())
        logger.info(f"Положительных примеров: {pos} ({pos / max(1, len(result)):.2%})")
        return result
    else:
        logger.warning("Не удалось построить ни одного снапшота")
        return pd.DataFrame()


def split_and_save_simple(snaps_df, out_dir):
    """Простое разделение по времени и сохранение"""
    logger = _get_logger()
    if snaps_df.empty:
        logger.warning("Нет данных для сохранения")
        return

    # Сортируем по дате
    snaps_df = snaps_df.sort_values('snapshot_date')

    # Уникальные даты
    unique_dates = snaps_df['snapshot_date'].unique()
    n_dates = len(unique_dates)

    # Индексы для разделения
    train_end_idx = int(n_dates * TRAIN_RATIO)
    val_end_idx = train_end_idx + int(n_dates * VAL_RATIO)

    # Границы дат
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]

    # Разделяем
    train_df = snaps_df[snaps_df['snapshot_date'].isin(train_dates)].copy()
    val_df = snaps_df[snaps_df['snapshot_date'].isin(val_dates)].copy()
    test_df = snaps_df[snaps_df['snapshot_date'].isin(test_dates)].copy()

    # Сохраняем
    train_df.to_parquet(os.path.join(out_dir, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(out_dir, 'val.parquet'), index=False)
    test_df.to_parquet(os.path.join(out_dir, 'test.parquet'), index=False)

    logger.info(f"\nСохранено:")
    logger.info(f"  Train: {len(train_df)} строк")
    logger.info(f"  Val:   {len(val_df)} строк")
    logger.info(f"  Test:  {len(test_df)} строк")

    # Статистика
    logger.info(f"\nРаспределение:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos = df['will_purchase_next_7d'].sum()
        total = len(df)
        logger.info(f"  {name}: {pos}/{total} ({pos / max(1, total):.2%}) положительных")


def main():
    """Основная функция"""
    logger = _get_logger()
    logger.info("Простой билдер snapshots (оптимизированный)")
    logger.info("=" * 50)

    # 1. Загружаем данные
    logger.info("\n1. Загрузка данных...")
    df = load_events(EVENTS_DIR)

    # 1.1. Загружаем тренды (если есть)
    logger.info("\n1.1. Загрузка трендов (если есть)...")
    trends_daily = load_trends(TRENDS_PATH)

    # 2. Строим snapshots
    logger.info("\n2. Построение snapshots...")
    snaps = build_snapshots_simple(df, horizon_days=HORIZON_DAYS, trends_daily=trends_daily)

    # 3. Сохраняем
    logger.info("\n3. Сохранение...")
    if not snaps.empty:
        split_and_save_simple(snaps, OUT_DIR)
        logger.info(f"\nГотово! Данные в {OUT_DIR}")
    else:
        print("Ошибка: не удалось создать snapshots")


if __name__ == "__main__":
    main()