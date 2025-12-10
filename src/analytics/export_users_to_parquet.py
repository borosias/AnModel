#!/usr/bin/env python
"""
Экспорт таблицы users из Postgres в Parquet.

Запуск из корня проекта:
    python MarketApp/src/analytics/export_users_to_parquet.py

Переменные окружения (с дефолтами):
    PG_HOST=localhost
    PG_PORT=5432
    PG_DB=analytics
    PG_USER=admin
    PG_PASSWORD=admin
    PG_SCHEMA=public
    USERS_TABLE=users

Выход:
    MarketApp/src/analytics/data/users/users.parquet
"""

import os
from pathlib import Path
from typing import List

import pandas as pd
from sqlalchemy import create_engine


def get_pg_engine():
    host = os.getenv("PG_HOST", "localhost")
    port = int(os.getenv("PG_PORT", "5432"))
    db = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "postgres")

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def cast_uuid_columns_to_str(df: pd.DataFrame, uuid_columns: List[str]) -> pd.DataFrame:
    """Приводит указанные UUID‑колонки к строкам, чтобы pyarrow не падал."""
    df = df.copy()
    for col in uuid_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def export_users():
    # Пути
    base_dir = Path(__file__).resolve().parent          # MarketApp/src/analytics
    out_dir = base_dir / "data" / "users"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "users.parquet"

    schema = os.getenv("PG_SCHEMA", "public")
    table = os.getenv("USERS_TABLE", "users")
    full_table = f'"{schema}"."{table}"'

    print(f"Connecting to Postgres and loading {full_table}...")
    engine = get_pg_engine()

    query = f"SELECT * FROM {full_table}"
    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df):,} users, columns: {list(df.columns)}")

    # Явно конвертируем UUID‑колонки в строки
    # В твоём postgres_init.sql колонка называется user_uid
    df = cast_uuid_columns_to_str(df, uuid_columns=["user_uid"])

    print(f"Writing to {out_path} ...")
    df.to_parquet(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    export_users()
