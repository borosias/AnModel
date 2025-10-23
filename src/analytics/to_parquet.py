import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
from clickhouse_driver import Client
from datetime import datetime, timedelta

CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", 9000))
CH_DB = os.getenv("CLICKHOUSE_DB", "analytics")
CH_USER = os.getenv("CLICKHOUSE_USER", "admin")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "admin")
OUT_DIR = os.getenv("PARQUET_OUT", "./data/parquet")
BATCH = int(os.getenv("PARQUET_BATCH", 100000))
DAYS = int(os.getenv("EXPORT_DAYS", 30))

os.makedirs(OUT_DIR, exist_ok=True)
client = Client(host=CH_HOST, user=CH_USER, password=CH_PASSWORD, port=CH_PORT, database=CH_DB)


def export_data_fast():
    start_time = time.time()

    now = datetime.utcnow()
    start_ts = (now - timedelta(days=DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    # Получаем данные одним запросом с итерацией
    query = """
            SELECT event_id, \
                   event_type, \
                   ts, \
                   session_id, \
                   user_id, \
                   region, \
                   item_id, \
                   price, \
                   properties
            FROM events_raw
            WHERE ts >= toDateTime(%(start_ts)s)
            ORDER BY ts, event_id \
            """

    part = 0
    total_rows = 0

    # Используем итератор для обработки батчами
    for rows in client.execute_iter(query, {'start_ts': start_ts}, chunk_size=BATCH):
        if not rows:
            continue

        # Создаем PyArrow Table напрямую (быстрее чем через Pandas)
        arrays = [
            pa.array([row[0] for row in rows]),  # event_id
            pa.array([row[1] for row in rows]),  # event_type
            pa.array([row[2] for row in rows]),  # ts
            pa.array([row[3] for row in rows]),  # session_id
            pa.array([row[4] for row in rows]),  # user_id
            pa.array([row[5] for row in rows]),  # region
            pa.array([row[6] for row in rows]),  # item_id
            pa.array([row[7] for row in rows]),  # price
            pa.array([row[8] for row in rows]),  # properties
        ]

        schema = pa.schema([
            ('event_id', pa.string()),
            ('event_type', pa.string()),
            ('ts', pa.timestamp('ms')),
            ('session_id', pa.string()),
            ('user_id', pa.string()),
            ('region', pa.string()),
            ('item_id', pa.string()),
            ('price', pa.float64()),
            ('properties', pa.string())
        ])

        table = pa.Table.from_arrays(arrays, schema=schema)

        out_path = f"{OUT_DIR}/events_part_{part:04d}.parquet"
        pq.write_table(table, out_path, compression='snappy')

        total_rows += len(rows)
        print(f"Part {part}: {len(rows):,} rows | Total: {total_rows:,}")
        part += 1

    total_time = time.time() - start_time
    rate = total_rows / total_time if total_time > 0 else 0

    print(f"\nExport finished: {part} files, {total_rows:,} rows")
    print(f"Time: {total_time:.2f}s, Rate: {rate:.0f} rows/sec")


if __name__ == "__main__":
    export_data_fast()
