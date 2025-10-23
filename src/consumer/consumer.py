import asyncio
import json
import os
import time
from collections import deque
from aiokafka import AIOKafkaConsumer
from clickhouse_driver import Client
from datetime import datetime
import logging
import threading

# Fast JSON
try:
    import orjson


    def loads(s):
        return orjson.loads(s)


    def dumps(o):
        return orjson.dumps(o).decode('utf-8')
except ImportError:
    def loads(s):
        return json.loads(s)


    def dumps(o):
        return json.dumps(o, ensure_ascii=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consumer_fast")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw-events")
CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", 9000))
CH_DATABASE = os.getenv("CLICKHOUSE_DB", "analytics")
CH_USER = os.getenv("CLICKHOUSE_USER", "admin")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "admin")

# Более консервативные настройки для Windows
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 5000))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 0.5))
MAX_INFLIGHT_INSERTS = int(os.getenv("MAX_INFLIGHT_INSERTS", 2))  # Уменьшил для Windows
GETMANY_TIMEOUT_MS = int(os.getenv("GETMANY_TIMEOUT_MS", 200))
STATS_INTERVAL = float(os.getenv("STATS_INTERVAL", 2.0))

# Глобальные метрики
processed_total = 0
processed_recent = 0
recent_start_time = time.time()
overall_start_time = time.time()
throughput_history = deque(maxlen=20)


class ClickHousePool:
    """Пул соединений для ClickHouse"""

    def __init__(self):
        self._pool = []
        self._lock = threading.Lock()
        self._max_connections = MAX_INFLIGHT_INSERTS

    def get_client(self):
        """Получить клиент из пула или создать новый"""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            else:
                return self._create_client()

    def return_client(self, client):
        """Вернуть клиент в пул"""
        with self._lock:
            if len(self._pool) < self._max_connections:
                self._pool.append(client)
            else:
                client.disconnect()

    def _create_client(self):
        """Создать новый клиент ClickHouse"""
        return Client(
            host=CH_HOST,
            port=CH_PORT,
            user=CH_USER,
            password=CH_PASSWORD,
            database=CH_DATABASE,
            settings={
                'send_timeout': 30,
                'receive_timeout': 30,
                'insert_block_size': 10000,
                'max_insert_block_size': 50000,
                'max_execution_time': 30,
            }
        )

    def close_all(self):
        """Закрыть все соединения"""
        with self._lock:
            for client in self._pool:
                client.disconnect()
            self._pool.clear()


def normalize_event_fast(ev: dict):
    """Ультра-быстрая нормализация с минимальными проверками"""
    props = ev.get("properties", {})
    ts_str = ev.get("timestamp", "")

    # Быстрое преобразование timestamp
    try:
        if 'Z' in ts_str:
            ts_str = ts_str.replace('Z', '+00:00')
        ts_parsed = datetime.fromisoformat(ts_str)
    except Exception:
        ts_parsed = datetime.utcnow()

    # Быстрое извлечение price
    price = props.get("price")
    if price is not None:
        try:
            price = float(price)
        except (ValueError, TypeError):
            price = None

    return (
        ev.get("event_id", ""),
        ev.get("event_type", ""),
        ts_parsed,
        ev.get("session_id", ""),
        ev.get("user_id", ""),
        ev.get("region", ""),
        props.get("item_id", ""),
        price,
        dumps(props) if props else "{}"
    )


def ensure_table():
    client = Client(
        host=CH_HOST,
        port=CH_PORT,
        user=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE,
        settings={
            'max_execution_time': 60,
            'receive_timeout': 60,
            'send_timeout': 60,
        }
    )
    client.execute("""
                   CREATE TABLE IF NOT EXISTS events_raw
                   (
                       event_id   String,
                       event_type String,
                       ts         DateTime64(3),
                       session_id String,
                       user_id    String,
                       region     String,
                       item_id    String,
                       price      Nullable(Float64),
                       properties String
                   ) ENGINE = MergeTree()
                         ORDER BY (ts)
                   """)
    client.disconnect()
    logger.info("Table ensured")


async def run_insert_in_executor(ch_pool, batch):
    """Вставка с использованием пула соединений"""

    def _do_insert():
        client = None
        try:
            client = ch_pool.get_client()
            client.execute(
                "INSERT INTO events_raw (event_id,event_type,ts,session_id,user_id,region,item_id,price,properties) VALUES",
                batch
            )
            return len(batch), True
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return 0, False
        finally:
            if client:
                ch_pool.return_client(client)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_insert)


async def consumer_loop():
    global processed_total, processed_recent, recent_start_time

    # Инициализация пула ClickHouse
    ch_pool = ClickHousePool()

    # Kafka consumer с более консервативными настройками
    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="clickhouse-consumer-group",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        fetch_max_bytes=10 * 1024 * 1024,  # 10MB вместо 100MB
        max_partition_fetch_bytes=5 * 1024 * 1024,  # 5MB вместо 50MB
        fetch_max_wait_ms=100,
        max_poll_records=5000,  # Уменьшил для стабильности
        session_timeout_ms=30000,
        heartbeat_interval_ms=5000,
    )

    await consumer.start()
    logger.info("Kafka consumer started with stable settings")

    buffer = []
    last_flush = time.time()
    inflight_tasks = set()
    pending_offsets = {}
    successful_inserts_since_commit = 0
    commit_every_n = 2

    try:
        while True:
            # Получаем сообщения с таймаутом
            msgs = await consumer.getmany(timeout_ms=GETMANY_TIMEOUT_MS, max_records=2000)

            for tp, records in msgs.items():
                if not records:
                    continue

                # Сохраняем offset для коммита
                last_offset = records[-1].offset
                pending_offsets[tp] = last_offset + 1

                # Обработка сообщений
                for rec in records:
                    try:
                        payload = loads(rec.value)
                        buffer.append(normalize_event_fast(payload))
                        processed_total += 1
                        processed_recent += 1
                    except Exception as e:
                        if processed_total % 10000 == 0:
                            logger.warning(f"Parse error: {e}")
                        continue

            now = time.time()

            # Условия для флаша батча
            should_flush = (
                    len(buffer) >= BATCH_SIZE or
                    (now - last_flush) >= BATCH_TIMEOUT or
                    len(buffer) > 10000  # Защита от переполнения
            )

            if buffer and should_flush and len(inflight_tasks) < MAX_INFLIGHT_INSERTS:
                batch_to_insert = buffer[:BATCH_SIZE]
                buffer = buffer[BATCH_SIZE:] if len(buffer) > BATCH_SIZE else []
                last_flush = now

                # Запускаем вставку
                task = asyncio.create_task(
                    _process_batch(
                        ch_pool=ch_pool,
                        batch=batch_to_insert,
                        consumer=consumer,
                        pending_offsets=pending_offsets.copy() if successful_inserts_since_commit >= commit_every_n else None
                    )
                )
                inflight_tasks.add(task)
                task.add_done_callback(inflight_tasks.discard)

            # Очищаем завершенные задачи
            done_tasks = [t for t in inflight_tasks if t.done()]
            for task in done_tasks:
                try:
                    success = task.result()
                    if success:
                        successful_inserts_since_commit += 1
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                inflight_tasks.discard(task)

            # Коммит каждые N успешных вставок
            if successful_inserts_since_commit >= commit_every_n:
                try:
                    await consumer.commit()
                    successful_inserts_since_commit = 0
                    pending_offsets.clear()
                except Exception as e:
                    logger.error(f"Commit failed: {e}")

            # Статистика
            if time.time() - recent_start_time >= STATS_INTERVAL:
                recent_elapsed = time.time() - recent_start_time
                recent_rate = processed_recent / recent_elapsed if recent_elapsed > 0 else 0
                overall_rate = processed_total / (time.time() - overall_start_time)

                throughput_history.append(recent_rate)
                avg_rate = sum(throughput_history) / len(throughput_history) if throughput_history else 0

                logger.info(
                    f"Rate: {recent_rate:.0f}/s | Avg: {avg_rate:.0f}/s | "
                    f"Total: {processed_total} | Buffer: {len(buffer)} | "
                    f"Inflight: {len(inflight_tasks)}"
                )

                processed_recent = 0
                recent_start_time = time.time()

            # Небольшая пауза для кооперативной многозадачности
            if not msgs and not buffer:
                await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"Consumer loop error: {e}")
        raise
    finally:
        logger.info("Shutting down...")

        # Финальный флаш буфера
        if buffer:
            logger.info(f"Flushing final batch of {len(buffer)} events")
            try:
                await run_insert_in_executor(ch_pool, buffer)
                await consumer.commit()
            except Exception as e:
                logger.error(f"Final insert failed: {e}")

        # Ожидаем завершения всех задач
        if inflight_tasks:
            logger.info(f"Waiting for {len(inflight_tasks)} inflight tasks")
            await asyncio.gather(*inflight_tasks, return_exceptions=True)

        await consumer.stop()
        ch_pool.close_all()
        logger.info("Consumer stopped")


async def _process_batch(ch_pool, batch, consumer, pending_offsets):
    """Обрабатывает батч и коммитит при необходимости"""
    try:
        count, success = await run_insert_in_executor(ch_pool, batch)

        if success and pending_offsets:
            await consumer.commit(pending_offsets)
            return True
        return success

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return False


if __name__ == "__main__":
    try:
        ensure_table()
        time.sleep(2)
        asyncio.run(consumer_loop())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")