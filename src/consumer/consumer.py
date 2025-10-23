import asyncio
import json
import os
import time
from collections import deque
from aiokafka import AIOKafkaConsumer
from clickhouse_driver import Client
from datetime import datetime

# Конфигурация
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw-events")
CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", 9000))
CH_DATABASE = os.getenv("CLICKHOUSE_DB", "analytics")
CH_USER = os.getenv("CLICKHOUSE_USER", "admin")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "admin")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 5000))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 0.1))

# Для мгновенной скорости
processed_total = 0
processed_recent = 0
recent_start_time = time.time()
overall_start_time = time.time()

# Для скользящего среднего (последние 10 секунд)
throughput_history = deque(maxlen=10)


def normalize_event_fast(ev: dict):
    props = ev.get("properties", {})
    ts = ev.get("timestamp", "")

    try:
        ts_parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        ts_parsed = datetime.utcnow()

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
        json.dumps(props, ensure_ascii=False) if props else "{}"
    )


async def insert_batch(ch_client, batch):
    try:
        ch_client.execute("INSERT INTO events_raw VALUES", batch)
        return True, len(batch)
    except Exception as e:
        print(f"Insert error: {e}")
        return False, 0


async def produce_loop():
    global processed_total, processed_recent, recent_start_time

    try:
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id="clickhouse-consumer-group",
            enable_auto_commit=False,
            auto_offset_reset="latest",
            fetch_max_bytes=10485760,
            fetch_max_wait_ms=100,
            max_partition_fetch_bytes=1048576,
            consumer_timeout_ms=100,
        )
        await consumer.start()
        print("Connected to Kafka")
    except Exception as e:
        print(f"Kafka connection failed: {e}")
        return

    try:
        ch = Client(
            host=CH_HOST,
            user=CH_USER,
            password=CH_PASSWORD,
            port=CH_PORT,
            database=CH_DATABASE,
            settings={'send_timeout': 10, 'receive_timeout': 10}
        )
        ch.execute("SELECT 1")
        print("Connected to ClickHouse")
    except Exception as e:
        print(f"ClickHouse connection failed: {e}")
        await consumer.stop()
        return

    buffer = []
    last_flush = time.time()
    last_stats = time.time()
    insert_tasks = set()
    consecutive_errors = 0

    try:
        while True:
            try:
                msg = await asyncio.wait_for(consumer.getone(), timeout=0.001)
                if msg:
                    try:
                        payload = json.loads(msg.value.decode("utf-8"))
                        buffer.append(normalize_event_fast(payload))
                        processed_total += 1
                        processed_recent += 1
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        if consecutive_errors % 100 == 0:
                            print(f"Parse errors: {consecutive_errors}")
                        continue
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors % 100 == 0:
                    print(f"Kafka read errors: {consecutive_errors}")
                continue

            current_time = time.time()
            if buffer and (len(buffer) >= BATCH_SIZE or current_time - last_flush >= BATCH_TIMEOUT):
                batch = buffer.copy()
                buffer.clear()
                last_flush = current_time

                task = asyncio.create_task(insert_batch(ch, batch))
                insert_tasks.add(task)
                task.add_done_callback(insert_tasks.discard)

            if insert_tasks:
                done, _ = await asyncio.wait(insert_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        success, count = await task
                        if success:
                            await consumer.commit()
                    except Exception as e:
                        print(f"Task error: {e}")

            # Обновленная статистика с мгновенной и средней скоростью
            if current_time - last_stats >= 5:
                # Средняя скорость за всё время
                overall_rate = processed_total / (current_time - overall_start_time)

                # Мгновенная скорость за последние 5 секунд
                recent_rate = processed_recent / (
                            current_time - recent_start_time) if current_time - recent_start_time > 0 else 0

                # Добавляем в историю для скользящего среднего
                throughput_history.append(recent_rate)
                avg_rate = sum(throughput_history) / len(throughput_history) if throughput_history else 0

                print(
                    f"Rate: {recent_rate:.0f}/s (current), {avg_rate:.0f}/s (avg 10s), {overall_rate:.0f}/s (overall), Total: {processed_total}")

                # Сбрасываем счетчики для следующего интервала
                processed_recent = 0
                recent_start_time = current_time
                last_stats = current_time

            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if buffer:
            try:
                success, count = await insert_batch(ch, buffer)
                if success:
                    await consumer.commit()
                    print(f"Final insert: {count} events")
            except Exception as e:
                print(f"Final insert failed: {e}")

        if insert_tasks:
            try:
                await asyncio.wait(insert_tasks, timeout=2.0)
            except Exception as e:
                print(f"Wait tasks error: {e}")

        try:
            await consumer.stop()
            print("Consumer stopped")
        except Exception as e:
            print(f"Consumer stop error: {e}")


def ensure_table():
    try:
        client = Client(
            host=CH_HOST,
            port=CH_PORT,
            user=CH_USER,
            password=CH_PASSWORD,
            database=CH_DATABASE
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
                       )
                           ENGINE = MergeTree()
                               ORDER BY (ts, event_type)
                       """)
        print("Table ensured")
        client.disconnect()
    except Exception as e:
        print(f"Table creation error: {e}")
        raise


if __name__ == "__main__":
    try:
        ensure_table()
        time.sleep(1)
        asyncio.run(produce_loop())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Main error: {e}")