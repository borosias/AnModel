# AnModel — інформаційна система прогнозування споживчого інтересу (E-commerce)

Репозиторій містить прототип end-to-end конвеєра для:

- збору поведінкових подій через HTTP;
- доставки подій у Kafka;
- потокового запису в ClickHouse;
- експорту подій у Parquet;
- побудови snapshot-ознак (включно з micro‑trend та простими sequence‑embedding ознаками);
- навчання моделі ContextAwareModel (класифікація + регресії);
- подачі прогнозів через FastAPI;
- демонстраційного UI (Vite + React).

Ключова ідея: гібридна архітектура **Streaming + Batch**, де події збираються та зберігаються у потоковому режимі, а ознаки та моделі формуються пакетно на Parquet‑зрізах.

---

## 1. Компоненти та потік даних

1) **MarketApp UI** (React) → звертається до Models API (FastAPI) для:
   - отримання списку сервісів (`/services`);
   - отримання користувачів та їх ознак/прогнозів (`/users`);
   - отримання прогнозів (`/predict`).

2) **Collector API** (FastAPI, `/collect`) → приймає подію та публікує у Kafka topic `raw-events`.

3) **Kafka** → буфер/шина подій.

4) **Consumer** → читає `raw-events`, нормалізує події та пише у ClickHouse таблицю `events_raw` батчами (у коді є консервативні налаштування під Windows).

5) **ETL / Export** → `src/analytics/to_parquet.py` експортує ClickHouse → Parquet частинами.

6) **Feature builders**:
   - `snapshot_builder1.py` (training snapshot з таргетами);
   - `daily_snapshot_builder1.py` (daily snapshot без таргетів, але з micro‑trend та sequence‑embedding).

7) **Model training** → `train_context_aware.py` навчає ContextAwareModel на training snapshot і зберігає модель у `src/models/production_models/`.

8) **Models API** (FastAPI) → на старті намагається завантажити останній daily snapshot і модель, предобчислює прогнозні значення та віддає їх клієнту.

---

## 2. Структура репозиторію (ключові файли)

- `docker-compose.yml` — інфраструктура (Kafka/ZooKeeper, ClickHouse, Postgres, Redis, MinIO, Kafka UI).
- `requrements.txt` — Python залежності (увага: файл у UTF‑16).
- `src/collector/app.py` — Collector API (`POST /collect` → Kafka).
- `src/consumer/consumer.py` — Kafka → ClickHouse consumer з батч‑вставкою.
- `src/data_gen/generate.py` — генератор подій (враховує cohorts, promo window, burst-ланцюжки).
- `src/data_gen/postgres_load.py` — читання user_uid з Postgres.
- `src/analytics/to_parquet.py` — експорт ClickHouse → Parquet (`events_part_*.parquet`).
- `src/analytics/export_users_to_parquet.py` — експорт таблиці `users` з Postgres у `src/analytics/data/users/users.parquet`.
- `src/analytics/one_time_trends_loader.py` — одноразове формування `trends_master.parquet` (опціонально).
- `src/analytics/builders/snapshot_builder1.py` — training snapshot (+таргети).
- `src/analytics/builders/daily_snapshot_builder1.py` — daily snapshot (micro‑trend + sequence embeddings + optional trends).
- `src/models/train_models/train_context_aware.py` — тренування моделі.
- `src/models/api/server.py` — Models API (`/services`, `/predict`, `/users`, `/health`).
- `MarketApp/` — фронтенд (Vite + React), API за замовчуванням: `http://localhost:8000`.

---

## 3. Передумови

- Docker / Docker Desktop (для інфраструктури).
- Python 3.10+ (рекомендовано 3.10–3.12).
- Node.js 18+ (для UI).
- (Windows) бажано запускати Python-скрипти у PowerShell або віртуальному середовищі, Docker — через Docker Desktop.

---

## 4. Важливі нюанси перед стартом

### 4.1. Конфлікт портів 8000 (Collector vs Models API)

- `Collector API` традиційно стартує на 8000, а `Models API` у `src/models/api/server.py` також використовує 8000.
- Для коректної роботи **UI** краще лишити **Models API на 8000**, а **Collector перенести на 8001**.

Далі в інструкції використовується саме цей варіант:
- Collector: `http://localhost:8001/collect`
- Models API: `http://localhost:8000`

### 4.2. `requrements.txt` у UTF‑16

Файл `requrements.txt` (саме так названий у репозиторії) має кодування UTF‑16 LE з CRLF. Деякі версії `pip` можуть відмовлятись коректно парсити такий файл. Рекомендований шлях:

- конвертувати у UTF‑8 і (опціонально) перейменувати у `requirements.txt`.

Приклади команд наведені нижче.

### 4.3. Відносні шляхи в feature builders

`snapshot_builder1.py` та `daily_snapshot_builder1.py` використовують відносні директорії на кшталт `../data/parquet`. Це означає, що їх потрібно запускати **з директорії `src/analytics/builders`**, інакше скрипт шукатиме дані «не там».

---

## 5. Швидкий старт (повний сценарій)

Нижче наведено «повний» сценарій — від підняття інфраструктури до UI.

### Крок 0. Підняти інфраструктуру

```bash
docker compose up -d
```

Перевірка:
- Kafka UI: `http://localhost:8080`
- ClickHouse: `localhost:8123` (HTTP) / `localhost:9000` (native)

### Крок 1. Підготувати Python середовище

macOS/Linux (bash):
```bash
python -m venv .venv
source .venv/bin/activate

# конвертація UTF-16 → UTF-8
python - <<'PY'
from pathlib import Path
p = Path("requrements.txt")
data = p.read_text(encoding="utf-16")
Path("requirements.txt").write_text(data, encoding="utf-8", newline="\n")
print("Wrote requirements.txt (UTF-8)")
PY

pip install -U pip
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1

# конвертація UTF-16 → UTF-8
python - <<'PY'
from pathlib import Path
p = Path("requrements.txt")
data = p.read_text(encoding="utf-16")
Path("requirements.txt").write_text(data, encoding="utf-8", newline="\\n")
print("Wrote requirements.txt (UTF-8)")
PY

python -m pip install -U pip
pip install -r requirements.txt
```

Примітка: якщо ви не хочете створювати новий `requirements.txt`, можна спробувати встановити напряму з `requrements.txt`, але це менш стабільно.

### Крок 2. Ініціалізувати таблицю users в Postgres (для генератора)

У репозиторії є `infra/postgres_init.sql`, але там порядок команд можна поліпшити (розширення `pgcrypto` краще створити ДО вставок). Рекомендований мінімальний варіант:

1) Підключитись до Postgres:
```bash
docker compose exec postgres psql -U postgres -d postgres
```

2) Виконати:
[postgres_init.sql](infra/postgres_init.sql)

Перевірка:
```sql
SELECT COUNT(*) FROM users;
```

Експортувати users у Parquet:
```bash
python src/analytics/export_users_to_parquet.py
# результат: src/analytics/data/users/users.parquet
```

### Крок 3. Запустити Collector API (на 8001)

```bash
uvicorn src.collector.app:app --host 0.0.0.0 --port 8001 --reload
```

### Крок 4. Запустити Consumer (Kafka → ClickHouse)

У новому терміналі (активне venv):
```bash
python src/consumer/consumer.py
```

Consumer сам створить таблицю `events_raw` у ClickHouse (якщо її ще немає).

### Крок 5. Згенерувати події (дані)

У новому терміналі:
```bash
python src/data_gen/generate.py --url http://localhost:8001/collect --events-per-user 200 --concurrency 100
```

Пояснення: генератор створює «реалістичні» ланцюжки подій (Markov-переходи), користувацькі когорти (heavy/medium/light/one_time), а також промо‑вікна та burst‑ланцюжки біля snapshot‑дати для формування позитивного класу.

### Крок 6. Експортувати ClickHouse → Parquet

Рекомендується запускати з директорії `src/analytics`, щоб дефолтний `./data/parquet` потрапив у `src/analytics/data/parquet`.

```bash
cd src/analytics
python to_parquet.py
cd ../..
```

Результат:
- `src/analytics/data/parquet/events_part_0000.parquet`, … (частини)

### Крок 7. Побудувати snapshot-датасети

Важливо: запускати з `src/analytics/builders`, бо шляхи відносні.

```bash
cd src/analytics/builders

# 7.1 training snapshot (train/val/test) → src/analytics/data/snapshots/model1/
python snapshot_builder1.py

# 7.2 daily snapshot → src/analytics/data/daily_features/snapshot_YYYY_MM_DD/daily_snapshot1.parquet
python daily_snapshot_builder1.py

cd ../../..
```

### Крок 8. Навчити модель ContextAwareModel

```bash
python src/models/train_models/train_context_aware.py
```

Результати:
- модель: `src/models/production_models/context_aware_model1.pkl` (створюється автоматично);
- артефакти експерименту: `src/models/experiments/context_aware_model1/` (логи, графіки ROC/PR тощо).

### Крок 9. Запустити Models API (на 8000)

```bash
cd src/models/api
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Перевірка:
- `GET http://localhost:8000/health`
- `GET http://localhost:8000/services`

### Крок 10. Запустити UI

```bash
cd MarketApp
npm install
npm run dev
```

UI за замовчуванням відкриється на:
- `http://localhost:5173`

---

## 6. Поетапний сценарій запуску (для дебагу)

Нижче — той самий процес, але з перевірками після кожного етапу.

1) `docker compose up -d`
   - Kafka UI: `http://localhost:8080`

2) Collector:
   - запуск: `uvicorn src.collector.app:app --port 8001`
   - перевірка:
     ```bash
     curl -X POST http://localhost:8001/collect -H "Content-Type: application/json" -d "{\"event_id\":\"1\",\"event_type\":\"click\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"session_id\":\"s\",\"user_id\":\"u\",\"region\":\"UA-30\",\"properties\":{\"item_id\":\"item_1\",\"price\":10}}"
     ```

3) Consumer:
   - запуск: `python src/consumer/consumer.py`
   - перевірка ClickHouse (будь-який клієнт або через контейнер):
     ```bash
     docker compose exec clickhouse clickhouse-client -u admin --password admin -d analytics -q "SELECT count() FROM events_raw"
     ```

4) Генератор:
   - запуск: `python src/data_gen/generate.py --url http://localhost:8001/collect`
   - перевірка: у логах consumer має зростати throughput; у ClickHouse — `count()`.

5) Export → Parquet:
   - запуск: `cd src/analytics && python to_parquet.py`
   - перевірка: на диску є `src/analytics/data/parquet/events_part_*.parquet`.

6) Feature builders:
   - запуск: `cd src/analytics/builders && python snapshot_builder1.py`
   - перевірка: є `src/analytics/data/snapshots/model1/train.parquet` і т.д.
   - запуск: `python daily_snapshot_builder1.py`
   - перевірка: є `src/analytics/data/daily_features/snapshot_YYYY_MM_DD/daily_snapshot1.parquet`

7) Training:
   - запуск: `python src/models/train_models/train_context_aware.py`
   - перевірка: створився `src/models/production_models/context_aware_model1.pkl`.

8) Models API:
   - запуск: `cd src/models/api && uvicorn server:app --port 8000`
   - перевірка:
     - `GET /services`
     - `POST /predict?service=context_aware`

9) UI:
   - запуск: `cd MarketApp && npm run dev`
   - перевірка: UI бачить сервіси і користувачів.

---

## 7. Конфігурація через змінні середовища

### 7.1 Consumer (Kafka → ClickHouse)

- `KAFKA_BOOTSTRAP` (default: `localhost:9092`)
- `KAFKA_TOPIC` (default: `raw-events`)
- `CLICKHOUSE_HOST` (default: `localhost`)
- `CLICKHOUSE_PORT` (default: `9000`)
- `CLICKHOUSE_DB` (default: `analytics`)
- `CLICKHOUSE_USER` (default: `admin`)
- `CLICKHOUSE_PASSWORD` (default: `admin`)
- `BATCH_SIZE` (default: `5000`)
- `BATCH_TIMEOUT` (default: `0.5`)
- `MAX_INFLIGHT_INSERTS` (default: `2`) — у коді вказано як більш безпечне для Windows.

Приклад (bash):
```bash
export BATCH_SIZE=20000
export MAX_INFLIGHT_INSERTS=4
python src/consumer/consumer.py
```

### 7.2 Export ClickHouse → Parquet (`to_parquet.py`)

- `PARQUET_OUT` (default: `./data/parquet`)
- `PARQUET_BATCH` (default: `100000`)
- `EXPORT_DAYS` (default: `730`)

Приклад:
```bash
cd src/analytics
PARQUET_OUT=./data/parquet PARQUET_BATCH=50000 python to_parquet.py
cd ../..
```

### 7.3 Trends loader (опційно)

`src/analytics/one_time_trends_loader.py` використовує `SERPAPI_KEY`.
Рекомендація: зберігати ключ у змінній середовища, а не у коді.

---

## 8. Типові проблеми та рішення

1) **Порти 8000 зайняті**
   - перенесіть Collector на 8001 (як у цьому README), або змініть порт Models API/Front-end.

2) **Feature builders “не знаходять Parquet”**
   - запускати `snapshot_builder1.py` і `daily_snapshot_builder1.py` з директорії `src/analytics/builders`;
   - або адаптувати шляхи на абсолютні (в коді).

3) **pip не встановлює залежності з `requrements.txt`**
   - конвертувати UTF‑16 → UTF‑8 (див. розділ 5, крок 1).

4) **Consumer “відстає” або нестабільний на Windows**
   - у коді вже виставлені більш консервативні дефолти;
   - зменшити `BATCH_SIZE`, `MAX_INFLIGHT_INSERTS`, `concurrency` генератора.

5) **Models API стартує без snapshot/моделі**
   - це очікувано, якщо ви ще не виконали кроки 7–8;
   - API спробує працювати у fallback режимі (тільки список користувачів, без предобчислених прогнозів).

---

## 9. Ліцензія та призначення

Проєкт є навчально-дослідним (під магістерську роботу) і демонструє архітектурні та інженерні підходи до побудови системи прогнозування споживчого інтересу у задачах електронної комерції.
