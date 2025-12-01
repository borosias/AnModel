# AnModel

Порядок запуску:
1) Докер
2) Инфра
3) Колектор
4) Консумер
5) Дата генератор
6) 

```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Docker
```
docker compose up -d
```

```
uvicorn src.collector.app:app --reload --host 0.0.0.0 --port 8000
```

```
python src/consumer/consumer.py
```

```
python src/data_gen/generate.py
```

