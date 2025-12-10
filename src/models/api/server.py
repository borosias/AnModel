from __future__ import annotations

"""
Models API Server (FastAPI)

Сервер для инференса ML-моделей с автоматической документацией.
Запуск: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
from contextlib import asynccontextmanager
from enum import Enum
from importlib import import_module
from typing import Any, Optional

import asyncpg
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===== Настройка путей =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

MODELS_DIR = os.path.join(SRC_DIR, "models", "production_models")
DEFAULT_CONTEXT_AWARE_PATH = os.path.join(MODELS_DIR, "context_aware_model1.pkl")

# Безопасная инициализация снапшота: не рушим сервер, если данных/зависимостей нет
SNAPSHOT_PATH: Optional[str] = None
try:
    daily_features_dir = os.path.join(SRC_DIR, "analytics", "data", "daily_features")
    candidate_paths = []
    if os.path.isdir(daily_features_dir):
        for f in os.listdir(daily_features_dir):
            p = os.path.join(daily_features_dir, f, "daily_snapshot1.parquet")
            if os.path.isfile(p):
                candidate_paths.append(p)
    if candidate_paths:
        SNAPSHOT_PATH = sorted(candidate_paths, reverse=True)[0]
        daily_snapshot = pd.read_parquet(SNAPSHOT_PATH)
        if "user_id" in daily_snapshot.columns:
            daily_snapshot.set_index("user_id", inplace=True)
    else:
        daily_snapshot = pd.DataFrame()
except Exception as e:
    print(f"[WARN] Failed to load daily snapshot: {e}")
    daily_snapshot = pd.DataFrame()

# Предвычисленные предсказания модели для всех пользователей снапшота
predicted_users = pd.DataFrame()
try:
    if not daily_snapshot.empty and os.path.exists(DEFAULT_CONTEXT_AWARE_PATH):
        # Импортируем модель и считаем предсказания батчем
        from models.models.context_aware import ContextAwareModel  # type: ignore

        context_model = ContextAwareModel.load(DEFAULT_CONTEXT_AWARE_PATH)
        # daily_snapshot сейчас с индексом user_id, вернём столбец user_id для join
        ds_with_id = daily_snapshot.reset_index(drop=False)
        preds_df = context_model.predict(ds_with_id)
        predicted_users = ds_with_id.join(preds_df)
        predicted_users.set_index("user_id", inplace=True)
        print(f"[INFO] Precomputed predictions for {len(predicted_users)} users from snapshot")
    else:
        print("[INFO] No daily snapshot or model file for precomputed predictions")
except Exception as e:
    print(f"[WARN] Failed to precompute predictions: {e}")
    predicted_users = pd.DataFrame()

# ===== Pydantic-схемы =====
class PredictRequest(BaseModel):
    """Запрос на предсказание."""
    records: list[dict[str, Any]] = Field(
        ...,
        description="Список записей с фичами для предсказания",
        min_length=1,
        examples=[[{
            "total_events": 100,
            "total_purchases": 3,
            "days_since_last_event": 5,
            "avg_spend_per_purchase_30d": 1200.0
        }]]
    )
    service: str | None = Field(
        default=None,
        description="Имя сервиса/модели (опционально, по умолчанию context_aware)"
    )


class PredictResponse(BaseModel):
    """Ответ с предсказаниями."""
    predictions: list[dict[str, Any]] = Field(
        ...,
        description="Список предсказаний",
        examples=[[{
            "purchase_proba": 0.73,
            "will_purchase_pred": 1,
            "days_to_next_pred": 4.2,
            "next_purchase_amount_pred": 1450.0
        }]]
    )


class ServiceStatus(str, Enum):
    LOADED = "loaded"
    ERROR = "error"
    PENDING = "pending"


class ServiceInfo(BaseModel):
    """Информация о сервисе."""
    status: ServiceStatus
    model_path: str | None = None
    error: str | None = None


class ServicesResponse(BaseModel):
    """Список доступных сервисов."""
    services: list[str]
    details: dict[str, ServiceInfo]


class HealthResponse(BaseModel):
    """Ответ health check."""
    status: str = "ok"


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    error: str
    details: str | None = None


class UserDataRequest(BaseModel):
    """Запрос данных пользователя."""
    days: int = Field(default=30, ge=1, le=365)
    model: str = Field(default="context_aware", description="Имя модели для которой нужны фичи")


class UserFeaturesResponse(BaseModel):
    """Ответ с фичами пользователя."""
    user_id: int
    features: dict[str, Any]


class UserSearchResponse(BaseModel):
    """Ответ при поиске пользователей."""
    users: list[dict[str, Any]]


class User(BaseModel):
    user_uid: str


# ===== Сервис моделей =====
class ModelService:
    """Обёртка над моделью: динамическая загрузка класса и предикт."""

    def __init__(self, model_path: str, model_class_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            module_name, class_name = model_class_path.split(":", 1)
            module = import_module(module_name)
            model_cls = getattr(module, class_name)
            self.model = model_cls.load(model_path)
        except (ImportError, OSError) as e:
            raise RuntimeError(
                f"Failed to load model '{model_class_path}': {e}"
            ) from e

    def predict(self, records: list[dict]) -> list[dict]:
        """Предсказание по списку записей."""
        if not records:
            return []

        df = pd.DataFrame.from_records(records)
        preds_df = self.model.predict(df)

        # Конвертация в JSON-совместимый формат
        return preds_df.to_dict(orient="records")


# ===== Конфигурация сервисов =====
SERVICE_CONFIGS: dict[str, dict] = {
    "context_aware": {
        "model_path": os.getenv("CONTEXT_AWARE_MODEL_PATH", DEFAULT_CONTEXT_AWARE_PATH),
        "model_class_path": os.getenv(
            "CONTEXT_AWARE_CLASS_PATH",
            "models.models.context_aware:ContextAwareModel"
        ),
    },
    # Легко добавить новые модели:
    # "cross_region": {
    #     "model_path": os.getenv("CROSS_REGION_MODEL_PATH", "..."),
    #     "model_class_path": "models.models.cross_region:CrossRegionModel",
    # },
}

# Кэш сервисов и ошибок
SERVICES: dict[str, ModelService] = {}
SERVICE_ERRORS: dict[str, str] = {}


def get_service(name: str) -> ModelService:
    """Получает сервис по имени с lazy-загрузкой."""
    if name not in SERVICE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown service: {name}")

    if name in SERVICE_ERRORS:
        raise HTTPException(status_code=503, detail=SERVICE_ERRORS[name])

    if name not in SERVICES:
        cfg = SERVICE_CONFIGS[name]
        try:
            SERVICES[name] = ModelService(
                model_path=cfg["model_path"],
                model_class_path=cfg["model_class_path"]
            )
        except (FileNotFoundError, RuntimeError) as e:
            SERVICE_ERRORS[name] = str(e)
            raise HTTPException(status_code=503, detail=str(e))

    return SERVICES[name]


def get_services_status() -> dict[str, ServiceInfo]:
    """Возвращает статус всех сервисов."""
    status = {}
    for name, cfg in SERVICE_CONFIGS.items():
        if name in SERVICES:
            status[name] = ServiceInfo(
                status=ServiceStatus.LOADED,
                model_path=cfg["model_path"]
            )
        elif name in SERVICE_ERRORS:
            status[name] = ServiceInfo(
                status=ServiceStatus.ERROR,
                error=SERVICE_ERRORS[name]
            )
        else:
            status[name] = ServiceInfo(
                status=ServiceStatus.PENDING,
                model_path=cfg["model_path"]
            )
    return status


# ===== Lifespan (опциональная предзагрузка моделей) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: можно предзагружать модели при старте."""
    # Опционально: предзагрузка моделей
    # for name in SERVICE_CONFIGS:
    #     try:
    #         get_service(name)
    #         print(f"✅ Preloaded service: {name}")
    #     except HTTPException as e:
    #         print(f"⚠️ Failed to preload {name}: {e.detail}")

    print(f"🚀 Models API started. Available services: {list(SERVICE_CONFIGS.keys())}")
    yield
    print("👋 Shutting down...")


# ===== FastAPI App =====
app = FastAPI(
    title="AnModel API",
    description="API для инференса ML-моделей маркетинговой аналитики",
    version="2.0.0",
    lifespan=lifespan,
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    # Явно указываем dev-источники, чтобы браузер корректно принимал креды
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Эндпоинты =====
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Проверка работоспособности сервера."""
    return HealthResponse()


@app.get("/services", response_model=ServicesResponse, tags=["System"])
async def list_services():
    """Список доступных сервисов и их статус."""
    return ServicesResponse(
        services=list(SERVICE_CONFIGS.keys()),
        details=get_services_status()
    )


@app.get("/model-info/{service_name}", tags=["System"])
async def get_model_info(service_name: str):
    """
    Получить информацию о модели: ожидаемые фичи, порог классификации и т.д.
    """
    model_service = get_service(service_name)
    model = model_service.model

    info = {
        "service": service_name,
        "features": [],
        "optimal_threshold": None,
        "feature_importance_top10": None,
    }

    # Получаем список фичей
    if hasattr(model, "feature_columns_") and model.feature_columns_ is not None:
        info["features"] = model.feature_columns_.tolist()

    # Порог классификации
    if hasattr(model, "optimal_threshold_"):
        info["optimal_threshold"] = model.optimal_threshold_

    # Топ-10 важных фичей
    if hasattr(model, "feature_importance_") and model.feature_importance_ is not None:
        top10 = model.feature_importance_.head(10).to_dict(orient="records")
        info["feature_importance_top10"] = top10

    # Медианы числовых фичей (для понимания "нормальных" значений)
    if hasattr(model, "numeric_medians_"):
        info["feature_medians"] = model.numeric_medians_

    return info


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        503: {"model": ErrorResponse, "description": "Model unavailable"},
    }
)
async def predict(
        request: PredictRequest,
        service: str = Query(default="context_aware", description="Имя сервиса/модели")
):
    """
    Получить предсказания модели.

    Приоритет выбора сервиса:
    1. Query parameter `?service=...`
    2. Поле `service` в теле запроса
    3. По умолчанию: `context_aware`
    """
    # Определяем сервис: query param > body > default
    service_name = service or request.service or "context_aware"

    model_service = get_service(service_name)

    try:
        predictions = model_service.predict(request.records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(predictions=predictions)


async def get_pool():
    return asyncpg.create_pool(
        user="postgres",
        password="postgres",
        database="postgres",
        host="localhost",
        port=5432
    )


@app.post(
    "/predict/{service_name}",
    response_model=PredictResponse,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Service not found"},
        503: {"model": ErrorResponse, "description": "Model unavailable"},
    }
)
async def predict_by_service(service_name: str, request: PredictRequest):
    """
    Получить предсказания конкретной модели.

    Путь `/predict/context_aware` эквивалентен `/predict?service=context_aware`
    """
    model_service = get_service(service_name)

    try:
        predictions = model_service.predict(request.records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(predictions=predictions)


@app.get("/users", tags=["Users"])
async def search_users():
    # Если есть предвычисленные предсказания — используем их
    if not predicted_users.empty:
        users_sorted = predicted_users.sort_values("purchase_proba", ascending=False, na_position="last")
        results = []
        for user_id, row in users_sorted.iterrows():
            features = row.to_dict()
            results.append({
                "user_id": str(user_id),
                "features": features
            })
        return {"users": results}

    # Fallback: старое поведение, если по какой-то причине нет снапшота/модели
    try:
        df = pd.read_parquet("../../analytics/data/users/users.parquet")
        df = df.sort_values("user_uid")
        user_ids = df["user_uid"].astype(str).tolist()

    except Exception:
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT user_uid
                    FROM users
                    ORDER BY user_uid
                    """,
                )

            user_ids = [str(r["user_uid"]) for r in rows]

        except Exception:
            raise HTTPException(status_code=500, detail="No parquet file and no DB connection")

    results = []

    for user_id in user_ids:
        features = (
            daily_snapshot.loc[user_id].to_dict()
            if (not daily_snapshot.empty and user_id in daily_snapshot.index)
            else {}
        )

        results.append({
            "user_id": user_id,
            "features": features
        })

    return {"users": results}


# ===== Запуск через uvicorn =====
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Для разработки
    )
