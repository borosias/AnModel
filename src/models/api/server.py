import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import import_module
from urllib.parse import urlparse, parse_qs

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure project src is on sys.path for module imports like `models.*`
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Default path to production models (can be overridden by env vars)
MODELS_DIR = os.path.join(SRC_DIR, "models", "production_models")
DEFAULT_CONTEXT_AWARE_PATH = os.path.join(MODELS_DIR, "context_aware_model1.pkl")


class ModelService:
    """Обёртка над моделью: динамическая загрузка класса и веса, предикт.

    По умолчанию грузит ContextAwareModel, но класс можно переопределить через dotted path,
    например: "models.models.context_aware:ContextAwareModel".
    """

    def __init__(self, model_path: str, model_class_path: str = "models.models.context_aware:ContextAwareModel") -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        module_name, class_name = model_class_path.split(":", 1)
        module = import_module(module_name)
        model_cls = getattr(module, class_name)
        self.model = model_cls.load(model_path)

    def predict_from_records(self, records: list[dict]) -> list[dict]:
        """Принимает список словарей с фичами, возвращает список словарей с предсказаниями."""
        if not records:
            return []

        df = pd.DataFrame.from_records(records)
        preds_df = self.model.predict(df)

        # Сериализация: используем to_json -> json.loads, чтобы гарантировать JSON‑совместимые типы
        records_json = preds_df.to_json(orient="records")
        return json.loads(records_json)

# ===== Регистрация сервисов (можно расширять ассортимент моделей) =====
SERVICE_CONFIGS: dict[str, dict] = {
    "context_aware": {
        "model_path": os.getenv("CONTEXT_AWARE_MODEL_PATH", DEFAULT_CONTEXT_AWARE_PATH),
        "model_class_path": os.getenv(
            "CONTEXT_AWARE_CLASS_PATH", "models.models.context_aware:ContextAwareModel"
        ),
    }
}

# Lazy cache of instantiated services
SERVICES: dict[str, ModelService] = {}


def get_service(name: str) -> ModelService:
    cfg = SERVICE_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown service: {name}")
    if name not in SERVICES:
        SERVICES[name] = ModelService(
            model_path=cfg["model_path"], model_class_path=cfg["model_class_path"]
        )
    return SERVICES[name]


class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status_code: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")  # для dev: разрешаем CORS
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        """CORS preflight"""
        self._set_headers(200)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
            return

        if path == "/services":
            self._set_headers(200)
            self.wfile.write(json.dumps({"services": list(SERVICE_CONFIGS.keys())}).encode("utf-8"))
            return

        # 404 по умолчанию
        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # Поддерживаем /predict, /predict/<service> и ?service=
        if path == "/predict" or path.startswith("/predict/"):
            service_in_path = None
            parts = [p for p in path.split("/") if p]
            if len(parts) == 2:  # ["predict"]
                service_in_path = None
            elif len(parts) == 3:  # ["predict", "<service>"]
                service_in_path = parts[2]
            self.handle_predict(service_in_path)
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

    def handle_predict(self, service_in_path: str | None = None) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
        except Exception:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "invalid json"}).encode("utf-8"))
            return

        records = data.get("records")
        if not isinstance(records, list):
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "`records` must be a list"}).encode("utf-8"))
            return

        # Определяем сервис: приоритет path > body.service > query ?service=
        parsed = urlparse(self.path)
        query_service = parse_qs(parsed.query).get("service", [None])[0]
        body_service = data.get("service") if isinstance(data, dict) else None
        service_name = service_in_path or body_service or query_service or "context_aware"

        try:
            service = get_service(service_name)
            predictions = service.predict_from_records(records)
        except ValueError as exc:
            # Например, неизвестный сервис
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return
        except Exception as exc:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return

        self._set_headers(200)
        self.wfile.write(json.dumps({"predictions": predictions}).encode("utf-8"))


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server_address = (host, port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Serving Models API on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()