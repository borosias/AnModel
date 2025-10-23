# test_imports.py в корне проекта
try:
    from src.collector.app import app
    from src.consumer.consumer import normalize_event
    from src.features.build_features import compute_features_optimized
    print("✅ Все импорты работают!")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")