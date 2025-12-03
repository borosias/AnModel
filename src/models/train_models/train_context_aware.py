# train_snapshot_model1.py
import os
import pandas as pd

from src.models import ContextAwareModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "..", "..", "analytics", "data", "snapshots", "model1")
MODEL_PATH = os.path.join(BASE_DIR, "..", "production_models", "context_aware_model1.pkl")

def load_dataset(name: str) -> pd.DataFrame:
    path = os.path.join(SNAPSHOT_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def main():
    print("📂 Loading datasets...")
    train_df = load_dataset("train")
    val_df = load_dataset("val")
    test_df = load_dataset("test")

    print(f"Train: {len(train_df):,} rows")
    print(f"Val:   {len(val_df):,} rows")
    print(f"Test:  {len(test_df):,} rows")

    # Обучение
    model = ContextAwareModel()

    print("\n🚀 Training model...")
    val_metrics = model.fit(train_df, val_df=val_df)

    print("\n📊 Validation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    # Оценка на тесте
    print("\n🧪 Test metrics:")
    test_metrics = model.evaluate(test_df)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    print("\n📈 Top 10 important features:")
    fi = model.get_feature_importance(top_n=10)
    if fi is not None:
        for _, row in fi.iterrows():
            print(f"  {row['feature']}: {row['importance']:.1f}")

    # Сохранение модели
    print(f"\n💾 Saving model to {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    # Пример использования
    print("\n🔮 Example predictions on first 5 test rows:")
    preds = model.predict(test_df.head(5))
    print(preds)


if __name__ == "__main__":
    main()
