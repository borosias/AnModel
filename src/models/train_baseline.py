import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import argparse
import joblib
from pathlib import Path


def load_snapshots(snapshot_dir: str):
    """Загружает снэпшоты"""
    train = pd.read_parquet(f"{snapshot_dir}/train.parquet")
    val = pd.read_parquet(f"{snapshot_dir}/val.parquet")
    return train, val


def prepare_features(df: pd.DataFrame):
    """Подготавливает фичи для обучения (совместимо с snapshot features)"""

    # Выбираем числовые фичи (исключаем ID и даты)
    feature_columns = [
        'total_events', 'total_spent', 'days_since_last_event',
        'count_purchase', 'count_product_view', 'count_add_to_cart',
        'purchase_count', 'total_spent_purchases', 'avg_purchase_value',
        'avg_session_events', 'max_session_events', 'total_sessions'
    ]

    # Оставляем только существующие колонки
    available_features = [col for col in feature_columns if col in df.columns]

    X = df[available_features].fillna(0)
    y = df['target_purchase']

    return X, y, available_features


def train_baseline_model(X_train, y_train, X_val, y_val):
    """Обучает baseline модель"""

    print("Training Random Forest baseline...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Валидация
    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    print(f"Train AUC: {train_auc:.4f}")
    print(f"Val AUC: {val_auc:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 features:")
    print(feature_importance.head(10))

    return model, feature_importance


def main():
    parser = argparse.ArgumentParser(description='Train baseline model on snapshots')
    parser.add_argument('--snapshot-dir', type=str, default='data/snapshots', help='Snapshot directory')
    parser.add_argument('--model-dir', type=str, default='models', help='Model output directory')

    args = parser.parse_args()

    # Создаем директорию для моделей
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    print("=== Baseline Model Training ===")

    # Загружаем данные
    train_df, val_df = load_snapshots(args.snapshot_dir)
    print(f"Train: {len(train_df):,} samples")
    print(f"Val: {len(val_df):,} samples")

    # Подготавливаем фичи
    X_train, y_train, features = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)

    print(f"Using {len(features)} features: {features}")

    # Обучаем модель
    model, feature_importance = train_baseline_model(X_train, y_train, X_val, y_val)

    # Сохраняем модель и фичи
    joblib.dump(model, f"{args.model_dir}/baseline_model.pkl")
    feature_importance.to_csv(f"{args.model_dir}/feature_importance.csv", index=False)

    print(f"\nModel saved to {args.model_dir}/")
    print(f"Feature importance saved to {args.model_dir}/feature_importance.csv")


if __name__ == "__main__":
    main()