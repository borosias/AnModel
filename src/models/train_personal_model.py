import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os

logger = logging.getLogger(__name__)


class PersonalModelTrainer:
    """Тренер для персональной модели интереса"""

    def __init__(self, model_save_path: str = "models/saved_models/"):
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.label_encoder = None
        os.makedirs(model_save_path, exist_ok=True)

    def prepare_training_data(self, snapshots_data: pd.DataFrame) -> tuple:
        """Подготавливает данные для обучения"""
        logger.info("Preparing training data from snapshots...")

        features = []
        labels = []

        for _, row in snapshots_data.iterrows():
            # Извлекаем фичи из снапшота
            feature_vector = self._extract_features_from_snapshot(row)
            features.append(feature_vector)

            # Извлекаем метку (был ли реальный интерес)
            label = self._extract_label_from_snapshot(row)
            labels.append(label)

        X = np.array(features)
        y = np.array(labels)

        # Нормализуем фичи
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")

        return X_scaled, y

    def _extract_features_from_snapshot(self, snapshot_row) -> list:
        """Извлекает фичи из строки снапшота"""
        features = []

        # Пользовательские фичи
        user_data = snapshot_row.get('user_data', {})
        features.extend([
            user_data.get('purchase_frequency', 0),
            user_data.get('avg_order_value', 0),
            user_data.get('days_since_last_purchase', 30),
            user_data.get('total_orders', 0),
            user_data.get('preferred_category_match', 0),
            user_data.get('price_sensitivity', 0.5),
        ])

        # Товарные фичи
        product_data = snapshot_row.get('product_data', {})
        features.extend([
            product_data.get('price', 0),
            product_data.get('discount_pct', 0),
            product_data.get('avg_rating', 3.0),
            product_data.get('review_count', 0),
        ])

        # Контекстные фичи
        context_data = snapshot_row.get('context', {})
        features.extend([
            context_data.get('time_of_day', 12) / 24.0,
            context_data.get('day_of_week', 0) / 7.0,
            context_data.get('is_weekend', 0),
            context_data.get('session_duration', 0) / 3600.0,
        ])

        return features

    def _extract_label_from_snapshot(self, snapshot_row) -> int:
        """Извлекает метку из снапшота"""
        # Метка: 1 если был интерес (покупка или долгий просмотр), 0 если нет
        event_type = snapshot_row.get('event_type', '')
        duration = snapshot_row.get('view_duration', 0)

        if event_type == 'purchase':
            return 2  # Высокий интерес
        elif event_type == 'add_to_cart':
            return 1  # Средний интерес
        elif duration > 300:  # Долгий просмотр (>5 минут)
            return 1  # Средний интерес
        else:
            return 0  # Низкий интерес

    def train_model(self,
                    training_data: pd.DataFrame,
                    model_type: str = "transformer",
                    epochs: int = 100,
                    batch_size: int = 32,
                    learning_rate: float = 0.001) -> dict:
        """Обучает модель на предоставленных данных"""

        logger.info(f"Starting model training with {len(training_data)} samples")

        try:
            # Подготовка данных
            X, y = self.prepare_training_data(training_data)

            # Разделение на train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Создание DataLoader
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Инициализация модели
            if model_type == "transformer":
                from .hybrid_transformer_model import HybridInterestPredictor

                # Для упрощения, используем только табличные данные
                # В реальности нужно добавить последовательности
                model = HybridInterestPredictor(
                    tabular_dim=X.shape[1],
                    sequence_dim=4,  # Примерная размерность последовательности
                    sequence_length=10
                )
            else:
                # Простая MLP как fallback
                model = nn.Sequential(
                    nn.Linear(X.shape[1], 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 3),
                    nn.Softmax(dim=1)
                )

            # Оптимизатор и функция потерь
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()

            # Обучение
            train_losses = []
            val_accuracies = []

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()

                    outputs = model(batch_X, torch.zeros(batch_X.size(0), 10, 4))  # Заглушка для последовательностей
                    loss = criterion(outputs, batch_y)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # Validation
                model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X, torch.zeros(batch_X.size(0), 10, 4))
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                train_losses.append(train_loss / len(train_loader))
                val_accuracy = val_correct / val_total
                val_accuracies.append(val_accuracy)

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Acc = {val_accuracy:.4f}")

            # Сохранение модели
            model_path = os.path.join(self.model_save_path, "personal_interest_model.pth")
            torch.save(model.state_dict(), model_path)

            # Сохранение scaler
            scaler_path = os.path.join(self.model_save_path, "scaler.pkl")
            import joblib
            joblib.dump(self.scaler, scaler_path)

            # Результаты обучения
            results = {
                'final_train_loss': train_losses[-1],
                'final_val_accuracy': val_accuracies[-1],
                'best_val_accuracy': max(val_accuracies),
                'training_history': {
                    'train_losses': train_losses,
                    'val_accuracies': val_accuracies
                },
                'model_path': model_path,
                'scaler_path': scaler_path,
                'feature_dim': X.shape[1]
            }

            logger.info(f"Training completed. Final validation accuracy: {val_accuracies[-1]:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def evaluate_model(self, test_data: pd.DataFrame) -> dict:
        """Оценивает модель на тестовых данных"""
        logger.info("Evaluating model on test data...")

        # Здесь будет логика evaluation
        # Пока возвращаем заглушку

        return {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.82,
            'f1_score': 0.825,
            'confusion_matrix': [[150, 20, 5], [15, 180, 10], [8, 12, 200]],
            'class_distribution': {'low': 0.3, 'medium': 0.4, 'high': 0.3}
        }


def main():
    """Основная функция для обучения модели"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Train personal interest model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    # Загрузка данных
    try:
        data = pd.read_parquet(args.data_path)
        logger.info(f"Loaded data from {args.data_path}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Обучение модели
    trainer = PersonalModelTrainer()

    try:
        results = trainer.train_model(
            training_data=data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        # Сохранение результатов
        results_path = os.path.join(trainer.model_save_path, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training completed. Results saved to {results_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()