import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelUtils:
    """Утилиты для работы с моделями"""

    @staticmethod
    def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
        """Вычисляет веса классов для несбалансированных данных"""
        class_counts = np.bincount(labels)
        total_samples = len(labels)

        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)

    @staticmethod
    def save_model_checkpoint(model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              epoch: int,
                              loss: float,
                              filepath: str):
        """Сохраняет чекпоинт модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    @staticmethod
    def load_model_checkpoint(model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              filepath: str) -> Dict:
        """Загружает чекпоинт модели"""
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint

    @staticmethod
    def predict_batch(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: str = 'cpu') -> np.ndarray:
        """Выполняет пакетное предсказание"""
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    inputs, _ = batch
                else:
                    inputs = batch

                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)


class PredictionAnalyzer:
    """Анализатор предсказаний модели"""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def analyze_predictions(self,
                            data_loader: torch.utils.data.DataLoader,
                            true_labels: np.ndarray,
                            class_names: List[str] = None) -> Dict:
        """Анализирует предсказания модели"""

        predictions = ModelUtils.predict_batch(self.model, data_loader)

        # Метрики классификации
        report = classification_report(true_labels, predictions,
                                       output_dict=True,
                                       target_names=class_names)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        # Дополнительные метрики
        accuracy = np.mean(predictions == true_labels)

        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions_distribution': pd.Series(predictions).value_counts().to_dict(),
            'true_distribution': pd.Series(true_labels).value_counts().to_dict()
        }

        return results

    def plot_confusion_matrix(self,
                              data_loader: torch.utils.data.DataLoader,
                              true_labels: np.ndarray,
                              class_names: List[str],
                              save_path: str = None):
        """Визуализирует confusion matrix"""

        predictions = ModelUtils.predict_batch(self.model, data_loader)
        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def feature_importance_analysis(self,
                                    model: torch.nn.Module,
                                    feature_names: List[str],
                                    test_data: torch.Tensor,
                                    num_samples: int = 100) -> Dict:
        """Анализирует важность фич с помощью permutation importance"""

        model.eval()
        original_predictions = ModelUtils.predict_batch(model, [(test_data,)])
        original_accuracy = np.mean(original_predictions == original_predictions)  # Baseline

        importance_scores = {}

        for i, feature_name in enumerate(feature_names):
            # Permute feature
            permuted_data = test_data.clone()
            permuted_data[:, i] = permuted_data[torch.randperm(permuted_data.size(0)), i]

            # Predict with permuted feature
            permuted_predictions = ModelUtils.predict_batch(model, [(permuted_data,)])
            permuted_accuracy = np.mean(permuted_predictions == original_predictions)

            # Importance score
            importance = original_accuracy - permuted_accuracy
            importance_scores[feature_name] = max(0, importance)

        # Нормализуем scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {k: v / total_importance for k, v in importance_scores.items()}

        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))