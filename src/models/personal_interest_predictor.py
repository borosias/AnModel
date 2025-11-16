import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class PersonalInterestModel:
    """Модель для предсказания персонального интереса пользователя"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_builder = None
        self.is_trained = False

        if model_path:
            self.load_model(model_path)

    def predict_individual_interest(self, user_data: Dict, product_data: Dict, context: Dict) -> Dict:
        """Предсказывает персональный интерес пользователя к товару"""

        try:
            # Если модель не загружена, используем эвристику
            if not self.is_trained:
                return self._heuristic_prediction(user_data, product_data, context)

            # Подготовка фичей для модели
            features = self._prepare_features(user_data, product_data, context)

            # Предсказание моделью
            with torch.no_grad():
                prediction = self.model(features)
                probabilities = torch.softmax(prediction, dim=1)

            return {
                'personal_interest_score': probabilities[0][2].item(),  # Вероятность высокого интереса
                'expected_engagement': probabilities[0][1].item(),  # Вероятность среднего интереса
                'purchase_probability': self._calculate_purchase_prob(user_data, product_data),
                'confidence': self._calculate_confidence(features),
                'feature_importance': self._get_feature_importance(user_data, product_data)
            }

        except Exception as e:
            logger.error(f"Error in personal interest prediction: {e}")
            return self._fallback_prediction(user_data, product_data)

    def _prepare_features(self, user_data: Dict, product_data: Dict, context: Dict) -> torch.Tensor:
        """Подготавливает фичи для модели"""
        features = []

        # Пользовательские фичи
        features.extend([
            user_data.get('purchase_frequency', 0),
            user_data.get('avg_order_value', 0),
            user_data.get('days_since_last_purchase', 30),
            user_data.get('total_orders', 0),
            user_data.get('preferred_category_match', 0),
            user_data.get('price_sensitivity', 0.5),
        ])

        # Товарные фичи
        features.extend([
            product_data.get('price', 0),
            product_data.get('discount_pct', 0),
            product_data.get('avg_rating', 3.0),
            product_data.get('review_count', 0),
        ])

        # Контекстные фичи
        features.extend([
            context.get('time_of_day', 12) / 24.0,
            context.get('day_of_week', 0) / 7.0,
            context.get('is_weekend', 0),
            context.get('session_duration', 0) / 3600.0,  # в часах
        ])

        return torch.FloatTensor([features])

    def _heuristic_prediction(self, user_data: Dict, product_data: Dict, context: Dict) -> Dict:
        """Эвристическое предсказание когда модель не обучена"""

        # Простая логика на основе правил
        base_score = 0.5

        # Учет истории покупок
        if user_data.get('total_orders', 0) > 5:
            base_score += 0.1

        # Учет соответствия категории
        if user_data.get('preferred_category') == product_data.get('category'):
            base_score += 0.2

        # Учет скидки
        if product_data.get('discount_pct', 0) > 10:
            base_score += 0.15

        # Учет времени суток (вечером выше вероятность)
        hour = context.get('time_of_day', 12)
        if 18 <= hour <= 22:
            base_score += 0.1

        return {
            'personal_interest_score': min(0.95, base_score),
            'expected_engagement': min(0.9, base_score * 0.8),
            'purchase_probability': min(0.85, base_score * 0.7),
            'confidence': 0.6,
            'feature_importance': {'heuristic': 1.0}
        }

    def _calculate_purchase_prob(self, user_data: Dict, product_data: Dict) -> float:
        """Вычисляет вероятность покупки на основе исторических данных"""
        # Упрощенная логика - можно заменить ML моделью
        price_ratio = user_data.get('avg_order_value', 1000) / max(product_data.get('price', 1), 1)
        if price_ratio > 2:
            return 0.7
        elif price_ratio > 1:
            return 0.5
        else:
            return 0.3

    def _calculate_confidence(self, features: torch.Tensor) -> float:
        """Вычисляет уверенность предсказания"""
        # Простая эвристика - можно заменить на оценку неопределенности модели
        feature_variance = torch.var(features).item()
        return max(0.1, 1.0 - feature_variance / 10.0)

    def _get_feature_importance(self, user_data: Dict, product_data: Dict) -> Dict:
        """Возвращает важность фич для отладки"""
        return {
            'purchase_frequency': 0.8,
            'category_match': 0.7,
            'price_affordability': 0.6,
            'discount': 0.5,
            'time_context': 0.3
        }

    def _fallback_prediction(self, user_data: Dict, product_data: Dict) -> Dict:
        """Резервное предсказание при ошибках"""
        return {
            'personal_interest_score': 0.5,
            'expected_engagement': 0.4,
            'purchase_probability': 0.3,
            'confidence': 0.1,
            'feature_importance': {'fallback': 1.0}
        }

    def load_model(self, model_path: str):
        """Загружает обученную модель"""
        try:
            # Здесь будет логика загрузки модели
            logger.info(f"Loading model from {model_path}")
            self.is_trained = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_trained = False

    def train_model(self, training_data: pd.DataFrame):
        """Обучает модель на исторических данных"""
        logger.info("Training personal interest model...")
        # Здесь будет логика обучения
        self.is_trained = True