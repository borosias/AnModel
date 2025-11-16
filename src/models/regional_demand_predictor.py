import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class RegionalDemandModel:
    """Модель для предсказания регионального спроса"""

    def __init__(self):
        self.regional_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.region_profiles = {}
        self.is_trained = False

    def predict_regional_demand(self, region_data: Dict, product_data: Dict, time_horizon: str = "7d") -> Dict:
        """Предсказывает спрос в регионе"""

        try:
            region_id = region_data.get('region_id', 'unknown')
            region_profile = self.region_profiles.get(region_id, {})

            # Базовый спрос на основе региона
            base_demand = self._calculate_base_demand(region_profile, product_data)

            # Сезонный фактор
            seasonal_factor = self._calculate_seasonal_factor(region_data, time_horizon)

            # Оптимизация логистики
            delivery_optimization = self._calculate_delivery_optimization(region_profile, product_data)

            # Популярность категории в регионе
            category_popularity = self._get_category_popularity(region_profile, product_data)

            return {
                'regional_demand_score': min(1.0, base_demand * seasonal_factor),
                'seasonal_factor': seasonal_factor,
                'delivery_optimization': delivery_optimization,
                'category_popularity': category_popularity,
                'expected_volume': self._estimate_volume(base_demand, region_profile),
                'regional_specifics': self._get_regional_specifics(region_id)
            }

        except Exception as e:
            logger.error(f"Error in regional demand prediction: {e}")
            return self._fallback_regional_prediction()

    def _calculate_base_demand(self, region_profile: Dict, product_data: Dict) -> float:
        """Вычисляет базовый спрос в регионе"""
        category = product_data.get('category', 'general')
        price = product_data.get('price', 0)

        # Спрос на категорию в регионе
        category_demand = region_profile.get('category_demands', {}).get(category, 0.5)

        # Корректировка на цену (дорогие товары имеют меньший спрос)
        price_factor = 1.0 - min(0.7, price / 10000.0)  # Нормализуем цену

        # Корректировка на экономический профиль региона
        economic_factor = region_profile.get('economic_index', 0.5)

        return category_demand * price_factor * economic_factor

    def _calculate_seasonal_factor(self, region_data: Dict, time_horizon: str) -> float:
        """Вычисляет сезонный фактор"""
        # Простая сезонная эвристика
        month = region_data.get('current_month', 1)
        region_id = region_data.get('region_id', 'general')

        # Сезонные паттерны по регионам
        seasonal_patterns = {
            'kyiv': [0.8, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8],  # Пик летом
            'lviv': [0.7, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7],  # Пик летом
            'odesa': [0.9, 0.8, 1.0, 1.1, 1.3, 1.4, 1.5, 1.4, 1.2, 1.1, 1.0, 0.9],  # Туристический пик
        }

        pattern = seasonal_patterns.get(region_id, [1.0] * 12)
        return pattern[month - 1]

    def _calculate_delivery_optimization(self, region_profile: Dict, product_data: Dict) -> float:
        """Вычисляет оптимальность доставки"""
        delivery_infrastructure = region_profile.get('delivery_score', 0.5)
        product_size = product_data.get('size_category', 'medium')

        # Корректировка на размер товара
        size_factors = {'small': 1.0, 'medium': 0.9, 'large': 0.7, 'xlarge': 0.5}
        size_factor = size_factors.get(product_size, 0.8)

        return delivery_infrastructure * size_factor

    def _get_category_popularity(self, region_profile: Dict, product_data: Dict) -> float:
        """Возвращает популярность категории в регионе"""
        category = product_data.get('category', 'general')
        return region_profile.get('category_popularity', {}).get(category, 0.5)

    def _estimate_volume(self, demand_score: float, region_profile: Dict) -> int:
        """Оценивает объем продаж"""
        population_factor = region_profile.get('population_density', 0.5)
        base_volume = 100  # Базовый объем

        return int(base_volume * demand_score * population_factor * 10)

    def _get_regional_specifics(self, region_id: str) -> Dict:
        """Возвращает специфику региона"""
        regional_info = {
            'kyiv': {"type": "столица", "consumption_level": "высокий", "trend_sensitivity": "высокая"},
            'lviv': {"type": "культурный центр", "consumption_level": "средний", "trend_sensitivity": "высокая"},
            'odesa': {"type": "туристический", "consumption_level": "средний", "trend_sensitivity": "средняя"},
            'kharkiv': {"type": "промышленный", "consumption_level": "средний", "trend_sensitivity": "средняя"},
            'dnipro': {"type": "промышленный", "consumption_level": "средний", "trend_sensitivity": "низкая"},
        }
        return regional_info.get(region_id,
                                 {"type": "неизвестно", "consumption_level": "средний", "trend_sensitivity": "средняя"})

    def _fallback_regional_prediction(self) -> Dict:
        """Резервное предсказание для регионов"""
        return {
            'regional_demand_score': 0.5,
            'seasonal_factor': 1.0,
            'delivery_optimization': 0.7,
            'category_popularity': 0.5,
            'expected_volume': 50,
            'regional_specifics': {"type": "общий", "consumption_level": "средний", "trend_sensitivity": "средняя"}
        }

    def train_regional_model(self, regional_data: pd.DataFrame):
        """Обучает модель регионального спроса"""
        logger.info("Training regional demand model...")

        try:
            # Здесь будет логика обучения на исторических данных по регионам
            # Пока используем эвристические профили

            self._build_region_profiles(regional_data)
            self.is_trained = True

            logger.info("Regional model trained with heuristic profiles")

        except Exception as e:
            logger.error(f"Failed to train regional model: {e}")
            self.is_trained = False

    def _build_region_profiles(self, regional_data: pd.DataFrame):
        """Строит профили регионов на основе данных"""
        # Примерные профили - в реальности будут вычисляться из данных
        self.region_profiles = {
            'kyiv': {
                'population_density': 0.9,
                'economic_index': 0.8,
                'delivery_score': 0.9,
                'category_demands': {'electronics': 0.8, 'clothing': 0.7, 'home': 0.6},
                'category_popularity': {'electronics': 0.9, 'clothing': 0.8, 'home': 0.7}
            },
            'lviv': {
                'population_density': 0.7,
                'economic_index': 0.6,
                'delivery_score': 0.8,
                'category_demands': {'electronics': 0.6, 'clothing': 0.8, 'home': 0.5},
                'category_popularity': {'electronics': 0.7, 'clothing': 0.9, 'home': 0.6}
            },
            'odesa': {
                'population_density': 0.6,
                'economic_index': 0.5,
                'delivery_score': 0.7,
                'category_demands': {'electronics': 0.5, 'clothing': 0.7, 'home': 0.4},
                'category_popularity': {'electronics': 0.6, 'clothing': 0.8, 'home': 0.5}
            }
        }