import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class SegmentTrendModel:
    """Модель для предсказания трендов в сегментах пользователей"""

    def __init__(self, n_segments: int = 5):
        self.n_segments = n_segments
        self.kmeans = KMeans(n_clusters=n_segments, random_state=42)
        self.scaler = StandardScaler()
        self.segment_profiles = {}
        self.is_trained = False

    def predict_segment_trends(self, user_data: Dict, product_data: Dict) -> Dict:
        """Предсказывает тренды для сегмента пользователя"""

        try:
            segment_id = self._assign_segment(user_data)
            segment_profile = self.segment_profiles.get(segment_id, {})

            # Анализ аффинности сегмента к категории товара
            category_affinity = self._calculate_category_affinity(segment_profile, product_data)

            # Анализ кросс-селл потенциала
            cross_sell_potential = self._calculate_cross_sell(segment_profile, product_data)

            # Анализ скорости роста тренда
            trend_velocity = self._calculate_trend_velocity(segment_id, product_data)

            return {
                'segment_affinity': category_affinity,
                'trend_velocity': trend_velocity,
                'cross_sell_potential': cross_sell_potential,
                'segment_size': segment_profile.get('size', 100),
                'segment_characteristics': self._get_segment_characteristics(segment_id)
            }

        except Exception as e:
            logger.error(f"Error in segment trend prediction: {e}")
            return self._fallback_segment_prediction()

    def _assign_segment(self, user_data: Dict) -> int:
        """Определяет сегмент пользователя"""
        if not self.is_trained:
            return 0  # Дефолтный сегмент если модель не обучена

        features = self._extract_segment_features(user_data)
        features_scaled = self.scaler.transform([features])
        return self.kmeans.predict(features_scaled)[0]

    def _extract_segment_features(self, user_data: Dict) -> List[float]:
        """Извлекает фичи для кластеризации сегментов"""
        return [
            user_data.get('purchase_frequency', 0),
            user_data.get('avg_order_value', 0),
            user_data.get('days_since_last_purchase', 30),
            user_data.get('total_orders', 0),
            user_data.get('preferred_category_diversity', 0),
            user_data.get('price_sensitivity', 0.5),
        ]

    def _calculate_category_affinity(self, segment_profile: Dict, product_data: Dict) -> float:
        """Вычисляет аффинность сегмента к категории товара"""
        category = product_data.get('category', 'general')
        category_affinities = segment_profile.get('category_affinities', {})

        return category_affinities.get(category, 0.5)

    def _calculate_cross_sell(self, segment_profile: Dict, product_data: Dict) -> float:
        """Вычисляет потенциал кросс-селла"""
        # Анализ сопутствующих товаров в сегменте
        base_potential = 0.3
        category = product_data.get('category', 'general')

        # Увеличиваем потенциал для категорий с высокой частотой покупок
        if category in segment_profile.get('frequent_categories', []):
            base_potential += 0.3

        # Учитываем ценовой диапазон
        price = product_data.get('price', 0)
        avg_segment_order = segment_profile.get('avg_order_value', 1000)
        if price <= avg_segment_order * 0.3:  # Сопутствующий товар дешевле основного заказа
            base_potential += 0.2

        return min(0.9, base_potential)

    def _calculate_trend_velocity(self, segment_id: int, product_data: Dict) -> float:
        """Вычисляет скорость роста тренда в сегменте"""
        # Эвристика на основе сегмента и категории
        segment_trendiness = self.segment_profiles.get(segment_id, {}).get('trend_adoption', 0.5)
        category_trend = product_data.get('category_trend_score', 0.5)

        return (segment_trendiness + category_trend) / 2

    def _get_segment_characteristics(self, segment_id: int) -> Dict:
        """Возвращает характеристики сегмента"""
        profiles = {
            0: {"name": "Бюджетные покупатели", "description": "Частые покупки недорогих товаров"},
            1: {"name": "Премиум клиенты", "description": "Редкие но дорогие покупки"},
            2: {"name": "Трендсеттеры", "description": "Ранние последователи модных трендов"},
            3: {"name": "Лояльные покупатели", "description": "Постоянные клиенты определенных брендов"},
            4: {"name": "Сезонные покупатели", "description": "Активны в определенные периоды"}
        }
        return profiles.get(segment_id, {"name": "Неизвестный сегмент", "description": ""})

    def _fallback_segment_prediction(self) -> Dict:
        """Резервное предсказание для сегментов"""
        return {
            'segment_affinity': 0.5,
            'trend_velocity': 0.3,
            'cross_sell_potential': 0.4,
            'segment_size': 1000,
            'segment_characteristics': {"name": "Общий сегмент", "description": "Резервный сегмент"}
        }

    def train_segment_model(self, user_data: pd.DataFrame):
        """Обучает модель сегментации на исторических данных"""
        logger.info("Training segment trend model...")

        try:
            # Подготовка данных
            segment_features = []
            for _, row in user_data.iterrows():
                features = self._extract_segment_features(row.to_dict())
                segment_features.append(features)

            X = np.array(segment_features)
            X_scaled = self.scaler.fit_transform(X)

            # Кластеризация
            self.kmeans.fit(X_scaled)

            # Создание профилей сегментов
            self._build_segment_profiles(user_data, self.kmeans.labels_)
            self.is_trained = True

            logger.info(f"Segment model trained with {self.n_segments} segments")

        except Exception as e:
            logger.error(f"Failed to train segment model: {e}")
            self.is_trained = False

    def _build_segment_profiles(self, user_data: pd.DataFrame, labels: np.ndarray):
        """Строит профили для каждого сегмента"""
        user_data['segment'] = labels

        for segment_id in range(self.n_segments):
            segment_users = user_data[user_data['segment'] == segment_id]

            self.segment_profiles[segment_id] = {
                'size': len(segment_users),
                'avg_order_value': segment_users['avg_order_value'].mean(),
                'purchase_frequency': segment_users['purchase_frequency'].mean(),
                'category_affinities': self._calculate_category_affinities(segment_users),
                'frequent_categories': self._get_frequent_categories(segment_users),
                'trend_adoption': self._calculate_trend_adoption(segment_users)
            }

    def _calculate_category_affinities(self, segment_users: pd.DataFrame) -> Dict:
        """Вычисляет аффинности к категориям для сегмента"""
        category_counts = segment_users['preferred_category'].value_counts(normalize=True)
        return category_counts.to_dict()

    def _get_frequent_categories(self, segment_users: pd.DataFrame) -> List:
        """Возвращает частые категории для сегмента"""
        return segment_users['preferred_category'].value_counts().head(3).index.tolist()

    def _calculate_trend_adoption(self, segment_users: pd.DataFrame) -> float:
        """Вычисляет скорость принятия трендов сегментом"""
        # Эвристика на основе разнообразия категорий и частоты покупок
        category_diversity = segment_users['preferred_category'].nunique()
        purchase_frequency = segment_users['purchase_frequency'].mean()

        return min(1.0, (category_diversity * purchase_frequency) / 100.0)