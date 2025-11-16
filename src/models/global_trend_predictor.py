import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class GlobalTrendModel:
    """Модель для анализа глобальных трендов из внешних источников"""

    def __init__(self):
        self.trend_keywords = {}
        self.sentiment_lexicon = {}
        self.trend_history = {}
        self.is_initialized = False

    def predict_global_trends(self, product_data: Dict, external_sources: Dict) -> Dict:
        """Анализирует глобальные тренды для товара"""

        try:
            product_name = product_data.get('name', '')
            product_category = product_data.get('category', '')

            # Анализ поисковых трендов
            search_trend_score = self._analyze_search_trends(product_name, product_category, external_sources)

            # Анализ социальных медиа
            social_engagement = self._analyze_social_engagement(product_name, product_category, external_sources)

            # Анализ вирального потенциала
            viral_potential = self._calculate_viral_potential(product_data, external_sources)

            # Анализ упоминаний в медиа
            news_mentions = self._count_news_mentions(product_name, external_sources)

            # Общий тренд-скор
            overall_trend_score = self._calculate_overall_trend(
                search_trend_score, social_engagement, viral_potential, news_mentions
            )

            return {
                'viral_potential': viral_potential,
                'search_trend_score': search_trend_score,
                'social_engagement': social_engagement,
                'news_mentions': news_mentions,
                'overall_trend_score': overall_trend_score,
                'trend_momentum': self._calculate_trend_momentum(product_name),
                'seasonal_relevance': self._check_seasonal_relevance(product_category),
                'trend_prediction': self._predict_trend_future(overall_trend_score, search_trend_score)
            }

        except Exception as e:
            logger.error(f"Error in global trend prediction: {e}")
            return self._fallback_trend_prediction()

    def _analyze_search_trends(self, product_name: str, category: str, external_sources: Dict) -> float:
        """Анализирует поисковые тренды"""
        search_data = external_sources.get('search_trends', {})

        # Эвристика на основе ключевых слов
        keywords = self._extract_keywords(product_name)
        trend_score = 0.0
        keyword_count = 0

        for keyword in keywords:
            if keyword in search_data:
                trend_score += search_data[keyword].get('trend_score', 0.5)
                keyword_count += 1

        if keyword_count > 0:
            return trend_score / keyword_count

        # Fallback: оценка на основе категории
        category_trends = {
            'electronics': 0.7, 'clothing': 0.8, 'home': 0.6,
            'beauty': 0.75, 'sports': 0.65, 'books': 0.5
        }
        return category_trends.get(category, 0.5)

    def _analyze_social_engagement(self, product_name: str, category: str, external_sources: Dict) -> float:
        """Анализирует вовлеченность в социальных сетях"""
        social_data = external_sources.get('social_media', {})

        engagement_metrics = []

        # Анализ упоминаний
        mentions = social_data.get('mentions', 0)
        if mentions > 0:
            normalized_mentions = min(1.0, mentions / 1000.0)  # Нормализуем до 0-1
            engagement_metrics.append(normalized_mentions)

        # Анализ вовлеченности (лайки, репосты)
        engagement_rate = social_data.get('engagement_rate', 0)
        if engagement_rate > 0:
            normalized_engagement = min(1.0, engagement_rate / 10.0)  # Нормализуем
            engagement_metrics.append(normalized_engagement)

        # Анализ настроений
        sentiment = social_data.get('sentiment', 0.5)
        engagement_metrics.append(sentiment)

        if engagement_metrics:
            return sum(engagement_metrics) / len(engagement_metrics)

        # Fallback на основе категории
        category_engagement = {
            'electronics': 0.6, 'clothing': 0.8, 'home': 0.5,
            'beauty': 0.9, 'sports': 0.7, 'books': 0.4
        }
        return category_engagement.get(category, 0.5)

    def _calculate_viral_potential(self, product_data: Dict, external_sources: Dict) -> float:
        """Вычисляет виральный потенциал товара"""
        viral_factors = []

        # Уникальность продукта
        uniqueness = product_data.get('uniqueness_factor', 0.5)
        viral_factors.append(uniqueness)

        # Визуальная привлекательность
        visual_appeal = product_data.get('visual_appeal', 0.5)
        viral_factors.append(visual_appeal)

        # Ценовая доступность для масс
        price = product_data.get('price', 0)
        price_accessibility = 1.0 - min(0.8, price / 5000.0)  # Дешевые товары более виральные
        viral_factors.append(price_accessibility)

        # Наличие социального доказательства
        social_proof = external_sources.get('social_media', {}).get('influencer_mentions', 0)
        normalized_social_proof = min(1.0, social_proof / 10.0)
        viral_factors.append(normalized_social_proof)

        return sum(viral_factors) / len(viral_factors)

    def _count_news_mentions(self, product_name: str, external_sources: Dict) -> int:
        """Считает упоминания в новостях"""
        news_data = external_sources.get('news', [])

        mentions = 0
        keywords = self._extract_keywords(product_name)

        for news_item in news_data:
            content = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
            for keyword in keywords:
                if keyword.lower() in content:
                    mentions += 1
                    break

        return mentions

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлекает ключевые слова из текста"""
        if not text:
            return []

        # Простая токенизация
        words = re.findall(r'\b\w+\b', text.lower())
        # Убираем стоп-слова и короткие слова
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'a', 'an'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        return keywords[:5]  # Возвращаем до 5 ключевых слов

    def _calculate_trend_momentum(self, product_name: str) -> float:
        """Вычисляет момент тренда (растет/падает)"""
        # Эвристика - в реальности нужно анализировать исторические данные
        current_time = datetime.now()

        # Простая логика: если товар новый, момент положительный
        product_age = getattr(self, '_get_product_age', lambda x: 365)(product_name)
        if product_age < 30:  # Новый товар
            return 0.8
        elif product_age < 90:  # Недавний товар
            return 0.6
        else:  # Старый товар
            return 0.4

    def _check_seasonal_relevance(self, category: str) -> float:
        """Проверяет сезонную релевантность"""
        current_month = datetime.now().month

        seasonal_categories = {
            'winter_clothing': [12, 1, 2],
            'summer_clothing': [6, 7, 8],
            'swimwear': [5, 6, 7, 8],
            'winter_sports': [11, 12, 1, 2],
            'beach_equipment': [5, 6, 7, 8],
            'school_supplies': [8, 9],
            'christmas_decor': [11, 12]
        }

        for cat, months in seasonal_categories.items():
            if cat in category.lower() and current_month in months:
                return 0.9

        return 0.5  # Нейтральная сезонность

    def _predict_trend_future(self, current_score: float, search_score: float) -> Dict:
        """Предсказывает будущее тренда"""
        momentum = self._calculate_trend_momentum_from_scores(current_score, search_score)

        if current_score > 0.8 and momentum > 0:
            return {"direction": "up", "confidence": 0.8, "timeframe": "1-2 недели"}
        elif current_score > 0.6 and momentum > 0:
            return {"direction": "up", "confidence": 0.6, "timeframe": "2-4 недели"}
        elif current_score < 0.4 and momentum < 0:
            return {"direction": "down", "confidence": 0.7, "timeframe": "1-2 недели"}
        else:
            return {"direction": "stable", "confidence": 0.5, "timeframe": "неопределенно"}

    def _calculate_trend_momentum_from_scores(self, current_score: float, search_score: float) -> float:
        """Вычисляет момент тренда на основе скоринга"""
        # Простая эвристика - разница между текущим и поисковым скором
        return search_score - current_score

    def _calculate_overall_trend(self, search_score: float, social_score: float,
                                 viral_score: float, news_mentions: int) -> float:
        """Вычисляет общий тренд-скор"""
        weights = {
            'search': 0.4,
            'social': 0.3,
            'viral': 0.2,
            'news': 0.1
        }

        news_score = min(1.0, news_mentions / 10.0)  # Нормализуем новости

        overall = (
                weights['search'] * search_score +
                weights['social'] * social_score +
                weights['viral'] * viral_score +
                weights['news'] * news_score
        )

        return min(1.0, overall)

    def _fallback_trend_prediction(self) -> Dict:
        """Резервное предсказание трендов"""
        return {
            'viral_potential': 0.5,
            'search_trend_score': 0.5,
            'social_engagement': 0.5,
            'news_mentions': 0,
            'overall_trend_score': 0.5,
            'trend_momentum': 0.0,
            'seasonal_relevance': 0.5,
            'trend_prediction': {"direction": "stable", "confidence": 0.3, "timeframe": "неопределенно"}
        }

    def update_trend_data(self, new_trend_data: Dict):
        """Обновляет данные о трендах"""
        try:
            # Обновление поисковых трендов
            if 'search_trends' in new_trend_data:
                self.trend_keywords.update(new_trend_data['search_trends'])

            # Обновление социальных данных
            if 'social_metrics' in new_trend_data:
                self._update_social_metrics(new_trend_data['social_metrics'])

            self.is_initialized = True
            logger.info("Trend data updated successfully")

        except Exception as e:
            logger.error(f"Failed to update trend data: {e}")

    def _update_social_metrics(self, social_metrics: Dict):
        """Обновляет метрики социальных сетей"""
        # Здесь будет логика обновления социальных метрик
        pass