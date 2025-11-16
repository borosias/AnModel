import numpy as np
from typing import Dict, List, Any
import logging
from .personal_interest_predictor import PersonalInterestModel
from .segment_trend_predictor import SegmentTrendModel
from .regional_demand_predictor import RegionalDemandModel
from .global_trend_predictor import GlobalTrendModel

logger = logging.getLogger(__name__)


class DemandForecastingOrchestrator:
    """Оркестратор для объединения прогнозов со всех уровней"""

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()

        # Инициализация моделей
        self.personal_model = PersonalInterestModel()
        self.segment_model = SegmentTrendModel()
        self.regional_model = RegionalDemandModel()
        self.trend_model = GlobalTrendModel()

        self.initialized = False

    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию"""
        return {
            'weights': {
                'personal': 0.35,
                'segment': 0.25,
                'regional': 0.20,
                'trend': 0.20
            },
            'thresholds': {
                'high_confidence': 0.7,
                'medium_confidence': 0.5,
                'urgent_action': 0.8
            },
            'business_rules': {
                'min_inventory_days': 3,
                'max_recommendations_per_user': 10,
                'regional_optimization': True
            }
        }

    def initialize_models(self, training_data: Dict = None):
        """Инициализирует все модели"""
        try:
            logger.info("Initializing forecasting models...")

            # Здесь будет загрузка предобученных моделей или обучение на данных
            if training_data:
                self._train_models(training_data)
            else:
                logger.info("Using heuristic models without training data")

            self.initialized = True
            logger.info("All models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.initialized = False

    def predict_optimal_offer(self, user_data: Dict, product_data: Dict,
                              context: Dict) -> Dict:
        """Главный метод - предсказывает оптимальное предложение"""

        try:
            # 1. Собираем прогнозы со всех уровней
            personal_pred = self.personal_model.predict_individual_interest(
                user_data, product_data, context
            )

            segment_pred = self.segment_model.predict_segment_trends(
                user_data, product_data
            )

            regional_pred = self.regional_model.predict_regional_demand(
                user_data.get('region_data', {}), product_data, context.get('time_horizon', '7d')
            )

            trend_pred = self.trend_model.predict_global_trends(
                product_data, context.get('external_data', {})
            )

            # 2. Объединяем прогнозы
            final_score = self._ensemble_predictions(
                personal_pred, segment_pred, regional_pred, trend_pred
            )

            # 3. Формируем рекомендацию
            recommendation = self._build_recommendation(
                final_score, personal_pred, segment_pred, regional_pred, trend_pred,
                user_data, product_data, context
            )

            logger.info(f"Prediction completed for user {user_data.get('user_id', 'unknown')}")
            return recommendation

        except Exception as e:
            logger.error(f"Error in prediction orchestration: {e}")
            return self._fallback_recommendation(user_data, product_data)

    def _ensemble_predictions(self, personal: Dict, segment: Dict,
                              regional: Dict, trend: Dict) -> float:
        """Объединяет прогнозы с весами"""
        weights = self.config['weights']

        # Взвешенное среднее
        score = (
                weights['personal'] * personal['personal_interest_score'] +
                weights['segment'] * segment['segment_affinity'] +
                weights['regional'] * regional['regional_demand_score'] +
                weights['trend'] * trend['overall_trend_score']
        )

        # Корректировки на основе дополнительных факторов
        score = self._apply_business_rules(score, personal, segment, regional, trend)

        return min(1.0, max(0.0, score))

    def _apply_business_rules(self, score: float, personal: Dict, segment: Dict,
                              regional: Dict, trend: Dict) -> float:
        """Применяет бизнес-правила к итоговому скору"""

        # Буст для высокого персонального интереса
        if personal['personal_interest_score'] > 0.8:
            score *= 1.1

        # Буст для виральных трендов
        if trend['viral_potential'] > 0.8:
            score *= 1.15

        # Буст для регионов с высокой доставкой
        if regional['delivery_optimization'] > 0.9:
            score *= 1.05

        # Снижение для низкого кросс-селл потенциала
        if segment['cross_sell_potential'] < 0.3:
            score *= 0.9

        return score

    def _build_recommendation(self, final_score: float, personal: Dict, segment: Dict,
                              regional: Dict, trend: Dict, user_data: Dict,
                              product_data: Dict, context: Dict) -> Dict:
        """Строит финальную рекомендацию"""

        # Определяем уровень уверенности
        confidence = self._calculate_confidence(personal, segment, regional, trend)

        # Определяем рекомендуемое действие
        recommended_action = self._suggest_action(final_score, confidence, context)

        # Оцениваем бизнес-влияние
        expected_impact = self._estimate_business_impact(final_score, product_data, regional)

        # Определяем срочность
        time_sensitivity = self._calculate_urgency(trend, regional, context)

        # Определяем уровень персонализации
        personalization_level = self._determine_personalization_level(
            final_score, personal, segment
        )

        return {
            'user_id': user_data.get('user_id'),
            'product_id': product_data.get('product_id'),
            'final_interest_score': final_score,
            'confidence': confidence,
            'recommended_action': recommended_action,
            'expected_impact': expected_impact,
            'time_sensitivity': time_sensitivity,
            'personalization_level': personalization_level,
            'component_scores': {
                'personal': personal['personal_interest_score'],
                'segment': segment['segment_affinity'],
                'regional': regional['regional_demand_score'],
                'trend': trend['overall_trend_score']
            },
            'metadata': {
                'segment_info': segment['segment_characteristics'],
                'regional_info': regional['regional_specifics'],
                'trend_info': trend['trend_prediction'],
                'timestamp': self._get_current_timestamp()
            }
        }

    def _calculate_confidence(self, personal: Dict, segment: Dict,
                              regional: Dict, trend: Dict) -> float:
        """Вычисляет общую уверенность предсказания"""
        confidences = [
            personal.get('confidence', 0.5),
            min(segment.get('segment_affinity', 0.5), segment.get('cross_sell_potential', 0.5)),
            regional.get('delivery_optimization', 0.5),
            trend.get('overall_trend_score', 0.5)
        ]

        return sum(confidences) / len(confidences)

    def _suggest_action(self, score: float, confidence: float, context: Dict) -> str:
        """Предлагает действие на основе скора и уверенности"""
        thresholds = self.config['thresholds']

        if score > thresholds['urgent_action'] and confidence > thresholds['high_confidence']:
            return "show_on_main_page"
        elif score > 0.7 and confidence > thresholds['medium_confidence']:
            return "send_personalized_email"
        elif score > 0.6:
            return "include_in_recommendations"
        elif score > 0.4:
            return "show_on_category_page"
        else:
            return "no_action"

    def _estimate_business_impact(self, score: float, product_data: Dict,
                                  regional: Dict) -> str:
        """Оценивает бизнес-влияние рекомендации"""
        price = product_data.get('price', 0)
        expected_volume = regional.get('expected_volume', 0)

        potential_revenue = price * expected_volume * score

        if potential_revenue > 10000:
            return "high_impact"
        elif potential_revenue > 5000:
            return "medium_impact"
        elif potential_revenue > 1000:
            return "low_impact"
        else:
            return "minimal_impact"

    def _calculate_urgency(self, trend: Dict, regional: Dict, context: Dict) -> str:
        """Определяет срочность действия"""
        trend_momentum = trend.get('trend_momentum', 0)
        trend_direction = trend.get('trend_prediction', {}).get('direction', 'stable')

        if trend_direction == "up" and trend_momentum > 0.5:
            return "urgent"
        elif regional.get('seasonal_factor', 1.0) > 1.2:
            return "seasonal_urgent"
        elif trend_direction == "down" and trend_momentum < -0.3:
            return "declining"
        else:
            return "normal"

    def _determine_personalization_level(self, final_score: float,
                                         personal: Dict, segment: Dict) -> str:
        """Определяет уровень персонализации"""
        personal_score = personal['personal_interest_score']
        segment_score = segment['segment_affinity']

        if personal_score > 0.8 and final_score > 0.7:
            return "hyper_personalized"
        elif segment_score > 0.7 and final_score > 0.6:
            return "segment_optimized"
        elif final_score > 0.5:
            return "context_aware"
        else:
            return "generic"

    def _get_current_timestamp(self) -> str:
        """Возвращает текущую timestamp строку"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _fallback_recommendation(self, user_data: Dict, product_data: Dict) -> Dict:
        """Резервная рекомендация при ошибках"""
        return {
            'user_id': user_data.get('user_id'),
            'product_id': product_data.get('product_id'),
            'final_interest_score': 0.5,
            'confidence': 0.1,
            'recommended_action': 'no_action',
            'expected_impact': 'minimal_impact',
            'time_sensitivity': 'normal',
            'personalization_level': 'generic',
            'component_scores': {
                'personal': 0.5,
                'segment': 0.5,
                'regional': 0.5,
                'trend': 0.5
            },
            'metadata': {
                'segment_info': {"name": "Fallback", "description": "Error occurred"},
                'regional_info': {"type": "Fallback", "consumption_level": "unknown"},
                'trend_info': {"direction": "stable", "confidence": 0.1, "timeframe": "unknown"},
                'timestamp': self._get_current_timestamp(),
                'error': True
            }
        }

    def _train_models(self, training_data: Dict):
        """Обучает модели на предоставленных данных"""
        logger.info("Training models with provided data...")

        # Здесь будет логика обучения всех моделей
        # Пока просто логируем
        logger.info(f"Training data keys: {list(training_data.keys())}")

        # В реальности здесь будет:
        # self.personal_model.train_model(training_data['personal'])
        # self.segment_model.train_segment_model(training_data['segment'])
        # self.regional_model.train_regional_model(training_data['regional'])

        logger.info("Model training completed (heuristic mode)")

    def batch_predict(self, user_product_pairs: List[Dict]) -> List[Dict]:
        """Пакетное предсказание для нескольких пар пользователь-товар"""
        results = []

        for pair in user_product_pairs:
            user_data = pair.get('user_data', {})
            product_data = pair.get('product_data', {})
            context = pair.get('context', {})

            prediction = self.predict_optimal_offer(user_data, product_data, context)
            results.append(prediction)

        return results

    def get_system_status(self) -> Dict:
        """Возвращает статус системы"""
        return {
            'initialized': self.initialized,
            'models_ready': {
                'personal': self.personal_model.is_trained,
                'segment': self.segment_model.is_trained,
                'regional': self.regional_model.is_trained,
                'trend': self.trend_model.is_initialized
            },
            'config': self.config,
            'timestamp': self._get_current_timestamp()
        }