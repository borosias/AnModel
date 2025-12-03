import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CrossRegionModel:
    """ПРАВИЛЬНАЯ модель кросс-регионального спроса - БЕЗ НЕЙРОСЕТИ!"""

    def __init__(self):
        self.models = {}  # model for each region: demand and revenue
        self.scalers = {}  # scaler for each region
        self.region_stats = {}  # статистики по регионам
        self.feature_importance = {}  # важность фичей

    def _prepare_features(self, df: pd.DataFrame, region: str, is_training: bool = True):
        """Подготовка фичей для конкретного региона"""

        region_df = df[df['region'] == region].copy()

        # Автоматически выбираем фичи (исключая таргеты и служебные)
        exclude_patterns = ['target_', 'snapshot_date', 'region']
        feature_cols = [
            col for col in region_df.columns
            if not any(pattern in col for pattern in exclude_patterns)
        ]

        # Только числовые фичи
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(region_df[col])]

        X = region_df[numeric_cols].fillna(0)

        # Масштабирование
        if is_training:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[region] = scaler
        else:
            scaler = self.scalers.get(region)
            X_scaled = scaler.transform(X) if scaler else X.values

        return X_scaled, numeric_cols

    def train(self, df: pd.DataFrame):
        """Обучение отдельной модели для каждого региона"""

        logger.info("🌍 Training Cross-Region Models (Gradient Boosting)...")

        regions = df['region'].unique()
        logger.info(f"  Found {len(regions)} regions: {regions}")

        for region in regions:
            try:
                region_df = df[df['region'] == region]

                if len(region_df) < 10:
                    logger.warning(f"  ⚠️ Region {region}: Only {len(region_df)} samples, skipping")
                    continue

                # Подготовка фичей
                X, feature_names = self._prepare_features(df, region, is_training=True)

                # Целевые переменные
                if 'target_purchase_count' in region_df.columns:
                    y_demand = region_df['target_purchase_count'].values
                else:
                    logger.warning(f"  ⚠️ Region {region}: No target_purchase_count, skipping")
                    continue

                if 'target_total_spent' in region_df.columns:
                    y_revenue = region_df['target_total_spent'].values
                else:
                    y_revenue = None

                # Обучаем модель спроса
                demand_model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                demand_model.fit(X, y_demand)

                # Обучаем модель выручки (если есть данные)
                revenue_model = None
                if y_revenue is not None:
                    revenue_model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42
                    )
                    revenue_model.fit(X, y_revenue)

                # Сохраняем статистики
                self.region_stats[region] = {
                    'n_samples': len(region_df),
                    'avg_demand': np.mean(y_demand),
                    'std_demand': np.std(y_demand),
                    'features': feature_names
                }

                # Сохраняем модель
                self.models[region] = {
                    'demand': demand_model,
                    'revenue': revenue_model,
                    'feature_importance': demand_model.feature_importances_
                }

                logger.info(f"  ✅ Region {region}: Trained on {len(region_df)} samples")

            except Exception as e:
                logger.error(f"  ❌ Region {region}: Error {e}")

        logger.info(f"✅ Trained models for {len(self.models)} regions")

    def predict(self, df: pd.DataFrame) -> Dict:
        """Предсказание спроса для регионов"""

        results = {}

        for region in df['region'].unique():
            region_df = df[df['region'] == region]

            if region in self.models:
                # Подготовка фичей
                X, _ = self._prepare_features(df, region, is_training=False)

                # Предсказание
                demand_pred = self.models[region]['demand'].predict(X)

                if self.models[region]['revenue']:
                    revenue_pred = self.models[region]['revenue'].predict(X)
                else:
                    # Оценка выручки: спрос * средняя цена
                    revenue_pred = demand_pred * self.region_stats[region].get('avg_price', 100)

                # Анализ трансфера спроса
                transfer_ops = self._analyze_transfer(region, demand_pred, df)

                results[region] = {
                    'predicted_demand': float(np.mean(demand_pred)),
                    'predicted_revenue': float(np.mean(revenue_pred)),
                    'confidence_interval': [
                        float(np.percentile(demand_pred, 25)),
                        float(np.percentile(demand_pred, 75))
                    ],
                    'transfer_opportunities': transfer_ops,
                    'n_samples': len(region_df)
                }
            else:
                # Регион не обучался - используем среднее по всем регионам
                global_avg = np.mean([stats['avg_demand'] for stats in self.region_stats.values()])
                results[region] = {
                    'predicted_demand': float(global_avg),
                    'predicted_revenue': float(global_avg * 100),
                    'warning': 'Region not trained, using global average',
                    'transfer_opportunities': []
                }

        return results

    def _analyze_transfer(self, region: str, predicted_demand: np.ndarray, df: pd.DataFrame) -> List[Dict]:
        """Анализ возможностей трансфера спроса"""

        opportunities = []

        if region not in self.region_stats:
            return opportunities

        current_avg = self.region_stats[region]['avg_demand']
        predicted = np.mean(predicted_demand)

        # Если прогноз выше обычного на 20% - возможен избыток
        if predicted > current_avg * 1.2:
            # Ищем регионы с низким спросом
            for other_region, stats in self.region_stats.items():
                if other_region != region:
                    other_avg = stats['avg_demand']

                    # Если у другого региона спрос ниже
                    if other_avg < current_avg * 0.8:
                        transfer_amount = min(
                            predicted - current_avg,  # избыток
                            current_avg - other_avg  # дефицит другого
                        ) * 0.3  # 30% от разницы

                        if transfer_amount > 0:
                            opportunities.append({
                                'from_region': region,
                                'to_region': other_region,
                                'transfer_amount': float(transfer_amount),
                                'reason': f'High demand in {region}, low demand in {other_region}',
                                'estimated_impact': f'Revenue increase: ${transfer_amount * 100:.0f}'
                            })

        return opportunities

    def get_recommendations(self, region_data: Dict) -> Dict:
        """Конкретные рекомендации для региона"""

        region = region_data.get('region')

        if region not in self.models:
            return {'error': f'Region {region} not trained'}

        # Создаем DataFrame для предсказания
        df = pd.DataFrame([region_data])

        predictions = self.predict(df)

        if region in predictions:
            pred = predictions[region]

            recommendations = {
                'region': region,
                'demand_forecast': pred['predicted_demand'],
                'revenue_forecast': pred['predicted_revenue'],
                'actions': []
            }

            # Генерация действий
            if pred['transfer_opportunities']:
                for transfer in pred['transfer_opportunities']:
                    recommendations['actions'].append({
                        'type': 'demand_transfer',
                        'description': f"Transfer {transfer['transfer_amount']:.1f} units to {transfer['to_region']}",
                        'reason': transfer['reason'],
                        'priority': 'high' if transfer['transfer_amount'] > 10 else 'medium'
                    })

            # Добавляем общие рекомендации
            if pred['predicted_demand'] > self.region_stats[region]['avg_demand'] * 1.5:
                recommendations['actions'].append({
                    'type': 'stock_increase',
                    'description': 'Increase inventory by 20%',
                    'reason': 'High demand forecast',
                    'priority': 'high'
                })

            return recommendations

        return {'error': 'No predictions available'}

    def save(self, path: str):
        """Сохранение модели"""
        state = {
            'models': self.models,
            'scalers': self.scalers,
            'region_stats': self.region_stats
        }
        joblib.dump(state, path)
        logger.info(f"💾 Model saved to {path}")

    def load(self, path: str):
        """Загрузка модели"""
        state = joblib.load(path)
        self.models = state['models']
        self.scalers = state['scalers']
        self.region_stats = state['region_stats']
        logger.info(f"📂 Model loaded from {path}")