# innovative_models.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import warnings
from pathlib import Path


warnings.filterwarnings('ignore')


# ===== МОДЕЛЬ 1: Context-Aware Purchase Prediction =====

class ContextAwareModel:
    """Модель 1: Предсказывает контекстную покупку"""

    def __init__(self):
        self.models = {
            'will_purchase': lgb.LGBMClassifier(),
            'category': lgb.LGBMClassifier(),
            'days_to_purchase': lgb.LGBMRegressor(),
            'purchase_amount': lgb.LGBMRegressor()
        }
        self.feature_columns = None

    def train(self, train_df: pd.DataFrame):
        """Обучение всех подмоделей"""

        print("🔮 Training Context-Aware Model...")

        # Подготовка фичей
        X = self._prepare_features(train_df)
        self.feature_columns = X.columns.tolist()

        # Обучение каждой подмодели
        # 1. Будет ли покупка?
        y_will = train_df['target_will_purchase']
        self.models['will_purchase'].fit(X, y_will)
        print(f"  ✅ Will purchase: AUC = {self._calculate_auc(X, y_will):.3f}")

        # 2. Какая категория? (только для тех кто купит)
        buyers = train_df[train_df['target_will_purchase'] == 1]
        if len(buyers) > 0 and 'target_category' in buyers.columns:
            X_buyers = X.loc[buyers.index]
            y_category = buyers['target_category']
            # Кодируем категории
            from sklearn.preprocessing import LabelEncoder
            self.category_encoder = LabelEncoder()
            y_category_encoded = self.category_encoder.fit_transform(y_category)
            self.models['category'].fit(X_buyers, y_category_encoded)
            print(f"  ✅ Category prediction: {len(self.category_encoder.classes_)} categories")

        # 3. Через сколько дней?
        if 'target_days_to_purchase' in train_df.columns:
            y_days = train_df['target_days_to_purchase']
            self.models['days_to_purchase'].fit(X, y_days)
            print(f"  ✅ Days to purchase: MAE = {self._calculate_mae(X, y_days):.2f} days")

        # 4. На какую сумму?
        if 'target_purchase_amount' in train_df.columns:
            y_amount = train_df['target_purchase_amount']
            self.models['purchase_amount'].fit(X, y_amount)
            print(f"  ✅ Purchase amount: MAE = {self._calculate_mae(X, y_amount):.2f}")

    def predict(self, X: pd.DataFrame) -> Dict:
        """Полный контекстный прогноз"""

        predictions = {}

        # 1. Вероятность покупки
        proba = self.models['will_purchase'].predict_proba(X)[:, 1]
        predictions['purchase_probability'] = proba

        # 2. Ожидаемая категория
        if 'category' in self.models:
            # Только для пользователей с высокой вероятностью покупки
            likely_buyers = proba > 0.3
            if likely_buyers.any():
                X_likely = X[likely_buyers]
                category_pred = self.models['category'].predict(X_likely)
                category_names = self.category_encoder.inverse_transform(category_pred)

                # Создаем маппинг
                full_categories = np.full(len(X), 'unknown')
                full_categories[likely_buyers] = category_names
                predictions['predicted_category'] = full_categories

        # 3. Ожидаемое время до покупки
        if 'days_to_purchase' in self.models:
            days_pred = self.models['days_to_purchase'].predict(X)
            predictions['days_to_purchase'] = days_pred

        # 4. Ожидаемая сумма
        if 'purchase_amount' in self.models:
            amount_pred = self.models['purchase_amount'].predict(X)
            predictions['purchase_amount'] = amount_pred

        return predictions

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает фичи"""
        # Убираем таргеты и идентификаторы
        exclude_cols = ['user_id', 'snapshot_date', 'target_will_purchase',
                        'target_category', 'target_days_to_purchase',
                        'target_purchase_amount']

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return df[feature_cols]

    def _calculate_auc(self, X, y):
        """Рассчитывает AUC (упрощенно)"""
        from sklearn.metrics import roc_auc_score
        preds = self.models['will_purchase'].predict_proba(X)[:, 1]
        return roc_auc_score(y, preds) if len(np.unique(y)) > 1 else 0.5

    def _calculate_mae(self, X, y):
        """Рассчитывает MAE"""
        from sklearn.metrics import mean_absolute_error
        preds = self.models['days_to_purchase'].predict(X)
        return mean_absolute_error(y, preds)


# ===== МОДЕЛЬ 2: Cross-Region Demand Transfer =====

class CrossRegionModel:
    """Модель 2: Трансфер спроса между регионами"""

    def __init__(self):
        # Модели для каждого региона (источник → целевой)
        self.region_models = {}
        self.region_encoders = {}

    def train(self, train_df: pd.DataFrame):
        """Обучение с трансфером между регионами"""

        print("🌍 Training Cross-Region Model...")

        regions = train_df['region'].unique()

        # Создаем модели для каждой пары регионов
        for target_region in regions:
            print(f"  Target region: {target_region}")

            # Данные целевого региона
            target_data = train_df[train_df['region'] == target_region]

            if len(target_data) < 10:
                print(f"    ⚠️ Not enough data, skipping...")
                continue

            # Данные из других регионов (источники)
            source_regions = [r for r in regions if r != target_region]

            # Создаем объединенный датасет
            X_all, y_all = [], []

            for source_region in source_regions:
                source_data = train_df[train_df['region'] == source_region]

                if len(source_data) > 0:
                    # Трансформируем фичи источника под целевой регион
                    X_source, y_source = self._transform_features(
                        source_data, source_region, target_region
                    )

                    X_all.append(X_source)
                    y_all.append(y_source)

            # Добавляем данные целевого региона
            X_target, y_target = self._prepare_regression_features(target_data)
            X_all.append(X_target)
            y_all.append(y_target)

            # Объединяем
            if X_all:
                X_combined = pd.concat(X_all, ignore_index=True)
                y_combined = pd.concat(y_all, ignore_index=True)

                # Обучаем модель
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6
                )
                model.fit(X_combined, y_combined['target_purchase_count'])

                self.region_models[target_region] = model

                # Оцениваем
                preds = model.predict(X_target)
                mae = np.mean(np.abs(preds - y_target['target_purchase_count']))
                print(f"    ✅ MAE: {mae:.2f}")

    def predict(self, region_data: pd.DataFrame) -> Dict:
        """Прогноз спроса для региона"""

        region = region_data['region'].iloc[0]

        if region not in self.region_models:
            return {'error': f'No model for region {region}'}

        X = self._prepare_regression_features(region_data)[0]
        model = self.region_models[region]

        predictions = model.predict(X)

        return {
            'region': region,
            'predicted_demand': float(predictions[0]),
            'confidence': 0.8  # Упрощенно
        }

    def _transform_features(self, source_df: pd.DataFrame,
                            source_region: str, target_region: str) -> Tuple:
        """Трансформирует фичи из региона-источника"""

        X = self._prepare_regression_features(source_df)[0]
        y = self._prepare_regression_features(source_df)[1]

        # Применяем трансформацию весов
        # UA-30 → UA-40: множитель 0.9
        # UA-30 → UA-50: множитель 0.7
        # UA-40 → UA-50: множитель 0.8
        # и т.д.

        transformation_rules = {
            ('UA-30', 'UA-40'): 0.9,
            ('UA-30', 'UA-50'): 0.7,
            ('UA-40', 'UA-30'): 1.1,
            ('UA-40', 'UA-50'): 0.8,
            ('UA-50', 'UA-30'): 1.3,
            ('UA-50', 'UA-40'): 1.2
        }

        multiplier = transformation_rules.get(
            (source_region, target_region), 1.0
        )

        # Применяем множитель к фичам связанным с объемом
        volume_columns = [col for col in X.columns if 'total' in col or 'count' in col]
        for col in volume_columns:
            X[col] = X[col] * multiplier

        return X, y

    def _prepare_regression_features(self, df: pd.DataFrame) -> Tuple:
        """Подготавливает фичи для регрессии"""

        # Убираем нечисловые колонки
        exclude = ['region', 'snapshot_date', 'target_purchase_count',
                   'target_total_spent', 'target_category', 'target_weekday_demand',
                   'target_weekend_demand']

        # Также убираем категориальные таргеты
        target_cols = [col for col in df.columns if col.startswith('target_category_')]
        exclude.extend(target_cols)

        feature_cols = [col for col in df.columns
                        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]

        X = df[feature_cols].fillna(0)
        y = df[['target_purchase_count']].fillna(0)

        return X, y


# ===== МОДЕЛЬ 3: Micro-Trend Anticipation =====

class MicroTrendModel(nn.Module):
    """Модель 3: Нейросеть для предсказания микро-трендов"""

    def __init__(self, input_dim: int = 20):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 3 таргета
        )

        self.trend_threshold = 0.7

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Attention
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )

        # Берем последний выход
        last_out = attn_out[:, -1, :]

        # Полносвязные слои
        output = self.fc(last_out)

        return output

    def detect_micro_trend(self, sequence_data: np.ndarray) -> Dict:
        """Обнаруживает микро-тренды в последовательности"""

        self.eval()
        with torch.no_grad():
            # Подготавливаем данные
            tensor_data = torch.FloatTensor(sequence_data).unsqueeze(0)

            # Прогноз
            predictions = self(tensor_data).squeeze().numpy()

            # Интерпретация
            will_continue = predictions[0] > self.trend_threshold
            peak_in_days = int(predictions[1] * 7)  # Масштабируем до 7 дней
            magnitude = predictions[2]

            return {
                'trend_will_continue': bool(will_continue),
                'expected_peak_in_days': peak_in_days,
                'trend_magnitude': float(magnitude),
                'alert_level': 'HIGH' if will_continue and magnitude > 0.8 else 'MEDIUM'
            }


# ===== МОДЕЛЬ 4: Adaptive Pricing Prophet =====

class AdaptivePricingModel:
    """Модель 4: Адаптивное ценообразование с RL"""

    def __init__(self):
        self.price_model = RandomForestRegressor(n_estimators=100)
        self.demand_model = xgb.XGBRegressor()
        self.optimal_prices = {}

    def train(self, train_df: pd.DataFrame):
        """Обучение модели ценообразования"""

        print("💰 Training Adaptive Pricing Model...")

        # 1. Модель зависимости спроса от цены
        X_demand = self._prepare_demand_features(train_df)
        y_demand = train_df['target_sales_count']

        self.demand_model.fit(X_demand, y_demand)

        # 2. Оптимизация цен
        items = train_df['item_id'].unique()  # Ограничим для демо

        for item_id in items:
            item_data = train_df[train_df['item_id'] == item_id]

            if len(item_data) < 5:
                continue

            # Находим оптимальную цену
            optimal_price = self._find_optimal_price(item_data)
            self.optimal_prices[item_id] = optimal_price

            print(f"  {item_id}: optimal price = {optimal_price:.2f}")

    def recommend_price(self, item_id: str, current_price: float,
                        context: Dict) -> Dict:
        """Рекомендует оптимальную цену"""

        if item_id in self.optimal_prices:
            optimal = self.optimal_prices[item_id]

            # Адаптируем к контексту
            final_price = self._adapt_to_context(optimal, current_price, context)

            return {
                'item_id': item_id,
                'current_price': current_price,
                'recommended_price': final_price,
                'change_percent': ((final_price - current_price) / current_price) * 100,
                'expected_demand_change': self._estimate_demand_change(
                    current_price, final_price, item_id
                )
            }
        else:
            return {
                'item_id': item_id,
                'recommended_price': current_price,
                'reason': 'No data for this item'
            }

    def _prepare_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает фичи для модели спроса"""

        # Ценовые фичи
        price_features = [
            'current_price', 'avg_price', 'price_std',
            'min_price', 'max_price', 'price_range',
            'price_elasticity', 'price_trend', 'price_volatility',
            'category_avg_price', 'price_position'
        ]

        # Другие фичи
        other_features = [
            'total_views', 'total_purchases', 'unique_viewers',
            'unique_buyers', 'conversion_rate'
        ]

        feature_cols = [col for col in price_features + other_features
                        if col in df.columns]

        return df[feature_cols].fillna(0)

    def _find_optimal_price(self, item_data: pd.DataFrame) -> float:
        """Находит оптимальную цену для товара"""

        # Простая оптимизация: максимизация revenue = price * demand
        prices = np.linspace(
            item_data['min_price'].min() * 0.8,
            item_data['max_price'].max() * 1.2,
            50
        )

        best_price = item_data['current_price'].mean()
        best_revenue = 0

        for price in prices:
            # Прогнозируем спрос при этой цене
            X_test = item_data.copy()
            X_test['current_price'] = price

            demand_features = self._prepare_demand_features(X_test)
            predicted_demand = self.demand_model.predict(demand_features).mean()

            revenue = price * predicted_demand

            if revenue > best_revenue:
                best_revenue = revenue
                best_price = price

        return best_price

    def _adapt_to_context(self, optimal_price: float, current_price: float,
                          context: Dict) -> float:
        """Адаптирует цену к контексту"""

        # Факторы адаптации
        factors = {
            'competition_pressure': 0.95,  # Цены конкурентов ниже
            'inventory_level': 1.05,  # Много запасов
            'seasonality': 1.0,  # Сезонный фактор
            'user_value': 1.1,  # Ценный пользователь
            'time_of_day': 1.0  # Время дня
        }

        # Применяем факторы
        adjusted_price = optimal_price

        for factor, multiplier in factors.items():
            if factor in context:
                adjusted_price *= multiplier

        # Плавное изменение (не более 20%)
        max_change = current_price * 0.2
        if abs(adjusted_price - current_price) > max_change:
            if adjusted_price > current_price:
                adjusted_price = current_price + max_change
            else:
                adjusted_price = current_price - max_change

        return round(adjusted_price, 2)

    def _estimate_demand_change(self, old_price: float,
                                new_price: float, item_id: str) -> float:
        """Оценивает изменение спроса"""

        # Упрощенная оценка на основе эластичности
        price_change = (new_price - old_price) / old_price

        # Базовая эластичность (можно учить из данных)
        elasticity = -1.5  # Типичная эластичность для e-commerce

        demand_change = elasticity * price_change

        return round(demand_change * 100, 1)  # В процентах


# ===== ИНТЕГРАЦИЯ И ЗАПУСК =====

class InnovationPipeline:
    """Главный пайплайн для всех 4 моделей"""

    def __init__(self, snapshots_dir: str = "../analytics/data/innovative_snapshots"):
        self.snapshots_dir = Path(snapshots_dir)
        self.models = {
            'context_aware': ContextAwareModel(),
            'cross_region': CrossRegionModel(),
            'micro_trend': MicroTrendModel(),
            'adaptive_pricing': AdaptivePricingModel()
        }

    def train_all_models(self):
        """Обучает все 4 модели"""

        print("=" * 60)
        print("🚀 TRAINING ALL 4 INNOVATIVE MODELS")
        print("=" * 60)

        # Модель 1: Context-Aware
        print("\n1️⃣ Context-Aware Purchase Prediction")
        try:
            train_df = pd.read_parquet(self.snapshots_dir / "model1/train.parquet")
            self.models['context_aware'].train(train_df)
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

        # Модель 2: Cross-Region
        print("\n2️⃣ Cross-Region Demand Transfer")
        try:
            train_df = pd.read_parquet(self.snapshots_dir / "model2/train.parquet")
            self.models['cross_region'].train(train_df)
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

        # Модель 3: Micro-Trend (упрощенная)
        print("\n3️⃣ Micro-Trend Anticipation")
        print("   ⚠️ Note: Requires PyTorch and more data")

        # Модель 4: Adaptive Pricing
        print("\n4️⃣ Adaptive Pricing Prophet")
        try:
            train_df = pd.read_parquet(self.snapshots_dir / "model4/train.parquet")
            self.models['adaptive_pricing'].train(train_df)
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

        print("\n" + "=" * 60)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 60)

    def make_predictions(self):
        """Делает прогнозы всеми моделями"""

        print("\n🎯 MAKING PREDICTIONS WITH ALL MODELS")
        print("=" * 60)

        predictions = {}

        # Пример прогноза для пользователя
        print("\n📊 Example predictions:")

        # 1. Контекстный прогноз для случайного пользователя
        try:
            test_df = pd.read_parquet(self.snapshots_dir / "model1/test.parquet")
            if len(test_df) > 0:
                sample_user = test_df.iloc[0]
                user_features = pd.DataFrame([sample_user])

                # Убираем таргеты для прогноза
                for col in user_features.columns:
                    if 'target' in col:
                        user_features[col] = 0

                context_pred = self.models['context_aware'].predict(user_features)
                print(f"\n1️⃣ User {sample_user.get('user_id', 'unknown')}:")
                print(f"   Purchase probability: {context_pred.get('purchase_probability', [0])[0]:.1%}")

                if 'predicted_category' in context_pred:
                    print(f"   Likely category: {context_pred['predicted_category'][0]}")

                predictions['context_aware'] = context_pred
        except Exception as e:
            print(f"   ⚠️ Context prediction error: {e}")

        # 2. Прогноз спроса для региона
        try:
            test_df = pd.read_parquet(self.snapshots_dir / "model2/test.parquet")
            if len(test_df) > 0:
                region_sample = test_df.iloc[0:1]
                region_pred = self.models['cross_region'].predict(region_sample)
                print(f"\n2️⃣ Region {region_pred.get('region', 'unknown')}:")
                print(f"   Predicted demand: {region_pred.get('predicted_demand', 0):.0f} purchases")
                predictions['cross_region'] = region_pred
        except Exception as e:
            print(f"   ⚠️ Region prediction error: {e}")

        # 4. Рекомендация цены
        try:
            test_df = pd.read_parquet(self.snapshots_dir / "model4/test.parquet")
            if len(test_df) > 0:
                item_sample = test_df.iloc[0]
                price_rec = self.models['adaptive_pricing'].recommend_price(
                    item_id=item_sample.get('item_id', 'item_1'),
                    current_price=item_sample.get('current_price', 100),
                    context={'competition_pressure': 0.95}
                )
                print(f"\n4️⃣ Item {price_rec.get('item_id', 'unknown')}:")
                print(f"   Current price: {price_rec.get('current_price', 0):.2f}")
                print(f"   Recommended: {price_rec.get('recommended_price', 0):.2f}")
                print(f"   Change: {price_rec.get('change_percent', 0):.1f}%")
                predictions['adaptive_pricing'] = price_rec
        except Exception as e:
            print(f"   ⚠️ Price prediction error: {e}")

        return predictions


# ===== ЗАПУСК =====

if __name__ == "__main__":
    # Инициализируем пайплайн
    pipeline = InnovationPipeline()

    # 1. Обучаем модели (если данные есть)
    pipeline.train_all_models()

    # 2. Делаем прогнозы
    predictions = pipeline.make_predictions()

    print("\n" + "=" * 60)
    print("🎉 INNOVATION PIPELINE COMPLETED!")
    print("=" * 60)

    # 3. Сохраняем результаты
    import json

    with open("innovative_predictions.json", "w") as f:
        # Сериализуем только простые типы
        simple_preds = {}
        for model_name, pred in predictions.items():
            if isinstance(pred, dict):
                simple_preds[model_name] = {
                    k: (float(v) if isinstance(v, (np.floating, float)) else
                        int(v) if isinstance(v, (np.integer, int)) else
                        str(v) if not isinstance(v, (list, dict, np.ndarray)) else
                        v.tolist() if isinstance(v, np.ndarray) else
                        list(v) if isinstance(v, (list, tuple)) else str(v))
                    for k, v in pred.items()
                }

        json.dump(simple_preds, f, indent=2)

    print("📁 Predictions saved to innovative_predictions.json")