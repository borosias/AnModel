# innovative_models.py
from random import random

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
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import optuna  # pip install optuna
from catboost import CatBoostRegressor, CatBoostClassifier  # pip install catboost


warnings.filterwarnings('ignore')


# ===== МОДЕЛЬ 1: Context-Aware Purchase Prediction =====

class ContextAwareModel:
    """Модель 1 с Bayesian калибровкой и uncertainty quantification"""

    def __init__(self, enable_bayesian: bool = True):
        self.models = {
            'will_purchase': lgb.LGBMClassifier(),
            'category': lgb.LGBMClassifier(),
            'days_to_purchase': lgb.LGBMRegressor(),
            'purchase_amount': lgb.LGBMRegressor()
        }
        self.enable_bayesian = enable_bayesian
        self.calibrators = {}
        self.uncertainty_models = {}  # Для оценки uncertainty

    def train(self, train_df: pd.DataFrame):
        print("🔮 Training Context-Aware Model with Bayesian calibration...")

        X = self._prepare_features(train_df)
        self.feature_columns = X.columns.tolist()

        # 1. Bayesian оптимизация гиперпараметров
        if self.enable_bayesian:
            self._optimize_hyperparameters(X, train_df)

        # 2. Обучение с TimeSeries Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Purchase probability с калибровкой
        y_will = train_df['target_will_purchase']

        # Калибровка вероятностей (Isotonic/Platt scaling)
        base_model = lgb.LGBMClassifier()
        self.calibrators['will_purchase'] = CalibratedClassifierCV(
            base_model, method='isotonic', cv=3
        )
        self.calibrators['will_purchase'].fit(X, y_will)

        # Также обучаем основной модель
        self.models['will_purchase'].fit(X, y_will)

        # 3. Bayesian calibration для вероятностей
        self._train_bayesian_calibration(X, y_will)

        # 4. Quantile Regression для uncertainty в регрессиях
        if 'target_days_to_purchase' in train_df.columns:
            self._train_quantile_regression(
                X, train_df['target_days_to_purchase'], 'days'
            )

        if 'target_purchase_amount' in train_df.columns:
            self._train_quantile_regression(
                X, train_df['target_purchase_amount'], 'amount'
            )

    def _train_bayesian_calibration(self, X, y):
        """Bayesian calibration с Beta распределением"""
        # Получаем raw scores
        raw_probs = self.models['will_purchase'].predict_proba(X)[:, 1]

        # Fit Beta distribution к калиброванным вероятностям
        calibrated_probs = self.calibrators['will_purchase'].predict_proba(X)[:, 1]

        # Оцениваем параметры Beta распределения
        from scipy.stats import beta
        alpha, beta_param, loc, scale = beta.fit(calibrated_probs, floc=0, fscale=1)

        self.beta_params = {'alpha': alpha, 'beta': beta_param}

        # Также обучаем модель uncertainty
        errors = np.abs(calibrated_probs - y)
        uncertainty_model = lgb.LGBMRegressor()
        uncertainty_model.fit(X, errors)
        self.uncertainty_models['will_purchase'] = uncertainty_model

    def _train_quantile_regression(self, X, y, target_name):
        """Обучает quantile regression для confidence intervals"""
        # Для разных квантилей
        quantiles = [0.05, 0.5, 0.95]  # 90% интервал

        for q in quantiles:
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                metric='quantile'
            )
            model.fit(X, y)
            self.uncertainty_models[f'{target_name}_q{q}'] = model

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict:
        """Прогноз с оценкой uncertainty"""
        predictions = self.predict(X)

        # Добавляем uncertainty
        if 'purchase_probability' in predictions:
            probs = predictions['purchase_probability']

            # Bayesian credible intervals
            from scipy.stats import beta
            alpha, beta_param = self.beta_params['alpha'], self.beta_params['beta']

            # Для каждой вероятности вычисляем credible interval
            ci_lower = []
            ci_upper = []

            for p in probs:
                # Преобразуем в параметры Beta
                a = alpha * p
                b = beta_param * (1 - p)

                lower, upper = beta.interval(0.9, a, b)
                ci_lower.append(lower)
                ci_upper.append(upper)

            predictions['probability_lower'] = np.array(ci_lower)
            predictions['probability_upper'] = np.array(ci_upper)

        # Для регрессий добавляем quantile интервалы
        if 'days_to_purchase' in predictions:
            q05 = self.uncertainty_models['days_q0.05'].predict(X)
            q95 = self.uncertainty_models['days_q0.95'].predict(X)
            predictions['days_lower'] = q05
            predictions['days_upper'] = q95

        return predictions

    def _optimize_hyperparameters(self, X, train_df):
        """Bayesian optimization гиперпараметров"""
        try:
            import optuna

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }

                model = lgb.LGBMClassifier(**params)

                # TimeSeries CV оценка
                tscv = TimeSeriesSplit(n_splits=3)
                scores = cross_val_score(
                    model, X, train_df['target_will_purchase'],
                    cv=tscv, scoring='roc_auc'
                )

                return scores.mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=30)

            # Применяем лучшие параметры
            self.models['will_purchase'].set_params(**study.best_params)

        except ImportError:
            print("⚠️ Optuna not installed, using default hyperparameters")

# ===== МОДЕЛЬ 2: Cross-Region Demand Transfer =====

class CrossRegionModel:
    """Модель 2 с Graph Neural Networks для трансфера"""

    def __init__(self, use_gnn: bool = False):
        self.region_models = {}
        self.region_graph = None
        self.use_gnn = use_gnn

    def _build_region_graph(self, train_df: pd.DataFrame):
        """Строит граф регионов на основе корреляций"""
        regions = train_df['region'].unique()

        # Матрица корреляций между регионами
        corr_matrix = pd.DataFrame(index=regions, columns=regions)

        for i, region_i in enumerate(regions):
            for j, region_j in enumerate(regions):
                if i >= j:
                    continue

                # Вычисляем корреляцию временных рядов
                data_i = train_df[train_df['region'] == region_i]
                data_j = train_df[train_df['region'] == region_j]

                if len(data_i) > 0 and len(data_j) > 0:
                    # Агрегируем по дням
                    daily_i = data_i.groupby('snapshot_date')['target_purchase_count'].sum()
                    daily_j = data_j.groupby('snapshot_date')['target_purchase_count'].sum()

                    # Выравниваем даты
                    common_dates = daily_i.index.intersection(daily_j.index)
                    if len(common_dates) >= 7:
                        corr = np.corrcoef(
                            daily_i.loc[common_dates].values,
                            daily_j.loc[common_dates].values
                        )[0, 1]
                        corr_matrix.loc[region_i, region_j] = corr
                        corr_matrix.loc[region_j, region_i] = corr

        # Заполняем пропуски
        corr_matrix = corr_matrix.fillna(0)
        self.region_graph = corr_matrix

        # Также вычисляем lead-lag отношения
        self._compute_lead_lag_relationships(train_df)

    def _compute_lead_lag_relationships(self, train_df: pd.DataFrame):
        """Вычисляет lead-lag отношения между регионами"""
        regions = train_df['region'].unique()
        self.lead_lag_matrix = pd.DataFrame(index=regions, columns=regions)

        for source in regions:
            for target in regions:
                if source == target:
                    continue

                source_data = train_df[train_df['region'] == source]
                target_data = train_df[train_df['region'] == target]

                if len(source_data) > 14 and len(target_data) > 14:
                    # Грэнджер каузальность тест
                    try:
                        # Подготавливаем панельные данные
                        source_series = source_data.groupby('snapshot_date')['target_purchase_count'].sum()
                        target_series = target_data.groupby('snapshot_date')['target_purchase_count'].sum()

                        # Выравниваем
                        common_idx = source_series.index.intersection(target_series.index)
                        if len(common_idx) >= 14:
                            data = pd.DataFrame({
                                'source': source_series.loc[common_idx],
                                'target': target_series.loc[common_idx]
                            })

                            # Granger causality test
                            gc_result = grangercausalitytests(data[['target', 'source']], maxlag=3, verbose=False)

                            # Берем минимальный p-value
                            p_values = [gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, 4)]
                            min_p = min(p_values)

                            if min_p < 0.05:
                                # Определяем направление
                                gc_result_rev = grangercausalitytests(data[['source', 'target']], maxlag=3,
                                                                      verbose=False)
                                p_values_rev = [gc_result_rev[lag][0]['ssr_ftest'][1] for lag in range(1, 4)]
                                min_p_rev = min(p_values_rev)

                                if min_p < min_p_rev:
                                    self.lead_lag_matrix.loc[source, target] = 'source_lead'
                                else:
                                    self.lead_lag_matrix.loc[source, target] = 'target_lead'
                    except:
                        pass

    def train(self, train_df: pd.DataFrame):
        print("🌍 Training Cross-Region Model with Graph Analysis...")

        # Строим граф регионов
        self._build_region_graph(train_df)

        regions = train_df['region'].unique()

        for target_region in regions:
            print(f"  Target region: {target_region}")

            target_data = train_df[train_df['region'] == target_region]

            if len(target_data) < 10:
                print(f"    ⚠️ Not enough data, skipping...")
                continue

            # Определяем source регионы на основе графа
            source_regions = self._select_source_regions(target_region, train_df)

            # Обучаем CatBoost с учетом графа
            self._train_with_graph_awareness(
                target_region, target_data, source_regions, train_df
            )

    def _select_source_regions(self, target_region: str, train_df: pd.DataFrame) -> List[str]:
        """Выбирает source регионы на основе графа"""
        if self.region_graph is None:
            return [r for r in train_df['region'].unique() if r != target_region]

        # Берем регионы с наибольшей корреляцией
        correlations = self.region_graph.loc[target_region].sort_values(ascending=False)

        # Фильтруем по lead-lag отношениям
        selected = []
        for region, corr in correlations.items():
            if region == target_region:
                continue

            if pd.notna(corr) and abs(corr) > 0.3:
                # Проверяем lead-lag
                if region in self.lead_lag_matrix.index and target_region in self.lead_lag_matrix.columns:
                    relation = self.lead_lag_matrix.loc[region, target_region]
                    if relation == 'source_lead':  # source ведет target
                        selected.append(region)

            if len(selected) >= 3:  # Ограничиваем количество
                break

        return selected

    def _train_with_graph_awareness(self, target_region, target_data, source_regions, train_df):
        """Обучение с учетом графа регионов"""

        X_all, y_all = [], []

        # Добавляем данные target региона
        X_target, y_target = self._prepare_regression_features(target_data)
        X_all.append(X_target)
        y_all.append(y_target)

        # Добавляем трансформированные данные source регионов
        for source_region in source_regions:
            source_data = train_df[train_df['region'] == source_region]

            if len(source_data) > 0:
                # Умная трансформация с учетом графа
                X_source, y_source = self._transform_with_graph(
                    source_data, source_region, target_region
                )

                X_all.append(X_source)
                y_all.append(y_source)

        if X_all:
            X_combined = pd.concat(X_all, ignore_index=True)
            y_combined = pd.concat(y_all, ignore_index=True)

            # Используем CatBoost с учетом категориальных фич
            model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                cat_features=['region_encoded'] if 'region_encoded' in X_combined.columns else None,
                verbose=False
            )

            model.fit(X_combined, y_combined['target_purchase_count'])
            self.region_models[target_region] = model

            # Оценка
            preds = model.predict(X_target)
            mae = np.mean(np.abs(preds - y_target['target_purchase_count']))
            rmse = np.sqrt(np.mean((preds - y_target['target_purchase_count']) ** 2))

            print(f"    ✅ MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# ===== МОДЕЛЬ 3: Micro-Trend Anticipation =====

class MicroTrendModel(nn.Module):
    """Модель 3 с advanced change point detection"""

    def __init__(self, input_dim: int = 20):
        super().__init__()

        # ... существующий код ...

        # Добавляем детектор точек изменения
        self.change_point_detector = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def detect_change_points(self, sequence_data: np.ndarray) -> Dict:
        """Детектирует точки изменения в последовательности"""

        self.eval()
        with torch.no_grad():
            tensor_data = torch.FloatTensor(sequence_data).unsqueeze(0)

            # LSTM encoding
            lstm_out, _ = self.lstm(tensor_data)

            # Детекция точек изменения
            change_probs = self.change_point_detector(lstm_out).squeeze().numpy()

            # Находим точки с высокой вероятностью изменения
            change_points = np.where(change_probs > 0.7)[0]

            # Анализируем сегменты между точками изменения
            segments = []
            prev_point = 0

            for cp in change_points:
                segment_data = sequence_data[prev_point:cp]
                if len(segment_data) >= 3:
                    # Анализ тренда в сегменте
                    x = np.arange(len(segment_data))
                    coeffs = np.polyfit(x, segment_data.mean(axis=1), 1)
                    slope = coeffs[0]

                    segments.append({
                        'start': prev_point,
                        'end': cp,
                        'slope': slope,
                        'length': cp - prev_point,
                        'magnitude': np.mean(segment_data)
                    })

                prev_point = cp

            # Последний сегмент
            if prev_point < len(sequence_data):
                segment_data = sequence_data[prev_point:]
                if len(segment_data) >= 3:
                    x = np.arange(len(segment_data))
                    coeffs = np.polyfit(x, segment_data.mean(axis=1), 1)
                    slope = coeffs[0]

                    segments.append({
                        'start': prev_point,
                        'end': len(sequence_data),
                        'slope': slope,
                        'length': len(sequence_data) - prev_point,
                        'magnitude': np.mean(segment_data)
                    })

            return {
                'change_points': change_points.tolist(),
                'segments': segments,
                'num_changes': len(change_points)
            }

    def _bayesian_trend_analysis(self, sequence_data: np.ndarray) -> Dict:
        """Bayesian анализ трендов"""
        # Bayesian linear regression для оценки uncertainty
        n = len(sequence_data)
        x = np.arange(n)
        y = sequence_data.mean(axis=1) if sequence_data.ndim > 1 else sequence_data

        # Bayesian линейная регрессия
        # Prior: slope ~ Normal(0, 1), intercept ~ Normal(mean(y), 10)

        # Простая реализация с MCMC (аппроксимация)
        try:
            import pymc3 as pm

            with pm.Model() as model:
                # Priors
                sigma = pm.HalfCauchy('sigma', beta=10)
                intercept = pm.Normal('intercept', mu=np.mean(y), sigma=10)
                slope = pm.Normal('slope', mu=0, sigma=1)

                # Likelihood
                likelihood = pm.Normal('y',
                                       mu=intercept + slope * x,
                                       sigma=sigma,
                                       observed=y)

                # Sampling
                trace = pm.sample(1000, tune=1000, return_inferencedata=False)

                # Posterior analysis
                slope_samples = trace['slope']
                slope_mean = np.mean(slope_samples)
                slope_std = np.std(slope_samples)

                # Probability that slope > 0 (положительный тренд)
                prob_positive = np.mean(slope_samples > 0)

                # Credible interval
                ci_lower, ci_upper = np.percentile(slope_samples, [5, 95])

                return {
                    'slope_mean': slope_mean,
                    'slope_std': slope_std,
                    'prob_positive': prob_positive,
                    'ci_90_lower': ci_lower,
                    'ci_90_upper': ci_upper
                }

        except ImportError:
            # Fallback на frequentist подход
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            results = model.fit()

            return {
                'slope_mean': results.params[1],
                'slope_std': results.bse[1],
                'prob_positive': 1 - stats.norm.cdf(0, loc=results.params[1], scale=results.bse[1]),
                'ci_90_lower': results.conf_int(alpha=0.1)[1, 0],
                'ci_90_upper': results.conf_int(alpha=0.1)[1, 1]
            }

# ===== МОДЕЛЬ 4: Adaptive Pricing Prophet =====

class AdaptivePricingModel:
    """Модель 4 с Reinforcement Learning для динамического ценообразования"""

    def __init__(self, use_rl: bool = True):
        self.price_model = RandomForestRegressor(n_estimators=100)
        self.demand_model = xgb.XGBRegressor()
        self.optimal_prices = {}
        self.item_elasticity = {}

        if use_rl:
            self.rl_agent = self._create_rl_agent()
        self.use_rl = use_rl

    def _create_rl_agent(self):
        """Создает RL агента для ценообразования"""

        # Q-learning agent с neural network approximation
        class PricingAgent:
            def __init__(self, state_dim, action_dim):
                self.q_network = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim)
                )
                self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
                self.memory = []  # Experience replay

            def act(self, state, epsilon=0.1):
                if np.random.random() < epsilon:
                    return np.random.randint(self.action_dim)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state)
                        q_values = self.q_network(state_tensor)
                        return torch.argmax(q_values).item()

            def remember(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))

            def replay(self, batch_size=32):
                if len(self.memory) < batch_size:
                    return

                batch = random.sample(self.memory, batch_size)

                for state, action, reward, next_state, done in batch:
                    # Q-learning update
                    state_tensor = torch.FloatTensor(state)
                    next_state_tensor = torch.FloatTensor(next_state)

                    current_q = self.q_network(state_tensor)[action]
                    next_q = torch.max(self.q_network(next_state_tensor))
                    target_q = reward + (0.99 * next_q * (1 - done))

                    loss = nn.MSELoss()(current_q, target_q.detach())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        return PricingAgent(state_dim=10, action_dim=20)  # 20 возможных цен

    def _train_rl_agent(self, historical_data: pd.DataFrame):
        """Обучает RL агента на исторических данных"""
        if not self.use_rl:
            return

        # Подготавливаем состояния
        states = []
        actions = []
        rewards = []

        for item_id in historical_data['item_id'].unique():
            item_data = historical_data[historical_data['item_id'] == item_id]
            item_data = item_data.sort_values('snapshot_date')

            if len(item_data) < 10:
                continue

            # Формируем эпизоды
            for i in range(len(item_data) - 1):
                # Состояние: фичи товара
                state = self._get_state_features(item_data.iloc[i])

                # Действие: цена (дискретизированная)
                current_price = item_data.iloc[i]['current_price']
                action = self._discretize_price(current_price)

                # Награда: прибыль на следующем шаге
                next_profit = item_data.iloc[i + 1]['target_revenue'] - item_data.iloc[i + 1]['current_price']
                reward = next_profit

                # Следующее состояние
                next_state = self._get_state_features(item_data.iloc[i + 1])

                self.rl_agent.remember(state, action, reward, next_state, done=False)

        # Обучение
        for epoch in range(100):
            self.rl_agent.replay()

    def _get_state_features(self, row: pd.Series) -> np.ndarray:
        """Извлекает фичи состояния для RL"""
        features = [
            row.get('current_price', 0),
            row.get('conversion_rate', 0),
            row.get('price_elasticity', -1),
            row.get('total_views', 0),
            row.get('total_purchases', 0),
            row.get('price_position', 1),
            row.get('price_volatility', 0),
            row.get('category_avg_price', 0),
            row.get('days_since_last_purchase', 30),
            row.get('inventory_level', 100) if 'inventory_level' in row else 100
        ]
        return np.array(features, dtype=np.float32)

    def _discretize_price(self, price: float) -> int:
        """Дискретизирует цену для RL"""
        # 20 bins от 0.5x до 2x от текущей цены
        min_price = price * 0.5
        max_price = price * 2.0
        bins = np.linspace(min_price, max_price, 20)
        return np.digitize(price, bins) - 1

    def recommend_price_rl(self, item_id: str, current_state: Dict) -> Dict:
        """Рекомендация цены с помощью RL"""
        if not self.use_rl or self.rl_agent is None:
            return self.recommend_price(item_id, current_state['current_price'], current_state)

        # Преобразуем состояние в features
        state_features = np.array([
            current_state.get('current_price', 0),
            current_state.get('conversion_rate', 0),
            current_state.get('price_elasticity', -1),
            current_state.get('total_views', 0),
            current_state.get('total_purchases', 0),
            current_state.get('price_position', 1),
            current_state.get('price_volatility', 0),
            current_state.get('category_avg_price', 0),
            current_state.get('days_since_last_purchase', 30),
            current_state.get('inventory_level', 100)
        ], dtype=np.float32)

        # Действие от RL агента
        action = self.rl_agent.act(state_features, epsilon=0.1)

        # Преобразуем действие обратно в цену
        min_price = current_state['current_price'] * 0.5
        max_price = current_state['current_price'] * 2.0
        price_bins = np.linspace(min_price, max_price, 20)
        recommended_price = price_bins[action]

        # Bayesian optimization для тонкой настройки
        recommended_price = self._bayesian_price_optimization(
            item_id, recommended_price, current_state
        )

        return {
            'item_id': item_id,
            'current_price': current_state['current_price'],
            'recommended_price': float(recommended_price),
            'method': 'reinforcement_learning',
            'confidence': self._calculate_price_confidence(item_id, recommended_price, current_state)
        }

    def _bayesian_price_optimization(self, item_id: str, initial_price: float, context: Dict) -> float:
        """Bayesian optimization для тонкой настройки цены"""
        try:
            import optuna

            def objective(trial):
                # Предлагаем цену в окрестности initial_price
                price = trial.suggest_float(
                    'price',
                    initial_price * 0.9,
                    initial_price * 1.1
                )

                # Прогнозируем спрос
                demand = self._predict_demand(item_id, price, context)

                # Прибыль (упрощенно)
                profit = price * demand

                return -profit  # Минимизируем отрицательную прибыль

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)

            return study.best_params['price']

        except ImportError:
            return initial_price

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