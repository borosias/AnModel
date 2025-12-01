# innovative_snapshot_builder.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
from scipy import stats
from sklearn.impute import SimpleImputer  # вместо KNNImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from unicodedata import category

warnings.filterwarnings('ignore')


class InnovativeSnapshotBuilder:
    """Создает снапшоты для 4 инновационных моделей"""

    def __init__(self, parquet_dir: str = "../analytics/data/parquet"):
        self.parquet_dir = Path(parquet_dir)
        self.google_trends = self._load_google_trends()

    def quick_test_builders(self, test_date: str = "2025-01-01", n_samples: int = 5):
        """Быстро тестирует все билды снапшотов на маленькой выборке"""
        print("\n🧪 QUICK TESTING ALL BUILDERS...")

        # Загружаем данные
        events = self._load_and_prepare_events()
        if events.empty:
            print("❌ No events loaded!")
            return

        # Берем небольшой поднабор для теста
        events_sample = events.head(10000)
        print(f"✅ Loaded {len(events_sample):,} events for testing")

        # Тестовая дата
        test_dt = pd.to_datetime(test_date)

        # В quick_test_builders добавьте:
        if 'region' in events_sample.columns and not events_sample['region'].empty:
            most_common_region = events_sample['region'].mode()[0]
        else:
            most_common_region = 'UA-30'  # значение по умолчанию

        # Автоподбор окон
        window_back, window_forward = self._optimize_window_sizes(events_sample)
        print(f"📅 Test date: {test_dt.date()}")
        print(f"🔧 Windows: back={window_back}d, forward={window_forward}d")

        # Определяем регион для теста Model 2 (берем самый частый)
        most_common_region = events_sample['region'].mode()[0] if 'region' in events_sample.columns else 'UA-30'

        # Тестируем каждую модель отдельно
        models = [
            ('model1',
             lambda ev, dt: self._build_model1_features(ev, dt),
             lambda ev, idx, dt: self._build_model1_targets(ev, idx, dt)),
            ('model2',
             lambda ev, dt: self._build_model2_features(ev, dt, most_common_region),
             lambda ev, dt: self._build_model2_targets(ev, dt)),
            ('model3',
             lambda ev, dt: self._build_model3_features(ev, dt),
             lambda ev, idx: self._build_model3_targets(ev, idx)),
            ('model4',
             lambda ev, dt: self._build_model4_features(ev, dt),
             lambda ev, idx: self._build_model4_targets(ev, idx)),
        ]

        for model_name, feature_builder, target_builder in models:
            print(f"\n{'=' * 40}")
            print(f"Testing {model_name.upper()}...")

            try:
                # Тестовые данные
                feature_start = test_dt - timedelta(days=window_back)
                feature_end = test_dt
                target_start = test_dt
                target_end = test_dt + timedelta(days=window_forward)

                feature_events = events_sample[
                    (events_sample['ts'] >= feature_start) &
                    (events_sample['ts'] < feature_end)
                    ]

                target_events = events_sample[
                    (events_sample['ts'] >= target_start) &
                    (events_sample['ts'] < target_end)
                    ]

                print(f"  Feature events: {len(feature_events):,}")
                print(f"  Target events: {len(target_events):,}")

                # Строим фичи
                features = feature_builder(feature_events, test_dt)
                print(f"  Features shape: {features.shape if not features.empty else 'Empty'}")

                if not features.empty:
                    # Строим таргеты
                    if model_name == 'model1':
                        targets = target_builder(target_events, features.index, test_dt)
                    elif model_name == 'model2':
                        targets = target_builder(target_events, test_dt)
                    else:
                        targets = target_builder(target_events, features.index)

                    print(f"  Targets shape: {targets.shape if not targets.empty else 'Empty'}")

                    if not targets.empty:
                        # Объединяем
                        if model_name == 'model2':
                            snapshot = pd.concat([features, targets], axis=1)
                        else:
                            snapshot = features.join(targets, how='left').fillna(0)

                        print(f"  ✅ {model_name} SUCCESS!")
                        print(f"  Snapshot shape: {snapshot.shape}")
                        print(f"  Features: {list(features.columns)[:5]}..." if len(
                            features.columns) > 5 else f"  Features: {list(features.columns)}")
                        print(f"  Targets: {list(targets.columns)}")
                    else:
                        print(f"  ⚠️ {model_name}: No targets generated")
                else:
                    print(f"  ⚠️ {model_name}: No features generated")

            except Exception as e:
                print(f"  ❌ {model_name} FAILED: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\n{'=' * 40}")
        print("✅ QUICK TEST COMPLETED")
    # Добавьте этот метод в класс InnovativeSnapshotBuilder

    def test_model3_specific(self, test_date: str = "2025-01-01"):
        """Тестирует только Model 3"""
        print("\n🔍 TESTING MODEL 3 SPECIFICALLY...")

        events = self._load_and_prepare_events()
        if events.empty:
            print("❌ No events loaded!")
            return

        # Небольшая выборка
        events_sample = events.head(20000)
        test_dt = pd.to_datetime(test_date)
        window_back, window_forward = 30, 14

        print(f"Total events: {len(events_sample):,}")
        print(f"Search events: {(events_sample['event_type'] == 'search').sum():,}")
        print(f"Search queries: {events_sample[events_sample['event_type'] == 'search']['search_query'].nunique():,}")

        # Подготовка событий
        trend_events = events_sample[events_sample['event_type'].isin(['search', 'product_view'])]

        feature_start = test_dt - timedelta(days=7)  # Короткое окно для трендов
        feature_end = test_dt

        feature_events = trend_events[
            (trend_events['ts'] >= feature_start) &
            (trend_events['ts'] < feature_end)
            ]

        print(f"\nFeature events: {len(feature_events):,}")
        print(f"Search events in features: {(feature_events['event_type'] == 'search').sum():,}")

        if 'search_query' in feature_events.columns:
            print(
                f"Non-empty search queries: {feature_events[feature_events['event_type'] == 'search']['search_query'].notna().sum():,}")

        # Тест features
        print("\nBuilding features...")
        features = self._build_model3_features(feature_events, test_dt)

        if features is None:
            print("❌ Features returned None!")
        elif features.empty:
            print("⚠️ Features DataFrame is empty")

            # Диагностика
            search_events = feature_events[feature_events['event_type'] == 'search']
            if not search_events.empty:
                print("\nSearch query samples:")
                print(search_events['search_query'].head(10).tolist())
                print(f"\nUnique search queries: {search_events['search_query'].nunique()}")
        else:
            print(f"✅ Features shape: {features.shape}")
            print(f"Sample queries: {features.index[:5].tolist()}")

    def _load_google_trends(self) -> pd.DataFrame:
        """Загружаем Google Trends данные (если есть)"""
        trends_file = Path("trends_data/trends_master.parquet")
        if trends_file.exists():
            return pd.read_parquet(trends_file)
        return pd.DataFrame()

    def _optimize_window_sizes(self, events: pd.DataFrame, target_col: str = 'target_purchase_7d'):
        """Автоматически подбирает оптимальные размеры окон"""

        if events.empty or target_col not in events.columns:
            return 30, 7  # значения по умолчанию

        # Анализ автокорреляции для определения feature window
        try:
            # Берем временной ряд покупок
            purchase_dates = events[events['event_type'] == 'purchase']['ts']
            if len(purchase_dates) >= 30:
                daily_purchases = purchase_dates.groupby(purchase_dates.dt.date).size()

                # Вычисляем ACF
                from statsmodels.tsa.stattools import acf
                acf_values = acf(daily_purchases.values, nlags=30, fft=False)

                # Находим точку где ACF падает ниже significance threshold
                threshold = 1.96 / np.sqrt(len(daily_purchases))
                significant_lags = np.where(np.abs(acf_values) > threshold)[0]

                if len(significant_lags) > 1:
                    optimal_window = min(significant_lags[-1], 90)  # не более 90 дней
                else:
                    optimal_window = 30
            else:
                optimal_window = 30
        except:
            optimal_window = 30

        # Анализ для target window
        try:
            # Смотрим на распределение времени между покупками
            if len(purchase_dates) >= 10:
                sorted_dates = purchase_dates.sort_values()
                time_diffs = (sorted_dates.shift(-1) - sorted_dates).dt.days.dropna()

                if len(time_diffs) > 0:
                    median_interval = time_diffs.median()
                    # Берем 75-й перцентиль или 14 дней максимум
                    optimal_forward = min(int(time_diffs.quantile(0.75)), 14)
                else:
                    optimal_forward = 7
            else:
                optimal_forward = 7
        except:
            optimal_forward = 7

        return optimal_window, optimal_forward

    # Используй в build_all_snapshots:
    def build_all_snapshots(self, train_end: str, val_end: str, test_end: str,
                            window_back_days: int = None,  # сделай опциональным
                            window_forward_days: int = None):

        # Загружаем события
        events = self._load_and_prepare_events()

        # Автоподбор окон если не указаны
        if window_back_days is None or window_forward_days is None:
            optimal_back, optimal_forward = self._optimize_window_sizes(events)
            window_back_days = window_back_days or optimal_back
            window_forward_days = window_forward_days or optimal_forward

            print(f"🎯 Auto-optimized windows: back={window_back_days}d, forward={window_forward_days}d")

        # Генерируем даты снапшотов (ежедневные для микро-трендов)
        snapshot_dates = self._generate_snapshot_dates(events, window_back_days, window_forward_days)

        # Разделяем на train/val/test
        train_dates, val_dates, test_dates = self._split_dates(
            snapshot_dates, train_end, val_end, test_end
        )

        print(f"📊 Total snapshots: {len(snapshot_dates)}")
        print(f"  Train: {len(train_dates)}, Val: {len(val_dates)}, Test: {len(test_dates)}")

        # Строим снапшоты для каждой модели
        snapshots = {}

        # Модель 1: Context-Aware Purchase Prediction
        snapshots['model1'] = self._build_model1_snapshots(
            events, train_dates, val_dates, test_dates, window_back_days, window_forward_days
        )

        # Модель 2: Cross-Region Demand Transfer
        snapshots['model2'] = self._build_model2_snapshots(
            events, train_dates, val_dates, test_dates, window_back_days, window_forward_days
        )

        # Модель 3: Micro-Trend Anticipation
        snapshots['model3'] = self._build_model3_snapshots(
            events, train_dates, val_dates, test_dates, window_back_days, window_forward_days
        )

        # Модель 4: Adaptive Pricing
        snapshots['model4'] = self._build_model4_snapshots(
            events, train_dates, val_dates, test_dates, window_back_days, window_forward_days
        )

        return snapshots

    def _load_and_prepare_events(self) -> pd.DataFrame:
        """Загружаем и готовим события"""
        parquet_files = list(self.parquet_dir.glob("events_part_*.parquet"))

        if not parquet_files:
            raise ValueError("No parquet files found!")

        print(f"📂 Loading {len(parquet_files)} parquet files...")

        # Быстрая загрузка первых 5 файлов (для демо)
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            df['ts'] = pd.to_datetime(df['ts'])

            # Парсим properties
            df = self._parse_properties(df)

            dfs.append(df)

        events = pd.concat(dfs, ignore_index=True).sort_values('ts')

        print(f"✅ Loaded {len(events):,} events from {events['ts'].min()} to {events['ts'].max()}")

        return events

    def _parse_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Парсим JSON свойства"""
        import json

        def parse_json(x):
            if isinstance(x, str):
                try:
                    return json.loads(x.replace("'", '"'))
                except:
                    return {}
            return x if isinstance(x, dict) else {}

        df['properties_dict'] = df['properties'].apply(parse_json)

        # Извлекаем важные поля
        df['search_query'] = df['properties_dict'].apply(lambda x: x.get('search_query', ''))
        df['category'] = df['properties_dict'].apply(lambda x: x.get('category', ''))
        df['device'] = df['properties_dict'].apply(lambda x: x.get('device', 'desktop'))
        df['price_from_props'] = df['properties_dict'].apply(lambda x: float(x.get('price', 0)))

        # Используем правильную цену
        df['price'] = df.apply(
            lambda row: row['price'] if pd.notna(row['price']) else row['price_from_props'],
            axis=1
        )

        return df

    def _generate_snapshot_dates(self, events: pd.DataFrame,
                                 window_back: int, window_forward: int) -> List[datetime]:
        """Генерируем даты снапшотов (ежедневно для микро-трендов)"""
        min_date = events['ts'].min().floor('D') + timedelta(days=window_back)
        max_date = events['ts'].max().ceil('D') - timedelta(days=window_forward)

        return pd.date_range(start=min_date, end=max_date, freq='1D')

    def _split_dates(self, dates: List[datetime],
                     train_end: str, val_end: str, test_end: str) -> Tuple:
        """Разделяем даты на train/val/test"""
        train_end_dt = pd.to_datetime(train_end)
        val_end_dt = pd.to_datetime(val_end)
        test_end_dt = pd.to_datetime(test_end)

        train = [d for d in dates if d <= train_end_dt]
        val = [d for d in dates if train_end_dt < d <= val_end_dt]
        test = [d for d in dates if val_end_dt < d <= test_end_dt]

        return train, val, test

    # ===== МОДЕЛЬ 1: Context-Aware Purchase Prediction =====

    def _build_model1_snapshots(self, events, train_dates, val_dates, test_dates,
                                window_back, window_forward):
        """Снапшоты для предсказания КОНТЕКСТНОЙ покупки"""

        print("\n🔮 Building Model 1: Context-Aware Purchase Prediction...")

        datasets = {}

        for dataset_name, dates in [('train', train_dates), ('val', val_dates), ('test', test_dates)]:
            snapshots = []

            for snapshot_date in dates:  # Ограничим для демо
                # Окно фич
                feature_start = snapshot_date - timedelta(days=window_back)
                feature_end = snapshot_date

                # Окно таргетов
                target_start = snapshot_date
                target_end = snapshot_date + timedelta(days=window_forward)

                # Фильтруем события
                feature_events = events[
                    (events['ts'] >= feature_start) &
                    (events['ts'] < feature_end)
                    ]

                target_events = events[
                    (events['ts'] >= target_start) &
                    (events['ts'] < target_end)
                    ]

                # Строим фичи
                features = self._build_model1_features(feature_events, snapshot_date)

                # Строим МУЛЬТИЗАДАЧНЫЕ таргеты
                targets = self._build_model1_targets(target_events, features.index, snapshot_date)

                # Объединяем
                snapshot_df = features.join(targets, how='left').fillna(0)
                snapshot_df['snapshot_date'] = snapshot_date

                snapshots.append(snapshot_df.reset_index())

            if snapshots:
                datasets[dataset_name] = pd.concat(snapshots, ignore_index=True)
                print(f"  {dataset_name}: {len(datasets[dataset_name]):,} samples")
            else:
                datasets[dataset_name] = pd.DataFrame()

        return datasets

    def _build_model1_features(self, events: pd.DataFrame, snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для Model 1 с учетом контекста"""

        if events.empty:
            return pd.DataFrame()

        # Группируем по пользователям
        user_features = events.groupby('user_id').agg({
            'event_id': 'count',
            'ts': ['max', 'min', 'nunique'],
            'event_type': lambda x: (x == 'purchase').sum(),
            'price': 'sum'
        })

        # Flatten columns
        user_features.columns = ['total_events', 'last_event', 'first_event',
                                 'active_days', 'total_purchases', 'total_spent']

        # Временные фичи с контекстом
        user_features['days_since_first'] = (snapshot_date - user_features['first_event']).dt.days
        user_features['days_since_last'] = (snapshot_date - user_features['last_event']).dt.days
        user_features['events_per_day'] = user_features['total_events'] / (user_features['days_since_first'] + 1)

        # Сезонность и время
        user_features['snapshot_month'] = snapshot_date.month
        user_features['snapshot_day_of_week'] = snapshot_date.weekday()
        user_features['snapshot_hour'] = snapshot_date.hour

        # Поведенческие паттерны
        # 1. Сессионные паттерны
        sessions = events.groupby(['user_id', 'session_id']).agg({
            'ts': ['min', 'max', 'count'],
            'event_type': lambda x: list(x)
        })

        sessions.columns = ['session_start', 'session_end', 'session_events', 'session_event_types']

        user_sessions = sessions.groupby('user_id').agg({
            'session_events': ['mean', 'std', 'count'],
            'session_start': 'min',
            'session_end': 'max'
        })

        user_sessions.columns = ['avg_session_events', 'std_session_events', 'session_count',
                                 'first_session', 'last_session']

        user_sessions['avg_session_duration'] = (
                                                        user_sessions['last_session'] - user_sessions['first_session']
                                                ).dt.total_seconds() / 3600 / (user_sessions['session_count'] + 1)

        # 2. Категориальные предпочтения
        if 'category' in events.columns:
            category_counts = events.groupby(['user_id', 'category']).size().unstack(fill_value=0)
            category_counts = category_counts.add_prefix('category_')
            user_features = user_features.join(category_counts, how='left')

        # 3. Временные паттерны (утро/день/вечер)
        events['hour'] = events['ts'].dt.hour
        time_patterns = pd.crosstab(events['user_id'], pd.cut(events['hour'],
                                                              bins=[0, 8, 16, 24],
                                                              labels=['night', 'day', 'evening']))
        time_patterns = time_patterns.add_prefix('activity_')

        # 4. Ценовая чувствительность с Bayesian подходом
        purchases = events[events['event_type'] == 'purchase']
        if not purchases.empty:
            for user_id, user_purchases in purchases.groupby('user_id'):
                if len(user_purchases) >= 3:
                    # Bayesian оценка эластичности
                    try:
                        prices = user_purchases['price'].values
                        # Убираем нули и отрицательные цены
                        mask = prices > 0
                        if np.sum(mask) >= 2:
                            log_prices = np.log(prices[mask])
                            # Каждая покупка = 1, логарифмируем
                            log_quantities = np.log(np.ones(np.sum(mask)))

                            # Простая OLS оценка эластичности
                            X = sm.add_constant(log_prices)
                            model = sm.OLS(log_quantities, X)
                            results = model.fit()

                            user_features.loc[user_id, 'price_elasticity'] = results.params[1]
                            user_features.loc[user_id, 'elasticity_se'] = results.bse[1]
                            user_features.loc[user_id, 'elasticity_pval'] = results.pvalues[1]

                            # Bayesian credible interval (аппроксимация)
                            n = len(log_prices)
                            t_critical = stats.t.ppf(0.975, n - 2)
                            ci_lower = results.params[1] - t_critical * results.bse[1]
                            ci_upper = results.params[1] + t_critical * results.bse[1]
                            user_features.loc[user_id, 'elasticity_ci_width'] = ci_upper - ci_lower
                    except:
                        pass

        # 5. SARIMA для временных рядов активности (вместо Prophet)
        if 'ts' in events.columns and len(events) > 30:
            user_activity = events.groupby(['user_id', pd.Grouper(key='ts', freq='D')]).size()

            for user_id in user_features.index:  # Ограничим для скорости
                if user_id in user_activity.index:
                    try:
                        user_series = user_activity.loc[user_id]
                        if isinstance(user_series, pd.Series) and len(user_series) > 14:
                            # Проверка стационарности
                            adf_result = adfuller(user_series.values, autolag='AIC')
                            user_features.loc[user_id, 'adf_statistic'] = adf_result[0]
                            user_features.loc[user_id, 'adf_pvalue'] = adf_result[1]

                            # Простая сезонная декомпозиция
                            if len(user_series) >= 30:
                                decomposition = seasonal_decompose(
                                    user_series.values,
                                    model='additive',
                                    period=7,
                                    extrapolate_trend='freq'
                                )

                                user_features.loc[user_id, 'trend_strength'] = np.maximum(
                                    0,
                                    1 - np.var(decomposition.resid) / np.var(decomposition.trend + decomposition.resid)
                                )
                                user_features.loc[user_id, 'seasonal_strength'] = np.maximum(
                                    0,
                                    1 - np.var(decomposition.resid) / np.var(
                                        decomposition.seasonal + decomposition.resid)
                                )
                    except:
                        pass

        # 6. Ручной feature engineering для временных рядов (вместо tsfresh)
        if len(events) > 100:
            # Группируем по пользователям и вычисляем статистики временных рядов
            time_stats = events.groupby('user_id').agg({
                'ts': lambda x: self._compute_time_series_stats(x, snapshot_date)
            })
            time_stats_df = pd.DataFrame(
                time_stats['ts'].tolist(),
                index=time_stats.index
            )
            user_features = user_features.join(time_stats_df, how='left')

        # 7. Улучшенная обработка пропусков и scaling
        if not user_features.empty:
            # Удаляем константные колонки
            numeric_cols = user_features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Удаляем колонки с нулевой дисперсией
                variances = user_features[numeric_cols].var()
                cols_to_keep = variances[variances > 1e-10].index
                user_features = user_features[cols_to_keep.tolist() +
                                              [c for c in user_features.columns if c not in numeric_cols]]

                # Импутация медианой
                imputer = SimpleImputer(strategy='median')
                numeric_data = imputer.fit_transform(user_features.select_dtypes(include=[np.number]))
                user_features[user_features.select_dtypes(include=[np.number]).columns] = numeric_data

                # PowerTransformer вместо Quantile (лучше для 3.10)
                try:
                    transformer = PowerTransformer(method='yeo-johnson')
                    scaled_data = transformer.fit_transform(user_features.select_dtypes(include=[np.number]))
                    user_features[user_features.select_dtypes(include=[np.number]).columns] = scaled_data
                except:
                    # Fallback на StandardScaler
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(user_features.select_dtypes(include=[np.number]))
                    user_features[user_features.select_dtypes(include=[np.number]).columns] = scaled_data

        return user_features

    def _compute_time_series_stats(self, timestamps, snapshot_date):
        """Вычисляет статистики временного ряда"""
        if len(timestamps) < 2:
            return {}

        # ПРАВИЛЬНОЕ преобразование в datetime
        if isinstance(timestamps, pd.Series):
            times = pd.to_datetime(timestamps).sort_values()
        else:
            times = pd.to_datetime(pd.Series(timestamps)).sort_values()

        # Проверяем что times не пусто
        if len(times) == 0:
            return {}

        deltas = np.diff(times).astype('timedelta64[s]').astype(float)

        # БЕЗОПАСНОЕ вычисление энтропии
        try:
            # Извлекаем часы безопасно
            if hasattr(times, 'dt'):
                hours = times.dt.hour
            else:
                # Если times уже обработаны
                hours = pd.Series(times).dt.hour if hasattr(pd.Series(times), 'dt') else pd.Series([0] * len(times))

            entropy = self._calculate_entropy(hours) if len(hours) > 0 else 0
        except:
            entropy = 0

        stats_dict = {
            'interarrival_mean': np.mean(deltas) if len(deltas) > 0 else 0,
            'interarrival_std': np.std(deltas) if len(deltas) > 0 else 0,
            'interarrival_cv': np.std(deltas) / np.mean(deltas) if len(deltas) > 0 and np.mean(deltas) > 0 else 0,
            'burstiness': (np.std(deltas) - np.mean(deltas)) / (np.std(deltas) + np.mean(deltas)) if len(
                deltas) > 0 else 0,
            'activity_entropy': entropy,
        }

        # Автокорреляция лага 1 (с защитой)
        if len(times) >= 5:
            try:
                daily_counts = pd.Series(times).dt.date.value_counts().sort_index()
                if len(daily_counts) >= 3:
                    stats_dict['autocorr_lag1'] = daily_counts.autocorr(lag=1)
            except:
                stats_dict['autocorr_lag1'] = 0

        return stats_dict

    def _calculate_entropy(self, values):
        """Рассчитывает энтропию Шеннона"""
        from collections import Counter
        counts = Counter(values)
        probs = np.array(list(counts.values())) / len(values)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _build_model1_targets(self, target_events: pd.DataFrame,
                              user_index: pd.Index, snapshot_date: datetime) -> pd.DataFrame:
        """Мультизадачные таргеты для Model 1"""

        targets = pd.DataFrame(index=user_index)

        # 1. Бинарный таргет: купит ли вообще
        targets['target_will_purchase'] = 0

        # 2. Категория покупки (если купит)
        targets['target_category'] = ''

        # 3. Время до покупки (в днях)
        targets['target_days_to_purchase'] = 999

        # 4. Сумма покупки
        targets['target_purchase_amount'] = 0

        if target_events.empty:
            return targets

        purchases = target_events[target_events['event_type'] == 'purchase']

        if not purchases.empty:
            # Группируем по пользователям
            for user_id in purchases['user_id'].unique():
                user_purchases = purchases[purchases['user_id'] == user_id]

                targets.loc[user_id, 'target_will_purchase'] = 1

                # Самая частая категория
                if 'category' in user_purchases.columns:
                    top_category = user_purchases['category'].mode()
                    if not top_category.empty:
                        targets.loc[user_id, 'target_category'] = top_category.iloc[0]

                # Время до первой покупки
                first_purchase_time = user_purchases['ts'].min()
                days_to_purchase = (first_purchase_time - snapshot_date).days
                targets.loc[user_id, 'target_days_to_purchase'] = max(0, days_to_purchase)

                # Сумма покупок
                targets.loc[user_id, 'target_purchase_amount'] = user_purchases['price'].sum()

        return targets

    def _add_google_trends_features(self, snapshot_date: datetime) -> pd.DataFrame:
        """Добавляем фичи из Google Trends"""
        if self.google_trends.empty:
            return pd.DataFrame()

        # Берем тренды за последние 7 дней
        start_date = snapshot_date - timedelta(days=7)

        recent_trends = self.google_trends[
            (self.google_trends['date'] >= start_date.date()) &
            (self.google_trends['date'] <= snapshot_date.date())
            ]

        if recent_trends.empty:
            return pd.DataFrame()

        # Агрегируем по запросам
        trends_features = {}

        for query in recent_trends['query'].unique():
            query_trends = recent_trends[recent_trends['query'] == query]

            # Средняя популярность
            trends_features[f'trend_{query}_avg'] = query_trends['popularity'].mean()

            # Тренд (рост/падение)
            if len(query_trends) > 1:
                first = query_trends.iloc[0]['popularity']
                last = query_trends.iloc[-1]['popularity']
                trends_features[f'trend_{query}_growth'] = (last - first) / (first + 1)
            else:
                trends_features[f'trend_{query}_growth'] = 0

        return pd.DataFrame([trends_features])

    # ===== МОДЕЛЬ 2: Cross-Region Demand Transfer =====

    def _build_model2_snapshots(self, events, train_dates, val_dates, test_dates,
                                window_back, window_forward):
        """Снапшоты для трансфера спроса между регионами"""

        print("\n🌍 Building Model 2: Cross-Region Demand Transfer...")

        # Группируем события по регионам
        regions = events['region'].unique()
        print(f"  Regions: {regions}")

        datasets = {'train': [], 'val': [], 'test': []}

        for region in regions:
            region_events = events[events['region'] == region]

            for dataset_name, dates in [('train', train_dates), ('val', val_dates), ('test', test_dates)]:
                region_snapshots = []

                for snapshot_date in dates:  # Ограничим
                    feature_start = snapshot_date - timedelta(days=window_back)
                    feature_end = snapshot_date
                    target_start = snapshot_date
                    target_end = snapshot_date + timedelta(days=window_forward)

                    feature_events = region_events[
                        (region_events['ts'] >= feature_start) &
                        (region_events['ts'] < feature_end)
                        ]

                    target_events = region_events[
                        (region_events['ts'] >= target_start) &
                        (region_events['ts'] < target_end)
                        ]

                    # Фичи на уровне региона
                    features = self._build_model2_features(feature_events, snapshot_date, region)
                    targets = self._build_model2_targets(target_events, snapshot_date)

                    snapshot_df = pd.concat([features, targets], axis=1)
                    snapshot_df['snapshot_date'] = snapshot_date
                    snapshot_df['region'] = region

                    region_snapshots.append(snapshot_df.reset_index(drop=True))

                if region_snapshots:
                    region_df = pd.concat(region_snapshots, ignore_index=True)
                    datasets[dataset_name].append(region_df)

        # Объединяем все регионы
        result = {}
        for dataset_name in ['train', 'val', 'test']:
            if datasets[dataset_name]:
                result[dataset_name] = pd.concat(datasets[dataset_name], ignore_index=True)
                print(f"  {dataset_name}: {len(result[dataset_name]):,} region-snapshots")
            else:
                result[dataset_name] = pd.DataFrame()

        return result

    def _build_model2_features(self, events: pd.DataFrame,
                               snapshot_date: datetime, region: str) -> pd.DataFrame:
        """Фичи для Model 2 (региональные)"""

        features = {}

        # Базовые метрики региона
        features['region_total_events'] = len(events)
        features['region_unique_users'] = events['user_id'].nunique()
        features['region_purchase_count'] = (events['event_type'] == 'purchase').sum()
        features['region_total_spent'] = events[events['event_type'] == 'purchase']['price'].sum()

        # Активность по времени
        if len(events) > 0:
            events['hour'] = events['ts'].dt.hour
            morning_events = ((events['hour'] >= 6) & (events['hour'] < 12)).sum()
            evening_events = ((events['hour'] >= 18) & (events['hour'] < 24)).sum()
            features['region_morning_ratio'] = morning_events / len(events)
            features['region_evening_ratio'] = evening_events / len(events)

        # Категориальное распределение
        if 'category' in events.columns:
            top_categories = events['category'].value_counts().head(3)
            for i, (cat, count) in enumerate(top_categories.items()):
                features[f'region_top_category_{i + 1}'] = cat
                features[f'region_top_category_{i + 1}_count'] = count

        # Ценовое распределение
        purchase_events = events[events['event_type'] == 'purchase']
        if len(purchase_events) > 0:
            features['region_avg_price'] = purchase_events['price'].mean()
            features['region_price_std'] = purchase_events['price'].std()

        # Временные фичи
        iso_tuple = snapshot_date.isocalendar()  # (year, week, weekday)
        features['snapshot_year'] = iso_tuple[0]
        features['snapshot_week'] = iso_tuple[1]
        features['snapshot_weekday'] = iso_tuple[2]

        # Соседние регионы (для трансфера)
        if region == 'UA-30':
            features['neighbor_UA40_proximity'] = 1.0
            features['neighbor_UA50_proximity'] = 0.7
        elif region == 'UA-40':
            features['neighbor_UA30_proximity'] = 1.0
            features['neighbor_UA50_proximity'] = 0.8
        else:  # UA-50
            features['neighbor_UA30_proximity'] = 0.7
            features['neighbor_UA40_proximity'] = 0.8

        return pd.DataFrame([features])

    def _build_model2_targets(self, target_events: pd.DataFrame,
                              snapshot_date: datetime) -> pd.DataFrame:
        """Таргеты для Model 2 (региональный спрос)"""

        targets = {}

        # Общий спрос
        purchases = target_events[target_events['event_type'] == 'purchase']
        targets['target_purchase_count'] = len(purchases)
        targets['target_total_spent'] = purchases['price'].sum() if not purchases.empty else 0

        # Спрос по категориям (top 3)
        if 'category' in purchases.columns and not purchases.empty:
            top_categories = purchases['category'].value_counts().head(3)
            for i, (cat, count) in enumerate(top_categories.items()):
                targets[f'target_category_{cat}_demand'] = count

        # Временное распределение спроса
        if not purchases.empty:
            purchases['day_of_week'] = purchases['ts'].dt.dayofweek
            weekday_demand = purchases[purchases['day_of_week'] < 5]['price'].sum()
            weekend_demand = purchases[purchases['day_of_week'] >= 5]['price'].sum()
            targets['target_weekday_demand'] = weekday_demand
            targets['target_weekend_demand'] = weekend_demand

        return pd.DataFrame([targets])

    # ===== МОДЕЛЬ 3: Micro-Trend Anticipation =====

    def _build_model3_snapshots(self, events, train_dates, val_dates, test_dates,
                                window_back, window_forward):
        """Снапшоты для предсказания микро-трендов (улучшенная версия)"""

        print("\n📈 Building Model 3: Micro-Trend Anticipation...")

        # Фокусируемся на поисковых запросах и просмотрах
        trend_events = events[events['event_type'].isin(['search', 'product_view'])]

        if trend_events.empty:
            print("  ⚠️ No search or view events for trend analysis")
            return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}

        print(f"  Trend events: {len(trend_events):,}")
        print(f"  Search events: {(trend_events['event_type'] == 'search').sum():,}")
        print(f"  View events: {(trend_events['event_type'] == 'product_view').sum():,}")

        datasets = {}

        # Функция для обработки одного набора дат
        def process_dates(dataset_name, dates):
            snapshots = []
            total_processed = 0
            total_skipped = 0

            # Ограничиваем количество дат для производительности
            if len(dates) > 50:
                print(f"  {dataset_name}: Too many dates ({len(dates)}), sampling 50...")
                dates = dates[:50]

            for i, snapshot_date in enumerate(dates):
                if i % 10 == 0:
                    print(f"    Processing {dataset_name} date {i + 1}/{len(dates)}: {snapshot_date.date()}")

                # Короткое окно для микро-трендов
                feature_start = snapshot_date - timedelta(days=7)
                feature_end = snapshot_date
                target_start = snapshot_date
                target_end = snapshot_date + timedelta(days=7)

                feature_events = trend_events[
                    (trend_events['ts'] >= feature_start) &
                    (trend_events['ts'] < feature_end)
                    ]

                target_events = trend_events[
                    (trend_events['ts'] >= target_start) &
                    (trend_events['ts'] < target_end)
                    ]

                # Фичи для поисковых запросов/товаров
                features = self._build_model3_features(feature_events, snapshot_date)

                if features.empty:
                    total_skipped += 1
                    continue

                # Проверяем что features.index не пустой
                if len(features.index) == 0:
                    total_skipped += 1
                    continue

                targets = self._build_model3_targets(target_events, features.index)

                # Если targets пустые, создаем пустые таргеты с правильным индексом
                if targets.empty:
                    targets = pd.DataFrame(index=features.index)
                    for col in ['target_future_searches', 'target_trend_continues', 'target_peak_in_days']:
                        targets[col] = 0

                # Проверяем совпадение индексов
                if not features.index.equals(targets.index):
                    print(f"    Warning: Index mismatch for {snapshot_date.date()}")
                    # Выравниваем индексы
                    common_idx = features.index.intersection(targets.index)
                    if len(common_idx) == 0:
                        total_skipped += 1
                        continue
                    features = features.loc[common_idx]
                    targets = targets.loc[common_idx]

                try:
                    snapshot_df = features.join(targets, how='left').fillna(0)
                    snapshot_df['snapshot_date'] = snapshot_date
                    snapshot_df['dataset'] = dataset_name
                    snapshots.append(snapshot_df.reset_index())
                    total_processed += 1
                except Exception as e:
                    print(f"    Error joining features and targets: {str(e)}")
                    total_skipped += 1
                    continue

            print(f"    {dataset_name}: Processed {total_processed}, skipped {total_skipped}")

            if snapshots:
                return pd.concat(snapshots, ignore_index=True)
            else:
                return pd.DataFrame()

        # Обрабатываем все датасеты
        for dataset_name, dates in [('train', train_dates), ('val', val_dates), ('test', test_dates)]:
            print(f"\n  Processing {dataset_name} ({len(dates)} dates)...")
            result_df = process_dates(dataset_name, dates)
            datasets[dataset_name] = result_df

            if not result_df.empty:
                print(f"  ✅ {dataset_name}: {len(result_df):,} trend-snapshots, {len(result_df.columns)} features")
            else:
                print(f"  ⚠️ {dataset_name}: No snapshots generated")

        return datasets

    def _build_model3_features(self, events: pd.DataFrame,
                               snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для обнаружения микро-трендов (полная версия)"""

        print(f"    Building Model 3 features for {snapshot_date.date()}...")

        # Разделяем события по типам
        search_events = events[events['event_type'] == 'search']
        view_events = events[events['event_type'] == 'product_view']

        print(f"      Total events: {len(events):,}")
        print(f"      Search events: {len(search_events):,}")
        print(f"      View events: {len(view_events):,}")

        # Список для сбора всех фич
        all_features = []

        # 1. Тренды поисковых запросов (если есть)
        if not search_events.empty and 'search_query' in search_events.columns:
            query_features = self._build_search_query_features(search_events, snapshot_date)
            if not query_features.empty:
                all_features.append(query_features)
                print(f"      Query features: {len(query_features)} queries")

        # 2. Тренды по категориям (если есть просмотры и категории)
        if not view_events.empty and 'category' in view_events.columns:
            category_features = self._build_category_features(view_events, snapshot_date)
            if not category_features.empty:
                all_features.append(category_features)
                print(f"      Category features: {len(category_features)} categories")

        # 3. Тренды по товарам (если нет категорий)
        elif not view_events.empty and 'item_id' in view_events.columns:
            item_features = self._build_item_features(view_events, snapshot_date)
            if not item_features.empty:
                all_features.append(item_features)
                print(f"      Item features: {len(item_features)} items")

        # 4. Временные тренды по активности
        temporal_features = self._build_temporal_features(events, snapshot_date)
        if not temporal_features.empty:
            all_features.append(temporal_features)
            print(f"      Temporal features: {len(temporal_features)} time periods")

        # Объединяем все фичи
        if all_features:
            result = pd.concat(all_features, axis=0)
            print(f"      Total features generated: {len(result)} trend entities")
            return result
        else:
            print(f"      No features generated")
            return pd.DataFrame()

    def _build_search_query_features(self, search_events: pd.DataFrame,
                                     snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для поисковых запросов"""

        # Фильтруем валидные запросы
        valid_searches = search_events[
            search_events['search_query'].notna() &
            (search_events['search_query'].astype(str).str.strip() != '')
            ]

        if valid_searches.empty:
            return pd.DataFrame()

        # Берем топ-100 запросов по частоте
        query_counts = valid_searches['search_query'].value_counts()
        top_queries = query_counts.head(100).index

        query_features_list = []

        for query in top_queries:
            try:
                query_data = valid_searches[valid_searches['search_query'] == query]

                # Базовые фичи
                features = {
                    'trend_id': f"query_{query}",
                    'trend_type': 'search_query',
                    'entity': query,
                    'total_count': len(query_data),
                    'unique_users': query_data['user_id'].nunique(),
                    'first_seen': query_data['ts'].min(),
                    'last_seen': query_data['ts'].max(),
                }

                # Временные фичи
                query_data['date'] = query_data['ts'].dt.date
                date_range = pd.date_range(start=features['first_seen'].date(),
                                           end=snapshot_date.date(), freq='D')
                daily_counts = query_data.groupby('date').size().reindex(date_range.date, fill_value=0)

                # Активных дней
                active_days = max((features['last_seen'] - features['first_seen']).days + 1, 1)
                features['frequency'] = features['total_count'] / active_days
                features['active_days'] = (daily_counts > 0).sum()

                # Статистики по дням
                if len(daily_counts) > 1:
                    features['mean_daily'] = daily_counts.mean()
                    features['std_daily'] = daily_counts.std()
                    features['cv_daily'] = features['std_daily'] / features['mean_daily'] if features[
                                                                                                 'mean_daily'] > 0 else 0

                    # Тренд и ускорение
                    features['trend_slope'] = self._calculate_linear_trend(daily_counts.values)
                    features['trend_acceleration'] = self._calculate_acceleration(daily_counts.values)

                    # Рост (CAGR)
                    if daily_counts.iloc[0] > 0 and daily_counts.iloc[-1] > 0:
                        periods = len(daily_counts) - 1
                        features['cagr'] = (daily_counts.iloc[-1] / daily_counts.iloc[0]) ** (1 / periods) - 1
                    else:
                        features['cagr'] = 0

                    # Последние 3 дня vs первые 3 дня
                    if len(daily_counts) >= 6:
                        first_avg = daily_counts.iloc[:3].mean()
                        last_avg = daily_counts.iloc[-3:].mean()
                        features['recent_growth'] = (last_avg - first_avg) / (first_avg + 1)
                    else:
                        features['recent_growth'] = 0
                else:
                    features.update({
                        'mean_daily': features['total_count'],
                        'std_daily': 0,
                        'cv_daily': 0,
                        'trend_slope': 0,
                        'trend_acceleration': 0,
                        'cagr': 0,
                        'recent_growth': 0
                    })

                # Географические фичи
                if 'region' in query_data.columns:
                    regions = query_data['region'].unique()
                    features['region_count'] = len(regions)
                    features['is_multi_region'] = len(regions) > 1

                    # Доминирующий регион
                    if len(regions) > 0:
                        top_region = query_data['region'].mode()
                        if not top_region.empty:
                            features['top_region'] = top_region.iloc[0]
                else:
                    features['region_count'] = 1
                    features['is_multi_region'] = False
                    features['top_region'] = 'unknown'

                # Демографические фичи (если есть)
                if 'device' in query_data.columns:
                    devices = query_data['device'].unique()
                    features['device_count'] = len(devices)

                # Время суток
                if 'ts' in query_data.columns:
                    query_data['hour'] = query_data['ts'].dt.hour
                    morning = ((query_data['hour'] >= 6) & (query_data['hour'] < 12)).sum()
                    evening = ((query_data['hour'] >= 18) & (query_data['hour'] < 24)).sum()
                    features['morning_ratio'] = morning / len(query_data) if len(query_data) > 0 else 0
                    features['evening_ratio'] = evening / len(query_data) if len(query_data) > 0 else 0

                query_features_list.append(features)

            except Exception as e:
                # Пропускаем проблемные запросы
                continue

        if query_features_list:
            df = pd.DataFrame(query_features_list)
            df.set_index('trend_id', inplace=True)
            return df
        else:
            return pd.DataFrame()

    def _build_category_features(self, view_events: pd.DataFrame,
                                 snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для трендов по категориям"""

        # Фильтруем валидные категории
        valid_views = view_events[view_events['category'].notna()]

        if valid_views.empty:
            return pd.DataFrame()

        # Берем топ-50 категорий
        category_counts = valid_views['category'].value_counts()
        top_categories = category_counts.head(50).index

        category_features_list = []

        for category in top_categories:
            try:
                category_data = valid_views[valid_views['category'] == category]

                # Базовые фичи
                features = {
                    'trend_id': f"category_{category}",
                    'trend_type': 'category',
                    'entity': category,
                    'total_views': len(category_data),
                    'unique_viewers': category_data['user_id'].nunique(),
                    'first_view': category_data['ts'].min(),
                    'last_view': category_data['ts'].max(),
                }

                # Временные фичи
                category_data['date'] = category_data['ts'].dt.date
                date_range = pd.date_range(start=features['first_view'].date(),
                                           end=snapshot_date.date(), freq='D')
                daily_counts = category_data.groupby('date').size().reindex(date_range.date, fill_value=0)

                # Активных дней
                active_days = max((features['last_view'] - features['first_view']).days + 1, 1)
                features['frequency'] = features['total_views'] / active_days
                features['active_days'] = (daily_counts > 0).sum()

                # Статистики по дням
                if len(daily_counts) > 1:
                    features['mean_daily'] = daily_counts.mean()
                    features['std_daily'] = daily_counts.std()
                    features['cv_daily'] = features['std_daily'] / features['mean_daily'] if features[
                                                                                                 'mean_daily'] > 0 else 0

                    # Тренд
                    features['trend_slope'] = self._calculate_linear_trend(daily_counts.values)
                    features['trend_acceleration'] = self._calculate_acceleration(daily_counts.values)

                    # Рост
                    if daily_counts.iloc[0] > 0 and daily_counts.iloc[-1] > 0:
                        periods = len(daily_counts) - 1
                        features['cagr'] = (daily_counts.iloc[-1] / daily_counts.iloc[0]) ** (1 / periods) - 1
                    else:
                        features['cagr'] = 0
                else:
                    features.update({
                        'mean_daily': features['total_views'],
                        'std_daily': 0,
                        'cv_daily': 0,
                        'trend_slope': 0,
                        'trend_acceleration': 0,
                        'cagr': 0
                    })

                # Дополнительные фичи для категорий
                # Конверсия (если есть данные о покупках)
                if 'event_type' in category_data.columns:
                    purchases_in_cat = category_data[category_data['event_type'] == 'purchase']
                    features['purchase_count'] = len(purchases_in_cat)
                    features['conversion_rate'] = len(purchases_in_cat) / len(category_data) if len(
                        category_data) > 0 else 0

                # Ценовые фичи (если есть цена)
                if 'price' in category_data.columns:
                    price_data = category_data[category_data['price'] > 0]['price']
                    if len(price_data) > 0:
                        features['avg_price'] = price_data.mean()
                        features['price_std'] = price_data.std()
                        features['min_price'] = price_data.min()
                        features['max_price'] = price_data.max()

                # Географические фичи
                if 'region' in category_data.columns:
                    regions = category_data['region'].unique()
                    features['region_count'] = len(regions)
                    features['is_multi_region'] = len(regions) > 1

                category_features_list.append(features)

            except Exception as e:
                continue

        if category_features_list:
            df = pd.DataFrame(category_features_list)
            df.set_index('trend_id', inplace=True)
            return df
        else:
            return pd.DataFrame()

    def _build_item_features(self, view_events: pd.DataFrame,
                             snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для трендов по товарам"""

        # Берем топ-50 товаров
        item_counts = view_events['item_id'].value_counts()
        top_items = item_counts.head(50).index

        item_features_list = []

        for item_id in top_items:
            try:
                item_data = view_events[view_events['item_id'] == item_id]

                # Базовые фичи
                features = {
                    'trend_id': f"item_{item_id}",
                    'trend_type': 'item',
                    'entity': str(item_id),
                    'total_views': len(item_data),
                    'unique_viewers': item_data['user_id'].nunique(),
                    'first_view': item_data['ts'].min(),
                    'last_view': item_data['ts'].max(),
                }

                # Временные фичи
                item_data['date'] = item_data['ts'].dt.date
                date_range = pd.date_range(start=features['first_view'].date(),
                                           end=snapshot_date.date(), freq='D')
                daily_counts = item_data.groupby('date').size().reindex(date_range.date, fill_value=0)

                # Активных дней
                active_days = max((features['last_view'] - features['first_view']).days + 1, 1)
                features['frequency'] = features['total_views'] / active_days
                features['active_days'] = (daily_counts > 0).sum()

                # Статистики по дням
                if len(daily_counts) > 1:
                    features['mean_daily'] = daily_counts.mean()
                    features['std_daily'] = daily_counts.std()
                    features['cv_daily'] = features['std_daily'] / features['mean_daily'] if features[
                                                                                                 'mean_daily'] > 0 else 0

                    # Тренд
                    features['trend_slope'] = self._calculate_linear_trend(daily_counts.values)

                    # Рост за последние 3 дня
                    if len(daily_counts) >= 4:
                        recent_avg = daily_counts.iloc[-3:].mean()
                        prev_avg = daily_counts.iloc[-6:-3].mean() if len(daily_counts) >= 6 else daily_counts.iloc[0]
                        features['recent_growth'] = (recent_avg - prev_avg) / (prev_avg + 1)
                    else:
                        features['recent_growth'] = 0
                else:
                    features.update({
                        'mean_daily': features['total_views'],
                        'std_daily': 0,
                        'cv_daily': 0,
                        'trend_slope': 0,
                        'recent_growth': 0
                    })

                # Конверсия
                if 'event_type' in item_data.columns:
                    purchases = item_data[item_data['event_type'] == 'purchase']
                    features['purchase_count'] = len(purchases)
                    features['conversion_rate'] = len(purchases) / len(item_data) if len(item_data) > 0 else 0

                # Цена
                if 'price' in item_data.columns:
                    price_data = item_data[item_data['price'] > 0]['price']
                    if len(price_data) > 0:
                        features['avg_price'] = price_data.mean()

                item_features_list.append(features)

            except Exception as e:
                continue

        if item_features_list:
            df = pd.DataFrame(item_features_list)
            df.set_index('trend_id', inplace=True)
            return df
        else:
            return pd.DataFrame()

    def _build_temporal_features(self, events: pd.DataFrame,
                                 snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для временных трендов (глобальные)"""

        if events.empty:
            return pd.DataFrame()

        # Агрегация по часам
        events['hour'] = events['ts'].dt.hour
        hourly_counts = events.groupby('hour').size()

        # Агрегация по дням недели
        events['day_of_week'] = events['ts'].dt.dayofweek
        dow_counts = events.groupby('day_of_week').size()

        features = {
            'trend_id': 'global_temporal',
            'trend_type': 'temporal',
            'entity': 'global',
            'total_events': len(events),
            'unique_users': events['user_id'].nunique(),

            # Часовые паттерны
            'peak_hour': hourly_counts.idxmax() if not hourly_counts.empty else 0,
            'peak_hour_count': hourly_counts.max() if not hourly_counts.empty else 0,
            'morning_events': ((events['hour'] >= 6) & (events['hour'] < 12)).sum(),
            'afternoon_events': ((events['hour'] >= 12) & (events['hour'] < 18)).sum(),
            'evening_events': ((events['hour'] >= 18) & (events['hour'] < 24)).sum(),
            'night_events': ((events['hour'] >= 0) & (events['hour'] < 6)).sum(),

            # Дневные паттерны
            'peak_dow': dow_counts.idxmax() if not dow_counts.empty else 0,
            'weekend_events': (events['day_of_week'] >= 5).sum(),
            'weekday_events': (events['day_of_week'] < 5).sum(),

            # Распределение по типам событий
            'search_ratio': (events['event_type'] == 'search').sum() / len(events) if len(events) > 0 else 0,
            'view_ratio': (events['event_type'] == 'product_view').sum() / len(events) if len(events) > 0 else 0,
            'purchase_ratio': (events['event_type'] == 'purchase').sum() / len(events) if len(events) > 0 else 0,
        }

        # Добавляем волатильность по часам
        if not hourly_counts.empty:
            features['hourly_cv'] = hourly_counts.std() / hourly_counts.mean() if hourly_counts.mean() > 0 else 0

        df = pd.DataFrame([features])
        df.set_index('trend_id', inplace=True)
        return df

    def _calculate_linear_trend(self, values):
        """Рассчитывает линейный тренд (наклон)"""
        if len(values) < 2:
            return 0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)

    def _calculate_acceleration(self, values):
        """Рассчитывает ускорение тренда"""
        if len(values) < 3:
            return 0

        try:
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 2)
            return float(2 * coeffs[0])  # Ускорение = 2 * коэффициент при x²
        except:
            return float(values[-1] - 2 * values[-2] + values[-3])

    def _build_model3_targets(self, target_events: pd.DataFrame,
                              trend_index: pd.Index) -> pd.DataFrame:
        """Таргеты для микро-трендов (полная версия)"""

        targets = pd.DataFrame(index=trend_index)

        # Базовые таргеты для всех трендов
        for col in ['target_future_count', 'target_growth', 'target_peak',
                    'target_continues', 'target_cross_region']:
            targets[col] = 0

        if target_events.empty:
            return targets

        # Анализируем каждый тренд
        for trend_id in trend_index:
            try:
                # Определяем тип тренда и сущность
                if trend_id.startswith('query_'):
                    entity_type = 'search_query'
                    entity = trend_id.split('query_', 1)[1]
                    filter_col = 'search_query'
                elif trend_id.startswith('category_'):
                    entity_type = 'category'
                    entity = trend_id.split('category_', 1)[1]
                    filter_col = 'category'
                elif trend_id.startswith('item_'):
                    entity_type = 'item'
                    entity = trend_id.split('item_', 1)[1]
                    filter_col = 'item_id'
                else:  # global_temporal
                    # Глобальный тренд
                    targets.loc[trend_id, 'target_future_count'] = len(target_events)
                    targets.loc[trend_id, 'target_continues'] = 1
                    continue

                # Фильтруем события для этой сущности
                if entity_type == 'search_query':
                    entity_events = target_events[
                        (target_events['event_type'] == 'search') &
                        (target_events[filter_col] == entity)
                        ]
                else:
                    entity_events = target_events[
                        (target_events[filter_col] == entity)
                    ]

                if not entity_events.empty:
                    # Базовые метрики
                    targets.loc[trend_id, 'target_future_count'] = len(entity_events)
                    targets.loc[trend_id, 'target_continues'] = 1

                    # Уникальные пользователи
                    targets.loc[trend_id, 'target_unique_users'] = entity_events['user_id'].nunique()

                    # Временное распределение
                    if 'ts' in entity_events.columns:
                        entity_events['date'] = entity_events['ts'].dt.date
                        daily_counts = entity_events.groupby('date').size()

                        if len(daily_counts) > 0:
                            targets.loc[trend_id, 'target_peak'] = daily_counts.max()

                            # Рост (последний день vs первый день)
                            if len(daily_counts) >= 2:
                                first_day = daily_counts.iloc[0]
                                last_day = daily_counts.iloc[-1]
                                if first_day > 0:
                                    targets.loc[trend_id, 'target_growth'] = (last_day - first_day) / first_day

                    # Географическое распространение
                    if 'region' in entity_events.columns:
                        regions = entity_events['region'].unique()
                        targets.loc[trend_id, 'target_region_count'] = len(regions)
                        targets.loc[trend_id, 'target_cross_region'] = 1 if len(regions) > 1 else 0

                    # Конверсия для товаров/категорий
                    if entity_type in ['category', 'item'] and 'event_type' in entity_events.columns:
                        purchases = entity_events[entity_events['event_type'] == 'purchase']
                        targets.loc[trend_id, 'target_purchases'] = len(purchases)
                        if len(entity_events) > 0:
                            targets.loc[trend_id, 'target_conversion'] = len(purchases) / len(entity_events)

            except Exception as e:
                # Пропускаем ошибки
                continue

        # Заполняем пропуски
        targets = targets.fillna(0)

        return targets

    # ===== МОДЕЛЬ 4: Adaptive Pricing =====

    def _build_model4_snapshots(self, events, train_dates, val_dates, test_dates,
                                window_back, window_forward):
        """Снапшоты для адаптивного ценообразования"""

        print("\n💰 Building Model 4: Adaptive Pricing...")

        # Фокусируемся на покупках и просмотрах
        price_events = events[events['event_type'].isin(['purchase', 'product_view'])]

        datasets = {}

        for dataset_name, dates in [('train', train_dates), ('val', val_dates), ('test', test_dates)]:
            snapshots = []

            for snapshot_date in dates:
                # Окно для анализа ценовой чувствительности
                feature_start = snapshot_date - timedelta(days=window_back)
                feature_end = snapshot_date

                # Будущие покупки для анализа конверсии
                target_start = snapshot_date
                target_end = snapshot_date + timedelta(days=window_forward)

                feature_events = price_events[
                    (price_events['ts'] >= feature_start) &
                    (price_events['ts'] < feature_end)
                    ]

                target_events = price_events[
                    (price_events['ts'] >= target_start) &
                    (price_events['ts'] < target_end)
                    ]

                # Фичи на уровне товара - С ПРОВЕРКОЙ!
                features = self._build_model4_features(feature_events, snapshot_date)

                if features is None or features.empty:
                    print(f"    {snapshot_date.date()}: No features generated")
                    continue

                targets = self._build_model4_targets(target_events, features.index)

                if not features.empty:
                    snapshot_df = features.join(targets, how='left').fillna(0)
                    snapshot_df['snapshot_date'] = snapshot_date
                    snapshots.append(snapshot_df.reset_index())

            if snapshots:
                datasets[dataset_name] = pd.concat(snapshots, ignore_index=True)
                print(f"  {dataset_name}: {len(datasets[dataset_name]):,} price-snapshots")
            else:
                datasets[dataset_name] = pd.DataFrame()
                print(f"  {dataset_name}: No snapshots generated")

        return datasets

    def _build_model4_features(self, events: pd.DataFrame,
                               snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для адаптивного ценообразования"""

        if events.empty or 'item_id' not in events.columns:
            return pd.DataFrame()

        item_features = []

        # Ограничим количество товаров для тестирования
        unique_items = events['item_id'].unique()
        if len(unique_items) > 100:  # Берем только топ-100 для скорости
            # Выбираем товары с наибольшим количеством событий
            item_counts = events['item_id'].value_counts().head(100)
            unique_items = item_counts.index.tolist()

        for item_id in unique_items:
            try:
                item_events = events[events['item_id'] == item_id]

                features = {
                    'item_id': item_id,
                    'total_views': (item_events['event_type'] == 'product_view').sum(),
                    'total_purchases': (item_events['event_type'] == 'purchase').sum(),
                    'unique_viewers': item_events[item_events['event_type'] == 'product_view']['user_id'].nunique(),
                    'unique_buyers': item_events[item_events['event_type'] == 'purchase']['user_id'].nunique(),
                }

                # Конверсия
                features['conversion_rate'] = (
                    features['total_purchases'] / features['total_views']
                    if features['total_views'] > 0 else 0
                )

                # Ценовая статистика
                purchases = item_events[item_events['event_type'] == 'purchase']
                if not purchases.empty:
                    features['current_price'] = purchases['price'].iloc[-1] if len(purchases) > 0 else 0
                    features['avg_price'] = purchases['price'].mean()
                    features['price_std'] = purchases['price'].std()
                    features['min_price'] = purchases['price'].min()
                    features['max_price'] = purchases['price'].max()
                    features['price_range'] = features['max_price'] - features['min_price']
                else:
                    # Если не было покупок, берем цену из просмотров или 0
                    views_with_price = item_events[
                        (item_events['event_type'] == 'product_view') &
                        (item_events['price'] > 0)
                        ]
                    features['current_price'] = views_with_price['price'].iloc[-1] if len(views_with_price) > 0 else 0
                    features['avg_price'] = features['current_price']
                    features['price_std'] = 0
                    features['min_price'] = features['current_price']
                    features['max_price'] = features['current_price']
                    features['price_range'] = 0

                # Bayesian Beta-Binomial для конверсии
                if features['total_views'] > 0:
                    # Prior: Beta(α=2, β=8) - предполагаем конверсию ~20%
                    alpha_prior = 2
                    beta_prior = 8

                    alpha_post = alpha_prior + features['total_purchases']
                    beta_post = beta_prior + features['total_views'] - features['total_purchases']

                    # MAP оценка (mode of Beta distribution)
                    features['conversion_rate_map'] = (alpha_post - 1) / (alpha_post + beta_post - 2) if (
                                                                                                                 alpha_post + beta_post) > 2 else 0

                    # Mean
                    features['conversion_rate_mean'] = alpha_post / (alpha_post + beta_post)

                    # Standard deviation
                    var = (alpha_post * beta_post) / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
                    features['conversion_rate_std'] = np.sqrt(var) if var > 0 else 0

                    # 90% credible interval (аппроксимация через нормальное распределение)
                    if features['conversion_rate_std'] > 0:
                        z_score = stats.norm.ppf(0.95)
                        margin = z_score * features['conversion_rate_std']
                        features['conversion_rate_lower'] = max(0, features['conversion_rate_mean'] - margin)
                        features['conversion_rate_upper'] = min(1, features['conversion_rate_mean'] + margin)

                # Продвинутая ценовая эластичность
                if not purchases.empty and len(purchases) >= 10:
                    # Группируем по неделям для устойчивости
                    purchases['week'] = purchases['ts'].dt.isocalendar().week
                    weekly_data = purchases.groupby('week').agg({
                        'price': ['mean', 'std'],
                        'event_id': 'count'
                    })
                    weekly_data.columns = ['price_mean', 'price_std', 'quantity']

                    if len(weekly_data) >= 4:
                        # Лог-линейная модель
                        valid_mask = (weekly_data['price_mean'] > 0) & (weekly_data['quantity'] > 0)
                        if valid_mask.sum() >= 3:
                            log_price = np.log(weekly_data.loc[valid_mask, 'price_mean'].values)
                            log_quantity = np.log(weekly_data.loc[valid_mask, 'quantity'].values)

                            # OLS с гетероскедастично-устойчивыми стандартными ошибками
                            X = sm.add_constant(log_price)
                            model = sm.OLS(log_quantity, X)
                            results = model.fit(cov_type='HC3')

                            features['price_elasticity_ols'] = results.params[1]
                            features['elasticity_pvalue'] = results.pvalues[1]
                            features['elasticity_r2'] = results.rsquared

                item_features.append(features)

            except Exception as e:
                # Пропускаем проблемные товары
                continue

        # ВОЗВРАЩАЕМ DATAFRAME - ИСПРАВЛЕНО!
        if item_features:
            df = pd.DataFrame(item_features)
            df.set_index('item_id', inplace=True)
            return df
        else:
            return pd.DataFrame()  # Всегда возвращаем DataFrame, даже пустой

    def _build_model4_targets(self, target_events: pd.DataFrame,
                              item_index: pd.Index) -> pd.DataFrame:
        """Таргеты для адаптивного ценообразования"""

        targets = pd.DataFrame(index=item_index)
        targets['target_sales_count'] = 0
        targets['target_revenue'] = 0
        targets['target_optimal_price'] = 0
        targets['target_price_change_effect'] = 0.0

        if target_events.empty:
            return targets

        purchases = target_events[target_events['event_type'] == 'purchase']

        for item_id in item_index:
            item_purchases = purchases[purchases['item_id'] == item_id]

            if not item_purchases.empty:
                targets.loc[item_id, 'target_sales_count'] = len(item_purchases)
                targets.loc[item_id, 'target_revenue'] = item_purchases['price'].sum()

                # Оптимальная цена (средняя цена при которой были покупки)
                targets.loc[item_id, 'target_optimal_price'] = item_purchases['price'].mean()

                # Эффект изменения цены: корреляция между дневной ценой и дневными продажами в окне target
                item_purchases['date'] = item_purchases['ts'].dt.date
                daily_qty = item_purchases.groupby('date').size()
                daily_price = item_purchases.groupby('date')['price'].mean()
                # Выравниваем индексы
                common_idx = daily_qty.index.intersection(daily_price.index)
                if len(common_idx) >= 3:
                    qty_series = daily_qty.loc[common_idx].astype(float)
                    price_series = daily_price.loc[common_idx].astype(float)
                    if price_series.std() > 0 and qty_series.std() > 0:
                        corr = float(np.corrcoef(price_series.values, qty_series.values)[0, 1])
                        # Инвертируем знак, чтобы положительное значение означало улучшение спроса при снижении цены
                        targets.loc[item_id, 'target_price_change_effect'] = -corr
                    else:
                        targets.loc[item_id, 'target_price_change_effect'] = 0.0
                else:
                    targets.loc[item_id, 'target_price_change_effect'] = 0.0

        return targets

    def save_all_snapshots(self, snapshots_dict: Dict,
                           output_dir: str = "../analytics/data/innovative_snapshots"):
        """Сохраняет все снапшоты"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, datasets in snapshots_dict.items():
            model_dir = output_path / model_name
            model_dir.mkdir(exist_ok=True)

            for dataset_name, df in datasets.items():
                if not df.empty:
                    file_path = model_dir / f"{dataset_name}.parquet"
                    df.to_parquet(file_path, index=False)

                    print(f"💾 Saved {model_name}/{dataset_name}: {len(df):,} rows")

                    # Сохраняем метаданные
                    meta = {
                        'model': model_name,
                        'dataset': dataset_name,
                        'rows': len(df),
                        'columns': df.columns.tolist(),  # Первые 20 колонок
                        'saved_at': datetime.now().isoformat()
                    }

                    with open(model_dir / f"{dataset_name}_meta.json", 'w') as f:
                        json.dump(meta, f, indent=2)


# ===== ЗАПУСК =====

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 INNOVATIVE SNAPSHOT BUILDER FOR 4 MODELS")
    print("=" * 60)

    builder = InnovativeSnapshotBuilder()

    # 1. Тестируем Model 3 отдельно
    builder.test_model3_specific(test_date="2025-01-01")

    # 2. Быстрый тест всех моделей
    builder.quick_test_builders(test_date="2025-01-01")

    # 3. Если тесты проходят, запускаем полный билд
    print("\n" + "=" * 60)
    print("🏗️ STARTING FULL BUILD...")
    print("=" * 60)

    try:
        snapshots = builder.build_all_snapshots(
            train_end="2025-06-01",
            val_end="2025-08-01",
            test_end="2025-09-01",
            window_back_days=30,
            window_forward_days=14
        )

        # Сохраняем
        builder.save_all_snapshots(snapshots)

        print("\n" + "=" * 60)
        print("✅ ВСЕ 4 НАБОРА ДАННЫХ ГОТОВЫ!")
        print("=" * 60)

        # Статистика
        for model_name, datasets in snapshots.items():
            print(f"\n📊 {model_name}:")
            for dataset_name, df in datasets.items():
                if not df.empty:
                    print(f"  {dataset_name}: {len(df):,} samples, {len(df.columns)} features")

    except Exception as e:
        print(f"\n❌ BUILD FAILED: {str(e)}")
        import traceback

        traceback.print_exc()