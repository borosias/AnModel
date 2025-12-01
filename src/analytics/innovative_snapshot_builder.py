# innovative_snapshot_builder.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class InnovativeSnapshotBuilder:
    """Создает снапшоты для 4 инновационных моделей"""

    def __init__(self, parquet_dir: str = "../analytics/data/parquet"):
        self.parquet_dir = Path(parquet_dir)
        self.google_trends = self._load_google_trends()

    def _load_google_trends(self) -> pd.DataFrame:
        """Загружаем Google Trends данные (если есть)"""
        trends_file = Path("trends_data/trends_master.parquet")
        if trends_file.exists():
            return pd.read_parquet(trends_file)
        return pd.DataFrame()

    def build_all_snapshots(self,
                            train_end: str,
                            val_end: str,
                            test_end: str,
                            window_back_days: int = 90,  # Увеличили для лучших паттернов
                            window_forward_days: int = 14):  # 14 дней для микро-трендов

        print("🚀 Building innovative snapshots for 4 models...")

        # Загружаем события
        events = self._load_and_prepare_events()

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

        # 4. Ценовая чувствительность
        purchases = events[events['event_type'] == 'purchase']
        if not purchases.empty:
            price_sensitivity = purchases.groupby('user_id')['price'].agg(['mean', 'std', 'min', 'max'])
            price_sensitivity.columns = ['avg_purchase_price', 'price_std', 'min_price', 'max_price']
            price_sensitivity['price_range'] = price_sensitivity['max_price'] - price_sensitivity['min_price']
            user_features = user_features.join(price_sensitivity, how='left')

        # Объединяем все фичи
        user_features = user_features.join(user_sessions, how='left')
        user_features = user_features.join(time_patterns, how='left')

        # Добавляем Google Trends если есть
        if not self.google_trends.empty:
            trends_features = self._add_google_trends_features(snapshot_date)
            # Добавляем ко всем пользователям
            for col in trends_features.columns:
                user_features[col] = trends_features[col].iloc[0] if len(trends_features) > 0 else 0

        # Заполняем пропуски
        user_features = user_features.fillna(0)

        return user_features

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
        features['snapshot_month'] = snapshot_date.month
        features['snapshot_week'] = snapshot_date.isocalendar().week
        features['is_weekend'] = snapshot_date.weekday() >= 5

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
        """Снапшоты для предсказания микро-трендов"""

        print("\n📈 Building Model 3: Micro-Trend Anticipation...")

        # Фокусируемся на поисковых запросах и просмотрах
        trend_events = events[events['event_type'].isin(['search', 'product_view'])]

        datasets = {}

        for dataset_name, dates in [('train', train_dates), ('val', val_dates), ('test', test_dates)]:
            snapshots = []

            for snapshot_date in dates:  # Нужно больше точек для трендов
                # Короткое окно для микро-трендов (7 дней назад)
                feature_start = snapshot_date - timedelta(days=7)
                feature_end = snapshot_date

                # Будущий тренд (следующие 3-7 дней)
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
                targets = self._build_model3_targets(target_events, features.index)

                if not features.empty and not targets.empty:
                    snapshot_df = features.join(targets, how='left').fillna(0)
                    snapshot_df['snapshot_date'] = snapshot_date
                    snapshots.append(snapshot_df.reset_index())

            if snapshots:
                datasets[dataset_name] = pd.concat(snapshots, ignore_index=True)
                print(f"  {dataset_name}: {len(datasets[dataset_name]):,} trend-snapshots")
            else:
                datasets[dataset_name] = pd.DataFrame()

        return datasets

    def _build_model3_features(self, events: pd.DataFrame,
                               snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для обнаружения микро-трендов"""

        # Анализируем поисковые запросы
        search_events = events[events['event_type'] == 'search']

        if search_events.empty or 'search_query' not in search_events.columns:
            return pd.DataFrame()

        # Группируем по запросам
        query_features = []

        for query in search_events['search_query'].unique():
            if not query or query == '':
                continue

            query_events = search_events[search_events['search_query'] == query]

            features = {
                'query': query,
                'total_searches': len(query_events),
                'unique_users': query_events['user_id'].nunique(),
                'first_seen': query_events['ts'].min(),
                'last_seen': query_events['ts'].max(),
                'search_frequency': len(query_events) / 7  # на день
            }

            # Временное распределение (рост/падение)
            query_events['date'] = query_events['ts'].dt.date
            daily_counts = query_events.groupby('date').size()

            if len(daily_counts) > 1:
                features['trend_growth_rate'] = (daily_counts.iloc[-1] - daily_counts.iloc[0]) / (
                            daily_counts.iloc[0] + 1)
                features['trend_acceleration'] = self._calculate_acceleration(daily_counts.values)
            else:
                features['trend_growth_rate'] = 0
                features['trend_acceleration'] = 0

            # Географическое распространение
            regions = query_events['region'].unique()
            features['region_spread'] = len(regions)
            features['is_multiregion'] = len(regions) > 1

            query_features.append(features)

        if not query_features:
            return pd.DataFrame()

        df = pd.DataFrame(query_features)
        df.set_index('query', inplace=True)

        # Добавляем Google Trends если есть
        if not self.google_trends.empty:
            df = self._enrich_with_google_trends(df, snapshot_date)

        return df

    def _calculate_acceleration(self, values):
        """Рассчитывает ускорение тренда"""
        if len(values) < 3:
            return 0

        # Простая вторая производная
        return (values[-1] - 2 * values[-2] + values[-3]) / max(values[-3], 1)

    def _enrich_with_google_trends(self, queries_df: pd.DataFrame,
                                   snapshot_date: datetime) -> pd.DataFrame:
        """Обогащает запросы Google Trends данными"""

        start_date = snapshot_date - timedelta(days=7)

        for query in queries_df.index:
            query_trends = self.google_trends[
                (self.google_trends['query'] == query) &
                (self.google_trends['date'] >= start_date.date()) &
                (self.google_trends['date'] <= snapshot_date.date())
                ]

            if not query_trends.empty:
                queries_df.loc[query, 'google_avg_popularity'] = query_trends['popularity'].mean()
                queries_df.loc[query, 'google_trend'] = self._calculate_trend(query_trends['popularity'].values)
            else:
                queries_df.loc[query, 'google_avg_popularity'] = 0
                queries_df.loc[query, 'google_trend'] = 0

        return queries_df

    def _calculate_trend(self, values):
        """Рассчитывает тренд из значений"""
        if len(values) < 2:
            return 0

        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Наклон линии

    def _build_model3_targets(self, target_events: pd.DataFrame,
                              query_index: pd.Index) -> pd.DataFrame:
        """Таргеты для микро-трендов"""

        targets = pd.DataFrame(index=query_index)
        targets['target_future_searches'] = 0
        targets['target_trend_continues'] = 0
        targets['target_peak_in_days'] = 999

        if target_events.empty or 'search_query' not in target_events.columns:
            return targets

        search_events = target_events[target_events['event_type'] == 'search']

        for query in query_index:
            query_events = search_events[search_events['search_query'] == query]

            if not query_events.empty:
                targets.loc[query, 'target_future_searches'] = len(query_events)
                targets.loc[query, 'target_trend_continues'] = 1

                # Когда пик? (день с максимальным количеством поисков)
                daily_counts = query_events.groupby(query_events['ts'].dt.date).size()
                if len(daily_counts) > 0:
                    peak_day = daily_counts.idxmax()
                    targets.loc[query, 'target_peak_in_days'] = (peak_day - query_events['ts'].min().date()).days

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

                # Фичи на уровне товара
                features = self._build_model4_features(feature_events, snapshot_date)
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

        return datasets

    def _build_model4_features(self, events: pd.DataFrame,
                               snapshot_date: datetime) -> pd.DataFrame:
        """Фичи для адаптивного ценообразования"""

        if events.empty or 'item_id' not in events.columns:
            return pd.DataFrame()

        item_features = []

        for item_id in events['item_id'].unique():
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

            # Эластичность спроса (упрощенная)
            if len(purchases) > 1:
                # Разделяем на высокие и низкие цены
                median_price = purchases['price'].median()
                high_price_sales = purchases[purchases['price'] > median_price].shape[0]
                low_price_sales = purchases[purchases['price'] <= median_price].shape[0]

                features['price_elasticity'] = (
                        (high_price_sales - low_price_sales) / (high_price_sales + low_price_sales + 1)
                )
            else:
                features['price_elasticity'] = 0

            # Временные паттерны цен
            item_events['date'] = item_events['ts'].dt.date
            price_over_time = purchases.groupby('date')['price'].mean()

            if len(price_over_time) > 1:
                features['price_trend'] = self._calculate_trend(price_over_time.values)
                features['price_volatility'] = price_over_time.std()
            else:
                features['price_trend'] = 0
                features['price_volatility'] = 0

            # Конкурентная среда (похожие товары по категории)
            if 'category' in item_events.columns:
                category = item_events['category'].iloc[0] if not item_events['category'].empty else ''
                features['category'] = category

                # Анализ цен в категории
                category_events = events[events['category'] == category]
                category_purchases = category_events[category_events['event_type'] == 'purchase']

                if not category_purchases.empty:
                    features['category_avg_price'] = category_purchases['price'].mean()
                    features['price_position'] = (
                        features['current_price'] / features['category_avg_price']
                        if features['category_avg_price'] > 0 else 1
                    )
                else:
                    features['category_avg_price'] = features['current_price']
                    features['price_position'] = 1

            item_features.append(features)

        if not item_features:
            return pd.DataFrame()

        df = pd.DataFrame(item_features)
        df.set_index('item_id', inplace=True)

        return df

    def _build_model4_targets(self, target_events: pd.DataFrame,
                              item_index: pd.Index) -> pd.DataFrame:
        """Таргеты для адаптивного ценообразования"""

        targets = pd.DataFrame(index=item_index)
        targets['target_sales_count'] = 0
        targets['target_revenue'] = 0
        targets['target_optimal_price'] = 0
        targets['target_price_change_effect'] = 0

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

                # Эффект изменения цены (пока упрощенно)
                # В реальности нужно сравнить с предыдущими ценами
                targets.loc[item_id, 'target_price_change_effect'] = 1.0

        return targets

    def save_all_snapshots(self, snapshots_dict: Dict,
                           output_dir: str = "src/analytics/data/innovative_snapshots"):
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

    # Строим все снапшоты
    snapshots = builder.build_all_snapshots(
        train_end="2024-01-20",
        val_end="2024-01-27",
        test_end="2024-02-03",
        window_back_days=90,
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