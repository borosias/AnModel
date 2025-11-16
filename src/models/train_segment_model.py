import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import os

logger = logging.getLogger(__name__)


class SegmentModelTrainer:
    """Тренер для модели сегментации пользователей"""

    def __init__(self, model_save_path: str = "models/saved_models/"):
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        os.makedirs(model_save_path, exist_ok=True)

    def find_optimal_clusters(self, data: pd.DataFrame, max_clusters: int = 10) -> int:
        """Находит оптимальное количество кластеров"""
        logger.info("Finding optimal number of clusters...")

        features = self._extract_segmentation_features(data)
        X_scaled = self.scaler.fit_transform(features)

        best_score = -1
        best_n = 2

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Silhouette score для оценки качества кластеризации
            score = silhouette_score(X_scaled, cluster_labels)

            logger.info(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_n = n_clusters

        logger.info(f"Optimal number of clusters: {best_n} with score {best_score:.4f}")
        return best_n

    def train_segmentation_model(self,
                                 user_data: pd.DataFrame,
                                 n_clusters: int = None) -> dict:
        """Обучает модель сегментации"""

        logger.info("Training user segmentation model...")

        try:
            # Извлечение фич
            features = self._extract_segmentation_features(user_data)
            X_scaled = self.scaler.fit_transform(features)

            # Определение оптимального числа кластеров если не задано
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters(user_data)

            # Кластеризация
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Создание профилей сегментов
            segment_profiles = self._build_segment_profiles(user_data, cluster_labels)

            # Сохранение модели
            model_path = os.path.join(self.model_save_path, "segment_model.pkl")
            import joblib
            joblib.dump({
                'kmeans': kmeans,
                'scaler': self.scaler,
                'segment_profiles': segment_profiles
            }, model_path)

            # Результаты
            results = {
                'n_clusters': n_clusters,
                'cluster_distribution': pd.Series(cluster_labels).value_counts().to_dict(),
                'segment_profiles': segment_profiles,
                'model_path': model_path,
                'silhouette_score': silhouette_score(X_scaled, cluster_labels)
            }

            logger.info(f"Segmentation model trained with {n_clusters} clusters")

            return results

        except Exception as e:
            logger.error(f"Error in segmentation training: {e}")
            raise

    def _extract_segmentation_features(self, user_data: pd.DataFrame) -> np.ndarray:
        """Извлекает фичи для сегментации"""
        features = []

        for _, row in user_data.iterrows():
            feature_vector = [
                row.get('purchase_frequency', 0),
                row.get('avg_order_value', 0),
                row.get('days_since_last_purchase', 30),
                row.get('total_orders', 0),
                row.get('preferred_category_diversity', 0),
                row.get('price_sensitivity', 0.5),
                row.get('avg_session_duration', 0) / 3600.0,
                row.get('return_rate', 0),
            ]
            features.append(feature_vector)

        return np.array(features)

    def _build_segment_profiles(self, user_data: pd.DataFrame, cluster_labels: np.ndarray) -> dict:
        """Строит профили для каждого сегмента"""
        user_data = user_data.copy()
        user_data['segment'] = cluster_labels

        segment_profiles = {}

        for segment_id in range(len(np.unique(cluster_labels))):
            segment_users = user_data[user_data['segment'] == segment_id]

            # Базовые метрики
            profile = {
                'size': len(segment_users),
                'avg_purchase_frequency': segment_users['purchase_frequency'].mean(),
                'avg_order_value': segment_users['avg_order_value'].mean(),
                'avg_days_since_purchase': segment_users['days_since_last_purchase'].mean(),
                'total_users': len(segment_users)
            }

            # Предпочтения по категориям
            if 'preferred_category' in segment_users.columns:
                category_dist = segment_users['preferred_category'].value_counts(normalize=True)
                profile['category_preferences'] = category_dist.head(5).to_dict()

            # Поведенческие паттерны
            if 'avg_session_duration' in segment_users.columns:
                profile['avg_session_duration'] = segment_users['avg_session_duration'].mean()

            if 'device_preference' in segment_users.columns:
                device_dist = segment_users['device_preference'].value_counts(normalize=True)
                profile['device_preference'] = device_dist.to_dict()

            # Название сегмента на основе характеристик
            profile['segment_name'] = self._generate_segment_name(profile)
            profile['description'] = self._generate_segment_description(profile)

            segment_profiles[segment_id] = profile

        return segment_profiles

    def _generate_segment_name(self, profile: dict) -> str:
        """Генерирует название для сегмента"""
        freq = profile['avg_purchase_frequency']
        value = profile['avg_order_value']

        if freq > 0.8 and value > 1500:
            return "VIP_клиенты"
        elif freq > 0.5 and value > 1000:
            return "Лояльные_покупатели"
        elif freq > 0.3:
            return "Активные_покупатели"
        elif value > 2000:
            return "Крупные_покупатели"
        else:
            return "Случайные_покупатели"

    def _generate_segment_description(self, profile: dict) -> str:
        """Генерирует описание для сегмента"""
        name = profile['segment_name']

        descriptions = {
            "VIP_клиенты": "Высокочастотные покупки с большим средним чеком",
            "Лояльные_покупатели": "Регулярные покупки со стабильным средним чеком",
            "Активные_покупатели": "Частые покупки с умеренным средним чеком",
            "Крупные_покупатели": "Редкие но крупные покупки",
            "Случайные_покупатели": "Нерегулярные покупки с низким средним чеком"
        }

        return descriptions.get(name, "Сегмент пользователей")


def main():
    """Основная функция для обучения модели сегментации"""
    import argparse

    parser = argparse.ArgumentParser(description='Train user segmentation model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to user data')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters (optional)')

    args = parser.parse_args()

    # Загрузка данных
    try:
        data = pd.read_parquet(args.data_path)
        logger.info(f"Loaded data from {args.data_path}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Обучение модели
    trainer = SegmentModelTrainer()

    try:
        results = trainer.train_segmentation_model(
            user_data=data,
            n_clusters=args.n_clusters
        )

        # Сохранение результатов
        results_path = os.path.join(trainer.model_save_path, "segmentation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Segmentation training completed. Results saved to {results_path}")

    except Exception as e:
        logger.error(f"Segmentation training failed: {e}")
        raise


if __name__ == "__main__":
    main()