import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Утилиты для загрузки и подготовки данных для моделей"""

    @staticmethod
    def load_training_data(data_path: str) -> pd.DataFrame:
        """Загружает данные для обучения"""
        try:
            if data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            elif data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            logger.info(f"Loaded data with shape: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    @staticmethod
    def prepare_sequences(user_events: List[Dict], sequence_length: int = 20) -> np.ndarray:
        """Подготавливает последовательности событий для модели"""
        sequences = []

        for events in user_events:
            sequence = []

            for event in events[-sequence_length:]:
                # Кодируем событие в вектор
                event_vector = DataLoader._encode_event(event)
                sequence.append(event_vector)

            # Паддинг если последовательность короче
            while len(sequence) < sequence_length:
                sequence.append([0.0] * 6)  # 6-мерный вектор события

            sequences.append(sequence)

        return np.array(sequences)

    @staticmethod
    def _encode_event(event: Dict) -> List[float]:
        """Кодирует событие в числовой вектор"""
        # Маппинг типов событий
        event_type_mapping = {
            'view': 0.1, 'click': 0.3, 'add_to_cart': 0.7,
            'purchase': 1.0, 'wishlist': 0.5, 'share': 0.6
        }

        # Маппинг категорий
        category_mapping = {
            'electronics': 0.1, 'clothing': 0.2, 'home': 0.3,
            'beauty': 0.4, 'sports': 0.5, 'books': 0.6
        }

        return [
            event_type_mapping.get(event.get('event_type', 'view'), 0.1),
            category_mapping.get(event.get('category', 'general'), 0.0),
            event.get('duration_seconds', 0) / 3600.0,  # Нормализуем до часов
            1.0 if event.get('is_mobile', False) else 0.0,
            event.get('scroll_depth', 0) / 100.0,  # Нормализуем проскролл
            event.get('price', 0) / 10000.0  # Нормализуем цену
        ]


class SequenceGenerator:
    """Генератор последовательностей для обучения временных моделей"""

    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length

    def generate_training_sequences(self, user_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Генерирует последовательности для обучения"""
        sequences = []
        labels = []

        for user_id, user_events in user_data.groupby('user_id'):
            events_sorted = user_events.sort_values('timestamp')

            # Создаем скользящие окна
            for i in range(len(events_sorted) - self.sequence_length):
                sequence = events_sorted.iloc[i:i + self.sequence_length]
                next_event = events_sorted.iloc[i + self.sequence_length]

                # Кодируем последовательность
                encoded_sequence = []
                for _, event in sequence.iterrows():
                    encoded_event = DataLoader._encode_event(event.to_dict())
                    encoded_sequence.append(encoded_event)

                sequences.append(encoded_sequence)
                labels.append(self._encode_label(next_event['event_type']))

        return np.array(sequences), np.array(labels)

    def _encode_label(self, event_type: str) -> int:
        """Кодирует метку для предсказания"""
        label_mapping = {
            'view': 0, 'click': 1, 'add_to_cart': 2, 'purchase': 3
        }
        return label_mapping.get(event_type, 0)