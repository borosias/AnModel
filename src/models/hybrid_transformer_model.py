import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridInterestPredictor(nn.Module):
    """Гибридная трансформерная модель для прогнозирования интереса"""

    def __init__(self,
                 tabular_dim: int,
                 sequence_dim: int,
                 sequence_length: int = 20,
                 num_heads: int = 8,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()

        self.tabular_dim = tabular_dim
        self.sequence_dim = sequence_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        # Ветка для табличных данных
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Позиционное кодирование для последовательностей
        self.positional_encoding = PositionalEncoding(64, sequence_length, dropout)

        # Ветка для временных последовательностей (Transformer)
        self.sequence_projection = nn.Linear(sequence_dim, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention механизм для последовательности
        self.sequence_attention = nn.MultiheadAttention(64, num_heads, dropout=dropout, batch_first=True)

        # Мультимодальное слияние с attention
        self.fusion_attention = nn.MultiheadAttention(128, num_heads, dropout=dropout, batch_first=True)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # 3 класса: низкий/средний/высокий интерес
            nn.Softmax(dim=1)
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self,
                tabular_data: torch.Tensor,
                sequence_data: torch.Tensor,
                sequence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass модели

        Args:
            tabular_data: Тензор табличных данных [batch_size, tabular_dim]
            sequence_data: Тензор последовательностей [batch_size, seq_len, sequence_dim]
            sequence_mask: Маска последовательности [batch_size, seq_len]

        Returns:
            Предсказания модели [batch_size, 3]
        """
        batch_size = tabular_data.size(0)

        # 1. Кодируем табличные данные
        tabular_encoded = self.tabular_encoder(tabular_data)  # [batch_size, 64]

        # 2. Кодируем последовательность
        sequence_proj = self.sequence_projection(sequence_data)  # [batch_size, seq_len, 64]
        sequence_proj = self.positional_encoding(sequence_proj)

        # Применяем Transformer encoder
        sequence_encoded = self.sequence_encoder(sequence_proj, src_key_padding_mask=sequence_mask)

        # Применяем attention к последовательности
        attended_sequence, _ = self.sequence_attention(
            sequence_encoded, sequence_encoded, sequence_encoded,
            key_padding_mask=sequence_mask
        )

        # Берем взвешенное среднее по последовательности
        if sequence_mask is not None:
            # Маскируем padded элементы
            sequence_mask = sequence_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            attended_sequence = attended_sequence.masked_fill(sequence_mask, 0)
            sequence_lengths = (~sequence_mask).sum(dim=1)  # [batch_size, 1]
            sequence_output = attended_sequence.sum(dim=1) / sequence_lengths.clamp(min=1)
        else:
            sequence_output = attended_sequence.mean(dim=1)  # [batch_size, 64]

        # 3. Мультимодальное слияние
        # Повторяем табличные данные для конкатенации
        tabular_expanded = tabular_encoded.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Конкатенируем табличные и последовательные данные
        multimodal_data = torch.cat([tabular_expanded, sequence_encoded], dim=2)  # [batch_size, seq_len, 128]

        # Применяем cross-attention между модальностями
        fused_output, _ = self.fusion_attention(
            multimodal_data, multimodal_data, multimodal_data,
            key_padding_mask=sequence_mask
        )

        # Усредняем по последовательности
        if sequence_mask is not None:
            fused_output = fused_output.masked_fill(sequence_mask.unsqueeze(-1), 0)
            fused_mean = fused_output.sum(dim=1) / sequence_lengths.clamp(min=1)
        else:
            fused_mean = fused_output.mean(dim=1)

        # 4. Классификация
        output = self.classifier(fused_mean)

        return output

    def predict_interest_level(self,
                               tabular_data: torch.Tensor,
                               sequence_data: torch.Tensor,
                               sequence_mask: Optional[torch.Tensor] = None) -> dict:
        """Предсказывает уровень интереса с дополнительной информацией"""
        with torch.no_grad():
            predictions = self.forward(tabular_data, sequence_data, sequence_mask)

            # Получаем вероятности для каждого класса
            probs = predictions.cpu().numpy()

            # Определяем предсказанный класс
            predicted_class = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)

            # Маппинг классов на уровни интереса
            interest_levels = {0: 'low', 1: 'medium', 2: 'high'}

            results = []
            for i in range(len(predicted_class)):
                results.append({
                    'interest_level': interest_levels[predicted_class[i]],
                    'confidence': confidence[i],
                    'probabilities': {
                        'low': probs[i][0],
                        'medium': probs[i][1],
                        'high': probs[i][2]
                    },
                    'recommendation_strength': self._calculate_recommendation_strength(
                        predicted_class[i], confidence[i]
                    )
                })

            return results if len(results) > 1 else results[0]

    def _calculate_recommendation_strength(self, predicted_class: int, confidence: float) -> str:
        """Вычисляет силу рекомендации на основе предсказания"""
        if predicted_class == 2 and confidence > 0.8:
            return "strong_recommend"
        elif predicted_class == 2 and confidence > 0.6:
            return "recommend"
        elif predicted_class == 1 and confidence > 0.7:
            return "suggest"
        elif predicted_class == 0 and confidence > 0.8:
            return "avoid"
        else:
            return "neutral"


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Создаем позиционное кодирование
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeSeriesFeatureExtractor(nn.Module):
    """Извлекает фичи из временных рядов для использования в гибридной модели"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.attention_pool = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, hidden_dim]
        """
        # Транспонируем для conv1d
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]

        # Применяем свертки
        conv_out = self.conv1d(x)  # [batch_size, hidden_dim, seq_len]

        # Транспонируем обратно для attention
        conv_out = conv_out.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]

        # Применяем attention pooling
        attended, _ = self.attention_pool(conv_out, conv_out, conv_out)

        # Усредняем по последовательности
        output = attended.mean(dim=1)  # [batch_size, hidden_dim]

        return output