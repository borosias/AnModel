# innovative_models_pro.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, LabelEncoder
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime
import pickle


# ============ UTILITY CLASSES ============

class BayesianCalibrator:
    """Bayesian калибровка вероятностей с Beta распределением"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.fitted = False

    def fit(self, y_true, y_pred):
        """Обучает Beta распределение на калиброванных вероятностях"""
        from scipy.special import digamma, gammaln
        from scipy.optimize import minimize

        # Преобразуем в numpy
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Фильтруем некорректные значения
        mask = (y_pred > 0) & (y_pred < 1) & (~np.isnan(y_pred)) & (~np.isnan(y_true))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 10:  # Слишком мало данных для калибровки
            self.alpha = self.alpha_prior
            self.beta = self.beta_prior
            self.fitted = True
            return self

        # Метод моментов для начального приближения
        if np.any(y_true == 1):
            mean = np.mean(y_pred[y_true == 1])
            var = np.var(y_pred[y_true == 1])
        else:
            mean = np.mean(y_pred)
            var = np.var(y_pred)

        if var > mean * (1 - mean):
            var = mean * (1 - mean) * 0.9

        if mean * (1 - mean) / var - 1 <= 0:
            # Невалидные параметры, используем приоры
            self.alpha = self.alpha_prior
            self.beta = self.beta_prior
            self.fitted = True
            return self

        alpha0 = mean * (mean * (1 - mean) / var - 1)
        beta0 = (1 - mean) * (mean * (1 - mean) / var - 1)

        # Защита от некорректных начальных значений
        alpha0 = max(alpha0, 0.1)
        beta0 = max(beta0, 0.1)

        # MLE оценка через gradient descent
        def neg_log_likelihood(params):
            alpha, beta = params
            # Prevent invalid parameters
            alpha = max(alpha, 0.1)
            beta = max(beta, 0.1)

            # Log-likelihood using scipy.gammaln instead of torch.lgamma
            ll = np.sum((alpha - 1) * np.log(y_pred + 1e-10) +
                        (beta - 1) * np.log(1 - y_pred + 1e-10) -
                        gammaln(alpha + beta) +
                        gammaln(alpha) +
                        gammaln(beta))

            # Add prior
            prior = (self.alpha_prior - 1) * np.log(alpha) + (self.beta_prior - 1) * np.log(beta)

            return -(ll + prior)

        try:
            # Optimization
            result = minimize(neg_log_likelihood, [alpha0, beta0],
                              bounds=[(0.1, 100), (0.1, 100)],
                              method='L-BFGS-B',
                              options={'maxiter': 100})

            self.alpha = float(result.x[0])
            self.beta = float(result.x[1])
            self.fitted = True

        except Exception as e:
            print(f"Warning: Bayesian calibration failed, using prior: {e}")
            self.alpha = self.alpha_prior
            self.beta = self.beta_prior
            self.fitted = True

        return self

    def calibrate(self, y_pred):
        """Калибрует предсказанные вероятности"""
        if not self.fitted:
            return y_pred

        y_pred = np.array(y_pred).flatten()

        # Bayesian update: posterior mean
        calibrated = (y_pred * self.alpha + (1 - y_pred) * self.alpha_prior) / \
                     (y_pred * self.alpha + (1 - y_pred) * self.alpha_prior +
                      y_pred * self.beta + (1 - y_pred) * self.beta_prior)

        return np.clip(calibrated, 1e-10, 1 - 1e-10)

    def confidence_interval(self, y_pred, confidence=0.9):
        """Возвращает Bayesian credible intervals"""
        from scipy.stats import beta

        if not self.fitted:
            return y_pred - 0.1, y_pred + 0.1

        y_pred = np.array(y_pred).flatten()
        lower = []
        upper = []

        for p in y_pred:
            # Posterior parameters
            a = p * self.alpha + (1 - p) * self.alpha_prior
            b = p * self.beta + (1 - p) * self.beta_prior

            # Защита от некорректных параметров
            a = max(a, 0.1)
            b = max(b, 0.1)

            ci = beta.interval(confidence, a, b)
            lower.append(ci[0])
            upper.append(ci[1])

        return np.array(lower), np.array(upper)


class QuantileLoss(nn.Module):
    """Pinball loss для quantile regression"""

    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """preds: [batch, num_quantiles]"""
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


# ============ МОДЕЛЬ 1: CONTEXT-AWARE PURCHASE PREDICTION ============

class MultiModalTransformer(nn.Module):
    """
    Упрощенная архитектура для tabular данных
    """

    def __init__(self,
                 num_numerical_features: int = 50,
                 categorical_dims: Dict[str, int] = None,
                 hidden_dim: int = 128,
                 dropout: float = 0.3):  # Увеличиваем dropout
        super().__init__()

        if categorical_dims is None:
            categorical_dims = {}

        # 1. Embedding для КАТЕГОРИАЛЬНЫХ фич (только настоящие категории)
        self.categorical_embeddings = nn.ModuleDict()
        for name, num_categories in categorical_dims.items():
            if num_categories > 1 and num_categories < 100:  # Только настоящие категории
                # Размер embedding = min(50, num_categories // 2)
                emb_dim = min(50, max(2, num_categories // 2))
                self.categorical_embeddings[name] = nn.Embedding(num_categories, emb_dim)

        # 2. Рассчитываем общую размерность входа
        total_embedding_dim = sum([emb.embedding_dim for emb in self.categorical_embeddings.values()])
        total_input_dim = num_numerical_features + total_embedding_dim

        # 3. Основная MLP архитектура
        self.mlp = nn.Sequential(
            # Первый слой
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Второй слой
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Третий слой
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Выходной слой
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, numerical_features, categorical_features=None):
        # 1. Обрабатываем числовые фичи
        features = [numerical_features]

        # 2. Обрабатываем категориальные фичи
        if categorical_features and self.categorical_embeddings:
            for name, embedding in self.categorical_embeddings.items():
                if name in categorical_features:
                    cat_emb = embedding(categorical_features[name])
                    features.append(cat_emb)

        # 3. Конкатенируем все фичи
        x = torch.cat(features, dim=-1)

        # 4. Пропускаем через MLP
        outputs = {}
        outputs['purchase_prob'] = self.mlp(x).squeeze(-1)

        return outputs


class ContextAwareModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classifier = lgb.LGBMClassifier(random_state=self.random_state, n_estimators=200)
        self.classifier_calibrator: Optional[CalibratedClassifierCV] = None
        self.reg_days = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.reg_amount = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.oencoder: Optional[OrdinalEncoder] = None
        self.scaler: Optional[RobustScaler] = None
        self.feature_columns: List[str] = []
        self.trained = False

    def _detect_columns(self, df: pd.DataFrame):
        cols = [c for c in df.columns if not c.startswith("target_") and c not in ("snapshot_date", "user_id", "customer_id", "snapshot")]
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in cat if df[c].nunique() > 1 and df[c].nunique() <= 200]
        self.num_cols = num
        self.cat_cols = cat

    def _fit_preprocessors(self, df: pd.DataFrame):
        if self.num_cols:
            self.scaler = RobustScaler()
            self.scaler.fit(df[self.num_cols].fillna(0).astype(float))
        if self.cat_cols:
            self.oencoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.oencoder.fit(df[self.cat_cols].astype(str).fillna("missing"))

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        X_num = pd.DataFrame(index=df.index)
        if self.num_cols:
            X_num = pd.DataFrame(self.scaler.transform(df[self.num_cols].fillna(0).astype(float)), columns=self.num_cols, index=df.index)
        X_cat = pd.DataFrame(index=df.index)
        if self.cat_cols:
            enc = self.oencoder.transform(df[self.cat_cols].astype(str).fillna("missing"))
            X_cat = pd.DataFrame(enc, columns=self.cat_cols, index=df.index)
        X = pd.concat([X_num, X_cat], axis=1)
        if not self.feature_columns:
            self.feature_columns = list(X.columns)
        missing = [c for c in self.feature_columns if c not in X.columns]
        for c in missing:
            X[c] = 0.0
        X = X[self.feature_columns]
        return X

    def train(self, df: pd.DataFrame, calibrate: bool = True, test_size: float = 0.15):
        if df is None or df.empty:
            raise ValueError("Empty training dataframe")
        if "target_will_purchase" not in df.columns:
            raise ValueError("target_will_purchase missing in training data")
        self._detect_columns(df)
        self._fit_preprocessors(df)
        X = self._transform_df(df)
        y_prob = df["target_will_purchase"].astype(float).fillna(0).values
        X_train, X_hold, y_train, y_hold = train_test_split(X, y_prob, test_size=test_size, random_state=self.random_state, stratify=np.where(y_prob>0,1,0))
        self.classifier.fit(X_train, y_train)
        if calibrate:
            try:
                self.classifier_calibrator = CalibratedClassifierCV(self.classifier, method="isotonic", cv="prefit")
                self.classifier_calibrator.fit(X_hold, y_hold)
            except Exception:
                self.classifier_calibrator = None
        if "target_days_to_purchase" in df.columns:
            y_days = df["target_days_to_purchase"].astype(float).fillna(0).values
            self.reg_days.fit(X, y_days)
        if "target_purchase_amount" in df.columns:
            y_amt = df["target_purchase_amount"].astype(float).fillna(0).values
            self.reg_amount.fit(X, y_amt)
        self.trained = True

    def predict(self, df: pd.DataFrame, return_proba: bool = True) -> Dict:
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        X = self._transform_df(df)
        out: Dict = {}
        try:
            if self.classifier_calibrator is not None:
                probs = self.classifier_calibrator.predict_proba(X)[:, 1]
            else:
                probs = self.classifier.predict_proba(X)[:, 1]
            out["purchase_probability"] = probs
            if not return_proba:
                out["will_purchase"] = (probs > 0.5).astype(int)
        except Exception:
            out["purchase_probability"] = np.zeros(len(X))
            if not return_proba:
                out["will_purchase"] = np.zeros(len(X), dtype=int)
        try:
            if hasattr(self.reg_days, "predict"):
                out["days_to_purchase"] = self.reg_days.predict(X)
        except Exception:
            out["days_to_purchase"] = np.zeros(len(X))
        try:
            if hasattr(self.reg_amount, "predict"):
                out["purchase_amount"] = self.reg_amount.predict(X)
        except Exception:
            out["purchase_amount"] = np.zeros(len(X))
        return out

    def save(self, path: str):
        state = {
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "oencoder": self.oencoder,
            "classifier": self.classifier,
            "classifier_calibrator": self.classifier_calibrator,
            "reg_days": self.reg_days,
            "reg_amount": self.reg_amount,
            "random_state": self.random_state
        }
        joblib.dump(state, path)

    def load(self, path: str):
        state = joblib.load(path)
        self.num_cols = state.get("num_cols", [])
        self.cat_cols = state.get("cat_cols", [])
        self.feature_columns = state.get("feature_columns", [])
        self.scaler = state.get("scaler", None)
        self.oencoder = state.get("oencoder", None)
        self.classifier = state.get("classifier", self.classifier)
        self.classifier_calibrator = state.get("classifier_calibrator", None)
        self.reg_days = state.get("reg_days", self.reg_days)
        self.reg_amount = state.get("reg_amount", self.reg_amount)
        self.random_state = state.get("random_state", self.random_state)
        self.trained = True

# ============ МОДЕЛЬ 2: CROSS-REGION DEMAND TRANSFER ============

class GraphAttentionNetwork(nn.Module):
    """GAT с multi-head attention и residual connections"""

    def __init__(self,
                 node_features: int,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_edge_features: bool = True):
        super().__init__()

        self.use_edge_features = use_edge_features

        # Node feature projection
        self.node_projection = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Edge feature projection (если есть)
        if use_edge_features:
            self.edge_projection = nn.Sequential(
                nn.Linear(1, hidden_dim // num_heads),
                nn.GELU()
            )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2)  # [demand, revenue]
        )

        # Residual connections
        self.residual_projection = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i > 0 else nn.Identity()
            for i in range(num_layers)
        ])

    def forward(self, node_features, adjacency_matrix, edge_features=None):
        # Project node features
        h = self.node_projection(node_features)  # [batch, nodes, hidden]

        # Project edge features если есть
        if self.use_edge_features and edge_features is not None:
            edge_h = self.edge_projection(edge_features.unsqueeze(-1))
        else:
            edge_h = None

        # Multiple GAT layers
        for i, layer in enumerate(self.gat_layers):
            residual = self.residual_projection[i](h)
            h = layer(h, adjacency_matrix, edge_h) + residual
            h = F.gelu(h)

        # Global pooling (attention pooling)
        attention_weights = torch.softmax(
            torch.matmul(h, h.transpose(-2, -1)).mean(dim=-1),
            dim=-1
        ).unsqueeze(-1)
        global_representation = (h * attention_weights).sum(dim=1)

        # Predict
        predictions = self.output_projection(global_representation)

        return predictions


class GraphAttentionLayer(nn.Module):
    """Один слой Graph Attention"""

    def __init__(self, in_features, out_features, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        # Linear projections
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)

        # Edge attention (если есть edge features)
        self.edge_attention = nn.Linear(self.head_dim * 2, 1)

        # Output projection
        self.out_proj = nn.Linear(out_features, out_features)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, edge_features=None):
        batch_size, num_nodes, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, nodes, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply adjacency mask
        adj_mask = adj.unsqueeze(1)  # [batch, 1, nodes, nodes]
        scores = scores.masked_fill(adj_mask == 0, -1e9)

        # Add edge features если есть
        if edge_features is not None:
            edge_scores = self.edge_attention(
                torch.cat([
                    Q.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1),
                    K.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
                ], dim=-1)
            ).squeeze(-1)
            scores = scores + edge_scores

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        out = torch.matmul(attention_weights, V)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        out = self.out_proj(out)

        return out


class CrossRegionModel:
    """Реальная Graph Neural Network для трансфера спроса"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.region_encoder = {}
        self.feature_scalers = {}
        self.graph_builder = None

    def _build_graph(self, df: pd.DataFrame, method='correlation'):
        """Строит граф регионов с разными типами связей"""

        regions = sorted(df['region'].unique())
        num_regions = len(regions)

        # Инициализируем матрицы
        adjacency = np.zeros((num_regions, num_regions))
        edge_weights = np.zeros((num_regions, num_regions))

        # Метод 1: Корреляция временных рядов
        if method == 'correlation':
            region_data = {}
            for i, region in enumerate(regions):
                region_df = df[df['region'] == region]
                if not region_df.empty:
                    # Агрегируем спрос по дням
                    daily_demand = region_df.groupby('snapshot_date')['target_purchase_count'].sum()
                    region_data[region] = daily_demand

            # Вычисляем попарные корреляции
            for i in range(num_regions):
                for j in range(i + 1, num_regions):
                    region_i = regions[i]
                    region_j = regions[j]

                    if region_i in region_data and region_j in region_data:
                        series_i = region_data[region_i]
                        series_j = region_data[region_j]

                        # Выравниваем временные ряды
                        common_dates = series_i.index.intersection(series_j.index)
                        if len(common_dates) >= 7:
                            corr = np.corrcoef(
                                series_i.loc[common_dates].values,
                                series_j.loc[common_dates].values
                            )[0, 1]

                            if abs(corr) > 0.3:  # Порог корреляции
                                adjacency[i, j] = 1
                                adjacency[j, i] = 1
                                edge_weights[i, j] = corr
                                edge_weights[j, i] = corr

        # Метод 2: Географическая близость (если есть координаты)
        elif method == 'geographic':
            # Здесь нужно добавить логику для географических координат
            # Для примера используем случайные расстояния
            pass

        # Добавляем self-connections
        np.fill_diagonal(adjacency, 1)
        np.fill_diagonal(edge_weights, 1)

        return adjacency, edge_weights

    def _prepare_graph_data(self, df: pd.DataFrame):
        """Подготавливает данные для GNN"""

        # Кодируем регионы
        regions = sorted(df['region'].unique())
        if not self.region_encoder:
            self.region_encoder = {region: idx for idx, region in enumerate(regions)}
            self.region_decoder = {idx: region for region, idx in self.region_encoder.items()}

        # Строим граф
        adjacency, edge_weights = self._build_graph(df)

        # Агрегируем фичи по регионам
        feature_cols = [col for col in df.columns
                        if not col.startswith('target_') and
                        col not in ['snapshot_date', 'region']]

        region_features = []
        region_targets = []

        for region in regions:
            region_df = df[df['region'] == region]
            if not region_df.empty:
                # Используем несколько статистик: mean, std, min, max, last
                features = []
                for col in feature_cols:
                    if col in region_df.columns:
                        col_data = region_df[col].dropna()
                        if len(col_data) > 0:
                            features.extend([
                                col_data.mean(),
                                col_data.std(),
                                col_data.min(),
                                col_data.max(),
                                col_data.iloc[-1] if len(col_data) > 0 else 0
                            ])
                        else:
                            features.extend([0, 0, 0, 0, 0])

                region_features.append(features)

                # Таргеты (будущий спрос)
                targets = [
                    region_df['target_purchase_count'].mean() if 'target_purchase_count' in region_df.columns else 0,
                    region_df['target_total_spent'].mean() if 'target_total_spent' in region_df.columns else 0
                ]
                region_targets.append(targets)

        # Нормализация фич
        region_features = np.array(region_features)
        if not hasattr(self, 'feature_stats'):
            self.feature_stats = {
                'mean': region_features.mean(axis=0),
                'std': region_features.std(axis=0).clip(1e-6, None)
            }

        region_features = (region_features - self.feature_stats['mean']) / self.feature_stats['std']

        # Конвертируем в тензоры
        node_features = torch.FloatTensor(region_features).unsqueeze(0).to(self.device)
        adjacency = torch.FloatTensor(adjacency).unsqueeze(0).to(self.device)
        edge_weights = torch.FloatTensor(edge_weights).unsqueeze(0).to(self.device)
        targets = torch.FloatTensor(np.array(region_targets)).unsqueeze(0).to(self.device)

        return node_features, adjacency, edge_weights, targets

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
              epochs: int = 200, batch_size: int = 32, learning_rate: float = 1e-3):
        """Обучение GNN"""

        print("🌍 Training Real Cross-Region GNN...")

        # Подготавливаем данные
        node_features, adjacency, edge_weights, targets = self._prepare_graph_data(train_df)

        # Создаем модель
        self.model = GraphAttentionNetwork(
            node_features=node_features.shape[-1],
            hidden_dim=128,
            num_heads=8,
            num_layers=3,
            use_edge_features=True
        ).to(self.device)

        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )

        # Loss function (Huber loss для robustness)
        criterion = nn.HuberLoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            predictions = self.model(node_features, adjacency, edge_weights)

            # Loss
            loss = criterion(predictions, targets)

            # Add graph regularization
            graph_reg = torch.norm(adjacency - torch.eye(adjacency.shape[-1]).to(self.device))
            loss = loss + 0.01 * graph_reg

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Validation
            if val_df is not None and (epoch + 1) % 10 == 0:
                val_loss = self._validate(val_df)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

                if val_loss < getattr(self, 'best_val_loss', float('inf')):
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_region_model.pt')
            elif (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        print("✅ GNN training completed")

    def _validate(self, val_df):
        """Валидация GNN"""
        self.model.eval()

        node_features, adjacency, edge_weights, targets = self._prepare_graph_data(val_df)

        with torch.no_grad():
            predictions = self.model(node_features, adjacency, edge_weights)
            loss = nn.HuberLoss()(predictions, targets)

        return loss.item()

    def predict(self, df: pd.DataFrame):
        """Прогноз спроса по регионам"""
        self.model.eval()

        node_features, adjacency, edge_weights, _ = self._prepare_graph_data(df)

        with torch.no_grad():
            predictions = self.model(node_features, adjacency, edge_weights)

        # Извлекаем предсказания
        predictions = predictions.squeeze(0).cpu().numpy()

        # Форматируем результаты
        results = {}
        for idx, (demand_pred, revenue_pred) in enumerate(predictions):
            region = self.region_decoder.get(idx, f'Region_{idx}')

            # Анализ transfer opportunities
            transfer_ops = self._analyze_transfer_opportunities(idx, predictions, adjacency.squeeze(0).cpu().numpy())

            results[region] = {
                'predicted_demand': float(demand_pred),
                'predicted_revenue': float(revenue_pred),
                'demand_ci': [
                    float(demand_pred * 0.8),  # lower
                    float(demand_pred * 1.2)  # upper
                ],
                'transfer_opportunities': transfer_ops,
                'graph_centrality': self._calculate_centrality(idx, adjacency.squeeze(0).cpu().numpy())
            }

        return results

    def _analyze_transfer_opportunities(self, region_idx, predictions, adjacency):
        """Анализирует возможности трансфера спроса"""
        opportunities = []

        region_demand = predictions[region_idx, 0]
        all_demands = predictions[:, 0]

        # Находим регионы с избытком спроса
        demand_mean = np.mean(all_demands)
        excess_mask = all_demands > demand_mean * 1.2
        deficit_mask = all_demands < demand_mean * 0.8

        if deficit_mask[region_idx]:
            # Этот регион испытывает дефицит
            for other_idx in np.where(excess_mask)[0]:
                if adjacency[region_idx, other_idx] > 0:  # Есть связь в графе
                    excess_demand = all_demands[other_idx] - demand_mean
                    deficit = demand_mean - region_demand

                    transfer_amount = min(excess_demand, deficit)
                    if transfer_amount > 0:
                        opportunities.append({
                            'from_region': self.region_decoder.get(other_idx, f'Region_{other_idx}'),
                            'to_region': self.region_decoder.get(region_idx, f'Region_{region_idx}'),
                            'transfer_amount': float(transfer_amount),
                            'confidence': float(adjacency[region_idx, other_idx]),
                            'estimated_impact': float(transfer_amount * 0.3)  # 30% эффективность трансфера
                        })

        return opportunities

    def _calculate_centrality(self, region_idx, adjacency):
        """Вычисляет centrality меры для региона"""
        # Degree centrality
        degree = np.sum(adjacency[region_idx])

        # Eigenvector centrality (упрощенная)
        eigenvalues, eigenvectors = np.linalg.eig(adjacency)
        max_eigen_idx = np.argmax(eigenvalues.real)
        eigen_centrality = np.abs(eigenvectors[:, max_eigen_idx].real)

        # Betweenness centrality (упрощенная)
        n = adjacency.shape[0]
        betweenness = 0

        for i in range(n):
            for j in range(i + 1, n):
                if i != region_idx and j != region_idx:
                    # Проверяем лежит ли region_idx на кратчайшем пути
                    if adjacency[i, region_idx] > 0 and adjacency[region_idx, j] > 0:
                        betweenness += 1

        return {
            'degree_centrality': float(degree / (n - 1)),
            'eigenvector_centrality': float(eigen_centrality[region_idx]),
            'betweenness_centrality': float(betweenness / ((n - 1) * (n - 2) / 2)) if n > 2 else 0
        }


# ============ МОДЕЛЬ 3: MICRO-TREND ANTICIPATION ============

class MicroTrendTFT(pl.LightningModule):
    """Реальный Temporal Fusion Transformer для прогнозирования временных рядов"""

    def __init__(self,
                 hidden_size: int = 160,
                 lstm_layers: int = 2,
                 dropout: float = 0.1,
                 attention_head_size: int = 4,
                 learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()

        # LSTM для encoding прошлых значений
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        # LSTM для decoding будущих значений
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_head_size,
            dropout=dropout,
            batch_first=True
        )

        # Position-wise feedforward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        # Output layers для quantile regression
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(3)  # 0.1, 0.5, 0.9 quantiles
        ])

        # Change point detection
        self.change_point_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Learning rate
        self.learning_rate = learning_rate

    def forward(self, x_historic, x_future=None):
        # x_historic: [batch, seq_len_hist, features]
        # x_future: [batch, seq_len_future, features]

        batch_size = x_historic.shape[0]

        # 1. Encode historical sequence
        lstm_out_hist, (hidden, cell) = self.lstm_encoder(x_historic)

        if x_future is not None:
            # 2. Decode future sequence
            lstm_out_future, _ = self.lstm_decoder(x_future, (hidden, cell))

            # 3. Combine sequences for attention
            combined = torch.cat([lstm_out_hist, lstm_out_future], dim=1)
        else:
            combined = lstm_out_hist

        # 4. Self-attention
        attn_out, attn_weights = self.attention(combined, combined, combined)
        attn_out = self.ln1(combined + self.dropout(attn_out))

        # 5. Position-wise feedforward
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ln2(attn_out + self.dropout(ffn_out))

        # 6. Extract future predictions
        if x_future is not None:
            future_features = ffn_out[:, -x_future.shape[1]:, :]
        else:
            future_features = ffn_out[:, -1:, :]  # Predict one step ahead

        # 7. Quantile predictions
        quantile_preds = []
        for quantile_layer in self.quantile_outputs:
            pred = quantile_layer(future_features)
            quantile_preds.append(pred)

        # 8. Change point detection
        change_points = self.change_point_detector(ffn_out)

        return {
            'quantile_predictions': torch.stack(quantile_preds, dim=-1).squeeze(-2),
            'attention_weights': attn_weights,
            'change_points': change_points,
            'encoded_features': ffn_out
        }

    def training_step(self, batch, batch_idx):
        x_hist, x_fut, y = batch
        outputs = self(x_hist, x_fut)

        # Quantile loss
        quantiles = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        loss = 0
        for i, q in enumerate(quantiles):
            errors = y - outputs['quantile_predictions'][:, :, i]
            loss += torch.mean(torch.max((q - 1) * errors, q * errors))

        # Change point detection loss (unsupervised)
        # Используем вариацию временного ряда как pseudo-labels
        ts_variance = torch.var(x_hist, dim=1, keepdim=True)
        change_labels = (ts_variance > torch.median(ts_variance)).float()
        change_loss = F.binary_cross_entropy(
            outputs['change_points'][:, -x_hist.shape[1]:, :].squeeze(-1),
            change_labels.squeeze(-1)
        )

        total_loss = loss + 0.1 * change_loss

        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x_hist, x_fut, y = batch
        outputs = self(x_hist, x_fut)

        # MAE на медиане
        mae = F.l1_loss(outputs['quantile_predictions'][:, :, 1], y)
        self.log('val_mae', mae, prog_bar=True)
        return mae

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_mae'
        }


class MicroTrendModel:
    """Реальная TFT модель для прогнозирования микро-трендов"""

    def __init__(self,
                 prediction_horizon: int = 7,
                 context_length: int = 14,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.prediction_horizon = prediction_horizon
        self.context_length = context_length
        self.model = None
        self.scaler = None
        self.trend_encoder = None

    def _create_sequences(self, df: pd.DataFrame):
        """Создает последовательности для TFT"""

        sequences = []
        targets = []
        change_points = []

        # Группируем по trend_id
        for trend_id in df['trend_id'].unique():
            trend_df = df[df['trend_id'] == trend_id].sort_values('snapshot_date')

            if len(trend_df) < self.context_length + self.prediction_horizon:
                continue

            # Извлекаем фичи
            feature_cols = [col for col in trend_df.columns
                            if not col.startswith('target_') and
                            col not in ['snapshot_date', 'trend_id', 'trend_type']]

            features = trend_df[feature_cols].values.astype(np.float32)

            # Извлекаем таргеты
            target_cols = [col for col in trend_df.columns if col.startswith('target_')]
            if target_cols:
                trend_targets = trend_df[target_cols].values.astype(np.float32)
            else:
                # Если нет таргетов, используем последнюю фичу
                trend_targets = features[:, -1:]

            # Нормализация
            if self.scaler is None:
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = self.scaler.transform(features)

            # Создаем скользящие окна
            for i in range(len(features_scaled) - self.context_length - self.prediction_horizon + 1):
                # Исторический контекст
                hist_seq = features_scaled[i:i + self.context_length]

                # Будущие ковариаты (можно использовать исторические фичи как прокси)
                fut_seq = features_scaled[i + self.context_length:i + self.context_length + self.prediction_horizon]

                # Таргеты
                target_seq = trend_targets[i + self.context_length:i + self.context_length + self.prediction_horizon]

                sequences.append((hist_seq, fut_seq))
                targets.append(target_seq)

                # Детекция точек изменения (на основе дисперсии)
                if i > 0:
                    prev_window = features_scaled[i - 1:i + self.context_length - 1]
                    curr_window = hist_seq

                    # Простая детекция через сравнение статистик
                    prev_mean = np.mean(prev_window[:, 0])
                    curr_mean = np.mean(curr_window[:, 0])
                    change = abs(curr_mean - prev_mean) / (prev_mean + 1e-10)

                    change_points.append(1 if change > 0.3 else 0)
                else:
                    change_points.append(0)

        if len(sequences) == 0:
            return None, None, None

        return sequences, targets, change_points

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Обучение TFT модели"""

        print("📈 Training Real Micro-Trend TFT...")

        # Создаем последовательности
        sequences, targets, change_points = self._create_sequences(train_df)

        if sequences is None:
            print("⚠️ No sequences created for training")
            return

        # Подготовка DataLoader
        X_hist = torch.FloatTensor(np.array([s[0] for s in sequences])).to(self.device)
        X_fut = torch.FloatTensor(np.array([s[1] for s in sequences])).to(self.device)
        y = torch.FloatTensor(np.array(targets)).to(self.device)

        train_dataset = TensorDataset(X_hist, X_fut, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Создаем модель
        self.model = TemporalFusionTransformer(
            hidden_size=160,
            lstm_layers=2,
            dropout=0.1,
            attention_head_size=4,
            learning_rate=learning_rate
        ).to(self.device)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            callbacks=[
                EarlyStopping(monitor='val_mae', patience=10, mode='min'),
                ModelCheckpoint(monitor='val_mae', mode='min')
            ],
            enable_progress_bar=True,
            logger=True
        )

        # Validation если есть
        if val_df is not None:
            val_sequences, val_targets, _ = self._create_sequences(val_df)
            if val_sequences:
                X_hist_val = torch.FloatTensor(np.array([s[0] for s in val_sequences])).to(self.device)
                X_fut_val = torch.FloatTensor(np.array([s[1] for s in val_sequences])).to(self.device)
                y_val = torch.FloatTensor(np.array(val_targets)).to(self.device)

                val_dataset = TensorDataset(X_hist_val, X_fut_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                trainer.fit(self.model, train_loader, val_loader)
            else:
                trainer.fit(self.model, train_loader)
        else:
            trainer.fit(self.model, train_loader)

        print("✅ TFT training completed")

    def predict(self, trend_data: pd.DataFrame, forecast_days: int = 7):
        """Прогнозирование тренда"""

        self.model.eval()

        # Подготавливаем последовательность
        trend_data = trend_data.sort_values('snapshot_date')

        # Извлекаем фичи
        feature_cols = [col for col in trend_data.columns
                        if not col.startswith('target_') and
                        col not in ['snapshot_date', 'trend_id', 'trend_type']]

        features = trend_data[feature_cols].values.astype(np.float32)

        # Нормализация
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        # Берем последний контекст
        if len(features_scaled) < self.context_length:
            # Паддим если данных недостаточно
            padding = np.zeros((self.context_length - len(features_scaled), features_scaled.shape[1]))
            hist_seq = np.vstack([padding, features_scaled])
        else:
            hist_seq = features_scaled[-self.context_length:]

        # Создаем будущие ковариаты (используем последние значения)
        fut_seq = np.tile(hist_seq[-1:], (forecast_days, 1))

        # Конвертируем в тензоры
        hist_tensor = torch.FloatTensor(hist_seq).unsqueeze(0).to(self.device)
        fut_tensor = torch.FloatTensor(fut_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(hist_tensor, fut_tensor)

        # Извлекаем предсказания
        quantile_preds = outputs['quantile_predictions'].cpu().numpy()[0]  # [forecast_days, 3]
        change_points = outputs['change_points'].cpu().numpy()[0, -self.context_length:, 0]

        # Анализ тренда
        trend_analysis = self._analyze_trend(quantile_preds[:, 1])  # Используем медиану

        # Детекция точек изменения
        detected_changes = np.where(change_points > 0.7)[0]

        return {
            'trend_id': trend_data['trend_id'].iloc[0] if 'trend_id' in trend_data.columns else 'unknown',
            'forecast_quantiles': {
                'lower': quantile_preds[:, 0].tolist(),  # 0.1 quantile
                'median': quantile_preds[:, 1].tolist(),  # 0.5 quantile
                'upper': quantile_preds[:, 2].tolist()  # 0.9 quantile
            },
            'trend_strength': trend_analysis['strength'],
            'trend_direction': trend_analysis['direction'],
            'volatility': float(np.std(quantile_preds[:, 2] - quantile_preds[:, 0])),  # Spread volatility
            'change_points': detected_changes.tolist(),
            'change_probabilities': change_points.tolist(),
            'attention_pattern': outputs['attention_weights'].cpu().numpy()[0].tolist(),
            'growth_rate': trend_analysis['growth_rate'],
            'acceleration': trend_analysis['acceleration']
        }

    def _analyze_trend(self, forecast):
        """Анализирует тренд на основе прогноза"""

        if len(forecast) < 3:
            return {'strength': 0, 'direction': 'neutral', 'growth_rate': 0, 'acceleration': 0}

        # Linear regression для определения тренда
        x = np.arange(len(forecast))
        slope, intercept = np.polyfit(x, forecast, 1)

        # R-squared для силы тренда
        y_pred = slope * x + intercept
        ss_res = np.sum((forecast - y_pred) ** 2)
        ss_tot = np.sum((forecast - np.mean(forecast)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        # Ускорение (вторая производная)
        if len(forecast) >= 3:
            acceleration = forecast[-1] - 2 * forecast[-2] + forecast[-3]
        else:
            acceleration = 0

        # Определяем направление
        if slope > 0.01:
            direction = 'up'
        elif slope < -0.01:
            direction = 'down'
        else:
            direction = 'neutral'

        return {
            'strength': float(r_squared),
            'direction': direction,
            'growth_rate': float(slope),
            'acceleration': float(acceleration)
        }


# ============ МОДЕЛЬ 4: ADAPTIVE PRICING ============

class DemandTransformer(nn.Module):
    """Transformer для моделирования спроса с учетом цены"""

    def __init__(self,
                 feature_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # Price embedding (separate для нелинейности)
        self.price_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim - 1, hidden_dim),  # -1 т.к. цена отдельно
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,  # price + features
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Demand prediction head
        self.demand_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Спрос всегда положительный
        )

        # Price elasticity head
        self.elasticity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Эластичность обычно отрицательная
        )

        # Optimal price head
        self.optimal_price_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def forward(self, price, features):
        # price: [batch, 1]
        # features: [batch, feature_dim-1]

        # Embed price
        price_emb = self.price_embedding(price.unsqueeze(-1))

        # Embed features
        feature_emb = self.feature_embedding(features)

        # Concatenate
        combined = torch.cat([price_emb, feature_emb], dim=-1)

        # Add sequence dimension для transformer
        combined = combined.unsqueeze(1)

        # Transformer encoding
        encoded = self.transformer(combined)
        encoded = encoded.squeeze(1)

        # Heads
        demand = self.demand_head(encoded)
        elasticity = self.elasticity_head(encoded)
        optimal_price = self.optimal_price_head(encoded)
        uncertainty = self.uncertainty_head(encoded)

        return {
            'demand': demand,
            'elasticity': elasticity,
            'optimal_price': optimal_price,
            'uncertainty': uncertainty,
            'encoded': encoded
        }


class ReinforcementLearningPricer:
    """Глубокое обучение с подкреплением для ценообразования"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=0.001
        )

        # Memory
        self.memory = []
        self.gamma = 0.99

    def select_action(self, state, epsilon=0.1):
        """Выбор действия с exploration"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = F.softmax(self.policy_net(state_tensor), dim=-1)
            action = torch.multinomial(action_probs, 1).item()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Сохраняет переход в memory"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size=64):
        """Обучает policy и value networks"""
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Calculate returns
        with torch.no_grad():
            next_values = self.value_net(next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)

        # Value loss
        values = self.value_net(states)
        value_loss = F.mse_loss(values, target_values)

        # Policy loss (REINFORCE)
        action_probs = F.softmax(self.policy_net(states), dim=-1)
        selected_probs = action_probs.gather(1, actions)
        advantages = target_values - values.detach()
        policy_loss = -torch.mean(torch.log(selected_probs + 1e-10) * advantages)

        # Total loss
        total_loss = value_loss + policy_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            1.0
        )
        self.optimizer.step()

        return total_loss.item()


class AdaptivePricingModel:
    """Реальная модель адаптивного ценообразования с RL"""

    def __init__(self,
                 price_min: float = 0.5,
                 price_max: float = 2.0,
                 num_price_bins: int = 50,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.price_min = price_min
        self.price_max = price_max
        self.num_price_bins = num_price_bins
        self.price_bins = np.linspace(price_min, price_max, num_price_bins)

        self.demand_model = None
        self.rl_agent = None
        self.scaler = None
        self.price_stats = None

    def _prepare_features(self, df: pd.DataFrame, is_training=True):
        """Подготавливает фичи для модели спроса"""

        # Важные фичи для моделирования спроса
        feature_cols = [
            'current_price', 'conversion_rate', 'total_views', 'total_purchases',
            'category_avg_price', 'competition_pressure', 'inventory_level',
            'days_since_last_purchase', 'trend_growth_rate', 'price_elasticity'
        ]

        # Выбираем только существующие колонки
        available_cols = [col for col in feature_cols if col in df.columns]

        X = df[available_cols].copy()

        # Заполняем пропуски
        X = X.fillna(0)

        # Нормализация
        if is_training:
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = available_cols
        else:
            X_scaled = self.scaler.transform(X)

        # Отделяем цену от других фич
        if 'current_price' in available_cols:
            price_idx = available_cols.index('current_price')
            price = X_scaled[:, price_idx:price_idx + 1]
            features = np.delete(X_scaled, price_idx, axis=1)
        else:
            price = np.zeros((len(X_scaled), 1))
            features = X_scaled

        return price, features

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Обучение модели спроса и RL агента"""

        print("💰 Training Real Adaptive Pricing Model...")

        # Подготавливаем данные
        price_train, features_train = self._prepare_features(train_df, is_training=True)

        # Таргеты
        if 'target_sales_count' in train_df.columns:
            demand_target = train_df['target_sales_count'].values.astype(np.float32)
        else:
            # Если нет таргета, используем псевдо-таргет
            demand_target = np.ones(len(train_df))

        # Конвертируем в тензоры
        price_tensor = torch.FloatTensor(price_train).to(self.device)
        features_tensor = torch.FloatTensor(features_train).to(self.device)
        demand_tensor = torch.FloatTensor(demand_target).unsqueeze(1).to(self.device)

        # Создаем модель спроса
        self.demand_model = DemandTransformer(
            feature_dim=features_train.shape[1] + 1,  # +1 for price
            hidden_dim=128,
            num_heads=8,
            num_layers=4
        ).to(self.device)

        # Optimizer
        optimizer = optim.AdamW(self.demand_model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_df) // batch_size + 1
        )

        # Loss functions
        demand_criterion = nn.HuberLoss()

        # Training loop для модели спроса
        self.demand_model.train()
        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(len(price_tensor))

            total_loss = 0
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]

                batch_price = price_tensor[batch_indices]
                batch_features = features_tensor[batch_indices]
                batch_demand = demand_tensor[batch_indices]

                # Forward pass
                outputs = self.demand_model(batch_price, batch_features)

                # Demand loss
                loss = demand_criterion(outputs['demand'], batch_demand)

                # Elasticity regularization (должна быть отрицательной)
                elasticity_reg = F.relu(outputs['elasticity'] + 0.1).mean()  # Наказываем положительную эластичность
                loss += 0.1 * elasticity_reg

                # Price consistency
                price_consistency = torch.abs(outputs['optimal_price'] - batch_price).mean()
                loss += 0.05 * price_consistency

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.demand_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            # Validation
            if val_df is not None and (epoch + 1) % 10 == 0:
                val_loss = self._validate(val_df, demand_criterion)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
            elif (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        # Инициализируем RL агент
        state_dim = features_train.shape[1] + 3  # features + price + elasticity + uncertainty
        self.rl_agent = ReinforcementLearningPricer(state_dim, self.num_price_bins)

        # Обучаем RL агента на исторических данных
        self._train_rl_agent(train_df)

        print("✅ Adaptive Pricing Model training completed")

    def _validate(self, val_df, criterion):
        """Валидация модели спроса"""
        self.demand_model.eval()

        price_val, features_val = self._prepare_features(val_df, is_training=False)

        if 'target_sales_count' in val_df.columns:
            demand_target = val_df['target_sales_count'].values.astype(np.float32)
        else:
            return 0

        price_tensor = torch.FloatTensor(price_val).to(self.device)
        features_tensor = torch.FloatTensor(features_val).to(self.device)
        demand_tensor = torch.FloatTensor(demand_target).unsqueeze(1).to(self.device)

        with torch.no_grad():
            outputs = self.demand_model(price_tensor, features_tensor)
            loss = criterion(outputs['demand'], demand_tensor)

        return loss.item()

    def _train_rl_agent(self, df: pd.DataFrame, num_episodes: int = 1000):
        """Обучает RL агента на исторических данных"""

        print("🤖 Training RL Pricing Agent...")

        # Подготавливаем фичи
        price, features = self._prepare_features(df, is_training=False)

        # Создаем эпизоды для RL
        episodes = []
        for i in range(len(df) - 1):
            current_state = self._create_rl_state(
                price[i],
                features[i],
                df.iloc[i] if 'price_elasticity' in df.columns else 0,
                0.1  # начальная uncertainty
            )

            # Действие = текущая цена (дискретизированная)
            current_price = df.iloc[i]['current_price'] if 'current_price' in df.columns else price[i]
            action = self._discretize_price(current_price)

            # Награда = прибыль на следующем шаге
            if i + 1 < len(df):
                next_price = df.iloc[i + 1]['current_price'] if 'current_price' in df.columns else price[i + 1]
                next_demand = df.iloc[i + 1]['target_sales_count'] if 'target_sales_count' in df.columns else 1
                reward = next_price * next_demand
            else:
                reward = 0

            # Следующее состояние
            next_state = self._create_rl_state(
                price[i + 1] if i + 1 < len(df) else price[i],
                features[i + 1] if i + 1 < len(df) else features[i],
                df.iloc[i + 1]['price_elasticity'] if i + 1 < len(df) and 'price_elasticity' in df.columns else 0,
                0.1
            )

            done = (i == len(df) - 2)

            episodes.append((current_state, action, reward, next_state, done))

        # Обучение RL агента
        for episode in range(num_episodes):
            total_reward = 0
            for state, action, reward, next_state, done in episodes:
                self.rl_agent.store_transition(state, action, reward, next_state, done)
                total_reward += reward

            # Обучение на батче
            loss = self.rl_agent.train_step()

            if (episode + 1) % 100 == 0:
                print(f"RL Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    def _create_rl_state(self, price, features, elasticity, uncertainty):
        """Создает состояние для RL агента"""
        state = np.concatenate([
            features.flatten(),
            [price, elasticity, uncertainty]
        ])
        return state

    def _discretize_price(self, price: float) -> int:
        """Дискретизирует цену для RL"""
        price_normalized = (price - self.price_min) / (self.price_max - self.price_min)
        price_bin = int(price_normalized * (self.num_price_bins - 1))
        return np.clip(price_bin, 0, self.num_price_bins - 1)

    def _undiscretize_price(self, price_bin: int, reference_price: float) -> float:
        """Обратно преобразует дискретизированную цену"""
        price_normalized = price_bin / (self.num_price_bins - 1)
        price_range = reference_price * (self.price_max - self.price_min)
        price = reference_price * self.price_min + price_normalized * price_range
        return price

    def recommend_price(self, item_data: pd.DataFrame,
                        market_context: Dict = None,
                        use_rl: bool = True,
                        risk_aversion: float = 0.5) -> Dict:
        """Рекомендует оптимальную цену"""

        self.demand_model.eval()

        # Подготавливаем фичи
        price, features = self._prepare_features(item_data, is_training=False)

        # Конвертируем в тензоры
        price_tensor = torch.FloatTensor(price).to(self.device)
        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            outputs = self.demand_model(price_tensor, features_tensor)

        # Базовые предсказания
        current_price = price[0, 0]
        predicted_demand = outputs['demand'].item()
        predicted_elasticity = outputs['elasticity'].item()
        optimal_price_nn = outputs['optimal_price'].item()
        uncertainty = outputs['uncertainty'].item()

        # RL коррекция если включено
        if use_rl and self.rl_agent is not None:
            # Создаем состояние для RL
            rl_state = self._create_rl_state(
                current_price,
                features[0],
                predicted_elasticity,
                uncertainty
            )

            # Выбор действия RL агентом
            rl_action = self.rl_agent.select_action(rl_state, epsilon=0.1)

            # Преобразуем действие в цену
            rl_price = self._undiscretize_price(rl_action, current_price)

            # Блендим RL и NN предсказания
            blend_weight = 0.3  # 30% RL, 70% NN
            final_price = (1 - blend_weight) * optimal_price_nn + blend_weight * rl_price
        else:
            final_price = optimal_price_nn

        # Учет контекста рынка
        if market_context:
            # Конкуренция
            if 'competition_price' in market_context:
                competition_weight = market_context.get('competition_weight', 0.3)
                final_price = final_price * (1 - competition_weight) + \
                              market_context['competition_price'] * competition_weight

            # Сегмент рынка
            if 'market_segment' in market_context:
                if market_context['market_segment'] == 'premium':
                    final_price *= 1.2
                elif market_context['market_segment'] == 'budget':
                    final_price *= 0.8

            # Спрос и предложение
            if 'supply_demand_ratio' in market_context:
                ratio = market_context['supply_demand_ratio']
                if ratio < 0.8:  # Дефицит
                    final_price *= 1.1
                elif ratio > 1.2:  # Избыток
                    final_price *= 0.9

        # Учет aversion к риску
        if risk_aversion > 0:
            # Более консервативная цена при высоком aversion
            risk_adjustment = 1 - 0.2 * risk_aversion
            final_price = current_price * risk_adjustment + final_price * (1 - risk_adjustment)

        # Ограничения
        min_price = current_price * 0.5
        max_price = current_price * 2.0
        final_price = np.clip(final_price, min_price, max_price)

        # Пересчет спроса с новой ценой
        price_change = (final_price - current_price) / (current_price + 1e-10)
        elasticity_effect = 1 + predicted_elasticity * price_change
        adjusted_demand = predicted_demand * elasticity_effect

        # Анализ чувствительности
        sensitivity = self._price_sensitivity_analysis(
            current_price, final_price, predicted_elasticity, predicted_demand
        )

        return {
            'item_id': item_data['item_id'].iloc[0] if 'item_id' in item_data.columns else 'unknown',
            'current_price': float(current_price),
            'recommended_price': float(final_price),
            'price_change_percent': float(price_change * 100),
            'predicted_demand': float(adjusted_demand),
            'predicted_revenue': float(final_price * adjusted_demand),
            'price_elasticity': float(predicted_elasticity),
            'uncertainty': float(uncertainty),
            'confidence': float(1 / (1 + uncertainty)),  # confidence = 1/(1+uncertainty)
            'method': 'transformer_rl' if use_rl else 'transformer_only',
            'sensitivity_analysis': sensitivity,
            'risk_adjusted': risk_aversion > 0
        }

    def _price_sensitivity_analysis(self, current_price, recommended_price,
                                    elasticity, base_demand):
        """Анализ чувствительности к цене"""

        price_points = np.linspace(current_price * 0.5, current_price * 2.0, 20)
        demands = []
        revenues = []
        profits = []

        for price in price_points:
            price_change = (price - current_price) / current_price
            demand = base_demand * (1 + elasticity * price_change)
            revenue = price * demand
            profit = revenue - price * demand * 0.6  # 40% margin

            demands.append(demand)
            revenues.append(revenue)
            profits.append(profit)

        # Находим оптимальные точки
        optimal_revenue_idx = np.argmax(revenues)
        optimal_profit_idx = np.argmax(profits)

        # Эластичность в точке
        point_elasticity = elasticity * (current_price / (base_demand + 1e-10))

        return {
            'price_points': price_points.tolist(),
            'demand_curve': demands,
            'revenue_curve': revenues,
            'profit_curve': profits,
            'optimal_revenue_price': float(price_points[optimal_revenue_idx]),
            'optimal_profit_price': float(price_points[optimal_profit_idx]),
            'current_elasticity': float(point_elasticity),
            'revenue_optimal': optimal_revenue_idx == np.argmin(np.abs(price_points - recommended_price)),
            'profit_optimal': optimal_profit_idx == np.argmin(np.abs(price_points - recommended_price))
        }


# ============ ПРОДАКШЕН ПАЙПЛАЙН ============

class ProductionPipeline:
    """Продакшен пайплайн для всех моделей"""

    def __init__(self, models_dir: str = "./production_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.models = {
            'context_aware': ContextAwareModel(),
            'cross_region': CrossRegionModel(),
            'micro_trend': MicroTrendModel(),
            'adaptive_pricing': AdaptivePricingModel()
        }

        self.is_trained = {name: False for name in self.models}
        self.metadata = {}

    def train_all_models(self, snapshots_dir: str = "../analytics/data/innovative_snapshots"):
        """Обучение всех моделей в production режиме"""

        print("=" * 80)
        print("🚀 PRODUCTION TRAINING OF ALL 4 REAL MODELS")
        print("=" * 80)

        snapshots_path = Path(snapshots_dir)

        # Model 1: Context-Aware
        print("\n" + "=" * 40)
        print("1️⃣ REAL Context-Aware Purchase Prediction")
        print("=" * 40)
        try:
            train_df = pd.read_parquet(snapshots_path / "model1/train.parquet")
            val_df = pd.read_parquet(snapshots_path / "model1/val.parquet")

            # Фильтруем для тестирования (можно убрать)
            train_df = train_df.head(10000)
            val_df = val_df.head(2000)

            self.models['context_aware'].train(train_df, val_df, epochs=50)

            # Сохраняем модель
            torch.save({
                'model_state': self.models['context_aware'].model.state_dict(),
                'feature_config': self.models['context_aware'].feature_config,
                'scalers': self.models['context_aware'].scalers,
                'encoders': self.models['context_aware'].encoders,
                'category_encoder': getattr(self.models['context_aware'], 'category_encoder', None),
                'calibrator': self.models['context_aware'].calibrator
            }, self.models_dir / "context_model.pt")

            self.is_trained['context_aware'] = True
            print("   ✅ Model trained and saved")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

        # Model 2: Cross-Region
        print("\n" + "=" * 40)
        print("2️⃣ REAL Cross-Region Demand Transfer")
        print("=" * 40)
        try:
            train_df = pd.read_parquet(snapshots_path / "model2/train.parquet")
            val_df = pd.read_parquet(snapshots_path / "model2/val.parquet")

            train_df = train_df.head(5000)
            val_df = val_df.head(1000)

            self.models['cross_region'].train(train_df, val_df, epochs=100)

            torch.save({
                'model_state': self.models['cross_region'].model.state_dict(),
                'region_encoder': self.models['cross_region'].region_encoder,
                'feature_stats': getattr(self.models['cross_region'], 'feature_stats', None)
            }, self.models_dir / "region_model.pt")

            self.is_trained['cross_region'] = True
            print("   ✅ Model trained and saved")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

        # Model 3: Micro-Trend
        print("\n" + "=" * 40)
        print("3️⃣ REAL Micro-Trend Anticipation")
        print("=" * 40)
        try:
            train_df = pd.read_parquet(snapshots_path / "model3/train.parquet")
            val_df = pd.read_parquet(snapshots_path / "model3/val.parquet")

            train_df = train_df.head(10000)
            val_df = val_df.head(2000)

            self.models['micro_trend'].train(train_df, val_df, epochs=50)

            torch.save({
                'model_state': self.models['micro_trend'].model.state_dict(),
                'scaler': self.models['micro_trend'].scaler
            }, self.models_dir / "trend_model.pt")

            self.is_trained['micro_trend'] = True
            print("   ✅ Model trained and saved")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

        # Model 4: Adaptive Pricing
        print("\n" + "=" * 40)
        print("4️⃣ REAL Adaptive Pricing with RL")
        print("=" * 40)
        try:
            train_df = pd.read_parquet(snapshots_path / "model4/train.parquet")
            val_df = pd.read_parquet(snapshots_path / "model4/val.parquet")

            train_df = train_df.head(8000)
            val_df = val_df.head(1500)

            self.models['adaptive_pricing'].train(train_df, val_df, epochs=80)

            # Сохраняем модель спроса
            torch.save({
                'demand_model_state': self.models['adaptive_pricing'].demand_model.state_dict(),
                'scaler': self.models['adaptive_pricing'].scaler,
                'feature_names': getattr(self.models['adaptive_pricing'], 'feature_names', None),
                'price_bins': self.models['adaptive_pricing'].price_bins
            }, self.models_dir / "pricing_model.pt")

            # Сохраняем RL агента
            if self.models['adaptive_pricing'].rl_agent:
                torch.save(
                    self.models['adaptive_pricing'].rl_agent.policy_net.state_dict(),
                    self.models_dir / "rl_policy.pt"
                )

            self.is_trained['adaptive_pricing'] = True
            print("   ✅ Model trained and saved")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

        # Сохраняем метаданные
        self.metadata = {
            'trained_models': [name for name, trained in self.is_trained.items() if trained],
            'training_date': datetime.now().isoformat(),
            'model_versions': {
                'context_aware': '2.0',
                'cross_region': '2.0',
                'micro_trend': '2.0',
                'adaptive_pricing': '2.0'
            },
            'training_stats': {
                'context_aware': {
                    'samples': len(train_df) if 'context_aware' in locals() else 0
                }
            }
        }

        with open(self.models_dir / "production_metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print("\n" + "=" * 80)
        trained_count = sum(self.is_trained.values())
        print(f"✅ {trained_count}/4 REAL MODELS TRAINED FOR PRODUCTION!")
        print("=" * 80)

    def load_all_models(self):
        """Загрузка всех моделей"""
        print("🔄 Loading production models...")

        # Model 1
        try:
            checkpoint = torch.load(self.models_dir / "context_model.pt",
                                    map_location=self.models['context_aware'].device)

            # Recreate model
            self.models['context_aware'].feature_config = checkpoint['feature_config']
            self.models['context_aware'].model = MultiModalTransformer(
                num_numerical_features=len(checkpoint['feature_config']['numerical_features']),
                categorical_dims=checkpoint['feature_config']['categorical_features']
            ).to(self.models['context_aware'].device)

            self.models['context_aware'].model.load_state_dict(checkpoint['model_state'])
            self.models['context_aware'].scalers = checkpoint['scalers']
            self.models['context_aware'].encoders = checkpoint['encoders']
            self.models['context_aware'].category_encoder = checkpoint['category_encoder']
            self.models['context_aware'].calibrator = checkpoint['calibrator']

            self.is_trained['context_aware'] = True
            print("  ✅ Context-Aware: Loaded")
        except Exception as e:
            print(f"  ❌ Context-Aware: {e}")

        # Model 2
        try:
            checkpoint = torch.load(self.models_dir / "region_model.pt",
                                    map_location=self.models['cross_region'].device)

            # Recreate model
            node_features = len(checkpoint['feature_stats']['mean']) if checkpoint['feature_stats'] else 50
            self.models['cross_region'].model = GraphAttentionNetwork(node_features=node_features)
            self.models['cross_region'].model.load_state_dict(checkpoint['model_state'])
            self.models['cross_region'].region_encoder = checkpoint['region_encoder']

            self.is_trained['cross_region'] = True
            print("  ✅ Cross-Region: Loaded")
        except Exception as e:
            print(f"  ❌ Cross-Region: {e}")

        # Model 3
        try:
            checkpoint = torch.load(self.models_dir / "trend_model.pt",
                                    map_location=self.models['micro_trend'].device)

            self.models['micro_trend'].model = MicroTrendModel()
            self.models['micro_trend'].model.load_state_dict(checkpoint['model_state'])
            self.models['micro_trend'].scaler = checkpoint['scaler']

            self.is_trained['micro_trend'] = True
            print("  ✅ Micro-Trend: Loaded")
        except Exception as e:
            print(f"  ❌ Micro-Trend: {e}")

        # Model 4
        try:
            checkpoint = torch.load(self.models_dir / "pricing_model.pt",
                                    map_location=self.models['adaptive_pricing'].device)

            # Recreate model
            feature_dim = len(checkpoint['feature_names']) if checkpoint['feature_names'] else 10
            self.models['adaptive_pricing'].demand_model = DemandTransformer(feature_dim=feature_dim)
            self.models['adaptive_pricing'].demand_model.load_state_dict(checkpoint['demand_model_state'])
            self.models['adaptive_pricing'].scaler = checkpoint['scaler']

            # Load RL agent
            rl_path = self.models_dir / "rl_policy.pt"
            if rl_path.exists():
                state_dim = feature_dim + 3
                self.models['adaptive_pricing'].rl_agent = ReinforcementLearningPricer(
                    state_dim,
                    len(checkpoint['price_bins'])
                )
                self.models['adaptive_pricing'].rl_agent.policy_net.load_state_dict(
                    torch.load(rl_path)
                )

            self.is_trained['adaptive_pricing'] = True
            print("  ✅ Adaptive Pricing: Loaded")
        except Exception as e:
            print(f"  ❌ Adaptive Pricing: {e}")

    def get_api_predictions(self, request_data: Dict) -> Dict:
        """API endpoint для всех предсказаний"""

        results = {
            'context_aware': None,
            'cross_region': None,
            'micro_trend': None,
            'adaptive_pricing': None,
            'metadata': self.metadata
        }

        # Model 1: User prediction
        if 'user_data' in request_data and self.is_trained['context_aware']:
            try:
                user_df = pd.DataFrame([request_data['user_data']])
                results['context_aware'] = self.models['context_aware'].predict(user_df)
            except Exception as e:
                results['context_aware'] = {'error': str(e)}

        # Model 2: Region prediction
        if 'region_data' in request_data and self.is_trained['cross_region']:
            try:
                region_df = pd.DataFrame([request_data['region_data']])
                results['cross_region'] = self.models['cross_region'].predict(region_df)
            except Exception as e:
                results['cross_region'] = {'error': str(e)}

        # Model 3: Trend prediction
        if 'trend_data' in request_data and self.is_trained['micro_trend']:
            try:
                trend_df = pd.DataFrame(request_data['trend_data'])
                results['micro_trend'] = self.models['micro_trend'].predict(trend_df)
            except Exception as e:
                results['micro_trend'] = {'error': str(e)}

        # Model 4: Price recommendation
        if 'item_data' in request_data and self.is_trained['adaptive_pricing']:
            try:
                item_df = pd.DataFrame([request_data['item_data']])
                context = request_data.get('market_context', {})
                results['adaptive_pricing'] = self.models['adaptive_pricing'].recommend_price(
                    item_df, context
                )
            except Exception as e:
                results['adaptive_pricing'] = {'error': str(e)}

        return results


# ============ FASTAPI СЕРВЕР ============

def create_fastapi_app(pipeline: ProductionPipeline):
    """Создает FastAPI приложение"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Production AI Models API")

    # Pydantic models
    class UserData(BaseModel):
        user_id: str
        features: Dict[str, Any]

    class RegionData(BaseModel):
        region: str
        features: Dict[str, Any]

    class TrendData(BaseModel):
        trend_id: str
        snapshot_date: str
        features: Dict[str, Any]

    class ItemData(BaseModel):
        item_id: str
        current_price: float
        features: Dict[str, Any]

    class MarketContext(BaseModel):
        competition_price: Optional[float] = None
        competition_weight: Optional[float] = 0.3
        market_segment: Optional[str] = "standard"
        supply_demand_ratio: Optional[float] = 1.0

    class PredictionRequest(BaseModel):
        user_data: Optional[UserData] = None
        region_data: Optional[RegionData] = None
        trend_data: Optional[List[TrendData]] = None
        item_data: Optional[ItemData] = None
        market_context: Optional[MarketContext] = None

    @app.get("/")
    async def root():
        return {
            "message": "Production AI Models API",
            "models_ready": pipeline.is_trained,
            "versions": pipeline.metadata.get('model_versions', {})
        }

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "models_loaded": sum(pipeline.is_trained.values()),
            "timestamp": datetime.now().isoformat()
        }

    @app.post("/predict/all")
    async def predict_all(request: PredictionRequest):
        """Единый endpoint для всех предсказаний"""
        request_dict = request.dict()
        results = pipeline.get_api_predictions(request_dict)

        return {
            "predictions": results,
            "timestamp": datetime.now().isoformat(),
            "request_id": f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

    @app.post("/predict/user")
    async def predict_user(user_data: UserData):
        """Прогноз для пользователя"""
        if not pipeline.is_trained['context_aware']:
            raise HTTPException(status_code=503, detail="Context-Aware model not loaded")

        try:
            user_df = pd.DataFrame([user_data.features])
            user_df['user_id'] = user_data.user_id

            prediction = pipeline.models['context_aware'].predict(user_df)

            return {
                "user_id": user_data.user_id,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/region")
    async def predict_region(region_data: RegionData):
        """Прогноз спроса для региона"""
        if not pipeline.is_trained['cross_region']:
            raise HTTPException(status_code=503, detail="Cross-Region model not loaded")

        try:
            region_df = pd.DataFrame([region_data.features])
            region_df['region'] = region_data.region

            prediction = pipeline.models['cross_region'].predict(region_df)

            return {
                "region": region_data.region,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/trend")
    async def predict_trend(trend_data: List[TrendData]):
        """Прогноз тренда"""
        if not pipeline.is_trained['micro_trend']:
            raise HTTPException(status_code=503, detail="Micro-Trend model not loaded")

        try:
            trend_df = pd.DataFrame([td.dict() for td in trend_data])

            prediction = pipeline.models['micro_trend'].predict(trend_df)

            return {
                "trend_id": prediction['trend_id'],
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/optimize/price")
    async def optimize_price(item_data: ItemData, market_context: MarketContext = None):
        """Оптимизация цены"""
        if not pipeline.is_trained['adaptive_pricing']:
            raise HTTPException(status_code=503, detail="Adaptive Pricing model not loaded")

        try:
            item_df = pd.DataFrame([item_data.features])
            item_df['item_id'] = item_data.item_id
            item_df['current_price'] = item_data.current_price

            context = market_context.dict() if market_context else {}

            recommendation = pipeline.models['adaptive_pricing'].recommend_price(
                item_df, context
            )

            return {
                "item_id": item_data.item_id,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ============ ЗАПУСК ============

if __name__ == "__main__":
    print("=" * 80)
    print("🏭 PRODUCTION AI MODELS - READY FOR DEPLOYMENT")
    print("=" * 80)

    # Инициализируем пайплайн
    pipeline = ProductionPipeline()

    # Пытаемся загрузить существующие модели
    pipeline.load_all_models()

    # Если не все модели загружены, обучаем
    if not all(pipeline.is_trained.values()):
        print("\n⚠️ Some models not loaded. Starting training...")
        pipeline.train_all_models()

    # Тестовые предсказания
    print("\n🧪 TEST PREDICTIONS:")

    # Test Model 1
    if pipeline.is_trained['context_aware']:
        try:
            test_features = {
                'total_events': 100,
                'days_since_last': 5,
                'events_per_day': 2.5,
                'total_purchases': 10,
                'total_spent': 1000
            }

            test_df = pd.DataFrame([test_features])
            prediction = pipeline.models['context_aware'].predict(test_df)

            print(f"\n✅ Context-Aware Test:")
            print(f"   Purchase probability: {prediction['purchase_probability'][0]:.1%}")
            if 'predicted_category' in prediction:
                print(f"   Predicted category: {prediction['predicted_category'][0]}")
        except Exception as e:
            print(f"❌ Context-Aware test failed: {e}")

    # Test Model 4
    if pipeline.is_trained['adaptive_pricing']:
        try:
            test_features = {
                'current_price': 100,
                'conversion_rate': 0.05,
                'total_views': 1000,
                'total_purchases': 50,
                'price_elasticity': -1.5
            }

            test_df = pd.DataFrame([test_features])
            recommendation = pipeline.models['adaptive_pricing'].recommend_price(test_df)

            print(f"\n✅ Adaptive Pricing Test:")
            print(f"   Current: ${recommendation['current_price']:.2f}")
            print(f"   Recommended: ${recommendation['recommended_price']:.2f}")
            print(f"   Change: {recommendation['price_change_percent']:.1f}%")
            print(f"   Confidence: {recommendation['confidence']:.1%}")
        except Exception as e:
            print(f"❌ Adaptive Pricing test failed: {e}")

    print("\n" + "=" * 80)
    print("🚀 PRODUCTION MODELS READY FOR API DEPLOYMENT")
    print("=" * 80)

    # Сохраняем конфигурацию для деплоя
    deploy_config = {
        'models_loaded': pipeline.is_trained,
        'api_endpoints': [
            {'path': '/predict/all', 'method': 'POST', 'description': 'All predictions'},
            {'path': '/predict/user', 'method': 'POST', 'description': 'User predictions'},
            {'path': '/predict/region', 'method': 'POST', 'description': 'Region predictions'},
            {'path': '/predict/trend', 'method': 'POST', 'description': 'Trend predictions'},
            {'path': '/optimize/price', 'method': 'POST', 'description': 'Price optimization'}
        ],
        'deployment_instructions': [
            '1. Install requirements: pip install fastapi uvicorn',
            '2. Run server: uvicorn innovative_models_pro:create_fastapi_app --host 0.0.0.0 --port 8000',
            '3. Test endpoint: curl -X POST http://localhost:8000/predict/all -H "Content-Type: application/json" -d \'{"user_data": {...}}\''
        ],
        'timestamp': datetime.now().isoformat()
    }

    with open("deploy_config.json", "w") as f:
        json.dump(deploy_config, f, indent=2)

    print("\n📁 Deployment config saved to deploy_config.json")
    print("\n🎯 TO DEPLOY THE API SERVER:")
    print("   uvicorn innovative_models_pro:create_fastapi_app --reload --host 0.0.0.0 --port 8000")
