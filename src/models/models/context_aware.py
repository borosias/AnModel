# snapshot_model1.py
import os
import joblib
import numpy as np
import pandas as pd

from typing import Dict, Optional, Tuple

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
)

class ContextAwareModel:
    """
    Модель для снапшотов из snapshot_builder1:
    - will_purchase_next_7d (классификация)
    - days_to_next_purchase (регрессия)
    - next_purchase_amount (регрессия)
    """

    def __init__(
        self,
        random_state: int = 42,
    ):
        self.random_state = random_state

        # Модели
        self.clf = None
        self.reg_days = None
        self.reg_amount = None

        # Список колонок после препроцессинга (для выравнивания при predict)
        self.feature_columns_: Optional[pd.Index] = None

    # ========= ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =========

    def _split_features_targets(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Разделяет DataFrame на:
        - X: фичи
        - y_clf: таргет для классификации (will_purchase_next_7d)
        - y_days: регрессия (days_to_next_purchase)
        - y_amount: регрессия (next_purchase_amount)
        """

        target_cols = [
            "will_purchase_next_7d",
            "days_to_next_purchase",
            "next_purchase_amount",
        ]

        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns in data: {missing}")

        y_clf = df["will_purchase_next_7d"].astype(int)
        y_days = df["days_to_next_purchase"].astype(float)
        y_amount = df["next_purchase_amount"].astype(float)

        # Исключаем таргеты и служебные колонки из фичей
        drop_cols = set(target_cols) | {
            "snapshot_date",
            "user_id",
            "last_ts",
        }

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return X, y_clf, y_days, y_amount

    def _prepare_features_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Препроцессинг фичей при обучении:
        - числовые колонки оставляем как есть (NaN → 0)
        - категориальные → one-hot через get_dummies
        - сохраняем порядок колонок
        """

        # Быстрый one‑hot по всем object/categorical
        X_processed = pd.get_dummies(X, dummy_na=True)

        # NaN в числах заменяем нулями
        X_processed = X_processed.fillna(0.0)

        self.feature_columns_ = X_processed.columns

        return X_processed

    def _prepare_features_infer(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Препроцессинг фичей при предсказании:
        - те же шаги, что и при обучении
        - приводим к тому же набору колонок self.feature_columns_
        """

        if self.feature_columns_ is None:
            raise ValueError("Model is not fitted: feature_columns_ is None")

        X_processed = pd.get_dummies(X, dummy_na=True)
        X_processed = X_processed.fillna(0.0)

        # Добавляем/убираем колонки и упорядочиваем за один проход, чтобы избежать
        # фрагментации DataFrame (PerformanceWarning при множественных insert)
        X_processed = X_processed.reindex(columns=self.feature_columns_, fill_value=0.0)

        return X_processed

    # ========= ОБУЧЕНИЕ =========

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Обучает три модели:
        - классификатор will_purchase_next_7d
        - регрессор days_to_next_purchase
        - регрессор next_purchase_amount

        Если val_df задан, сразу считает метрики на валидации.
        """

        # 1. Разделяем фичи и таргеты
        X_train_raw, y_clf_train, y_days_train, y_amount_train = self._split_features_targets(
            train_df
        )

        # 2. Препроцессинг фичей
        X_train = self._prepare_features_fit(X_train_raw)

        # 3. Модели
        self.clf = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=self.random_state,
        )

        self.reg_days = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=self.random_state,
        )

        self.reg_amount = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=self.random_state,
        )

        # 4. Обучение
        self.clf.fit(X_train, y_clf_train)

        # Для регрессии иногда разумно обучать только на тех, у кого была покупка
        mask_pos = y_clf_train == 1
        if mask_pos.sum() > 0:
            self.reg_days.fit(X_train[mask_pos], y_days_train[mask_pos])
            self.reg_amount.fit(X_train[mask_pos], y_amount_train[mask_pos])
        else:
            # fallback: обучаем на всех, чтобы модель не падала
            self.reg_days.fit(X_train, y_days_train)
            self.reg_amount.fit(X_train, y_amount_train)

        # 5. Метрики на валидации (если есть)
        metrics = {}
        if val_df is not None and not val_df.empty:
            metrics = self.evaluate(val_df)

        return metrics

    # ========= ПРЕДСКАЗАНИЕ =========

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает DataFrame с предсказаниями:
        - purchase_proba
        - will_purchase_pred (0/1)
        - days_to_next_pred
        - amount_pred
        """

        if any(m is None for m in [self.clf, self.reg_days, self.reg_amount]):
            raise ValueError("Model is not fitted. Call fit() first.")

        # Для удобства: можно передавать как полный снапшот, так и только фичи
        # Если таргеты есть — просто игнорируем их.
        X_raw = df.copy()
        for col in ["will_purchase_next_7d", "days_to_next_purchase", "next_purchase_amount"]:
            if col in X_raw.columns:
                X_raw = X_raw.drop(columns=[col])

        # Препроцессинг
        X = self._prepare_features_infer(X_raw)

        # Классификация
        proba = self.clf.predict_proba(X)[:, 1]
        will_purchase_pred = (proba >= 0.5).astype(int)

        # Регрессия
        days_pred = self.reg_days.predict(X)
        amount_pred = self.reg_amount.predict(X)

        # Маскируем регрессионные предсказания для тех, у кого модель считает, что покупки не будет
        days_pred = np.where(will_purchase_pred == 1, days_pred, 999.0)
        amount_pred = np.where(will_purchase_pred == 1, amount_pred, 0.0)

        result = pd.DataFrame(
            {
                "purchase_proba": proba,
                "will_purchase_pred": will_purchase_pred,
                "days_to_next_pred": days_pred,
                "next_purchase_amount_pred": amount_pred,
            },
            index=df.index,
        )

        return result

    # ========= ОЦЕНКА =========

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Считает метрики на переданном DataFrame с таргетами:
        - AUC, PR-AUC для will_purchase_next_7d
        - RMSE / MAE для дней до покупки (только на тех, где была покупка)
        - RMSE / MAE для суммы покупки (только на тех, где была покупка)
        """

        X_raw, y_clf, y_days, y_amount = self._split_features_targets(df)
        X = self._prepare_features_infer(X_raw)

        proba = self.clf.predict_proba(X)[:, 1]

        metrics = {}

        # Классификация
        if len(np.unique(y_clf)) > 1:
            metrics["auc_roc"] = roc_auc_score(y_clf, proba)
            metrics["auc_pr"] = average_precision_score(y_clf, proba)
        else:
            metrics["auc_roc"] = float("nan")
            metrics["auc_pr"] = float("nan")

        # Регрессии — только на положительных примерах
        mask_pos = y_clf == 1
        if mask_pos.sum() > 0:
            days_pred = self.reg_days.predict(X[mask_pos])
            amount_pred = self.reg_amount.predict(X[mask_pos])

            metrics["rmse_days"] = mean_squared_error(y_days[mask_pos], days_pred)
            metrics["mae_days"] = mean_absolute_error(y_days[mask_pos], days_pred)

            metrics["rmse_amount"] = mean_squared_error(
                y_amount[mask_pos], amount_pred)
            metrics["mae_amount"] = mean_absolute_error(y_amount[mask_pos], amount_pred)
        else:
            metrics["rmse_days"] = float("nan")
            metrics["mae_days"] = float("nan")
            metrics["rmse_amount"] = float("nan")
            metrics["mae_amount"] = float("nan")

        return metrics

    # ========= СЕРИАЛИЗАЦИЯ =========

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "random_state": self.random_state,
            "clf": self.clf,
            "reg_days": self.reg_days,
            "reg_amount": self.reg_amount,
            "feature_columns_": self.feature_columns_,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "ContextAwareModel":
        state = joblib.load(path)
        model = cls(random_state=state.get("random_state", 42))
        model.clf = state["clf"]
        model.reg_days = state["reg_days"]
        model.reg_amount = state["reg_amount"]
        model.feature_columns_ = state["feature_columns_"]
        return model