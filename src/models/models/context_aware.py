"""
Улучшенная версия ContextAwareModel v2.0

Ключевые улучшения:
- LightGBM вместо sklearn GradientBoosting (быстрее, точнее)
- Автоматический подбор гиперпараметров через Optuna
- Умный препроцессинг с обработкой выбросов и правильным заполнением NaN
- Калибровка вероятностей
- Cross-validation для надежной оценки
- Feature importance анализ
- Оптимальный порог классификации
- Исправленные метрики (RMSE вместо MSE)
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Пытаемся импортировать LightGBM, если нет — fallback на sklearn
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Пытаемся импортировать Optuna для подбора гиперпараметров
try:
    import optuna
    from optuna.samplers import TPESampler

    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore", category=UserWarning)


class ContextAwareModel:
    """
    Модель для снапшотов из snapshot_builder1:
    - will_purchase_next_7d (классификация)
    - days_to_next_purchase (регрессия)
    - next_purchase_amount (регрессия)

    Улучшенная версия с:
    - LightGBM (или fallback на sklearn)
    - Автоматическим подбором гиперпараметров
    - Калибровкой вероятностей
    - Оптимизацией порога классификации
    """

    def __init__(
            self,
            random_state: int = 42,
            use_optuna: bool = True,
            optuna_trials: int = 50,
            calibrate_proba: bool = True,
            verbose: bool = True,
    ):
        self.random_state = random_state
        self.use_optuna = use_optuna and HAS_OPTUNA
        self.optuna_trials = optuna_trials
        self.calibrate_proba = calibrate_proba
        self.verbose = verbose

        # Модели
        self.clf = None
        self.clf_calibrated = None
        self.reg_days = None
        self.reg_amount = None

        # Препроцессинг
        self.feature_columns_: Optional[pd.Index] = None
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.numeric_medians_: Dict[str, float] = {}
        self.numeric_stds_: Dict[str, float] = {}

        # Оптимальный порог классификации
        self.optimal_threshold_: float = 0.5

        # Feature importance
        self.feature_importance_: Optional[pd.DataFrame] = None

        # Лучшие гиперпараметры
        self.best_params_clf_: Optional[Dict] = None
        self.best_params_reg_days_: Optional[Dict] = None
        self.best_params_reg_amount_: Optional[Dict] = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ========= ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =========

    def _split_features_targets(
            self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Разделяет DataFrame на фичи и таргеты."""
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
            "index",
        }

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return X, y_clf, y_days, y_amount

    def _detect_outliers_iqr(self, series: pd.Series, factor: float = 3.0) -> pd.Series:
        """Обнаружение выбросов методом IQR."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return series.clip(lower, upper)

    def _prepare_features_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Препроцессинг фичей при обучении:
        - Обработка выбросов в числовых колонках
        - Заполнение NaN медианой для числовых, модой для категориальных
        - Label encoding для категориальных (LightGBM умеет с ними работать)
        """
        X = X.copy()

        # Определяем типы колонок
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Обработка числовых колонок
        for col in numeric_cols:
            # Сохраняем медиану для заполнения NaN
            median_val = X[col].median()
            self.numeric_medians_[col] = median_val if pd.notna(median_val) else 0.0

            # Сохраняем std для нормализации (опционально)
            std_val = X[col].std()
            self.numeric_stds_[col] = std_val if pd.notna(std_val) and std_val > 0 else 1.0

            # Заполняем NaN медианой
            X[col] = X[col].fillna(self.numeric_medians_[col])

            # Обрабатываем выбросы
            X[col] = self._detect_outliers_iqr(X[col])

        # Обработка категориальных колонок
        for col in categorical_cols:
            # Заполняем NaN специальной категорией
            X[col] = X[col].fillna("__MISSING__")

            # Label encoding
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders_[col] = le

        self.feature_columns_ = X.columns

        return X

    def _prepare_features_infer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Препроцессинг фичей при предсказании."""
        if self.feature_columns_ is None:
            raise ValueError("Model is not fitted: feature_columns_ is None")

        X = X.copy()

        # Убираем служебные колонки, если они есть
        drop_cols = {"snapshot_date", "user_id", "last_ts", "index"}
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

        # Числовые колонки
        numeric_cols = [c for c in X.columns if c in self.numeric_medians_]
        for col in numeric_cols:
            X[col] = X[col].fillna(self.numeric_medians_.get(col, 0.0))
            X[col] = self._detect_outliers_iqr(X[col])

        # Категориальные колонки
        for col, le in self.label_encoders_.items():
            if col in X.columns:
                X[col] = X[col].fillna("__MISSING__")
                # Обрабатываем новые категории
                X[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Приводим к нужному набору колонок
        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns_]

        return X

    # ========= ПОДБОР ГИПЕРПАРАМЕТРОВ =========

    def _get_lgb_params(self, trial: "optuna.Trial", task: str = "classification") -> Dict:
        """Генерирует параметры LightGBM для Optuna trial."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }

        if task == "classification":
            params["objective"] = "binary"
            params["metric"] = "auc"
        else:
            params["objective"] = "regression"
            params["metric"] = "rmse"

        return params

    def _optimize_classifier(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Оптимизация гиперпараметров классификатора через Optuna."""

        def objective(trial):
            params = self._get_lgb_params(trial, "classification")

            cv_scores = []
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

            for train_idx, val_idx in skf.split(X, y):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                )

                proba = model.predict_proba(X_val_cv)[:, 1]
                score = roc_auc_score(y_val_cv, proba)
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=self.verbose)

        return self._get_lgb_params(study.best_trial, "classification")

    def _optimize_regressor(self, X: np.ndarray, y: np.ndarray, metric: str = "rmse") -> Dict:
        """Оптимизация гиперпараметров регрессора через Optuna."""

        def objective(trial):
            params = self._get_lgb_params(trial, "regression")

            cv_scores = []
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

            for train_idx, val_idx in kf.split(X):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                )

                preds = model.predict(X_val_cv)
                score = np.sqrt(mean_squared_error(y_val_cv, preds))
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=self.verbose)

        return self._get_lgb_params(study.best_trial, "regression")

    def _find_optimal_threshold(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        """Находит оптимальный порог классификации по F1-score."""
        precision, recall, thresholds = precision_recall_curve(y_true, proba)

        # Избегаем деления на ноль
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * (precision * recall) / (precision + recall),
            0
        )

        # thresholds на 1 элемент короче, чем precision/recall
        if len(thresholds) > 0:
            best_idx = np.argmax(f1_scores[:-1])
            return float(thresholds[best_idx])
        return 0.5

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
        self._log("🔧 Preparing features...")

        # 1. Разделяем фичи и таргеты
        X_train_raw, y_clf_train, y_days_train, y_amount_train = self._split_features_targets(train_df)

        # 2. Препроцессинг фичей
        X_train = self._prepare_features_fit(X_train_raw)
        X_train_np = X_train.values
        y_clf_np = y_clf_train.values

        self._log(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        self._log(f"📊 Positive class ratio: {y_clf_train.mean():.2%}")

        # 3. Создаем модели
        if HAS_LIGHTGBM:
            self._log("🚀 Using LightGBM")

            # Подбор гиперпараметров для классификатора
            if self.use_optuna:
                self._log(f"🔍 Optimizing classifier hyperparameters ({self.optuna_trials} trials)...")
                self.best_params_clf_ = self._optimize_classifier(X_train_np, y_clf_np)
            else:
                self.best_params_clf_ = {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "num_leaves": 63,
                    "min_child_samples": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "random_state": self.random_state,
                    "verbosity": -1,
                    "n_jobs": -1,
                }

            self.clf = lgb.LGBMClassifier(**self.best_params_clf_)
        else:
            self._log("⚠️ LightGBM not found, using sklearn GradientBoosting (slower)")
            self.clf = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=self.random_state,
            )

        # 4. Обучение классификатора
        self._log("🎯 Training classifier...")
        self.clf.fit(X_train_np, y_clf_np)

        # 5. Калибровка вероятностей
        if self.calibrate_proba:
            self._log("📐 Calibrating probabilities...")
            self.clf_calibrated = CalibratedClassifierCV(
                self.clf, method="isotonic", cv=3
            )
            self.clf_calibrated.fit(X_train_np, y_clf_np)

        # 6. Находим оптимальный порог
        proba_train = self._get_proba(X_train_np)
        self.optimal_threshold_ = self._find_optimal_threshold(y_clf_np, proba_train)
        self._log(f"📊 Optimal classification threshold: {self.optimal_threshold_:.3f}")

        # 7. Обучение регрессоров (только на положительных примерах)
        mask_pos = y_clf_train == 1
        n_positive = mask_pos.sum()
        self._log(f"📈 Training regressors on {n_positive} positive samples...")

        if n_positive >= 10:  # Минимум 10 примеров для адекватной регрессии
            X_pos = X_train_np[mask_pos]
            y_days_pos = y_days_train[mask_pos].values
            y_amount_pos = y_amount_train[mask_pos].values

            if HAS_LIGHTGBM:
                # Подбор гиперпараметров для регрессоров
                if self.use_optuna and n_positive >= 30:
                    self._log("🔍 Optimizing days regressor...")
                    self.best_params_reg_days_ = self._optimize_regressor(X_pos, y_days_pos)
                    self._log("🔍 Optimizing amount regressor...")
                    self.best_params_reg_amount_ = self._optimize_regressor(X_pos, y_amount_pos)
                else:
                    base_reg_params = {
                        "n_estimators": 200,
                        "learning_rate": 0.05,
                        "max_depth": 5,
                        "num_leaves": 31,
                        "min_child_samples": 10,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_state": self.random_state,
                        "verbosity": -1,
                        "n_jobs": -1,
                    }
                    self.best_params_reg_days_ = base_reg_params.copy()
                    self.best_params_reg_amount_ = base_reg_params.copy()

                self.reg_days = lgb.LGBMRegressor(**self.best_params_reg_days_)
                self.reg_amount = lgb.LGBMRegressor(**self.best_params_reg_amount_)
            else:
                self.reg_days = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=self.random_state,
                )
                self.reg_amount = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=self.random_state,
                )

            self.reg_days.fit(X_pos, y_days_pos)
            self.reg_amount.fit(X_pos, y_amount_pos)
        else:
            self._log("⚠️ Too few positive samples, training regressors on all data")
            if HAS_LIGHTGBM:
                self.reg_days = lgb.LGBMRegressor(random_state=self.random_state, verbosity=-1)
                self.reg_amount = lgb.LGBMRegressor(random_state=self.random_state, verbosity=-1)
            else:
                self.reg_days = GradientBoostingRegressor(random_state=self.random_state)
                self.reg_amount = GradientBoostingRegressor(random_state=self.random_state)

            self.reg_days.fit(X_train_np, y_days_train.values)
            self.reg_amount.fit(X_train_np, y_amount_train.values)

        # 8. Feature importance
        self._compute_feature_importance(X_train)

        # 9. Метрики на валидации (если есть)
        metrics = {}
        if val_df is not None and not val_df.empty:
            self._log("📊 Evaluating on validation set...")
            metrics = self.evaluate(val_df)

        self._log("✅ Training complete!")
        return metrics

    def _get_proba(self, X: np.ndarray) -> np.ndarray:
        """Получает вероятности с учетом калибровки."""
        if self.calibrate_proba and self.clf_calibrated is not None:
            return self.clf_calibrated.predict_proba(X)[:, 1]
        return self.clf.predict_proba(X)[:, 1]

    def _compute_feature_importance(self, X_train: pd.DataFrame):
        """Вычисляет важность признаков."""
        if HAS_LIGHTGBM and hasattr(self.clf, "feature_importances_"):
            importance = self.clf.feature_importances_
            self.feature_importance_ = pd.DataFrame({
                "feature": X_train.columns,
                "importance": importance
            }).sort_values("importance", ascending=False)

    # ========= ПРЕДСКАЗАНИЕ =========

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает DataFrame с предсказаниями:
        - purchase_proba
        - will_purchase_pred (0/1)
        - days_to_next_pred
        - next_purchase_amount_pred
        """
        if any(m is None for m in [self.clf, self.reg_days, self.reg_amount]):
            raise ValueError("Model is not fitted. Call fit() first.")

        # Убираем таргеты, если есть
        X_raw = df.copy()
        for col in ["will_purchase_next_7d", "days_to_next_purchase", "next_purchase_amount"]:
            if col in X_raw.columns:
                X_raw = X_raw.drop(columns=[col])

        # Препроцессинг
        X = self._prepare_features_infer(X_raw)
        X_np = X.values

        # Классификация (с калиброванными вероятностями)
        proba = self._get_proba(X_np)
        will_purchase_pred = (proba >= self.optimal_threshold_).astype(int)

        # Регрессия
        days_pred = self.reg_days.predict(X_np)
        amount_pred = self.reg_amount.predict(X_np)

        # Маскируем регрессионные предсказания для тех, у кого модель считает, что покупки не будет
        days_pred = np.where(will_purchase_pred == 1, days_pred, 999.0)
        amount_pred = np.where(will_purchase_pred == 1, amount_pred, 0.0)

        # Ограничиваем предсказания разумными значениями
        days_pred = np.clip(days_pred, 0, 999)
        amount_pred = np.clip(amount_pred, 0, None)

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
        - AUC-ROC, PR-AUC, F1 для will_purchase_next_7d
        - RMSE / MAE для дней до покупки (только на тех, где была покупка)
        - RMSE / MAE для суммы покупки (только на тех, где была покупка)
        """
        X_raw, y_clf, y_days, y_amount = self._split_features_targets(df)
        X = self._prepare_features_infer(X_raw)
        X_np = X.values

        proba = self._get_proba(X_np)
        preds = (proba >= self.optimal_threshold_).astype(int)

        metrics = {}

        # Классификация
        if len(np.unique(y_clf)) > 1:
            metrics["auc_roc"] = roc_auc_score(y_clf, proba)
            metrics["auc_pr"] = average_precision_score(y_clf, proba)
            metrics["f1"] = f1_score(y_clf, preds)
            metrics["threshold"] = self.optimal_threshold_
        else:
            metrics["auc_roc"] = float("nan")
            metrics["auc_pr"] = float("nan")
            metrics["f1"] = float("nan")
            metrics["threshold"] = self.optimal_threshold_

        # Регрессии — только на положительных примерах
        mask_pos = y_clf == 1
        if mask_pos.sum() > 0:
            days_pred = self.reg_days.predict(X_np[mask_pos])
            amount_pred = self.reg_amount.predict(X_np[mask_pos])

            # ИСПРАВЛЕНО: теперь это действительно RMSE, а не MSE
            metrics["rmse_days"] = np.sqrt(mean_squared_error(y_days[mask_pos], days_pred))
            metrics["mae_days"] = mean_absolute_error(y_days[mask_pos], days_pred)

            metrics["rmse_amount"] = np.sqrt(mean_squared_error(y_amount[mask_pos], amount_pred))
            metrics["mae_amount"] = mean_absolute_error(y_amount[mask_pos], amount_pred)
        else:
            metrics["rmse_days"] = float("nan")
            metrics["mae_days"] = float("nan")
            metrics["rmse_amount"] = float("nan")
            metrics["mae_amount"] = float("nan")

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Возвращает топ-N важных признаков."""
        if self.feature_importance_ is not None:
            return self.feature_importance_.head(top_n)
        return None

    # ========= СЕРИАЛИЗАЦИЯ =========

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "random_state": self.random_state,
            "use_optuna": self.use_optuna,
            "optuna_trials": self.optuna_trials,
            "calibrate_proba": self.calibrate_proba,
            "clf": self.clf,
            "clf_calibrated": self.clf_calibrated,
            "reg_days": self.reg_days,
            "reg_amount": self.reg_amount,
            "feature_columns_": self.feature_columns_,
            "label_encoders_": self.label_encoders_,
            "numeric_medians_": self.numeric_medians_,
            "numeric_stds_": self.numeric_stds_,
            "optimal_threshold_": self.optimal_threshold_,
            "feature_importance_": self.feature_importance_,
            "best_params_clf_": self.best_params_clf_,
            "best_params_reg_days_": self.best_params_reg_days_,
            "best_params_reg_amount_": self.best_params_reg_amount_,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "ContextAwareModel":
        state = joblib.load(path)
        model = cls(
            random_state=state.get("random_state", 42),
            use_optuna=state.get("use_optuna", False),  # При загрузке не нужен Optuna
            calibrate_proba=state.get("calibrate_proba", True),
        )
        model.clf = state["clf"]
        model.clf_calibrated = state.get("clf_calibrated")
        model.reg_days = state["reg_days"]
        model.reg_amount = state["reg_amount"]
        model.feature_columns_ = state["feature_columns_"]
        model.label_encoders_ = state.get("label_encoders_", {})
        model.numeric_medians_ = state.get("numeric_medians_", {})
        model.numeric_stds_ = state.get("numeric_stds_", {})
        model.optimal_threshold_ = state.get("optimal_threshold_", 0.5)
        model.feature_importance_ = state.get("feature_importance_")
        model.best_params_clf_ = state.get("best_params_clf_")
        model.best_params_reg_days_ = state.get("best_params_reg_days_")
        model.best_params_reg_amount_ = state.get("best_params_reg_amount_")
        return model