"""
ContextAwareModel v3.1 (Stable Demo Version)

Changes:
- Fixed AttributeError: get_feature_importance
- Removed CalibratedClassifierCV
- Removed masking (always predicts days/amount)
- Enforced class_weight='balanced'
- Added strict output clipping
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

# Try importing LightGBM
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Try importing Optuna
try:
    import optuna
    from optuna.samplers import TPESampler

    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore", category=UserWarning)


class ContextAwareModel:
    def __init__(
            self,
            random_state: int = 42,
            use_optuna: bool = True,
            optuna_trials: int = 20,
            verbose: bool = True,
    ):
        self.random_state = random_state
        self.use_optuna = use_optuna and HAS_OPTUNA
        self.optuna_trials = optuna_trials
        self.verbose = verbose

        # Models
        self.clf = None
        self.reg_days = None
        self.reg_amount = None

        # Preprocessing
        self.feature_columns_: Optional[pd.Index] = None
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.numeric_medians_: Dict[str, float] = {}

        # Threshold
        self.optimal_threshold_: float = 0.5
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.best_params_clf_: Optional[Dict] = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _split_features_targets(self, df: pd.DataFrame):
        target_cols = ["will_purchase_next_7d", "days_to_next_purchase", "next_purchase_amount"]
        drop_cols = set(target_cols) | {"snapshot_date", "user_id", "last_ts", "index"}

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        y_clf = df["will_purchase_next_7d"].astype(int)
        y_days = df["days_to_next_purchase"].astype(float)
        y_amount = df["next_purchase_amount"].astype(float)
        return X, y_clf, y_days, y_amount

    def _detect_outliers_iqr(self, series: pd.Series, factor: float = 3.0) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return series.clip(lower, upper)

    def _prepare_features_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in numeric_cols:
            median_val = X[col].median()
            self.numeric_medians_[col] = median_val if pd.notna(median_val) else 0.0
            X[col] = X[col].fillna(self.numeric_medians_[col])
            X[col] = self._detect_outliers_iqr(X[col])

        for col in categorical_cols:
            X[col] = X[col].fillna("__MISSING__")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders_[col] = le

        self.feature_columns_ = X.columns
        return X

    def _prepare_features_infer(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns_ is None:
            raise ValueError("Model is not fitted")

        X = X.copy()
        drop_cols = {"snapshot_date", "user_id", "last_ts", "index",
                     "will_purchase_next_7d", "days_to_next_purchase", "next_purchase_amount"}
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns_]

        numeric_cols = [c for c in X.columns if c in self.numeric_medians_]
        for col in numeric_cols:
            X[col] = X[col].fillna(self.numeric_medians_.get(col, 0.0))

        for col, le in self.label_encoders_.items():
            if col in X.columns:
                X[col] = X[col].fillna("__MISSING__")
                X[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        return X

    def _optimize_classifier(self, X, y) -> Dict:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "num_leaves": trial.suggest_int("num_leaves", 20, 64),
                "class_weight": "balanced",
                "random_state": self.random_state,
                "verbosity": -1,
                "n_jobs": -1
            }
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            cv_scores = []
            for train_idx, val_idx in skf.split(X, y):
                model = lgb.LGBMClassifier(**params)
                model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])])
                score = roc_auc_score(y[val_idx], model.predict_proba(X[val_idx])[:, 1])
                cv_scores.append(score)
            return np.mean(cv_scores)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=self.verbose)
        return study.best_params

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict:
        self._log("🚀 Training ContextAwareModel v3.2 (Smart Regressor)...")

        X_raw, y_clf, y_days, y_amount = self._split_features_targets(train_df)
        X = self._prepare_features_fit(X_raw)

        # --- 1. CLASSIFIER (Остается как был) ---
        if HAS_LIGHTGBM:
            if self.use_optuna:
                self._log("🔍 Tuning classifier...")
                self.best_params_clf_ = self._optimize_classifier(X.values, y_clf.values)
                self.best_params_clf_["class_weight"] = "balanced"
            else:
                self.best_params_clf_ = {
                    "n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
                    "class_weight": "balanced", "random_state": self.random_state
                }
            self.clf = lgb.LGBMClassifier(**self.best_params_clf_)
        else:
            self.clf = GradientBoostingClassifier(random_state=self.random_state)

        self.clf.fit(X.values, y_clf.values)

        # Threshold logic...
        proba = self.clf.predict_proba(X.values)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_clf.values, proba)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1)
        calc_threshold = float(thresholds[best_idx]) if len(thresholds) > best_idx else 0.5
        self.optimal_threshold_ = min(calc_threshold, 0.6)
        self._log(f"📊 Threshold: {self.optimal_threshold_:.3f}")

        # --- 2. REGRESSORS (ИСПРАВЛЕНО) ---
        # Раньше мы учили только на тех, кто купит (y_days < 300).
        # Теперь учим НА ВСЕХ, но для тех, кто не купит, ставим "потолок" (например, 30 дней).
        # Это научит модель для "плохих" юзеров предсказывать долгий срок, а не 3 дня.

        X_reg = X.values

        # Если y_days > 60 (значит покупки не было или она очень далеко),
        # ставим 30 (наш горизонт UI).
        # То есть учим модель: "Если он выглядит как труп, предсказывай 30 дней".
        y_days_reg = np.clip(y_days.values, 0, 30)

        # Для суммы берем только реальные покупки, т.к. предсказывать чек для тех кто не купит - сложно
        # Но чтобы не терять объем, можно заполнить средним чеком или 0.
        # Лучше оставить обучение amount только на позитивах или заполнить медианой.
        # Вариант: учим amount только на тех, кто реально покупал.
        mask_buyers = y_clf.values == 1

        # Чтобы размерности совпали для .fit, если используем разные выборки, нужны разные X
        if mask_buyers.sum() > 50:
            X_reg_amt = X.values[mask_buyers]
            y_amt_reg = y_amount.values[mask_buyers]
        else:
            X_reg_amt = X.values
            y_amt_reg = y_amount.values

        if HAS_LIGHTGBM:
            reg_params = {
                "n_estimators": 150, "learning_rate": 0.05, "max_depth": 5,
                "random_state": self.random_state, "verbosity": -1
            }
            self.reg_days = lgb.LGBMRegressor(**reg_params)
            self.reg_amount = lgb.LGBMRegressor(**reg_params)
        else:
            self.reg_days = GradientBoostingRegressor(random_state=self.random_state)
            self.reg_amount = GradientBoostingRegressor(random_state=self.random_state)

        # Дни учим на ВСЕХ (с клипом)
        self.reg_days.fit(X_reg, y_days_reg)

        # Сумму учим только на ПОКУПАТЕЛЯХ (чтобы не учить модель предсказывать чек 0)
        self.reg_amount.fit(X_reg_amt, y_amt_reg)

        # Feature Importance
        if HAS_LIGHTGBM:
            self.feature_importance_ = pd.DataFrame({
                "feature": X.columns,
                "importance": self.clf.feature_importances_
            }).sort_values("importance", ascending=False)
        elif hasattr(self.clf, "feature_importances_"):
            self.feature_importance_ = pd.DataFrame({
                "feature": X.columns,
                "importance": self.clf.feature_importances_
            }).sort_values("importance", ascending=False)

        return {"threshold": self.optimal_threshold_}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.clf is None:
            raise ValueError("Not fitted")

        X = self._prepare_features_infer(df)
        X_np = X.values

        proba = self.clf.predict_proba(X_np)[:, 1]
        pred_cls = (proba >= self.optimal_threshold_).astype(int)

        pred_days = self.reg_days.predict(X_np)
        pred_amount = self.reg_amount.predict(X_np)

        pred_days = np.clip(pred_days, 0.1, 30)
        pred_amount = np.clip(pred_amount, 10, None)

        return pd.DataFrame({
            "purchase_proba": proba,
            "will_purchase_pred": pred_cls,
            "days_to_next_pred": pred_days,
            "next_purchase_amount_pred": pred_amount
        }, index=df.index)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        X_raw, y_clf, _, _ = self._split_features_targets(df)
        X = self._prepare_features_infer(X_raw)
        proba = self.clf.predict_proba(X.values)[:, 1]
        return {
            "auc": roc_auc_score(y_clf, proba) if len(np.unique(y_clf)) > 1 else 0,
            "avg_proba": float(proba.mean())
        }

    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Возвращает топ-N важных признаков."""
        if self.feature_importance_ is not None:
            return self.feature_importance_.head(top_n)
        return None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
