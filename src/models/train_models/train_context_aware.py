"""Training script for ContextAwareModel.

This script loads snapshot datasets, trains the ContextAwareModel,
evaluates it on validation and test sets, logs metrics and plots to
dedicated experiment folders, and saves the trained model.  It is
intended to produce reproducible artefacts for the experimentation
section of the thesis.
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_curve,
)

from src.models.models import ContextAwareModel

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "..", "..", "analytics", "data", "snapshots", "model1")

EXPERIMENT_DIR = os.path.join(BASE_DIR, "..", "experiments", "context_aware_model1")
MODEL_PATH = os.path.join(BASE_DIR, "..", "production_models", "context_aware_model1.pkl")

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# --- LOGGING ---
LOG_PATH = os.path.join(EXPERIMENT_DIR, "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- UTILITIES ---
def load_dataset(name: str) -> pd.DataFrame:
    """Load snapshot dataset by name (train/val/test)."""
    path = os.path.join(SNAPSHOT_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def save_plot(fig, name: str) -> None:
    """Save matplotlib figure to the experiment directory and log it."""
    path = os.path.join(EXPERIMENT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path}")


def evaluate_full(model: ContextAwareModel, df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Compute a suite of metrics for both classification and regression tasks.

    Returns a dictionary of metrics and the probability predictions used for
    plotting curves.
    """
    # Ground truth values
    y_true = df["will_purchase_next_7d"].values
    y_days = df["days_to_next_purchase"].values
    y_amount = df["next_purchase_amount"].values

    # Predictions
    preds = model.predict(df)
    proba = preds["purchase_proba"].values
    y_pred = preds["will_purchase_pred"].values

    metrics: dict[str, float] = {}

    # --- Classification metrics ---
    # Handle degenerate cases (only one class) gracefully
    metrics["roc_auc"] = (
        roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else 0.0
    )
    metrics["pr_auc"] = average_precision_score(y_true, proba)
    metrics["f1"] = f1_score(y_true, y_pred)

    # --- Regression metrics ---
    # Days: compute MAE and RMSE; RMSE manually to support older sklearn versions
    metrics["mae_days"] = mean_absolute_error(y_days, preds["days_to_next_pred"])  # type: ignore
    mse_days = mean_squared_error(y_days, preds["days_to_next_pred"])  # type: ignore
    metrics["rmse_days"] = float(np.sqrt(mse_days))

    # Amount: evaluate only on buyers
    mask_buyers = y_true == 1
    if mask_buyers.sum() > 0:
        metrics["mae_amount"] = mean_absolute_error(
            y_amount[mask_buyers], preds.loc[mask_buyers, "next_purchase_amount_pred"]
        )
        mse_amount = mean_squared_error(
            y_amount[mask_buyers], preds.loc[mask_buyers, "next_purchase_amount_pred"]
        )
        metrics["rmse_amount"] = float(np.sqrt(mse_amount))
    else:
        metrics["mae_amount"] = 0.0
        metrics["rmse_amount"] = 0.0

    return metrics, proba


def plot_roc(y_true: np.ndarray, proba: np.ndarray) -> None:
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    save_plot(fig, "roc_curve.png")


def plot_pr(y_true: np.ndarray, proba: np.ndarray) -> None:
    """Plot and save the Precision–Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    save_plot(fig, "pr_curve.png")


def plot_feature_importance(model: ContextAwareModel) -> None:
    """Plot and save the top 20 feature importances if available."""
    fi = model.get_feature_importance(top_n=20)
    if fi is None or fi.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(fi["feature"], fi["importance"])
    ax.invert_yaxis()
    ax.set_title("Top-20 Feature Importance")
    save_plot(fig, "feature_importance.png")


def plot_proba_hist(proba: np.ndarray) -> None:
    """Plot and save a histogram of predicted probabilities."""
    fig, ax = plt.subplots()
    ax.hist(proba, bins=30)
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("purchase_proba")
    ax.set_ylabel("count")
    save_plot(fig, "proba_hist.png")


def main() -> None:
    logger.info("Loading datasets...")
    train_df = load_dataset("train")
    val_df = load_dataset("val")
    test_df = load_dataset("test")

    logger.info(f"Train: {len(train_df):,}")
    logger.info(f"Val:   {len(val_df):,}")
    logger.info(f"Test:  {len(test_df):,}")

    # Save experiment configuration
    config = {
        "model": "ContextAwareModel",
        "snapshot_dir": SNAPSHOT_DIR,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(os.path.join(EXPERIMENT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train model
    model = ContextAwareModel(verbose=True)
    logger.info("Training model...")
    model.fit(train_df, val_df=val_df)

    # Evaluate on validation set
    logger.info("Validation evaluation...")
    val_metrics, val_proba = evaluate_full(model, val_df)

    # Evaluate on test set
    logger.info("Test evaluation...")
    test_metrics, test_proba = evaluate_full(model, test_df)

    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(EXPERIMENT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Metrics summary:")
    for split, metrics in all_metrics.items():
        logger.info(f"--- {split.upper()} ---")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

    # Generate plots using validation probabilities
    plot_roc(val_df["will_purchase_next_7d"].values, val_proba)
    plot_pr(val_df["will_purchase_next_7d"].values, val_proba)
    plot_feature_importance(model)
    plot_proba_hist(val_proba)

    # Save model
    logger.info(f"Saving model - {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    logger.info("Training pipeline finished successfully")


if __name__ == "__main__":
    main()
