"""
src/models/train.py
-------------------
Trains production LightGBM booster using the Tweedie deviance objective
directly on raw transactional demand (total_quantity).
Saves booster artifact, feature metadata schema, and holdout scores.
"""

from pathlib import Path
import json
import time
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.common import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

log = get_logger("model_train")


def train_production_model():
    config = load_config()
    start_time = time.time()

    features_path = PROJECT_ROOT / config["paths"]["features_parquet"]
    log.info(f"Loading data from {features_path.name}...")
    df = pd.read_parquet(features_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Target switches directly to raw unit quantity for Tweedie loss
    target_col = "total_quantity"
    ignored_cols = {
        "transaction_date",
        "total_quantity",
        "total_revenue",
        "order_count",
        "target_log1p",
    }
    feature_cols = [c for c in df.columns if c not in ignored_cols]

    # Convert high-cardinality SKU identifier to category
    df["stock_code"] = df["stock_code"].astype("category")

    # Chronological holdout test split (28-day window)
    max_date = df["transaction_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=28)

    train_df = df[df["transaction_date"] <= cutoff_date].copy()
    test_df = df[df["transaction_date"] > cutoff_date].copy()

    log.info(
        f"Train window : {train_df['transaction_date'].min().date()} to "
        f"{train_df['transaction_date'].max().date()} ({len(train_df):,} rows)"
    )
    log.info(
        f"Test window  : {test_df['transaction_date'].min().date()} to "
        f"{max_date.date()} ({len(test_df):,} rows)"
    )

    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_test, y_test = test_df[feature_cols], test_df[target_col].values

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=["stock_code"],
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_test,
        label=y_test,
        reference=dtrain,
        categorical_feature=["stock_code"],
        free_raw_data=False,
    )

    model_cfg = config["model"]
    params = {
        "objective": model_cfg.get("objective", "tweedie"),
        "tweedie_variance_power": model_cfg.get("tweedie_variance_power", 1.5),
        "metric": model_cfg.get("metric", "mae"),
        "learning_rate": model_cfg.get("learning_rate", 0.05),
        "num_leaves": model_cfg.get("num_leaves", 63),
        "feature_fraction": model_cfg.get("feature_fraction", 0.8),
        "bagging_fraction": model_cfg.get("bagging_fraction", 0.8),
        "bagging_freq": model_cfg.get("bagging_freq", 1),
        "min_child_samples": model_cfg.get("min_child_samples", 50),
        "seed": config["project"].get("random_seed", 42),
        "verbose": model_cfg.get("verbose", -1),
    }

    log.info("Fitting LightGBM Tweedie booster...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=model_cfg.get("n_estimators", 1000),
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=model_cfg.get("early_stopping_rounds", 50),
                verbose=False,
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    log.info(f"Training complete. Best iteration: {model.best_iteration}")

    # Generate predictions directly in unit scale
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds = np.clip(test_preds, 0, None)

    # Compute operational metrics
    total_actual = np.sum(y_test)
    total_abs_error = np.sum(np.abs(y_test - test_preds))
    wape = (total_abs_error / total_actual) * 100
    mae = float(np.mean(np.abs(y_test - test_preds)))
    rmse = float(np.sqrt(np.mean((y_test - test_preds) ** 2)))

    eval_results = {
        "model": "LightGBM (Tweedie)",
        "best_iteration": int(model.best_iteration),
        "holdout_WAPE_pct": round(float(wape), 2),
        "holdout_MAE": round(mae, 2),
        "holdout_RMSE": round(rmse, 2),
    }

    log.info(
        f"Final Tweedie Holdout Results -> WAPE: {eval_results['holdout_WAPE_pct']}% | "
        f"MAE: {eval_results['holdout_MAE']} | RMSE: {eval_results['holdout_RMSE']}"
    )

    # Serialize booster model artifact
    models_dir = PROJECT_ROOT / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = PROJECT_ROOT / config["paths"]["lightgbm_model"]
    model.save_model(str(model_file))
    log.info(f"Tweedie booster artifact saved -> {model_file}")

    # Serialize feature metadata
    meta_file = PROJECT_ROOT / config["paths"]["feature_metadata"]
    metadata = {
        "features": feature_cols,
        "categorical_features": ["stock_code"],
        "num_features": len(feature_cols),
        "target": target_col,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    log.info(f"Feature metadata schema saved -> {meta_file}")

    # Save evaluation metrics
    metrics_file = PROJECT_ROOT / config["paths"]["eval_metrics"]
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    log.info(f"Model evaluation saved -> {metrics_file}")

    elapsed = time.time() - start_time
    log.info(f"Training workflow executed in {elapsed:.2f}s.")


if __name__ == "__main__":
    train_production_model()