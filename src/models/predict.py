"""
src/models/predict.py
---------------------
Generates 28-day forward demand predictions using the trained LightGBM model.
"""

from pathlib import Path
import json
import time
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.common import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

log = get_logger("batch_predict")


def run_batch_prediction() -> pd.DataFrame:
    """Loads the model artifact and generates 28-day demand forecasts for each SKU."""
    config = load_config()
    start_time = time.time()

    # Define file paths
    model_path = PROJECT_ROOT / config["paths"]["lightgbm_model"]
    meta_path = PROJECT_ROOT / config["paths"]["feature_metadata"]
    features_path = PROJECT_ROOT / config["paths"]["features_parquet"]
    output_path = PROJECT_ROOT / config["paths"]["forecast_predictions_parquet"]

    # Basic safety checks
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Feature metadata not found at {meta_path}. Run train.py first.")

    log.info(f"Loading trained LightGBM model from: {model_path.name}")
    model = lgb.Booster(model_file=str(model_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    log.info(f"Loading feature dataset from: {features_path.name}")
    df = pd.read_parquet(features_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Score the most recent 28-day horizon for all items
    max_date = df["transaction_date"].max()
    eval_start_date = max_date - pd.Timedelta(days=28)

    forecast_window = df[df["transaction_date"] > eval_start_date].copy()
    log.info(
        f"Scoring 28-day window ({eval_start_date.date()} to {max_date.date()}) "
        f"across {forecast_window['stock_code'].nunique():,} SKUs..."
    )

    # Convert category column to match the format used during training
    forecast_window["stock_code"] = forecast_window["stock_code"].astype("category")
    
# Tweedie outputs directly in real sales unit scale
    X = forecast_window[feature_cols]
    preds_raw = model.predict(X)
    preds_actual = np.clip(preds_raw, 0, None)
    forecast_window["forecasted_quantity"] = np.round(preds_actual, 2)

    # Estimate expected revenue = forecasted units * average unit price
    forecast_window["forecasted_revenue"] = np.round(
        forecast_window["forecasted_quantity"] * forecast_window["avg_unit_price"], 2
    )

    # Select only the columns needed by the dashboard and inventory engine
    output_cols = [
        "stock_code",
        "transaction_date",
        "total_quantity",
        "forecasted_quantity",
        "avg_unit_price",
        "forecasted_revenue",
    ]
    result_df = forecast_window[output_cols].copy()

    # Save to data/predictions/
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False, compression="snappy")

    elapsed = round(time.time() - start_time, 2)
    log.info(f"Batch prediction finished: {len(result_df):,} rows generated in {elapsed}s.")
    log.info(f"Saved forecast dataset -> {output_path}")

    return result_df


if __name__ == "__main__":
    run_batch_prediction()