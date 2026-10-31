"""
src/models/baseline.py
----------------------
Calculates standard retail heuristic benchmarks (Naive lag-1, Lag-7 seasonal,
and rolling averages) on the holdout test window to set the baseline WAPE.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils.common import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

log = get_logger("baseline_eval")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes standard retail forecasting performance metrics."""
    # Clip negative predictions to zero (demand cannot be negative)
    y_pred = np.clip(y_pred, 0, None)
    
    total_actual = np.sum(y_true)
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    
    # Weighted Absolute Percentage Error (WAPE)
    wape = (total_abs_error / total_actual) * 100 if total_actual > 0 else np.nan
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "WAPE_pct": round(float(wape), 2),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    }


def evaluate_baselines():
    config = load_config()
    features_path = PROJECT_ROOT / config["paths"]["features_parquet"]
    
    log.info(f"Loading feature matrix from {features_path.name}...")
    df = pd.read_parquet(features_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    
    # 28-day holdout test split
    max_date = df["transaction_date"].max()
    split_date = max_date - pd.Timedelta(days=28)
    
    test_df = df[df["transaction_date"] > split_date].copy()
    y_true = test_df["total_quantity"].values
    
    log.info(f"Evaluating baselines on holdout window: {split_date.date()} to {max_date.date()} ({len(test_df):,} rows)...")
    
    baselines = {
        "naive_yesterday (Lag 1)": test_df["lag_1"].values,
        "seasonal_last_week (Lag 7)": test_df["lag_7"].values,
        "rolling_mean_7d": test_df["rolling_mean_7"].values,
        "rolling_mean_14d": test_df["rolling_mean_14"].values,
        "rolling_mean_28d": test_df["rolling_mean_28"].values,
    }
    
    results = {}
    for name, preds in baselines.items():
        # Fill warm-up NaNs with 0
        clean_preds = np.nan_to_num(preds, nan=0.0)
        res = calculate_metrics(y_true, clean_preds)
        results[name] = res
        log.info(f"  {name:<30} -> WAPE: {res['WAPE_pct']}% | MAE: {res['MAE']} | RMSE: {res['RMSE']}")
        
    # Save baseline metrics
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_file = metrics_dir / "baseline_results.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    log.info(f"Baseline benchmarking finished. Saved -> {out_file}")


if __name__ == "__main__":
    evaluate_baselines()