"""
src/features/build_features.py
------------------------------
Transforms irregular transaction lines into a regular daily time-series panel:
1. Filters qualified SKUs (active trading days >= 100).
2. Expands transactions onto a continuous Cartesian date-spine (SKU x Days).
3. Computes volume-weighted unit price with forward-fill propagation.
4. Generates leakage-free autoregressive lags and rolling statistics.
5. Applies log1p target transformation and persists to features.parquet.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.utils.common import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

log = get_logger("build_features")


def load_clean_data(parquet_path: Path) -> pd.DataFrame:
    """Loads clean transactional records produced by src/data/clean.py."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Clean transactions not found at {parquet_path}. "
            "Please run `python -m src.data.clean` first."
        )

    log.info(f"Loading verified transaction records from {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def get_qualified_skus(df: pd.DataFrame, min_active_days: int = 100) -> list[str]:
    """
    Identifies catalog items with sufficient trading density.
    SKUs with fewer than 100 active transaction days represent dead stock,
    seasonal one-offs, or irregular orders that act as pure noise for ML.
    """
    active_days = df.groupby("stock_code")["transaction_date"].nunique()
    qualified = active_days[active_days >= min_active_days].index.tolist()

    log.info(
        f"Qualified {len(qualified):,} dense SKUs (>= {min_active_days} active days) "
        f"out of {len(active_days):,} total observed items."
    )
    return qualified


def compute_daily_sku_aggregates(df: pd.DataFrame, qualified_skus: list[str]) -> pd.DataFrame:
    """
    Aggregates granular sales to a daily level per SKU.
    Uses volume-weighted average price so wholesale bulk discounts
    are accurately represented instead of naive arithmetic averages.
    """
    log.info("Aggregating transactional lines to daily (SKU, Date) sales...")
    subset = df[df["stock_code"].isin(qualified_skus)].copy()
    subset["line_revenue"] = subset["quantity"] * subset["price"]

    daily = (
        subset.groupby(["stock_code", "transaction_date"], observed=False)
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("line_revenue", "sum"),
            order_count=("invoice", "nunique"),
        )
        .reset_index()
    )

    # Volume-weighted realized unit price
    daily["avg_unit_price"] = daily["total_revenue"] / daily["total_quantity"]
    return daily


def create_continuous_panel(daily_df: pd.DataFrame, qualified_skus: list[str]) -> pd.DataFrame:
    """
    Builds a full Cartesian date-spine grid (All SKUs x Full Calendar Horizon).
    Days without orders are populated with quantity = 0 to reflect real catalog sparsity.
    """
    start_date = daily_df["transaction_date"].min()
    end_date = daily_df["transaction_date"].max()
    log.info(f"Expanding continuous grid across calendar window: {start_date.date()} to {end_date.date()}...")

    full_dates = pd.date_range(start=start_date, end=end_date, freq="D", name="transaction_date")
    sku_index = pd.Index(qualified_skus, name="stock_code")

    grid = pd.MultiIndex.from_product([sku_index, full_dates]).to_frame().reset_index(drop=True)
    panel = pd.merge(grid, daily_df, on=["stock_code", "transaction_date"], how="left")

    # Inactive sales days have 0 quantity and 0 revenue
    panel["total_quantity"] = panel["total_quantity"].fillna(0.0)
    panel["total_revenue"] = panel["total_revenue"].fillna(0.0)
    panel["order_count"] = panel["order_count"].fillna(0.0)

    # Sort deterministically by item and calendar date
    panel = panel.sort_values(["stock_code", "transaction_date"]).reset_index(drop=True)

    # Forward-fill unit price so inactive days retain the last known price signal
    panel["avg_unit_price"] = (
        panel.groupby("stock_code")["avg_unit_price"]
        .ffill()
        .bfill()
    )

    sparsity = (panel["total_quantity"] == 0).mean() * 100
    log.info(f"Panel finalized: {len(panel):,} rows ({sparsity:.2f}% zero-demand days).")
    return panel


def engineer_features(panel: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    """
    Generates autoregressive lags, rolling statistics, and calendar features.
    Strictly applies shift(1) to all rolling windows to eliminate lookahead leakage.
    """
    log.info("Computing autoregressive features and calendar signals...")
    df = panel.copy()

    # 1. Calendar encodings
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df["month"] = df["transaction_date"].dt.month
    df["day_of_month"] = df["transaction_date"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical day-of-week transform
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # 2. Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("stock_code")["total_quantity"].shift(lag)

    # 3. Rolling window statistics (shifted by 1 day so t's forecast never peeks at t)
    for w in windows:
        shifted_demand = df.groupby("stock_code")["total_quantity"].shift(1)
        grouped_shifted = shifted_demand.groupby(df["stock_code"])

        df[f"rolling_mean_{w}"] = (
            grouped_shifted.rolling(window=w, min_periods=max(1, w // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"rolling_std_{w}"] = (
            grouped_shifted.rolling(window=w, min_periods=max(1, w // 2))
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    # 4. Target variable: log1p stabilizes extreme wholesale volume spikes
    df["target_log1p"] = np.log1p(df["total_quantity"])

    # 5. Drop the initial warm-up period where the longest lag (28 days) is undefined
    max_history_needed = max(lags + windows)
    clean_panel = df.dropna(subset=[f"lag_{max_history_needed}"]).reset_index(drop=True)

    log.info(
        f"Feature engineering complete: {len(clean_panel):,} training rows "
        f"x {clean_panel.shape[1]} features (warm-up dropped: {len(panel) - len(clean_panel):,} rows)."
    )
    return clean_panel


def run():
    """CLI execution entrypoint."""
    config = load_config()
    start_time = time.time()

    clean_parquet = PROJECT_ROOT / config.get("paths", {}).get(
        "clean_transactions_parquet", "data/processed/clean_transactions.parquet"
    )
    output_parquet = PROJECT_ROOT / config.get("paths", {}).get(
        "features_parquet", "data/processed/features.parquet"
    )

    # Pipeline stages
    df_clean = load_clean_data(clean_parquet)
    min_days = config.get("qualification", {}).get("min_active_days", 100)
    qualified = get_qualified_skus(df_clean, min_active_days=min_days)

    daily_agg = compute_daily_sku_aggregates(df_clean, qualified)
    panel = create_continuous_panel(daily_agg, qualified)

    feature_cfg = config.get("features", {})
    features_df = engineer_features(
        panel,
        lags=feature_cfg.get("lags", [1, 2, 7, 14, 21, 28]),
        windows=feature_cfg.get("rolling_windows", [7, 14, 28]),
    )

    # Save final feature matrix
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_parquet, index=False, compression="snappy")

    elapsed = time.time() - start_time
    log.info(f"features.parquet successfully written -> {output_parquet} ({elapsed:.2f}s total run time).")


if __name__ == "__main__":
    run()