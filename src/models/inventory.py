"""
src/models/inventory.py
-----------------------
Calculates inventory replenishment policies: Safety Stock (SS),
Reorder Point (ROP), and automated stockout risk warnings.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.utils.common import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

log = get_logger("inventory_engine")


def run_inventory_optimization() -> pd.DataFrame:
    """Translates point forecasts into supply chain replenishment targets."""
    config = load_config()
    start_time = time.time()

    # Define file paths
    forecast_path = PROJECT_ROOT / config["paths"]["forecast_predictions_parquet"]
    catalogue_path = PROJECT_ROOT / config["paths"]["sku_catalogue_csv"]
    output_path = PROJECT_ROOT / config["paths"]["replenishment_plan_parquet"]

    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing forecast data at {forecast_path}. Run predict.py first.")

    log.info(f"Reading forward forecasts from: {forecast_path.name}")
    df_forecast = pd.read_parquet(forecast_path)

    # Supply chain parameters from config (or default fallback values)
    inv_cfg = config.get("inventory", {})
    lead_time_days = inv_cfg.get("lead_time_days", 7)     # 7 days vendor delivery wait
    service_level_z = inv_cfg.get("service_level_z", 1.65) # 1.65 corresponds to a 95% service level

    log.info(f"Using Lead Time: {lead_time_days} days | Service Level Z: {service_level_z} (95% target)")

    # Aggregate 28-day projected performance per SKU
    sku_summary = (
        df_forecast.groupby("stock_code")
        .agg(
            total_projected_demand=("forecasted_quantity", "sum"),
            daily_forecast_mean=("forecasted_quantity", "mean"),
            daily_forecast_std=("forecasted_quantity", "std"),
            avg_unit_price=("avg_unit_price", "mean"),
            total_projected_revenue=("forecasted_revenue", "sum"),
        )
        .reset_index()
    )

    # If daily standard deviation is NaN (e.g. constant demand), use a modest fallback
    sku_summary["daily_forecast_std"] = sku_summary["daily_forecast_std"].fillna(
        sku_summary["daily_forecast_mean"] * 0.3
    )

    # --- Standard Supply Chain Formulas ---
    # 1. Lead Time Demand: Expected sales during the vendor delivery period
    sku_summary["lead_time_demand"] = np.round(
        sku_summary["daily_forecast_mean"] * lead_time_days, 1
    )

    # 2. Safety Stock (SS) = Z * std_dev * sqrt(Lead_Time)
    sku_summary["safety_stock"] = np.ceil(
        service_level_z * sku_summary["daily_forecast_std"] * np.sqrt(lead_time_days)
    ).astype(int)

    # 3. Reorder Point (ROP) = Lead Time Demand + Safety Stock
    sku_summary["reorder_point"] = np.ceil(
        sku_summary["lead_time_demand"] + sku_summary["safety_stock"]
    ).astype(int)

    # Simulated realistic warehouse stock (equal to ~10 days of average demand for testing)
    sku_summary["current_stock"] = np.round(
        sku_summary["daily_forecast_mean"] * 10, 0
    ).astype(int)

    # Days of supply remaining with current stock
    sku_summary["days_of_supply"] = np.where(
        sku_summary["daily_forecast_mean"] > 0,
        np.round(sku_summary["current_stock"] / sku_summary["daily_forecast_mean"], 1),
        999.0
    )

   # Inventory Health Status Flag
    def check_health(row):
        # FIX: ML models output tiny fractions (e.g., 0.42 units total). 
        # If 28-day demand is less than 1 whole unit, treat it as zero demand.
        if row["total_projected_demand"] < 1:
            if row["current_stock"] <= 0:
                return "NO DEMAND (Do Not Reorder)"
            else:
                return "OVERSTOCKED (Capital Tied)"
                
        elif row["current_stock"] <= row["reorder_point"]:
            return "REORDER NOW (Stockout Risk)"
            
        elif row["days_of_supply"] > 45:
            return "OVERSTOCKED (Capital Tied)"
            
        return "OPTIMAL"

    sku_summary["inventory_status"] = sku_summary.apply(check_health, axis=1)

    # Merge human-readable product names if catalogue exists
    if catalogue_path.exists():
        cat_df = pd.read_csv(catalogue_path)
        sku_summary = sku_summary.merge(cat_df, on="stock_code", how="left")
        sku_summary["description"] = sku_summary["description"].fillna("MERCHANDISE ITEM")
    else:
        sku_summary["description"] = "MERCHANDISE ITEM"

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sku_summary.to_parquet(output_path, index=False, compression="snappy")

    elapsed = round(time.time() - start_time, 2)
    log.info(f"Inventory policy calculated for {len(sku_summary):,} SKUs in {elapsed}s.")
    log.info(f"Replenishment plan saved -> {output_path}")

    # Log summary counts so you can easily review the results in the console
    status_breakdown = sku_summary["inventory_status"].value_counts().to_dict()
    for status, count in status_breakdown.items():
        log.info(f"  * {status:<30}: {count:,} items")

    return sku_summary


if __name__ == "__main__":
    run_inventory_optimization()