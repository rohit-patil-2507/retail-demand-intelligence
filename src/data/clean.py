"""
src/data/clean.py
-----------------
Applies audited data cleansing rules to produce reliable transactional records.
Also extracts the static SKU metadata dimension for business-facing displays.
"""

from pathlib import Path
import pandas as pd

from src.utils.common import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

log = get_logger("data_clean")


def apply_cleaning_filters(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transactions using deterministic business criteria:
    - Drops returns to avoid future-lookahead negative spikes.
    - Drops internal system breakages (quantity <= 0).
    - Drops zero-price items.
    - Drops postage and administrative lines.
    - Retains guest checkouts (NaN customer_id) to protect 22.76% of sales volume.
    """
    initial_rows = len(raw)
    log.info(f"Applying data cleansing filters to {initial_rows:,} raw records...")

    # 1. Strip and standardize identifiers
    df = raw.copy()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["invoice"] = df["invoice"].astype(str).str.strip()

    # 2. Boolean filter masks
    is_cancelled = df["invoice"].str.startswith("C", na=False)
    is_bad_qty = df["quantity"] <= 0
    is_bad_price = df["price"] <= 0

    admin_codes = {
        "POST", "D", "M", "BANK CHARGES", "AMAZONFEE",
        "CRUK", "TEST", "PADS", "ADJUST", "ADJUST2"
    }
    is_admin = df["stock_code"].isin(admin_codes)

    # 3. Retain only valid outbound customer sales
    valid_mask = (~is_cancelled) & (~is_bad_qty) & (~is_bad_price) & (~is_admin)
    clean = df[valid_mask].copy()

    # 4. Standardize dates to pure dates (discards random invoice second-level noise)
    clean["transaction_date"] = pd.to_datetime(clean["invoice_date"]).dt.date
    clean["transaction_date"] = pd.to_datetime(clean["transaction_date"])

    dropped_rows = initial_rows - len(clean)
    retention_pct = (len(clean) / initial_rows) * 100
    log.info(f"Cleaning complete: dropped {dropped_rows:,} noisy rows ({100 - retention_pct:.2f}%).")
    log.info(f"Retained {len(clean):,} genuine transactions ({retention_pct:.2f}% retention).")
    log.info(f"Active physical catalog: {clean['stock_code'].nunique():,} unique SKUs.")

    return clean


def extract_catalogue_dimension(clean: pd.DataFrame, out_path: Path) -> None:
    """
    Builds a canonical product metadata table.
    Takes the mode (most common) description for each SKU to resolve cashier typos.
    """
    log.info("Generating canonical SKU product description catalog...")
    valid_desc = clean.dropna(subset=["description"]).copy()
    valid_desc["description"] = valid_desc["description"].astype(str).str.strip().str.upper()

    def get_mode_description(series: pd.Series) -> str:
        modes = series.mode()
        return modes.iloc[0] if not modes.empty else "MERCHANDISE ITEM"

    catalogue = (
        valid_desc.groupby("stock_code")["description"]
        .agg(get_mode_description)
        .reset_index()
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalogue.to_csv(out_path, index=False)
    log.info(f"Product catalog dimension saved ({len(catalogue):,} SKUs) -> {out_path}")


def run():
    """CLI execution entrypoint."""
    config = load_config()
    raw_path = PROJECT_ROOT / "data" / "raw" / "raw_transactions.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw snapshot at {raw_path}. Run ingest.py first.")

    raw_data = pd.read_parquet(raw_path)
    clean_data = apply_cleaning_filters(raw_data)

    # Save static product catalog
    cat_path = PROJECT_ROOT / config.get("paths", {}).get("sku_catalogue_csv", "data/processed/sku_catalogue.csv")
    extract_catalogue_dimension(clean_data, cat_path)

    # Save verified transaction table
    clean_path = PROJECT_ROOT / config.get("paths", {}).get(
        "clean_transactions_parquet", "data/processed/clean_transactions.parquet"
    )
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    keep_cols = [
        "invoice", "stock_code", "quantity", "price",
        "transaction_date", "customer_id", "country"
    ]
    clean_data[keep_cols].to_parquet(clean_path, index=False, compression="snappy")
    log.info(f"Clean transactional records saved -> {clean_path}")


if __name__ == "__main__":
    run()