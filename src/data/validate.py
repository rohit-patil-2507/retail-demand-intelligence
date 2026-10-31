"""
src/data/validate.py
--------------------
Audits raw transactional data against known ledger anomalies:
- Cancellations ('C%' prefix)
- Warehouse shrinkage/breakage (qty <= 0)
- Price <= 0
- Non-merchandise administrative codes
- Missing customer IDs (logged, not dropped)
"""

from pathlib import Path
import pandas as pd

from src.utils.common import PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger("data_validate")


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Verifies that all required operational columns exist and have content."""
    required_cols = [
        "invoice", "stock_code", "description",
        "quantity", "invoice_date", "price", "customer_id", "country"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing mandatory columns: {missing}")
    log.info(f"Schema check passed: all {len(required_cols)} mandatory columns present.")


def audit_anomalies(df: pd.DataFrame) -> dict:
    """Profiles the dataset for ledger anomalies identified in Notebook 02."""
    total_records = len(df)
    log.info(f"Beginning data profiling audit across {total_records:,} records...")

    # 1. Cancellations
    cancellations = df["invoice"].astype(str).str.startswith("C", na=False).sum()

    # 2. Damaged/test entries (negative qty without 'C')
    is_neg = (df["quantity"] <= 0) & (~df["invoice"].astype(str).str.startswith("C", na=False))
    neg_qty = is_neg.sum()

    # 3. Bad prices (freebies, internal writeoffs)
    zero_price = (df["price"] <= 0).sum()

    # 4. Known administrative non-product codes
    admin_codes = {
        "POST", "D", "M", "BANK CHARGES", "AMAZONFEE",
        "CRUK", "TEST", "PADS", "ADJUST", "ADJUST2"
    }
    clean_stock = df["stock_code"].astype(str).str.upper().str.strip()
    admin_skus = clean_stock.isin(admin_codes).sum()

    # 5. Missing customer checkouts (guest shoppers)
    missing_cust = df["customer_id"].isna().sum()

    summary = {
        "total_rows": total_records,
        "cancellations": (cancellations, (cancellations / total_records) * 100),
        "negative_qty_unlinked": (neg_qty, (neg_qty / total_records) * 100),
        "zero_or_negative_price": (zero_price, (zero_price / total_records) * 100),
        "admin_service_codes": (admin_skus, (admin_skus / total_records) * 100),
        "missing_customer_ids": (missing_cust, (missing_cust / total_records) * 100),
    }

    log.info("--- Data Profiling Audit Summary ---")
    log.info(f"  * Cancellations ('C%')        : {summary['cancellations'][0]:,} rows ({summary['cancellations'][1]:.2f}%)")
    log.info(f"  * Unlinked negative qty       : {summary['negative_qty_unlinked'][0]:,} rows ({summary['negative_qty_unlinked'][1]:.2f}%)")
    log.info(f"  * Zero or negative price      : {summary['zero_or_negative_price'][0]:,} rows ({summary['zero_or_negative_price'][1]:.2f}%)")
    log.info(f"  * Admin service codes         : {summary['admin_service_codes'][0]:,} rows ({summary['admin_service_codes'][1]:.2f}%)")
    log.info(f"  * Missing customer IDs (kept) : {summary['missing_customer_ids'][0]:,} rows ({summary['missing_customer_ids'][1]:.2f}%)")

    return summary


def run():
    """CLI execution entrypoint."""
    raw_path = PROJECT_ROOT / "data" / "raw" / "raw_transactions.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw snapshot not found at {raw_path}. Run ingest.py first.")

    df_raw = pd.read_parquet(raw_path)
    validate_raw_schema(df_raw)
    audit_anomalies(df_raw)


if __name__ == "__main__":
    run()