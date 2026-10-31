"""
src/data/ingest.py
------------------
Pulls raw transactions from the MySQL staging table and creates a local snapshot.
Uses batch chunking to avoid choking system RAM when pulling 1M+ rows.
"""

from pathlib import Path
import time
import pandas as pd
from sqlalchemy import text

from src.database.db_connector import get_engine
from src.utils.common import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

log = get_logger("data_ingest")


def fetch_raw_data(chunk_size: int = 100_000) -> pd.DataFrame:
    """
    Streams records in chunks from the staging table so memory stays flat.
    Returns the complete raw DataFrame.
    """
    config = load_config()
    staging_table = config.get("database", {}).get("staging_table", "raw_retail_transactions")
    engine = get_engine()

    query = f"""
    SELECT 
        invoice,
        stock_code,
        description,
        quantity,
        invoice_date,
        price,
        customer_id,
        country
    FROM {staging_table}
    """

    log.info(f"Initiating extraction from MySQL table `{staging_table}` (chunk size: {chunk_size:,})...")
    start_time = time.time()

    chunks = []
    total_rows = 0

    with engine.connect() as conn:
        for idx, chunk in enumerate(pd.read_sql(text(query), conn, chunksize=chunk_size)):
            total_rows += len(chunk)
            chunks.append(chunk)
            log.info(f"Streamed batch #{idx + 1:02d} -> {total_rows:,} rows pulled so far.")

    raw_df = pd.concat(chunks, ignore_index=True)
    elapsed = time.time() - start_time
    log.info(f"Extraction finished: {len(raw_df):,} total rows fetched in {elapsed:.2f}s.")
    return raw_df


def save_raw_snapshot(df: pd.DataFrame) -> Path:
    """Persists raw staging data to data/raw/ as a fast, immutable Parquet snapshot."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file = raw_dir / "raw_transactions.parquet"

    # Save with snappy compression to keep size reasonable (~30MB instead of 150MB CSV)
    df.to_parquet(out_file, index=False, compression="snappy")
    log.info(f"Raw transaction snapshot persisted -> {out_file}")
    return out_file


def run():
    """CLI execution entrypoint."""
    df_raw = fetch_raw_data()
    save_raw_snapshot(df_raw)


if __name__ == "__main__":
    run()