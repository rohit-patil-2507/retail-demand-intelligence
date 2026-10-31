"""
src/database/db_connector.py
Centralized MySQL connection manager reading credentials from root .env.
Supports both SQLAlchemy Engine (for Pandas) and raw DBAPI connections.
"""

import os
from pathlib import Path
import urllib.parse
from dotenv import find_dotenv, load_dotenv
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.logger import get_logger

logger = get_logger("DatabaseConnector")

# Locate and load root .env
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate_path = project_root / ".env"
    if candidate_path.is_file():
        dotenv_path = str(candidate_path)

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    load_dotenv(override=True)


def get_db_config() -> dict:
    """Fetch database configuration parameters from environment variables."""
    password = os.getenv("DB_PASSWORD")
    if password is None or password.strip() == "":
        logger.error(f"DB_PASSWORD missing or empty. Checked path: {dotenv_path}")
        raise ValueError("DB_PASSWORD environment variable is missing or empty.")

    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "database": os.getenv("DB_NAME", "retail_intelligence"),
        "user": os.getenv("DB_USER", "root"),
        "password": password.strip(),
    }


def get_engine() -> Engine:
    """
    Constructs and returns a SQLAlchemy connection Engine for Pandas I/O.
    Encodes passwords safely to prevent getaddrinfo resolution issues.
    """
    config = get_db_config()
    safe_password = urllib.parse.quote_plus(config["password"])
    
    conn_uri = (
        f"mysql+pymysql://{config['user']}:{safe_password}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )

    try:
        engine = create_engine(
            conn_uri,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
        )
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize SQLAlchemy Engine: {e}")
        raise


def get_raw_connection():
    """Returns a direct PyMySQL connection for lightweight scripting or cursor execution."""
    try:
        config = get_db_config()
        conn = pymysql.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
            cursorclass=pymysql.cursors.DictCursor,
        )
        return conn
    except Exception as e:
        logger.error(f"Direct PyMySQL Connection failed: {e}")
        raise


def test_connection() -> bool:
    """Smoke-tests the database connectivity."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1;")).scalar()
            if result == 1:
                logger.info("Database connection smoke test passed (SELECT 1 -> OK).")
                return True
        return False
    except Exception as e:
        logger.error(f"Database connection smoke test failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()