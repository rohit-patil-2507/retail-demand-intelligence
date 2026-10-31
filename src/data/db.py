"""
db.py - Robust Centralized MySQL connection management using python-dotenv.
"""
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import mysql.connector

# Strategy 1: Attempt to find .env starting from current working directory upwards
dotenv_path = find_dotenv(usecwd=True)

# Strategy 2: If find_dotenv fails, explicitly check the parent directory of this file (project root)
if not dotenv_path:
    project_root = Path(__file__).resolve().parent.parent
    candidate_path = project_root / ".env"
    if candidate_path.is_file():
        dotenv_path = str(candidate_path)

# Load the found .env file
if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    # Fallback default load
    load_dotenv(override=True)


def get_db_config() -> dict:
    """Read database configuration directly from environment variables."""
    password = os.getenv("DB_PASSWORD")
    
    # Detailed diagnosis if password is missing
    if password is None or password.strip() == "":
        raise ValueError(
            f"DB_PASSWORD environment variable is missing or empty.\n"
            f"Resolved .env location: {dotenv_path if dotenv_path else 'NOT FOUND'}\n"
            f"Current Working Directory: {os.getcwd()}\n"
            f"Make sure your file is strictly named '.env' (not '.env.txt') and contains: DB_PASSWORD=your_password"
        )

    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "database": os.getenv("DB_NAME", "retail_forecasting"),
        "user": os.getenv("DB_USER", "root"),
        "password": password.strip(),
    }


def get_connection():
    """Establish and return a MySQL database connection."""
    config = get_db_config()
    
    return mysql.connector.connect(**config)