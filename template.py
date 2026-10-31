import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s"
)

# Comprehensive production-grade project structure
list_of_files = [
    # 1. Source Package: Database & Connection Layer
    "src/__init__.py",
    "src/database/__init__.py",
    "src/database/db_connector.py",
    
    # 2. Source Package: Ingestion, Validation & Cleaning
    "src/data/__init__.py",
    "src/data/ingest.py",
    "src/data/validate.py",
    "src/data/clean.py",
    "src/data/db.py",
    
    # 3. Source Package: Feature Engineering
    "src/features/__init__.py",
    "src/features/build_features.py",
    
    # 4. Source Package: Modeling, Inference & Inventory Policy
    "src/models/__init__.py",
    "src/models/baseline.py",
    "src/models/train.py",
    "src/models/evaluate.py",
    "src/models/predict.py",       # Batch recursive forecasting for next 28 days
    "src/models/inventory.py",     # Safety stock & Reorder Point (ROP) logic
    
    # 5. Source Package: Shared Utilities
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/common.py",
    
    # 6. Streamlit Business User Application
    "app/__init__.py",
    "app/streamlit_app.py",
    
    # 7. Data Storage Directories
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "data/predictions/.gitkeep",   # Holds batch inference outputs
    
    # 8. Models & Metrics Artifact Stores
    "models/.gitkeep",             # Serialized LightGBM booster & feature metadata
    "metrics/.gitkeep",            # Holds WAPE, RMSE, MAE evaluation JSONs
    
    # 9. SQL Migration & Mart Scripts
    "sql/01_create_database.sql",
    "sql/02_create_tables.sql",
    "sql/03_data_cleaning.sql",
    "sql/04_aggregation_views.sql",
    
    # 10. Analytical & Prototyping Notebooks
    "notebooks/01_database_smoke_test.ipynb",
    "notebooks/02_data_quality_profiling.ipynb",
    "notebooks/03_time_series_eda.ipynb",
    "notebooks/04_baseline_forecasting.ipynb",
    "notebooks/05_ml_forecasting_backtest.ipynb",
    
    # 11. Configuration & Logging
    "config/config.yaml",
    "logs/.gitkeep",
    
    # 12. Orchestration Pipeline
    "dags/retail_forecasting_dag.py",
    
    # 13. Root Operational & Environment Files
    ".env",
    ".env.example",                # Safe credentials template for GitHub
    ".gitignore",
    "requirements.txt",
    "setup.py",
    "README.md",
    "main.py"
]


def create_project_structure():
    """Scaffolds all required project directories and placeholder files."""
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        # Create parent directories if they don't exist
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Directory verified/created: {filedir}")

        # Create file if missing or empty
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            with open(filepath, "w", encoding="utf-8") as f:
                pass
            logging.info(f"Scaffolded file: {filepath}")
        else:
            logging.info(f"Existing file retained: {filepath}")


if __name__ == "__main__":
    create_project_structure()
    logging.info("Project directory structure scaffolded successfully.")