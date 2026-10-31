import argparse
import logging
import subprocess
import sys

# Configure centralized logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_step(module_name):
    """Executes a Python module just like running it from the terminal."""
    try:
        # sys.executable ensures it uses the Python from your active virtual environment (.venv)
        subprocess.run([sys.executable, "-m", module_name], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed during execution of {module_name}.")
        raise RuntimeError(f"Pipeline halted at {module_name}") from e

def run_pipeline(train_model=True):
    try:
        logging.info("Step 1: Starting Database Ingestion...")
        run_step("src.data.ingest")
        
        logging.info("Step 2: Engineering Time-Series Features...")
        run_step("src.features.build_features")
        
        if train_model:
            logging.info("Step 3: Training LightGBM Tweedie Booster...")
            run_step("src.models.train")
            
            logging.info("Step 4: Evaluating Model on Holdout Set...")
            run_step("src.models.evaluate")
            
        logging.info("Step 5: Generating 28-Day P50 Forecasts...")
        run_step("src.models.predict")
        
        logging.info("Step 6: Calculating Safety Stock & ROP...")
        run_step("src.models.inventory")
        
        logging.info("✅ Pipeline executed successfully.")
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Intelligence Pipeline Orchestrator")
    parser.add_argument("--predict-only", action="store_true", help="Skip training and only run inference")
    
    args = parser.parse_args()
    
    # If --predict-only is passed, train_model becomes False
    run_pipeline(train_model=not args.predict_only)