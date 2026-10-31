from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default settings applied to all tasks
default_args = {
    'owner': 'rohit_patil',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG (Directed Acyclic Graph)
with DAG(
    dag_id='retail_demand_mlops_pipeline',
    default_args=default_args,
    description='Weekly pipeline to extract data, retrain LightGBM, and generate inventory policies.',
    schedule_interval='0 0 * * 0', # Runs at midnight every Sunday
    start_date=datetime(2026, 10, 1),
    catchup=False, # Do not backfill past runs
    tags=['machine-learning', 'supply-chain', 'forecasting'],
) as dag:

    # Task 1: Connect to database and extract raw data
    ingest_data = BashOperator(
        task_id='extract_live_transactions',
        bash_command='python -m src.data.ingest'
    )

    # Task 2: Clean data and engineer rolling time-series features
    build_features = BashOperator(
        task_id='engineer_features',
        bash_command='python -m src.features.build_features'
    )

    # Task 3: Retrain the LightGBM Tweedie model on the latest data
    train_model = BashOperator(
        task_id='train_lightgbm_booster',
        bash_command='python -m src.models.train'
    )

    # Task 4: Benchmark model against naive heuristics on holdout set
    evaluate_model = BashOperator(
        task_id='evaluate_holdout_metrics',
        bash_command='python -m src.models.evaluate'
    )

    # Task 5: Generate 28-day forward predictions
    generate_forecast = BashOperator(
        task_id='predict_forward_demand',
        bash_command='python -m src.models.predict'
    )

    # Task 6: Calculate Safety Stock, ROP, and final procurement quantities
    optimize_inventory = BashOperator(
        task_id='calculate_inventory_policy',
        bash_command='python -m src.models.inventory'
    )

    # Define the execution sequence (Downstream Dependencies)
    ingest_data >> build_features >> train_model >> evaluate_model >> generate_forecast >> optimize_inventory