# 📦 Retail Demand Forecasting & Inventory Intelligence System

demo_badge = """
[![Live Demo](https://img.shields.io/badge/Live_Demo-10B981?style=for-the-badge&logo=streamlit)](https://retail-demand-forecasting-demo.streamlit.app/)

Welcome to my end-to-end applied data science project! This system is designed to bridge the gap between machine learning (demand forecasting) and stochastic supply chain operations.

Rather than just plotting historical sales on a chart, I built this system to act as an **automated decision engine** for category managers. It transforms probabilistic forecasts into exact procurement strategies, actively balancing warehouse working capital against the risk of stockouts.

## 🎯 Why I Built This (The Business Problem)

Retail supply chains suffer when relying on naive forecasting (like 30-day moving averages). Highly erratic or intermittent sales usually lead to one of two expensive outcomes: massive overstocking (tying up valuable capital) or sudden stockouts (losing out on revenue).

I built this platform to solve that exact problem by treating demand forecasting as a machine learning challenge and inventory replenishment as an operations research problem.

## 📊 Where The Data Comes From

This project is built using the [**Online Retail II Dataset**](https://archive.ics.uci.edu/dataset/502/online+retail+ii) from the UCI Machine Learning Repository.

It contains over 1 million actual historical transactional records for a UK-based registered non-store online retailer. Using real-world, messy data provided an authentic environment for me to tackle advanced data engineering, outlier handling (wholesale bulk purchases/returns), and the modeling of unpredictable demand patterns.

## 🧠 Under the Hood: Analytics & Machine Learning

* **Zero-Inflated Demand Forecasting:** Powered by a production-grade LightGBM regressor using a **Tweedie deviance objective** ($\rho = 1.5$). This handles the sparse, lumpy, and right-skewed nature of intermittent retail transactions without collapsing predictions to zero.

* **Syntetos-Boylan Demand Profiling:** The system algorithmically classifies every product in the catalog into *Smooth, Intermittent, Erratic,* or *Lumpy* time-series patterns based on Average Demand Interval (ADI) and Coefficient of Variation (CV²).

* **Dual-Variance Safety Stock:** Moving beyond static rules of thumb, the system computes dynamic safety stock buffers mathematically. It accounts for both historical demand variance ($\sigma_D$) and supplier lead time volatility ($\sigma_L$).

* **Forward-Depletion Simulation:** It simulates true day-by-day inventory consumption against the 28-day P50 forecast to predict exact stockout dates and generate precise Recommended Order Quantities (ROQ).

## 🚀 Enterprise Features & MLOps

* **Live Database Connector:** A secure SQLAlchemy ingestion pipeline fetches live transactions directly from a **MySQL** database into compressed `.parquet` files for rapid training.

* **Continuous Integration (CI):** A fully automated GitHub Actions pipeline runs `pytest` smoke tests and `flake8` linting on every push to ensure code integrity before deployment.

* **Automated Retraining (CD/MLOps):** A cron-scheduled GitHub Actions workflow automatically extracts new data, retrains the LightGBM booster, generates fresh 28-day forecasts, and commits the updated predictions back to the repository every Sunday at midnight.

* **1-Click ERP Integration:** Built-in `fpdf` document generation allows procurement teams to instantly export filtered landscape PDF Action Plans and CSV purchase orders.

* **Executive SaaS UI:** A responsive Streamlit dashboard featuring custom CSS grid KPI cards, dynamic ABC revenue segmentation, and interactive Plotly visualizations.

## 🚧 What Broke & How I Fixed It (Production Challenges)

Building an end-to-end ML system for volatile retail data required solving several structural and statistical roadblocks. Here is a transparent look at the challenges I faced and how I engineered around them:

* **Challenge 1: Intermittent Sales Broke Standard Models**
  Traditional time-series models (like ARIMA or standard XGBoost) failed on slow-moving items. They often predicted negative sales or continuous fractional units (e.g., 0.3 units/day), which completely breaks downstream inventory math.

  * **The Fix:** I transitioned the model to **LightGBM with a Tweedie Deviance objective (**$\rho = 1.5$**)**. This mathematical approach is explicitly designed to handle compound Poisson-Gamma distributions, cleanly mapping long periods of zero demand alongside sudden, massive volume spikes.

* **Challenge 2: Dashboard State-Management Loops**
  Streamlit’s top-down execution model caused infinite refresh loops and broken states when I tried to dynamically sync the catalog dropdown with the executive UI radio buttons.

  * **The Fix:** I decoupled the logic and built a custom dynamic evaluator using `st.session_state` and Streamlit callbacks (`on_change`). The app now evaluates Syntetos-Boylan demand profiles on the fly and snaps to the correct risk profile smoothly without trapping the user in an execution loop.

* **Challenge 3: Brittle MLOps Execution**
  Initially, running 6 separate modular scripts (`ingest.py`, `train.py`, etc.) via GitHub Actions was prone to cascading failures. If database ingestion failed, the ML training script would still try to run on stale data and crash the pipeline.

  * **The Fix:** I engineered a centralized **`main.py` pipeline orchestrator** using `subprocess` and `argparse`. This ensures strict sequential execution, centralized error logging, and allows for conditional flow control (like passing a `--predict-only` flag to skip heavy model retraining during daily operational cron runs).

## 📂 Project Architecture

```
├── .github/workflows/       # CI/CD and MLOps automation pipelines
├── app/                     # Streamlit frontend and UI views
├── src/
│   ├── data/                # SQL extraction and Parquet serialization
│   ├── database/            # Secure SQLAlchemy connection layer
│   ├── features/            # Feature engineering and rolling aggregations
│   ├── models/              # LightGBM training, inference, and inventory math
│   └── utils/               # Logging and shared helpers
├── data/                    # Local storage for raw and predicted artifacts
├── metrics/                 # JSON evaluation scorecards (WAPE, RMSE, MAE)
├── notebooks/               # Jupyter notebooks for EDA and algorithmic prototyping
├── tests/                   # Pytest smoke tests for pipeline integrity
├── main.py                  # Central ML pipeline orchestrator and execution entry point
├── requirements.txt         # Package dependencies
└── setup.py                 # Local package configuration

```

## 📓 Research & Prototyping (Jupyter Notebooks)

While the `src/` folder contains the modular, production-ready pipeline, the `notebooks/` directory houses the rigorous Research & Development phase.

If you are exploring my codebase for the first time, I highly recommend walking through the notebooks in this exact order to understand the statistical rationale behind the system:

1. **`01_database_smoke_test.ipynb`**: Validates the SQLAlchemy connection to the live MySQL database and runs initial SQL sanity checks.

2. **`02_data_quality_profiling.ipynb`**: Investigates missing values, flags negative quantities (returns/damages), and handles extreme wholesale outliers.

3. **`03_time_series_eda.ipynb`**: Deep dive into seasonality, temporal trends, and the application of the Syntetos-Boylan classification (ADI/CV²) to categorize SKU volatility.

4. **`04_baseline_forecasting.ipynb`**: Establishes standard industry heuristics (28-day rolling means, seasonal naive) to serve as benchmarks.

5. **`05_ml_forecasting_backtest.ipynb`**: Prototyping the LightGBM Tweedie model, tuning hyper-parameters, evaluating SHAP feature importance, and proving the WAPE reduction before refactoring it into the production scripts.

## 🛠️ Getting Started (Local Setup)

Want to run the production system locally? Follow these steps to set up the environment, connect your MySQL database, and launch the dashboard.

**1. Clone the repository & set up the environment**

```
git clone https://github.com/yourusername/retail-demand-forecasting.git
cd retail-demand-forecasting

# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate

# Install dependencies and the local package
pip install -r requirements.txt
pip install -e .

```

**2. Configure the Database Connection (MySQL)**
The system requires a live MySQL database connection. For security, never hardcode your credentials!

* Rename the provided `.env.example` file to `.env`.

* Update the `.env` file with your actual MySQL database credentials:

```
DB_DIALECT=mysql
DB_USER=your_db_username
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=retail_forecasting_db

```

**3. Execute the Machine Learning Pipeline (`main.py`)**
Rather than manually executing individual modules, this project uses `main.py` as a central orchestrator. This design provides single-command execution, centralized error handling, and dynamic argument parsing.

Run the full end-to-end pipeline (data ingestion, feature engineering, LightGBM training, evaluation, and stochastic inventory math) with one command:

```
python main.py

```


**4. Launch the Application**
Once the pipeline has generated predictions in the `data/predictions/` folder, you can start the interactive UI:

```
streamlit run app/streamlit_app.py

```

## 👨‍💻 About Me

Developed with ❤️ by **Rohit Chhotulal Patil**

I built this project to demonstrate how data science can move beyond theoretical analysis and standard model building. Instead of merely forecasting sales on a Jupyter notebook, I wanted to prove how machine learning can be fully operationalized to drive real, production-grade business value.

* **Role:**- Data Scientist

* **Email:** [work.patilrohit@gmail.com](mailto:work.patilrohit@gmail.com)

* **LinkedIn:** [Rohit Patil](https://www.linkedin.com/in/rohit-patil-646074361)