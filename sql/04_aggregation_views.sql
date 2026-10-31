-- sql/04_aggregation_views.sql
-- Purpose: Aggregate transaction-level data to daily product-level metrics.
-- Business Logic: Group clean records by stock_code and transaction_date.

USE retail_forecasting;

DROP TABLE IF EXISTS daily_product_sales;

CREATE TABLE daily_product_sales (
    stock_code VARCHAR(30) NOT NULL,
    transaction_date DATE NOT NULL,
    total_quantity INT NOT NULL,
    total_revenue DECIMAL(12, 4) NOT NULL,
    order_count INT NOT NULL,
    avg_unit_price DECIMAL(10, 4) NOT NULL,
    PRIMARY KEY (stock_code, transaction_date),
    INDEX idx_trans_date (transaction_date),
    INDEX idx_stock (stock_code)
);

-- Populate daily aggregated metrics
INSERT INTO daily_product_sales (
    stock_code,
    transaction_date,
    total_quantity,
    total_revenue,
    order_count,
    avg_unit_price
)
SELECT 
    stock_code,
    transaction_date,
    SUM(quantity) AS total_quantity,
    ROUND(SUM(revenue), 4) AS total_revenue,
    COUNT(DISTINCT invoice) AS order_count,
    ROUND(SUM(revenue) / SUM(quantity), 4) AS avg_unit_price
FROM clean_retail_transactions
GROUP BY 
    stock_code, 
    transaction_date;