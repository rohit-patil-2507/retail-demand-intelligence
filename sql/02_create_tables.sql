-- ====================================================================
-- SCRIPT 02: RAW STAGING TABLE SCHEMA
-- Project: Retail Demand Forecasting & Intelligence Platform
-- ====================================================================

USE retail_forecasting;

-- Drop existing staging table if needed for clean re-runs
DROP TABLE IF EXISTS raw_retail_transactions;

CREATE TABLE raw_retail_transactions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    invoice VARCHAR(20) NULL,
    stock_code VARCHAR(30) NULL,
    description VARCHAR(255) NULL,
    quantity INT NULL,
    invoice_date DATETIME NULL,
    price DECIMAL(10, 4) NULL,
    customer_id VARCHAR(20) NULL,
    country VARCHAR(100) NULL,
    sheet_source VARCHAR(30) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite & single indexes for fast lookups and joins
    INDEX idx_invoice (invoice),
    INDEX idx_stock_code (stock_code),
    INDEX idx_invoice_date (invoice_date),
    INDEX idx_stock_date (stock_code, invoice_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Verification of table structure
DESCRIBE raw_retail_transactions;