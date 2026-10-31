USE retail_forecasting;

-- ====================================================================
-- STEP 1: DATA QUALITY PROFILING (Investigating Messiness)
-- ====================================================================

-- 1.1 Total record count
SELECT COUNT(*) AS total_raw_records FROM raw_retail_transactions;

-- 1.2 Missing Customer IDs
SELECT 
    COUNT(*) AS missing_customer_id_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM raw_retail_transactions), 2) AS missing_customer_pct
FROM raw_retail_transactions
WHERE customer_id IS NULL;

-- 1.3 Invoice Cancellations (Invoices starting with 'C')
SELECT 
    COUNT(*) AS cancellation_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM raw_retail_transactions), 2) AS cancellation_pct
FROM raw_retail_transactions
WHERE invoice LIKE 'C%';

-- 1.4 Non-positive Quantities and Prices (Damaged stock, adjustments, giveaways)
SELECT 
    SUM(CASE WHEN quantity <= 0 THEN 1 ELSE 0 END) AS zero_or_negative_quantity,
    SUM(CASE WHEN price <= 0 THEN 1 ELSE 0 END) AS zero_or_negative_price
FROM raw_retail_transactions;

-- 1.5 Administrative / Non-Product StockCodes (POST, D, M, BANK CHARGES, etc.)
SELECT stock_code, description, COUNT(*) AS occurrences
FROM raw_retail_transactions
WHERE stock_code REGEXP '^[a-zA-Z]+$' OR LENGTH(stock_code) > 8
GROUP BY stock_code, description
ORDER BY occurrences DESC
LIMIT 15;


-- ====================================================================
-- STEP 2: BUILD CLEAN PRODUCTION TABLE (clean_retail_transactions)
-- ====================================================================

DROP TABLE IF EXISTS clean_retail_transactions;

CREATE TABLE clean_retail_transactions AS
SELECT 
    id AS raw_id,
    invoice,
    stock_code,
    description,
    quantity,
    invoice_date,
    DATE(invoice_date) AS order_date,
    price AS unit_price,
    ROUND(quantity * price, 2) AS line_total,
    customer_id,
    country
FROM raw_retail_transactions
WHERE 
    -- Exclude order cancellations
    invoice NOT LIKE 'C%'
    -- Require positive quantity (actual purchases)
    AND quantity > 0
    -- Require valid price (exclude administrative zero-price gifts/write-offs)
    AND price > 0
    -- Exclude non-product operational codes
    AND stock_code NOT IN ('POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'DOT', 'CRUK')
    -- Filter out descriptive manual entries
    AND description NOT LIKE 'adjust%'
    AND description NOT LIKE 'check%'
    AND description NOT LIKE 'damages%';

-- Add primary key and query indexes for downstream performance
ALTER TABLE clean_retail_transactions 
    ADD PRIMARY KEY (raw_id),
    ADD INDEX idx_order_date (order_date),
    ADD INDEX idx_clean_stock_code (stock_code);

-- ====================================================================
-- STEP 3: POST-CLEANING SANITY AUDIT
-- ====================================================================
SELECT 
    (SELECT COUNT(*) FROM raw_retail_transactions) AS raw_count,
    (SELECT COUNT(*) FROM clean_retail_transactions) AS clean_count,
    (SELECT COUNT(*) FROM raw_retail_transactions) - (SELECT COUNT(*) FROM clean_retail_transactions) AS dropped_records;