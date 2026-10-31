CREATE DATABASE IF NOT EXISTS retail_forecasting;
USE retail_forecasting;

DROP TABLE IF EXISTS sales_practice;

CREATE TABLE sales_practice (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    order_date DATE NOT NULL
);

INSERT INTO sales_practice (product_name, quantity, unit_price, order_date) VALUES
('Product A', 10, 15.50, '2026-10-20'),
('Product B', 5, 25.00, '2026-10-20'),
('Product A', 7, 15.50, '2026-10-21'),
('Product C', 2, 99.00, '2026-10-21'),
('Product B', 12, 25.00, '2026-10-22');

SELECT 
    order_date,
    COUNT(order_id) AS total_orders,
    SUM(quantity) AS total_units_sold,
    ROUND(SUM(quantity * unit_price), 2) AS daily_revenue
FROM sales_practice
GROUP BY order_date
ORDER BY order_date ASC;