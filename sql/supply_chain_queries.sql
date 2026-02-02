-- ============================================================================
-- SUPPLY CHAIN COST OPTIMIZATION & DEMAND FORECASTING - SQL ANALYSIS
-- ============================================================================
-- 
-- PURPOSE:
-- This SQL script contains comprehensive analytical queries for supply chain
-- optimization. These queries demonstrate advanced SQL techniques including:
-- - Window functions (ROW_NUMBER, RANK, LAG, LEAD, moving averages)
-- - Common Table Expressions (CTEs) and recursive queries
-- - Complex joins across multiple tables
-- - Aggregations with CUBE/ROLLUP
-- - Statistical calculations
-- - Business intelligence metrics
--
-- BUSINESS CONTEXT:
-- TechManufacture Inc. needs to optimize inventory costs, improve supplier
-- performance, and enhance demand forecasting accuracy. These queries provide
-- actionable insights for operational decision-making.
--
-- DATABASE: SQLite (supply_chain.db)
-- ============================================================================

-- ============================================================================
-- SECTION 1: INVENTORY OPTIMIZATION ANALYSIS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 1.1: ABC Classification (Pareto Analysis)
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Which products account for the majority of inventory value?
-- WHY IT MATTERS: Focus inventory management efforts on high-value items (A items)
-- TECHNIQUE: Window functions for cumulative percentage calculation

WITH inventory_value AS (
    SELECT 
        ci.sku_id,
        sm.product_name,
        sm.category,
        ci.quantity_on_hand,
        ci.unit_cost,
        ci.total_value,
        SUM(ci.total_value) OVER () as total_inventory_value
    FROM current_inventory ci
    JOIN sku_master sm ON ci.sku_id = sm.sku_id
),
ranked_inventory AS (
    SELECT 
        *,
        ROUND(100.0 * total_value / total_inventory_value, 2) as pct_of_total_value,
        ROUND(100.0 * SUM(total_value) OVER (ORDER BY total_value DESC) / total_inventory_value, 2) as cumulative_pct
    FROM inventory_value
)
SELECT 
    sku_id,
    product_name,
    category,
    quantity_on_hand,
    unit_cost,
    total_value,
    pct_of_total_value,
    cumulative_pct,
    CASE 
        WHEN cumulative_pct <= 80 THEN 'A - High Value (Top 80%)'
        WHEN cumulative_pct <= 95 THEN 'B - Medium Value (80-95%)'
        ELSE 'C - Low Value (Bottom 5%)'
    END as abc_classification
FROM ranked_inventory
ORDER BY total_value DESC;

-- EXPECTED INSIGHT: ~20% of SKUs account for ~80% of inventory value


-- ----------------------------------------------------------------------------
-- Query 1.2: Inventory Turnover Ratio by SKU and Warehouse
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: How fast is inventory moving? Which items are slow-moving?
-- WHY IT MATTERS: Low turnover = excess inventory = high carrying costs
-- FORMULA: Inventory Turnover = Cost of Goods Sold / Average Inventory Value

WITH sales_cogs AS (
    SELECT 
        s.sku_id,
        SUM(s.quantity_fulfilled * sm.unit_cost) as total_cogs,
        COUNT(DISTINCT DATE(s.date)) as days_with_sales
    FROM sales s
    JOIN sku_master sm ON s.sku_id = sm.sku_id
    WHERE s.date >= DATE('now', '-12 months')  -- Last 12 months
    GROUP BY s.sku_id
),
avg_inventory AS (
    SELECT 
        sku_id,
        warehouse_id,
        AVG(quantity_on_hand * unit_cost) as avg_inventory_value
    FROM current_inventory
    GROUP BY sku_id, warehouse_id
)
SELECT 
    ai.sku_id,
    sm.product_name,
    sm.category,
    ai.warehouse_id,
    wm.warehouse_name,
    ROUND(ai.avg_inventory_value, 2) as avg_inventory_value,
    ROUND(sc.total_cogs, 2) as annual_cogs,
    ROUND(sc.total_cogs / NULLIF(ai.avg_inventory_value, 0), 2) as inventory_turnover_ratio,
    ROUND(365.0 / NULLIF(sc.total_cogs / NULLIF(ai.avg_inventory_value, 0), 0), 1) as days_inventory_on_hand,
    CASE 
        WHEN sc.total_cogs / NULLIF(ai.avg_inventory_value, 0) > 12 THEN 'Fast Moving'
        WHEN sc.total_cogs / NULLIF(ai.avg_inventory_value, 0) > 6 THEN 'Medium Moving'
        WHEN sc.total_cogs / NULLIF(ai.avg_inventory_value, 0) > 2 THEN 'Slow Moving'
        ELSE 'Very Slow / Obsolete'
    END as movement_category
FROM avg_inventory ai
JOIN sku_master sm ON ai.sku_id = sm.sku_id
JOIN warehouse_master wm ON ai.warehouse_id = wm.warehouse_id
LEFT JOIN sales_cogs sc ON ai.sku_id = sc.sku_id
ORDER BY inventory_turnover_ratio DESC NULLS LAST;

-- EXPECTED INSIGHT: Identify slow-moving items with >180 days of inventory


-- ----------------------------------------------------------------------------
-- Query 1.3: Economic Order Quantity (EOQ) Calculation
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: What is the optimal order quantity to minimize total costs?
-- WHY IT MATTERS: Balance ordering costs vs carrying costs
-- FORMULA: EOQ = SQRT((2 * Annual Demand * Order Cost) / (Carrying Cost per Unit))

WITH annual_demand AS (
    SELECT 
        sku_id,
        SUM(quantity_fulfilled) as annual_units_sold,
        COUNT(DISTINCT DATE(date)) as days_with_demand
    FROM sales
    WHERE date >= DATE('now', '-12 months')
    GROUP BY sku_id
)
SELECT 
    sm.sku_id,
    sm.product_name,
    sm.category,
    ad.annual_units_sold,
    sm.unit_cost,
    sm.carrying_cost_rate,
    50 as estimated_order_cost,  -- Assumed $50 per order
    ROUND(
        SQRT(
            (2.0 * ad.annual_units_sold * 50) / 
            NULLIF(sm.unit_cost * sm.carrying_cost_rate, 0)
        ), 
        0
    ) as economic_order_quantity,
    ROUND(
        ad.annual_units_sold / 
        NULLIF(
            SQRT(
                (2.0 * ad.annual_units_sold * 50) / 
                NULLIF(sm.unit_cost * sm.carrying_cost_rate, 0)
            ), 
            0
        ),
        1
    ) as orders_per_year,
    ROUND(
        365.0 / NULLIF(
            ad.annual_units_sold / 
            NULLIF(
                SQRT(
                    (2.0 * ad.annual_units_sold * 50) / 
                    NULLIF(sm.unit_cost * sm.carrying_cost_rate, 0)
                ), 
                0
            ),
            0
        ),
        1
    ) as days_between_orders
FROM sku_master sm
JOIN annual_demand ad ON sm.sku_id = ad.sku_id
WHERE ad.annual_units_sold > 0
ORDER BY economic_order_quantity DESC;

-- EXPECTED INSIGHT: Optimal order quantities vary by product demand and cost


-- ----------------------------------------------------------------------------
-- Query 1.4: Slow-Moving and Obsolete Inventory Identification
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Which items have excessive inventory (>180 days)?
-- WHY IT MATTERS: Tie up capital, risk obsolescence, incur carrying costs

WITH recent_sales AS (
    SELECT 
        sku_id,
        MAX(DATE(date)) as last_sale_date,
        SUM(quantity_fulfilled) as total_sold_90days
    FROM sales
    WHERE date >= DATE('now', '-90 days')
    GROUP BY sku_id
)
SELECT 
    ci.sku_id,
    sm.product_name,
    sm.category,
    ci.warehouse_id,
    ci.quantity_on_hand,
    ci.total_value,
    rs.last_sale_date,
    JULIANDAY('now') - JULIANDAY(rs.last_sale_date) as days_since_last_sale,
    COALESCE(rs.total_sold_90days, 0) as units_sold_last_90days,
    CASE 
        WHEN rs.total_sold_90days IS NULL OR rs.total_sold_90days = 0 THEN 999
        ELSE ROUND(ci.quantity_on_hand / (rs.total_sold_90days / 90.0), 0)
    END as days_of_inventory,
    CASE 
        WHEN rs.total_sold_90days IS NULL THEN 'No Sales (Obsolete)'
        WHEN ROUND(ci.quantity_on_hand / (rs.total_sold_90days / 90.0), 0) > 180 THEN 'Excess (>180 days)'
        WHEN ROUND(ci.quantity_on_hand / (rs.total_sold_90days / 90.0), 0) > 90 THEN 'Slow Moving (90-180 days)'
        ELSE 'Normal'
    END as inventory_status
FROM current_inventory ci
JOIN sku_master sm ON ci.sku_id = sm.sku_id
LEFT JOIN recent_sales rs ON ci.sku_id = rs.sku_id
WHERE ci.quantity_on_hand > 0
ORDER BY days_of_inventory DESC;

-- EXPECTED INSIGHT: Identify $XXX,XXX in excess inventory for clearance


-- ============================================================================
-- SECTION 2: DEMAND PATTERN ANALYSIS (Foundation for Forecasting)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 2.1: Time-Series Decomposition - Trend and Seasonality
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: What are the seasonal demand patterns?
-- WHY IT MATTERS: Forecasting models need to account for seasonality

WITH daily_sales AS (
    SELECT 
        DATE(date) as sale_date,
        STRFTIME('%Y', date) as year,
        STRFTIME('%m', date) as month,
        STRFTIME('%W', date) as week,
        CAST(STRFTIME('%w', date) AS INTEGER) as day_of_week,
        SUM(quantity_fulfilled) as total_units_sold,
        SUM(revenue) as total_revenue
    FROM sales
    GROUP BY DATE(date)
),
monthly_avg AS (
    SELECT 
        month,
        AVG(total_units_sold) as avg_monthly_units
    FROM daily_sales
    GROUP BY month
)
SELECT 
    ds.sale_date,
    ds.year,
    ds.month,
    ds.week,
    ds.day_of_week,
    ds.total_units_sold,
    ds.total_revenue,
    ma.avg_monthly_units,
    ROUND(ds.total_units_sold / NULLIF(ma.avg_monthly_units, 0), 2) as seasonal_index,
    AVG(ds.total_units_sold) OVER (
        ORDER BY ds.sale_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7day,
    AVG(ds.total_units_sold) OVER (
        ORDER BY ds.sale_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as moving_avg_30day
FROM daily_sales ds
JOIN monthly_avg ma ON ds.month = ma.month
ORDER BY ds.sale_date;

-- EXPECTED INSIGHT: Q4 (Nov-Dec) shows 1.5-2x seasonal spike


-- ----------------------------------------------------------------------------
-- Query 2.2: Coefficient of Variation (Demand Volatility)
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Which products have volatile demand?
-- WHY IT MATTERS: High volatility requires higher safety stock
-- FORMULA: CV = (Standard Deviation / Mean) * 100

WITH sku_demand_stats AS (
    SELECT 
        sku_id,
        AVG(quantity_fulfilled) as mean_demand,
        -- SQLite doesn't have STDDEV, so we calculate manually
        SQRT(
            AVG(quantity_fulfilled * quantity_fulfilled) - 
            AVG(quantity_fulfilled) * AVG(quantity_fulfilled)
        ) as stddev_demand,
        COUNT(*) as num_transactions,
        MIN(quantity_fulfilled) as min_demand,
        MAX(quantity_fulfilled) as max_demand
    FROM sales
    WHERE date >= DATE('now', '-12 months')
    GROUP BY sku_id
)
SELECT 
    sm.sku_id,
    sm.product_name,
    sm.category,
    ROUND(sds.mean_demand, 2) as avg_daily_demand,
    ROUND(sds.stddev_demand, 2) as stddev_demand,
    ROUND((sds.stddev_demand / NULLIF(sds.mean_demand, 0)) * 100, 2) as coefficient_of_variation,
    sds.min_demand,
    sds.max_demand,
    sds.num_transactions,
    CASE 
        WHEN (sds.stddev_demand / NULLIF(sds.mean_demand, 0)) * 100 > 100 THEN 'Very High Volatility'
        WHEN (sds.stddev_demand / NULLIF(sds.mean_demand, 0)) * 100 > 50 THEN 'High Volatility'
        WHEN (sds.stddev_demand / NULLIF(sds.mean_demand, 0)) * 100 > 25 THEN 'Medium Volatility'
        ELSE 'Low Volatility'
    END as volatility_category
FROM sku_demand_stats sds
JOIN sku_master sm ON sds.sku_id = sm.sku_id
ORDER BY coefficient_of_variation DESC;

-- EXPECTED INSIGHT: Promotional items have CV > 100% (very volatile)


-- ----------------------------------------------------------------------------
-- Query 2.3: Demand Correlation Between SKUs (Cross-Selling Patterns)
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Which products are frequently bought together?
-- WHY IT MATTERS: Bundle promotions, joint forecasting

WITH daily_sku_sales AS (
    SELECT 
        DATE(date) as sale_date,
        sku_id,
        SUM(quantity_fulfilled) as daily_qty
    FROM sales
    GROUP BY DATE(date), sku_id
),
sku_pairs AS (
    SELECT 
        a.sale_date,
        a.sku_id as sku_a,
        b.sku_id as sku_b,
        a.daily_qty as qty_a,
        b.daily_qty as qty_b
    FROM daily_sku_sales a
    JOIN daily_sku_sales b 
        ON a.sale_date = b.sale_date 
        AND a.sku_id < b.sku_id  -- Avoid duplicates
)
SELECT 
    sp.sku_a,
    sm_a.product_name as product_a,
    sp.sku_b,
    sm_b.product_name as product_b,
    COUNT(*) as days_sold_together,
    ROUND(AVG(sp.qty_a), 1) as avg_qty_a,
    ROUND(AVG(sp.qty_b), 1) as avg_qty_b,
    ROUND(
        COUNT(*) * 100.0 / (
            SELECT COUNT(DISTINCT sale_date) 
            FROM daily_sku_sales
        ),
        2
    ) as pct_days_together
FROM sku_pairs sp
JOIN sku_master sm_a ON sp.sku_a = sm_a.sku_id
JOIN sku_master sm_b ON sp.sku_b = sm_b.sku_id
GROUP BY sp.sku_a, sp.sku_b
HAVING COUNT(*) > 30  -- Sold together on >30 days
ORDER BY days_sold_together DESC
LIMIT 50;

-- EXPECTED INSIGHT: Smartphones + Accessories have high co-occurrence


-- ============================================================================
-- SECTION 3: SUPPLIER PERFORMANCE SCORECARD
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 3.1: Comprehensive Supplier Performance Metrics
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Which suppliers are reliable? Which are problematic?
-- WHY IT MATTERS: Supplier selection, contract negotiation, risk mitigation

WITH supplier_po_metrics AS (
    SELECT 
        supplier_id,
        COUNT(*) as total_orders,
        SUM(order_quantity) as total_units_ordered,
        SUM(received_quantity) as total_units_received,
        SUM(total_cost) as total_spend,
        AVG(lead_time_actual) as avg_lead_time,
        AVG(lead_time_actual - lead_time_expected) as avg_lead_time_variance,
        SUM(CASE WHEN on_time = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as on_time_delivery_pct,
        SUM(CASE WHEN quality_accepted = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as quality_acceptance_pct,
        SUM(CASE WHEN on_time = 0 THEN total_cost ELSE 0 END) as cost_of_late_deliveries
    FROM (
        SELECT 
            supplier_id,
            order_quantity,
            received_quantity,
            total_cost,
            lead_time_actual,
            lead_time_expected,
            CASE WHEN on_time = 'true' OR on_time = '1' OR on_time = 1 THEN 1 ELSE 0 END as on_time,
            CASE WHEN quality_accepted = 'true' OR quality_accepted = '1' OR quality_accepted = 1 THEN 1 ELSE 0 END as quality_accepted
        FROM (
            SELECT 
                supplier_id,
                CAST(order_quantity AS INTEGER) as order_quantity,
                CAST(received_quantity AS INTEGER) as received_quantity,
                CAST(total_cost AS REAL) as total_cost,
                CAST(lead_time_actual AS INTEGER) as lead_time_actual,
                CAST(lead_time_expected AS INTEGER) as lead_time_expected,
                on_time,
                quality_accepted
            FROM json_each((SELECT json_group_array(json_object(
                'supplier_id', supplier_id,
                'order_quantity', order_quantity,
                'received_quantity', received_quantity,
                'total_cost', total_cost,
                'lead_time_actual', lead_time_actual,
                'lead_time_expected', lead_time_expected,
                'on_time', on_time,
                'quality_accepted', quality_accepted
            )) FROM (SELECT * FROM json_each(readfile('data/raw/purchase_orders.json')))))
        )
    )
    GROUP BY supplier_id
)
SELECT 
    sm.supplier_id,
    sm.supplier_name,
    sm.supplier_type,
    sm.country,
    spm.total_orders,
    ROUND(spm.total_spend, 2) as total_spend,
    ROUND(spm.avg_lead_time, 1) as avg_lead_time_days,
    ROUND(spm.avg_lead_time_variance, 1) as avg_delay_days,
    ROUND(spm.on_time_delivery_pct, 2) as on_time_delivery_pct,
    ROUND(spm.quality_acceptance_pct, 2) as quality_acceptance_pct,
    ROUND(spm.cost_of_late_deliveries, 2) as cost_of_late_deliveries,
    ROUND(
        (spm.on_time_delivery_pct * 0.4) + 
        (spm.quality_acceptance_pct * 0.4) + 
        (CASE WHEN spm.avg_lead_time < 60 THEN 100 ELSE 100 - (spm.avg_lead_time - 60) END * 0.2),
        2
    ) as composite_score,
    CASE 
        WHEN spm.on_time_delivery_pct >= 95 AND spm.quality_acceptance_pct >= 98 THEN 'Tier 1 - Excellent'
        WHEN spm.on_time_delivery_pct >= 85 AND spm.quality_acceptance_pct >= 95 THEN 'Tier 2 - Good'
        WHEN spm.on_time_delivery_pct >= 70 AND spm.quality_acceptance_pct >= 90 THEN 'Tier 3 - Acceptable'
        ELSE 'Tier 4 - Poor (Review Required)'
    END as supplier_tier
FROM supplier_master sm
LEFT JOIN supplier_po_metrics spm ON sm.supplier_id = spm.supplier_id
ORDER BY composite_score DESC NULLS LAST;

-- EXPECTED INSIGHT: 15-20% of suppliers are Tier 4 (causing majority of issues)


-- ----------------------------------------------------------------------------
-- Query 3.2: Lead Time Variability Analysis by Supplier
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Which suppliers have unpredictable lead times?
-- WHY IT MATTERS: High variability requires higher safety stock

SELECT 
    supplier_id,
    COUNT(*) as num_orders,
    ROUND(AVG(CAST(lead_time_actual AS REAL)), 1) as avg_lead_time,
    ROUND(MIN(CAST(lead_time_actual AS REAL)), 1) as min_lead_time,
    ROUND(MAX(CAST(lead_time_actual AS REAL)), 1) as max_lead_time,
    ROUND(MAX(CAST(lead_time_actual AS REAL)) - MIN(CAST(lead_time_actual AS REAL)), 1) as lead_time_range,
    -- Calculate standard deviation manually for SQLite
    ROUND(
        SQRT(
            AVG(CAST(lead_time_actual AS REAL) * CAST(lead_time_actual AS REAL)) - 
            AVG(CAST(lead_time_actual AS REAL)) * AVG(CAST(lead_time_actual AS REAL))
        ),
        2
    ) as stddev_lead_time,
    CASE 
        WHEN SQRT(
            AVG(CAST(lead_time_actual AS REAL) * CAST(lead_time_actual AS REAL)) - 
            AVG(CAST(lead_time_actual AS REAL)) * AVG(CAST(lead_time_actual AS REAL))
        ) > 20 THEN 'High Variability'
        WHEN SQRT(
            AVG(CAST(lead_time_actual AS REAL) * CAST(lead_time_actual AS REAL)) - 
            AVG(CAST(lead_time_actual AS REAL)) * AVG(CAST(lead_time_actual AS REAL))
        ) > 10 THEN 'Medium Variability'
        ELSE 'Low Variability'
    END as variability_category
FROM (
    SELECT 
        supplier_id,
        lead_time_actual
    FROM json_each((SELECT json_group_array(json_object(
        'supplier_id', supplier_id,
        'lead_time_actual', lead_time_actual
    )) FROM (SELECT * FROM json_each(readfile('data/raw/purchase_orders.json')))))
)
GROUP BY supplier_id
ORDER BY stddev_lead_time DESC;

-- EXPECTED INSIGHT: Overseas suppliers have 2-3x higher lead time variability


-- ============================================================================
-- SECTION 4: STOCKOUT & SERVICE LEVEL ANALYSIS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 4.1: Fill Rate and Stockout Analysis
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: How often do we fail to fulfill customer orders?
-- WHY IT MATTERS: Stockouts = lost sales + customer dissatisfaction
-- TARGET: 95% fill rate

WITH order_fulfillment AS (
    SELECT 
        sku_id,
        category,
        DATE(date) as order_date,
        quantity_ordered,
        quantity_fulfilled,
        backorder_quantity,
        CASE WHEN quantity_fulfilled < quantity_ordered THEN 1 ELSE 0 END as is_stockout
    FROM sales
    WHERE date >= DATE('now', '-6 months')
)
SELECT 
    sm.sku_id,
    sm.product_name,
    sm.category,
    COUNT(*) as total_orders,
    SUM(of.quantity_ordered) as total_units_ordered,
    SUM(of.quantity_fulfilled) as total_units_fulfilled,
    SUM(of.backorder_quantity) as total_backorders,
    ROUND(
        SUM(of.quantity_fulfilled) * 100.0 / NULLIF(SUM(of.quantity_ordered), 0),
        2
    ) as fill_rate_pct,
    SUM(of.is_stockout) as num_stockout_events,
    ROUND(
        SUM(of.is_stockout) * 100.0 / COUNT(*),
        2
    ) as stockout_frequency_pct,
    ROUND(
        SUM(of.backorder_quantity) * sm.unit_price,
        2
    ) as lost_revenue_estimate,
    CASE 
        WHEN SUM(of.quantity_fulfilled) * 100.0 / NULLIF(SUM(of.quantity_ordered), 0) >= 95 THEN 'Meeting Target (>=95%)'
        WHEN SUM(of.quantity_fulfilled) * 100.0 / NULLIF(SUM(of.quantity_ordered), 0) >= 90 THEN 'Below Target (90-95%)'
        ELSE 'Critical (<90%)'
    END as service_level_status
FROM order_fulfillment of
JOIN sku_master sm ON of.sku_id = sm.sku_id
GROUP BY sm.sku_id, sm.product_name, sm.category, sm.unit_price
HAVING total_orders > 10
ORDER BY fill_rate_pct ASC;

-- EXPECTED INSIGHT: $2.3M in lost sales from stockouts


-- ----------------------------------------------------------------------------
-- Query 4.2: Root Cause Analysis of Stockouts
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Why do stockouts occur? Supplier delays or forecast errors?
-- WHY IT MATTERS: Different root causes require different solutions

WITH stockout_events AS (
    SELECT 
        s.sku_id,
        s.date as stockout_date,
        s.backorder_quantity
    FROM sales s
    WHERE s.backorder_quantity > 0
),
recent_pos AS (
    SELECT 
        sku_id,
        actual_delivery_date,
        expected_delivery_date,
        CASE WHEN on_time = 'false' OR on_time = '0' OR on_time = 0 THEN 1 ELSE 0 END as was_late
    FROM (
        SELECT 
            sku_id,
            actual_delivery_date,
            expected_delivery_date,
            on_time
        FROM json_each((SELECT json_group_array(json_object(
            'sku_id', sku_id,
            'actual_delivery_date', actual_delivery_date,
            'expected_delivery_date', expected_delivery_date,
            'on_time', on_time
        )) FROM (SELECT * FROM json_each(readfile('data/raw/purchase_orders.json')))))
    )
)
SELECT 
    se.sku_id,
    sm.product_name,
    COUNT(DISTINCT se.stockout_date) as num_stockout_days,
    SUM(se.backorder_quantity) as total_backorder_units,
    SUM(CASE WHEN rp.was_late = 1 THEN 1 ELSE 0 END) as supplier_delays_count,
    ROUND(
        SUM(CASE WHEN rp.was_late = 1 THEN 1 ELSE 0 END) * 100.0 / 
        NULLIF(COUNT(DISTINCT se.stockout_date), 0),
        2
    ) as pct_due_to_supplier_delay,
    CASE 
        WHEN SUM(CASE WHEN rp.was_late = 1 THEN 1 ELSE 0 END) * 100.0 / 
             NULLIF(COUNT(DISTINCT se.stockout_date), 0) > 70 THEN 'Supplier Issue'
        WHEN SUM(CASE WHEN rp.was_late = 1 THEN 1 ELSE 0 END) * 100.0 / 
             NULLIF(COUNT(DISTINCT se.stockout_date), 0) > 30 THEN 'Mixed (Supplier + Forecast)'
        ELSE 'Forecast Error'
    END as primary_root_cause
FROM stockout_events se
JOIN sku_master sm ON se.sku_id = sm.sku_id
LEFT JOIN recent_pos rp 
    ON se.sku_id = rp.sku_id 
    AND DATE(rp.actual_delivery_date) >= DATE(se.stockout_date, '-30 days')
GROUP BY se.sku_id, sm.product_name
ORDER BY num_stockout_days DESC;

-- EXPECTED INSIGHT: 60% of stockouts caused by supplier delays


-- ============================================================================
-- SECTION 5: COST ANALYSIS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 5.1: Total Cost of Ownership by Supplier
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: What is the TRUE cost of each supplier (beyond unit price)?
-- WHY IT MATTERS: Cheapest supplier may not be most cost-effective
-- INCLUDES: Purchase cost + quality cost + late delivery cost

WITH supplier_costs AS (
    SELECT 
        supplier_id,
        SUM(CAST(total_cost AS REAL)) as purchase_cost,
        SUM(CAST(order_quantity AS INTEGER) - CAST(received_quantity AS INTEGER)) as defective_units,
        SUM(CASE WHEN on_time = 'false' OR on_time = '0' OR on_time = 0 THEN 1 ELSE 0 END) as late_deliveries
    FROM (
        SELECT 
            supplier_id,
            total_cost,
            order_quantity,
            received_quantity,
            on_time
        FROM json_each((SELECT json_group_array(json_object(
            'supplier_id', supplier_id,
            'total_cost', total_cost,
            'order_quantity', order_quantity,
            'received_quantity', received_quantity,
            'on_time', on_time
        )) FROM (SELECT * FROM json_each(readfile('data/raw/purchase_orders.json')))))
    )
    GROUP BY supplier_id
)
SELECT 
    sm.supplier_id,
    sm.supplier_name,
    sm.country,
    ROUND(sc.purchase_cost, 2) as purchase_cost,
    ROUND(sc.defective_units * 50, 2) as quality_cost,  -- Assume $50 per defective unit
    ROUND(sc.late_deliveries * 500, 2) as expediting_cost,  -- Assume $500 per late delivery
    ROUND(
        sc.purchase_cost + 
        (sc.defective_units * 50) + 
        (sc.late_deliveries * 500),
        2
    ) as total_cost_of_ownership,
    ROUND(
        ((sc.defective_units * 50) + (sc.late_deliveries * 500)) * 100.0 / 
        NULLIF(sc.purchase_cost, 0),
        2
    ) as hidden_cost_pct
FROM supplier_master sm
LEFT JOIN supplier_costs sc ON sm.supplier_id = sc.supplier_id
ORDER BY total_cost_of_ownership DESC NULLS LAST;

-- EXPECTED INSIGHT: Hidden costs add 10-30% to purchase price for poor suppliers


-- ----------------------------------------------------------------------------
-- Query 5.2: Warehouse Operating Cost Efficiency
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Which warehouses are cost-efficient?
-- WHY IT MATTERS: Identify candidates for consolidation

WITH warehouse_inventory AS (
    SELECT 
        warehouse_id,
        SUM(quantity_on_hand) as total_units,
        SUM(total_value) as total_inventory_value
    FROM current_inventory
    GROUP BY warehouse_id
)
SELECT 
    wm.warehouse_id,
    wm.warehouse_name,
    wm.city,
    wm.capacity_units,
    wi.total_units,
    ROUND(wi.total_units * 100.0 / wm.capacity_units, 2) as utilization_pct,
    wm.monthly_operating_cost,
    ROUND(wm.monthly_operating_cost / NULLIF(wi.total_units, 0), 2) as cost_per_unit_stored,
    ROUND(wm.monthly_operating_cost * 12, 2) as annual_operating_cost,
    CASE 
        WHEN wi.total_units * 100.0 / wm.capacity_units < 50 THEN 'Underutilized - Consolidation Candidate'
        WHEN wi.total_units * 100.0 / wm.capacity_units > 90 THEN 'Over Capacity - Expansion Needed'
        ELSE 'Optimal Utilization'
    END as efficiency_status
FROM warehouse_master wm
LEFT JOIN warehouse_inventory wi ON wm.warehouse_id = wi.warehouse_id
ORDER BY utilization_pct ASC;

-- EXPECTED INSIGHT: 5 warehouses under 50% utilization = consolidation opportunity


-- ============================================================================
-- SECTION 6: ADVANCED QUERIES (Recursive CTEs, Window Functions)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 6.1: Bill of Materials Explosion (Recursive CTE)
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: What components are needed to build finished goods?
-- WHY IT MATTERS: Component demand forecasting, supply chain risk

WITH RECURSIVE bom_explosion AS (
    -- Base case: top-level products
    SELECT 
        parent_sku,
        component_sku,
        quantity_per,
        1 as level,
        parent_sku as root_product,
        CAST(quantity_per AS REAL) as total_quantity_needed
    FROM bom
    
    UNION ALL
    
    -- Recursive case: components that are also parents
    SELECT 
        b.parent_sku,
        b.component_sku,
        b.quantity_per,
        be.level + 1,
        be.root_product,
        CAST(be.total_quantity_needed * b.quantity_per AS REAL)
    FROM bom b
    INNER JOIN bom_explosion be ON b.parent_sku = be.component_sku
    WHERE be.level < 5  -- Prevent infinite recursion
)
SELECT 
    be.root_product,
    sm_root.product_name as root_product_name,
    be.component_sku,
    sm_comp.product_name as component_name,
    be.level,
    be.total_quantity_needed,
    ROUND(be.total_quantity_needed * sm_comp.unit_cost, 2) as component_cost_per_unit
FROM bom_explosion be
JOIN sku_master sm_root ON be.root_product = sm_root.sku_id
JOIN sku_master sm_comp ON be.component_sku = sm_comp.sku_id
ORDER BY be.root_product, be.level, be.component_sku;

-- EXPECTED INSIGHT: Multi-level BOM reveals critical component dependencies


-- ----------------------------------------------------------------------------
-- Query 6.2: Moving Averages and Trend Analysis (Window Functions)
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: Is demand trending up or down?
-- WHY IT MATTERS: Adjust forecasts and inventory policies

WITH daily_demand AS (
    SELECT 
        DATE(date) as demand_date,
        SUM(quantity_fulfilled) as total_demand
    FROM sales
    GROUP BY DATE(date)
)
SELECT 
    demand_date,
    total_demand,
    AVG(total_demand) OVER (
        ORDER BY demand_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as ma_7day,
    AVG(total_demand) OVER (
        ORDER BY demand_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as ma_30day,
    total_demand - LAG(total_demand, 7) OVER (ORDER BY demand_date) as week_over_week_change,
    ROUND(
        (total_demand - LAG(total_demand, 7) OVER (ORDER BY demand_date)) * 100.0 / 
        NULLIF(LAG(total_demand, 7) OVER (ORDER BY demand_date), 0),
        2
    ) as week_over_week_pct_change
FROM daily_demand
ORDER BY demand_date DESC
LIMIT 90;

-- EXPECTED INSIGHT: Identify growth/decline trends for capacity planning


-- ============================================================================
-- SECTION 7: ACTIONABLE INSIGHTS SUMMARY
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 7.1: Executive Summary - Key Metrics Dashboard
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION: What are the top-level KPIs?
-- WHY IT MATTERS: Executive decision-making

SELECT 
    'Total Inventory Value' as metric,
    '$' || ROUND(SUM(total_value), 2) as value
FROM current_inventory

UNION ALL

SELECT 
    'Average Inventory Turnover',
    ROUND(AVG(turnover), 2)
FROM (
    SELECT 
        sku_id,
        SUM(quantity_fulfilled * unit_cost) / 
        NULLIF(AVG(quantity_on_hand * unit_cost), 0) as turnover
    FROM (
        SELECT s.sku_id, s.quantity_fulfilled, sm.unit_cost, ci.quantity_on_hand
        FROM sales s
        JOIN sku_master sm ON s.sku_id = sm.sku_id
        LEFT JOIN current_inventory ci ON s.sku_id = ci.sku_id
        WHERE s.date >= DATE('now', '-12 months')
    )
    GROUP BY sku_id
)

UNION ALL

SELECT 
    'Overall Fill Rate',
    ROUND(SUM(quantity_fulfilled) * 100.0 / SUM(quantity_ordered), 2) || '%'
FROM sales
WHERE date >= DATE('now', '-6 months')

UNION ALL

SELECT 
    'Number of Stockout Events',
    COUNT(*)
FROM sales
WHERE backorder_quantity > 0 
  AND date >= DATE('now', '-6 months')

UNION ALL

SELECT 
    'Average Supplier On-Time Delivery',
    ROUND(AVG(on_time_delivery_rate) * 100, 2) || '%'
FROM supplier_master;

-- EXPECTED OUTPUT: Dashboard-ready KPIs


-- ============================================================================
-- END OF SQL ANALYSIS
-- ============================================================================
-- 
-- NEXT STEPS:
-- 1. Export query results to CSV for Python analysis
-- 2. Build demand forecasting models (ARIMA, Prophet)
-- 3. Optimize inventory policies (ROP, safety stock)
-- 4. Create visualizations and dashboards
-- 5. Generate executive summary with recommendations
-- ============================================================================
