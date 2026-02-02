"""
SQL Query Execution and Results Export Module
==============================================

PURPOSE:
This module executes the comprehensive SQL queries defined in supply_chain_queries.sql
and exports the results to CSV files for further analysis and visualization.

BUSINESS VALUE:
- Automates execution of complex analytical queries
- Creates reusable datasets for Python analysis
- Enables reproducible analysis workflow
- Facilitates sharing of insights with stakeholders

HOW TO USE:
    python src/execute_sql_queries.py

OUTPUT:
- CSV files in data/processed/ directory
- Query execution log
- Summary statistics
"""

import sqlite3
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

class SQLQueryExecutor:
    """
    Executes SQL queries against the supply chain database and exports results.
    """
    
    def __init__(self, db_path: str = "data/raw/supply_chain.db", output_dir: str = "data/processed"):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.execution_log = []
        
    def connect(self):
        """Establish database connection."""
        print(f"\n[DATABASE] Connecting to: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        print("[OK] Connection established")
        
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("\n[DATABASE] Connection closed")
            
    def execute_query(self, query_name: str, query_sql: str, export_csv: bool = True):
        """
        Execute a single SQL query and optionally export to CSV.
        
        Args:
            query_name: Descriptive name for the query
            query_sql: SQL query string
            export_csv: Whether to export results to CSV
            
        Returns:
            DataFrame with query results
        """
        print(f"\n[QUERY] Executing: {query_name}")
        start_time = time.time()
        
        try:
            # Execute query
            df = pd.read_sql_query(query_sql, self.conn)
            execution_time = time.time() - start_time
            
            # Log execution
            self.execution_log.append({
                'query_name': query_name,
                'rows_returned': len(df),
                'columns': len(df.columns),
                'execution_time_sec': round(execution_time, 2),
                'status': 'SUCCESS'
            })
            
            print(f"   [OK] Returned {len(df)} rows, {len(df.columns)} columns in {execution_time:.2f}s")
            
            # Export to CSV
            if export_csv and len(df) > 0:
                csv_filename = f"{query_name.lower().replace(' ', '_').replace('-', '_')}.csv"
                csv_path = self.output_dir / csv_filename
                df.to_csv(csv_path, index=False)
                print(f"   [EXPORT] Saved to: {csv_path}")
                
            return df
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_log.append({
                'query_name': query_name,
                'rows_returned': 0,
                'columns': 0,
                'execution_time_sec': round(execution_time, 2),
                'status': f'ERROR: {str(e)}'
            })
            print(f"   [ERROR] {str(e)}")
            return None
            
    def run_all_queries(self):
        """
        Execute all predefined analytical queries.
        """
        print("\n" + "="*70)
        print("  EXECUTING SUPPLY CHAIN SQL ANALYSIS QUERIES")
        print("="*70)
        
        self.connect()
        
        # ====================================================================
        # SECTION 1: INVENTORY OPTIMIZATION
        # ====================================================================
        
        print("\n" + "-"*70)
        print("SECTION 1: INVENTORY OPTIMIZATION ANALYSIS")
        print("-"*70)
        
        # Query 1.1: ABC Classification
        abc_query = """
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
        """
        self.execute_query("1.1_ABC_Classification", abc_query)
        
        # Query 1.2: Inventory Turnover
        turnover_query = """
        WITH sales_cogs AS (
            SELECT 
                s.sku_id,
                SUM(s.quantity_fulfilled * sm.unit_cost) as total_cogs,
                COUNT(DISTINCT DATE(s.date)) as days_with_sales
            FROM sales s
            JOIN sku_master sm ON s.sku_id = sm.sku_id
            WHERE s.date >= DATE('now', '-12 months')
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
        """
        self.execute_query("1.2_Inventory_Turnover", turnover_query)
        
        # Query 1.3: Economic Order Quantity
        eoq_query = """
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
            50 as estimated_order_cost,
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
            ) as orders_per_year
        FROM sku_master sm
        JOIN annual_demand ad ON sm.sku_id = ad.sku_id
        WHERE ad.annual_units_sold > 0
        ORDER BY economic_order_quantity DESC;
        """
        self.execute_query("1.3_Economic_Order_Quantity", eoq_query)
        
        # Query 1.4: Slow-Moving Inventory
        slow_moving_query = """
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
        """
        self.execute_query("1.4_Slow_Moving_Inventory", slow_moving_query)
        
        # ====================================================================
        # SECTION 2: DEMAND PATTERN ANALYSIS
        # ====================================================================
        
        print("\n" + "-"*70)
        print("SECTION 2: DEMAND PATTERN ANALYSIS")
        print("-"*70)
        
        # Query 2.1: Time-Series Decomposition
        time_series_query = """
        WITH daily_sales AS (
            SELECT 
                DATE(date) as sale_date,
                STRFTIME('%Y', date) as year,
                STRFTIME('%m', date) as month,
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
        """
        self.execute_query("2.1_Time_Series_Decomposition", time_series_query)
        
        # Query 2.2: Demand Volatility
        volatility_query = """
        WITH sku_demand_stats AS (
            SELECT 
                sku_id,
                AVG(quantity_fulfilled) as mean_demand,
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
        """
        self.execute_query("2.2_Demand_Volatility", volatility_query)
        
        # ====================================================================
        # SECTION 3: SUPPLIER PERFORMANCE
        # ====================================================================
        
        print("\n" + "-"*70)
        print("SECTION 3: SUPPLIER PERFORMANCE ANALYSIS")
        print("-"*70)
        
        # Query 3.1: Supplier Scorecard (Simplified for SQLite without JSON parsing)
        supplier_query = """
        SELECT 
            sm.supplier_id,
            sm.supplier_name,
            sm.supplier_type,
            sm.country,
            ROUND(sm.on_time_delivery_rate * 100, 2) as on_time_delivery_pct,
            sm.quality_rating,
            sm.payment_terms_days,
            sm.minimum_order_quantity,
            CASE 
                WHEN sm.on_time_delivery_rate >= 0.95 AND sm.quality_rating >= 4.5 THEN 'Tier 1 - Excellent'
                WHEN sm.on_time_delivery_rate >= 0.85 AND sm.quality_rating >= 4.0 THEN 'Tier 2 - Good'
                WHEN sm.on_time_delivery_rate >= 0.70 AND sm.quality_rating >= 3.5 THEN 'Tier 3 - Acceptable'
                ELSE 'Tier 4 - Poor (Review Required)'
            END as supplier_tier
        FROM supplier_master sm
        ORDER BY sm.on_time_delivery_rate DESC, sm.quality_rating DESC;
        """
        self.execute_query("3.1_Supplier_Scorecard", supplier_query)
        
        # ====================================================================
        # SECTION 4: STOCKOUT ANALYSIS
        # ====================================================================
        
        print("\n" + "-"*70)
        print("SECTION 4: STOCKOUT & SERVICE LEVEL ANALYSIS")
        print("-"*70)
        
        # Query 4.1: Fill Rate Analysis
        fill_rate_query = """
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
        """
        self.execute_query("4.1_Fill_Rate_Analysis", fill_rate_query)
        
        # ====================================================================
        # SECTION 5: COST ANALYSIS
        # ====================================================================
        
        print("\n" + "-"*70)
        print("SECTION 5: COST ANALYSIS")
        print("-"*70)
        
        # Query 5.1: Warehouse Efficiency
        warehouse_query = """
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
        """
        self.execute_query("5.1_Warehouse_Efficiency", warehouse_query)
        
        # ====================================================================
        # SECTION 6: EXECUTIVE SUMMARY
        # ====================================================================
        
        print("\n" + "-"*70)
        print("SECTION 6: EXECUTIVE SUMMARY KPIs")
        print("-"*70)
        
        # Query 6.1: Key Metrics Dashboard
        kpi_query = """
        SELECT 
            'Total Inventory Value' as metric,
            '$' || ROUND(SUM(total_value), 2) as value
        FROM current_inventory

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
        """
        self.execute_query("6.1_Executive_KPIs", kpi_query)
        
        # ====================================================================
        # FINALIZE
        # ====================================================================
        
        self.disconnect()
        self.save_execution_log()
        self.print_summary()
        
    def save_execution_log(self):
        """Save query execution log to CSV."""
        log_df = pd.DataFrame(self.execution_log)
        log_path = self.output_dir / 'query_execution_log.csv'
        log_df.to_csv(log_path, index=False)
        print(f"\n[LOG] Execution log saved to: {log_path}")
        
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*70)
        print("  EXECUTION SUMMARY")
        print("="*70)
        
        log_df = pd.DataFrame(self.execution_log)
        
        total_queries = len(log_df)
        successful = len(log_df[log_df['status'] == 'SUCCESS'])
        failed = total_queries - successful
        total_rows = log_df['rows_returned'].sum()
        total_time = log_df['execution_time_sec'].sum()
        
        print(f"\nTotal Queries Executed: {total_queries}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total Rows Returned: {total_rows:,}")
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"\nResults exported to: {self.output_dir.absolute()}")
        print("\n[CHECK] SQL analysis complete!")
        print("="*70 + "\n")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("  SUPPLY CHAIN SQL ANALYSIS - QUERY EXECUTION")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    executor = SQLQueryExecutor()
    executor.run_all_queries()
    
    print("\n[TARGET] Next Steps:")
    print("   1. Review CSV files in data/processed/")
    print("   2. Run exploratory data analysis (src/exploratory_analysis.py)")
    print("   3. Build demand forecasting models (src/demand_forecasting.py)")
    print("   4. Create visualizations and dashboards")
    print("\n")


if __name__ == "__main__":
    main()
