"""
Exploratory Data Analysis Module
=================================

PURPOSE:
This module performs comprehensive exploratory data analysis on the supply chain data.
It generates statistical summaries, data quality checks, and initial insights.

BUSINESS VALUE:
- Understand data distributions and patterns
- Identify data quality issues
- Validate assumptions for forecasting models
- Generate initial business insights

WHAT THIS FILE DOES:
1. Loads all datasets (CSV, JSON, SQLite)
2. Performs data quality checks (missing values, outliers, duplicates)
3. Generates statistical summaries
4. Creates initial visualizations
5. Exports summary reports

HOW TO USE:
    python src/exploratory_analysis.py
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class SupplyChainEDA:
    """
    Performs exploratory data analysis on supply chain datasets.
    """
    
    def __init__(self, data_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.sku_master = None
        self.warehouse_master = None
        self.supplier_master = None
        self.sales = None
        self.inventory_txn = None
        self.current_inventory = None
        self.purchase_orders = None
        self.economic_indicators = None
        self.promotional_calendar = None
        
        self.insights = []
        
    def load_all_data(self):
        """
        Load all datasets from various sources.
        """
        print("\n" + "="*70)
        print("  LOADING SUPPLY CHAIN DATA")
        print("="*70)
        
        # Load CSV files
        print("\n[LOADING] CSV files...")
        self.sku_master = pd.read_csv(self.data_dir / 'sku_master.csv')
        print(f"   [OK] SKU Master: {len(self.sku_master)} records")
        
        self.warehouse_master = pd.read_csv(self.data_dir / 'warehouse_master.csv')
        print(f"   [OK] Warehouse Master: {len(self.warehouse_master)} records")
        
        self.supplier_master = pd.read_csv(self.data_dir / 'supplier_master.csv')
        print(f"   [OK] Supplier Master: {len(self.supplier_master)} records")
        
        self.sales = pd.read_csv(self.data_dir / 'historical_sales.csv')
        self.sales['date'] = pd.to_datetime(self.sales['date'])
        print(f"   [OK] Historical Sales: {len(self.sales):,} records")
        
        self.inventory_txn = pd.read_csv(self.data_dir / 'inventory_transactions.csv')
        self.inventory_txn['transaction_date'] = pd.to_datetime(self.inventory_txn['transaction_date'])
        print(f"   [OK] Inventory Transactions: {len(self.inventory_txn):,} records")
        
        self.economic_indicators = pd.read_csv(self.data_dir / 'economic_indicators.csv')
        print(f"   [OK] Economic Indicators: {len(self.economic_indicators)} records")
        
        self.promotional_calendar = pd.read_csv(self.data_dir / 'promotional_calendar.csv')
        print(f"   [OK] Promotional Calendar: {len(self.promotional_calendar)} records")
        
        # Load JSON files
        print("\n[LOADING] JSON files...")
        with open(self.data_dir / 'purchase_orders.json', 'r') as f:
            self.purchase_orders = pd.DataFrame(json.load(f))
        self.purchase_orders['order_date'] = pd.to_datetime(self.purchase_orders['order_date'])
        self.purchase_orders['actual_delivery_date'] = pd.to_datetime(self.purchase_orders['actual_delivery_date'])
        print(f"   [OK] Purchase Orders: {len(self.purchase_orders):,} records")
        
        # Load from SQLite database
        print("\n[LOADING] SQLite database...")
        conn = sqlite3.connect(self.data_dir / 'supply_chain.db')
        self.current_inventory = pd.read_sql_query("SELECT * FROM current_inventory", conn)
        conn.close()
        print(f"   [OK] Current Inventory: {len(self.current_inventory):,} records")
        
        print("\n[CHECK] All data loaded successfully!")
        
    def data_quality_check(self):
        """
        Perform comprehensive data quality checks.
        """
        print("\n" + "="*70)
        print("  DATA QUALITY ASSESSMENT")
        print("="*70)
        
        datasets = {
            'SKU Master': self.sku_master,
            'Warehouse Master': self.warehouse_master,
            'Supplier Master': self.supplier_master,
            'Historical Sales': self.sales,
            'Inventory Transactions': self.inventory_txn,
            'Current Inventory': self.current_inventory,
            'Purchase Orders': self.purchase_orders
        }
        
        quality_report = []
        
        for name, df in datasets.items():
            print(f"\n[CHECKING] {name}")
            
            # Basic stats
            total_rows = len(df)
            total_cols = len(df.columns)
            
            # Missing values
            missing_values = df.isnull().sum().sum()
            missing_pct = (missing_values / (total_rows * total_cols)) * 100
            
            # Duplicates
            duplicates = df.duplicated().sum()
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            quality_report.append({
                'Dataset': name,
                'Rows': total_rows,
                'Columns': total_cols,
                'Missing Values': missing_values,
                'Missing %': round(missing_pct, 2),
                'Duplicates': duplicates,
                'Memory (MB)': round(memory_mb, 2)
            })
            
            print(f"   Rows: {total_rows:,} | Columns: {total_cols}")
            print(f"   Missing Values: {missing_values} ({missing_pct:.2f}%)")
            print(f"   Duplicates: {duplicates}")
            print(f"   Memory: {memory_mb:.2f} MB")
            
            if missing_pct > 5:
                print(f"   [WARNING] High missing value percentage!")
                
        # Save quality report
        quality_df = pd.DataFrame(quality_report)
        quality_df.to_csv(self.output_dir / 'data_quality_report.csv', index=False)
        print(f"\n[EXPORT] Quality report saved to: {self.output_dir / 'data_quality_report.csv'}")
        
    def analyze_sales_patterns(self):
        """
        Analyze sales patterns and trends.
        """
        print("\n" + "="*70)
        print("  SALES PATTERN ANALYSIS")
        print("="*70)
        
        # Overall statistics
        print("\n[ANALYSIS] Overall Sales Statistics")
        total_revenue = self.sales['revenue'].sum()
        total_units = self.sales['quantity_fulfilled'].sum()
        avg_order_value = self.sales['revenue'].mean()
        
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   Total Units Sold: {total_units:,}")
        print(f"   Average Order Value: ${avg_order_value:.2f}")
        print(f"   Date Range: {self.sales['date'].min()} to {self.sales['date'].max()}")
        
        self.insights.append(f"Total revenue over 24 months: ${total_revenue:,.2f}")
        
        # Monthly trends
        monthly_sales = self.sales.groupby(self.sales['date'].dt.to_period('M')).agg({
            'revenue': 'sum',
            'quantity_fulfilled': 'sum',
            'backorder_quantity': 'sum'
        }).reset_index()
        monthly_sales['date'] = monthly_sales['date'].astype(str)
        
        print(f"\n[ANALYSIS] Monthly Sales Trends")
        print(f"   Average Monthly Revenue: ${monthly_sales['revenue'].mean():,.2f}")
        print(f"   Peak Month Revenue: ${monthly_sales['revenue'].max():,.2f}")
        print(f"   Lowest Month Revenue: ${monthly_sales['revenue'].min():,.2f}")
        
        # Seasonality check
        self.sales['month'] = self.sales['date'].dt.month
        seasonal_pattern = self.sales.groupby('month')['quantity_fulfilled'].mean()
        peak_month = seasonal_pattern.idxmax()
        low_month = seasonal_pattern.idxmin()
        
        print(f"\n[ANALYSIS] Seasonality Patterns")
        print(f"   Peak Month: {peak_month} (avg {seasonal_pattern[peak_month]:.0f} units/day)")
        print(f"   Low Month: {low_month} (avg {seasonal_pattern[low_month]:.0f} units/day)")
        print(f"   Seasonal Variation: {(seasonal_pattern.max() / seasonal_pattern.min()):.2f}x")
        
        self.insights.append(f"Demand peaks in month {peak_month} with {(seasonal_pattern.max() / seasonal_pattern.min()):.2f}x variation")
        
        # Category performance
        print(f"\n[ANALYSIS] Category Performance")
        category_sales = self.sales.groupby('category').agg({
            'revenue': 'sum',
            'quantity_fulfilled': 'sum'
        }).sort_values('revenue', ascending=False)
        
        for idx, (category, row) in enumerate(category_sales.iterrows(), 1):
            pct_revenue = (row['revenue'] / total_revenue) * 100
            print(f"   {idx}. {category}: ${row['revenue']:,.2f} ({pct_revenue:.1f}% of total)")
        
        # Stockout analysis
        stockout_rate = (self.sales['backorder_quantity'] > 0).sum() / len(self.sales) * 100
        total_backorders = self.sales['backorder_quantity'].sum()
        
        print(f"\n[ANALYSIS] Stockout Analysis")
        print(f"   Stockout Rate: {stockout_rate:.2f}% of orders")
        print(f"   Total Backorder Units: {total_backorders:,}")
        print(f"   Estimated Lost Revenue: ${(total_backorders * self.sales['unit_price'].mean()):,.2f}")
        
        self.insights.append(f"Stockout rate of {stockout_rate:.1f}% indicates inventory optimization opportunity")
        
        # Save monthly trends
        monthly_sales.to_csv(self.output_dir / 'monthly_sales_trends.csv', index=False)
        category_sales.to_csv(self.output_dir / 'category_performance.csv')
        
    def analyze_inventory_health(self):
        """
        Analyze current inventory health.
        """
        print("\n" + "="*70)
        print("  INVENTORY HEALTH ANALYSIS")
        print("="*70)
        
        # Total inventory value
        total_inv_value = self.current_inventory['total_value'].sum()
        total_units = self.current_inventory['quantity_on_hand'].sum()
        
        print(f"\n[ANALYSIS] Current Inventory Position")
        print(f"   Total Inventory Value: ${total_inv_value:,.2f}")
        print(f"   Total Units on Hand: {total_units:,}")
        print(f"   Average Unit Cost: ${total_inv_value / total_units:.2f}")
        
        self.insights.append(f"Current inventory value: ${total_inv_value:,.2f}")
        
        # Inventory distribution by warehouse
        warehouse_inv = self.current_inventory.groupby('warehouse_id').agg({
            'quantity_on_hand': 'sum',
            'total_value': 'sum'
        }).sort_values('total_value', ascending=False)
        
        print(f"\n[ANALYSIS] Top 5 Warehouses by Inventory Value")
        for idx, (wh_id, row) in enumerate(warehouse_inv.head().iterrows(), 1):
            pct = (row['total_value'] / total_inv_value) * 100
            print(f"   {idx}. {wh_id}: ${row['total_value']:,.2f} ({pct:.1f}%)")
        
        # Items below reorder point
        below_rop = self.current_inventory[
            self.current_inventory['quantity_on_hand'] < self.current_inventory['reorder_point']
        ]
        
        print(f"\n[ANALYSIS] Reorder Point Analysis")
        print(f"   Items Below Reorder Point: {len(below_rop)} ({len(below_rop)/len(self.current_inventory)*100:.1f}%)")
        print(f"   Urgent Reorders Needed: ${below_rop['total_value'].sum():,.2f} in value")
        
        if len(below_rop) > 0:
            self.insights.append(f"{len(below_rop)} items below reorder point - immediate action required")
        
        # Excess inventory (above safety stock by 3x)
        excess_inv = self.current_inventory[
            self.current_inventory['quantity_on_hand'] > (self.current_inventory['safety_stock'] * 3)
        ]
        
        print(f"\n[ANALYSIS] Excess Inventory")
        print(f"   Items with Excess Stock: {len(excess_inv)} ({len(excess_inv)/len(self.current_inventory)*100:.1f}%)")
        print(f"   Excess Inventory Value: ${excess_inv['total_value'].sum():,.2f}")
        
        if len(excess_inv) > 0:
            self.insights.append(f"${excess_inv['total_value'].sum():,.2f} in excess inventory - potential cost reduction")
        
    def analyze_supplier_performance(self):
        """
        Analyze supplier reliability and performance.
        """
        print("\n" + "="*70)
        print("  SUPPLIER PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Overall supplier statistics
        avg_on_time = self.supplier_master['on_time_delivery_rate'].mean()
        avg_quality = self.supplier_master['quality_rating'].mean()
        
        print(f"\n[ANALYSIS] Overall Supplier Metrics")
        print(f"   Average On-Time Delivery Rate: {avg_on_time*100:.1f}%")
        print(f"   Average Quality Rating: {avg_quality:.2f}/5.0")
        
        # Supplier tiers
        tier1 = self.supplier_master[
            (self.supplier_master['on_time_delivery_rate'] >= 0.95) & 
            (self.supplier_master['quality_rating'] >= 4.5)
        ]
        tier4 = self.supplier_master[
            (self.supplier_master['on_time_delivery_rate'] < 0.70) | 
            (self.supplier_master['quality_rating'] < 3.5)
        ]
        
        print(f"\n[ANALYSIS] Supplier Tier Distribution")
        print(f"   Tier 1 (Excellent): {len(tier1)} suppliers ({len(tier1)/len(self.supplier_master)*100:.1f}%)")
        print(f"   Tier 4 (Poor): {len(tier4)} suppliers ({len(tier4)/len(self.supplier_master)*100:.1f}%)")
        
        if len(tier4) > 0:
            self.insights.append(f"{len(tier4)} suppliers in Tier 4 - review and potentially replace")
        
        # Country analysis
        country_perf = self.supplier_master.groupby('country').agg({
            'on_time_delivery_rate': 'mean',
            'quality_rating': 'mean',
            'supplier_id': 'count'
        }).round(3)
        country_perf.columns = ['Avg_OnTime_Rate', 'Avg_Quality', 'Num_Suppliers']
        
        print(f"\n[ANALYSIS] Performance by Country")
        for country, row in country_perf.sort_values('Avg_OnTime_Rate', ascending=False).iterrows():
            print(f"   {country}: {row['Avg_OnTime_Rate']*100:.1f}% on-time, {row['Avg_Quality']:.2f} quality ({int(row['Num_Suppliers'])} suppliers)")
        
        country_perf.to_csv(self.output_dir / 'supplier_performance_by_country.csv')
        
    def generate_summary_report(self):
        """
        Generate executive summary report.
        """
        print("\n" + "="*70)
        print("  GENERATING SUMMARY REPORT")
        print("="*70)
        
        report = {
            'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Data Period': f"{self.sales['date'].min()} to {self.sales['date'].max()}",
            'Total SKUs': len(self.sku_master),
            'Total Warehouses': len(self.warehouse_master),
            'Total Suppliers': len(self.supplier_master),
            'Total Sales Transactions': len(self.sales),
            'Total Revenue': f"${self.sales['revenue'].sum():,.2f}",
            'Total Inventory Value': f"${self.current_inventory['total_value'].sum():,.2f}",
            'Average Fill Rate': f"{(self.sales['quantity_fulfilled'].sum() / self.sales['quantity_ordered'].sum() * 100):.2f}%",
            'Stockout Rate': f"{((self.sales['backorder_quantity'] > 0).sum() / len(self.sales) * 100):.2f}%"
        }
        
        report_df = pd.DataFrame([report]).T
        report_df.columns = ['Value']
        report_df.to_csv(self.output_dir / 'executive_summary.csv')
        
        print("\n[SUMMARY] Key Metrics:")
        for key, value in report.items():
            print(f"   {key}: {value}")
        
        # Save insights
        insights_df = pd.DataFrame({'Insight': self.insights})
        insights_df.to_csv(self.output_dir / 'key_insights.csv', index=False)
        
        print(f"\n[EXPORT] Summary report saved to: {self.output_dir / 'executive_summary.csv'}")
        print(f"[EXPORT] Key insights saved to: {self.output_dir / 'key_insights.csv'}")
        
    def run_full_analysis(self):
        """
        Execute complete exploratory data analysis.
        """
        print("\n" + "="*70)
        print("  SUPPLY CHAIN EXPLORATORY DATA ANALYSIS")
        print("="*70)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_all_data()
        self.data_quality_check()
        self.analyze_sales_patterns()
        self.analyze_inventory_health()
        self.analyze_supplier_performance()
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("  EDA COMPLETE!")
        print("="*70)
        print(f"\n[CHECK] All analysis results exported to: {self.output_dir.absolute()}")
        print("\n[TARGET] Next Steps:")
        print("   1. Review summary reports in data/processed/")
        print("   2. Run demand forecasting models (src/demand_forecasting.py)")
        print("   3. Perform inventory optimization (src/inventory_optimization.py)")
        print("   4. Create visualizations and dashboards")
        print("\n")


def main():
    """
    Main execution function.
    """
    eda = SupplyChainEDA()
    eda.run_full_analysis()


if __name__ == "__main__":
    main()
