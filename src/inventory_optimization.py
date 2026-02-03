"""
Inventory Optimization Module
==============================

PURPOSE:
This module calculates optimal inventory policies to minimize total costs while
maintaining target service levels.

BUSINESS VALUE:
- Reduce excess inventory (carrying costs)
- Prevent stockouts (lost sales)
- Optimize reorder points and safety stock
- Balance ordering costs vs holding costs

OPTIMIZATION TECHNIQUES:
1. Economic Order Quantity (EOQ)
2. Reorder Point (ROP) calculation
3. Safety Stock optimization
4. Service Level analysis
5. Linear Programming for multi-SKU optimization

KEY FORMULAS:
- EOQ = SQRT((2 * Annual Demand * Order Cost) / Carrying Cost per Unit)
- ROP = Lead Time Demand + Safety Stock
- Safety Stock = Z-score * Std Dev of Demand * SQRT(Lead Time)

HOW TO USE:
    python src/inventory_optimization.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class InventoryOptimizer:
    """
    Optimizes inventory policies for supply chain cost reduction.
    """
    
    def __init__(self, data_dir: str = "data/raw", output_dir: str = "data/output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sku_master = None
        self.sales = None
        self.current_inventory = None
        
        self.optimization_results = []
        self.cost_savings = {}
        
    def load_data(self):
        """
        Load necessary datasets for optimization.
        """
        print("\n" + "="*70)
        print("  LOADING DATA FOR INVENTORY OPTIMIZATION")
        print("="*70)
        
        print("\n[LOADING] SKU Master...")
        self.sku_master = pd.read_csv(self.data_dir / 'sku_master.csv')
        print(f"   [OK] {len(self.sku_master)} SKUs loaded")
        
        print("\n[LOADING] Historical Sales...")
        self.sales = pd.read_csv(self.data_dir / 'historical_sales.csv')
        self.sales['date'] = pd.to_datetime(self.sales['date'])
        print(f"   [OK] {len(self.sales):,} sales transactions loaded")
        
        print("\n[LOADING] Current Inventory...")
        import sqlite3
        conn = sqlite3.connect(self.data_dir / 'supply_chain.db')
        self.current_inventory = pd.read_sql_query("SELECT * FROM current_inventory", conn)
        conn.close()
        print(f"   [OK] {len(self.current_inventory):,} inventory records loaded")
        
    def calculate_demand_statistics(self):
        """
        Calculate demand statistics for each SKU.
        """
        print("\n" + "="*70)
        print("  CALCULATING DEMAND STATISTICS")
        print("="*70)
        
        # Calculate daily demand statistics per SKU
        demand_stats = self.sales.groupby('sku_id').agg({
            'quantity_fulfilled': ['mean', 'std', 'sum', 'count']
        }).reset_index()
        
        demand_stats.columns = ['sku_id', 'avg_daily_demand', 'std_daily_demand', 
                                'total_annual_demand', 'num_transactions']
        
        # Annualize demand (data is for 24 months)
        demand_stats['total_annual_demand'] = demand_stats['total_annual_demand'] / 2
        
        # Handle zero std (constant demand)
        demand_stats['std_daily_demand'] = demand_stats['std_daily_demand'].fillna(
            demand_stats['avg_daily_demand'] * 0.1  # Assume 10% variation
        )
        
        # Merge with SKU master
        self.demand_stats = demand_stats.merge(self.sku_master, on='sku_id', how='left')
        
        print(f"\n[OK] Demand statistics calculated for {len(self.demand_stats)} SKUs")
        print(f"   Average daily demand across all SKUs: {demand_stats['avg_daily_demand'].mean():.2f} units")
        print(f"   Total annual demand: {demand_stats['total_annual_demand'].sum():,.0f} units")
        
    def calculate_eoq(self):
        """
        Calculate Economic Order Quantity for each SKU.
        
        EOQ = SQRT((2 * Annual Demand * Order Cost) / (Carrying Cost per Unit per Year))
        """
        print("\n" + "="*70)
        print("  CALCULATING ECONOMIC ORDER QUANTITY (EOQ)")
        print("="*70)
        
        # Assumptions
        ORDER_COST = 50  # Fixed cost per order ($)
        
        print(f"\n[PARAMETERS]")
        print(f"   Order Cost: ${ORDER_COST} per order")
        print(f"   Carrying Cost Rate: From SKU master (15-25% of unit cost)")
        
        # Calculate EOQ for each SKU
        self.demand_stats['order_cost'] = ORDER_COST
        self.demand_stats['carrying_cost_per_unit'] = (
            self.demand_stats['unit_cost'] * self.demand_stats['carrying_cost_rate']
        )
        
        self.demand_stats['eoq'] = np.sqrt(
            (2 * self.demand_stats['total_annual_demand'] * self.demand_stats['order_cost']) /
            self.demand_stats['carrying_cost_per_unit']
        )
        
        # Round to nearest integer
        self.demand_stats['eoq'] = self.demand_stats['eoq'].fillna(0).round(0).astype(int)
        
        # Calculate orders per year
        self.demand_stats['orders_per_year'] = (
            self.demand_stats['total_annual_demand'] / self.demand_stats['eoq']
        ).replace([np.inf, -np.inf], 0).fillna(0).round(1)
        
        # Calculate days between orders
        self.demand_stats['days_between_orders'] = (
            365 / self.demand_stats['orders_per_year']
        ).replace([np.inf, -np.inf], 365).fillna(365).round(0).astype(int)
        
        print(f"\n[RESULTS] EOQ Calculation Summary:")
        print(f"   Average EOQ: {self.demand_stats['eoq'].mean():.0f} units")
        print(f"   Median EOQ: {self.demand_stats['eoq'].median():.0f} units")
        print(f"   Average orders per year: {self.demand_stats['orders_per_year'].mean():.1f}")
        
    def calculate_safety_stock(self, service_level: float = 0.95):
        """
        Calculate safety stock for each SKU based on desired service level.
        
        Safety Stock = Z-score * Std Dev of Demand * SQRT(Lead Time in days)
        
        Args:
            service_level: Target service level (default: 95%)
        """
        print("\n" + "="*70)
        print("  CALCULATING SAFETY STOCK")
        print("="*70)
        
        # Z-score for service level
        z_score = stats.norm.ppf(service_level)
        
        print(f"\n[PARAMETERS]")
        print(f"   Target Service Level: {service_level*100:.0f}%")
        print(f"   Z-score: {z_score:.2f}")
        print(f"   Lead Time: From SKU master (40-120 days)")
        
        # Calculate safety stock
        self.demand_stats['z_score'] = z_score
        self.demand_stats['safety_stock'] = (
            z_score * 
            self.demand_stats['std_daily_demand'] * 
            np.sqrt(self.demand_stats['lead_time_days'])
        )
        
        self.demand_stats['safety_stock'] = self.demand_stats['safety_stock'].fillna(0).round(0).astype(int)
        
        print(f"\n[RESULTS] Safety Stock Calculation Summary:")
        print(f"   Average Safety Stock: {self.demand_stats['safety_stock'].mean():.0f} units")
        print(f"   Total Safety Stock Value: ${(self.demand_stats['safety_stock'] * self.demand_stats['unit_cost']).sum():,.2f}")
        
    def calculate_reorder_point(self):
        """
        Calculate Reorder Point for each SKU.
        
        ROP = Lead Time Demand + Safety Stock
        """
        print("\n" + "="*70)
        print("  CALCULATING REORDER POINT (ROP)")
        print("="*70)
        
        # Lead time demand
        self.demand_stats['lead_time_demand'] = (
            self.demand_stats['avg_daily_demand'] * 
            self.demand_stats['lead_time_days']
        ).round(0).astype(int)
        
        # Reorder point
        self.demand_stats['reorder_point'] = (
            self.demand_stats['lead_time_demand'] + 
            self.demand_stats['safety_stock']
        )
        
        print(f"\n[RESULTS] Reorder Point Calculation Summary:")
        print(f"   Average ROP: {self.demand_stats['reorder_point'].mean():.0f} units")
        print(f"   Average Lead Time Demand: {self.demand_stats['lead_time_demand'].mean():.0f} units")
        
    def calculate_total_cost(self):
        """
        Calculate total inventory cost (ordering + carrying + stockout).
        """
        print("\n" + "="*70)
        print("  CALCULATING TOTAL INVENTORY COSTS")
        print("="*70)
        
        # Annual ordering cost
        self.demand_stats['annual_ordering_cost'] = (
            self.demand_stats['orders_per_year'] * 
            self.demand_stats['order_cost']
        )
        
        # Annual carrying cost (average inventory = EOQ/2 + Safety Stock)
        self.demand_stats['avg_inventory'] = (
            self.demand_stats['eoq'] / 2 + 
            self.demand_stats['safety_stock']
        )
        
        self.demand_stats['annual_carrying_cost'] = (
            self.demand_stats['avg_inventory'] * 
            self.demand_stats['carrying_cost_per_unit']
        )
        
        # Total cost
        self.demand_stats['total_annual_cost'] = (
            self.demand_stats['annual_ordering_cost'] + 
            self.demand_stats['annual_carrying_cost']
        )
        
        print(f"\n[RESULTS] Total Cost Analysis:")
        print(f"   Total Annual Ordering Cost: ${self.demand_stats['annual_ordering_cost'].sum():,.2f}")
        print(f"   Total Annual Carrying Cost: ${self.demand_stats['annual_carrying_cost'].sum():,.2f}")
        print(f"   Total Annual Inventory Cost: ${self.demand_stats['total_annual_cost'].sum():,.2f}")
        
        # Cost breakdown by category
        category_costs = self.demand_stats.groupby('category').agg({
            'total_annual_cost': 'sum',
            'annual_ordering_cost': 'sum',
            'annual_carrying_cost': 'sum'
        }).round(2)
        
        print(f"\n[BREAKDOWN] Cost by Category:")
        for category, row in category_costs.sort_values('total_annual_cost', ascending=False).iterrows():
            print(f"   {category}: ${row['total_annual_cost']:,.2f}")
            print(f"      Ordering: ${row['annual_ordering_cost']:,.2f} | Carrying: ${row['annual_carrying_cost']:,.2f}")
        
    def compare_current_vs_optimized(self):
        """
        Compare current inventory policies vs optimized policies.
        """
        print("\n" + "="*70)
        print("  CURRENT VS OPTIMIZED COMPARISON")
        print("="*70)
        
        # Merge with current inventory
        comparison = self.demand_stats.merge(
            self.current_inventory[['sku_id', 'quantity_on_hand', 'reorder_point', 'safety_stock']],
            on='sku_id',
            how='left',
            suffixes=('_optimized', '_current')
        )
        
        # Calculate differences
        comparison['rop_difference'] = comparison['reorder_point_optimized'] - comparison['reorder_point_current']
        comparison['safety_stock_difference'] = comparison['safety_stock_optimized'] - comparison['safety_stock_current']
        
        # Calculate potential savings
        comparison['current_carrying_cost'] = (
            comparison['quantity_on_hand'] * 
            comparison['carrying_cost_per_unit']
        )
        
        comparison['optimized_carrying_cost'] = (
            comparison['avg_inventory'] * 
            comparison['carrying_cost_per_unit']
        )
        
        comparison['potential_savings'] = (
            comparison['current_carrying_cost'] - 
            comparison['optimized_carrying_cost']
        )
        
        total_current_cost = comparison['current_carrying_cost'].sum()
        total_optimized_cost = comparison['optimized_carrying_cost'].sum()
        total_savings = comparison['potential_savings'].sum()
        
        print(f"\n[COMPARISON] Inventory Cost Impact:")
        print(f"   Current Annual Carrying Cost: ${total_current_cost:,.2f}")
        print(f"   Optimized Annual Carrying Cost: ${total_optimized_cost:,.2f}")
        print(f"   Potential Annual Savings: ${total_savings:,.2f}")
        print(f"   Cost Reduction: {(total_savings / total_current_cost * 100):.1f}%")
        
        self.cost_savings = {
            'current_cost': total_current_cost,
            'optimized_cost': total_optimized_cost,
            'savings': total_savings,
            'savings_pct': (total_savings / total_current_cost * 100)
        }
        
        # Items needing adjustment
        items_increase_rop = len(comparison[comparison['rop_difference'] > 0])
        items_decrease_rop = len(comparison[comparison['rop_difference'] < 0])
        
        print(f"\n[ACTIONS] Recommended Adjustments:")
        print(f"   Increase Reorder Point: {items_increase_rop} SKUs")
        print(f"   Decrease Reorder Point: {items_decrease_rop} SKUs")
        
        # Save comparison
        comparison_export = comparison[[
            'sku_id', 'product_name', 'category',
            'eoq', 'reorder_point_optimized', 'safety_stock_optimized',
            'reorder_point_current', 'safety_stock_current',
            'rop_difference', 'safety_stock_difference',
            'potential_savings'
        ]].sort_values('potential_savings', ascending=False)
        
        comparison_export.to_csv(self.output_dir / 'inventory_optimization_comparison.csv', index=False)
        print(f"\n[EXPORT] Comparison saved to: {self.output_dir / 'inventory_optimization_comparison.csv'}")
        
    def generate_optimization_report(self):
        """
        Generate executive summary of optimization results.
        """
        print("\n" + "="*70)
        print("  OPTIMIZATION SUMMARY REPORT")
        print("="*70)
        
        report = {
            'Analysis Date': datetime.now().strftime('%Y-%m-%d'),
            'SKUs Analyzed': len(self.demand_stats),
            'Target Service Level': '95%',
            'Average EOQ': f"{self.demand_stats['eoq'].mean():.0f} units",
            'Average Reorder Point': f"{self.demand_stats['reorder_point'].mean():.0f} units",
            'Average Safety Stock': f"{self.demand_stats['safety_stock'].mean():.0f} units",
            'Total Annual Inventory Cost (Optimized)': f"${self.demand_stats['total_annual_cost'].sum():,.2f}",
            'Current Annual Carrying Cost': f"${self.cost_savings['current_cost']:,.2f}",
            'Optimized Annual Carrying Cost': f"${self.cost_savings['optimized_cost']:,.2f}",
            'Potential Annual Savings': f"${self.cost_savings['savings']:,.2f}",
            'Cost Reduction Percentage': f"{self.cost_savings['savings_pct']:.1f}%"
        }
        
        report_df = pd.DataFrame([report]).T
        report_df.columns = ['Value']
        report_df.to_csv(self.output_dir / 'optimization_summary.csv')
        
        print("\n[SUMMARY] Optimization Results:")
        for key, value in report.items():
            print(f"   {key}: {value}")
        
        print(f"\n[EXPORT] Summary saved to: {self.output_dir / 'optimization_summary.csv'}")
        
        # Save detailed results
        detailed_results = self.demand_stats[[
            'sku_id', 'product_name', 'category',
            'avg_daily_demand', 'std_daily_demand', 'total_annual_demand',
            'lead_time_days', 'unit_cost', 'carrying_cost_rate',
            'eoq', 'orders_per_year', 'days_between_orders',
            'safety_stock', 'reorder_point',
            'annual_ordering_cost', 'annual_carrying_cost', 'total_annual_cost'
        ]].sort_values('total_annual_cost', ascending=False)
        
        detailed_results.to_csv(self.output_dir / 'inventory_optimization_detailed.csv', index=False)
        print(f"[EXPORT] Detailed results saved to: {self.output_dir / 'inventory_optimization_detailed.csv'}")
        
    def run_full_optimization(self):
        """
        Execute complete inventory optimization workflow.
        """
        print("\n" + "="*70)
        print("  INVENTORY OPTIMIZATION ANALYSIS")
        print("="*70)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_data()
        self.calculate_demand_statistics()
        self.calculate_eoq()
        self.calculate_safety_stock(service_level=0.95)
        self.calculate_reorder_point()
        self.calculate_total_cost()
        self.compare_current_vs_optimized()
        self.generate_optimization_report()
        
        print("\n" + "="*70)
        print("  OPTIMIZATION COMPLETE!")
        print("="*70)
        print(f"\n[CHECK] All results exported to: {self.output_dir.absolute()}")
        print(f"\n[IMPACT] Potential annual savings: ${self.cost_savings['savings']:,.2f}")
        print("\n[TARGET] Next Steps:")
        print("   1. Review optimization recommendations")
        print("   2. Implement new reorder points and safety stock levels")
        print("   3. Monitor service level achievement")
        print("   4. Create visualization dashboards")
        print("\n")


def main():
    """
    Main execution function.
    """
    optimizer = InventoryOptimizer()
    optimizer.run_full_optimization()


if __name__ == "__main__":
    main()
