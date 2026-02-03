"""
Visualization and Dashboard Module
===================================

PURPOSE:
This module creates comprehensive visualizations and interactive dashboards
for supply chain analytics insights.

BUSINESS VALUE:
- Communicate insights to stakeholders
- Enable data-driven decision making
- Monitor KPIs and trends
- Identify patterns and anomalies

VISUALIZATIONS CREATED:
1. Sales trends and seasonality
2. Inventory turnover analysis
3. Supplier performance scorecards
4. Forecast accuracy charts
5. ABC classification Pareto charts
6. Cost breakdown analysis
7. Stockout analysis
8. Warehouse utilization heatmaps

HOW TO USE:
    python src/visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class SupplyChainVisualizer:
    """
    Creates visualizations and dashboards for supply chain analytics.
    """
    
    def __init__(self, data_dir: str = "data/raw", processed_dir: str = "data/processed", 
                 output_dir: str = "reports/figures"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sales = None
        self.sku_master = None
        self.supplier_master = None
        self.warehouse_master = None
        
    def load_data(self):
        """
        Load all necessary datasets.
        """
        print("\n" + "="*70)
        print("  LOADING DATA FOR VISUALIZATION")
        print("="*70)
        
        print("\n[LOADING] Historical sales...")
        self.sales = pd.read_csv(self.data_dir / 'historical_sales.csv')
        self.sales['date'] = pd.to_datetime(self.sales['date'])
        print(f"   [OK] {len(self.sales):,} records")
        
        print("\n[LOADING] SKU Master...")
        self.sku_master = pd.read_csv(self.data_dir / 'sku_master.csv')
        print(f"   [OK] {len(self.sku_master)} SKUs")
        
        print("\n[LOADING] Supplier Master...")
        self.supplier_master = pd.read_csv(self.data_dir / 'supplier_master.csv')
        print(f"   [OK] {len(self.supplier_master)} suppliers")
        
        print("\n[LOADING] Warehouse Master...")
        self.warehouse_master = pd.read_csv(self.data_dir / 'warehouse_master.csv')
        print(f"   [OK] {len(self.warehouse_master)} warehouses")
        
    def plot_sales_trends(self):
        """
        Create sales trend visualizations.
        """
        print("\n[CREATING] Sales Trend Visualizations...")
        
        # Daily sales trend
        daily_sales = self.sales.groupby('date').agg({
            'revenue': 'sum',
            'quantity_fulfilled': 'sum'
        }).reset_index()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Revenue Trend', 'Daily Units Sold'),
            vertical_spacing=0.12
        )
        
        # Revenue trend
        fig.add_trace(
            go.Scatter(
                x=daily_sales['date'],
                y=daily_sales['revenue'],
                mode='lines',
                name='Revenue',
                line=dict(color='#2E86AB', width=2)
            ),
            row=1, col=1
        )
        
        # Units trend
        fig.add_trace(
            go.Scatter(
                x=daily_sales['date'],
                y=daily_sales['quantity_fulfilled'],
                mode='lines',
                name='Units Sold',
                line=dict(color='#A23B72', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        fig.update_yaxes(title_text="Units", row=2, col=1)
        
        fig.update_layout(
            title_text="Sales Performance Over Time (24 Months)",
            height=700,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.write_html(self.output_dir / 'sales_trends.html')
        print(f"   [OK] Saved: sales_trends.html")
        
    def plot_seasonality(self):
        """
        Create seasonality analysis visualization.
        """
        print("\n[CREATING] Seasonality Analysis...")
        
        # Monthly aggregation
        self.sales['month'] = self.sales['date'].dt.month
        self.sales['year'] = self.sales['date'].dt.year
        
        monthly_sales = self.sales.groupby(['year', 'month']).agg({
            'quantity_fulfilled': 'sum'
        }).reset_index()
        
        # Average by month across years
        avg_by_month = self.sales.groupby('month')['quantity_fulfilled'].mean().reset_index()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        avg_by_month['month_name'] = avg_by_month['month'].apply(lambda x: month_names[x-1])
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=avg_by_month['month_name'],
            y=avg_by_month['quantity_fulfilled'],
            marker_color='#F18F01',
            text=avg_by_month['quantity_fulfilled'].round(0),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Average Daily Demand by Month (Seasonality Pattern)',
            xaxis_title='Month',
            yaxis_title='Average Daily Units Sold',
            template='plotly_white',
            height=500
        )
        
        fig.write_html(self.output_dir / 'seasonality_pattern.html')
        print(f"   [OK] Saved: seasonality_pattern.html")
        
    def plot_category_performance(self):
        """
        Create category performance visualization.
        """
        print("\n[CREATING] Category Performance Analysis...")
        
        category_sales = self.sales.groupby('category').agg({
            'revenue': 'sum',
            'quantity_fulfilled': 'sum'
        }).reset_index().sort_values('revenue', ascending=False)
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=category_sales['category'],
                y=category_sales['revenue'],
                name='Revenue',
                marker_color='#006BA6'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=category_sales['category'],
                y=category_sales['quantity_fulfilled'],
                name='Units Sold',
                mode='lines+markers',
                marker=dict(size=10, color='#C1121F'),
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Product Category")
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Units Sold", secondary_y=True)
        
        fig.update_layout(
            title='Sales Performance by Product Category',
            template='plotly_white',
            height=500
        )
        
        fig.write_html(self.output_dir / 'category_performance.html')
        print(f"   [OK] Saved: category_performance.html")
        
    def plot_abc_classification(self):
        """
        Create ABC classification Pareto chart.
        """
        print("\n[CREATING] ABC Classification (Pareto Chart)...")
        
        # Load ABC classification data if available
        try:
            abc_data = pd.read_csv(self.processed_dir / '1.1_abc_classification.csv')
        except:
            # Calculate on the fly
            import sqlite3
            conn = sqlite3.connect(self.data_dir / 'supply_chain.db')
            current_inv = pd.read_sql_query("SELECT * FROM current_inventory", conn)
            conn.close()
            
            abc_data = current_inv.merge(self.sku_master, on='sku_id')
            abc_data = abc_data.sort_values('total_value', ascending=False).head(50)
            abc_data['cumulative_pct'] = (abc_data['total_value'].cumsum() / abc_data['total_value'].sum() * 100)
        
        # Create Pareto chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=abc_data['sku_id'].head(30),
                y=abc_data['total_value'].head(30),
                name='Inventory Value',
                marker_color='#4361EE'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=abc_data['sku_id'].head(30),
                y=abc_data['cumulative_pct'].head(30),
                name='Cumulative %',
                mode='lines+markers',
                marker=dict(size=8, color='#F72585'),
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        # Add 80% line
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="80% Threshold", secondary_y=True)
        
        fig.update_xaxes(title_text="SKU ID")
        fig.update_yaxes(title_text="Inventory Value ($)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 100])
        
        fig.update_layout(
            title='ABC Classification - Pareto Analysis (Top 30 SKUs)',
            template='plotly_white',
            height=500
        )
        
        fig.write_html(self.output_dir / 'abc_pareto_chart.html')
        print(f"   [OK] Saved: abc_pareto_chart.html")
        
    def plot_supplier_scorecard(self):
        """
        Create supplier performance scorecard.
        """
        print("\n[CREATING] Supplier Performance Scorecard...")
        
        # Top 15 suppliers by on-time delivery
        top_suppliers = self.supplier_master.nlargest(15, 'on_time_delivery_rate')
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color by tier
        colors = []
        for _, row in top_suppliers.iterrows():
            if row['on_time_delivery_rate'] >= 0.95 and row['quality_rating'] >= 4.5:
                colors.append('#06D6A0')  # Green - Tier 1
            elif row['on_time_delivery_rate'] >= 0.85 and row['quality_rating'] >= 4.0:
                colors.append('#FFD166')  # Yellow - Tier 2
            elif row['on_time_delivery_rate'] >= 0.70:
                colors.append('#EF476F')  # Orange - Tier 3
            else:
                colors.append('#C1121F')  # Red - Tier 4
        
        fig.add_trace(go.Scatter(
            x=top_suppliers['on_time_delivery_rate'] * 100,
            y=top_suppliers['quality_rating'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color='white')
            ),
            text=top_suppliers['supplier_id'],
            textposition='top center',
            textfont=dict(size=9)
        ))
        
        # Add quadrant lines
        fig.add_hline(y=4.0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=85, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title='Supplier Performance Scorecard (Top 15)',
            xaxis_title='On-Time Delivery Rate (%)',
            yaxis_title='Quality Rating (1-5)',
            template='plotly_white',
            height=600,
            xaxis=dict(range=[60, 100]),
            yaxis=dict(range=[2, 5.5])
        )
        
        fig.write_html(self.output_dir / 'supplier_scorecard.html')
        print(f"   [OK] Saved: supplier_scorecard.html")
        
    def plot_stockout_analysis(self):
        """
        Create stockout analysis visualization.
        """
        print("\n[CREATING] Stockout Analysis...")
        
        # Calculate stockout rate by category
        stockout_by_category = self.sales.groupby('category').agg({
            'backorder_quantity': lambda x: (x > 0).sum(),
            'sku_id': 'count'
        }).reset_index()
        
        stockout_by_category['stockout_rate'] = (
            stockout_by_category['backorder_quantity'] / 
            stockout_by_category['sku_id'] * 100
        )
        stockout_by_category = stockout_by_category.sort_values('stockout_rate', ascending=False)
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=stockout_by_category['category'],
            y=stockout_by_category['stockout_rate'],
            marker_color='#E63946',
            text=stockout_by_category['stockout_rate'].round(1),
            texttemplate='%{text}%',
            textposition='outside'
        ))
        
        # Add target line (5% stockout rate)
        fig.add_hline(y=5, line_dash="dash", line_color="green", 
                     annotation_text="Target: 5%")
        
        fig.update_layout(
            title='Stockout Rate by Product Category',
            xaxis_title='Category',
            yaxis_title='Stockout Rate (%)',
            template='plotly_white',
            height=500
        )
        
        fig.write_html(self.output_dir / 'stockout_analysis.html')
        print(f"   [OK] Saved: stockout_analysis.html")
        
    def plot_warehouse_utilization(self):
        """
        Create warehouse utilization heatmap.
        """
        print("\n[CREATING] Warehouse Utilization Analysis...")
        
        # Create bar chart for utilization
        wh_sorted = self.warehouse_master.sort_values('current_utilization_rate', ascending=False)
        
        # Color code by utilization level
        colors = []
        for util in wh_sorted['current_utilization_rate']:
            if util > 0.90:
                colors.append('#E63946')  # Red - Over capacity
            elif util > 0.70:
                colors.append('#06D6A0')  # Green - Optimal
            elif util > 0.50:
                colors.append('#FFD166')  # Yellow - Acceptable
            else:
                colors.append('#C1121F')  # Dark red - Underutilized
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=wh_sorted['warehouse_name'],
            y=wh_sorted['current_utilization_rate'] * 100,
            marker_color=colors,
            text=(wh_sorted['current_utilization_rate'] * 100).round(1),
            texttemplate='%{text}%',
            textposition='outside'
        ))
        
        # Add reference lines
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="Over Capacity (90%)")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                     annotation_text="Underutilized (50%)")
        
        fig.update_layout(
            title='Warehouse Capacity Utilization',
            xaxis_title='Warehouse',
            yaxis_title='Utilization Rate (%)',
            template='plotly_white',
            height=600,
            xaxis_tickangle=-45
        )
        
        fig.write_html(self.output_dir / 'warehouse_utilization.html')
        print(f"   [OK] Saved: warehouse_utilization.html")
        
    def create_executive_dashboard(self):
        """
        Create comprehensive executive dashboard.
        """
        print("\n[CREATING] Executive Dashboard...")
        
        # Calculate KPIs
        total_revenue = self.sales['revenue'].sum()
        total_units = self.sales['quantity_fulfilled'].sum()
        stockout_rate = (self.sales['backorder_quantity'] > 0).sum() / len(self.sales) * 100
        fill_rate = (self.sales['quantity_fulfilled'].sum() / self.sales['quantity_ordered'].sum() * 100)
        
        # Create dashboard with multiple subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Revenue Trend',
                'Category Distribution',
                'Supplier Performance',
                'Warehouse Utilization'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Monthly revenue trend
        monthly_rev = self.sales.groupby(self.sales['date'].dt.to_period('M'))['revenue'].sum().reset_index()
        monthly_rev['date'] = monthly_rev['date'].astype(str)
        
        fig.add_trace(
            go.Scatter(
                x=monthly_rev['date'],
                y=monthly_rev['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#2E86AB', width=3)
            ),
            row=1, col=1
        )
        
        # 2. Category pie chart
        category_rev = self.sales.groupby('category')['revenue'].sum().reset_index()
        
        fig.add_trace(
            go.Pie(
                labels=category_rev['category'],
                values=category_rev['revenue'],
                name='Category'
            ),
            row=1, col=2
        )
        
        # 3. Supplier performance
        top_suppliers = self.supplier_master.nlargest(10, 'on_time_delivery_rate')
        
        fig.add_trace(
            go.Bar(
                x=top_suppliers['supplier_id'],
                y=top_suppliers['on_time_delivery_rate'] * 100,
                name='On-Time %',
                marker_color='#06D6A0'
            ),
            row=2, col=1
        )
        
        # 4. Warehouse utilization
        fig.add_trace(
            go.Bar(
                x=self.warehouse_master['warehouse_id'],
                y=self.warehouse_master['current_utilization_rate'] * 100,
                name='Utilization %',
                marker_color='#F18F01'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"Supply Chain Executive Dashboard | Revenue: ${total_revenue:,.0f} | Fill Rate: {fill_rate:.1f}%",
            height=900,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.write_html(self.output_dir / 'executive_dashboard.html')
        print(f"   [OK] Saved: executive_dashboard.html")
        
    def generate_all_visualizations(self):
        """
        Generate all visualizations.
        """
        print("\n" + "="*70)
        print("  GENERATING SUPPLY CHAIN VISUALIZATIONS")
        print("="*70)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_data()
        
        self.plot_sales_trends()
        self.plot_seasonality()
        self.plot_category_performance()
        self.plot_abc_classification()
        self.plot_supplier_scorecard()
        self.plot_stockout_analysis()
        self.plot_warehouse_utilization()
        self.create_executive_dashboard()
        
        print("\n" + "="*70)
        print("  VISUALIZATION COMPLETE!")
        print("="*70)
        print(f"\n[CHECK] All visualizations saved to: {self.output_dir.absolute()}")
        print("\n[TARGET] Next Steps:")
        print("   1. Open HTML files in browser to view interactive charts")
        print("   2. Share executive_dashboard.html with stakeholders")
        print("   3. Use insights for decision-making")
        print("\n")


def main():
    """
    Main execution function.
    """
    visualizer = SupplyChainVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
