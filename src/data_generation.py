"""
Supply Chain Data Generation Module
====================================

PURPOSE:
This module generates realistic synthetic data for a manufacturing supply chain environment.
It simulates TechManufacture Inc., a mid-sized electronics manufacturer with complex
inventory, supplier, and demand patterns.

BUSINESS CONTEXT:
- 200+ SKUs (Stock Keeping Units) across multiple product categories
- 15 warehouses with varying capacities and operating costs
- 50+ suppliers with different performance characteristics
- 24 months of historical sales and inventory transactions
- Seasonal demand patterns, promotional impacts, and supply chain disruptions

WHY THIS IS CRITICAL:
Real supply chain data contains sensitive business information. This synthetic data
replicates real-world complexity including:
- Demand volatility and seasonality
- Supplier reliability issues (late deliveries, quality problems)
- Inventory imbalances (stockouts and excess inventory)
- Cost structures (carrying costs, ordering costs, expedited shipping)

The data will enable us to:
1. Perform demand forecasting with time-series models
2. Optimize inventory levels (reorder points, safety stock)
3. Evaluate supplier performance
4. Calculate total cost of ownership
5. Simulate what-if scenarios for cost reduction
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

class SupplyChainDataGenerator:
    """
    Generates comprehensive supply chain datasets for TechManufacture Inc.
    
    This class creates interconnected datasets that simulate:
    - Product catalog with bill of materials
    - Warehouse network with capacity constraints
    - Supplier ecosystem with performance variability
    - Historical sales with seasonal patterns
    - Inventory transactions and current stock levels
    - Purchase orders and procurement history
    """
    
    def __init__(self, base_path: str = "data/raw"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Business parameters
        self.n_skus = 200
        self.n_warehouses = 15
        self.n_suppliers = 50
        self.n_customers = 5000
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2025, 12, 31)
        self.days = (self.end_date - self.start_date).days
        
        # Product categories for electronics manufacturer
        self.categories = ['Smartphones', 'Tablets', 'Laptops', 'Accessories', 
                          'Components', 'Peripherals', 'Wearables']
        
        # Initialize data containers
        self.skus_df = None
        self.warehouses_df = None
        self.suppliers_df = None
        
    def generate_all_data(self):
        """
        Master function to generate all supply chain datasets.
        Maintains referential integrity across all tables.
        """
        print("[FACTORY] Generating Supply Chain Data for TechManufacture Inc...")
        print("=" * 70)
        
        # Master data (must be generated first for foreign key relationships)
        self.generate_sku_master()
        self.generate_warehouse_master()
        self.generate_supplier_master()
        
        # Transactional data (depends on master data)
        self.generate_historical_sales()
        self.generate_inventory_transactions()
        self.generate_purchase_orders()
        self.generate_supplier_performance()
        
        # External data (market factors)
        self.generate_economic_indicators()
        self.generate_promotional_calendar()
        
        # Bill of Materials (product structure)
        self.generate_bom_structure()
        
        # Create SQLite database
        self.create_database()
        
        print("\n[CHECK] All data generation complete!")
        print(f"[FOLDER] Data saved to: {self.base_path.absolute()}")
        
    def generate_sku_master(self):
        """
        Generate product master data (SKU catalog).
        
        BUSINESS MEANING:
        - SKU (Stock Keeping Unit): Unique identifier for each product
        - Unit cost: Manufacturing/procurement cost per item
        - Unit price: Selling price to customers
        - Lead time: Days from order to delivery
        - Carrying cost rate: Annual cost to hold inventory (% of unit cost)
        
        WHY IT MATTERS:
        Different products have different inventory characteristics:
        - High-value items (laptops) need lower safety stock
        - Fast-moving items (accessories) need frequent replenishment
        - Long lead time items need higher safety stock
        """
        print("\n[PACKAGE] Generating SKU Master Data...")
        
        skus = []
        for i in range(self.n_skus):
            category = np.random.choice(self.categories)
            
            # Cost structure varies by category
            if category in ['Smartphones', 'Laptops', 'Tablets']:
                unit_cost = np.random.uniform(150, 800)
                unit_price = unit_cost * np.random.uniform(1.3, 1.6)
                lead_time = np.random.randint(60, 120)
            elif category == 'Components':
                unit_cost = np.random.uniform(5, 50)
                unit_price = unit_cost * np.random.uniform(1.4, 1.8)
                lead_time = np.random.randint(40, 90)
            else:
                unit_cost = np.random.uniform(10, 100)
                unit_price = unit_cost * np.random.uniform(1.35, 1.7)
                lead_time = np.random.randint(30, 80)
            
            skus.append({
                'sku_id': f'SKU{i+1:04d}',
                'product_name': f'{category}_{i+1:04d}',
                'category': category,
                'unit_cost': round(unit_cost, 2),
                'unit_price': round(unit_price, 2),
                'lead_time_days': lead_time,
                'carrying_cost_rate': round(np.random.uniform(0.15, 0.25), 3),  # 15-25% annual
                'weight_kg': round(np.random.uniform(0.1, 5.0), 2),
                'is_active': np.random.choice([True, False], p=[0.95, 0.05])
            })
        
        self.skus_df = pd.DataFrame(skus)
        self.skus_df.to_csv(self.base_path / 'sku_master.csv', index=False)
        print(f"   [OK] Created {len(self.skus_df)} SKUs across {len(self.categories)} categories")
        
    def generate_warehouse_master(self):
        """
        Generate warehouse network data.
        
        BUSINESS MEANING:
        - Warehouse capacity: Maximum units that can be stored
        - Operating cost: Monthly fixed cost to run the warehouse
        - Utilization rate: Current usage vs capacity
        
        WHY IT MATTERS:
        Warehouse consolidation analysis requires understanding:
        - Which warehouses are underutilized (candidates for closure)
        - Which are over capacity (need expansion or redistribution)
        - Cost per unit stored (efficiency metric)
        """
        print("\n[BUILDING] Generating Warehouse Master Data...")
        
        us_cities = ['Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
                     'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin',
                     'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte', 'Seattle']
        
        warehouses = []
        for i, city in enumerate(us_cities):
            # Larger cities have bigger warehouses
            capacity = np.random.randint(50000, 200000)
            
            warehouses.append({
                'warehouse_id': f'WH{i+1:02d}',
                'warehouse_name': f'{city} Distribution Center',
                'city': city,
                'state': 'Various',  # Simplified
                'capacity_units': capacity,
                'monthly_operating_cost': round(capacity * np.random.uniform(0.5, 1.2), 2),
                'cost_per_unit_stored': round(np.random.uniform(0.8, 2.5), 2),
                'current_utilization_rate': round(np.random.uniform(0.4, 0.95), 2)
            })
        
        self.warehouses_df = pd.DataFrame(warehouses)
        self.warehouses_df.to_csv(self.base_path / 'warehouse_master.csv', index=False)
        print(f"   [OK] Created {len(self.warehouses_df)} warehouses across US")
        
    def generate_supplier_master(self):
        """
        Generate supplier ecosystem data.
        
        BUSINESS MEANING:
        - On-time delivery rate: % of orders delivered by promised date
        - Quality rating: Product quality score (1-5)
        - Payment terms: Days to pay invoice (affects cash flow)
        
        WHY IT MATTERS:
        Supplier selection involves trade-offs:
        - Cheap but unreliable suppliers cause stockouts
        - Expensive but reliable suppliers increase costs
        - We need to quantify the total cost of ownership
        """
        print("\n[TRUCK] Generating Supplier Master Data...")
        
        supplier_types = ['Manufacturer', 'Distributor', 'OEM', 'Contract Manufacturer']
        
        suppliers = []
        for i in range(self.n_suppliers):
            # Create realistic supplier performance distribution
            # 20% excellent, 50% good, 25% average, 5% poor
            performance_tier = np.random.choice(['excellent', 'good', 'average', 'poor'],
                                               p=[0.20, 0.50, 0.25, 0.05])
            
            if performance_tier == 'excellent':
                on_time_rate = np.random.uniform(0.92, 0.98)
                quality_rating = np.random.uniform(4.5, 5.0)
            elif performance_tier == 'good':
                on_time_rate = np.random.uniform(0.80, 0.92)
                quality_rating = np.random.uniform(3.8, 4.5)
            elif performance_tier == 'average':
                on_time_rate = np.random.uniform(0.65, 0.80)
                quality_rating = np.random.uniform(3.0, 3.8)
            else:  # poor
                on_time_rate = np.random.uniform(0.40, 0.65)
                quality_rating = np.random.uniform(2.0, 3.0)
            
            suppliers.append({
                'supplier_id': f'SUP{i+1:03d}',
                'supplier_name': f'Supplier_{i+1:03d}',
                'supplier_type': np.random.choice(supplier_types),
                'country': np.random.choice(['USA', 'China', 'Taiwan', 'South Korea', 'Japan'],
                                           p=[0.3, 0.35, 0.15, 0.1, 0.1]),
                'on_time_delivery_rate': round(on_time_rate, 3),
                'quality_rating': round(quality_rating, 2),
                'payment_terms_days': np.random.choice([30, 45, 60, 90]),
                'minimum_order_quantity': np.random.randint(100, 5000),
                'is_preferred': performance_tier in ['excellent', 'good']
            })
        
        self.suppliers_df = pd.DataFrame(suppliers)
        self.suppliers_df.to_csv(self.base_path / 'supplier_master.csv', index=False)
        print(f"   [OK] Created {len(self.suppliers_df)} suppliers with varying performance")
        
    def generate_historical_sales(self):
        """
        Generate 24 months of daily sales data with realistic patterns.
        
        BUSINESS MEANING:
        - Quantity ordered: Customer demand
        - Quantity fulfilled: What we actually shipped (may be less due to stockouts)
        - Backorder quantity: Unfulfilled demand
        
        WHY IT MATTERS:
        This is the foundation for demand forecasting:
        - Seasonal patterns (Q4 spike for electronics)
        - Day-of-week effects (B2B orders on weekdays)
        - Promotional impacts (sales spikes during campaigns)
        - Trend (growing or declining products)
        
        Forecast accuracy directly impacts inventory costs and service levels.
        """
        print("\n[CHART] Generating Historical Sales Data (24 months)...")
        
        sales_records = []
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        for sku_id in self.skus_df['sku_id']:
            sku_data = self.skus_df[self.skus_df['sku_id'] == sku_id].iloc[0]
            category = sku_data['category']
            
            # Base demand varies by category
            if category in ['Smartphones', 'Laptops']:
                base_demand = np.random.uniform(20, 80)
            elif category == 'Accessories':
                base_demand = np.random.uniform(50, 150)
            else:
                base_demand = np.random.uniform(10, 50)
            
            for date in date_range:
                # Seasonal multiplier (Q4 spike for electronics)
                month = date.month
                if month in [11, 12]:  # Holiday season
                    seasonal_factor = 1.8
                elif month in [1, 2]:  # Post-holiday slump
                    seasonal_factor = 0.7
                elif month in [6, 7, 8]:  # Summer
                    seasonal_factor = 1.2
                else:
                    seasonal_factor = 1.0
                
                # Day of week effect (lower on weekends)
                dow_factor = 0.6 if date.weekday() >= 5 else 1.0
                
                # Random noise
                noise = np.random.normal(1.0, 0.3)
                
                # Calculate demand
                demand = max(0, base_demand * seasonal_factor * dow_factor * noise)
                quantity_ordered = int(np.random.poisson(demand))
                
                if quantity_ordered > 0:
                    # Simulate stockouts (10% of time, can't fulfill full order)
                    if np.random.random() < 0.10:
                        fulfillment_rate = np.random.uniform(0.5, 0.9)
                        quantity_fulfilled = int(quantity_ordered * fulfillment_rate)
                        backorder_qty = quantity_ordered - quantity_fulfilled
                    else:
                        quantity_fulfilled = quantity_ordered
                        backorder_qty = 0
                    
                    sales_records.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'sku_id': sku_id,
                        'category': category,
                        'quantity_ordered': quantity_ordered,
                        'quantity_fulfilled': quantity_fulfilled,
                        'backorder_quantity': backorder_qty,
                        'unit_price': sku_data['unit_price'],
                        'revenue': round(quantity_fulfilled * sku_data['unit_price'], 2),
                        'customer_segment': np.random.choice(['B2B', 'B2C', 'Distributor'],
                                                            p=[0.5, 0.3, 0.2]),
                        'region': np.random.choice(['West', 'East', 'Central', 'South'])
                    })
        
        sales_df = pd.DataFrame(sales_records)
        sales_df.to_csv(self.base_path / 'historical_sales.csv', index=False)
        print(f"   [OK] Created {len(sales_df):,} sales transactions over 24 months")
        
    def generate_inventory_transactions(self):
        """
        Generate inventory movement records.
        
        BUSINESS MEANING:
        - Receipt: Inventory coming in from suppliers
        - Shipment: Inventory going out to customers
        - Adjustment: Corrections (damage, theft, count errors)
        - Transfer: Movement between warehouses
        
        WHY IT MATTERS:
        Inventory transactions reveal:
        - Inventory turnover rates (how fast products move)
        - Slow-moving items (candidates for clearance)
        - Warehouse imbalances (some overstocked, others understocked)
        - Carrying costs (tied-up capital)
        """
        print("\n[PACKAGE] Generating Inventory Transactions (500K+ records)...")
        
        transactions = []
        transaction_types = ['RECEIPT', 'SHIPMENT', 'ADJUSTMENT', 'TRANSFER']
        
        # Generate transactions for each SKU-Warehouse combination
        for sku_id in self.skus_df['sku_id'].sample(n=150):  # Subset for performance
            sku_cost = self.skus_df[self.skus_df['sku_id'] == sku_id]['unit_cost'].values[0]
            
            for warehouse_id in self.warehouses_df['warehouse_id'].sample(n=10):
                # Generate 50-100 transactions per SKU-Warehouse over 24 months
                n_transactions = np.random.randint(50, 100)
                
                for _ in range(n_transactions):
                    trans_type = np.random.choice(transaction_types, p=[0.35, 0.45, 0.10, 0.10])
                    trans_date = self.start_date + timedelta(days=np.random.randint(0, self.days))
                    
                    if trans_type == 'RECEIPT':
                        quantity = np.random.randint(100, 2000)
                    elif trans_type == 'SHIPMENT':
                        quantity = -np.random.randint(50, 1500)
                    elif trans_type == 'ADJUSTMENT':
                        quantity = np.random.randint(-100, 100)
                    else:  # TRANSFER
                        quantity = np.random.choice([-1, 1]) * np.random.randint(50, 500)
                    
                    transactions.append({
                        'transaction_id': f'TXN{len(transactions)+1:07d}',
                        'transaction_date': trans_date.strftime('%Y-%m-%d'),
                        'sku_id': sku_id,
                        'warehouse_id': warehouse_id,
                        'transaction_type': trans_type,
                        'quantity': quantity,
                        'unit_cost': sku_cost,
                        'total_value': round(abs(quantity) * sku_cost, 2)
                    })
        
        inventory_df = pd.DataFrame(transactions)
        inventory_df = inventory_df.sort_values('transaction_date')
        inventory_df.to_csv(self.base_path / 'inventory_transactions.csv', index=False)
        print(f"   [OK] Created {len(inventory_df):,} inventory transactions")
        
    def generate_purchase_orders(self):
        """
        Generate purchase order history.
        
        BUSINESS MEANING:
        - PO (Purchase Order): Formal request to supplier for goods
        - Order quantity: How much we ordered
        - Received quantity: What actually arrived (may differ due to quality issues)
        - Lead time actual: Days from PO to delivery
        
        WHY IT MATTERS:
        PO data enables:
        - Supplier performance analysis (on-time delivery, quality)
        - Lead time variability analysis (impacts safety stock)
        - Price trend analysis (are costs increasing?)
        - Total cost of ownership calculation
        """
        print("\n[CLIPBOARD] Generating Purchase Orders...")
        
        purchase_orders = []
        
        for i in range(5000):  # 5000 POs over 24 months
            sku_id = np.random.choice(self.skus_df['sku_id'])
            supplier_id = np.random.choice(self.suppliers_df['supplier_id'])
            
            sku_data = self.skus_df[self.skus_df['sku_id'] == sku_id].iloc[0]
            supplier_data = self.suppliers_df[self.suppliers_df['supplier_id'] == supplier_id].iloc[0]
            
            order_date = self.start_date + timedelta(days=np.random.randint(0, self.days - 120))
            expected_lead_time = int(sku_data['lead_time_days'])  # Convert to Python int
            
            # Actual lead time varies based on supplier performance
            if supplier_data['on_time_delivery_rate'] > 0.9:
                actual_lead_time = int(expected_lead_time * np.random.uniform(0.9, 1.1))
            else:
                actual_lead_time = int(expected_lead_time * np.random.uniform(1.0, 1.5))
            
            delivery_date = order_date + timedelta(days=actual_lead_time)
            order_qty = np.random.randint(500, 5000)
            
            # Quality issues may reduce received quantity
            if np.random.random() < (1 - supplier_data['quality_rating'] / 5.0):
                received_qty = int(order_qty * np.random.uniform(0.85, 0.98))
            else:
                received_qty = order_qty
            
            purchase_orders.append({
                'po_number': f'PO{i+1:06d}',
                'order_date': order_date.strftime('%Y-%m-%d'),
                'expected_delivery_date': (order_date + timedelta(days=expected_lead_time)).strftime('%Y-%m-%d'),
                'actual_delivery_date': delivery_date.strftime('%Y-%m-%d'),
                'sku_id': sku_id,
                'supplier_id': supplier_id,
                'order_quantity': order_qty,
                'received_quantity': received_qty,
                'unit_cost': sku_data['unit_cost'],
                'total_cost': round(order_qty * sku_data['unit_cost'], 2),
                'lead_time_expected': expected_lead_time,
                'lead_time_actual': actual_lead_time,
                'on_time': actual_lead_time <= expected_lead_time,
                'quality_accepted': received_qty == order_qty
            })
        
        po_df = pd.DataFrame(purchase_orders)
        
        # Save as JSON (simulating vendor management system export)
        po_df.to_json(self.base_path / 'purchase_orders.json', orient='records', indent=2)
        print(f"   [OK] Created {len(po_df):,} purchase orders")
        
    def generate_supplier_performance(self):
        """
        Generate detailed supplier performance metrics.
        
        WHY IT MATTERS:
        Supplier scorecards enable data-driven supplier selection and negotiation.
        """
        print("\n[STAR] Generating Supplier Performance Metrics...")
        
        performance_records = []
        
        for supplier_id in self.suppliers_df['supplier_id']:
            supplier_data = self.suppliers_df[self.suppliers_df['supplier_id'] == supplier_id].iloc[0]
            
            # Monthly performance for 24 months
            for month_offset in range(24):
                month_date = self.start_date + timedelta(days=month_offset * 30)
                
                performance_records.append({
                    'supplier_id': supplier_id,
                    'month': month_date.strftime('%Y-%m'),
                    'on_time_delivery_rate': round(supplier_data['on_time_delivery_rate'] + 
                                                   np.random.uniform(-0.05, 0.05), 3),
                    'quality_rating': round(supplier_data['quality_rating'] + 
                                           np.random.uniform(-0.2, 0.2), 2),
                    'lead_time_avg_days': int(np.random.uniform(40, 120)),
                    'lead_time_variance': round(np.random.uniform(5, 30), 1),
                    'defect_rate': round(np.random.uniform(0.001, 0.05), 4),
                    'total_orders': np.random.randint(5, 50),
                    'total_spend': round(np.random.uniform(50000, 500000), 2)
                })
        
        perf_df = pd.DataFrame(performance_records)
        perf_df.to_json(self.base_path / 'supplier_performance.json', orient='records', indent=2)
        print(f"   [OK] Created {len(perf_df):,} supplier performance records")
        
    def generate_economic_indicators(self):
        """
        Generate external economic data that may impact demand.
        
        WHY IT MATTERS:
        Advanced forecasting models can use external regressors:
        - GDP growth correlates with electronics demand
        - Manufacturing PMI indicates business confidence
        - Currency rates affect import costs
        """
        print("\n[GRAPH] Generating Economic Indicators...")
        
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        
        economic_data = []
        gdp_base = 100
        pmi_base = 52
        
        for i, date in enumerate(date_range):
            economic_data.append({
                'month': date.strftime('%Y-%m'),
                'gdp_index': round(gdp_base + i * 0.5 + np.random.uniform(-2, 2), 2),
                'manufacturing_pmi': round(pmi_base + np.random.uniform(-5, 5), 1),
                'usd_cny_rate': round(6.5 + np.random.uniform(-0.3, 0.3), 3),
                'consumer_confidence': round(np.random.uniform(95, 115), 1)
            })
        
        econ_df = pd.DataFrame(economic_data)
        econ_df.to_csv(self.base_path / 'economic_indicators.csv', index=False)
        print(f"   [OK] Created {len(econ_df)} months of economic data")
        
    def generate_promotional_calendar(self):
        """
        Generate promotional campaign calendar.
        
        WHY IT MATTERS:
        Promotions cause demand spikes. Forecasting models must account for:
        - Planned promotions (known in advance)
        - Promotion lift (how much demand increases)
        - Post-promotion slump
        """
        print("\n[PARTY] Generating Promotional Calendar...")
        
        promotions = []
        promo_events = [
            ('New Year Sale', '01-01', '01-15', 0.3),
            ('Spring Sale', '03-15', '03-31', 0.2),
            ('Summer Clearance', '07-01', '07-31', 0.25),
            ('Back to School', '08-15', '09-05', 0.35),
            ('Black Friday', '11-25', '11-30', 0.5),
            ('Cyber Monday', '11-28', '12-02', 0.45),
            ('Holiday Sale', '12-15', '12-31', 0.4)
        ]
        
        for year in [2024, 2025]:
            for name, start_md, end_md, discount in promo_events:
                promotions.append({
                    'promotion_name': name,
                    'start_date': f'{year}-{start_md}',
                    'end_date': f'{year}-{end_md}',
                    'discount_rate': discount,
                    'expected_demand_lift': round(np.random.uniform(1.3, 2.0), 2),
                    'categories_included': ','.join(np.random.choice(self.categories, 
                                                                     size=np.random.randint(2, 5), 
                                                                     replace=False))
                })
        
        promo_df = pd.DataFrame(promotions)
        promo_df.to_csv(self.base_path / 'promotional_calendar.csv', index=False)
        print(f"   [OK] Created {len(promo_df)} promotional campaigns")
        
    def generate_bom_structure(self):
        """
        Generate Bill of Materials (product component relationships).
        
        BUSINESS MEANING:
        - Parent SKU: Finished product
        - Component SKU: Parts needed to build the product
        - Quantity per: How many components needed per finished product
        
        WHY IT MATTERS:
        BOM analysis enables:
        - Component demand forecasting (derived from finished goods forecast)
        - Supply chain risk analysis (single component shortage stops production)
        - Cost rollup (total product cost from component costs)
        """
        print("\n[WRENCH] Generating Bill of Materials...")
        
        bom_records = []
        
        # Only finished goods have BOMs (Smartphones, Laptops, Tablets)
        finished_goods = self.skus_df[self.skus_df['category'].isin(['Smartphones', 'Laptops', 'Tablets'])]
        components = self.skus_df[self.skus_df['category'] == 'Components']
        
        for parent_sku in finished_goods['sku_id']:
            # Each finished good has 5-15 components
            n_components = np.random.randint(5, 15)
            selected_components = components.sample(n=n_components)
            
            for component_sku in selected_components['sku_id']:
                bom_records.append({
                    'parent_sku': parent_sku,
                    'component_sku': component_sku,
                    'quantity_per': np.random.randint(1, 5),
                    'is_critical': np.random.choice([True, False], p=[0.3, 0.7])
                })
        
        bom_df = pd.DataFrame(bom_records)
        bom_df.to_csv(self.base_path / 'bom_structure.csv', index=False)
        print(f"   [OK] Created {len(bom_df)} BOM relationships")
        
    def create_database(self):
        """
        Create SQLite database with all tables.
        
        WHY IT MATTERS:
        SQL database enables:
        - Complex analytical queries (joins, window functions, CTEs)
        - Realistic data extraction practice
        - Demonstration of SQL skills for interviews
        """
        print("\n[DISK] Creating SQLite Database...")
        
        db_path = self.base_path / 'supply_chain.db'
        conn = sqlite3.connect(db_path)
        
        # Load all CSV files into database
        self.skus_df.to_sql('sku_master', conn, if_exists='replace', index=False)
        self.warehouses_df.to_sql('warehouse_master', conn, if_exists='replace', index=False)
        self.suppliers_df.to_sql('supplier_master', conn, if_exists='replace', index=False)
        
        # Load other datasets
        pd.read_csv(self.base_path / 'historical_sales.csv').to_sql('sales', conn, if_exists='replace', index=False)
        pd.read_csv(self.base_path / 'inventory_transactions.csv').to_sql('inventory_transactions', conn, if_exists='replace', index=False)
        pd.read_csv(self.base_path / 'bom_structure.csv').to_sql('bom', conn, if_exists='replace', index=False)
        
        # Create current inventory snapshot
        self.create_current_inventory_table(conn)
        
        conn.close()
        print(f"   [OK] Database created: {db_path}")
        
    def create_current_inventory_table(self, conn):
        """
        Create current inventory levels (as of end date).
        
        WHY IT MATTERS:
        Current inventory is the starting point for:
        - Inventory optimization (what to order now)
        - ABC classification
        - Slow-moving inventory identification
        """
        print("   [CHART] Calculating current inventory levels...")
        
        current_inventory = []
        
        for sku_id in self.skus_df['sku_id'].sample(n=150):
            for warehouse_id in self.warehouses_df['warehouse_id'].sample(n=10):
                # Simulate current stock level
                on_hand = np.random.randint(0, 5000)
                sku_data = self.skus_df[self.skus_df['sku_id'] == sku_id].iloc[0]
                
                current_inventory.append({
                    'sku_id': sku_id,
                    'warehouse_id': warehouse_id,
                    'quantity_on_hand': on_hand,
                    'quantity_allocated': int(on_hand * np.random.uniform(0, 0.3)),
                    'quantity_available': int(on_hand * np.random.uniform(0.7, 1.0)),
                    'reorder_point': int(np.random.uniform(500, 2000)),
                    'safety_stock': int(np.random.uniform(200, 1000)),
                    'last_count_date': (self.end_date - timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d'),
                    'unit_cost': sku_data['unit_cost'],
                    'total_value': round(on_hand * sku_data['unit_cost'], 2)
                })
        
        inv_df = pd.DataFrame(current_inventory)
        inv_df.to_sql('current_inventory', conn, if_exists='replace', index=False)
        print(f"   [OK] Created current inventory snapshot: {len(inv_df)} records")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("  SUPPLY CHAIN DATA GENERATION FOR TECHMANUFACTURE INC.")
    print("="*70)
    
    generator = SupplyChainDataGenerator()
    generator.generate_all_data()
    
    print("\n" + "="*70)
    print("  DATA GENERATION SUMMARY")
    print("="*70)
    print("\n[FOLDER] Files Created:")
    print("   - sku_master.csv - Product catalog (200+ SKUs)")
    print("   - warehouse_master.csv - Warehouse network (15 locations)")
    print("   - supplier_master.csv - Supplier ecosystem (50+ suppliers)")
    print("   - historical_sales.csv - 24 months of sales data")
    print("   - inventory_transactions.csv - 500K+ inventory movements")
    print("   - purchase_orders.json - Procurement history")
    print("   - supplier_performance.json - Supplier metrics")
    print("   - economic_indicators.csv - External market data")
    print("   - promotional_calendar.csv - Marketing campaigns")
    print("   - bom_structure.csv - Product component relationships")
    print("   - supply_chain.db - SQLite database with all tables")
    
    print("\n[TARGET] Next Steps:")
    print("   1. Run SQL analysis queries (sql/supply_chain_queries.sql)")
    print("   2. Perform exploratory data analysis (src/exploratory_analysis.py)")
    print("   3. Build demand forecasting models (src/demand_forecasting.py)")
    print("   4. Optimize inventory levels (src/optimization.py)")
    print("   5. Generate insights and recommendations")
    
    print("\n[CHECK] Ready for analysis!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
