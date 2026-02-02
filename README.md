# 🏭 Supply Chain Cost Optimization & Demand Forecasting Intelligence System

## 📋 Project Overview

This project tackles real-world operational challenges faced by **TechManufacture Inc.**, a mid-sized electronics manufacturer struggling with:
- **$8.5M locked in excess inventory** due to poor demand forecasting (28% accuracy)
- **$2.3M annual losses** from stockouts and expedited shipping
- **35% on-time supplier delivery rate** (target: 95%)
- **22% increase in warehouse costs** despite flat sales

### 🎯 Business Objectives

1. **Improve demand forecasting accuracy** from 28% to >75% using time-series models
2. **Optimize inventory levels** to reduce carrying costs while maintaining service levels
3. **Identify underperforming suppliers** and quantify financial impact
4. **Recommend warehouse consolidation** scenarios for cost reduction
5. **Calculate optimal reorder points and safety stock** for 200+ SKUs

---

## 🗂️ Project Structure

```
project 2/
├── data/
│   ├── raw/                    # Synthetic supply chain datasets
│   │   ├── sku_master.csv
│   │   ├── warehouse_master.csv
│   │   ├── supplier_master.csv
│   │   ├── historical_sales.csv
│   │   ├── inventory_transactions.csv
│   │   ├── purchase_orders.json
│   │   ├── supplier_performance.json
│   │   ├── economic_indicators.csv
│   │   ├── promotional_calendar.csv
│   │   ├── bom_structure.csv
│   │   └── supply_chain.db
│   ├── processed/              # Cleaned and transformed data
│   └── output/                 # Forecasts and results
├── src/
│   ├── data_generation.py      # Generate synthetic datasets
│   ├── data_cleaning.py        # ETL pipeline
│   ├── exploratory_analysis.py # EDA and statistical analysis
│   ├── demand_forecasting.py   # ARIMA, Prophet models
│   ├── inventory_optimization.py # EOQ, ROP, safety stock
│   ├── supplier_analytics.py   # Vendor scorecards
│   ├── cost_analysis.py        # TCO and what-if scenarios
│   └── visualization.py        # Dashboard and charts
├── sql/
│   └── supply_chain_queries.sql # Comprehensive SQL analysis
├── notebooks/
│   └── full_analysis.ipynb     # Interactive analysis
├── reports/
│   └── executive_summary.pdf   # Business recommendations
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Data Sources

### Master Data
- **SKU Master** (200+ products): Product catalog with costs, lead times, carrying costs
- **Warehouse Master** (15 locations): Capacity, operating costs, utilization rates
- **Supplier Master** (50+ suppliers): Performance ratings, payment terms, reliability

### Transactional Data
- **Historical Sales** (24 months): Daily sales with seasonal patterns and promotions
- **Inventory Transactions** (500K+ records): Receipts, shipments, adjustments, transfers
- **Purchase Orders** (5,000+ POs): Procurement history with lead time variance

### External Data
- **Economic Indicators**: GDP, manufacturing PMI, currency rates
- **Promotional Calendar**: Marketing campaigns and expected demand lift
- **Bill of Materials**: Product-component relationships

---

## 🛠️ Technologies Used

### Data Analysis & Processing
- **Python 3.x**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **SQLite**: Relational database for complex queries
- **SQLAlchemy**: Database connectivity

### Time-Series Forecasting
- **Facebook Prophet**: Multi-seasonality forecasting
- **ARIMA/SARIMA**: Statistical time-series models
- **Statsmodels**: Statistical analysis and validation

### Optimization
- **PuLP**: Linear programming for inventory optimization
- **SciPy**: Optimization algorithms

### Visualization
- **Matplotlib & Seaborn**: Statistical visualizations
- **Plotly**: Interactive dashboards
- **Dash**: Web-based analytics dashboard

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ARYANRAJ1121/Supply-Chain-Cost-Optimization-Demand-Forecasting-Intelligence-System.git
cd Supply-Chain-Cost-Optimization-Demand-Forecasting-Intelligence-System
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate synthetic data**
```bash
python src/data_generation.py
```

---

## 📈 Key Analyses

### 1. Demand Forecasting
- **Models**: ARIMA, SARIMA, Facebook Prophet
- **Features**: Seasonality, promotions, economic indicators
- **Validation**: MAPE, RMSE, MAE metrics
- **Output**: 90-day demand forecasts with confidence intervals

### 2. Inventory Optimization
- **ABC Classification**: Pareto analysis by value
- **Economic Order Quantity (EOQ)**: Optimal order sizes
- **Reorder Point (ROP)**: When to reorder
- **Safety Stock**: Buffer against demand variability

### 3. Supplier Performance Analysis
- **On-time Delivery Rate**: Reliability metrics
- **Lead Time Variance**: Consistency analysis
- **Quality Ratings**: Defect rates and acceptance
- **Total Cost of Ownership**: Comprehensive cost analysis

### 4. Cost Optimization
- **Carrying Costs**: Inventory holding expenses
- **Stockout Costs**: Lost sales and expedited shipping
- **Warehouse Consolidation**: Scenario modeling
- **What-If Analysis**: Impact of parameter changes

---

## 📊 Sample Insights (To Be Generated)

*This section will be populated after analysis completion with insights such as:*

- "Category A items account for 75% of inventory costs but only 8% have optimized reorder points"
- "Supplier 'TechParts Co' has 42% on-time delivery causing $340K in expedited shipping costs"
- "Prophet model achieved 78% forecast accuracy vs 28% baseline"
- "Warehouse consolidation could save $1.2M annually with <2% service level impact"

---

## 🎯 Business Recommendations

*To be completed after analysis*

---

## 📧 Contact

**Aryan Raj**
- GitHub: [@ARYANRAJ1121](https://github.com/ARYANRAJ1121)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/aryanraj11/)
- Email: your.email@example.com

---

## 📝 License

This project is created for portfolio demonstration purposes.

---

## 🙏 Acknowledgments

- Synthetic data generated to replicate real-world supply chain complexity
- Inspired by operational challenges in manufacturing and distribution
- Built to demonstrate advanced analytics skills for supply chain analyst roles

---

*Last Updated: February 2026*
