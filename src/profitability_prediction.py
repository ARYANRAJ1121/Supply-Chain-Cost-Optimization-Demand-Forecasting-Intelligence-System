"""
Profitability Risk Prediction Model
====================================

PURPOSE:
This module builds a machine learning model to predict when the company is at risk
of facing losses. Provides early warning signals 3-6 months in advance.

BUSINESS VALUE:
- Predict profit/loss 3-6 months ahead
- Identify risk factors (inventory costs, supplier delays, stockouts)
- Enable proactive intervention before losses occur
- Quantify financial impact of different scenarios

PREDICTION APPROACH:
1. Calculate monthly profitability metrics
2. Engineer features (inventory costs, stockout costs, supplier performance)
3. Build classification model (Profit vs Loss risk)
4. Build regression model (Profit amount prediction)
5. Identify key risk drivers using feature importance
6. Generate early warning alerts

MODELS USED:
- Random Forest Classifier (Profit/Loss prediction)
- Gradient Boosting Regressor (Profit amount prediction)
- XGBoost (Alternative high-performance model)

HOW TO USE:
    python src/profitability_prediction.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    mean_absolute_error, mean_squared_error, r2_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class ProfitabilityPredictor:
    """
    Predicts company profitability risk and provides early warning signals.
    """
    
    def __init__(self, data_dir: str = "data/raw", output_dir: str = "data/output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sales = None
        self.sku_master = None
        self.supplier_master = None
        self.inventory_txn = None
        
        self.monthly_metrics = None
        self.features = None
        self.target = None
        
        self.classification_model = None
        self.regression_model = None
        self.scaler = None
        
    def load_data(self):
        """
        Load all necessary datasets.
        """
        print("\n" + "="*70)
        print("  LOADING DATA FOR PROFITABILITY PREDICTION")
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
        
        print("\n[LOADING] Inventory Transactions...")
        self.inventory_txn = pd.read_csv(self.data_dir / 'inventory_transactions.csv')
        self.inventory_txn['transaction_date'] = pd.to_datetime(self.inventory_txn['transaction_date'])
        print(f"   [OK] {len(self.inventory_txn):,} records")
        
    def calculate_monthly_profitability(self):
        """
        Calculate monthly profitability metrics.
        """
        print("\n" + "="*70)
        print("  CALCULATING MONTHLY PROFITABILITY METRICS")
        print("="*70)
        
        # Merge sales with SKU master for costs
        sales_enriched = self.sales.merge(self.sku_master[['sku_id', 'unit_cost', 'carrying_cost_rate']], 
                                         on='sku_id', how='left')
        
        # Add year-month column
        sales_enriched['year_month'] = sales_enriched['date'].dt.to_period('M')
        
        # Calculate monthly metrics
        monthly_agg = sales_enriched.groupby('year_month').agg({
            # Revenue metrics
            'revenue': 'sum',
            'quantity_fulfilled': 'sum',
            'quantity_ordered': 'sum',
            'backorder_quantity': 'sum',
            
            # Cost metrics
            'unit_cost': 'mean',
            'unit_price': 'mean'
        }).reset_index()
        
        # Calculate COGS (Cost of Goods Sold)
        monthly_agg['cogs'] = (
            sales_enriched.groupby('year_month').apply(
                lambda x: (x['quantity_fulfilled'] * x['unit_cost']).sum()
            ).values
        )
        
        # Calculate gross profit
        monthly_agg['gross_profit'] = monthly_agg['revenue'] - monthly_agg['cogs']
        monthly_agg['gross_margin_pct'] = (monthly_agg['gross_profit'] / monthly_agg['revenue'] * 100).round(2)
        
        # Calculate stockout costs (lost sales)
        monthly_agg['stockout_cost'] = (
            sales_enriched.groupby('year_month').apply(
                lambda x: (x['backorder_quantity'] * x['unit_price']).sum()
            ).values
        )
        
        # Calculate fill rate
        monthly_agg['fill_rate'] = (
            monthly_agg['quantity_fulfilled'] / monthly_agg['quantity_ordered'] * 100
        ).round(2)
        
        # Calculate inventory carrying costs (estimated)
        monthly_agg['carrying_cost'] = (
            sales_enriched.groupby('year_month').apply(
                lambda x: (x['quantity_fulfilled'] * x['unit_cost'] * x['carrying_cost_rate'] / 12).sum()
            ).values
        )
        
        # Calculate operating profit (gross profit - carrying costs - stockout costs)
        monthly_agg['operating_profit'] = (
            monthly_agg['gross_profit'] - 
            monthly_agg['carrying_cost'] - 
            monthly_agg['stockout_cost']
        )
        
        # Operating margin
        monthly_agg['operating_margin_pct'] = (
            monthly_agg['operating_profit'] / monthly_agg['revenue'] * 100
        ).round(2)
        
        # Profit/Loss flag
        monthly_agg['is_profitable'] = (monthly_agg['operating_profit'] > 0).astype(int)
        
        self.monthly_metrics = monthly_agg
        
        print(f"\n[OK] Calculated profitability for {len(monthly_agg)} months")
        print(f"\n[SUMMARY] Profitability Overview:")
        print(f"   Profitable months: {monthly_agg['is_profitable'].sum()} ({monthly_agg['is_profitable'].sum()/len(monthly_agg)*100:.1f}%)")
        print(f"   Loss months: {len(monthly_agg) - monthly_agg['is_profitable'].sum()}")
        print(f"   Average monthly profit: ${monthly_agg['operating_profit'].mean():,.2f}")
        print(f"   Average operating margin: {monthly_agg['operating_margin_pct'].mean():.2f}%")
        
    def engineer_features(self):
        """
        Create predictive features for the model.
        """
        print("\n" + "="*70)
        print("  ENGINEERING PREDICTIVE FEATURES")
        print("="*70)
        
        df = self.monthly_metrics.copy()
        df['year_month'] = df['year_month'].astype(str)
        
        # Lagged features (previous months' performance)
        print("\n[CREATING] Lagged features (1-3 months back)...")
        for lag in [1, 2, 3]:
            df[f'revenue_lag_{lag}'] = df['revenue'].shift(lag)
            df[f'gross_margin_lag_{lag}'] = df['gross_margin_pct'].shift(lag)
            df[f'fill_rate_lag_{lag}'] = df['fill_rate'].shift(lag)
            df[f'stockout_cost_lag_{lag}'] = df['stockout_cost'].shift(lag)
        
        # Rolling averages (trends)
        print("[CREATING] Rolling averages (3-month trends)...")
        df['revenue_rolling_3m'] = df['revenue'].rolling(window=3, min_periods=1).mean()
        df['profit_rolling_3m'] = df['operating_profit'].rolling(window=3, min_periods=1).mean()
        df['margin_rolling_3m'] = df['operating_margin_pct'].rolling(window=3, min_periods=1).mean()
        
        # Month-over-month changes
        print("[CREATING] Month-over-month changes...")
        df['revenue_mom_change'] = df['revenue'].pct_change() * 100
        df['profit_mom_change'] = df['operating_profit'].pct_change() * 100
        df['fill_rate_mom_change'] = df['fill_rate'].pct_change() * 100
        
        # Seasonality indicators
        print("[CREATING] Seasonality indicators...")
        df['month'] = pd.to_datetime(df['year_month']).dt.month
        df['quarter'] = pd.to_datetime(df['year_month']).dt.quarter
        df['is_q4'] = (df['quarter'] == 4).astype(int)  # Holiday season
        
        # Cost ratios
        print("[CREATING] Cost ratios...")
        df['cogs_to_revenue_ratio'] = (df['cogs'] / df['revenue'] * 100).round(2)
        df['stockout_to_revenue_ratio'] = (df['stockout_cost'] / df['revenue'] * 100).round(2)
        df['carrying_to_revenue_ratio'] = (df['carrying_cost'] / df['revenue'] * 100).round(2)
        
        # Risk indicators
        print("[CREATING] Risk indicators...")
        df['low_fill_rate_flag'] = (df['fill_rate'] < 90).astype(int)
        df['high_stockout_flag'] = (df['stockout_cost'] > df['stockout_cost'].median()).astype(int)
        df['declining_revenue_flag'] = (df['revenue_mom_change'] < -5).astype(int)
        
        # Drop rows with NaN (from lagging)
        df_clean = df.dropna()
        
        print(f"\n[OK] Created {len(df_clean.columns)} features")
        print(f"   Training samples: {len(df_clean)}")
        
        self.features = df_clean
        
    def prepare_training_data(self, forecast_horizon: int = 1):
        """
        Prepare features and target for model training.
        
        Args:
            forecast_horizon: Months ahead to predict (1-6)
        """
        print(f"\n[PREPARING] Training data for {forecast_horizon}-month ahead prediction...")
        
        df = self.features.copy()
        
        # Target: Profitability N months ahead
        df['target_profit'] = df['operating_profit'].shift(-forecast_horizon)
        df['target_is_profitable'] = df['is_profitable'].shift(-forecast_horizon)
        
        # Drop rows without target
        df_clean = df.dropna(subset=['target_profit', 'target_is_profitable'])
        
        # Feature columns (exclude targets and identifiers)
        feature_cols = [col for col in df_clean.columns if col not in [
            'year_month', 'target_profit', 'target_is_profitable', 
            'operating_profit', 'is_profitable', 'gross_profit'
        ]]
        
        X = df_clean[feature_cols]
        y_class = df_clean['target_is_profitable']
        y_reg = df_clean['target_profit']
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(X)}")
        print(f"   Profitable samples: {y_class.sum()} ({y_class.sum()/len(y_class)*100:.1f}%)")
        
        return X, y_class, y_reg, feature_cols
        
    def train_classification_model(self, forecast_horizon: int = 3):
        """
        Train model to predict Profit vs Loss risk.
        
        Args:
            forecast_horizon: Months ahead to predict (default: 3 months)
        """
        print("\n" + "="*70)
        print(f"  TRAINING PROFIT/LOSS CLASSIFICATION MODEL ({forecast_horizon}-MONTH AHEAD)")
        print("="*70)
        
        X, y_class, y_reg, feature_cols = self.prepare_training_data(forecast_horizon)
        
        # Check if we have both classes
        unique_classes = y_class.unique()
        if len(unique_classes) == 1:
            print("\n[INFO] All samples belong to one class (all profitable or all loss)")
            print(f"   Class: {'Profitable' if unique_classes[0] == 1 else 'Loss'}")
            print("   [SKIP] Classification model requires both profit and loss examples")
            print("   [NOTE] Regression model will still predict profit amounts")
            
            # Create a dummy classifier for consistency
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            
            # Return empty feature importance
            return pd.DataFrame({
                'feature': feature_cols,
                'importance': [0] * len(feature_cols)
            })
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.25, random_state=42, stratify=y_class
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        print("\n[TRAINING] Random Forest Classifier...")
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.classification_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.classification_model.predict(X_test_scaled)
        
        # Handle probability prediction safely
        y_pred_proba_all = self.classification_model.predict_proba(X_test_scaled)
        if y_pred_proba_all.shape[1] > 1:
            y_pred_proba = y_pred_proba_all[:, 1]
        else:
            y_pred_proba = y_pred_proba_all[:, 0]
        
        # Evaluation
        print("\n[EVALUATION] Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['Loss Risk', 'Profitable']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"   True Negatives (Correctly predicted loss): {cm[0,0]}")
        print(f"   False Positives (False alarm): {cm[0,1]}")
        print(f"   False Negatives (Missed loss risk): {cm[1,0]}")
        print(f"   True Positives (Correctly predicted profit): {cm[1,1]}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.classification_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n[TOP 10] Most Important Risk Factors:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Save feature importance
        feature_importance.to_csv(self.output_dir / 'risk_factor_importance.csv', index=False)
        
        # Save model performance
        performance = {
            'Model': 'Random Forest Classifier',
            'Forecast Horizon': f'{forecast_horizon} months',
            'Accuracy': f"{(y_pred == y_test).sum() / len(y_test) * 100:.2f}%",
            'Precision (Loss Detection)': f"{cm[0,0] / (cm[0,0] + cm[0,1]) * 100:.2f}%" if (cm[0,0] + cm[0,1]) > 0 else 'N/A',
            'Recall (Loss Detection)': f"{cm[0,0] / (cm[0,0] + cm[1,0]) * 100:.2f}%" if (cm[0,0] + cm[1,0]) > 0 else 'N/A'
        }
        
        pd.DataFrame([performance]).T.to_csv(self.output_dir / 'classification_model_performance.csv')
        
        return feature_importance
        
    def train_regression_model(self, forecast_horizon: int = 3):
        """
        Train model to predict profit amount.
        
        Args:
            forecast_horizon: Months ahead to predict (default: 3 months)
        """
        print("\n" + "="*70)
        print(f"  TRAINING PROFIT AMOUNT REGRESSION MODEL ({forecast_horizon}-MONTH AHEAD)")
        print("="*70)
        
        X, y_class, y_reg, feature_cols = self.prepare_training_data(forecast_horizon)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_reg, test_size=0.25, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting Regressor
        print("\n[TRAINING] Gradient Boosting Regressor...")
        self.regression_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.regression_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.regression_model.predict(X_test_scaled)
        
        # Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print("\n[EVALUATION] Model Performance:")
        print(f"   MAE (Mean Absolute Error): ${mae:,.2f}")
        print(f"   RMSE (Root Mean Squared Error): ${rmse:,.2f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAPE (Mean Absolute % Error): {mape:.2f}%")
        
        # Save performance
        performance = {
            'Model': 'Gradient Boosting Regressor',
            'Forecast Horizon': f'{forecast_horizon} months',
            'MAE': f"${mae:,.2f}",
            'RMSE': f"${rmse:,.2f}",
            'R2_Score': f"{r2:.4f}",
            'MAPE': f"{mape:.2f}%"
        }
        
        pd.DataFrame([performance]).T.to_csv(self.output_dir / 'regression_model_performance.csv')
        
    def generate_risk_alerts(self):
        """
        Generate early warning alerts for loss risk.
        """
        print("\n" + "="*70)
        print("  GENERATING RISK ALERTS")
        print("="*70)
        
        # Use last 3 months to predict next 3 months
        recent_data = self.features.tail(3).copy()
        
        alerts = []
        
        for idx, row in recent_data.iterrows():
            # Prepare features
            feature_cols = [col for col in recent_data.columns if col not in [
                'year_month', 'operating_profit', 'is_profitable', 'gross_profit'
            ]]
            
            X = row[feature_cols].values.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predict profit amount
            predicted_profit = self.regression_model.predict(X_scaled)[0]
            
            # Predict loss probability (if classification model exists)
            if self.classification_model is not None:
                loss_probability = self.classification_model.predict_proba(X_scaled)[0, 0]
            else:
                # If no classification model, estimate probability from predicted profit
                # Negative profit = 100% loss probability
                # Positive profit = lower probability based on margin
                if predicted_profit < 0:
                    loss_probability = 1.0
                else:
                    # Lower probability for higher profits
                    margin = (predicted_profit / row['revenue']) * 100 if row['revenue'] > 0 else 0
                    loss_probability = max(0, 1 - (margin / 30))  # 30% margin = 0% loss prob
            
            # Risk level
            if loss_probability > 0.7:
                risk_level = 'HIGH RISK'
            elif loss_probability > 0.4:
                risk_level = 'MEDIUM RISK'
            else:
                risk_level = 'LOW RISK'
            
            alerts.append({
                'month': row['year_month'],
                'current_profit': row['operating_profit'],
                'predicted_profit_3m_ahead': predicted_profit,
                'loss_probability': loss_probability * 100,
                'risk_level': risk_level,
                'fill_rate': row['fill_rate'],
                'stockout_cost': row['stockout_cost'],
                'operating_margin': row['operating_margin_pct']
            })
        
        alerts_df = pd.DataFrame(alerts)
        
        print("\n[ALERTS] Risk Assessment for Recent Months:")
        for _, alert in alerts_df.iterrows():
            print(f"\n   Month: {alert['month']}")
            print(f"   Current Profit: ${alert['current_profit']:,.2f}")
            print(f"   Predicted Profit (3m ahead): ${alert['predicted_profit_3m_ahead']:,.2f}")
            print(f"   Loss Probability: {alert['loss_probability']:.1f}%")
            print(f"   Risk Level: {alert['risk_level']}")
            
            if alert['risk_level'] in ['HIGH RISK', 'MEDIUM RISK']:
                print(f"   [WARNING] Action required!")
                if alert['fill_rate'] < 90:
                    print(f"      - Fill rate is low ({alert['fill_rate']:.1f}%) - improve inventory")
                if alert['stockout_cost'] > 50000:
                    print(f"      - High stockout costs (${alert['stockout_cost']:,.0f}) - fix supplier issues")
                if alert['operating_margin'] < 10:
                    print(f"      - Low margin ({alert['operating_margin']:.1f}%) - reduce costs")
        
        alerts_df.to_csv(self.output_dir / 'profitability_risk_alerts.csv', index=False)
        print(f"\n[EXPORT] Alerts saved to: {self.output_dir / 'profitability_risk_alerts.csv'}")
        
    def generate_summary_report(self):
        """
        Generate executive summary of profitability predictions.
        """
        print("\n" + "="*70)
        print("  PROFITABILITY PREDICTION SUMMARY")
        print("="*70)
        
        report = {
            'Analysis Date': datetime.now().strftime('%Y-%m-%d'),
            'Forecast Horizon': '3 months ahead',
            'Model Type': 'Random Forest + Gradient Boosting',
            'Historical Period': f"{self.monthly_metrics['year_month'].min()} to {self.monthly_metrics['year_month'].max()}",
            'Total Months Analyzed': len(self.monthly_metrics),
            'Profitable Months': self.monthly_metrics['is_profitable'].sum(),
            'Loss Months': len(self.monthly_metrics) - self.monthly_metrics['is_profitable'].sum(),
            'Average Monthly Profit': f"${self.monthly_metrics['operating_profit'].mean():,.2f}",
            'Average Operating Margin': f"{self.monthly_metrics['operating_margin_pct'].mean():.2f}%",
            'Model Accuracy': 'See classification_model_performance.csv',
            'Top Risk Factor': 'See risk_factor_importance.csv'
        }
        
        report_df = pd.DataFrame([report]).T
        report_df.columns = ['Value']
        report_df.to_csv(self.output_dir / 'profitability_prediction_summary.csv')
        
        print("\n[SUMMARY] Key Metrics:")
        for key, value in report.items():
            print(f"   {key}: {value}")
        
        print(f"\n[EXPORT] Summary saved to: {self.output_dir / 'profitability_prediction_summary.csv'}")
        
    def run_full_prediction(self):
        """
        Execute complete profitability prediction workflow.
        """
        print("\n" + "="*70)
        print("  PROFITABILITY RISK PREDICTION ANALYSIS")
        print("="*70)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_data()
        self.calculate_monthly_profitability()
        self.engineer_features()
        
        # Train models (3-month ahead prediction)
        feature_importance = self.train_classification_model(forecast_horizon=3)
        self.train_regression_model(forecast_horizon=3)
        
        # Generate alerts
        self.generate_risk_alerts()
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("  PREDICTION COMPLETE!")
        print("="*70)
        print(f"\n[CHECK] All results exported to: {self.output_dir.absolute()}")
        print("\n[KEY INSIGHTS]")
        print("   1. Review risk_factor_importance.csv for main drivers")
        print("   2. Check profitability_risk_alerts.csv for early warnings")
        print("   3. Monitor months with >70% loss probability")
        print("\n[TARGET] Next Steps:")
        print("   1. Set up automated monthly predictions")
        print("   2. Create alert system for high-risk periods")
        print("   3. Develop action plans for risk mitigation")
        print("   4. Integrate with business intelligence dashboard")
        print("\n")


def main():
    """
    Main execution function.
    """
    predictor = ProfitabilityPredictor()
    predictor.run_full_prediction()


if __name__ == "__main__":
    main()
