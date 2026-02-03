"""
Demand Forecasting Module
==========================

PURPOSE:
This module builds and evaluates time-series forecasting models for demand prediction.
Implements multiple forecasting approaches and compares their accuracy.

BUSINESS VALUE:
- Improve forecast accuracy from 28% to >75% (target)
- Reduce excess inventory and stockouts
- Enable proactive procurement planning
- Support data-driven inventory optimization

FORECASTING MODELS IMPLEMENTED:
1. Naive Baseline (last period's demand)
2. Moving Average (7-day, 30-day)
3. Exponential Smoothing
4. ARIMA/SARIMA (Statistical time-series)
5. Facebook Prophet (Industry-standard with seasonality)

EVALUATION METRICS:
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

HOW TO USE:
    python src/demand_forecasting.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time-series libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Prophet (Facebook's forecasting library)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("[WARNING] Prophet not installed. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

class DemandForecaster:
    """
    Builds and evaluates demand forecasting models for supply chain optimization.
    """
    
    def __init__(self, data_dir: str = "data/raw", output_dir: str = "data/output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sales = None
        self.daily_demand = None
        self.train_data = None
        self.test_data = None
        
        self.forecast_results = {}
        self.model_performance = []
        
    def load_and_prepare_data(self):
        """
        Load sales data and prepare for forecasting.
        """
        print("\n" + "="*70)
        print("  LOADING AND PREPARING DATA FOR FORECASTING")
        print("="*70)
        
        # Load sales data
        print("\n[LOADING] Historical sales data...")
        self.sales = pd.read_csv(self.data_dir / 'historical_sales.csv')
        self.sales['date'] = pd.to_datetime(self.sales['date'])
        print(f"   [OK] Loaded {len(self.sales):,} sales transactions")
        
        # Aggregate to daily level
        print("\n[PROCESSING] Aggregating to daily demand...")
        self.daily_demand = self.sales.groupby('date').agg({
            'quantity_fulfilled': 'sum',
            'revenue': 'sum'
        }).reset_index()
        self.daily_demand.columns = ['date', 'demand', 'revenue']
        
        # Sort by date
        self.daily_demand = self.daily_demand.sort_values('date').reset_index(drop=True)
        
        print(f"   [OK] Daily demand series: {len(self.daily_demand)} days")
        print(f"   Date range: {self.daily_demand['date'].min()} to {self.daily_demand['date'].max()}")
        print(f"   Average daily demand: {self.daily_demand['demand'].mean():.0f} units")
        print(f"   Std deviation: {self.daily_demand['demand'].std():.0f} units")
        
    def train_test_split(self, test_days: int = 90):
        """
        Split data into training and testing sets.
        
        Args:
            test_days: Number of days to hold out for testing (default: 90 days = 3 months)
        """
        print(f"\n[SPLIT] Creating train/test split (last {test_days} days for testing)...")
        
        split_date = self.daily_demand['date'].max() - timedelta(days=test_days)
        
        self.train_data = self.daily_demand[self.daily_demand['date'] <= split_date].copy()
        self.test_data = self.daily_demand[self.daily_demand['date'] > split_date].copy()
        
        print(f"   Training set: {len(self.train_data)} days ({self.train_data['date'].min()} to {self.train_data['date'].max()})")
        print(f"   Test set: {len(self.test_data)} days ({self.test_data['date'].min()} to {self.test_data['date'].max()})")
        
    def baseline_models(self):
        """
        Create baseline forecasting models for comparison.
        """
        print("\n" + "="*70)
        print("  BASELINE FORECASTING MODELS")
        print("="*70)
        
        # Model 1: Naive Forecast (last value)
        print("\n[MODEL 1] Naive Baseline (Last Period)")
        naive_forecast = np.full(len(self.test_data), self.train_data['demand'].iloc[-1])
        self.forecast_results['Naive'] = naive_forecast
        self.evaluate_forecast('Naive', naive_forecast)
        
        # Model 2: Moving Average (7-day)
        print("\n[MODEL 2] Moving Average (7-day)")
        ma7_forecast = np.full(len(self.test_data), self.train_data['demand'].tail(7).mean())
        self.forecast_results['MA_7day'] = ma7_forecast
        self.evaluate_forecast('MA_7day', ma7_forecast)
        
        # Model 3: Moving Average (30-day)
        print("\n[MODEL 3] Moving Average (30-day)")
        ma30_forecast = np.full(len(self.test_data), self.train_data['demand'].tail(30).mean())
        self.forecast_results['MA_30day'] = ma30_forecast
        self.evaluate_forecast('MA_30day', ma30_forecast)
        
    def exponential_smoothing_model(self):
        """
        Build Exponential Smoothing model with trend and seasonality.
        """
        print("\n" + "="*70)
        print("  EXPONENTIAL SMOOTHING MODEL")
        print("="*70)
        
        try:
            print("\n[TRAINING] Holt-Winters Exponential Smoothing...")
            
            # Fit model with additive seasonality (weekly pattern = 7 days)
            model = ExponentialSmoothing(
                self.train_data['demand'],
                seasonal_periods=7,
                trend='add',
                seasonal='add',
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(optimized=True)
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(self.test_data))
            self.forecast_results['Exp_Smoothing'] = forecast.values
            
            print(f"   [OK] Model trained successfully")
            print(f"   Alpha (level): {fitted_model.params['smoothing_level']:.4f}")
            print(f"   Beta (trend): {fitted_model.params['smoothing_trend']:.4f}")
            print(f"   Gamma (seasonal): {fitted_model.params['smoothing_seasonal']:.4f}")
            
            self.evaluate_forecast('Exp_Smoothing', forecast.values)
            
        except Exception as e:
            print(f"   [ERROR] Exponential Smoothing failed: {str(e)}")
            
    def sarima_model(self):
        """
        Build SARIMA (Seasonal ARIMA) model.
        """
        print("\n" + "="*70)
        print("  SARIMA MODEL (Seasonal ARIMA)")
        print("="*70)
        
        try:
            print("\n[TRAINING] SARIMA model...")
            print("   Note: This may take a few minutes...")
            
            # SARIMA parameters: (p,d,q) x (P,D,Q,s)
            # p,d,q: non-seasonal parameters
            # P,D,Q: seasonal parameters
            # s: seasonal period (7 for weekly)
            
            # Using auto-selected parameters for speed
            # In production, would use auto_arima from pmdarima
            order = (1, 1, 1)  # (p, d, q)
            seasonal_order = (1, 1, 1, 7)  # (P, D, Q, s)
            
            model = SARIMAX(
                self.train_data['demand'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False, maxiter=50)
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(self.test_data))
            self.forecast_results['SARIMA'] = forecast.values
            
            print(f"   [OK] SARIMA{order}x{seasonal_order} trained")
            print(f"   AIC: {fitted_model.aic:.2f}")
            print(f"   BIC: {fitted_model.bic:.2f}")
            
            self.evaluate_forecast('SARIMA', forecast.values)
            
        except Exception as e:
            print(f"   [ERROR] SARIMA failed: {str(e)}")
            
    def prophet_model(self):
        """
        Build Facebook Prophet model with seasonality and holidays.
        """
        if not PROPHET_AVAILABLE:
            print("\n[SKIP] Prophet not available")
            return
            
        print("\n" + "="*70)
        print("  FACEBOOK PROPHET MODEL")
        print("="*70)
        
        try:
            print("\n[TRAINING] Prophet model with seasonality...")
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_train = self.train_data[['date', 'demand']].copy()
            prophet_train.columns = ['ds', 'y']
            
            # Initialize Prophet with seasonality
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )
            
            # Fit model
            model.fit(prophet_train)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=len(self.test_data))
            
            # Forecast
            forecast = model.predict(future)
            
            # Extract forecast for test period
            test_forecast = forecast.tail(len(self.test_data))['yhat'].values
            self.forecast_results['Prophet'] = test_forecast
            
            print(f"   [OK] Prophet model trained")
            print(f"   Seasonality components: Weekly, Yearly")
            
            self.evaluate_forecast('Prophet', test_forecast)
            
            # Save Prophet forecast details
            forecast_details = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(self.test_data))
            forecast_details.to_csv(self.output_dir / 'prophet_forecast_details.csv', index=False)
            
        except Exception as e:
            print(f"   [ERROR] Prophet failed: {str(e)}")
            
    def evaluate_forecast(self, model_name: str, forecast: np.ndarray):
        """
        Evaluate forecast accuracy using multiple metrics.
        
        Args:
            model_name: Name of the forecasting model
            forecast: Forecasted values
        """
        actual = self.test_data['demand'].values
        
        # Calculate metrics
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        # Forecast bias
        bias = np.mean(forecast - actual)
        
        # Store results
        self.model_performance.append({
            'Model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'Bias': round(bias, 2),
            'Forecast_Accuracy': round(100 - mape, 2)
        })
        
        print(f"   MAE: {mae:.2f} units")
        print(f"   RMSE: {rmse:.2f} units")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Forecast Accuracy: {100 - mape:.2f}%")
        print(f"   Bias: {bias:.2f} units")
        
    def compare_models(self):
        """
        Compare all models and identify the best performer.
        """
        print("\n" + "="*70)
        print("  MODEL COMPARISON & SELECTION")
        print("="*70)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.model_performance)
        comparison_df = comparison_df.sort_values('MAPE')
        
        print("\n[RESULTS] Model Performance Ranking (by MAPE):")
        print(comparison_df.to_string(index=False))
        
        # Identify best model
        best_model = comparison_df.iloc[0]
        baseline_mape = comparison_df[comparison_df['Model'] == 'Naive']['MAPE'].values[0]
        
        print(f"\n[WINNER] Best Model: {best_model['Model']}")
        print(f"   MAPE: {best_model['MAPE']:.2f}%")
        print(f"   Forecast Accuracy: {best_model['Forecast_Accuracy']:.2f}%")
        print(f"   Improvement over Naive: {baseline_mape - best_model['MAPE']:.2f} percentage points")
        
        # Business impact
        if best_model['Forecast_Accuracy'] >= 75:
            print(f"\n[SUCCESS] Target achieved! Forecast accuracy >75%")
        else:
            print(f"\n[INFO] Target: 75% accuracy | Current: {best_model['Forecast_Accuracy']:.2f}%")
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        print(f"\n[EXPORT] Model comparison saved to: {self.output_dir / 'model_comparison.csv'}")
        
        return best_model['Model']
        
    def save_forecasts(self):
        """
        Save all forecasts to CSV for visualization.
        """
        print("\n[EXPORT] Saving forecast results...")
        
        # Create forecast dataframe
        forecast_df = self.test_data[['date']].copy()
        forecast_df['actual'] = self.test_data['demand'].values
        
        for model_name, forecast in self.forecast_results.items():
            forecast_df[f'forecast_{model_name}'] = forecast
            forecast_df[f'error_{model_name}'] = self.test_data['demand'].values - forecast
        
        forecast_df.to_csv(self.output_dir / 'all_forecasts.csv', index=False)
        print(f"   [OK] Forecasts saved to: {self.output_dir / 'all_forecasts.csv'}")
        
    def generate_forecast_report(self):
        """
        Generate executive summary of forecasting results.
        """
        print("\n" + "="*70)
        print("  FORECASTING SUMMARY REPORT")
        print("="*70)
        
        comparison_df = pd.DataFrame(self.model_performance)
        best_model = comparison_df.loc[comparison_df['MAPE'].idxmin()]
        
        report = {
            'Analysis Date': datetime.now().strftime('%Y-%m-%d'),
            'Forecast Horizon': f"{len(self.test_data)} days",
            'Models Evaluated': len(self.model_performance),
            'Best Model': best_model['Model'],
            'Best Model MAPE': f"{best_model['MAPE']:.2f}%",
            'Best Model Accuracy': f"{best_model['Forecast_Accuracy']:.2f}%",
            'Target Accuracy': '75%',
            'Target Met': 'Yes' if best_model['Forecast_Accuracy'] >= 75 else 'No',
            'Baseline (Naive) MAPE': f"{comparison_df[comparison_df['Model'] == 'Naive']['MAPE'].values[0]:.2f}%",
            'Improvement vs Baseline': f"{comparison_df[comparison_df['Model'] == 'Naive']['MAPE'].values[0] - best_model['MAPE']:.2f} pp"
        }
        
        report_df = pd.DataFrame([report]).T
        report_df.columns = ['Value']
        report_df.to_csv(self.output_dir / 'forecast_summary.csv')
        
        print("\n[SUMMARY] Forecasting Results:")
        for key, value in report.items():
            print(f"   {key}: {value}")
        
        print(f"\n[EXPORT] Summary saved to: {self.output_dir / 'forecast_summary.csv'}")
        
    def run_full_forecasting(self):
        """
        Execute complete forecasting workflow.
        """
        print("\n" + "="*70)
        print("  DEMAND FORECASTING ANALYSIS")
        print("="*70)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_and_prepare_data()
        self.train_test_split(test_days=90)
        
        # Build models
        self.baseline_models()
        self.exponential_smoothing_model()
        self.sarima_model()
        self.prophet_model()
        
        # Compare and save
        best_model = self.compare_models()
        self.save_forecasts()
        self.generate_forecast_report()
        
        print("\n" + "="*70)
        print("  FORECASTING COMPLETE!")
        print("="*70)
        print(f"\n[CHECK] All forecasts exported to: {self.output_dir.absolute()}")
        print(f"\n[RECOMMENDATION] Use {best_model} for production forecasting")
        print("\n[TARGET] Next Steps:")
        print("   1. Review forecast accuracy in data/output/model_comparison.csv")
        print("   2. Use forecasts for inventory optimization")
        print("   3. Create forecast visualizations")
        print("   4. Implement automated retraining pipeline")
        print("\n")


def main():
    """
    Main execution function.
    """
    forecaster = DemandForecaster()
    forecaster.run_full_forecasting()


if __name__ == "__main__":
    main()
