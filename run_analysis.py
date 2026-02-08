"""
Master Execution Script
========================

PURPOSE:
This script orchestrates the complete supply chain analytics workflow.
Executes all analysis modules in the correct sequence.

WORKFLOW:
1. Generate synthetic data
2. Execute SQL queries
3. Perform exploratory data analysis
4. Build demand forecasting models
5. Optimize inventory policies
6. Create visualizations

HOW TO USE:
    python run_analysis.py

OPTIONS:
    --skip-data-gen: Skip data generation (use existing data)
    --skip-sql: Skip SQL query execution
    --quick: Run quick analysis (skip time-consuming models)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse

class AnalysisPipeline:
    """
    Orchestrates the complete supply chain analytics pipeline.
    """
    
    def __init__(self, skip_data_gen=False, skip_sql=False, quick_mode=False):
        self.skip_data_gen = skip_data_gen
        self.skip_sql = skip_sql
        self.quick_mode = quick_mode
        self.project_root = Path(__file__).parent
        
    def run_script(self, script_name: str, description: str):
        """
        Run a Python script and handle errors.
        """
        print("\n" + "="*70)
        print(f"  {description}")
        print("="*70)
        
        script_path = self.project_root / 'src' / script_name
        
        if not script_path.exists():
            print(f"[SKIP] {script_name} not found")
            return False
            
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            print(result.stdout)
            
            if result.returncode != 0:
                print(f"[ERROR] {script_name} failed:")
                print(result.stderr)
                return False
            else:
                print(f"[SUCCESS] {description} completed")
                return True
                
        except Exception as e:
            print(f"[ERROR] Failed to run {script_name}: {str(e)}")
            return False
            
    def run_full_pipeline(self):
        """
        Execute the complete analytics pipeline.
        """
        print("\n" + "="*70)
        print("  SUPPLY CHAIN ANALYTICS - MASTER EXECUTION PIPELINE")
        print("="*70)
        print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project Root: {self.project_root}")
        
        results = {}
        
        # Step 1: Data Generation
        if not self.skip_data_gen:
            results['Data Generation'] = self.run_script(
                'data_generation.py',
                'STEP 1: GENERATING SYNTHETIC SUPPLY CHAIN DATA'
            )
        else:
            print("\n[SKIP] Data generation (using existing data)")
            results['Data Generation'] = True
            
        # Step 2: SQL Query Execution
        if not self.skip_sql:
            results['SQL Analysis'] = self.run_script(
                'execute_sql_queries.py',
                'STEP 2: EXECUTING SQL ANALYTICAL QUERIES'
            )
        else:
            print("\n[SKIP] SQL query execution")
            results['SQL Analysis'] = True
            
        # Step 3: Exploratory Data Analysis
        results['EDA'] = self.run_script(
            'exploratory_analysis.py',
            'STEP 3: EXPLORATORY DATA ANALYSIS'
        )
        
        # Step 4: Demand Forecasting
        if not self.quick_mode:
            results['Forecasting'] = self.run_script(
                'demand_forecasting.py',
                'STEP 4: DEMAND FORECASTING MODELS'
            )
        else:
            print("\n[SKIP] Demand forecasting (quick mode)")
            results['Forecasting'] = True
            
        # Step 5: Inventory Optimization
        results['Optimization'] = self.run_script(
            'inventory_optimization.py',
            'STEP 5: INVENTORY OPTIMIZATION'
        )
        
        # Step 6: Visualizations
        results['Visualization'] = self.run_script(
            'visualization.py',
            'STEP 6: CREATING VISUALIZATIONS AND DASHBOARDS'
        )
        
        # Step 7: Profitability Risk Prediction
        results['Profitability Prediction'] = self.run_script(
            'profitability_prediction.py',
            'STEP 7: PROFITABILITY RISK PREDICTION (ML MODEL)'
        )
        
        # Print summary
        self.print_summary(results)
        
    def print_summary(self, results: dict):
        """
        Print execution summary.
        """
        print("\n" + "="*70)
        print("  PIPELINE EXECUTION SUMMARY")
        print("="*70)
        
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nStep Results:")
        
        for step, success in results.items():
            status = "[OK]" if success else "[FAILED]"
            print(f"   {status} {step}")
            
        total_steps = len(results)
        successful_steps = sum(results.values())
        
        print(f"\nTotal Steps: {total_steps}")
        print(f"Successful: {successful_steps}")
        print(f"Failed: {total_steps - successful_steps}")
        
        if successful_steps == total_steps:
            print("\n[SUCCESS] All analysis steps completed successfully!")
            print("\n[NEXT STEPS]")
            print("   1. Review results in data/processed/ and data/output/")
            print("   2. Open visualizations in reports/figures/")
            print("   3. Review executive_dashboard.html for key insights")
            print("   4. Check optimization recommendations")
        else:
            print("\n[WARNING] Some steps failed. Review error messages above.")
            
        print("\n" + "="*70 + "\n")


def main():
    """
    Main execution function with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Run complete supply chain analytics pipeline'
    )
    
    parser.add_argument(
        '--skip-data-gen',
        action='store_true',
        help='Skip data generation (use existing data)'
    )
    
    parser.add_argument(
        '--skip-sql',
        action='store_true',
        help='Skip SQL query execution'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (skip time-consuming models)'
    )
    
    args = parser.parse_args()
    
    pipeline = AnalysisPipeline(
        skip_data_gen=args.skip_data_gen,
        skip_sql=args.skip_sql,
        quick_mode=args.quick
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
