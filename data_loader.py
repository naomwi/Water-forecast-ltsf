import os
import pandas as pd
import streamlit as st
from pathlib import Path

# Project root directory (same level as dashboard)
PROJECT_DIR = Path(__file__).parent.parent

# Model folders mapping (same as compare_models.py)
MODEL_FOLDERS = {
    'SpikeDLinear': ('Proposed_Models', 'SpikeDLinear'),
    'CEEMD-DLinear': ('CEEMD_Baselines', 'dlinear'),
    'CEEMD-NLinear': ('CEEMD_Baselines', 'nlinear'),
    'LSTM': ('Deep_Baselines', 'lstm'),
    'PatchTST': ('Deep_Baselines', 'patchtst'),
    'Transformer': ('Deep_Baselines', 'transformer'),
}

@st.cache_data
def get_available_sites():
    """Dynamically extract available sites from trained metric files across all models."""
    sites = set()
    
    # Loop over all registered model paths to find trained sites
    for folder, file_prefix in MODEL_FOLDERS.values():
        for target in ['EC', 'pH']:
            results_dir = PROJECT_DIR / folder / target / "results"
            
            # Check for site_XXXX folders directly
            if results_dir.exists():
                for item in os.listdir(results_dir):
                    if item.startswith("site_"):
                        try:
                            site_str = item.split('_')[1]
                            sites.add(int(site_str))
                        except Exception:
                            pass
            
            # Legacy generic folder check fallback
            metrics_dir = results_dir / "metrics"
            if metrics_dir.exists():
                for file in os.listdir(metrics_dir):
                    if file.endswith(".csv") and "_site" in file:
                        try:
                            site_str = file.split('_site')[1].split('_')[0]
                            sites.add(int(site_str))
                        except Exception:
                            pass
    
    if not sites:
        return [1463500]  # Fallback to default if no results found
        
    return sorted(list(sites))

@st.cache_data
def get_global_kpi_summary():
    """Reads all basic KPIs to build a single string representing the entire benchmark results."""
    sites = get_available_sites()
    targets = ['EC', 'pH']
    horizons = [6, 12, 24, 48, 96, 168]
    
    summary_lines = ["--- GLOBAL WATER QUALITY BENCHMARK RESULTS ---"]
    for site in sites:
        summary_lines.append(f"\n[Site: {site}]")
        for target in targets:
            summary_lines.append(f"  Target: {target}")
            for h in horizons:
                horizon_stats = []
                for model_name in MODEL_FOLDERS.keys():
                    metrics = load_metrics(target, h, site, model_name)
                    if metrics:
                        mse = metrics.get('MSE', 'N/A')
                        r2 = metrics.get('R2', 'N/A')
                        if isinstance(mse, float):
                            mse = f"{mse:.4f}"
                        if isinstance(r2, float):
                            r2 = f"{r2:.4f}"
                        horizon_stats.append(f"{model_name}(MSE:{mse}, R2:{r2})")
                
                if horizon_stats:
                    summary_lines.append(f"    Horizon {h}h -> " + " | ".join(horizon_stats))
                    
    return "\n".join(summary_lines)

@st.cache_data
def load_metrics(target, horizon, site, model_name="SpikeDLinear"):
    """
    Load KPIs (MAE, R2, etc.) for a specific Target, Horizon, and Site.
    Returns a dictionary of metrics if successful, else None.
    """
    if model_name not in MODEL_FOLDERS:
        return None
        
    folder, file_prefix = MODEL_FOLDERS[model_name]
    
    # New format: site_XXXX/metrics/prefix_hYY.csv
    file_path_new = PROJECT_DIR / folder / target / "results" / f"site_{site}" / "metrics" / f"{file_prefix}_h{horizon}.csv"
    
    # Old format fallback: metrics/prefix_siteXXXX_hYY.csv
    file_path_old = PROJECT_DIR / folder / target / "results" / "metrics" / f"{file_prefix}_site{site}_h{horizon}.csv"
    
    file_path = file_path_new if file_path_new.exists() else file_path_old
    
    if not file_path.exists():
        return None
        
    try:
        df = pd.read_csv(file_path)
        if not df.empty:
            return df.iloc[0].to_dict() # Return the first row as a Dict
    except Exception as e:
        print(f"Error loading metrics: {e}")
    
    return None

@st.cache_data
def load_series(target, horizon, site, model_name="SpikeDLinear"):
    """
    Load the prediction series vs actual series data.
    Returns a Pandas DataFrame if successful, else None.
    """
    if model_name not in MODEL_FOLDERS:
        return None
        
    folder, file_prefix = MODEL_FOLDERS[model_name]
    
    # New format: site_XXXX/series/series_prefix_P{horizon}_{target}.csv
    file_path_new = PROJECT_DIR / folder / target / "results" / f"site_{site}" / "series" / f"series_{file_prefix}_P{horizon}_{target}.csv"
    
    # Old format fallback
    file_path_old = PROJECT_DIR / folder / target / "results" / "series" / f"series_{file_prefix}_site{site}_P{horizon}_{target}.csv"
    
    file_path = file_path_new if file_path_new.exists() else file_path_old
    
    if not file_path.exists():
        return None
        
    try:
        df = pd.read_csv(file_path)
        # Add dummy Index or Time column if present
        df['Timestep'] = range(len(df))
        return df
    except Exception as e:
        print(f"Error loading series: {e}")
    
    return None

def generate_data_summary(df_series, metrics):
    """
    Generate a statistical summary of the data and errors for Gemini context injection.
    """
    if df_series is None or metrics is None:
        return "No data available."
        
    # Calculate errors
    errors = df_series['Predicted'] - df_series['Actual']
    
    summary = f"""
    [KPI Metrics]
    - MAE: {metrics.get('MAE', 'N/A'):.4f}
    - MSE: {metrics.get('MSE', 'N/A'):.4f}
    - RMSE: {metrics.get('RMSE', 'N/A'):.4f}
    - R2 Score: {metrics.get('R2', 'N/A'):.4f}
    - MAPE: {metrics.get('MAPE', 'N/A')}
    
    [Error Statistics]
    - Mean Error: {errors.mean():.4f}
    - Std Error: {errors.std():.4f}
    - Max Overprediction: {errors.max():.4f} (Predicted much higher than actual)
    - Max Underprediction: {errors.min():.4f} (Predicted much lower than actual)
    
    [Data Trends]
    - Max Actual Value: {df_series['Actual'].max():.4f}
    - Max Predicted Spike: {df_series['Predicted'].max():.4f}
    - Total Data Points Displayed: {len(df_series)}
    """
    return summary
