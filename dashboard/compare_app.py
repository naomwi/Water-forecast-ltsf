import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from data_loader import get_available_sites, load_metrics, load_series, MODEL_FOLDERS

# ==========================================
# PAGE SETTINGS & CONSTANTS
# ==========================================
st.set_page_config(page_title="LTSF Models Comparison", layout="wide", page_icon="⚖️")

TARGETS = ["EC", "pH"]
HORIZONS = [6, 12, 24, 48, 96, 168]

# ==========================================
# FUNCTIONS
# ==========================================
@st.cache_data
def compile_comparison_metrics(target, site, horizon):
    """Load metrics for Proposed vs CEEMD."""
    # Proposed
    prop_metrics = load_metrics(target, horizon, site, "SpikeDLinear")
    # CEEMD
    ceemd_metrics = load_metrics(target, horizon, site, "CEEMD-DLinear")
    
    data = []
    if prop_metrics:
        prop_metrics['Model'] = 'Proposed (SpikeDLinear)'
        data.append(prop_metrics)
    if ceemd_metrics:
        ceemd_metrics['Model'] = 'CEEMD Baseline (DLinear)'
        data.append(ceemd_metrics)
        
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data
def load_all_metrics(target):
    """Compile an overview of all metrics across all sites and horizons."""
    all_data = []
    sites = get_available_sites()
    
    for site in sites:
        for h in HORIZONS:
            df = compile_comparison_metrics(target, site, h)
            if not df.empty:
                for _, row in df.iterrows():
                    row_dict = row.to_dict()
                    row_dict['Site'] = site
                    all_data.append(row_dict)
                    
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚖️ Model Comparison")

st_target = st.sidebar.selectbox("Target Variable", TARGETS)
sites = get_available_sites()
st_site = st.sidebar.selectbox("Monitoring Site", sites)

# Get valid horizons
all_metrics = load_all_metrics(st_target)
if not all_metrics.empty:
    valid_horizons = sorted(all_metrics[all_metrics['Site'] == st_site]['Horizon'].unique().tolist())
else:
    valid_horizons = HORIZONS

if not valid_horizons:
    valid_horizons = HORIZONS

st_horizon = st.sidebar.selectbox("Prediction Horizon (H)", valid_horizons)

st.sidebar.markdown("---")
st.sidebar.info("This dashboard compares the Proposed SpikeDLinear model against the CEEMD DLinear baseline head-to-head.")

# ==========================================
# MAIN CONTENT
# ==========================================
st.title(f"⚖️ Head-to-Head: Proposed vs CEEMD ({st_target}, Site {st_site})")

tab1, tab2, tab3 = st.tabs(["📊 KPI Comparison", "📈 Series Overlays", "🏆 Global Summary"])

with tab1:
    st.header(f"Performance Metrics (Horizon {st_horizon})")
    
    metrics_df = compile_comparison_metrics(st_target, st_site, st_horizon)
    
    if not metrics_df.empty:
        # Style dataframe for display
        display_df = metrics_df[['Model', 'MSE', 'MAE', 'RMSE', 'R2', 'MAPE']].copy()
        
        def highlight_best(s):
            is_min = s == s.min()
            is_max = s == s.max()
            if s.name == 'R2':
                return ['background-color: #2e7d32; color: white' if v else '' for v in is_max]
            elif s.name in ['MSE', 'MAE', 'RMSE', 'MAPE']:
                return ['background-color: #2e7d32; color: white' if v else '' for v in is_min]
            return ['' for _ in s]

        styled_df = display_df.style.apply(highlight_best, subset=['MSE', 'MAE', 'RMSE', 'R2', 'MAPE']).format(
            {'MSE': '{:.4f}', 'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R2': '{:.4f}', 'MAPE': '{:.2f}%'}
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Calculate improvement manually for KPIs
        st.markdown("### 🚀 Improvement Overview")
        if len(display_df) == 2:
            try:
                prop_mse = display_df[display_df['Model'].str.contains('Proposed')]['MSE'].values[0]
                ceemd_mse = display_df[display_df['Model'].str.contains('CEEMD')]['MSE'].values[0]
                
                prop_r2 = display_df[display_df['Model'].str.contains('Proposed')]['R2'].values[0]
                ceemd_r2 = display_df[display_df['Model'].str.contains('CEEMD')]['R2'].values[0]
                
                mse_diff = ((ceemd_mse - prop_mse) / ceemd_mse) * 100
                r2_diff = prop_r2 - ceemd_r2
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Proposed MSE vs Baseline", value=f"{prop_mse:.4f}", delta=f"{mse_diff:.1f}% Improvement", delta_color="normal")
                with col2:
                    st.metric(label="Proposed R² vs Baseline", value=f"{prop_r2:.4f}", delta=f"{r2_diff:.4f} Absolute", delta_color="normal")
            except IndexError:
                pass
    else:
        st.warning(f"No metrics found. Tests might still be running or haven't run for this configuration.")

with tab2:
    st.header(f"Time Series Overlay: Actual vs Predicted")
    
    prop_series = load_series(st_target, st_horizon, st_site, "SpikeDLinear")
    ceemd_series = load_series(st_target, st_horizon, st_site, "CEEMD-DLinear")
    
    if prop_series is not None or ceemd_series is not None:
        fig = go.Figure()
        
        actual_added = False
        
        if prop_series is not None:
            fig.add_trace(go.Scatter(x=prop_series['Timestep'], y=prop_series['Actual'], mode='lines', name='Actual', line=dict(color='gray', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=prop_series['Timestep'], y=prop_series['Predicted'], mode='lines', name='Proposed (SpikeDLinear)', line=dict(color='#ff4b4b', width=2)))
            actual_added = True
            
        if ceemd_series is not None:
            if not actual_added:
                fig.add_trace(go.Scatter(x=ceemd_series['Timestep'], y=ceemd_series['Actual'], mode='lines', name='Actual', line=dict(color='gray', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=ceemd_series['Timestep'], y=ceemd_series['Predicted'], mode='lines', name='CEEMD Baseline', line=dict(color='#0068c9', width=2)))
            
        fig.update_layout(
            title=f"Horizon {st_horizon} Predictions (Test Set)",
            xaxis_title="Time Steps Series",
            yaxis_title=st_target,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Series data not available yet.")

with tab3:
    st.header(f"Global Summary View for {st_target}")
    
    if not all_metrics.empty:
        st.subheader("MSE Heatmap (Lower is Better)")
        try:
            mse_pivot = all_metrics.pivot_table(index=['Site', 'Model'], columns='Horizon', values='MSE')
            st.dataframe(mse_pivot.style.background_gradient(cmap='YlOrRd', axis=None).format("{:.4f}"), use_container_width=True)
            
            st.subheader("R² Heatmap (Higher is Better)")
            r2_pivot = all_metrics.pivot_table(index=['Site', 'Model'], columns='Horizon', values='R2')
            st.dataframe(r2_pivot.style.background_gradient(cmap='Greens', axis=None).format("{:.4f}"), use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate pivot tables: {e}")
    else:
        st.warning("No data available yet across sites/horizons.")
