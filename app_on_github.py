import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from chatbot import init_gemini, display_chat
from pathlib import Path

import json

# ==========================================
# PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="HydroPred AI | FPT University",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PREMIUM CSS — ChatGPT / Claude Inspired
# ==========================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ============ GLOBAL RESET & FONT ============ */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0a0a0a !important;
    }
    
    /* Hide default Streamlit headers */
    #MainMenu {visibility: hidden;}
    header {background: transparent !important;}
    footer {visibility: hidden;}
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 1rem !important;
        background: radial-gradient(circle at 50% 0%, rgba(255,123,0,0.05) 0%, #0a0a0a 60%);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,123,0,0.2); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,123,0,0.5); }
    
    /* ============ SIDEBAR GLASSMORPHISM ============ */
    [data-testid="stSidebar"] {
        background: rgba(20, 20, 20, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255,255,255,0.03) !important;
    }
    [data-testid="stSidebarNav"] { display: none !important; }
    
    /* Sidebar Brand */
    .sb-logo-wrap {
        text-align: center;
        padding: 24px 10px;
        margin-bottom: 12px;
        position: relative;
    }
    .sb-logo-wrap img {
        height: 48px;
        margin-bottom: 16px;
        filter: drop-shadow(0 4px 12px rgba(255,123,0,0.3));
        transition: transform 0.3s ease;
    }
    .sb-logo-wrap:hover img { transform: scale(1.08); }
    .sb-brand {
        font-weight: 800;
        font-size: 1.45rem;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ff7b00, #ffba59);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        text-shadow: 0 2px 10px rgba(255,123,0,0.2);
    }
    .sb-tagline {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.4;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .sb-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        background: rgba(34, 197, 94, 0.1);
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(34,197,94,0.2);
        font-weight: 500;
    }
    .sb-status-dot {
        width: 6px; height: 6px;
        background: #4ade80;
        border-radius: 50%;
        box-shadow: 0 0 8px #4ade80;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0%,100%{opacity:1; transform: scale(1);}
        50%{opacity:.4; transform: scale(0.8);}
    }
    
    /* Sidebar Cards */
    .sb-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .sb-card:hover {
        background: rgba(255,123,0,0.05);
        border-color: rgba(255,123,0,0.2);
        transform: translateY(-2px);
    }
    .sb-card strong { 
        color: #fff !important; 
        font-size: 0.85rem; 
        display: block; 
        margin-bottom: 6px; 
    }
    .sb-card span { 
        color: rgba(255,255,255,0.5) !important; 
        font-size: 0.75rem; 
        line-height: 1.5; 
        display: block; 
    }
    
    /* ============ MAIN AREA ============ */
    
    /* Hero */
    .hero {
        text-align: center;
        padding: 56px 16px 24px;
        animation: fadeUp 0.6s ease-out;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hero-icon {
        font-size: 3.5rem;
        margin-bottom: 12px;
        filter: drop-shadow(0 0 20px rgba(255,123,0,0.3));
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff, #a3a3a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        max-width: 560px;
        margin: 12px auto 0;
        font-size: 0.95rem;
        color: rgba(255,255,255,0.5);
        line-height: 1.6;
    }
    
    /* Suggestion chips */
    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: center;
        padding: 16px 16px 40px;
    }
    .chip {
        padding: 10px 20px;
        border-radius: 24px;
        font-size: 0.85rem;
        font-weight: 500;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        color: #ccc;
        cursor: pointer;
        transition: all .25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .chip:hover {
        background: rgba(255,123,0,0.1);
        border-color: #ff7b00;
        color: #fff;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(255,123,0,0.15);
    }
    
    /* ============ CHAT BUBBLES ============ */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(255,255,255,0.03) !important;
        background: rgba(255,255,255,0.01) !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.2s ease;
    }
    .stChatMessage:hover {
        background: rgba(255,255,255,0.03) !important;
    }
    
    /* User avatar */
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg,#ff7b00,#ffa845) !important;
        box-shadow: 0 0 10px rgba(255,123,0,0.4);
    }
    
    /* ============ CHAT INPUT ============ */
    [data-testid="stChatInput"] {
        border-radius: 20px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        background: rgba(20,20,20,0.8) !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
        transition: all .3s ease !important;
        padding-left: 8px !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #ff7b00 !important;
        box-shadow: 0 0 0 3px rgba(255,123,0,0.15), 0 8px 32px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #fff !important;
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE NAVIGATION
# ==========================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"

def set_page(page_name):
    st.session_state.current_page = page_name

# Init model once
init_gemini()

# ==========================================
# SIDEBAR
# ==========================================
FPT_LOGO = "https://upload.wikimedia.org/wikipedia/commons/1/11/FPT_logo_2010.svg"

with st.sidebar:
    st.markdown(f"""
    <div class="sb-logo-wrap">
        <img src="{FPT_LOGO}" alt="FPT">
        <div class="sb-brand">HydroPred AI</div>
        <div class="sb-tagline">FPT University · Capstone</div>
        <div class="sb-status"><span class="sb-status-dot"></span>Online</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sb-card">
        <strong>🌊 Water Quality Expert</strong>
        <span>Trained on EC & pH forecasting data across multiple USGS monitoring sites.</span>
    </div>
    <div class="sb-card">
        <strong>🧠 6 AI Models</strong>
        <span>SpikeDLinear · CEEMD-DLinear · CEEMD-NLinear · LSTM · PatchTST · Transformer</span>
    </div>
    <div class="sb-card">
        <strong>📄 Full Report Access</strong>
        <span>Methodology, experimental results, and analysis from the team's capstone report.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation Buttons
    nav_pages = ["Chat", "About", "Dataset", "Retrain"]
    nav_icons = {"Chat": "💬 Chat", "About": "📖 About Project", "Dataset": "📊 Dataset", "Retrain": "🔄 Retrain Model"}
    
    for page in nav_pages:
        if st.session_state.current_page == page:
            st.button(f"✨ {page} (Active)", use_container_width=True, disabled=True)
        else:
            st.button(nav_icons[page], use_container_width=True, on_click=set_page, args=(page,))


# ==========================================
# MAIN CONTENT
# ==========================================
# ==========================================
# ABOUT PAGE VIEW
# ==========================================
if st.session_state.current_page == "About":
    st.markdown("""
    <style>
    .about-header {
        text-align: center;
        padding: 40px 0 20px;
        animation: fadeUp 0.6s ease-out;
    }
    .about-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #a3a3a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    .about-header p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.6);
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Section heading */
    .section-heading {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 28px 0 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Model grid */
    .model-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }
    .model-card {
        border: 1px solid rgba(128,128,128,0.12);
        border-radius: 12px;
        padding: 16px;
        transition: border-color 0.2s ease;
    }
    .model-card:hover {
        border-color: #ff7b00;
    }
    .model-card .mc-name {
        font-weight: 700;
        font-size: 0.92rem;
        margin-bottom: 4px;
    }
    .model-card .mc-type {
        font-size: 0.72rem;
        opacity: 0.5;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .model-card .mc-desc {
        font-size: 0.8rem;
        opacity: 0.7;
        line-height: 1.5;
    }
    .mc-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 600;
        background: rgba(255,123,0,0.15);
        color: #ff7b00;
        margin-bottom: 8px;
    }
    .mc-badge.proposed {
        background: rgba(251,146,60,0.15);
        color: #fb923c;
    }
    
    /* Team members */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 16px;
        margin-bottom: 40px;
    }
    .team-card {
        border: 1px solid rgba(128,128,128,0.1);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .team-card .tm-avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        margin: 0 auto 8px;
        background: rgba(128,128,128,0.1);
    }
    .team-card .tm-name {
        font-weight: 600;
        font-size: 0.82rem;
        margin-bottom: 2px;
    }
    .team-card .tm-role {
        font-size: 0.7rem;
        opacity: 0.5;
    }
    
    /* Methodology */
    .method-steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }
    .method-step {
        border: 1px solid rgba(128,128,128,0.12);
        border-radius: 12px;
        padding: 16px;
    }
    .method-step .ms-num {
        font-size: 0.7rem;
        font-weight: 700;
        opacity: 0.4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .method-step .ms-title {
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 6px;
    }
    .method-step .ms-desc {
        font-size: 0.78rem;
        opacity: 0.65;
        line-height: 1.5;
    }
    
    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(128,128,128,0.1);
        margin: 8px 0;
    }
    </style>
    
    <div class="about-header">
        <h1>About the Project</h1>
        <p>A deep dive into the methodology, architectures, and the team behind HydroPred AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- Methodology ----
    st.markdown('<div class="section-heading" style="margin-top:0;">🔬 Methodology Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="method-steps">
        <div class="method-step">
            <div class="ms-num">Phase 1</div>
            <div class="ms-title">CEEMDAN Decomposition</div>
            <div class="ms-desc">Decouple raw EC signal into 12 IMFs — isolating high-frequency spikes from deterministic trends on raw, unscaled data.</div>
        </div>
        <div class="method-step">
            <div class="ms-num">Phase 2</div>
            <div class="ms-title">Multi-Branch Architecture</div>
            <div class="ms-desc">Route high-freq IMFs (1-3) into a Deep MLP spike detector, and low-freq components into a linear trend branch.</div>
        </div>
        <div class="method-step">
            <div class="ms-num">Phase 3</div>
            <div class="ms-title">Asymmetric Optimization</div>
            <div class="ms-desc">Custom Spike-Aware Loss (α=5.0, γ=2.0) penalizes under-predictions 5× harder to prevent dangerous oversmoothing.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ---- Model Comparison ----
    st.markdown('<div class="section-heading">🧠 Models Benchmarked</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-grid">
        <div class="model-card">
            <span class="mc-badge proposed">★ PROPOSED</span>
            <div class="mc-name">SpikeDLinear</div>
            <div class="mc-type">Hybrid Decomposition-Ensemble</div>
            <div class="mc-desc">CEEMDAN + Multi-branch MLP/Linear with Asymmetric Spike-Aware Loss. Captures extreme EC spikes that baselines miss.</div>
        </div>
        <div class="model-card">
            <span class="mc-badge">BASELINE</span>
            <div class="mc-name">CEEMD-DLinear</div>
            <div class="mc-type">Decomposition + Linear</div>
            <div class="mc-desc">CEEMDAN signal decomposition paired with a DLinear mapping layer. Excels at global trend accuracy with lowest MSE.</div>
        </div>
        <div class="model-card">
            <span class="mc-badge">BASELINE</span>
            <div class="mc-name">CEEMD-NLinear</div>
            <div class="mc-type">Decomposition + Normalized Linear</div>
            <div class="mc-desc">Normalized variant of LTSF-Linear with CEEMDAN preprocessing for non-stationary time-series robustness.</div>
        </div>
        <div class="model-card">
            <span class="mc-badge">BASELINE</span>
            <div class="mc-name">LSTM</div>
            <div class="mc-type">Deep Recurrent Network</div>
            <div class="mc-desc">Classic Long Short-Term Memory network. Processes sequential dependencies but struggles with long-horizon forecasting.</div>
        </div>
        <div class="model-card">
            <span class="mc-badge">BASELINE</span>
            <div class="mc-name">PatchTST</div>
            <div class="mc-type">Transformer (Patched)</div>
            <div class="mc-desc">State-of-the-art Transformer using patching and channel independence for efficient local semantic capture.</div>
        </div>
        <div class="model-card">
            <span class="mc-badge">BASELINE</span>
            <div class="mc-name">Transformer</div>
            <div class="mc-type">Vanilla Self-Attention</div>
            <div class="mc-desc">Standard self-attention mechanism for deep global semantic modeling of temporal sequences.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ---- Team Members ----
    st.markdown('<div class="section-heading">👥 Research Team</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="team-grid">
        <div class="team-card">
            <div class="tm-avatar">🧑‍💻</div>
            <div class="tm-name">Khoi Nguyen</div>
            <div class="tm-role">Lead Researcher</div>
        </div>
        <div class="team-card">
            <div class="tm-avatar">👩‍🏫</div>
            <div class="tm-name">Thu Le</div>
            <div class="tm-role">Supervisor</div>
        </div>
        <div class="team-card">
            <div class="tm-avatar">🧑‍💻</div>
            <div class="tm-name">Thai Tran</div>
            <div class="tm-role">Researcher</div>
        </div>
        <div class="team-card">
            <div class="tm-avatar">🧑‍💻</div>
            <div class="tm-name">Khai Trinh</div>
            <div class="tm-role">Researcher</div>
        </div>
        <div class="team-card">
            <div class="tm-avatar">🧑‍💻</div>
            <div class="tm-name">Phuoc Phan</div>
            <div class="tm-role">Researcher</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# DATASET EXPLORER PAGE
# ==========================================
elif st.session_state.current_page == "Dataset":
    # ---- CSS for Dataset Explorer ----
    st.markdown("""
    <style>
    .ds-header {
        text-align: center;
        padding: 40px 0 20px;
        animation: fadeUp 0.6s ease-out;
    }
    .ds-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #a3a3a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    .ds-header p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.6);
        max-width: 700px;
        margin: 0 auto;
    }
    .ds-stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        padding: 20px 0 32px;
    }
    .ds-stat-card {
        border-radius: 14px;
        padding: 22px 16px;
        border: 1px solid rgba(255,123,0,0.15);
        background: rgba(255,123,0,0.04);
        text-align: center;
        transition: all 0.3s ease;
    }
    .ds-stat-card:hover {
        border-color: rgba(255,123,0,0.4);
        background: rgba(255,123,0,0.08);
        transform: translateY(-3px);
    }
    .ds-stat-card .ds-num {
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        background: linear-gradient(135deg, #ff7b00, #ffba59);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .ds-stat-card .ds-label {
        font-size: 0.78rem;
        opacity: 0.55;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .ds-section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 36px 0 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Header ----
    st.markdown("""
    <div class="ds-header">
        <h1>📊 Dataset Explorer</h1>
        <p>Explore the raw USGS water quality monitoring data used to train and evaluate all HydroPred AI models.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Load Data ----
    DATA_PATH = Path(__file__).parent.parent / "Deep_Baselines" / "data" / "USGs" / "water_data_2021_2025_clean.csv"

    @st.cache_data
    def load_dataset():
        df = pd.read_csv(DATA_PATH)
        df['Time'] = pd.to_datetime(df['Time'], utc=True)
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')
        return df

    df_full = load_dataset()
    feature_cols = ['Temp', 'Flow', 'EC', 'DO', 'pH', 'Turbidity']
    feature_units = {
        'Temp': '°C', 'Flow': 'ft³/s', 'EC': 'µS/cm',
        'DO': 'mg/L', 'pH': 'pH', 'Turbidity': 'NTU'
    }
    feature_descriptions = {
        'Temp': 'Water Temperature',
        'Flow': 'Stream Discharge Rate',
        'EC': 'Electrical Conductivity',
        'DO': 'Dissolved Oxygen',
        'pH': 'Acidity/Alkalinity Level',
        'Turbidity': 'Water Clarity'
    }
    sites = sorted(df_full['site_no'].unique())

    # ---- Overview Stats ----
    st.markdown(f"""
    <div class="ds-stats-row">
        <div class="ds-stat-card">
            <div class="ds-num">{len(df_full):,}</div>
            <div class="ds-label">Total Observations</div>
        </div>
        <div class="ds-stat-card">
            <div class="ds-num">{len(sites)}</div>
            <div class="ds-label">Monitoring Sites</div>
        </div>
        <div class="ds-stat-card">
            <div class="ds-num">{len(feature_cols)}</div>
            <div class="ds-label">Water Quality Features</div>
        </div>
        <div class="ds-stat-card">
            <div class="ds-num">2021–2025</div>
            <div class="ds-label">Time Span</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Site Selector ----
    st.markdown('<div class="ds-section-title">🏗️ Select Monitoring Site</div>', unsafe_allow_html=True)
    site_labels = {s: f"USGS Site {s}" for s in sites}
    selected_site = st.selectbox(
        "Choose a USGS monitoring site to explore",
        options=sites,
        format_func=lambda x: site_labels[x],
        label_visibility="collapsed"
    )
    df_site = df_full[df_full['site_no'] == selected_site].copy().reset_index(drop=True)
    st.caption(f"Showing **{len(df_site):,}** hourly observations for USGS Site **{selected_site}** — from **{df_site['Time'].min().strftime('%b %d, %Y')}** to **{df_site['Time'].max().strftime('%b %d, %Y')}**")

    # ---- Plotly dark theme template ----
    plotly_layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='rgba(255,255,255,0.7)', size=12),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        hoverlabel=dict(bgcolor='#1a1a1a', font_size=12, font_family='Inter'),
    )
    color_palette = ['#ff7b00', '#ffba59', '#ff4d4d', '#4dc9f6', '#4ade80', '#a78bfa']

    # ---- Raw Data Preview ----
    st.markdown('<div class="ds-section-title">📋 Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(
        df_site.head(100).style.format({
            'Temp': '{:.1f}', 'Flow': '{:,.1f}', 'EC': '{:.0f}',
            'DO': '{:.1f}', 'pH': '{:.1f}', 'Turbidity': '{:.1f}'
        }),
        use_container_width=True,
        height=350
    )

    # ---- Time Series Plots ----
    st.markdown('<div class="ds-section-title">📈 Time Series — All Parameters</div>', unsafe_allow_html=True)

    # Downsample for performance (every 6 hours)
    df_plot = df_site.set_index('Time').resample('6h').mean(numeric_only=True).reset_index()

    ts_tabs = st.tabs([f"{feature_descriptions[f]} ({f})" for f in feature_cols])
    for i, feat in enumerate(feature_cols):
        with ts_tabs[i]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot['Time'], y=df_plot[feat],
                mode='lines',
                name=feat,
                line=dict(color=color_palette[i], width=1.5),
                fill='tozeroy',
                fillcolor=f'rgba({int(color_palette[i][1:3],16)},{int(color_palette[i][3:5],16)},{int(color_palette[i][5:7],16)},0.08)'
            ))
            fig.update_layout(
                **plotly_layout,
                title=dict(text=f'{feature_descriptions[feat]} over Time', font=dict(size=14)),
                xaxis_title='Date',
                yaxis_title=f'{feat} ({feature_units[feat]})',
                height=380,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Distribution Charts ----
    st.markdown('<div class="ds-section-title">📊 Feature Distributions</div>', unsafe_allow_html=True)

    dist_cols = st.columns(3)
    for i, feat in enumerate(feature_cols):
        with dist_cols[i % 3]:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_site[feat],
                nbinsx=60,
                marker=dict(
                    color=color_palette[i],
                    line=dict(color='rgba(255,255,255,0.1)', width=0.5),
                    opacity=0.85
                ),
                name=feat
            ))
            fig.update_layout(
                **plotly_layout,
                title=dict(text=f'{feat} ({feature_units[feat]})', font=dict(size=13)),
                xaxis_title=f'{feat}',
                yaxis_title='Count',
                height=280,
                showlegend=False,
                bargap=0.03
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Correlation Heatmap ----
    st.markdown('<div class="ds-section-title">🔗 Feature Correlation Heatmap</div>', unsafe_allow_html=True)

    corr_matrix = df_site[feature_cols].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=feature_cols,
        y=feature_cols,
        colorscale=[
            [0.0, '#1a0a00'],
            [0.25, '#3d1a00'],
            [0.5, '#0a0a0a'],
            [0.75, '#7a3d00'],
            [1.0, '#ff7b00']
        ],
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=13, color='white'),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
        zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text='Corr', font=dict(color='rgba(255,255,255,0.6)')),
            tickfont=dict(color='rgba(255,255,255,0.5)')
        )
    ))
    corr_layout = {**plotly_layout}
    corr_layout['xaxis'] = dict(side='bottom', tickfont=dict(size=12), gridcolor='rgba(255,255,255,0.04)')
    corr_layout['yaxis'] = dict(autorange='reversed', tickfont=dict(size=12), gridcolor='rgba(255,255,255,0.04)')
    fig_corr.update_layout(**corr_layout, height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ---- Statistical Summary Table ----
    st.markdown('<div class="ds-section-title">📑 Statistical Summary</div>', unsafe_allow_html=True)

    desc = df_site[feature_cols].describe().T
    desc.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    desc['Unit'] = [feature_units[f] for f in desc.index]
    desc['Description'] = [feature_descriptions[f] for f in desc.index]
    desc = desc[['Description', 'Unit', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']]
    st.dataframe(
        desc.style.format({
            'Count': '{:.0f}', 'Mean': '{:.2f}', 'Std': '{:.2f}',
            'Min': '{:.2f}', '25%': '{:.2f}', '50%': '{:.2f}',
            '75%': '{:.2f}', 'Max': '{:.2f}'
        }),
        use_container_width=True,
        height=280
    )

    # ---- Label Distribution ----
    st.markdown('<div class="ds-section-title">🏷️ Event Label Distribution</div>', unsafe_allow_html=True)
    label_counts = df_site['Final_Label'].value_counts().sort_index()
    fig_label = go.Figure(data=[go.Bar(
        x=['Normal (0)', 'Event (1)'],
        y=[label_counts.get(0, 0), label_counts.get(1, 0)],
        marker=dict(
            color=['rgba(255,255,255,0.1)', '#ff7b00'],
            line=dict(color=['rgba(255,255,255,0.2)', '#ffba59'], width=1.5)
        ),
        text=[f"{label_counts.get(0, 0):,}", f"{label_counts.get(1, 0):,}"],
        textposition='outside',
        textfont=dict(color='rgba(255,255,255,0.7)', size=13)
    )])
    fig_label.update_layout(
        **plotly_layout,
        height=320,
        yaxis_title='Number of Observations',
        showlegend=False
    )
    st.plotly_chart(fig_label, use_container_width=True)


elif st.session_state.current_page == "Chat":
    if "chat_history" not in st.session_state or len(st.session_state.get("chat_history", [])) <= 1:
        
        # ---- CSS for new sections ----
        st.markdown("""
        <style>
        /* Stat cards row */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            padding: 0 0 24px;
        }
        .stat-card {
            border-radius: 14px;
            padding: 20px;
            border: 1px solid rgba(128,128,128,0.12);
            text-align: center;
        }
        .stat-card .stat-num {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 2px;
        }
        .stat-card .stat-label {
            font-size: 0.78rem;
            opacity: 0.55;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # ---- Hero ----
        st.markdown("""
        <div class="hero">
            <div class="hero-icon">🌊</div>
            <h1>HydroPred AI</h1>
            <p>Intelligent assistant for the FPT University Water Quality Forecasting Capstone Project — powered by Gemini AI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ---- Quick Stats ----
        st.markdown("""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-num">43,813</div>
                <div class="stat-label">Observations</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">6</div>
                <div class="stat-label">AI Models</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">6</div>
                <div class="stat-label">Horizons</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">2021–2025</div>
                <div class="stat-label">Time Span</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    
    display_chat()


# ==========================================
# RETRAIN MODEL PAGE
# ==========================================
elif st.session_state.current_page == "Retrain":
    import time
    from google.cloud import aiplatform, storage
    import os
    import json
    import tempfile
    
    def _get_gcp_credentials_path():
        """Resolve GCP credentials: Streamlit Secrets first, then local file."""
        # Option 1: Streamlit Secrets (for Streamlit Cloud)
        try:
            gcp_info = st.secrets["gcp_service_account"]
            # Write to a temp file so google-cloud SDK can read it
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(dict(gcp_info), tmp)
            tmp.close()
            return tmp.name
        except (KeyError, FileNotFoundError):
            pass
        
        # Option 2: Local file (for development)
        local_path = Path(__file__).parent / 'gcp-service-account.json'
        if local_path.exists():
            return str(local_path)
        
        return None
    
    st.markdown("""
    <div class="hero" style="padding: 20px 16px 10px;">
        <div class="hero-icon" style="font-size: 2.5rem; filter: hue-rotate(90deg);">🔄</div>
        <h1>Retrain Model (Vertex AI)</h1>
        <p>Upload new data to fine-tune the SpikeDLinear model directly on Google Cloud MLOps pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Training Configuration")
        seq_len = st.number_input("Sequence Length (History)", min_value=168, value=168, step=24)
        pred_len = st.number_input("Prediction Length (Horizon)", min_value=24, value=24, step=24)
        batch_size = st.number_input("Batch Size", min_value=32, value=64, step=32)
        epochs = st.number_input("Fine-tune Epochs", min_value=10, value=50, step=10)
        
        target_col = st.selectbox("Target Feature", options=['EC', 'pH', 'DO', 'Temp', 'Flow', 'Turbidity'])
        site_no_str = st.text_input("USGS Site No (Optional, ignored if no such column in data)", value="")
        site_no = int(site_no_str) if site_no_str.strip() else 0
    
    with col2:
        st.markdown("### 🗃️ Upload New Dataset")
        uploaded_file = st.file_uploader("Upload CSV file (Must contain target and features)", type=['csv'])
        
        if uploaded_file is not None:
            # 1. READ FILE
            df = pd.read_csv(uploaded_file)
            st.success(f"File loaded: {len(df):,} rows")
            
            # Simple handling for site_no: if column exists and user selected a site, filter it.
            # But the user said user data won't have site_no, so this just lets them train all.
            if site_no != 0 and 'site_no' in df.columns:
                df = df[df['site_no'] == site_no]
                st.info(f"📍 Filtered to Site **{site_no}**: {len(df):,} rows")
            else:
                st.info(f"ℹ️ Training on all {len(df):,} rows.")
            
            # 2. RUN CONSTRAINTS
            validation_passed = True
            
            with st.expander("🚦 Data Validation & Constraints", expanded=True):
                # Constraint 1: Length
                min_len = seq_len + pred_len + batch_size
                if len(df) < min_len:
                    st.error(f"❌ **Constraint 1 Failed:** Data too short. Minimum required: seq_len({seq_len}) + pred_len({pred_len}) + batch_size({batch_size}) = {min_len} rows. Found: {len(df)}.")
                    validation_passed = False
                else:
                    st.success(f"✅ Length acceptable: {len(df)} >= {min_len}")
                
                # Constraint 2: Schema
                required_cols = ['Time', 'Temp', 'Flow', 'EC', 'DO', 'pH', 'Turbidity']
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    st.error(f"❌ **Constraint 2 Failed:** Missing required columns: {missing_cols}")
                    validation_passed = False
                else:
                    st.success("✅ Schema matched (All 7 required columns present)")
                
                # Constraint 3: Quality
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    st.warning(f"⚠️ **Constraint 3 Warning:** Found {nan_count} missing values. Auto-interpolating...")
                    df = df.interpolate(method='linear').bfill().ffill()
                    st.success("✅ Missing values handled.")
                else:
                    st.success("✅ Data quality optimal (0 missing values).")
            
            st.markdown("---")
            
            # Action Button with PASSCODE LOCK
            if validation_passed:
                st.info("💡 Data is ready. Please enter the Admin Passcode to authorize Cloud Vertex AI resources.")
                
                # Use a form to prevent stream-reruns on every keystroke
                with st.form("admin_auth_form"):
                    passcode = st.text_input("Admin Passcode", type="password", help="Contact administrator for the deployment passcode.")
                    bucket_name = st.text_input("GCS Bucket Name", value="hydropred-bucket-2026")
                    gcp_project = st.text_input("GCP Project ID", value="hydropred")
                    auth_submitted = st.form_submit_button("Authenticate")
                
                if auth_submitted:
                    if passcode == "HydroPred2026":
                        st.session_state['is_admin_auth'] = True
                        st.session_state['gcs_bucket'] = bucket_name
                        st.session_state['gcp_project'] = gcp_project
                        st.success("✅ Authentication successful. Vertex AI deployment authorized.")
                    else:
                        st.session_state['is_admin_auth'] = False
                        st.error("❌ Incorrect passcode. Deployment locked.")
                
                # If authenticated, show the trigger button
                if st.session_state.get('is_admin_auth', False):
                    # Check for service account credentials
                    sa_key_path = _get_gcp_credentials_path()
                    if sa_key_path is None:
                        st.error("⚠️ **Missing GCP Credentials.** Either add `gcp_service_account` to Streamlit Secrets or place `gcp-service-account.json` in the dashboard folder.")
                        st.stop()
                    
                    trigger_btn = st.button("🚀 Confirm & Deploy Training Job to Vertex AI", type="primary", use_container_width=True)
                    
                    if trigger_btn:
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_key_path
                        bucket = st.session_state['gcs_bucket']
                        project = st.session_state['gcp_project']
                        region = "asia-southeast1"
                        
                        try:
                            with st.spinner("Initializing Cloud MLOps Pipeline..."):
                                # Step 1: Upload to GCS
                                progress_text = "1. Uploading modified data to GCS temp_train/..."
                                my_bar = st.progress(0, text=progress_text)
                                
                                storage_client = storage.Client()
                                bucket_obj = storage_client.bucket(bucket)
                                blob = bucket_obj.blob("temp_train/new_data.csv")
                                
                                # Convert df to CSV string and upload
                                csv_data = df.to_csv(index=False)
                                blob.upload_from_string(csv_data, content_type='text/csv')
                                gcs_uri = f"gs://{bucket}/temp_train/new_data.csv"
                                
                                my_bar.progress(100, text=f"1. Upload complete: {gcs_uri}")
                                time.sleep(1)
                                
                                # Step 2: Trigger Vertex AI
                                st.info("2. Triggering Vertex AI Custom Container Job...")
                                
                                aiplatform.init(project=project, location=region, staging_bucket=f"gs://{bucket}")
                                
                                # Define the container URI (assumes user pushed it to this path)
                                container_uri = f"{region}-docker.pkg.dev/{project}/hydropred-repo/hydropred-trainer:latest"
                                
                                job = aiplatform.CustomContainerTrainingJob(
                                    display_name="hydropred-retrain-job",
                                    container_uri=container_uri,
                                    command=["python", "/app/train_vertex.py"],
                                )
                                
                                st.success("🚀 Submitting job to Google Cloud...")
                                
                                # Run asynchronously so Streamlit doesn't freeze for 30 minutes
                                # The user can check the GCP console
                                model = job.submit(
                                    machine_type="c2-standard-16",
                                    replica_count=1,
                                    args=[
                                        "--gcs_bucket", bucket,
                                        "--data_uri", gcs_uri,
                                        "--target", target_col,
                                        "--site", str(site_no),
                                        "--seq_len", str(seq_len),
                                        "--pred_len", str(pred_len),
                                        "--epochs", str(epochs),
                                        "--batch_size", str(batch_size),
                                        "--timeout_minutes", "90"
                                    ]
                                )
                                
                                st.balloons()
                                st.success(f"✨ **Vertex AI Job successfully submitted!**")
                                
                                # Store job info for status checking
                                if hasattr(model, 'resource_name'):
                                    st.session_state['vertex_job_name'] = model.resource_name
                                
                                st.info("""
                                ⏳ **Training is now running on Google Cloud. This will take approximately 5-15 minutes.**
                                
                                What's happening right now:
                                1. 🖥️ Google is spinning up a 16-core CPU machine for you
                                2. 📊 CEEMDAN is decomposing your data into frequency components
                                3. 🧠 SpikeDLinear is being fine-tuned on your new data
                                4. 📈 The new model will be evaluated against the previous version
                                
                                **You can safely close this page.** Results will be saved to GCS.
                                Come back anytime and click **"🔍 Check Latest Training Results"** below to see if training is done.
                                """)
                                
                                st.markdown(f"""
                                > **Region:** `{region}` | **Machine:** `n1-standard-16`
                                > 🔗 [Monitor Live Logs on Google Cloud Console](https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project})
                                """)
                        
                        except Exception as e:
                            st.error(f"❌ **Cloud Deployment Failed:** {e}")
                            st.caption("Please verify your GCP Project ID, Bucket Name, and Service Account permissions.")

    # =========================================================================
    # Section: Check Training Results (always visible if authenticated)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📊 Training Results")
    st.caption("After the Vertex AI job finishes (5-15 min), click below to fetch and display the results.")
    
    check_btn = st.button("🔍 Check Latest Training Results", use_container_width=True)
    
    if check_btn:
        sa_key_path = _get_gcp_credentials_path()
        if sa_key_path is None:
            st.error("⚠️ **Missing GCP Credentials.** Either add `gcp_service_account` to Streamlit Secrets or place `gcp-service-account.json` in the dashboard folder.")
            st.stop()
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_key_path
        
        try:
            from google.cloud import storage as gcs_storage
            
            bucket_name = st.session_state.get('gcs_bucket', 'hydropred-bucket-2026')
            client = gcs_storage.Client()
            bucket_obj = client.bucket(bucket_name)
            blob = bucket_obj.blob("results/metrics_latest.json")
            
            if not blob.exists():
                st.warning("⏳ No results found yet. The training job may still be running. Try again in a few minutes.")
            else:
                metrics_data = json.loads(blob.download_as_text())
                
                # --- Verdict Banner ---
                is_better = metrics_data.get("model_is_better", False)
                new_ver = metrics_data.get("version", "?")
                old_ver = metrics_data.get("previous_version", "?")
                
                if is_better:
                    st.success(f"🏆 **Model {new_ver} is BETTER than {old_ver}!** Auto-deploy recommended.")
                else:
                    st.warning(f"⚠️ **Model {new_ver} is WORSE than {old_ver}.** Consider keeping the current model.")
                
                # --- Metrics Table ---
                col_new, col_old = st.columns(2)
                new_m = metrics_data.get("new_model_metrics", {})
                old_m = metrics_data.get("old_model_metrics", {})
                
                with col_new:
                    st.markdown(f"**🆕 New Model ({new_ver})**")
                    for k, v in new_m.items():
                        st.metric(label=k, value=f"{v:.6f}")
                
                with col_old:
                    if old_m:
                        st.markdown(f"**📦 Previous Model ({old_ver})**")
                        for k, v in old_m.items():
                            delta = new_m.get(k, 0) - v
                            delta_str = f"{delta:+.6f}"
                            # For MAE/MSE/RMSE lower is better; for R2 higher is better
                            inv = k == "R2"
                            st.metric(label=k, value=f"{v:.6f}", delta=delta_str, delta_color="inverse" if not inv else "normal")
                    else:
                        st.info("No previous model to compare (first training run).")
                
                # --- Loss Curves ---
                train_loss = metrics_data.get("train_loss_curve", [])
                val_loss = metrics_data.get("val_loss_curve", [])
                
                if train_loss and val_loss:
                    st.markdown("#### 📈 Training & Validation Loss Curves")
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=train_loss, name="Train Loss", mode="lines", line=dict(color="#FF6B6B")))
                    fig.add_trace(go.Scatter(y=val_loss, name="Val Loss", mode="lines", line=dict(color="#4ECDC4")))
                    fig.update_layout(
                        xaxis_title="Epoch", yaxis_title="Loss",
                        template="plotly_dark", height=350,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- Prediction Plot (Step-1 Forecast) ---
                pred_blob = bucket_obj.blob("results/predictions_latest.csv")
                if pred_blob.exists():
                    st.markdown("#### 🎯 Predicted vs Actual (Step-1 Forecast on Hold-out Test Set)")
                    import io
                    pred_df = pd.read_csv(io.BytesIO(pred_blob.download_as_bytes()))
                    
                    import plotly.graph_objects as go
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        y=pred_df["actual"], name="Actual",
                        mode="lines", line=dict(color="#5B8FF9", width=1.5),
                        opacity=0.7
                    ))
                    fig2.add_trace(go.Scatter(
                        y=pred_df["predicted"], name="Predicted",
                        mode="lines", line=dict(color="#FF6B6B", width=1.5, dash="dot"),
                        opacity=0.9
                    ))
                    target_name = metrics_data.get("target", "Value")
                    fig2.update_layout(
                        xaxis_title="Sample Index",
                        yaxis_title=target_name,
                        template="plotly_dark", height=400,
                        margin=dict(l=20, r=20, t=30, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption(f"Total samples: {len(pred_df)} | Target: {target_name}")
                
                # --- Meta Info ---
                with st.expander("🗂️ Full Training Metadata"):
                    st.json(metrics_data)
        
        except Exception as e:
            st.error(f"❌ Failed to fetch results: {e}")
