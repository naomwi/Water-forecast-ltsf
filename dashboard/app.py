import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from chatbot import init_gemini, display_chat
from pathlib import Path

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
    if st.session_state.current_page == "Chat":
        st.button("📖 About the Project", use_container_width=True, on_click=set_page, args=("About",))
        st.button("📊 Dataset Explorer", use_container_width=True, on_click=set_page, args=("Dataset",))
    elif st.session_state.current_page == "About":
        st.button("💬 Back to Chat", use_container_width=True, on_click=set_page, args=("Chat",))
        st.button("📊 Dataset Explorer", use_container_width=True, on_click=set_page, args=("Dataset",))
    else:
        st.button("💬 Back to Chat", use_container_width=True, on_click=set_page, args=("Chat",))
        st.button("📖 About the Project", use_container_width=True, on_click=set_page, args=("About",))


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

