import streamlit as st
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
    header {visibility: hidden;}
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

# ==========================================
# MAIN CONTENT
# ==========================================
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
        gap: 12px;
        margin-bottom: 20px;
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
        gap: 12px;
        margin-bottom: 24px;
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
        gap: 12px;
        margin-bottom: 24px;
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
    
    # ---- Methodology ----
    st.markdown('<div class="section-heading">🔬 Methodology Pipeline</div>', unsafe_allow_html=True)
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


display_chat()

