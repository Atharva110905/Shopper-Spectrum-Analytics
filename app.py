"""
Shopper Spectrum — Customer Segmentation & Product Recommendation
Ultra-Premium Streamlit Application v2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, json
from pathlib import Path

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

BASE = Path(__file__).parent
_data_sub   = BASE / "data"
_models_sub = BASE / "models"
DATA   = _data_sub   if _data_sub.exists()   else BASE
MODELS = _models_sub if _models_sub.exists() else BASE

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS — Aurora dark theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg:        #07080d;
  --bg1:       #0d0f1a;
  --bg2:       #111422;
  --bg3:       #161929;
  --border:    rgba(255,255,255,0.06);
  --border2:   rgba(255,255,255,0.1);
  --p1: #7c6fef;  --p2: #a78bfa;  --p3: #c4b5fd;
  --b1: #22d3ee;  --b2: #38bdf8;
  --g1: #34d399;  --g2: #22c55e;
  --o1: #fb923c;  --o2: #f97316;
  --r1: #f87171;  --r2: #ef4444;
  --y1: #fbbf24;
  --txt:  #f0f2ff;
  --txt2: #8892b0;
  --txt3: #4a5280;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: var(--bg) !important; }

/* Aurora background glow */
.stApp::before {
  content: '';
  position: fixed; top: -40%; left: -20%; width: 80%; height: 80%;
  background: radial-gradient(ellipse, rgba(124,111,239,0.07) 0%, transparent 70%);
  pointer-events: none; z-index: 0;
}
.stApp::after {
  content: '';
  position: fixed; bottom: -30%; right: -10%; width: 60%; height: 60%;
  background: radial-gradient(ellipse, rgba(34,211,238,0.05) 0%, transparent 70%);
  pointer-events: none; z-index: 0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1700px !important; position: relative; z-index: 1; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #09091a 0%, #0d0f1a 50%, #0a0b12 100%) !important;
  border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── NAV ITEMS ── */
div[data-testid="stRadio"] > div { gap: 0.2rem !important; }
div[data-testid="stRadio"] label {
  border-radius: 10px !important; padding: 0.55rem 0.9rem !important;
  color: var(--txt2) !important; font-size: 0.83rem !important;
  font-weight: 500 !important; cursor: pointer;
  transition: all 0.15s ease !important;
  border: 1px solid transparent !important;
}
div[data-testid="stRadio"] label:hover {
  background: rgba(124,111,239,0.1) !important;
  color: var(--txt) !important;
  border-color: rgba(124,111,239,0.2) !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] input:checked + div {
  background: rgba(124,111,239,0.15) !important;
  color: var(--p2) !important;
  border-color: rgba(124,111,239,0.3) !important;
}

/* ── HERO BANNER ── */
.hero {
  background: linear-gradient(135deg, #0d0f1a 0%, #111628 50%, #0a0d1c 100%);
  border: 1px solid var(--border2);
  border-radius: 20px;
  padding: 2rem 2.5rem;
  margin: 1.5rem 0 2rem 0;
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -60%; right: -10%; width: 55%; height: 200%;
  background: radial-gradient(ellipse, rgba(124,111,239,0.12) 0%, transparent 60%);
  pointer-events: none;
}
.hero::after {
  content: '';
  position: absolute; bottom: -50%; left: 30%; width: 40%; height: 150%;
  background: radial-gradient(ellipse, rgba(34,211,238,0.06) 0%, transparent 60%);
  pointer-events: none;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2rem; font-weight: 700; letter-spacing: -0.04em;
  background: linear-gradient(135deg, #fff 30%, var(--p2) 70%, var(--b1) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin-bottom: 0.4rem;
}
.hero-sub { font-size: 0.88rem; color: var(--txt2); line-height: 1.6; max-width: 500px; }
.hero-pills { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
.hero-pill {
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 4px 12px; border-radius: 20px; border: 1px solid;
}
.hero-pill.purple { color: var(--p2); border-color: rgba(167,139,250,0.3); background: rgba(167,139,250,0.08); }
.hero-pill.cyan   { color: var(--b1); border-color: rgba(34,211,238,0.3);  background: rgba(34,211,238,0.08); }
.hero-pill.green  { color: var(--g1); border-color: rgba(52,211,153,0.3);  background: rgba(52,211,153,0.08); }

/* ── KPI GRID ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 1.4rem 1.6rem;
  position: relative; overflow: hidden;
  transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
  cursor: default;
}
.kpi:hover {
  transform: translateY(-3px);
  border-color: var(--border2);
  box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05);
}
.kpi-glow {
  position: absolute; top: -30px; right: -30px;
  width: 100px; height: 100px; border-radius: 50%; opacity: 0.12; filter: blur(30px);
}
.kpi-glow.p { background: var(--p1); }
.kpi-glow.b { background: var(--b1); }
.kpi-glow.g { background: var(--g1); }
.kpi-glow.o { background: var(--o1); }
.kpi-icon { font-size: 1.3rem; margin-bottom: 0.8rem; display: block; }
.kpi-lbl { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--txt3); margin-bottom: 0.4rem; }
.kpi-val { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; letter-spacing: -0.04em; color: var(--txt); line-height: 1; }
.kpi-foot { margin-top: 0.6rem; font-size: 0.72rem; color: var(--txt3); }
.kpi-tag { display: inline-block; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; }
.kpi-tag.up   { background: rgba(52,211,153,0.12); color: var(--g1); }
.kpi-tag.info { background: rgba(124,111,239,0.12); color: var(--p2); }
.kpi-bar { height: 3px; border-radius: 2px; margin-top: 1rem; }
.kpi-bar.p { background: linear-gradient(90deg,var(--p1),var(--p2)); }
.kpi-bar.b { background: linear-gradient(90deg,var(--b2),var(--b1)); }
.kpi-bar.g { background: linear-gradient(90deg,var(--g2),var(--g1)); }
.kpi-bar.o { background: linear-gradient(90deg,var(--o2),var(--o1)); }

/* ── SECTION LABEL ── */
.sec-lbl {
  display: flex; align-items: center; gap: 0.7rem;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--txt3); margin: 2rem 0 1rem 0;
}
.sec-lbl::after { content:''; flex:1; height:1px; background: var(--border); }

/* ── GLASS CARD ── */
.g-card {
  background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
  border: 1px solid var(--border);
  backdrop-filter: blur(10px);
  border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
  position: relative; overflow: hidden;
}
.g-card-title {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--txt3); margin-bottom: 1rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.g-card-title span { color: var(--p2); }

/* ── PAGE HEADER ── */
.page-hdr { padding: 1.8rem 0 0.5rem 0; }
.page-hdr h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; color: var(--txt);
  margin: 0 0 0.3rem 0;
}
.page-hdr p { font-size: 0.85rem; color: var(--txt2); margin: 0; }

/* ── SEGMENT CARDS ── */
.seg-card {
  border-radius: 16px; padding: 1.4rem;
  border: 1px solid; position: relative; overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}
.seg-card:hover { transform: translateY(-2px); }
.seg-card.hv  { background: rgba(124,111,239,0.06); border-color: rgba(124,111,239,0.2); }
.seg-card.reg { background: rgba(34,211,238,0.05);  border-color: rgba(34,211,238,0.2); }
.seg-card.occ { background: rgba(251,146,60,0.05);  border-color: rgba(251,146,60,0.2); }
.seg-card.ar  { background: rgba(248,113,113,0.05); border-color: rgba(248,113,113,0.2); }
.seg-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.seg-name { font-family:'Space Grotesk',sans-serif; font-size: 1rem; font-weight: 700; margin-bottom: 0.2rem; }
.seg-count { font-size: 2rem; font-weight: 800; font-family:'Space Grotesk',sans-serif; letter-spacing:-0.04em; margin-bottom: 0.4rem; }
.seg-desc { font-size: 0.72rem; color: var(--txt2); line-height: 1.5; margin-bottom: 0.8rem; }
.seg-stat { font-size: 0.7rem; color: var(--txt3); }
.seg-stat b { color: var(--txt2); }

/* ── REC CARDS ── */
.rec-item {
  display: flex; align-items: center; gap: 1rem;
  padding: 0.8rem 1rem; border-radius: 12px;
  background: var(--bg2); border: 1px solid var(--border);
  margin-bottom: 0.5rem; transition: all 0.15s;
}
.rec-item:hover { border-color: var(--border2); background: var(--bg3); transform: translateX(4px); }
.rec-rank {
  width: 30px; height: 30px; border-radius: 8px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 800;
  background: linear-gradient(135deg, var(--p1), var(--p2));
  color: white;
}
.rec-name { font-size: 0.85rem; font-weight: 500; color: var(--txt); flex: 1; }
.rec-pct { font-size: 0.72rem; font-weight: 700; color: var(--p2); }

/* ── PREDICTION BOX ── */
.pred-result {
  border-radius: 20px; padding: 2rem; text-align: center;
  border: 1px solid; position: relative; overflow: hidden;
}
.pred-result.hv  { background: rgba(124,111,239,0.08); border-color: rgba(124,111,239,0.25); }
.pred-result.reg { background: rgba(34,211,238,0.06);  border-color: rgba(34,211,238,0.25); }
.pred-result.occ { background: rgba(251,146,60,0.06);  border-color: rgba(251,146,60,0.25); }
.pred-result.ar  { background: rgba(248,113,113,0.06); border-color: rgba(248,113,113,0.25); }
.pred-lbl { font-size: 0.7rem; text-transform:uppercase; letter-spacing:0.12em; color:var(--txt3); margin-bottom:0.5rem; }
.pred-seg { font-family:'Space Grotesk',sans-serif; font-size:2.5rem; font-weight:800; letter-spacing:-0.04em; line-height:1; margin-bottom:0.8rem; }
.pred-advice { font-size:0.83rem; color:var(--txt2); line-height:1.6; max-width:380px; margin:0 auto; }

/* ── INPUTS ── */
.stNumberInput input, .stTextInput input {
  background: var(--bg2) !important; border: 1px solid var(--border2) !important;
  border-radius: 10px !important; color: var(--txt) !important;
  font-size: 0.9rem !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
  border-color: rgba(124,111,239,0.5) !important;
  box-shadow: 0 0 0 3px rgba(124,111,239,0.1) !important;
}
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }
.stButton > button {
  background: linear-gradient(135deg, var(--p1) 0%, var(--p2) 100%) !important;
  color: #fff !important; border: none !important; border-radius: 12px !important;
  font-weight: 700 !important; font-size: 0.9rem !important;
  padding: 0.7rem 2rem !important; letter-spacing: 0.02em !important;
  transition: all 0.2s !important; box-shadow: 0 4px 20px rgba(124,111,239,0.3) !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(124,111,239,0.4) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg1) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; padding: 4px !important; gap: 2px !important;
}
.stTabs [data-baseweb="tab"] { border-radius: 9px !important; font-size:0.8rem !important; font-weight:600 !important; color: var(--txt3) !important; padding: 0.5rem 1rem !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--p1), var(--p2)) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ── SIDEBAR LOGO ── */
.sb-logo {
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.5rem 0 1.5rem 0;
}
.sb-logo-icon {
  width: 40px; height: 40px; border-radius: 12px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--p1), var(--b1));
  display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
  box-shadow: 0 4px 16px rgba(124,111,239,0.4);
}
.sb-logo-name { font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:700; color:var(--txt); letter-spacing:-0.02em; }
.sb-logo-tag  { font-size:0.65rem; color:var(--txt3); }

/* ── SIDEBAR NAV LABEL ── */
.sb-nav-lbl { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--txt3); padding: 0 0 0.4rem 0; }

/* ── SIDEBAR STAT ROW ── */
.sb-stat { display:flex; justify-content:space-between; align-items:center; padding:0.45rem 0; border-bottom:1px solid var(--border); }
.sb-stat-k { font-size:0.73rem; color:var(--txt3); }
.sb-stat-v { font-size:0.73rem; color:var(--txt2); font-weight:600; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { background: var(--bg1) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    rfm     = pd.read_csv(DATA/"rfm.csv")
    monthly = pd.read_csv(DATA/"monthly.csv")
    country = pd.read_csv(DATA/"country.csv")
    top_p   = pd.read_csv(DATA/"top_products.csv")
    heatmap = pd.read_csv(DATA/"heatmap.csv")
    kpis    = json.load(open(DATA/"kpis.json"))
    elbow   = json.load(open(DATA/"elbow.json"))
    prods   = json.load(open(DATA/"products.json"))
    return rfm, monthly, country, top_p, heatmap, kpis, elbow, prods

class TopSimLookup:
    """Lightweight wrapper around a {product: [(sim_product, score), ...]} dict
    that mimics the small subset of pandas.DataFrame API used elsewhere in app.py:
    .index, df[product] -> Series-like, df.loc[list, list] -> small matrix."""
    def __init__(self, records: dict):
        self._records = records
        self.index = list(records.keys())

    def __getitem__(self, product):
        pairs = self._records.get(product, [])
        return pd.Series({p: s for p, s in pairs})

    def loc_matrix(self, products):
        """Build a small dense similarity matrix for a short list of products,
        used only for heatmaps (top 15 products)."""
        n = len(products)
        mat = pd.DataFrame(0.0, index=products, columns=products)
        for p in products:
            pairs = dict(self._records.get(p, []))
            for q in products:
                if q == p:
                    mat.loc[p, q] = 1.0
                elif q in pairs:
                    mat.loc[p, q] = pairs[q]
        return mat


@st.cache_resource
def load_models():
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    scaler = pickle.load(open(MODELS/"scaler.pkl","rb"))
    kmeans = pickle.load(open(MODELS/"kmeans.pkl","rb"))

    compact_path = MODELS/"top_similar.pkl"
    full_path    = MODELS/"cosine_sim.pkl"

    if compact_path.exists():
        records = pickle.load(open(compact_path, "rb"))
        cosine  = TopSimLookup(records)
    elif full_path.exists():
        cosine = pd.read_pickle(full_path)
    else:
        # Auto-build: download CSV from Google Drive, compute cosine similarity
        with st.spinner("Building recommendation engine — downloading dataset (first run only ~2 min)..."):
            import urllib.request
            GDRIVE_ID  = "1EznS2aHv5UcRPd3SBxH7DFQYFDpeaDwM"
            GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}"
            try:
                csv_path = BASE / "online_retail.csv"
                if not csv_path.exists():
                    urllib.request.urlretrieve(GDRIVE_URL, csv_path)
                df_tmp = pd.read_csv(csv_path, encoding="latin1")
            except Exception as dl_err:
                st.error(f"Failed to download dataset: {dl_err}")
                st.stop()
            df_tmp = df_tmp[~df_tmp["InvoiceNo"].astype(str).str.startswith("C")]
            df_tmp = df_tmp.dropna(subset=["CustomerID"])
            df_tmp = df_tmp[df_tmp["Quantity"] > 0]
            df_tmp = df_tmp[df_tmp["UnitPrice"] > 0]
            df_tmp["Description"] = df_tmp["Description"].str.strip().str.upper()
            df_tmp["CustomerID"]  = df_tmp["CustomerID"].astype(int)
            pivot  = df_tmp.pivot_table(index="CustomerID", columns="Description",
                                        values="Quantity", fill_value=0)
            cos         = _cos_sim(pivot.T)
            cosine_full = pd.DataFrame(cos, index=pivot.columns, columns=pivot.columns)
            records = {}
            for prod in cosine_full.index:
                sims = cosine_full[prod].drop(prod).sort_values(ascending=False).head(20)
                records[prod] = list(zip(sims.index, sims.values.round(4)))
            pickle.dump(records, open(compact_path, "wb"))
            cosine = TopSimLookup(records)
    return scaler, kmeans, cosine

rfm, monthly, country, top_p, heatmap, kpis, elbow, prods = load_data()
scaler, kmeans, cosine_df = load_models()

SEG_COLORS = {'High-Value':'#7c6fef','Regular':'#22d3ee','Occasional':'#fb923c','At-Risk':'#f87171'}
PALETTE    = ['#7c6fef','#22d3ee','#34d399','#fb923c','#a78bfa','#f472b6','#facc15','#4ade80']

BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#8892b0', size=11),
    margin=dict(l=8, r=8, t=30, b=8),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10), bordercolor='rgba(255,255,255,0.06)', borderwidth=1),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)', tickfont=dict(size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)', tickfont=dict(size=10)),
)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>
      <div class='sb-logo-icon'>🛒</div>
      <div>
        <div class='sb-logo-name'>Shopper Spectrum</div>
        <div class='sb-logo-tag'>Customer Intelligence</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sb-nav-lbl'>Main Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "📊  Dashboard",
        "👥  Segmentation",
        "🎯  Recommendations",
        "📈  EDA Explorer",
        "🤖  RFM Predictor",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sb-nav-lbl'>Dataset Summary</div>", unsafe_allow_html=True)
    for k, v in [
        ("📅 Period", "Dec 2022 – Dec 2023"),
        ("🌍 Countries", "38"),
        ("📦 Products",  f"{kpis['total_products']:,}"),
        ("👤 Customers", f"{kpis['total_customers']:,}"),
        ("🧾 Orders",    f"{kpis['total_orders']:,}"),
    ]:
        st.markdown(f"<div class='sb-stat'><span class='sb-stat-k'>{k}</span><span class='sb-stat-v'>{v}</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(124,111,239,0.08);border:1px solid rgba(124,111,239,0.2);
                border-radius:12px;padding:1rem;font-size:0.72rem;color:#4a5280;line-height:1.7;'>
      <div style='color:#7c6fef;font-weight:700;margin-bottom:0.3rem;'>⚡ Tech Stack</div>
      KMeans · Collaborative Filtering<br>
      Cosine Similarity · RFM Analysis<br>
      Plotly · Streamlit · Scikit-learn
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":

    # Hero
    st.markdown(f"""
    <div class='hero'>
      <div style='position:relative;z-index:1;'>
        <div class='hero-title'>Shopper Spectrum Analytics</div>
        <div class='hero-sub'>
          Real-time intelligence across <b style='color:#c4b5fd;'>{kpis['total_customers']:,} customers</b>,
          <b style='color:#22d3ee;'>{kpis['total_products']:,} products</b> and
          <b style='color:#34d399;'>£{kpis['total_revenue']:,.0f}</b> in revenue.
          Dec 2022 → Dec 2023.
        </div>
        <div class='hero-pills'>
          <div class='hero-pill purple'>KMeans Segmentation</div>
          <div class='hero-pill cyan'>Collaborative Filtering</div>
          <div class='hero-pill green'>RFM Analysis</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # KPI Row
    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='kpi'>
        <div class='kpi-glow p'></div>
        <span class='kpi-icon'>💷</span>
        <div class='kpi-lbl'>Total Revenue</div>
        <div class='kpi-val'>£{kpis['total_revenue']/1e6:.2f}M</div>
        <div class='kpi-foot'>All clean transactions &nbsp;<span class='kpi-tag info'>Online Retail</span></div>
        <div class='kpi-bar p'></div>
      </div>
      <div class='kpi'>
        <div class='kpi-glow b'></div>
        <span class='kpi-icon'>🧾</span>
        <div class='kpi-lbl'>Total Orders</div>
        <div class='kpi-val'>{kpis['total_orders']:,}</div>
        <div class='kpi-foot'>Unique invoices &nbsp;<span class='kpi-tag up'>↑ 18.5K</span></div>
        <div class='kpi-bar b'></div>
      </div>
      <div class='kpi'>
        <div class='kpi-glow g'></div>
        <span class='kpi-icon'>👤</span>
        <div class='kpi-lbl'>Active Customers</div>
        <div class='kpi-val'>{kpis['total_customers']:,}</div>
        <div class='kpi-foot'>With purchase history &nbsp;<span class='kpi-tag up'>↑ Segmented</span></div>
        <div class='kpi-bar g'></div>
      </div>
      <div class='kpi'>
        <div class='kpi-glow o'></div>
        <span class='kpi-icon'>📦</span>
        <div class='kpi-lbl'>Avg Order Value</div>
        <div class='kpi-val'>£{kpis['avg_order_value']:.0f}</div>
        <div class='kpi-foot'>Per invoice &nbsp;<span class='kpi-tag info'>Per Transaction</span></div>
        <div class='kpi-bar o'></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Row 1 — Revenue trend + Segment donut
    st.markdown("<div class='sec-lbl'>Revenue & Segments</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([2.4, 1])

    with c1:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>📈</span> Monthly Revenue Trend</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly['Month'], y=monthly['Revenue'],
            fill='tozeroy',
            fillcolor='rgba(124,111,239,0.06)',
            line=dict(color='#7c6fef', width=2.5),
            mode='lines+markers',
            marker=dict(size=6, color='#7c6fef', line=dict(color='#c4b5fd', width=1.5)),
            hovertemplate='<b>%{x}</b><br>Revenue: £%{y:,.0f}<extra></extra>',
            name='Revenue'
        ))
        fig.add_trace(go.Scatter(
            x=monthly['Month'], y=monthly['Revenue'].rolling(3, center=True).mean(),
            line=dict(color='#22d3ee', width=1.5, dash='dot'),
            mode='lines', name='3-month avg',
            hovertemplate='Avg: £%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(**BASE_LAYOUT, height=240)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>👥</span> Segments</div>", unsafe_allow_html=True)
        seg_c = rfm['Segment'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=seg_c.index, values=seg_c.values, hole=0.68,
            marker=dict(colors=[SEG_COLORS.get(s,'#94a3b8') for s in seg_c.index],
                        line=dict(color='#07080d', width=3)),
            textinfo='none',
            hovertemplate='<b>%{label}</b><br>%{value} customers (%{percent})<extra></extra>'
        ))
        fig2.add_annotation(
            text=f'<b style="font-size:18px">{rfm.shape[0]:,}</b><br><span style="font-size:10px;color:#4a5280">customers</span>',
            x=0.5, y=0.5, showarrow=False, font=dict(color='#f0f2ff', size=14)
        )
        layout2 = {k:v for k,v in BASE_LAYOUT.items() if k != 'legend'}
        fig2.update_layout(**layout2, height=240, showlegend=True,
                           legend=dict(orientation='v', x=1.02, y=0.5, font=dict(size=10),
                                       bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2 — Products + Country
    st.markdown("<div class='sec-lbl'>Products & Geography</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>📦</span> Top 10 Products by Revenue</div>", unsafe_allow_html=True)
        tp = top_p.head(10).sort_values('Revenue')
        colors_bar = [f'rgba(124,111,239,{0.4 + 0.06*i})' for i in range(len(tp))]
        fig3 = go.Figure(go.Bar(
            x=tp['Revenue'], y=[d[:30]+'…' if len(d)>30 else d for d in tp['Description']],
            orientation='h',
            marker=dict(color=colors_bar, line=dict(color='rgba(0,0,0,0)')),
            hovertemplate='<b>%{y}</b><br>£%{x:,.0f}<extra></extra>'
        ))
        fig3.update_layout(**BASE_LAYOUT, height=290)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>🌍</span> Revenue by Country — Top 10</div>", unsafe_allow_html=True)
        c10 = country.head(10)
        fig4 = go.Figure(go.Bar(
            x=c10['Country'], y=c10['Revenue'],
            marker=dict(
                color=c10['Revenue'],
                colorscale=[[0,'#111422'],[0.5,'#1a1d38'],[1,'#22d3ee']],
                line=dict(color='rgba(0,0,0,0)')
            ),
            hovertemplate='<b>%{x}</b><br>£%{y:,.0f}<extra></extra>'
        ))
        fig4.update_layout(**BASE_LAYOUT, height=290)
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3 — Heatmap + Orders/Customers
    st.markdown("<div class='sec-lbl'>Time Patterns</div>", unsafe_allow_html=True)
    c5, c6 = st.columns([1.4, 1])

    with c5:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>🕐</span> Purchase Heatmap — Day × Hour</div>", unsafe_allow_html=True)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        hm = heatmap.pivot_table(index='DayOfWeek', columns='Hour', values='TotalPrice', aggfunc='sum', fill_value=0)
        hm = hm.reindex([d for d in day_order if d in hm.index])
        fig5 = go.Figure(go.Heatmap(
            z=hm.values,
            x=[f"{h:02d}h" for h in hm.columns],
            y=hm.index,
            colorscale=[[0,'#07080d'],[0.3,'#1a1238'],[0.7,'#4c1d95'],[1,'#7c6fef']],
            hovertemplate='<b>%{y}</b> %{x}<br>Revenue: £%{z:,.0f}<extra></extra>',
            showscale=False
        ))
        fig5.update_layout(**BASE_LAYOUT, height=230)
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>📋</span> Orders vs Customers / Month</div>", unsafe_allow_html=True)
        fig6 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        fig6.add_trace(go.Bar(x=monthly['Month'], y=monthly['Orders'],
                              marker_color='rgba(124,111,239,0.7)', name='Orders'), row=1, col=1)
        fig6.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Customers'],
                                  line=dict(color='#22d3ee', width=2), mode='lines+markers',
                                  marker=dict(size=4), name='Customers'), row=2, col=1)
        fig6.update_layout(**BASE_LAYOUT, height=230, showlegend=False)
        fig6.update_yaxes(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(size=9))
        st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════
elif page == "👥  Segmentation":
    st.markdown("<div class='page-hdr'><h1>Customer Segmentation</h1><p>RFM-based KMeans clustering across 4 behavioral groups</p></div>", unsafe_allow_html=True)

    seg_stats = rfm.groupby('Segment').agg(
        Count=('CustomerID','count'),
        AvgR=('Recency','mean'), AvgF=('Frequency','mean'), AvgM=('Monetary','mean')
    ).reset_index()

    cfg = {
        'High-Value': ('hv','🏆','#a78bfa','Top spenders, recent & frequent. Reward with VIP treatment.'),
        'Regular':    ('reg','🔄','#22d3ee','Steady buyers. Nurture with targeted upsells.'),
        'Occasional': ('occ','🕐','#fb923c','Low engagement. Re-activate with discounts.'),
        'At-Risk':    ('ar','⚠️','#f87171','Lapsed buyers. Launch win-back now.'),
    }

    st.markdown("<div class='sec-lbl'>Segment Profiles</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (_, row) in enumerate(seg_stats.iterrows()):
        seg = row['Segment']
        cls, icon, color, desc = cfg.get(seg, ('occ','●','#94a3b8',''))
        pct = row['Count'] / rfm.shape[0] * 100
        with cols[i]:
            st.markdown(f"""
            <div class='seg-card {cls}'>
              <div class='seg-icon'>{icon}</div>
              <div class='seg-name' style='color:{color};'>{seg}</div>
              <div class='seg-count' style='color:{color};'>{row['Count']:,}</div>
              <div class='seg-desc'>{desc}</div>
              <div class='seg-stat'>
                {pct:.1f}% of customers<br>
                <b>Recency:</b> {row['AvgR']:.0f}d &nbsp;
                <b>Freq:</b> {row['AvgF']:.1f} &nbsp;
                <b>£{row['AvgM']:,.0f}</b>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-lbl'>Cluster Visualizations</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>🔵</span> Recency vs Monetary (bubble = frequency)</div>", unsafe_allow_html=True)
        pd_plot = rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.98)].copy()
        fig = px.scatter(pd_plot, x='Recency', y='Monetary', color='Segment',
                         size='Frequency', size_max=22, opacity=0.7,
                         color_discrete_map=SEG_COLORS,
                         hover_data={'CustomerID':True,'Frequency':True,'Recency':True,'Monetary':':.0f'})
        fig.update_layout(**BASE_LAYOUT, height=330)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>📊</span> Frequency Distribution by Segment</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        for seg, col in SEG_COLORS.items():
            d = rfm[rfm['Segment']==seg]['Frequency']
            if len(d):
                r,g,b = int(col[1:3],16), int(col[3:5],16), int(col[5:7],16)
                fig2.add_trace(go.Histogram(x=d, name=seg, nbinsx=25,
                                            marker_color=f'rgba({r},{g},{b},0.7)'))
        fig2.update_layout(**BASE_LAYOUT, height=330, barmode='overlay')
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # 3D
    st.markdown("<div class='g-card'>", unsafe_allow_html=True)
    st.markdown("<div class='g-card-title'><span>🌐</span> 3D RFM Cluster Space</div>", unsafe_allow_html=True)
    p3d = rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.97)]
    fig3d = px.scatter_3d(p3d, x='Recency', y='Frequency', z='Monetary', color='Segment',
                           opacity=0.65, color_discrete_map=SEG_COLORS, hover_data=['CustomerID'])
    fig3d.update_traces(marker=dict(size=2.5))
    fig3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#8892b0'),
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.04)', color='#4a5280', title='Recency'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.04)', color='#4a5280', title='Frequency'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.04)', color='#4a5280', title='Monetary'),
        ),
        margin=dict(l=0,r=0,t=10,b=0), height=430,
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig3d, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Table
    st.markdown("<div class='sec-lbl'>Customer RFM Table</div>", unsafe_allow_html=True)
    sf = st.selectbox("Filter Segment", ["All"]+list(rfm['Segment'].unique()), label_visibility="collapsed")
    disp = rfm if sf=="All" else rfm[rfm['Segment']==sf]
    st.dataframe(
        disp[['CustomerID','Recency','Frequency','Monetary','Segment']].sort_values('Monetary',ascending=False).head(200).reset_index(drop=True),
        use_container_width=True, height=320
    )


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
elif page == "🎯  Recommendations":
    st.markdown("<div class='page-hdr'><h1>Product Recommendations</h1><p>Item-based Collaborative Filtering · Cosine Similarity · 3,866 products</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1.2])

    with c1:
        st.markdown("<div class='sec-lbl'>Find Similar Products</div>", unsafe_allow_html=True)
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)

        query = st.text_input("", placeholder="e.g.  WHITE HANGING HEART T-LIGHT HOLDER", label_visibility="collapsed")
        n     = st.slider("Recommendations to show", 3, 10, 5, label_visibility="visible")

        if st.button("🎯  Get Recommendations", use_container_width=True):
            if not query.strip():
                st.warning("Enter a product name above.")
            else:
                q = query.strip().upper()
                matches = [p for p in cosine_df.index if q in p]
                if not matches:
                    matches = [p for p in cosine_df.index if any(w in p for w in q.split())]
                if not matches:
                    st.error("No product matched. Try a shorter keyword.")
                else:
                    best = matches[0]
                    st.markdown(f"<div style='font-size:0.75rem;color:#4a5280;margin-bottom:1rem;padding:0.6rem 0.8rem;background:rgba(124,111,239,0.06);border-radius:8px;border:1px solid rgba(124,111,239,0.15);'>Matched: <b style='color:#a78bfa;'>{best.title()}</b></div>", unsafe_allow_html=True)
                    sims = cosine_df[best].drop(best).sort_values(ascending=False).head(n)
                    for i, (prod, score) in enumerate(sims.items(), 1):
                        bar_w = int(score * 100)
                        st.markdown(f"""
                        <div class='rec-item'>
                          <div class='rec-rank'>{i}</div>
                          <div class='rec-name'>{prod.title()}</div>
                          <div style='text-align:right;'>
                            <div class='rec-pct'>{score:.1%}</div>
                            <div style='width:60px;height:3px;border-radius:2px;background:rgba(255,255,255,0.05);margin-top:3px;'>
                              <div style='width:{bar_w}%;height:100%;border-radius:2px;background:linear-gradient(90deg,#7c6fef,#a78bfa);'></div>
                            </div>
                          </div>
                        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='sec-lbl'>Top Products</div>", unsafe_allow_html=True)
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>🏅</span> Top 12 by Revenue</div>", unsafe_allow_html=True)
        for i, (_, row) in enumerate(top_p.head(12).iterrows(), 1):
            name = row['Description'].title()
            if len(name) > 32: name = name[:32] + '…'
            pct = row['Revenue'] / top_p['Revenue'].max()
            st.markdown(f"""
            <div style='margin-bottom:0.5rem;'>
              <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                <span style='font-size:0.75rem;color:#8892b0;'><span style='color:#4a5280;'>#{i:02d}</span> {name}</span>
                <span style='font-size:0.72rem;font-weight:700;color:#7c6fef;'>£{row["Revenue"]:,.0f}</span>
              </div>
              <div style='height:2px;background:rgba(255,255,255,0.04);border-radius:1px;'>
                <div style='width:{pct*100:.0f}%;height:100%;background:linear-gradient(90deg,#7c6fef,#22d3ee);border-radius:1px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Similarity heatmap
    st.markdown("<div class='sec-lbl'>Product Similarity Heatmap</div>", unsafe_allow_html=True)
    st.markdown("<div class='g-card'>", unsafe_allow_html=True)
    top15 = list(top_p.head(15)['Description'])
    sub   = cosine_df.loc[top15, top15]
    labels15 = [p[:22]+'…' if len(p)>22 else p for p in sub.columns]
    fig_hm = go.Figure(go.Heatmap(
        z=sub.values, x=labels15, y=labels15,
        colorscale=[[0,'#07080d'],[0.3,'#1a1238'],[0.7,'#4c1d95'],[1,'#a78bfa']],
        showscale=True, colorbar=dict(thickness=10, tickfont=dict(size=9), tickcolor='#4a5280'),
        hovertemplate='%{y}<br>× %{x}<br>Sim: %{z:.2%}<extra></extra>'
    ))
    fig_hm.update_layout(**BASE_LAYOUT, height=440)
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════
elif page == "📈  EDA Explorer":
    st.markdown("<div class='page-hdr'><h1>EDA Explorer</h1><p>Deep-dive into transaction patterns, distributions, and model evaluation</p></div>", unsafe_allow_html=True)

    tabs = st.tabs(["📅 Time Analysis", "🌍 Geo Analysis", "📦 Products", "🎻 RFM Distributions", "📉 Model Evaluation"])

    with tabs[0]:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=['Monthly Revenue (£)', 'Orders & Customers'])
        fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Revenue'],
                                 fill='tozeroy', fillcolor='rgba(124,111,239,0.06)',
                                 line=dict(color='#7c6fef', width=2.5), name='Revenue'), row=1, col=1)
        fig.add_trace(go.Bar(x=monthly['Month'], y=monthly['Orders'],
                             marker_color='rgba(34,211,238,0.7)', name='Orders'), row=2, col=1)
        fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Customers'],
                                 line=dict(color='#34d399', width=2), mode='lines+markers',
                                 marker=dict(size=5, color='#34d399'), name='Customers'), row=2, col=1)
        fig.update_layout(**BASE_LAYOUT, height=480, showlegend=True)
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(size=9))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("<div class='g-card'>", unsafe_allow_html=True)
            st.markdown("<div class='g-card-title'><span>🌍</span> Revenue Share</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Pie(
                labels=country['Country'].head(10), values=country['Revenue'].head(10),
                hole=0.55, marker=dict(colors=PALETTE, line=dict(color='#07080d', width=2)),
                hovertemplate='<b>%{label}</b><br>£%{value:,.0f} (%{percent})<extra></extra>'
            ))
            fig.update_layout(**BASE_LAYOUT, height=320)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='g-card'>", unsafe_allow_html=True)
            st.markdown("<div class='g-card-title'><span>📋</span> Country Revenue Table</div>", unsafe_allow_html=True)
            st.dataframe(country.head(15).style.format({'Revenue':'£{:,.0f}','Orders':'{:,}'}),
                         use_container_width=True, height=320)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div class='g-card-title'><span>📦</span> Top 20 Products — Revenue vs Quantity</div>", unsafe_allow_html=True)
        fig = px.scatter(top_p, x='Quantity', y='Revenue', size='Revenue', color='Revenue',
                         hover_data=['Description'], color_continuous_scale='Purples',
                         text='Description')
        fig.update_traces(textposition='top center', textfont=dict(size=7, color='#8892b0'))
        fig.update_layout(**BASE_LAYOUT, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        cols3 = st.columns(3)
        for col_, metric, label in zip(cols3,
            ['Recency','Frequency','Monetary'],
            ['Recency (days)','Frequency (orders)','Monetary (£)']):
            with col_:
                st.markdown("<div class='g-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='g-card-title'><span>🎻</span> {label}</div>", unsafe_allow_html=True)
                cap = rfm[metric].quantile(0.95)
                fig = go.Figure()
                for seg, clr in SEG_COLORS.items():
                    d = rfm[rfm['Segment']==seg][metric]
                    d = d[d <= cap]
                    if len(d):
                        r,g,b = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
                        fig.add_trace(go.Violin(y=d, name=seg,
                                                fillcolor=f'rgba({r},{g},{b},0.18)',
                                                line_color=clr, box_visible=True,
                                                meanline_visible=True))
                fig.update_layout(**BASE_LAYOUT, height=310, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Elbow Curve (Inertia)', 'Silhouette Score'])
        fig.add_trace(go.Scatter(x=elbow['k'], y=elbow['inertia'],
                                 mode='lines+markers+text', text=[f'{v:,.0f}' for v in elbow['inertia']],
                                 textposition='top center', textfont=dict(size=9),
                                 marker=dict(color='#7c6fef', size=9, line=dict(color='#c4b5fd',width=1.5)),
                                 line=dict(color='#7c6fef', width=2.5), name='Inertia'), row=1, col=1)
        fig.add_trace(go.Scatter(x=elbow['k'], y=elbow['silhouette'],
                                 mode='lines+markers+text', text=[f'{v:.2f}' for v in elbow['silhouette']],
                                 textposition='top center', textfont=dict(size=9),
                                 marker=dict(color='#34d399', size=9, line=dict(color='#6ee7b7',width=1.5)),
                                 line=dict(color='#34d399', width=2.5), name='Silhouette'), row=1, col=2)
        fig.add_vline(x=4, line_dash='dot', line_color='rgba(251,146,60,0.6)',
                      annotation_text='k = 4  ✓', annotation_font_color='#fb923c', row=1, col=1)
        fig.add_vline(x=4, line_dash='dot', line_color='rgba(251,146,60,0.6)', row=1, col=2)
        fig.update_layout(**BASE_LAYOUT, height=360, showlegend=False)
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(size=9))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("""
        <div style='margin-top:0.5rem;padding:0.8rem 1rem;border-radius:10px;
                    background:rgba(124,111,239,0.06);border:1px solid rgba(124,111,239,0.12);
                    font-size:0.78rem;color:#8892b0;line-height:1.7;'>
          <b style='color:#a78bfa;'>Why k = 4?</b> &nbsp;Selected for business interpretability — maps cleanly
          to High-Value, Regular, Occasional, and At-Risk customer types.
          Silhouette score confirms good intra-cluster cohesion.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — RFM PREDICTOR
# ══════════════════════════════════════════════════════════════════
elif page == "🤖  RFM Predictor":
    st.markdown("<div class='page-hdr'><h1>RFM Segment Predictor</h1><p>Enter any customer's Recency · Frequency · Monetary values to predict their segment in real-time</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1.9])

    with c1:
        st.markdown("<div class='sec-lbl'>Customer Input</div>", unsafe_allow_html=True)
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#4a5280;margin-bottom:0.8rem;'>RFM Values</div>", unsafe_allow_html=True)
        recency   = st.number_input("🕐 Recency — days since last purchase", 1, 400, 30)
        frequency = st.number_input("🔄 Frequency — number of orders",       1, 300, 5)
        monetary  = st.number_input("💰 Monetary — total spend (£)",         1.0, 300000.0, 500.0, step=50.0)

        st.markdown("<br>", unsafe_allow_html=True)
        predict   = st.button("🤖  Predict Segment", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#4a5280;margin-bottom:0.5rem;'>Segment Guide</div>", unsafe_allow_html=True)
        for ico, seg, hint in [
            ('🏆','High-Value','Low R · High F · High M'),
            ('🔄','Regular',   'Low R · Med F · Med M'),
            ('🕐','Occasional','Med R · Low F · Low M'),
            ('⚠️','At-Risk',  'High R · Low F · Low M'),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:0.45rem 0;border-bottom:1px solid rgba(255,255,255,0.04);'>
              <span style='font-size:0.75rem;color:#8892b0;'>{ico} {seg}</span>
              <span style='font-size:0.68rem;color:#4a5280;'>{hint}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        if predict:
            inp     = np.array([[recency, frequency, monetary]])
            cluster = kmeans.predict(scaler.transform(inp))[0]
            labels  = json.load(open(DATA/"cluster_labels.json"))
            segment = labels.get(str(cluster), "Unknown")

            smap = {
                'High-Value': ('hv','🏆','#a78bfa','This is your most valuable customer type — recent visits, frequent orders, high spend. Reward with VIP perks, early access, and loyalty programs.'),
                'Regular':    ('reg','🔄','#22d3ee','A consistent buyer with solid engagement. Nurture with personalised upsells, cross-sell bundles, and targeted campaigns.'),
                'Occasional': ('occ','🕐','#fb923c','Low engagement. Win them back with a time-limited offer, re-engagement email, or a personalised discount nudge.'),
                'At-Risk':    ('ar','⚠️','#f87171','Danger zone — this customer hasn\'t purchased in a long time. Launch a win-back campaign immediately before they churn permanently.'),
            }
            cls, icon, color, advice = smap.get(segment, ('occ','●','#94a3b8',''))

            st.markdown("<div class='sec-lbl'>Prediction Result</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='pred-result {cls}' style='margin-bottom:1.2rem;'>
              <div class='pred-lbl'>Predicted Segment</div>
              <div class='pred-seg' style='color:{color};'>{icon} {segment}</div>
              <div class='pred-advice'>{advice}</div>
            </div>""", unsafe_allow_html=True)

            # Gauges
            rfm_max = rfm[['Recency','Frequency','Monetary']].quantile(0.95)
            rfm_med = rfm[['Recency','Frequency','Monetary']].median()

            st.markdown("<div class='g-card'>", unsafe_allow_html=True)
            st.markdown("<div class='g-card-title'><span>🎯</span> Customer Position vs Dataset</div>", unsafe_allow_html=True)
            gauges = make_subplots(rows=1, cols=3, specs=[[{'type':'indicator'}]*3])
            for i, (m, val, col_) in enumerate([
                ('Recency',   recency,   '#f87171'),
                ('Frequency', frequency, '#22d3ee'),
                ('Monetary',  monetary,  '#34d399'),
            ], 1):
                gauges.add_trace(go.Indicator(
                    mode='gauge+number',
                    value=val,
                    title=dict(text=m, font=dict(size=11, color='#8892b0')),
                    gauge=dict(
                        axis=dict(range=[0, float(rfm_max[m])], tickcolor='#4a5280', tickfont=dict(size=8)),
                        bar=dict(color=col_, thickness=0.25),
                        bgcolor='rgba(255,255,255,0.02)',
                        bordercolor='rgba(255,255,255,0.06)',
                        steps=[
                            dict(range=[0, float(rfm_med[m])],         color='rgba(124,111,239,0.05)'),
                            dict(range=[float(rfm_med[m]), float(rfm_max[m])], color='rgba(124,111,239,0.02)')
                        ],
                        threshold=dict(line=dict(color='#7c6fef', width=2), thickness=0.75, value=float(rfm_med[m]))
                    ),
                    number=dict(font=dict(color='#f0f2ff', size=16), suffix='d' if m=='Recency' else '')
                ), row=1, col=i)
            gauges.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#8892b0'),
                                 margin=dict(l=10,r=10,t=30,b=10), height=200)
            st.plotly_chart(gauges, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

            # Comparison bar
            st.markdown("<div class='g-card'>", unsafe_allow_html=True)
            st.markdown("<div class='g-card-title'><span>📊</span> vs Segment Averages</div>", unsafe_allow_html=True)
            seg_means = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean()
            metrics   = ['Recency','Frequency','Monetary']
            fig_bar   = go.Figure()
            for sn, sc in SEG_COLORS.items():
                if sn in seg_means.index:
                    r,g,b = int(sc[1:3],16), int(sc[3:5],16), int(sc[5:7],16)
                    fig_bar.add_trace(go.Bar(name=sn, x=metrics,
                                            y=[seg_means.loc[sn,m] for m in metrics],
                                            marker_color=f'rgba({r},{g},{b},0.55)'))
            fig_bar.add_trace(go.Bar(name='This Customer', x=metrics,
                                     y=[recency, frequency, monetary],
                                     marker_color='rgba(251,191,36,0.9)',
                                     marker_line=dict(color='#fbbf24', width=1.5)))
            fig_bar.update_layout(**BASE_LAYOUT, height=250, barmode='group')
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:460px;gap:1.2rem;'>
              <div style='width:80px;height:80px;border-radius:24px;
                          background:linear-gradient(135deg,rgba(124,111,239,0.15),rgba(34,211,238,0.08));
                          border:1px solid rgba(124,111,239,0.2);
                          display:flex;align-items:center;justify-content:center;font-size:2.5rem;'>
                🤖
              </div>
              <div style='font-family:"Space Grotesk",sans-serif;font-size:1.1rem;font-weight:700;color:#f0f2ff;'>
                Ready to Predict
              </div>
              <div style='font-size:0.82rem;color:#4a5280;text-align:center;max-width:260px;line-height:1.7;'>
                Enter Recency, Frequency, and Monetary values on the left,
                then click <b style="color:#8892b0;">Predict Segment</b>.
              </div>
              <div style='display:flex;gap:0.5rem;flex-wrap:wrap;justify-content:center;margin-top:0.5rem;'>
                <span style='font-size:0.7rem;padding:4px 12px;border-radius:20px;background:rgba(167,139,250,0.1);color:#a78bfa;border:1px solid rgba(167,139,250,0.2);'>🏆 High-Value</span>
                <span style='font-size:0.7rem;padding:4px 12px;border-radius:20px;background:rgba(34,211,238,0.1);color:#22d3ee;border:1px solid rgba(34,211,238,0.2);'>🔄 Regular</span>
                <span style='font-size:0.7rem;padding:4px 12px;border-radius:20px;background:rgba(251,146,60,0.1);color:#fb923c;border:1px solid rgba(251,146,60,0.2);'>🕐 Occasional</span>
                <span style='font-size:0.7rem;padding:4px 12px;border-radius:20px;background:rgba(248,113,113,0.1);color:#f87171;border:1px solid rgba(248,113,113,0.2);'>⚠️ At-Risk</span>
              </div>
            </div>""", unsafe_allow_html=True)
