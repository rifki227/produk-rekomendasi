"""
dashboard/main.py
Product Recommendation Analytics Dashboard
Capstone Project CC26-PRU466 — Neural Collaborative Filtering
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NCF Analytics · CC26-PRU466",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root ── */
:root {
    --bg:        #070b14;
    --bg2:       #0d1220;
    --bg3:       #111827;
    --border:    #1e2a3f;
    --border2:   #2a3a55;
    --text:      #e2e8f7;
    --muted:     #5a6a8a;
    --accent:    #38bdf8;
    --accent2:   #818cf8;
    --accent3:   #34d399;
    --accent4:   #fb923c;
    --accent5:   #f472b6;
    --danger:    #f87171;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── Base ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1600px !important; }
#MainMenu, footer { visibility: hidden !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

/* Fix sidebar overlap */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

[data-testid="collapsedControl"] {
    z-index: 9999 !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: var(--muted) !important;
    font-size:12px !important;
    text-transform:uppercase;
    letter-spacing:.06em;
}

[data-testid="stSidebar"] [data-baseweb="select"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] [data-testid="stSliderTrack"] {
    background: var(--border2) !important;
}

[data-testid="stSidebar"] [data-testid="stSliderThumb"] {
    background: var(--accent) !important;
}
/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
[data-testid="metric-container"]:hover { border-color: var(--border2) !important; }
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-family: var(--font-mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    font-family: var(--font-head) !important;
    line-height: 1.2 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: var(--font-head) !important; color: var(--text) !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: transparent !important; gap: 4px !important; }
[data-baseweb="tab"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: var(--bg) !important;
    font-weight: 600 !important;
}
[data-baseweb="tab-highlight"] { display:none !important; }
[data-baseweb="tab-border"] { display:none !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--bg2) !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; font-family: var(--font-mono) !important; font-size:13px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* ── Selectbox / input ── */
[data-baseweb="select"] { background: var(--bg3) !important; }
[data-baseweb="input"] { background: var(--bg3) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":    "rgba(0,0,0,0)",
    "grid":  "#1e2a3f",
    "text":  "#e2e8f7",
    "muted": "#5a6a8a",
    "a1":    "#38bdf8",   # cyan
    "a2":    "#818cf8",   # indigo
    "a3":    "#34d399",   # emerald
    "a4":    "#fb923c",   # orange
    "a5":    "#f472b6",   # pink
    "a6":    "#facc15",   # yellow
    "danger":"#f87171",
}
PALETTE = [C["a1"], C["a2"], C["a3"], C["a4"], C["a5"], C["a6"], C["danger"]]
RATING_COLORS = {1.0: C["danger"], 2.0: C["a4"], 3.0: C["a6"], 4.0: C["a3"], 5.0: C["a1"]}

BASE_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
    font=dict(color=C["text"], family="DM Sans, sans-serif", size=12),
    margin=dict(l=4, r=4, t=36, b=4),
    xaxis=dict(gridcolor=C["grid"], linecolor=C["grid"], tickcolor=C["muted"], zeroline=False),
    yaxis=dict(gridcolor=C["grid"], linecolor=C["grid"], tickcolor=C["muted"], zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["grid"], font=dict(size=11)),
    hoverlabel=dict(bgcolor="#111827", bordercolor="#1e2a3f", font=dict(color=C["text"])),
)

def layout(**overrides):
    return {**BASE_LAYOUT, **overrides}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Memuat dataset …")
def load_data():
    candidates = [
        Path("data/sample_data.csv"),
        Path("../data/sample_data.csv"),
        Path("CC26-PRU466-Product-Recommendation-main/data/sample_data.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        st.error("❌ File `data/sample_data.csv` tidak ditemukan.")
        st.stop()

    df = pd.read_csv(path)
    df["timestamp"]  = pd.to_datetime(df["timestamp"], unit="s")
    df["year"]       = df["timestamp"].dt.year
    df["month"]      = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.strftime("%b")
    df["quarter"]    = df["timestamp"].dt.quarter
    df["dayofweek"]  = df["timestamp"].dt.dayofweek
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["rating"]     = df["rating"].astype(float)
    return df


df_raw = load_data()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;"
        "color:#38bdf8;letter-spacing:-.02em;margin-bottom:4px'>🧠 NCF Analytics</div>"
        "<div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace;"
        "margin-bottom:20px'>CC26-PRU466 · E-Commerce</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    year_min, year_max = int(df_raw["year"].min()), int(df_raw["year"].max())
    year_range = st.slider("📅 Rentang Tahun", year_min, year_max, (2007, year_max))

    rating_filter = st.multiselect(
        "⭐ Filter Rating",
        options=[1.0, 2.0, 3.0, 4.0, 5.0],
        default=[1.0, 2.0, 3.0, 4.0, 5.0],
        format_func=lambda x: f"{'★'*int(x)} {'☆'*(5-int(x))}  {int(x)}/5",
    )

    top_n = st.slider("🏆 Top N untuk Ranking", 5, 30, 15)

    min_reviews = st.slider("🔎 Min. Reviews per Produk", 1, 50, 5)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace'>"
        "STATISTIK DATASET</div>", unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='font-size:13px;line-height:2;color:#8898b8'>"
        f"Total baris &nbsp;·&nbsp; <b style='color:#e2e8f7'>{len(df_raw):,}</b><br>"
        f"Unique users &nbsp;·&nbsp; <b style='color:#e2e8f7'>{df_raw['user_id'].nunique():,}</b><br>"
        f"Unique products &nbsp;·&nbsp; <b style='color:#e2e8f7'>{df_raw['product_id'].nunique():,}</b><br>"
        f"Periode &nbsp;·&nbsp; <b style='color:#e2e8f7'>{year_min} – {year_max}</b>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    page = st.radio(
        "🗂️ Halaman",
        ["Overview", "Produk & Ulasan", "Analisis Pengguna", "Tren & Waktu", "Model NCF"],
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY FILTERS
# ══════════════════════════════════════════════════════════════════════════════
mask = df_raw["year"].between(*year_range) & df_raw["rating"].isin(rating_filter)
df   = df_raw[mask].copy()

if df.empty:
    st.warning("⚠️ Tidak ada data untuk filter yang dipilih.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def section(title: str, icon: str = ""):
    st.markdown(
        f"<div style='font-family:Syne,sans-serif;font-size:17px;font-weight:700;"
        f"color:#e2e8f7;margin:28px 0 14px;padding-bottom:10px;"
        f"border-bottom:1px solid #1e2a3f'>{icon} {title}</div>",
        unsafe_allow_html=True
    )

def badge(label: str, value: str, color: str = "#38bdf8"):
    return (
        f"<span style='display:inline-block;background:{color}18;border:1px solid {color}44;"
        f"color:{color};font-family:DM Mono,monospace;font-size:12px;border-radius:6px;"
        f"padding:2px 10px;margin:2px'>{label}: <b>{value}</b></span>"
    )

def pct_change(a, b):
    return ((b - a) / a * 100) if a else 0


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
page_titles = {
    "Overview":           ("Overview", "Gambaran Besar Dataset & KPI Utama"),
    "Produk & Ulasan":    ("Produk & Ulasan", "Analisis Mendalam Katalog Produk"),
    "Analisis Pengguna":  ("Analisis Pengguna", "Segmentasi, Retensi & Perilaku User"),
    "Tren & Waktu":       ("Tren & Waktu", "Pola Temporal & Seasonality"),
    "Model NCF":          ("Model NCF", "Insight Model Neural Collaborative Filtering"),
}
ptitle, psub = page_titles[page]
st.markdown(
    f"<div style='margin-bottom:6px'>"
    f"<span style='font-family:Syne,sans-serif;font-size:30px;font-weight:800;"
    f"letter-spacing:-.03em;color:#e2e8f7'>{ptitle}</span>"
    f"<span style='font-size:13px;color:#5a6a8a;margin-left:14px;"
    f"font-family:DM Mono,monospace'>{psub}</span></div>",
    unsafe_allow_html=True
)

# Filter summary badges
st.markdown(
    badge("tahun", f"{year_range[0]}–{year_range[1]}") +
    badge("ulasan", f"{len(df):,}", "#818cf8") +
    badge("user", f"{df['user_id'].nunique():,}", "#34d399") +
    badge("produk", f"{df['product_id'].nunique():,}", "#fb923c"),
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    # ── KPI Row ──────────────────────────────────────────────────────────────
    avg_r   = df["rating"].mean()
    pct5    = (df["rating"] == 5).mean() * 100
    pct1    = (df["rating"] == 1).mean() * 100
    sparsity = (1 - len(df) / (df["user_id"].nunique() * df["product_id"].nunique())) * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📝 Total Ulasan",     f"{len(df):,}")
    c2.metric("👤 Pengguna Unik",    f"{df['user_id'].nunique():,}")
    c3.metric("📦 Produk Unik",      f"{df['product_id'].nunique():,}")
    c4.metric("⭐ Avg Rating",       f"{avg_r:.3f}")
    c5.metric("🟢 Rating 5★",        f"{pct5:.1f}%")
    c6.metric("🔴 Rating 1★",        f"{pct1:.1f}%")


    # ── Row 1: Tren Tahunan + Pie Distribusi ─────────────────────────────────
    section("Tren & Distribusi Rating", "📈")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        yearly = (
            df.groupby("year")
            .agg(count=("rating", "count"), avg=("rating", "mean"))
            .reset_index()
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=yearly["year"], y=yearly["count"],
            name="Jumlah Ulasan",
            marker=dict(
                color=yearly["count"],
                colorscale=[[0, "#1e2a3f"], [1, C["a1"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>%{y:,} ulasan<extra></extra>",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["avg"].round(3),
            name="Avg Rating",
            mode="lines+markers",
            line=dict(color=C["a4"], width=2.5),
            marker=dict(size=8, color=C["a4"],
                        line=dict(color=C["bg"], width=2)),
            hovertemplate="<b>%{x}</b><br>Avg: %{y:.3f}★<extra></extra>",
        ), secondary_y=True)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom"
            )
        )
        fig.update_yaxes(title_text="Ulasan", secondary_y=False, gridcolor=C["grid"])
        fig.update_yaxes(title_text="Rating", secondary_y=True,
                         range=[3.5, 5.0], gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        dist = df["rating"].value_counts().sort_index().reset_index()
        dist.columns = ["rating", "count"]
        dist["color"] = dist["rating"].map(RATING_COLORS)
        dist["label"] = dist["rating"].map(lambda x: f"{'★'*int(x)} ({int(x)})")
        dist["pct"]   = (dist["count"] / dist["count"].sum() * 100).round(1)

        fig2 = go.Figure(go.Pie(
            labels=dist["label"], values=dist["count"],
            hole=0.62,
            marker=dict(colors=dist["color"].tolist(),
                        line=dict(color="#070b14", width=3)),
            textinfo="percent",
            textfont=dict(size=12, color="#fff"),
            direction="clockwise",
            hovertemplate="<b>%{label}</b><br>%{value:,} ulasan · %{percent}<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"<b style='font-size:22px'>{avg_r:.2f}</b><br><span style='font-size:11px'>avg ★</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=C["text"]), align="center",
        )
        
        fig2.update_layout(
            title="Rating Chart",
            legend=dict(orientation="h", y=1.1, x=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Heatmap Year×Month + Sparsity Card ─────────────────────────
    section("Heatmap Aktivitas", "🗓️")
    col_h, col_s = st.columns([3, 2])

    with col_h:
        heat = (
            df.groupby(["year", "month"])["rating"]
            .count().reset_index()
            .pivot(index="year", columns="month", values="rating")
            .fillna(0)
        )
        month_labels = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
        cols_present = [c for c in range(1, 13) if c in heat.columns]

        fig4 = go.Figure(go.Heatmap(
            z=heat[cols_present].values,
            x=[month_labels[c-1] for c in cols_present],
            y=heat.index.astype(str),
            colorscale=[[0, "#0d1220"], [0.3, "#1e3a5f"], [0.7, C["a2"]], [1.0, C["a1"]]],
            hovertemplate="<b>%{y} %{x}</b><br>%{z:,} ulasan<extra></extra>",
            colorbar=dict(title="Ulasan", tickfont=dict(color=C["muted"]), len=0.8),
        ))
        fig4.update_layout(title="Volume Ulasan (Tahun × Bulan)", **layout())
        st.plotly_chart(fig4, use_container_width=True)

    with col_s:
        # Stat cards vertikal
        n_u = df["user_id"].nunique()
        n_p = df["product_id"].nunique()
        n_i = len(df)
        spar = (1 - n_i / (n_u * n_p)) * 100
        avg_per_user = n_i / n_u
        avg_per_prod = n_i / n_p

        stats_data = [
            ("Sparsity Matrix",    f"{spar:.2f}%",       C["danger"]),
            ("Avg Ulasan / User",  f"{avg_per_user:.1f}", C["a3"]),
            ("Avg Ulasan / Produk",f"{avg_per_prod:.2f}", C["a2"]),
            ("Interaksi Total",    f"{n_i:,}",            C["a1"]),
            ("Kombinasi Potensial",f"{n_u*n_p:,}",        C["muted"]),
        ]
        for label, value, color in stats_data:
            st.markdown(
                f"<div style='background:#0d1220;border:1px solid #1e2a3f;"
                f"border-left:3px solid {color};border-radius:0 10px 10px 0;"
                f"padding:12px 16px;margin-bottom:10px'>"
                f"<div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace;"
                f"text-transform:uppercase;letter-spacing:.06em'>{label}</div>"
                f"<div style='font-size:22px;font-weight:700;font-family:Syne,sans-serif;"
                f"color:{color};margin-top:2px'>{value}</div></div>",
                unsafe_allow_html=True
            )

    # ── Model Info Card ───────────────────────────────────────────────────
    section("Info Model", "🧠")
    c1, c2, c3, c4 = st.columns(4)
    info = [
        ("Arsitektur", "Neural Collaborative Filtering", C["a1"]),
        ("Input Users", f"13,345", C["a2"]),
        ("Input Products", f"27,447", C["a3"]),
        ("Output Scale", "0–1 → Invers ke 1–5", C["a4"]),
    ]
    for col, (lbl, val, clr) in zip([c1, c2, c3, c4], info):
        col.markdown(
            f"<div style='background:#0d1220;border:1px solid #1e2a3f;border-radius:12px;"
            f"padding:16px 18px;text-align:center'>"
            f"<div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace;"
            f"text-transform:uppercase;margin-bottom:6px'>{lbl}</div>"
            f"<div style='font-size:15px;font-weight:600;color:{clr}'>{val}</div></div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PRODUK & ULASAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Produk & Ulasan":

    prod_stats = (
        df.groupby("product_id")
        .agg(reviews=("rating", "count"), avg_rating=("rating", "mean"), total_score=("rating", "sum"))
        .reset_index()
    )
    C_bay = prod_stats["reviews"].mean()
    M_bay = prod_stats["avg_rating"].mean()
    prod_stats["bayesian"] = (
        (prod_stats["reviews"] * prod_stats["avg_rating"] + C_bay * M_bay)
        / (prod_stats["reviews"] + C_bay)
    )
    prod_stats_filt = prod_stats[prod_stats["reviews"] >= min_reviews]

    # ── KPI ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Produk Total",          f"{prod_stats['product_id'].nunique():,}")
    c2.metric("📦 Produk ≥ Filter",        f"{prod_stats_filt['product_id'].nunique():,}")
    c3.metric("⭐ Avg Rating Produk",      f"{prod_stats_filt['avg_rating'].mean():.3f}")
    c4.metric("🏅 Max Ulasan (1 Produk)", f"{int(prod_stats['reviews'].max()):,}")

    # ── Top Products: 2 tabs ─────────────────────────────────────────────────
    section("Ranking Produk", "🏆")
    tab1, tab2, tab3 = st.tabs(["📊 Terlaris (Volume)", "⭐ Tertinggi (Bayesian)", "🔴 Terendah"])

    with tab1:
        top_vol = prod_stats_filt.nlargest(top_n, "reviews")

        fig1 = go.Figure(go.Bar(
            x=top_vol["reviews"],
            y=top_vol["product_id"],
            orientation="h",
        ))

        fig1.update_layout(
            **layout(
                title=f"Top {top_n} Produk berdasarkan Volume Ulasan",
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True,
            key="top_review_chart"
        )

    with tab2:
        top_bay = prod_stats_filt.nlargest(top_n, "bayesian")

        fig2 = go.Figure(go.Bar(
            x=top_bay["bayesian"].round(3),
            y=top_bay["product_id"],
            orientation="h",
        ))

        fig2.update_layout(
            **layout(
                title=f"Top {top_n} Produk berdasarkan Bayesian Avg Rating",
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            key="bayesian_chart"
        )
    with tab3:
        bot_rat = prod_stats_filt.nsmallest(top_n, "avg_rating")
        fig = go.Figure(go.Bar(
            x=bot_rat["avg_rating"].round(3), y=bot_rat["product_id"],
            orientation="h",
            marker=dict(color=C["danger"], opacity=0.8, line=dict(width=0)),
            text=bot_rat["avg_rating"].map(lambda x: f"★ {x:.2f}"),
            textposition="outside", textfont=dict(color=C["muted"], size=11),
            hovertemplate="<b>%{y}</b><br>Avg: %{x:.2f}★<extra></extra>",
        ))
    fig.update_layout(
    **layout(
        title=f"Bottom {top_n} Produk berdasarkan Rating",
        height=max(380, top_n * 30),
        xaxis=dict(
            range=[0, 4],
            gridcolor=C["grid"]
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(0,0,0,0)"
        ),
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: Volume vs Rating ─────────────────────────────────────────
    section("Distribusi Produk", "🔬")
    col_sc, col_hist = st.columns([3, 2])

    with col_sc:
        scatter_df = prod_stats[prod_stats["reviews"] >= 3].copy()
        scatter_df["size"] = np.sqrt(scatter_df["reviews"]).clip(3, 30)
        fig = go.Figure(go.Scatter(
            x=scatter_df["reviews"],
            y=scatter_df["avg_rating"].round(3),
            mode="markers",
            marker=dict(
                size=scatter_df["size"],
                color=scatter_df["avg_rating"],
                colorscale=[[0, C["danger"]], [0.5, C["a6"]], [1, C["a1"]]],
                opacity=0.6,
                colorbar=dict(title="Avg ★", len=0.7, thickness=13,
                              tickfont=dict(color=C["muted"])),
                cmin=1, cmax=5,
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=scatter_df["product_id"],
            hovertemplate="<b>%{text}</b><br>Ulasan: %{x:,}<br>Avg: %{y:.2f}★<extra></extra>",
        ))
        fig.update_layout(
            **layout(
                title="Produk: Volume vs Avg Rating (ukuran = volume)",
                margin=dict(l=60, r=20, t=50, b=40),
                xaxis=dict(
                title="Jumlah Ulasan",
                type="log",
                gridcolor=C["grid"]
                ),
                yaxis=dict(
                title="Avg Rating",
                range=[0.5, 5.5],
                gridcolor=C["grid"]
                ),
            )
        )
    st.plotly_chart(fig, use_container_width=True)

    with col_hist:
        fig = px.histogram(
            prod_stats, x="reviews",
            nbins=50,
            color_discrete_sequence=[C["a2"]],
            labels={"reviews": "Jumlah Ulasan", "count": "Produk"},
            title="Distribusi Jumlah Ulasan per Produk",
        )
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.update_layout(**layout())
        st.plotly_chart(fig, use_container_width=True)

    # ── Data Table ────────────────────────────────────────────────────────
    section("Tabel Produk", "📋")
    show_prod = prod_stats_filt.nlargest(50, "reviews")[
        ["product_id", "reviews", "avg_rating", "bayesian"]
    ].copy()
    show_prod.columns = ["Product ID", "Jumlah Ulasan", "Avg Rating", "Bayesian Avg"]
    show_prod["Avg Rating"]    = show_prod["Avg Rating"].round(3)
    show_prod["Bayesian Avg"]  = show_prod["Bayesian Avg"].round(3)
    show_prod = show_prod.reset_index(drop=True)
    show_prod.index = show_prod.index + 1
    st.dataframe(show_prod, use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALISIS PENGGUNA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analisis Pengguna":

    user_stats = (
        df.groupby("user_id")
        .agg(reviews=("rating","count"), avg=("rating","mean"),
             min_r=("rating","min"), max_r=("rating","max"))
        .reset_index()
    )
    bins   = [0, 1, 5, 15, 30, 50, 10000]
    labels = ["Casual (1)", "Light (2–5)", "Moderate (6–15)",
              "Active (16–30)", "Heavy (31–50)", "Power (50+)"]
    user_stats["segment"] = pd.cut(user_stats["reviews"], bins=bins, labels=labels)

    # ── KPI ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 Total User",          f"{user_stats['user_id'].nunique():,}")
    c2.metric("📝 Avg Ulasan/User",     f"{user_stats['reviews'].mean():.1f}")
    c3.metric("🏅 Max Ulasan (1 User)", f"{int(user_stats['reviews'].max()):,}")
    c4.metric("⭐ Avg Rating User",     f"{user_stats['avg'].mean():.3f}")

    # ── Segmentasi ──────────────────────────────────────────────────────────
    section("Segmentasi Pengguna", "👥")
    col1, col2, col3 = st.columns(3)

    seg_colors = [C["danger"], C["a4"], C["a6"], C["a3"], C["a1"], C["a2"]]

    with col1:
        seg_count = user_stats["segment"].value_counts().reset_index()
        seg_count.columns = ["segment", "count"]
        seg_count = seg_count.sort_values("segment")
        fig = go.Figure(go.Pie(
            labels=seg_count["segment"], values=seg_count["count"],
            hole=0.55,
            marker=dict(colors=seg_colors, line=dict(color="#070b14", width=3)),
            textinfo="percent", textfont=dict(size=12),
            direction="clockwise",
            hovertemplate="<b>%{label}</b><br>%{value:,} user (%{percent})<extra></extra>",
        ))
        fig.update_layout(title="Distribusi Segmen", showlegend=False, **layout())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_avg = (
            user_stats.groupby("segment", observed=True)["avg"]
            .mean().reset_index().sort_values("segment")
        )
        fig = go.Figure(go.Bar(
            x=seg_avg["segment"], y=seg_avg["avg"].round(3),
            marker_color=seg_colors,
            text=seg_avg["avg"].round(3),
            textposition="outside", textfont=dict(color=C["muted"]),
            hovertemplate="<b>%{x}</b><br>Avg rating: %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(
            **layout(
            title="Avg Rating per Segmen",
            yaxis=dict(
            range=[3.5, 5.0],
            gridcolor=C["grid"]
            ),
            xaxis=dict(
            tickangle=20,
            gridcolor="rgba(0,0,0,0)"
            ),
        )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        seg_vol = (
            user_stats.groupby("segment", observed=True)["reviews"]
            .sum().reset_index().sort_values("segment")
        )
        fig = go.Figure(go.Bar(
            x=seg_vol["segment"], y=seg_vol["reviews"],
            marker_color=seg_colors, opacity=0.85,
            text=seg_vol["reviews"].map(lambda x: f"{x:,}"),
            textposition="outside", textfont=dict(color=C["muted"]),
        ))
        fig.update_layout(
            **layout(
            title="Total Kontribusi Ulasan per Segmen",
            yaxis=dict(
            gridcolor=C["grid"]
            ),
            xaxis=dict(
            tickangle=20,
            gridcolor="rgba(0,0,0,0)"
            ),
        )
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Retensi User ─────────────────────────────────────────────────────────
    section("Retensi & Kohort Pengguna", "🔄")
    col_ret, col_coh = st.columns(2)

    with col_ret:
        user_years = df_raw.groupby("user_id")["year"].nunique().value_counts().sort_index()
        total_users = user_years.sum()
        fig = go.Figure(go.Bar(
            x=user_years.index.astype(str),
            y=user_years.values,
            marker=dict(
                color=user_years.values,
                colorscale=[[0, "#1e2a3f"], [1, C["a3"]]],
                line=dict(width=0),
            ),
            text=(user_years.values / total_users * 100).round(1).astype(str) + "%",
            textposition="outside", textfont=dict(color=C["muted"]),
            hovertemplate="<b>%{x} tahun aktif</b><br>%{y:,} user<extra></extra>",
        ))
        fig.update_layout(
            **layout(
            title="Distribusi Tahun Aktif per User",
            yaxis=dict(
            gridcolor=C["grid"]
            ),
            xaxis=dict(
            tickangle=20,
            gridcolor="rgba(0,0,0,0)"
            ),
        )
        )
    st.plotly_chart(fig, use_container_width=True)

    with col_coh:
        cohort = df_raw.groupby("user_id")["timestamp"].min().dt.year.value_counts().sort_index()
        fig = go.Figure(go.Scatter(
            x=cohort.index, y=cohort.values,
            mode="lines+markers",
            fill="tozeroy",
            fillcolor=f"rgba(129,140,248,.15)",
            line=dict(color=C["a2"], width=2.5),
            marker=dict(size=8, color=C["a2"], line=dict(color=C["bg"], width=2)),
            hovertemplate="<b>%{x}</b><br>%{y:,} user baru<extra></extra>",
        ))
        fig.update_layout(
            **layout(
            title="Akuisisi User Baru per Tahun (Kohort)",
            yaxis=dict(
            gridcolor=C["grid"]
            ),
            xaxis=dict(
            tickangle=20,
            gridcolor="rgba(0,0,0,0)"
            ),
        )
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Top Reviewer ──────────────────────────────────────────────────────
    section(f"Top {top_n} Reviewer Paling Aktif", "🏆")
    col_tr, col_rb = st.columns([3, 2])

    with col_tr:
        top_users = user_stats.nlargest(top_n, "reviews")
        fig = go.Figure(go.Bar(
            x=top_users["reviews"], y=top_users["user_id"],
            orientation="h",
            marker=dict(
                color=top_users["avg"],
                colorscale=[[0, C["danger"]], [0.5, C["a6"]], [1, C["a1"]]],
                colorbar=dict(title="Avg ★", len=0.7, thickness=13,
                              tickfont=dict(color=C["muted"])),
                cmin=1, cmax=5, line=dict(width=0),
            ),
            text=top_users["reviews"].map(lambda x: f"{x:,}"),
            textposition="outside", textfont=dict(color=C["muted"], size=11),
            hovertemplate="<b>%{y}</b><br>%{x:,} ulasan<extra></extra>",
        ))
        fig.update_layout(
            **layout(
            title=f"Top {top_n} Reviewer",
            height=max(360, top_n * 30),
            xaxis=dict(
            gridcolor=C["grid"]
            ),
            yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(0,0,0,0)"
            ),
        )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_rb:
        # Box plot distribusi ulasan
        fig = go.Figure(go.Box(
            x=user_stats["reviews"],
            name="Ulasan/User",
            marker_color=C["a2"],
            line_color=C["a1"],
            boxmean=True,
            hovertemplate="<b>%{x}</b> ulasan<extra></extra>",
        ))
        fig.update_layout(
            **layout(
            title="Distribusi Ulasan per User",
            xaxis=dict(
            title="Jumlah Ulasan",
            gridcolor=C["grid"]
            ),
            showlegend=False,
        )
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Percentile stats
        pcts = [25, 50, 75, 90, 95, 99]
        vals = np.percentile(user_stats["reviews"], pcts)
        st.markdown(
            "<div style='background:#0d1220;border:1px solid #1e2a3f;border-radius:10px;"
            "padding:14px 18px;margin-top:8px'>"
            "<div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace;"
            "text-transform:uppercase;margin-bottom:10px'>Persentil Ulasan/User</div>" +
            "".join(
                f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                f"border-bottom:1px solid #1e2a3f'>"
                f"<span style='color:#5a6a8a;font-size:12px'>P{p}</span>"
                f"<span style='color:#e2e8f7;font-weight:600;font-family:DM Mono,monospace'>{int(v)}</span></div>"
                for p, v in zip(pcts, vals)
            ) +
            "</div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: TREN & WAKTU
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Tren & Waktu":

    # ── Monthly trend ────────────────────────────────────────────────────────
    section("Tren Bulanan", "📅")
    monthly = (
        df.groupby("year_month")
        .agg(count=("rating","count"), avg=("rating","mean"))
        .reset_index().sort_values("year_month")
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["count"],
        name="Volume", fill="tozeroy",
        fillcolor=f"rgba(56,189,248,.12)",
        line=dict(color=C["a1"], width=1.5),
        mode="lines",
        hovertemplate="<b>%{x}</b><br>%{y:,} ulasan<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["avg"].round(3),
        name="Avg Rating",
        line=dict(color=C["a4"], width=2, dash="dot"),
        mode="lines",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.3f}★<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(
        **layout(
        title="Tren Bulanan: Volume Ulasan & Avg Rating",
        legend=dict(
            orientation="h",
            y=1.1
        ),
    )
    )
    fig.update_yaxes(title_text="Ulasan", secondary_y=False, gridcolor=C["grid"])
    fig.update_yaxes(title_text="Avg Rating", secondary_y=True,
                     range=[3.5, 5.0], gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(nticks=24, tickangle=45, gridcolor=C["grid"])
    st.plotly_chart(fig, use_container_width=True)

    # ── Seasonality ──────────────────────────────────────────────────────────
    section("Pola Musiman", "🌊")
    col_mon, col_day = st.columns(2)

    with col_mon:
        mon_agg = (
            df.groupby("month")
            .agg(count=("rating","count"), avg=("rating","mean"))
            .reset_index()
        )
        mon_labels = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
        mon_agg["label"] = mon_agg["month"].apply(lambda x: mon_labels[x-1])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=mon_agg["label"], y=mon_agg["count"],
            name="Volume", marker_color=C["a1"], opacity=0.75,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=mon_agg["label"], y=mon_agg["avg"].round(3),
            name="Avg Rating", mode="lines+markers",
            line=dict(color=C["a4"], width=2.5),
            marker=dict(size=8, color=C["a4"]),
        ), secondary_y=True)
        fig.update_layout(title="Pola Bulanan (Seasonality)", **layout())
        fig.update_yaxes(secondary_y=False, gridcolor=C["grid"])
        fig.update_yaxes(secondary_y=True, range=[4.1, 4.4], gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_day:
        day_agg = (
            df.groupby("dayofweek")
            .agg(count=("rating","count"), avg=("rating","mean"))
            .reset_index()
        )
        day_labels = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
        day_agg["label"] = day_agg["dayofweek"].apply(lambda x: day_labels[x])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=day_agg["count"], theta=day_agg["label"],
            fill="toself", fillcolor=f"rgba(56,189,248,.15)",
            line=dict(color=C["a1"], width=2),
            name="Volume Ulasan",
        ))
        fig.update_layout(
            title="Volume Ulasan per Hari (Radar)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor=C["grid"],
                                tickcolor=C["muted"], tickfont=dict(size=10)),
                angularaxis=dict(gridcolor=C["grid"]),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"], family="DM Sans"),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap Rating per Year×Month ─────────────────────────────────────
    section("Heatmap Avg Rating (Tahun × Bulan)", "🌡️")

    heat_r = (
        df.groupby(["year", "month"])["rating"]
        .mean()
        .reset_index()
        .pivot(index="year", columns="month", values="rating")
        .fillna(3.5)
    )

    month_labels = [
        "Jan","Feb","Mar","Apr","Mei","Jun",
        "Jul","Agu","Sep","Okt","Nov","Des"
    ]

    cols_p = [c for c in range(1, 13) if c in heat_r.columns]

    fig = go.Figure(
        go.Heatmap(
            z=heat_r[cols_p].values.round(3),

            x=[month_labels[c - 1] for c in cols_p],

            y=heat_r.index.astype(str),

            xgap=2,
            ygap=2,

            colorscale=[
                [0, C["danger"]],
                [0.25, C["a4"]],
                [0.5, C["a6"]],
                [0.75, C["a3"]],
                [1.0, C["a1"]],
            ],

            zmin=3.5,
            zmax=5.0,

            text=heat_r[cols_p].values.round(2),

            texttemplate="%{text}",

            textfont=dict(
                size=10
            ),

            hovertemplate=(
                "<b>%{y} %{x}</b><br>"
                "Avg rating: %{z:.3f}<extra></extra>"
            ),

            colorbar=dict(
                title="Avg ★",
                tickfont=dict(
                    color=C["muted"]
                )
            ),
        )
    )

    fig.update_layout(
        **layout(
            title="Avg Rating per Tahun & Bulan",

            xaxis=dict(
                side="bottom",
                gridcolor="rgba(0,0,0,0)"
            ),

            yaxis=dict(
                gridcolor="rgba(0,0,0,0)"
            ),
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    # ── Quarter analysis ──────────────────────────────────────────────────
 section("Analisis per Kuartal", "📊")
 col_qv, col_qr = st.columns(2)

# 1. Siapkan data secara kronologis (diurutkan berdasarkan tahun dan kuartal)
 qtr = (
    df.groupby(["year", "quarter"])
    .agg(count=("rating", "count"), avg=("rating", "mean"))
    .reset_index()
    .sort_values(["year", "quarter"])  # Pastikan urut kronologis
 )
 qtr["label"] = qtr["year"].astype(str) + " Q" + qtr["quarter"].astype(str)

# --- KOLOM KIRI: TOTAL VOLUME (COUNT) ---
 with col_qv:
    fig_v = go.Figure()
    # Buat satu garis utuh agar tidak terputus
    fig_v.add_trace(go.Scatter(
        x=qtr["label"], 
        y=qtr["count"],
        mode="lines+markers",
        line=dict(color=C["a1"], width=3),  # Gunakan 1 warna utama untuk tren
        marker=dict(size=8, color=C["a2"]),
        name="Total Rating"
    ))
    
    fig_v.update_layout(
        **layout(
            title="Total Volume Rating per Kuartal",  # Judul diperbaiki
            xaxis=dict(tickangle=45, gridcolor=C["grid"]),
            yaxis=dict(gridcolor=C["grid"]),
        )
    )
    st.plotly_chart(fig_v, use_container_width=True)

# --- KOLOM KANAN: RATA-RATA RATING (AVG) ---
 with col_qr:
    fig_r = go.Figure()
    # Buat satu garis utuh agar tidak terputus
    fig_r.add_trace(go.Scatter(
        x=qtr["label"], 
        y=qtr["avg"].round(3),
        mode="lines+markers",
        line=dict(color=C["a3"], width=3),  # Gunakan warna kontras lain
        marker=dict(size=8, color=C["a4"]),
        name="Avg Rating"
    ))
    
    fig_r.update_layout(
        **layout(
            title="Avg Rating per Kuartal",
            xaxis=dict(tickangle=45, gridcolor=C["grid"]),
            # Range yaxis disesuaikan otomatis atau berikan ruang logis
            yaxis=dict(range=[qtr["avg"].min() - 0.2, qtr["avg"].max() + 0.2], gridcolor=C["grid"]),
        )
    )
    st.plotly_chart(fig_r, use_container_width=True)
    
# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL NCF
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model NCF":

    # ── Arsitektur card ───────────────────────────────────────────────────────
    section("Arsitektur Model", "🧠")
    st.markdown("""
    <div style='background:#0d1220;border:1px solid #1e2a3f;border-radius:14px;padding:24px 28px'>
    <div style='font-family:Syne,sans-serif;font-size:18px;font-weight:700;color:#38bdf8;margin-bottom:16px'>
    Neural Collaborative Filtering (NCF)</div>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px'>
        <div style='background:#111827;border:1px solid #1e2a3f;border-radius:10px;padding:14px'>
            <div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace;margin-bottom:6px'>INPUT</div>
            <div style='font-size:14px;color:#e2e8f7'>User ID <span style='color:#38bdf8'>→</span> Embedding (32)</div>
            <div style='font-size:14px;color:#e2e8f7;margin-top:4px'>Product ID <span style='color:#38bdf8'>→</span> Embedding (32)</div>
        </div>
        <div style='background:#111827;border:1px solid #1e2a3f;border-radius:10px;padding:14px'>
            <div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace;margin-bottom:6px'>HIDDEN LAYERS</div>
            <div style='font-size:14px;color:#818cf8'>Jalur 1: Concatenate → Dense(64→32→16)</div>
            <div style='font-size:14px;color:#34d399;margin-top:4px'>Jalur 2: Element-wise × (Custom Layer)</div>
        </div>
        <div style='background:#111827;border:1px solid #1e2a3f;border-radius:10px;padding:14px'>
            <div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace;margin-bottom:6px'>OUTPUT</div>
            <div style='font-size:14px;color:#e2e8f7'>Sigmoid → [0,1]</div>
            <div style='font-size:14px;color:#e2e8f7;margin-top:4px'>Invers Scaler → [1,5] ★</div>
        </div>
    </div>
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px'>
        <div style='text-align:center;padding:10px;background:#111827;border-radius:8px'>
            <div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace'>OPTIMIZER</div>
            <div style='color:#38bdf8;font-weight:600'>Adam</div>
        </div>
        <div style='text-align:center;padding:10px;background:#111827;border-radius:8px'>
            <div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace'>LOSS</div>
            <div style='color:#818cf8;font-weight:600'>Custom RMSE</div>
        </div>
        <div style='text-align:center;padding:10px;background:#111827;border-radius:8px'>
            <div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace'>EMBEDDING DIM</div>
            <div style='color:#34d399;font-weight:600'>32</div>
        </div>
        <div style='text-align:center;padding:10px;background:#111827;border-radius:8px'>
            <div style='font-size:11px;color:#5a6a8a;font-family:DM Mono,monospace'>DROPOUT</div>
            <div style='color:#fb923c;font-weight:600'>0.5</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data split info ────────────────────────────────────────────────────────
    section("Informasi Data Training", "📊")
    n_total = 155938
    n_train = int(n_total * 0.8)
    n_test  = n_total - n_train

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sampel (Cleaned)", f"{n_total:,}")
    c2.metric("Training Set (80%)",      f"{n_train:,}")
    c3.metric("Test Set (20%)",          f"{n_test:,}")
    c4.metric("Sparsity Matrix",         "99.98%")

    # ── Cleaned vs Raw comparison ─────────────────────────────────────────────
    section("Distribusi: Data Mentah vs Cleaned (Scaled)", "⚖️")
    col_r, col_c = st.columns(2)

    with col_r:
        raw_dist = df_raw["rating"].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=[f"{'★'*int(r)} ({int(r)})" for r in raw_dist.index],
            y=raw_dist.values,
            marker=dict(
                color=list(RATING_COLORS.values()),
                line=dict(width=0),
            ),
            text=[f"{v:,}" for v in raw_dist.values],
            textposition="outside",
            textfont=dict(color=C["muted"]),
            hovertemplate="<b>%{x}</b><br>%{y:,} ulasan<extra></extra>",
        ))
        fig.update_layout(
            **layout(
            title="Distribusi Rating — Data Mentah (1–5)",

            yaxis=dict(
                gridcolor=C["grid"]
            ),

            xaxis=dict(
                gridcolor="rgba(0,0,0,0)"
            ),
        )
    )
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        # Cleaned data scaled 0-1
        dfc = pd.read_csv(
            next(p for p in [
                Path("data/cleaned_sample_data_scaled.csv"),
                Path("../data/cleaned_sample_data_scaled.csv"),
            ] if p.exists()),
        )
        fig = px.histogram(
            dfc, x="rating", nbins=40,
            color_discrete_sequence=[C["a2"]],
            labels={"rating": "Rating (Scaled 0–1)", "count": "Frekuensi"},
            title="Distribusi Rating — Cleaned & Scaled (0–1)",
        )
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.update_layout(**layout())
        st.plotly_chart(fig, use_container_width=True)

    # ── Simulasi prediksi ─────────────────────────────────────────────────────
    section("Simulasi Rekomendasi via API", "🔌")

    st.markdown(
        "<div style='background:#0d1220;border:1px solid #1e2a3f;border-left:3px solid #38bdf8;"
        "border-radius:0 12px 12px 0;padding:14px 18px;font-size:13px;color:#8898b8;"
        "font-family:DM Mono,monospace;margin-bottom:20px'>"
        "Pastikan API sudah berjalan pada <b style='color:#38bdf8'>http://localhost:8000</b> "
        "sebelum menggunakan fitur ini.<br>"
        "Jalankan: <b style='color:#34d399'>uvicorn app:app --reload</b> dari folder <code>api/</code>"
        "</div>",
        unsafe_allow_html=True
    )

    col_form, col_res = st.columns([1, 2])

    # =========================
    # FORM INPUT
    # =========================
    with col_form:

        sample_users = (
            df_raw["user_id"]
            .value_counts()
            .head(20)
            .index
            .tolist()
        )

        sel_user = st.selectbox(
            "Pilih User ID",
            sample_users,
            key="recommend_user_select"
        )

        top_k_sim = st.slider(
            "Top K Rekomendasi",
            3,
            20,
            10,
            key="top_k_slider"
        )

        min_r_sim = st.slider(
            "Min Rating Filter",
            1.0,
            5.0,
            3.0,
            0.5,
            key="min_rating_slider"
        )

        if st.button(
            "🚀 Minta Rekomendasi",
            use_container_width=True,
            key="recommend_button"
        ):

            try:
                import requests

                resp = requests.post(
                    "http://localhost:8000/recommend",
                    json={
                        "user_id": sel_user,
                        "top_k": top_k_sim,
                        "min_rating": min_r_sim
                    },
                    timeout=30,
                )

                if resp.status_code == 200:
                    st.session_state["api_result"] = resp.json()

                else:
                    st.session_state["api_result"] = {
                        "error": resp.text
                    }

            except Exception as e:
                st.session_state["api_result"] = {
                    "error": str(e)
                }

    # =========================
    # HASIL REKOMENDASI
    # =========================
    with col_res:

        if "api_result" not in st.session_state:

            st.info(
                "Silakan pilih user lalu klik tombol rekomendasi."
            )

        else:

            result = st.session_state["api_result"]

            if "error" in result:

                st.error(
                    f"❌ API Error: {result['error']}"
                )

            else:

                recs = result.get(
                    "recommendations",
                    []
                )

                if recs:

                    rec_df = pd.DataFrame(recs)

                    fig = go.Figure(
                        go.Bar(
                            x=rec_df["predicted_rating"],
                            y=rec_df["product_id"],
                            orientation="h",
                            marker=dict(
                                color=rec_df["predicted_rating"],
                                colorscale=[
                                    [0, C["danger"]],
                                    [0.5, C["a6"]],
                                    [1, C["a1"]]
                                ],
                                cmin=1,
                                cmax=5,
                                line=dict(width=0),
                            ),
                            text=rec_df["predicted_rating"].map(
                                lambda x: f"★ {x:.3f}"
                            ),
                            textposition="outside",
                            textfont=dict(color=C["muted"]),
                        )
                    )

                    fig.update_layout(
                        **layout(
                            title=f"Rekomendasi untuk {sel_user}",
                            height=max(300, len(recs) * 36),

                            xaxis=dict(
                            range=[1, 5.5],
                            gridcolor=C["grid"]
                            ),

                            yaxis=dict(
                            autorange="reversed",
                            gridcolor="rgba(0,0,0,0)"
                            ),
                        )
                    )

            st.plotly_chart(
            fig,
            use_container_width=True,
            key="rekomendasi_chart"
        )

    # ── Data Mentah ────────────────────────────────────────────────────────────
    section("Preview Data Cleaned", "📋")

    try:

        dfc_preview = dfc.head(100).copy()

        dfc_preview["rating_bintang"] = (
            dfc_preview["rating"] * 4 + 1
        ).round(2)

        with st.expander(
            "Lihat 100 baris pertama data cleaned_sample_data_scaled.csv",
            expanded=False
        ):

            st.dataframe(
                dfc_preview,
                use_container_width=True,
                height=300
            )

    except Exception:

        st.info(
            "File `cleaned_sample_data_scaled.csv` tidak ditemukan."
        )
# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#2a3a55;font-size:12px;"
    "font-family:DM Mono,monospace;padding:8px 0'>"
    "NCF Analytics Dashboard &nbsp;·&nbsp; Capstone CC26-PRU466 "
    "&nbsp;·&nbsp; Neural Collaborative Filtering E-Commerce"
    "</div>",
    unsafe_allow_html=True
)
