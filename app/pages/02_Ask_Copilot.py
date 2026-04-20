
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os, sys

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COPILOT_DIR = os.path.join(BASE_DIR, "copilot")
if COPILOT_DIR not in sys.path:
    sys.path.append(COPILOT_DIR)
from query_router import answer_direct, answer_question

st.set_page_config(page_title="Ask Copilot", page_icon="🤖",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background:#FFFFFF; }
#MainMenu, footer, header { visibility:hidden; }
.main .block-container { padding:1.4rem 2rem 3rem; max-width:1300px; }

@keyframes fadeUp  { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes orbs    { 0%{transform:scale(1) rotate(0deg)} 100%{transform:scale(1.1) rotate(4deg)} }
@keyframes gradText{ 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0A0F1E 0%, #0F172A 55%, #1A0A2E 100%);
    border-radius: 20px; padding: 1.8rem 2.4rem;
    color: #F8FAFC; margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
    animation: fadeUp 0.5s ease;
}
.hero-orbs {
    position:absolute; inset:0; pointer-events:none;
    background:
        radial-gradient(ellipse 50% 65% at 5% 45%,  rgba(37,99,235,0.28) 0%, transparent 60%),
        radial-gradient(ellipse 42% 50% at 88% 15%, rgba(192,98,74,0.22) 0%, transparent 55%),
        radial-gradient(ellipse 35% 45% at 55% 90%, rgba(124,58,237,0.18) 0%, transparent 50%);
    animation: orbs 14s ease-in-out infinite alternate;
}
.hero-grid {
    position:absolute; inset:0; pointer-events:none;
    background-image: linear-gradient(rgba(255,255,255,0.03) 1px,transparent 1px),
                      linear-gradient(90deg,rgba(255,255,255,0.03) 1px,transparent 1px);
    background-size: 44px 44px;
}
.hero-content { position:relative; z-index:1; }
.hero-eyebrow { font-size:0.7rem; font-weight:700; letter-spacing:0.16em;
                text-transform:uppercase; color:rgba(248,250,252,0.45); margin-bottom:0.4rem; }
.hero h1 {
    font-family:'DM Serif Display',serif; font-size:2.2rem; font-weight:400; margin:0 0 0.4rem;
    background:linear-gradient(120deg,#FFFFFF 0%,#C0624A 50%,#A78BFA 100%);
    background-size:200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation: gradText 6s linear infinite;
}
.hero p { font-size:0.88rem; color:rgba(248,250,252,0.6); margin:0; line-height:1.6; }

/* ── Questions panel ── */
.q-panel-title {
    font-size: 0.85rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: #334155;
    padding: 0 0.4rem 0.6rem;
}

/* Expander header */
details > summary { list-style: none; }
details > summary p,
details > summary span,
details summary div,
details summary div p,
details summary div span {
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
details > summary svg { color: #0F172A !important; }

/* Buttons inside expanders */
details [data-testid="stButton"] button,
details [data-testid="stBaseButton-secondary"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}
details [data-testid="stButton"] button p,
details [data-testid="stBaseButton-secondary"] p {
    color: #334155 !important;
    font-weight: 500 !important;
}
details [data-testid="stButton"] button:hover {
    background: #EFF6FF !important;
    border-color: #2563EB !important;
}
details [data-testid="stButton"] button:hover p {
    color: #1D4ED8 !important;
}

/* ── Result card ── */
.result-card {
    background:#F8FAFC; border-radius:16px;
    padding:0.8rem 1.2rem; margin-bottom:0.8rem;
    border-left:4px solid #2563EB;
    animation: fadeUp 0.3s ease;
}
.result-title { font-size:1rem; font-weight:700; color:#0F172A; }

/* ── Placeholder ── */
.placeholder {
    background:#F8FAFC; border:2px dashed #E2E8F0;
    border-radius:20px; padding:3rem 2rem;
    text-align:center; color:#94A3B8;
    font-size:1rem;
}

/* ── Custom input ── */
.stTextInput input {
    border:2px solid #E2E8F0 !important; border-radius:12px !important;
    padding:0.7rem 1rem !important; font-size:0.95rem !important;
    font-family:'DM Sans',sans-serif !important;
    transition:border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput input:focus {
    border-color:#2563EB !important;
    box-shadow:0 0 0 4px rgba(37,99,235,0.1) !important;
    outline:none !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background:linear-gradient(135deg,#C0624A 0%,#E8724A 100%);
    border:none; border-radius:10px; color:#FFFFFF;
    font-weight:700; font-family:'DM Sans',sans-serif;
    transition:all 0.2s ease;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#A8533E 0%,#C0624A 100%);
    transform:translateY(-1px);
}

/* ── Question buttons in dark panel ── */
section[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
if "output"       not in st.session_state: st.session_state.output       = None
if "active_label" not in st.session_state: st.session_state.active_label = None

def run_direct(label, query_name, params={}):
    st.session_state.active_label = label
    st.session_state.output       = answer_direct(query_name, params)

def run_text(q):
    st.session_state.active_label = q
    st.session_state.output       = answer_question(q)

# ── Questions data ─────────────────────────────────────────────────────────────
CATEGORIES = {
    "📊 Risk Overview": [
        ("Top 10 high-risk accounts",        "top_risk_accounts",         {"limit":10}),
        ("Top 25 high-risk accounts",        "top_risk_accounts",         {"limit":25}),
        ("Top 50 high-risk accounts",        "top_risk_accounts",         {"limit":50}),
        ("Risk distribution by bucket",      "risk_bucket_distribution",  {}),
        ("Risk distribution by band",        "risk_band_distribution",    {}),
        ("Total high-risk count",            "high_risk_count",           {}),
        ("Accounts in Top 1%",               "top_1_percent",             {}),
        ("Accounts in Top 5%",               "top_5_percent",             {}),
        ("Accounts in Top 10%",              "top_10_percent",            {}),
    ],
    "📋 By Plan": [
        ("Churn risk by plan",               "avg_risk_by_plan",          {}),
        ("High-risk count by plan",          "high_risk_by_plan",         {}),
        ("Enterprise high-risk accounts",    "enterprise_high_risk",      {}),
        ("Pro high-risk accounts",           "pro_high_risk",             {}),
        ("Basic high-risk accounts",         "basic_high_risk",           {}),
    ],
    "🌍 By Region": [
        ("Churn risk by region",             "avg_risk_by_region",        {}),
        ("High-risk count by region",        "high_risk_by_region",       {}),
        ("High-risk in NA",                  "high_risk_na",              {}),
        ("High-risk in EU",                  "high_risk_eu",              {}),
        ("High-risk in APAC",                "high_risk_apac",            {}),
    ],
    "📝 By Contract": [
        ("Risk by contract type",            "avg_risk_by_contract",      {}),
        ("High-risk Monthly accounts",       "monthly_high_risk",         {}),
        ("High-risk Annual accounts",        "annual_high_risk",          {}),
    ],
    "🔀 Segments": [
        ("Risk by plan × region",            "risk_by_plan_and_region",      {}),
        ("Risk by plan × contract",          "risk_by_plan_and_contract",    {}),
        ("Enterprise + Monthly high-risk",   "enterprise_monthly_high_risk", {}),
    ],
    "💰 Revenue at Risk": [
        ("Total revenue at risk",            "revenue_at_risk",           {}),
        ("Revenue at risk by plan",          "revenue_at_risk_by_plan",   {}),
        ("Revenue at risk by region",        "revenue_at_risk_by_region", {}),
    ],
    "📈 Weekly Trends": [
        ("High-risk count over time",        "high_risk_trend",           {}),
        ("Average risk over time",           "avg_risk_trend",            {}),
        ("Week-over-week summary",           "week_over_week_summary",    {}),
        ("Newly high-risk this week",        "new_high_risk_accounts",    {}),
        ("Accounts that recovered",          "recovered_accounts",        {}),
    ],
}

# ── Top nav ───────────────────────────────────────────────────────────────────
cb, _ = st.columns([1, 9])
with cb:
    if st.button("← Dashboard"):
        st.switch_page("Home.py")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-orbs"></div>
  <div class="hero-grid"></div>
  <div class="hero-content">
    <div class="hero-eyebrow">Churn Intelligence</div>
    <h1>Ask Copilot</h1>
    <p>Pick a question from the panel or type your own below.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Custom input ──────────────────────────────────────────────────────────────
ci, cb2 = st.columns([8, 1])
with ci:
    custom = st.text_input("", placeholder="e.g. high risk Enterprise accounts in EU with Monthly contracts",
                           label_visibility="collapsed")
with cb2:
    st.markdown("<div style='padding-top:0.28rem'>", unsafe_allow_html=True)
    if st.button("Run →", type="primary", use_container_width=True):
        if custom.strip(): run_text(custom.strip())
        else: st.warning("Please enter a question.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)

# ── Two-column layout: questions left, result right ───────────────────────────
q_col, r_col = st.columns([3, 7], gap="large")

# ── Left: question panel ──────────────────────────────────────────────────────
with q_col:
    st.markdown('<div class="q-panel-title">Questions</div>', unsafe_allow_html=True)
    for cat, qs in CATEGORIES.items():
        with st.expander(cat, expanded=False):
            for i, (lbl, qn, params) in enumerate(qs):
                if st.button(lbl, key=f"q_{cat}_{i}", use_container_width=True):
                    run_direct(lbl, qn, params)

# ── Chart renderer ────────────────────────────────────────────────────────────
CHART = dict(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
             font=dict(family="DM Sans", color="#0F172A"),
             margin=dict(l=20, r=50, t=50, b=30))

def render_chart(df: pd.DataFrame, query_name: str):
    if df is None or df.empty or len(df) < 2: return
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")
    num  = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not num: return

    # Account-list queries: horizontal bar of risk percentile by account
    if "account_id" in df.columns and "risk_percentile" in df.columns:
        plot_df = df.sort_values("risk_percentile", ascending=True).head(20)
        fig = go.Figure(go.Bar(
            x=plot_df["risk_percentile"],
            y=plot_df["account_id"].astype(str),
            orientation="h",
            text=plot_df["risk_percentile"].round(1),
            textposition="outside",
            textfont=dict(size=11, color="#0F172A"),
            marker=dict(color="#2563EB", line=dict(color="#FFFFFF", width=1)),
            hovertemplate="<b>%{y}</b><br>Risk %ile: %{x:.1f}<extra></extra>"
        ))
        fig.update_layout(
            height=max(300, len(plot_df) * 32),
            showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(size=11),
                       title="Risk Percentile", range=[0, 115]),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
            **CHART)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        return

    if "run_date" in df.columns:
        fig = go.Figure(go.Scatter(
            x=df["run_date"].astype(str), y=df[num[0]],
            mode="lines+markers+text",
            text=df[num[0]], textposition="top center",
            textfont=dict(size=11, color="#0F172A"),
            cliponaxis=False,
            line=dict(color="#2563EB", width=3),
            marker=dict(size=10, color="#2563EB", line=dict(color="white", width=2)),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"
        ))
        ymax = df[num[0]].max() * 1.25
        fig.update_layout(height=360, showlegend=False,
                          xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                          yaxis=dict(showgrid=False, tickfont=dict(size=11), range=[0, ymax]),
                          **CHART)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        return

    if query_name in ("risk_bucket_distribution", "risk_band_distribution") and text:
        cmap = {"High":"#1E3A8A","Medium":"#2563EB","Low":"#93C5FD",
                "Top 1%":"#1E3A8A","Top 5%":"#1D4ED8","Top 10%":"#2563EB",
                "Top 25%":"#60A5FA","Rest":"#BFDBFE"}
        colors = [cmap.get(l, "#94A3B8") for l in df[text[0]]]
        fig = go.Figure(go.Pie(
            labels=df[text[0]], values=df[num[0]], hole=0.58,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
            textinfo="label+percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value} accounts<extra></extra>"
        ))
        fig.update_layout(height=360, showlegend=False, paper_bgcolor="#FFFFFF",
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        return

    # Grouped bar: 2 categorical columns (e.g. plan × region, plan × contract)
    if len(text) >= 2 and num:
        group_col = text[0]   # x-axis  (e.g. plan_type)
        color_col = text[1]   # grouping (e.g. region / contract_type)
        BLUES = ["#1E3A8A", "#2563EB", "#60A5FA", "#93C5FD", "#BFDBFE"]
        cats  = list(df[color_col].unique())
        cmap  = {c: BLUES[i % len(BLUES)] for i, c in enumerate(cats)}
        ymax  = df[num[0]].max() * 1.35
        fig   = go.Figure()
        for cat in cats:
            sub = df[df[color_col] == cat]
            fig.add_trace(go.Bar(
                name=str(cat),
                x=sub[group_col], y=sub[num[0]],
                text=sub[num[0]], textposition="outside",
                textfont=dict(size=11, color="#0F172A"),
                marker=dict(color=cmap[cat], line=dict(color="#FFFFFF", width=1)),
                hovertemplate=f"<b>%{{x}}</b> · {cat}<br>%{{y}}<extra></extra>"
            ))
        fig.update_layout(
            height=360, barmode="group", bargap=0.25, bargroupgap=0.08,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=11)),
            xaxis=dict(showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(showgrid=False, tickfont=dict(size=11), range=[0, ymax]),
            **CHART)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        return

    if text and num:
        fig = go.Figure(go.Bar(
            x=df[text[0]], y=df[num[0]],
            text=df[num[0]], textposition="outside",
            textfont=dict(size=12, color="#0F172A"),
            marker=dict(color=["#2563EB"] * len(df), line=dict(color="#FFFFFF", width=1)),
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"
        ))
        fig.update_layout(height=360, bargap=0.38, showlegend=False,
                          xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                          yaxis=dict(showgrid=False, tickfont=dict(size=11),
                                     range=[0, df[num[0]].max() * 1.3]),
                          **CHART)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Right: result panel ───────────────────────────────────────────────────────
with r_col:
    if st.session_state.output:
        out = st.session_state.output
        if out["matched_query"] is None:
            st.error(out["message"])
        else:
            st.markdown(
                f'<div class="result-card"><div class="result-title">{st.session_state.active_label}</div></div>',
                unsafe_allow_html=True)
            df = out["result"]
            if df is not None and not df.empty:
                for col in df.select_dtypes(include="object").columns:
                    df[col] = df[col].fillna("Unknown")
            qn = out.get("matched_query", "")
            if df is not None and not df.empty:
                render_chart(df, qn)
                st.dataframe(df, use_container_width=True, hide_index=True,
                             column_config={
                                 "risk_percentile":       st.column_config.NumberColumn("Risk %ile",  format="%.1f"),
                                 "churn_probability":     st.column_config.NumberColumn("Churn Prob", format="%.4f"),
                                 "churn_risk_calibrated": st.column_config.NumberColumn("Churn Prob", format="%.4f"),
                             })
            else:
                st.info("No results returned for this query.")
    else:
        st.markdown("""
        <div class="placeholder">
            👈 &nbsp; Select a question from the panel to see the chart here
        </div>
        """, unsafe_allow_html=True)
