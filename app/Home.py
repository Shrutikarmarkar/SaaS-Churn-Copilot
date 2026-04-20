import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os, sys

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPILOT_DIR = os.path.join(BASE_DIR, "copilot")
if COPILOT_DIR not in sys.path:
    sys.path.append(COPILOT_DIR)
from query_db import run_query

st.set_page_config(page_title="Churn Copilot", page_icon="📉",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #FFFFFF;
    color: #0F172A;
}
#MainMenu, footer, header { visibility: hidden; }
.main .block-container { padding: 1.8rem 2.8rem 3rem; max-width: 1300px; }

/* ── Keyframes ── */
@keyframes fadeUp    { from { opacity:0; transform:translateY(22px); } to { opacity:1; transform:translateY(0); } }
@keyframes slideIn   { from { opacity:0; transform:translateX(-14px); } to { opacity:1; transform:translateX(0); } }
@keyframes orbs      { 0% { transform:scale(1) rotate(0deg); } 100% { transform:scale(1.1) rotate(4deg); } }
@keyframes breathe   { 0%,100% { opacity:1; transform:scale(1); box-shadow:0 0 0 0 rgba(74,222,128,0.5); }
                        50%  { opacity:0.7; transform:scale(0.85); box-shadow:0 0 0 6px rgba(74,222,128,0); } }
@keyframes popIn     { 0% { opacity:0; transform:scale(0.75); } 70% { transform:scale(1.06); } 100% { opacity:1; transform:scale(1); } }
@keyframes ripple    { 0% { transform:scale(0); opacity:0.55; } 100% { transform:scale(3); opacity:0; } }
@keyframes gradText  { 0%,100% { background-position:0% 50%; } 50% { background-position:100% 50%; } }
@keyframes revealUp  { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:translateY(0); } }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0A0F1E 0%, #0F172A 55%, #1A0A2E 100%);
    border-radius: 24px;
    padding: 3.2rem 3.5rem;
    color: #F8FAFC;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.5s ease;
}
.hero-orbs {
    position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 55% 70% at 5% 45%,  rgba(192,98,74,0.30) 0%, transparent 60%),
        radial-gradient(ellipse 45% 55% at 88% 15%, rgba(37,99,235,0.22) 0%, transparent 55%),
        radial-gradient(ellipse 38% 48% at 55% 92%, rgba(124,58,237,0.18) 0%, transparent 50%),
        radial-gradient(ellipse 28% 38% at 72% 58%, rgba(16,163,74,0.12) 0%, transparent 45%);
    animation: orbs 14s ease-in-out infinite alternate;
}
.hero-grid {
    position: absolute; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 44px 44px;
}
.hero-content { position: relative; z-index: 1; }
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: rgba(248,250,252,0.45);
    margin-bottom: 0.9rem;
    animation: slideIn 0.5s ease 0.1s both;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem; font-weight: 400; margin: 0 0 0.7rem;
    background: linear-gradient(120deg, #FFFFFF 0%, #C0624A 50%, #A78BFA 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: slideIn 0.5s ease 0.2s both, gradText 6s linear infinite;
}
.hero p {
    font-size: 1rem; color: rgba(248,250,252,0.6);
    margin: 0 0 1.8rem; max-width: 500px; line-height: 1.65;
    animation: slideIn 0.5s ease 0.3s both;
}
.hero-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; animation: slideIn 0.5s ease 0.4s both; }
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px; padding: 0.32rem 1rem;
    font-size: 0.78rem; color: rgba(248,250,252,0.78); font-weight: 500;
    backdrop-filter: blur(8px);
    transition: background 0.2s, border-color 0.2s, transform 0.2s;
}
.badge:hover { background: rgba(255,255,255,0.13); border-color: rgba(255,255,255,0.28); transform: translateY(-1px); }
.live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #4ADE80; display: inline-block;
    animation: breathe 2.2s ease infinite;
}

/* ── Metric Cards ── */
.cards-row { display: flex; gap: 1.2rem; margin: 2rem 0; }
.mcard {
    flex: 1; background: #FFFFFF;
    border-radius: 20px; padding: 1.7rem 1.8rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 8px rgba(0,0,0,0.03);
    position: relative; overflow: hidden;
    transition: transform 0.3s cubic-bezier(.4,0,.2,1),
                box-shadow 0.3s cubic-bezier(.4,0,.2,1),
                border-color 0.3s;
    cursor: default;
}
.mcard:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.10);
    border-color: transparent;
}
.mcard::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 20px 20px 0 0;
    transition: height 0.3s ease;
}
.mcard:hover::before { height: 4px; }
.mcard-rust::before  { background: linear-gradient(90deg, #C0624A, #F97316); }
.mcard-sand::before  { background: linear-gradient(90deg, #2563EB, #06B6D4); }
.mcard-mauve::before { background: linear-gradient(90deg, #7C3AED, #A855F7); }
.mcard-sage::before  { background: linear-gradient(90deg, #16A34A, #22C55E); }

.mcard:hover.mcard-rust  { box-shadow: 0 20px 40px rgba(192,98,74,0.14); }
.mcard:hover.mcard-sand  { box-shadow: 0 20px 40px rgba(37,99,235,0.12); }
.mcard:hover.mcard-mauve { box-shadow: 0 20px 40px rgba(124,58,237,0.12); }
.mcard:hover.mcard-sage  { box-shadow: 0 20px 40px rgba(22,163,74,0.12); }

.mcard-icon  { font-size: 1.6rem; margin-bottom: 0.6rem; display: block; }
.mcard-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
               letter-spacing: 0.1em; color: #94A3B8; margin-bottom: 0.4rem; }
.mcard-value {
    font-size: 3.1rem; font-weight: 800; color: #0F172A;
    line-height: 1; font-variant-numeric: tabular-nums;
    display: block;
}
.mcard-delta-up   { font-size: 0.92rem; color: #DC2626; margin-top: 0.5rem; font-weight: 600; }
.mcard-delta-down { font-size: 0.92rem; color: #16A34A; margin-top: 0.5rem; font-weight: 600; }
.mcard-delta-flat { font-size: 0.92rem; color: #94A3B8;  margin-top: 0.5rem; }

/* ── Chart Cards ── */
.cccard {
    background: #FFFFFF; border-radius: 20px;
    padding: 1.6rem 1.8rem 0.8rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 8px rgba(0,0,0,0.03);
    margin-bottom: 1.2rem;
    transition: box-shadow 0.3s ease, border-color 0.3s ease, transform 0.3s ease;
}
.cccard:hover {
    box-shadow: 0 12px 32px rgba(0,0,0,0.09);
    border-color: #CBD5E1;
    transform: translateY(-2px);
}
.cccard-title    { font-size: 1.08rem; font-weight: 700; color: #0F172A; margin-bottom: 0.15rem; }
.cccard-subtitle { font-size: 0.86rem; color: #94A3B8; margin-bottom: 0.8rem; }

/* ── Section Headers ── */
.sec-hdr {
    font-size: 1rem; font-weight: 700; color: #0F172A;
    margin: 2.2rem 0 1rem;
    display: flex; align-items: center; gap: 0.5rem;
    padding-bottom: 0.7rem;
    border-bottom: 2px solid #F1F5F9;
}

/* ── Alerts ── */
.alert-warm {
    background: #FEF2F2; border: 1px solid #FECACA;
    border-left: 4px solid #DC2626;
    border-radius: 12px; padding: 1rem 1.2rem;
    color: #991B1B; font-size: 0.88rem; font-weight: 500;
    animation: fadeUp 0.4s ease;
}
.alert-ok {
    background: #F0FDF4; border: 1px solid #BBF7D0;
    border-left: 4px solid #16A34A;
    border-radius: 12px; padding: 1rem 1.2rem;
    color: #166534; font-size: 0.88rem; font-weight: 500;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #C0624A 0%, #E8724A 100%);
    border: none; border-radius: 12px;
    padding: 0.75rem 2.2rem; font-size: 0.93rem;
    font-weight: 700; color: #FFFFFF;
    box-shadow: 0 4px 14px rgba(192,98,74,0.35);
    transition: all 0.25s cubic-bezier(.4,0,.2,1);
    font-family: 'DM Sans', sans-serif;
    position: relative; overflow: hidden;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #A8533E 0%, #C0624A 100%);
    box-shadow: 0 8px 24px rgba(192,98,74,0.45);
    transform: translateY(-2px);
}
div[data-testid="stButton"] > button {
    border-radius: 10px; font-family: 'DM Sans', sans-serif;
    transition: all 0.2s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
}

/* ── Scroll reveal ── */
.reveal { opacity: 0; transform: translateY(24px); transition: opacity 0.65s ease, transform 0.65s ease; }
.revealed { opacity: 1 !important; transform: translateY(0) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    try:
        wow = run_query("""
            WITH runs AS (SELECT DISTINCT run_date FROM churn_scores_history ORDER BY run_date DESC LIMIT 2),
            tw AS (SELECT COUNT(*) c FROM churn_scores_history
                   WHERE risk_bucket='High' AND run_date=(SELECT MAX(run_date) FROM runs)),
            lw AS (SELECT COUNT(*) c FROM churn_scores_history
                   WHERE risk_bucket='High' AND run_date=(SELECT MIN(run_date) FROM runs)),
            tot AS (SELECT COUNT(DISTINCT account_id) t FROM churn_scores_history
                    WHERE run_date=(SELECT MAX(run_date) FROM churn_scores_history))
            SELECT (SELECT MAX(run_date) FROM runs) AS refreshed,
                   tw.c high_now, lw.c high_prev, tw.c-lw.c change, tot.t total
            FROM tw,lw,tot;
        """)
        trend = run_query("""
            SELECT run_date, COUNT(*) AS high_risk
            FROM churn_scores_history WHERE risk_bucket='High'
            GROUP BY run_date ORDER BY run_date;
        """)
        buckets = run_query("""
            SELECT risk_bucket, COUNT(*) AS accounts
            FROM churn_scores_latest_ranked GROUP BY risk_bucket;
        """)
        region = run_query("""
            SELECT COALESCE(d.region,'Unknown') region, COUNT(*) high_risk
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id=d.account_id
            WHERE c.risk_bucket='High'
            GROUP BY d.region ORDER BY high_risk DESC;
        """)
        plan = run_query("""
            SELECT d.plan_type,
                   ROUND(AVG(c.risk_percentile)::numeric,1) avg_percentile
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id=d.account_id
            GROUP BY d.plan_type ORDER BY avg_percentile DESC;
        """)
        new_accts = run_query("""
            WITH lt AS (SELECT MAX(run_date) d FROM churn_scores_history),
                 pv AS (SELECT MAX(run_date) d FROM churn_scores_history
                        WHERE run_date<(SELECT d FROM lt))
            SELECT h.account_id, da.plan_type, da.region, da.contract_type,
                   ROUND(h.churn_risk_calibrated::numeric,4) churn_probability,
                   ROUND(h.risk_percentile::numeric,1) risk_percentile, h.risk_band
            FROM churn_scores_history h
            JOIN dim_account da ON h.account_id=da.account_id
            JOIN lt ON h.run_date=lt.d
            WHERE h.risk_bucket='High'
              AND h.account_id NOT IN (
                  SELECT account_id FROM churn_scores_history
                  WHERE run_date=(SELECT d FROM pv) AND risk_bucket='High')
            ORDER BY h.risk_percentile DESC;
        """)
        return dict(ok=True, wow=wow, trend=trend, buckets=buckets,
                    region=region, plan=plan, new_accts=new_accts)
    except Exception as e:
        return dict(ok=False, error=str(e))

D = load_data()

CHART = dict(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
             font=dict(family="DM Sans", color="#0F172A"),
             margin=dict(l=10,r=10,t=20,b=10))
CHART_TALL = dict(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                  font=dict(family="DM Sans", color="#0F172A"),
                  margin=dict(l=10,r=10,t=45,b=10))
GRID  = "#E2E8F0"

# ── Hero ──────────────────────────────────────────────────────────────────────
refreshed = D["wow"].iloc[0]["refreshed"] if D.get("ok") and not D["wow"].empty else "—"
st.markdown(f"""
<div class="hero">
  <div class="hero-orbs"></div>
  <div class="hero-grid"></div>
  <div class="hero-content">
    <div class="hero-eyebrow">B2B SaaS Intelligence Platform</div>
    <h1>Churn Copilot</h1>
    <p>Spot at-risk accounts before they leave — updated every week, backed by machine learning.</p>
    <div class="hero-badges">
      <span class="badge"><span class="live-dot"></span>&nbsp;Live</span>
      <span class="badge">🔄 Refreshed {refreshed}</span>
      <span class="badge">📉 Churn Analytics</span>
      <span class="badge">🔒 Internal</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── CTA ───────────────────────────────────────────────────────────────────────
c1, c2, _ = st.columns([2,3,3])
with c1:
    if st.button("🤖  Open Churn Copilot →", type="primary", use_container_width=True):
        st.switch_page("pages/02_Ask_Copilot.py")
with c2:
    st.markdown("<p style='padding-top:0.65rem;color:#64748B;font-size:0.84rem'>Ask any question about your churn data in plain English</p>",
                unsafe_allow_html=True)

# ── Metric Cards ──────────────────────────────────────────────────────────────
if not D.get("ok"):
    st.error(f"⚠️ Database error: {D.get('error', 'Unknown')}")
if D.get("ok") and not D["wow"].empty:
    r = D["wow"].iloc[0]
    high_now  = int(r["high_now"]);  high_prev = int(r["high_prev"])
    change    = int(r["change"]);    total      = int(r["total"])
    new_n     = len(D["new_accts"]); pct        = round(high_now/total*100,1) if total else 0

    dc = "mcard-delta-up" if change>0 else "mcard-delta-down" if change<0 else "mcard-delta-flat"
    di = "▲" if change>0 else "▼" if change<0 else "●"

    st.markdown(f"""
    <div class="cards-row">
      <div class="mcard mcard-rust" style="animation:fadeUp .5s ease .05s both">
        <div class="mcard-label">High-Risk Accounts</div>
        <div class="mcard-value" data-count="{high_now}">{high_now}</div>
        <div class="{dc}">{di} {abs(change)} vs last week</div>
      </div>
      <div class="mcard mcard-sand" style="animation:fadeUp .5s ease .15s both">
        <div class="mcard-label">Total Accounts</div>
        <div class="mcard-value" data-count="{total}">{total}</div>
        <div class="mcard-delta-flat">{pct}% flagged high-risk</div>
      </div>
      <div class="mcard mcard-mauve" style="animation:fadeUp .5s ease .25s both">
        <div class="mcard-label">Newly At-Risk</div>
        <div class="mcard-value" data-count="{new_n}">{new_n}</div>
        <div class="mcard-delta-{'up' if new_n>0 else 'flat'}">{"▲ needs attention" if new_n>0 else "● no new accounts"}</div>
      </div>
      <div class="mcard mcard-sage" style="animation:fadeUp .5s ease .35s both">
        <div class="mcard-label">Last Week</div>
        <div class="mcard-value" data-count="{high_prev}">{high_prev}</div>
        <div class="mcard-delta-flat">high-risk accounts</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Charts row 1: trend + donut ───────────────────────────────────────────────
if D.get("ok"):
    col_l, col_r = st.columns([3,2])

    with col_l:
        st.markdown('<div class="cccard reveal">', unsafe_allow_html=True)
        st.markdown('<div class="cccard-title">High-Risk Account Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="cccard-subtitle">Weekly count of high-risk accounts</div>', unsafe_allow_html=True)
        tdf = D["trend"]
        if not tdf.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tdf["run_date"].astype(str), y=tdf["high_risk"],
                mode="lines+markers+text",
                text=tdf["high_risk"], textposition="top center",
                textfont=dict(size=11, color="#0F172A"),
                line=dict(color="#2563EB", width=3),
                marker=dict(size=10, color="#2563EB",
                            line=dict(color="white", width=2)),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
                hovertemplate="<b>%{x}</b><br>%{y} high-risk<extra></extra>"
            ))
            ymax = tdf["high_risk"].max() * 1.25
            fig.update_layout(height=320, showlegend=False,
                              xaxis=dict(showgrid=False, tickfont=dict(size=12)),
                              yaxis=dict(showgrid=False, tickfont=dict(size=12), range=[0, ymax]),
                              **CHART_TALL)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="cccard reveal">', unsafe_allow_html=True)
        st.markdown('<div class="cccard-title">Risk Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="cccard-subtitle">Current snapshot — all accounts</div>', unsafe_allow_html=True)
        bdf = D["buckets"]
        if not bdf.empty:
            bcolors = {"High":"#1E3A8A","Medium":"#2563EB","Low":"#93C5FD"}
            total_v = bdf["accounts"].sum()
            fig2 = go.Figure(go.Pie(
                labels=bdf["risk_bucket"], values=bdf["accounts"],
                hole=0.6,
                marker=dict(colors=[bcolors.get(b,"#BFDBFE") for b in bdf["risk_bucket"]],
                            line=dict(color="#FFFFFF", width=3)),
                textinfo="label+percent", textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>%{value} accounts<extra></extra>"
            ))
            fig2.update_layout(
                height=320, showlegend=False,
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10,r=10,t=10,b=10),
                annotations=[dict(
                    text=f"<b>{total_v}</b><br><span style='font-size:10px'>accounts</span>",
                    x=0.5, y=0.5, font_size=16, showarrow=False,
                    font=dict(family="DM Sans", color="#0F172A")
                )]
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

# ── Charts row 2: region + plan ───────────────────────────────────────────────
if D.get("ok"):
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="cccard reveal">', unsafe_allow_html=True)
        st.markdown('<div class="cccard-title">High-Risk by Region</div>', unsafe_allow_html=True)
        st.markdown('<div class="cccard-subtitle">Number of high-risk accounts per region</div>', unsafe_allow_html=True)
        rdf = D["region"]
        if not rdf.empty:
            fig3 = go.Figure(go.Bar(
                x=rdf["region"], y=rdf["high_risk"],
                text=rdf["high_risk"], textposition="outside",
                textfont=dict(size=12, color="#0F172A"),
                marker=dict(color="#2563EB",
                            line=dict(color="#FFFFFF", width=1)),
                hovertemplate="<b>%{x}</b><br>%{y} high-risk<extra></extra>"
            ))
            fig3.update_layout(height=300, bargap=0.4, showlegend=False,
                               xaxis=dict(showgrid=False, tickfont=dict(size=12)),
                               yaxis=dict(showgrid=False, tickfont=dict(size=12),
                                          range=[0, rdf["high_risk"].max()*1.3]),
                               **CHART_TALL)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="cccard reveal">', unsafe_allow_html=True)
        st.markdown('<div class="cccard-title">Avg Risk Percentile by Plan</div>', unsafe_allow_html=True)
        st.markdown('<div class="cccard-subtitle">Higher = riskier plan tier on average</div>', unsafe_allow_html=True)
        pdf = D["plan"]
        if not pdf.empty:
            fig4 = go.Figure(go.Bar(
                x=pdf["plan_type"], y=pdf["avg_percentile"],
                text=pdf["avg_percentile"].astype(str)+"th",
                textposition="outside",
                textfont=dict(size=12, color="#0F172A"),
                marker=dict(color="#2563EB",
                            line=dict(color="#FFFFFF", width=1)),
                hovertemplate="<b>%{x}</b><br>Avg percentile: %{y}<extra></extra>"
            ))
            fig4.update_layout(height=300, bargap=0.4, showlegend=False,
                               xaxis=dict(showgrid=False, tickfont=dict(size=12)),
                               yaxis=dict(showgrid=False, tickfont=dict(size=12), range=[0,115]),
                               **CHART_TALL)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

# ── Newly at-risk ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr reveal">⚡ Newly High-Risk This Week</div>', unsafe_allow_html=True)
if D.get("ok"):
    ndf = D["new_accts"]
    if ndf.empty:
        st.markdown('<div class="alert-ok reveal">✅ No accounts newly entered high-risk this week.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-warm reveal">⚠️ {len(ndf)} account(s) newly flagged high-risk this week — use Ask Copilot to drill in.</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:2.5rem'>", unsafe_allow_html=True)
st.divider()
c1, c2, _ = st.columns([2,3,2])
with c1:
    if st.button("🤖  Go to Ask Copilot →", type="primary", use_container_width=True):
        st.switch_page("pages/02_Ask_Copilot.py")
with c2:
    st.markdown("<p style='padding-top:0.65rem;color:#64748B;font-size:0.82rem'>Drill into any segment, region, or account — ask in plain English.</p>",
                unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── JS: counters + scroll reveal + ripple ────────────────────────────────────
st.markdown("""
<script>
(function() {
  function init() {

    /* 1. Animated number counters */
    document.querySelectorAll('[data-count]').forEach(function(el) {
      var target = parseInt(el.getAttribute('data-count'), 10);
      var dur = 1300, start = performance.now();
      function tick(now) {
        var t = Math.min((now - start) / dur, 1);
        var ease = 1 - Math.pow(1 - t, 4);
        el.textContent = Math.round(ease * target).toLocaleString();
        if (t < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });

    /* 2. Scroll reveal via IntersectionObserver */
    var obs = new IntersectionObserver(function(entries) {
      entries.forEach(function(e) {
        if (e.isIntersecting) {
          e.target.classList.add('revealed');
          obs.unobserve(e.target);
        }
      });
    }, { threshold: 0.06 });
    document.querySelectorAll('.reveal').forEach(function(el) { obs.observe(el); });

    /* 3. Ripple effect on every button */
    document.querySelectorAll('button').forEach(function(btn) {
      if (btn._hasRipple) return;
      btn._hasRipple = true;
      btn.style.position = 'relative';
      btn.style.overflow = 'hidden';
      btn.addEventListener('click', function(e) {
        var rect = btn.getBoundingClientRect();
        var size = Math.max(rect.width, rect.height) * 2;
        var span = document.createElement('span');
        span.style.cssText = [
          'position:absolute', 'border-radius:50%', 'pointer-events:none',
          'width:'  + size + 'px', 'height:' + size + 'px',
          'left:'   + (e.clientX - rect.left  - size / 2) + 'px',
          'top:'    + (e.clientY - rect.top   - size / 2) + 'px',
          'background:rgba(255,255,255,0.28)',
          'animation:ripple 0.55s ease-out forwards'
        ].join(';');
        btn.appendChild(span);
        setTimeout(function() { span.remove(); }, 600);
      });
    });

    /* 4. Card tilt on mouse move */
    document.querySelectorAll('.mcard').forEach(function(card) {
      card.addEventListener('mousemove', function(e) {
        var rect = card.getBoundingClientRect();
        var x = (e.clientX - rect.left) / rect.width  - 0.5;
        var y = (e.clientY - rect.top)  / rect.height - 0.5;
        card.style.transform = 'translateY(-5px) rotateX(' + (-y*6) + 'deg) rotateY(' + (x*6) + 'deg)';
        card.style.transition = 'transform 0.1s ease';
      });
      card.addEventListener('mouseleave', function() {
        card.style.transform = '';
        card.style.transition = 'transform 0.3s cubic-bezier(.4,0,.2,1), box-shadow 0.3s, border-color 0.3s';
      });
    });

  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() { setTimeout(init, 300); });
  } else {
    setTimeout(init, 300);
  }
})();
</script>
""", unsafe_allow_html=True)
