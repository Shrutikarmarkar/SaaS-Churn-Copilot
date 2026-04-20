
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background:#FFFFFF; }
#MainMenu, footer, header { visibility:hidden; }
.main .block-container { padding:1.8rem 2.8rem 3rem; max-width:1280px; }

/* ── Keyframes ── */
@keyframes fadeUp   { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes slideIn  { from{opacity:0;transform:translateX(-14px)} to{opacity:1;transform:translateX(0)} }
@keyframes orbs     { 0%{transform:scale(1) rotate(0deg)} 100%{transform:scale(1.1) rotate(4deg)} }
@keyframes breathe  { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.6;transform:scale(0.82)} }
@keyframes gradText { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
@keyframes ripple   { 0%{transform:scale(0);opacity:0.55} 100%{transform:scale(3);opacity:0} }
@keyframes resultIn { from{opacity:0;transform:translateY(18px) scale(0.98)} to{opacity:1;transform:translateY(0) scale(1)} }
@keyframes pulse    { 0%,100%{box-shadow:0 0 0 0 rgba(37,99,235,0.3)} 50%{box-shadow:0 0 0 8px rgba(37,99,235,0)} }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0A0F1E 0%, #0F172A 55%, #1A0A2E 100%);
    border-radius: 24px; padding: 2.8rem 3.2rem;
    color: #F8FAFC; margin-bottom: 1.8rem;
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
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 44px 44px;
}
.hero-content { position:relative; z-index:1; }
.hero-eyebrow {
    font-size:0.72rem; font-weight:700; letter-spacing:0.16em;
    text-transform:uppercase; color:rgba(248,250,252,0.45);
    margin-bottom:0.8rem; animation:slideIn 0.5s ease 0.1s both;
}
.hero h1 {
    font-family:'DM Serif Display',serif;
    font-size:2.8rem; font-weight:400; margin:0 0 0.6rem;
    background: linear-gradient(120deg, #FFFFFF 0%, #60A5FA 50%, #A78BFA 100%);
    background-size: 200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
    animation: slideIn 0.5s ease 0.2s both, gradText 6s linear infinite;
}
.hero p {
    font-size:0.96rem; color:rgba(248,250,252,0.6);
    margin:0; max-width:480px; line-height:1.65;
    animation:slideIn 0.5s ease 0.3s both;
}
.live-dot {
    width:8px; height:8px; border-radius:50%;
    background:#4ADE80; display:inline-block;
    animation:breathe 2.2s ease infinite;
}

/* ── Category label ── */
.cat-label {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#64748B; margin:0.8rem 0 0.5rem;
    display:flex; align-items:center; gap:0.4rem;
}
.cat-label::after {
    content:''; flex:1; height:1px; background:#F1F5F9;
}

/* ── Question buttons ── */
div[data-testid="stButton"] > button {
    border-radius:10px; font-size:0.83rem; font-weight:500;
    border:1px solid #E2E8F0; background:#FFFFFF; color:#1E293B;
    padding:0.55rem 0.85rem; text-align:left;
    box-shadow:0 1px 3px rgba(0,0,0,0.05);
    transition:all 0.2s cubic-bezier(.4,0,.2,1); width:100%;
    position:relative; overflow:hidden;
}
div[data-testid="stButton"] > button:hover {
    background:#EFF6FF; border-color:#93C5FD;
    color:#1D4ED8; box-shadow:0 4px 12px rgba(37,99,235,0.14);
    transform:translateY(-2px);
}
div[data-testid="stButton"] > button[kind="primary"] {
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    border:none; color:#FFFFFF; font-weight:700;
    box-shadow:0 4px 14px rgba(37,99,235,0.38);
    animation:pulse 2.5s ease infinite;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#1D4ED8,#1E40AF);
    box-shadow:0 6px 22px rgba(37,99,235,0.48);
    transform:translateY(-2px);
    animation:none;
}

/* ── Back button ── */
div[data-testid="stButton"] > button:not([kind="primary"]):first-of-type {
    border-radius:10px; font-size:0.82rem;
    border:1px solid #E2E8F0; background:#F8FAFC; color:#475569;
}
div[data-testid="stButton"] > button:not([kind="primary"]):first-of-type:hover {
    background:#FFFFFF; color:#0F172A; border-color:#CBD5E1;
    transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

/* ── Tabs ── */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#F8FAFC; border-radius:14px;
    border:1px solid #E2E8F0; padding:0.3rem; gap:0.15rem;
    box-shadow:0 1px 4px rgba(0,0,0,0.05);
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius:10px; font-size:0.8rem; font-weight:500;
    padding:0.42rem 0.8rem; color:#475569;
    transition:all 0.2s ease;
}
div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background:rgba(37,99,235,0.06); color:#2563EB;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    background:linear-gradient(135deg,#1D4ED8,#2563EB) !important;
    color:#FFFFFF !important; font-weight:700 !important;
    box-shadow:0 2px 8px rgba(37,99,235,0.3) !important;
}

/* ── Result card ── */
.result-card {
    background:#FFFFFF; border-radius:20px;
    border:1px solid #E2E8F0;
    box-shadow:0 4px 24px rgba(0,0,0,0.08);
    padding:1.8rem 2rem 1.4rem;
    margin-top:1.5rem;
    animation: resultIn 0.45s cubic-bezier(.4,0,.2,1);
}
.result-title {
    font-family:'DM Serif Display',serif;
    font-size:1.15rem; font-weight:400; color:#0F172A;
    margin-bottom:1.2rem; padding-bottom:0.8rem;
    border-bottom:1px solid #F1F5F9;
    display:flex; align-items:center; gap:0.5rem;
}
.result-title::before {
    content:''; width:3px; height:1.1rem;
    background:linear-gradient(180deg,#2563EB,#7C3AED);
    border-radius:2px; display:inline-block; flex-shrink:0;
}

/* ── Custom input section ── */
.input-section {
    background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:16px; padding:1.2rem 1.5rem;
    margin-top:0.5rem;
}

/* ── Text input ── */
.stTextInput input {
    border-radius:10px; border:1.5px solid #E2E8F0;
    background:#FFFFFF; font-size:0.88rem;
    padding:0.6rem 1rem; color:#0F172A;
    font-family:'DM Sans',sans-serif;
    transition:border 0.2s, box-shadow 0.2s;
}
.stTextInput input:focus {
    border-color:#2563EB;
    box-shadow:0 0 0 3px rgba(37,99,235,0.12);
    outline:none;
}
.stTextInput input::placeholder { color:#94A3B8; }

/* ── Scroll reveal ── */
.reveal { opacity:0; transform:translateY(20px); transition:opacity 0.6s ease, transform 0.6s ease; }
.revealed { opacity:1 !important; transform:translateY(0) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
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

# ── Header ────────────────────────────────────────────────────────────────────
cb, _ = st.columns([1,9])
with cb:
    if st.button("← Dashboard"):
        st.switch_page("Home.py")

st.markdown("""
<div class="hero">
  <div class="hero-orbs"></div>
  <div class="hero-grid"></div>
  <div class="hero-content">
    <div class="hero-eyebrow">Churn Intelligence</div>
    <h1>Ask Copilot</h1>
    <p>Click a preset question or type your own — every answer is backed by a live database query.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Presets ───────────────────────────────────────────────────────────────────
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

tabs = st.tabs(list(CATEGORIES.keys()))
for tab, (cat, qs) in zip(tabs, CATEGORIES.items()):
    with tab:
        st.markdown('<div class="cat-label">Select a question</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for i, (lbl, qn, params) in enumerate(qs):
            with (c1 if i%2==0 else c2):
                if st.button(lbl, key=f"q_{cat}_{i}", use_container_width=True):
                    run_direct(lbl, qn, params)

# ── Custom input ──────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<p class="cat-label">Or type your own question</p>', unsafe_allow_html=True)
ci, cb2 = st.columns([7,1])
with ci:
    custom = st.text_input("", placeholder="e.g.  high risk Enterprise accounts in EU with Monthly contracts",
                           label_visibility="collapsed")
with cb2:
    st.markdown("<div style='padding-top:0.28rem'>", unsafe_allow_html=True)
    if st.button("Run →", type="primary", use_container_width=True):
        if custom.strip(): run_text(custom.strip())
        else: st.warning("Please enter a question.")
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Chart renderer ────────────────────────────────────────────────────────────
CHART = dict(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
             font=dict(family="DM Sans", color="#0F172A"),
             margin=dict(l=20,r=20,t=50,b=30))
GRID  = "#E2E8F0"
PALET = ["#2563EB"] * 10

def render_chart(df: pd.DataFrame, query_name: str):
    if df is None or df.empty or len(df) < 2: return
    num  = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not num: return

    if "run_date" in df.columns:
        fig = go.Figure(go.Scatter(
            x=df["run_date"].astype(str), y=df[num[0]],
            mode="lines+markers+text",
            text=df[num[0]], textposition="top center",
            textfont=dict(size=11, color="#0F172A"),
            line=dict(color="#2563EB", width=3),
            marker=dict(size=10, color="#2563EB", line=dict(color="white", width=2)),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"
        ))
        fig.update_layout(height=380, showlegend=False,
                          xaxis=dict(showgrid=False, tickfont=dict(size=12)),
                          yaxis=dict(showgrid=True, gridcolor=GRID, tickfont=dict(size=12)),
                          **CHART)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        return

    if query_name in ("risk_bucket_distribution","risk_band_distribution") and text:
        cmap = {"High":"#1E3A8A","Medium":"#2563EB","Low":"#93C5FD",
                "Top 1%":"#1E3A8A","Top 5%":"#1D4ED8","Top 10%":"#2563EB",
                "Top 25%":"#60A5FA","Rest":"#BFDBFE"}
        colors = [cmap.get(l,"#94A3B8") for l in df[text[0]]]
        fig = go.Figure(go.Pie(
            labels=df[text[0]], values=df[num[0]], hole=0.58,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
            textinfo="label+percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value} accounts<extra></extra>"
        ))
        fig.update_layout(height=380, showlegend=False, paper_bgcolor="#FFFFFF",
                          margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        return

    if text and num:
        fig = go.Figure(go.Bar(
            x=df[text[0]], y=df[num[0]],
            text=df[num[0]], textposition="outside",
            textfont=dict(size=12, color="#0F172A"),
            marker=dict(color=PALET[:len(df)], line=dict(color="#FFFFFF", width=1)),
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"
        ))
        fig.update_layout(height=380, bargap=0.38, showlegend=False,
                          xaxis=dict(showgrid=False, tickfont=dict(size=12)),
                          yaxis=dict(showgrid=True, gridcolor=GRID, tickfont=dict(size=12)),
                          **CHART)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ── Result ────────────────────────────────────────────────────────────────────
if st.session_state.output:
    st.markdown('<div id="result-anchor"></div>', unsafe_allow_html=True)
    out = st.session_state.output
    if out["matched_query"] is None:
        st.error(out["message"])
    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">{st.session_state.active_label}</div>
        </div>""", unsafe_allow_html=True)

        df = out["result"]
        qn = out.get("matched_query","")

        if df is not None and not df.empty:
            render_chart(df, qn)
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={
                             "risk_percentile":       st.column_config.NumberColumn("Risk %ile", format="%.1f"),
                             "churn_probability":     st.column_config.NumberColumn("Churn Prob", format="%.4f"),
                             "churn_risk_calibrated": st.column_config.NumberColumn("Churn Prob", format="%.4f"),
                         })
        else:
            st.info("No results returned for this query.")


# ── JS: scroll reveal + ripple ────────────────────────────────────────────────
st.markdown("""
<script>
(function() {
  function init() {

    /* Scroll reveal */
    var obs = new IntersectionObserver(function(entries) {
      entries.forEach(function(e) {
        if (e.isIntersecting) { e.target.classList.add('revealed'); obs.unobserve(e.target); }
      });
    }, { threshold: 0.06 });
    document.querySelectorAll('.reveal').forEach(function(el) { obs.observe(el); });

    /* Ripple on buttons */
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
          'position:absolute','border-radius:50%','pointer-events:none',
          'width:' + size + 'px','height:' + size + 'px',
          'left:' + (e.clientX - rect.left - size/2) + 'px',
          'top:'  + (e.clientY - rect.top  - size/2) + 'px',
          'background:rgba(255,255,255,0.28)',
          'animation:ripple 0.55s ease-out forwards'
        ].join(';');
        btn.appendChild(span);
        setTimeout(function() { span.remove(); }, 600);
      });
    });

    /* Input focus glow */
    document.querySelectorAll('input[type="text"]').forEach(function(inp) {
      inp.addEventListener('focus', function() {
        inp.parentElement.style.transition = 'box-shadow 0.2s ease';
        inp.parentElement.style.boxShadow  = '0 0 0 4px rgba(37,99,235,0.1)';
        inp.parentElement.style.borderRadius = '12px';
      });
      inp.addEventListener('blur', function() {
        inp.parentElement.style.boxShadow = '';
      });
    });

  }

    /* Auto-scroll to result anchor */
    var anchor = document.getElementById('result-anchor');
    if (anchor) {
      anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() { setTimeout(init, 300); });
  } else {
    setTimeout(init, 300);
  }
})();
</script>
""", unsafe_allow_html=True)
