
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

EXPLAIN_KEY = "explain_input"

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

    st.markdown("---")
    st.markdown('<div class="q-panel-title">🔍 Explain Account</div>', unsafe_allow_html=True)
    acc_input = st.text_input("Account ID", placeholder="e.g. ACC_0086",
                              label_visibility="collapsed", key=EXPLAIN_KEY)
    if st.button("Explain →", use_container_width=True, key="explain_btn"):
        if acc_input.strip():
            run_direct(f"Why is {acc_input.strip()} high risk?",
                       "explain_account", {"account_id": acc_input.strip()})
        else:
            st.warning("Enter an account ID.")

# ── SHAP insight card renderer ────────────────────────────────────────────────
def _insight_sentence(fname: str, value: float, avg) -> str:
    v = float(value)
    try:
        a = float(avg)
        a = None if pd.isna(a) else a
    except (TypeError, ValueError):
        a = None

    has_avg = a is not None
    pct     = abs((v - a) / max(abs(a), 0.001) * 100) if has_avg else 0
    below   = has_avg and v < a
    above   = has_avg and v > a

    if fname == "seats":
        what = ("<b>What this means:</b> Seats = the number of licensed user accounts under this subscription — how many people from the customer's team can log in and use the product.")
        if has_avg:
            body = (f"This account has <b>{int(v)} seat(s)</b>, vs. the average of <b>{a:.1f} seats</b> across all accounts ({pct:.0f}% {'below' if below else 'above'} average). "
                    + ("With so few seats, only one or two people are using the product. If that person leaves or deprioritises it, the entire account churns." if below else
                       "A larger seat count means more stakeholders depend on the product — if engagement drops, there is more revenue at risk."))
        else:
            body = f"This account has <b>{int(v)} seat(s)</b>. A very small seat count suggests limited team adoption, which the model associates with higher cancellation risk."
        return f"{what}<br>{body}"

    if fname == "tenure_days":
        years = v / 365
        what  = "<b>What this means:</b> Account tenure = how long this company has been a paying customer (in days)."
        if has_avg:
            body = (f"This account has been active for <b>{int(v)} days (~{years:.1f} years)</b>, vs. the average of <b>{a:.0f} days (~{a/365:.1f} years)</b>. "
                    + ("Long-tenured accounts that show declining engagement are a strong churn signal — they may be staying out of habit rather than active value. A proactive check-in from a Customer Success rep is recommended." if above else
                       "Relatively newer accounts that show low engagement early are at elevated churn risk before they reach full adoption."))
        else:
            body = f"This account has been active for <b>{int(v)} days (~{years:.1f} years)</b>. Long-tenured accounts that become disengaged are a strong churn signal."
        return f"{what}<br>{body}"

    if "active_users" in fname:
        period = fname.split("_")[-1]
        what   = f"<b>What this means:</b> Active users (last {period}) = the average number of unique team members who logged into the product each day over the past {period}. This measures how regularly the team actually uses the product day-to-day."
        if has_avg:
            freq   = f"roughly 1 login every {int(round(1/v))} days" if 0 < v < 1 else f"{v:.2f} users/day"
            body   = (f"This account had <b>{freq}</b> of active usage, vs. the average of <b>{a:.2f} active users/day</b> across all accounts ({pct:.0f}% {'below' if below else 'above'} average). "
                      + ("Near-zero daily logins means the product is sitting unused. This is one of the strongest early signals that an account is about to cancel." if below else
                         "Above-average daily logins indicate the team is regularly engaged with the product."))
        else:
            body = f"This account averaged <b>{v:.2f} active users/day</b>. Very low daily active users indicates the product is rarely being used."
        return f"{what}<br>{body}"

    if "sessions" in fname and "mean" in fname:
        period = fname.split("_")[-1]
        what   = f"<b>What this means:</b> Sessions (last {period}) = how many times per day users from this account opened or actively used the product, averaged over the past {period}. Each session is one intentional visit to the product."
        if has_avg:
            body = (f"This account had <b>{v:.2f} sessions/day</b> over the past {period}, vs. the average of <b>{a:.2f} sessions/day</b> ({pct:.0f}% {'below' if below else 'above'} average). "
                    + ("Infrequent sessions suggest the product is not part of the team's regular workflow, making cancellation significantly more likely." if below else
                       "High session frequency indicates strong daily engagement with the product."))
        else:
            body = f"This account had <b>{v:.2f} sessions/day</b>. Low session frequency signals the product is not being regularly used."
        return f"{what}<br>{body}"

    if "events" in fname and "mean" in fname:
        period = fname.split("_")[-1]
        what   = f"<b>What this means:</b> Events (last {period}) = the average number of in-product actions per day — things like clicking a button, running a report, or saving a record — recorded over the past {period}. More events = deeper product engagement."
        if has_avg:
            body = (f"This account triggered <b>{v:.2f} events/day</b>, vs. the average of <b>{a:.2f} events/day</b> ({pct:.0f}% {'below' if below else 'above'} average). "
                    + ("Low event counts mean users are not exploring the product's features. Accounts that don't engage deeply rarely renew." if below else
                       "High event counts indicate users are actively using features — a healthy engagement signal."))
        else:
            body = f"This account averaged <b>{v:.2f} in-product actions/day</b>. Low in-product activity signals limited engagement with the product's features."
        return f"{what}<br>{body}"

    if "revenue" in fname:
        period = fname.split("_")[-1]
        what   = f"<b>What this means:</b> Revenue (last {period}) = the total subscription revenue collected from this account over the past {period}. In SaaS, this reflects the contract size — larger accounts pay more."
        if has_avg:
            body = (f"This account generated <b>${v:,.2f}</b> over the past {period}, vs. the average of <b>${a:,.2f}</b> per account ({pct:.0f}% {'below' if below else 'above'} average). "
                    + ("Smaller accounts have fewer switching costs and churn at higher rates. The model flags low-revenue accounts as higher risk." if below else
                       "Larger accounts have more at stake — if engagement drops on a high-value account, it represents significant revenue at risk."))
        else:
            body = f"This account generated <b>${v:,.2f}</b> over the past {period}."
        return f"{what}<br>{body}"

    if fname == "days_since_last_activity":
        what = "<b>What this means:</b> Days since last activity = how many days ago any user from this account last logged in. This is the most direct measure of whether the account is still actively using the product."
        if has_avg:
            body = (f"Last recorded login was <b>{int(v)} days ago</b>, vs. an average of <b>{a:.0f} days</b> across all accounts. "
                    + (f"This is {pct:.0f}% longer than typical — an extended gap in logins is a direct warning sign that the account may be heading toward cancellation." if above else
                       "This account is logging in more recently than most — a positive signal."))
        else:
            body = f"Last activity was <b>{int(v)} days ago</b>. Long gaps since last login are a direct warning sign of disengagement."
        return f"{what}<br>{body}"

    if fname == "sessions_drop_7v7":
        what = "<b>What this means:</b> Week-over-week session drop = the percentage change in login sessions between the most recent 7 days and the 7 days before that. A positive value means usage fell."
        body = (f"Sessions dropped <b>{v:.0%} week-over-week</b>. A sudden decline in logins is a strong early warning — it suggests the account's users are pulling back from the product right now." if v > 0.1 else
                f"Week-over-week session change: <b>{v:+.2f}</b>" + (f" (avg {a:+.2f})." if has_avg else "."))
        return f"{what}<br>{body}"

    if fname == "sessions_trend_7_minus_30":
        what = "<b>What this means:</b> Session trend = difference between the 7-day average and the 30-day average session count. A negative value means recent usage is lower than the longer-term baseline — a declining trend."
        body = (f"The 7-day average is <b>{abs(v):.2f} sessions/day below</b> the 30-day baseline — usage is actively declining, not just low." if v < -0.05 else
                f"Session trend: <b>{v:+.2f}</b>" + (f" vs. 30-day baseline (avg {a:+.2f})." if has_avg else "."))
        return f"{what}<br>{body}"

    # OHE / unknown feature
    return (f"The model identified this account's profile as a risk factor. "
            f"Value: <b>{v:.3g}</b>" + (f" vs. population avg <b>{a:.3g}</b>." if has_avg else "."))


def _render_shap_cards(df: pd.DataFrame):
    st.markdown("""
    <style>
    .shap-card {
        background: #F8FAFC; border-radius: 14px;
        padding: 1rem 1.2rem; margin-bottom: 0.7rem;
        border-left: 5px solid #EF4444;
    }
    .shap-card.green { border-left-color: #22C55E; }
    .shap-card-title {
        font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;
    }
    .shap-card-body { font-size: 0.9rem; color: #334155; line-height: 1.7; }
    .shap-badge {
        display: inline-block; font-size: 0.72rem; font-weight: 700;
        padding: 2px 8px; border-radius: 20px; margin-left: 8px;
        vertical-align: middle;
    }
    .shap-badge.red  { background: #FEE2E2; color: #B91C1C; }
    .shap-badge.grn  { background: #DCFCE7; color: #15803D; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("**Top risk drivers — what's making this account high risk:**")
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    for _, row in df.iterrows():
        fname  = str(row.get("feature_name", ""))
        label  = str(row["driver"])
        value  = float(row["value"])
        avg    = row.get("pop_avg", None)
        shap_v = float(row["shap_value"])

        is_risk  = shap_v > 0
        cls      = "" if is_risk else " green"
        badge_cls = "red" if is_risk else "grn"
        badge_txt = "↑ Increases risk" if is_risk else "↓ Reduces risk"
        sentence  = _insight_sentence(fname, value, avg)

        st.markdown(f"""
        <div class="shap-card{cls}">
            <div class="shap-card-title">
                {label}
                <span class="shap-badge {badge_cls}">{badge_txt}</span>
            </div>
            <div class="shap-card-body">{sentence}</div>
        </div>
        """, unsafe_allow_html=True)


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

    # SHAP explanation: render plain-English insight cards
    if "driver" in df.columns and "shap_value" in df.columns:
        _render_shap_cards(df)
        return

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
                if qn != "explain_account":
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
