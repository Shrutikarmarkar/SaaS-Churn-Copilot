
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os, sys

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COPILOT_DIR = os.path.join(BASE_DIR, "copilot")
if COPILOT_DIR not in sys.path:
    sys.path.append(COPILOT_DIR)
from query_router import answer_direct, answer_question
from llm_sql_router import answer_question_llm

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
    st.session_state.output       = answer_question_llm(q)

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
def _infer_fname(fname: str, label: str) -> str:
    """Fall back to inferring feature name from the human-readable label."""
    if fname: return fname
    l = label.lower()
    if "seat"                          in l: return "seats"
    if "tenure"                        in l: return "tenure_days"
    if "days since" in l or "last activ" in l: return "days_since_last_activity"
    if "active user" in l and "7"      in l: return "active_users_mean_7d"
    if "active user" in l and "14"     in l: return "active_users_mean_14d"
    if "active user"                   in l: return "active_users_mean_30d"
    if "session" in l and "drop"       in l: return "sessions_drop_7v7"
    if "session" in l and "trend"      in l: return "sessions_trend_7_minus_30"
    if "session" in l and "7"          in l: return "sessions_mean_7d"
    if "session" in l and "14"         in l: return "sessions_mean_14d"
    if "session"                       in l: return "sessions_mean_30d"
    if "event"   in l and "7"          in l: return "events_mean_7d"
    if "event"   in l and "14"         in l: return "events_mean_14d"
    if "event"                         in l: return "events_mean_30d"
    if "revenue" in l and "7"          in l: return "revenue_sum_7d"
    if "revenue" in l and "14"         in l: return "revenue_sum_14d"
    if "revenue"                       in l: return "revenue_sum_30d"
    return fname

def _period_days(fname: str) -> int:
    if "7d" in fname:  return 7
    if "14d" in fname: return 14
    return 30

def _insight_sentence(fname: str, label: str, value: float, avg) -> str:
    fname = _infer_fname(fname, label)
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
        what = "<b>What this means:</b> Seats = the number of licensed user accounts purchased under this subscription — i.e. how many people from this company can log in and use the product."
        val  = f"<b>{int(v)} seat{'s' if v != 1 else ''}</b>"
        if has_avg:
            body = (f"This account has {val}, compared to an average of <b>{a:.1f} seats</b> across all accounts ({pct:.0f}% {'below' if below else 'above'} average). "
                    + ("With only 1–2 people using the product, if that person leaves or deprioritises it, the whole account churns. Low seat counts strongly predict cancellation." if below else
                       "More seats means more stakeholders are using the product — cancellation requires buy-in from a larger group, reducing churn risk."))
        else:
            body = f"This account has {val}. Very few seats means limited team adoption, which the model associates with higher cancellation risk."
        return f"{what}<br>{body}"

    if fname == "tenure_days":
        yrs  = v / 365
        what = "<b>What this means:</b> Account tenure = how long this company has been a paying customer, measured in days."
        val  = f"<b>{int(v)} days (~{yrs:.1f} years)</b>"
        if has_avg:
            a_yrs = a / 365
            body = (f"This account has been a customer for {val}, vs. an average of <b>{a:.0f} days (~{a_yrs:.1f} years)</b>. "
                    + ("Long-tenured accounts that stop engaging are high churn risk — they may be staying out of habit rather than genuine value. A Customer Success check-in is recommended." if above else
                       "Newer accounts that haven't fully adopted the product are at higher risk of early cancellation before they see full value."))
        else:
            body = f"This account has been a customer for {val}. Long-tenured accounts that become disengaged are a strong churn signal."
        return f"{what}<br>{body}"

    if "active_users" in fname:
        days = _period_days(fname)
        total     = max(1, round(v * days))
        avg_total = max(1, round(a * days)) if has_avg else None
        what = (f"<b>What this means:</b> Active users (last {days} days) = how many unique team members from this company "
                f"logged into the product at least once over the past {days} days. "
                f"This tells us how many real people are actually using the product.")
        val  = f"<b>~{total} user{'s' if total != 1 else ''} active in the past {days} days</b>"
        if has_avg:
            body = (f"This account had {val}, vs. an average of <b>~{avg_total} users</b> across all accounts ({pct:.0f}% {'below' if below else 'above'} average). "
                    + (f"With only {total} user active, the product is essentially sitting unused. Near-zero user activity is one of the strongest predictors of cancellation." if total <= 1 else
                       "Above-average user activity indicates the team is regularly engaged with the product."))
        else:
            body = f"This account had {val}. Very few active users means the product is rarely being used by the team."
        return f"{what}<br>{body}"

    if "sessions" in fname and "mean" in fname:
        days  = _period_days(fname)
        total     = max(1, round(v * days))
        avg_total = max(1, round(a * days)) if has_avg else None
        what = (f"<b>What this means:</b> Sessions (last {days} days) = the total number of times any user from this company "
                f"opened or logged into the product over the past {days} days. Each session = one intentional visit.")
        val  = f"<b>~{total} login session{'s' if total != 1 else ''} in the past {days} days</b>"
        if has_avg:
            body = (f"This account had {val}, vs. an average of <b>~{avg_total} sessions</b> ({pct:.0f}% {'below' if below else 'above'} average). "
                    + ("Very few logins mean the product is not part of the team's daily routine — this strongly predicts cancellation." if below else
                       "A high number of sessions indicates the team is regularly using the product."))
        else:
            body = f"This account had {val}. Low session count suggests the product is not being regularly used."
        return f"{what}<br>{body}"

    if "events" in fname and "mean" in fname:
        days  = _period_days(fname)
        total     = max(1, round(v * days))
        avg_total = max(1, round(a * days)) if has_avg else None
        what = (f"<b>What this means:</b> In-product events (last {days} days) = the total number of actions taken inside the product "
                f"— such as clicking a button, submitting a form, or using a feature — over the past {days} days. "
                f"More events = users are actively exploring and using the product.")
        val  = f"<b>~{total} in-product action{'s' if total != 1 else ''} in the past {days} days</b>"
        if has_avg:
            body = (f"This account recorded {val}, vs. an average of <b>~{avg_total} actions</b> ({pct:.0f}% {'below' if below else 'above'} average). "
                    + ("Low in-product activity means users are not engaging with the product's features. Accounts that don't interact deeply rarely renew." if below else
                       "High in-product activity shows users are actively exploring features — a healthy retention signal."))
        else:
            body = f"This account recorded {val}. Low in-product activity signals limited engagement with the product's features."
        return f"{what}<br>{body}"

    if "revenue" in fname:
        days  = _period_days(fname)
        what = (f"<b>What this means:</b> Revenue (last {days} days) = the total subscription payment collected from this "
                f"account over the past {days} days. In SaaS, this reflects the account's contract size.")
        val  = f"<b>${v:,.2f} paid in the past {days} days</b>"
        if has_avg:
            body = (f"This account contributed {val}, vs. an average of <b>${a:,.2f}</b> per account ({pct:.0f}% {'below' if below else 'above'} average). "
                    + ("Smaller-paying accounts have fewer switching costs and churn at higher rates. The model flags low-revenue accounts as elevated risk." if below else
                       "Larger accounts have more at stake — if engagement drops on a high-value account, the revenue risk is significant."))
        else:
            body = f"This account contributed {val}."
        return f"{what}<br>{body}"

    if fname == "days_since_last_activity":
        what = "<b>What this means:</b> Days since last activity = how many days ago any user from this company last logged in. This is the most direct signal of whether the account is still actively using the product."
        val  = f"<b>last login was {int(v)} days ago</b>"
        if has_avg:
            body = (f"For this account, the {val}, vs. an average of <b>{a:.0f} days</b> across all accounts. "
                    + (f"This is {pct:.0f}% longer than typical — an extended gap without any login is a direct warning sign the account may be heading toward cancellation." if above else
                       "This account logged in more recently than average — a positive signal."))
        else:
            body = f"For this account, the {val}. Extended periods without login are a direct warning sign of disengagement."
        return f"{what}<br>{body}"

    if fname == "sessions_drop_7v7":
        what = "<b>What this means:</b> Week-over-week session drop = the change in login count between the most recent 7 days and the 7 days before that. A drop means users are pulling back from the product right now."
        body = (f"Sessions fell by <b>{v:.0%} this week vs. last week</b>. A sudden usage decline is a strong early warning — it means users are actively stepping away from the product." if v > 0.1 else
                f"Session count was relatively stable week-over-week (change: {v:+.2f}).")
        return f"{what}<br>{body}"

    if fname == "sessions_trend_7_minus_30":
        what = "<b>What this means:</b> Session trend = whether recent usage (last 7 days) is higher or lower than the longer-term baseline (last 30 days). A negative number means usage is declining."
        body = (f"Recent usage is <b>{abs(v):.2f} sessions/day below the 30-day baseline</b> — the account's activity is actively declining, not just low." if v < -0.05 else
                f"Usage trend is relatively flat (7d vs 30d difference: {v:+.2f}).")
        return f"{what}<br>{body}"

    # Generic fallback
    return (f"The model identified this account characteristic as a risk factor. "
            f"Value: <b>{v:.3g}</b>" + (f" vs. population avg <b>{a:.3g}</b>." if has_avg else "."))


def _render_shap_cards(df: pd.DataFrame):
    st.markdown("""
    <style>
    .verdict-banner {
        border-radius: 12px; padding: 1rem 1.4rem;
        margin-bottom: 1.2rem; border-left: 5px solid;
    }
    .verdict-banner.high  { background:#FEF2F2; border-color:#DC2626; }
    .verdict-banner.med   { background:#FFFBEB; border-color:#D97706; }
    .verdict-banner.low   { background:#F0FDF4; border-color:#16A34A; }
    .verdict-title {
        font-size: 1.05rem; font-weight: 800; margin-bottom: 0.3rem;
    }
    .verdict-title.high { color: #B91C1C; }
    .verdict-title.med  { color: #B45309; }
    .verdict-title.low  { color: #15803D; }
    .verdict-sub { font-size: 0.88rem; color: #475569; }

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
    .shap-badge.red { background: #FEE2E2; color: #B91C1C; }
    .shap-badge.grn { background: #DCFCE7; color: #15803D; }
    .drivers-header { font-size: 0.92rem; font-weight: 700; color: #0F172A;
                      margin-bottom: 0.6rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Risk verdict banner ───────────────────────────────────────────────────
    first = df.iloc[0]

    # If risk columns are absent (old cached query_router), fetch them directly
    if "risk_bucket" not in df.columns and "account_id" in df.columns:
        try:
            from query_db import run_query as _rq
            _acc = str(first["account_id"]).strip()
            _risk_df = _rq(f"""
                SELECT risk_bucket, risk_band, risk_percentile, churn_risk_calibrated
                FROM churn_scores_latest_ranked WHERE account_id = '{_acc}'
            """)
            if _risk_df is not None and not _risk_df.empty:
                first = first.copy()
                first["risk_bucket"]     = _risk_df.iloc[0]["risk_bucket"]
                first["risk_band"]       = _risk_df.iloc[0]["risk_band"]
                first["risk_percentile"] = _risk_df.iloc[0]["risk_percentile"]
                first["churn_probability"] = _risk_df.iloc[0]["churn_risk_calibrated"]
        except Exception:
            pass

    bucket     = str(first.get("risk_bucket", "")).strip()
    band       = str(first.get("risk_band",   "")).strip()
    percentile = first.get("risk_percentile",  None)
    churn_prob = first.get("churn_probability", None)

    has_score = bucket not in ("", "None", "nan") and percentile is not None
    try:
        pct_val = float(percentile)
    except (TypeError, ValueError):
        has_score = False
        pct_val   = 0.0

    try:
        prob_val = float(churn_prob) if churn_prob is not None else None
    except (TypeError, ValueError):
        prob_val = None

    if has_score:
        if bucket == "High":
            cls, verdict = "high", "HIGH RISK"
            prob_str = f" Estimated churn probability: <b>{prob_val:.1%}</b>." if prob_val is not None else ""
            sub = (f"Risk Percentile: <b>{pct_val:.1f}th</b> — this account is in the top "
                   f"{100 - pct_val:.0f}% most at-risk accounts.{prob_str} "
                   f"The factors below are driving the elevated risk score.")
        elif bucket == "Medium":
            cls, verdict = "med", "MEDIUM RISK"
            sub = (f"Risk Percentile: <b>{pct_val:.1f}th</b>. "
                   f"This account shows some warning signs but is not yet in the high-risk zone. "
                   f"The factors below are worth monitoring.")
        else:
            cls, verdict = "low", "LOW RISK"
            sub = (f"Risk Percentile: <b>{pct_val:.1f}th</b>. "
                   f"This account is healthy overall. "
                   f"The factors below are what the model monitors — they are currently within normal range "
                   f"and do not indicate imminent churn.")
    else:
        cls, verdict, sub = "med", "ACCOUNT FOUND", "Risk score not available for this account."

    st.markdown(f"""
    <div class="verdict-banner {cls}">
        <div class="verdict-title {cls}">{verdict}</div>
        <div class="verdict-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section header changes based on risk level ────────────────────────────
    if has_score and bucket == "High":
        header = "Key factors driving the elevated churn risk:"
    elif has_score and bucket == "Medium":
        header = "Factors to monitor for this account:"
    else:
        header = "Model factors — currently healthy for this account:"

    st.markdown(f'<div class="drivers-header">{header}</div>', unsafe_allow_html=True)

    # ── Driver cards ──────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        fname  = str(row.get("feature_name", ""))
        label  = str(row["driver"])
        value  = float(row["value"])
        avg    = row.get("pop_avg", None)
        shap_v = float(row["shap_value"])

        is_risk   = shap_v > 0
        card_cls  = "" if is_risk else " green"
        badge_cls = "red" if is_risk else "grn"
        badge_txt = "↑ Increases risk" if is_risk else "↓ Reduces risk"
        sentence  = _insight_sentence(fname, label, value, avg)

        st.markdown(f"""
        <div class="shap-card{card_cls}">
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
        has_plan = "plan_type" in plot_df.columns
        has_prob = "churn_probability" in plot_df.columns
        has_contract = "contract_type" in plot_df.columns

        PLAN_COLORS = {"Enterprise": "#1E3A8A", "Pro": "#3B82F6", "Basic": "#93C5FD"}

        fig = go.Figure()

        if has_plan:
            y_order = plot_df["account_id"].astype(str).tolist()
            for plan, color in PLAN_COLORS.items():
                sub = plot_df[plot_df["plan_type"] == plan]
                if sub.empty:
                    continue
                if has_prob:
                    bar_text = [f"{r:.1f}  ({p:.1%})" for r, p in
                                zip(sub["risk_percentile"], sub["churn_probability"])]
                else:
                    bar_text = sub["risk_percentile"].round(1).astype(str).tolist()
                hover_extra = ""
                if has_contract:
                    hover_extra = "<br>Contract: %{customdata}"
                    customdata = sub["contract_type"].tolist()
                else:
                    customdata = None
                fig.add_trace(go.Bar(
                    x=sub["risk_percentile"],
                    y=sub["account_id"].astype(str),
                    orientation="h",
                    name=plan,
                    text=bar_text,
                    textposition="outside",
                    textfont=dict(size=11, color="#0F172A"),
                    marker=dict(color=color, line=dict(color="#FFFFFF", width=1)),
                    customdata=customdata,
                    hovertemplate=(
                        f"<b>%{{y}}</b><br>Plan: {plan}{hover_extra}"
                        f"<br>Risk Percentile: %{{x:.1f}}"
                        + (f"<br>Churn Probability: %{{text}}" if not has_prob else "")
                        + "<extra></extra>"
                    )
                ))
        else:
            if has_prob:
                bar_text = [f"{r:.1f}  ({p:.1%})" for r, p in
                            zip(plot_df["risk_percentile"], plot_df["churn_probability"])]
            else:
                bar_text = plot_df["risk_percentile"].round(1).astype(str).tolist()
            fig.add_trace(go.Bar(
                x=plot_df["risk_percentile"],
                y=plot_df["account_id"].astype(str),
                orientation="h",
                text=bar_text,
                textposition="outside",
                textfont=dict(size=11, color="#0F172A"),
                marker=dict(color="#2563EB", line=dict(color="#FFFFFF", width=1)),
                hovertemplate="<b>%{y}</b><br>Risk %ile: %{x:.1f}<extra></extra>"
            ))
            y_order = plot_df["account_id"].astype(str).tolist()

        fig.update_layout(
            height=max(300, len(plot_df) * 36),
            showlegend=has_plan,
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=11)),
            xaxis=dict(showgrid=False, tickfont=dict(size=11),
                       title="Risk Percentile", range=[0, 125]),
            yaxis=dict(showgrid=False, tickfont=dict(size=11),
                       categoryorder="array", categoryarray=y_order),
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
            is_llm = out.get("matched_query") == "llm_generated"
            badge  = ' <span style="font-size:0.72rem;font-weight:700;background:#EFF6FF;color:#1D4ED8;padding:2px 9px;border-radius:20px;margin-left:8px;vertical-align:middle;">AI Generated</span>' if is_llm else ""
            st.markdown(
                f'<div class="result-card"><div class="result-title">{st.session_state.active_label}{badge}</div></div>',
                unsafe_allow_html=True)

            # Show generated SQL for LLM results
            if is_llm and out.get("sql"):
                with st.expander("View generated SQL", expanded=False):
                    st.code(out["sql"], language="sql")

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
            Select a preset question from the panel, or type any question above — the AI will write the SQL for you.<br><br>
            <span style="font-size:0.85rem;color:#64748B;">
            Try: "Which Enterprise accounts in EU haven't logged in for 30 days?"<br>
            Or: "How many Pro accounts recovered from high risk last week?"
            </span>
        </div>
        """, unsafe_allow_html=True)
