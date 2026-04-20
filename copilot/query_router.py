from query_db import run_query

QUERY_MAP = {

    # ── Risk Overview ────────────────────────────────────────────────────────

    "top_risk_accounts": {
        "keywords": ["top high risk accounts", "top risk accounts", "highest risk accounts",
                     "show high risk accounts", "show risk accounts", "show top accounts"],
        "sql_template": """
            SELECT c.account_id, d.plan_type, d.region, d.contract_type,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band, c.risk_bucket
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            ORDER BY c.risk_percentile DESC
            LIMIT {limit};
        """,
        "default_limit": 10
    },

    "risk_bucket_distribution": {
        "keywords": ["risk distribution by bucket", "distribution by bucket",
                     "risk bucket distribution", "show risk distribution", "bucket distribution",
                     "how many accounts in each bucket"],
        "sql": """
            SELECT risk_bucket,
                   COUNT(*) AS account_count,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
            FROM churn_scores_latest_ranked
            GROUP BY risk_bucket
            ORDER BY account_count DESC;
        """
    },

    "risk_band_distribution": {
        "keywords": ["risk band distribution", "distribution by band", "show risk bands",
                     "accounts by risk band", "how many in each band"],
        "sql": """
            SELECT risk_band,
                   COUNT(*) AS account_count,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
            FROM churn_scores_latest_ranked
            GROUP BY risk_band
            ORDER BY account_count DESC;
        """
    },

    "high_risk_count": {
        "keywords": ["how many accounts are high risk", "count high risk accounts",
                     "number of high risk accounts", "total high risk accounts"],
        "sql": """
            SELECT COUNT(*) AS high_risk_accounts
            FROM churn_scores_latest_ranked
            WHERE risk_bucket = 'High';
        """
    },

    "top_1_percent": {
        "keywords": ["top 1% accounts", "top 1 percent", "how many in top 1%",
                     "how many accounts are in the top 1%", "top one percent"],
        "sql": """
            SELECT c.account_id, d.plan_type, d.region, d.contract_type,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE c.risk_band = 'Top 1%'
            ORDER BY c.risk_percentile DESC;
        """
    },

    "top_5_percent": {
        "keywords": ["top 5% accounts", "top 5 percent", "how many in top 5%",
                     "how many accounts are in the top 5%", "top five percent"],
        "sql": """
            SELECT c.account_id, d.plan_type, d.region, d.contract_type,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE c.risk_band IN ('Top 1%', 'Top 5%')
            ORDER BY c.risk_percentile DESC;
        """
    },

    "top_10_percent": {
        "keywords": ["top 10% accounts", "top 10 percent", "how many in top 10%",
                     "accounts in top 10%", "top ten percent"],
        "sql": """
            SELECT c.account_id, d.plan_type, d.region, d.contract_type,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE c.risk_band IN ('Top 1%', 'Top 5%', 'Top 10%')
            ORDER BY c.risk_percentile DESC;
        """
    },

    # ── By Plan ──────────────────────────────────────────────────────────────

    "avg_risk_by_plan": {
        "keywords": ["average churn risk by plan", "avg risk by plan", "risk by plan",
                     "average risk by plan", "plan risk", "churn by plan"],
        "sql": """
            SELECT d.plan_type,
                   COUNT(*)                                            AS total_accounts,
                   SUM(CASE WHEN c.risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_accounts,
                   ROUND(AVG(c.risk_percentile)::numeric, 1)          AS avg_risk_percentile,
                   ROUND(AVG(c.churn_risk_calibrated)::numeric, 4)    AS avg_churn_probability
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            GROUP BY d.plan_type
            ORDER BY avg_risk_percentile DESC;
        """
    },

    "high_risk_by_plan": {
        "keywords": ["high risk by plan", "which plan has the most high risk",
                     "high risk accounts by plan", "most high risk accounts by plan",
                     "plan with most churn risk"],
        "sql": """
            SELECT d.plan_type,
                   COUNT(*) AS high_risk_accounts
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE c.risk_bucket = 'High'
            GROUP BY d.plan_type
            ORDER BY high_risk_accounts DESC;
        """
    },

    "enterprise_high_risk": {
        "keywords": ["enterprise high risk", "high risk enterprise accounts",
                     "enterprise accounts at risk", "how many enterprise are high risk",
                     "enterprise churn risk"],
        "sql": """
            SELECT c.account_id, d.region, d.contract_type, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.plan_type = 'Enterprise'
              AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    "pro_high_risk": {
        "keywords": ["pro high risk", "high risk pro accounts", "pro accounts at risk",
                     "how many pro are high risk", "pro plan churn risk"],
        "sql": """
            SELECT c.account_id, d.region, d.contract_type, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.plan_type = 'Pro'
              AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    "basic_high_risk": {
        "keywords": ["basic high risk", "high risk basic accounts", "basic accounts at risk",
                     "how many basic are high risk", "basic plan churn risk"],
        "sql": """
            SELECT c.account_id, d.region, d.contract_type, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.plan_type = 'Basic'
              AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    # ── By Region ────────────────────────────────────────────────────────────

    "avg_risk_by_region": {
        "keywords": ["average churn risk by region", "avg risk by region", "risk by region",
                     "average risk by region", "region risk", "churn by region"],
        "sql": """
            SELECT d.region,
                   COUNT(*)                                            AS total_accounts,
                   SUM(CASE WHEN c.risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_accounts,
                   ROUND(AVG(c.risk_percentile)::numeric, 1)          AS avg_risk_percentile,
                   ROUND(AVG(c.churn_risk_calibrated)::numeric, 4)    AS avg_churn_probability
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            GROUP BY d.region
            ORDER BY avg_risk_percentile DESC;
        """
    },

    "high_risk_by_region": {
        "keywords": ["high risk by region", "which region has the most high risk",
                     "high risk accounts by region", "most high risk accounts by region",
                     "region with most churn risk"],
        "sql": """
            SELECT d.region,
                   COUNT(*) AS high_risk_accounts
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE c.risk_bucket = 'High'
            GROUP BY d.region
            ORDER BY high_risk_accounts DESC;
        """
    },

    "high_risk_na": {
        "keywords": ["high risk accounts in na", "high risk na accounts",
                     "north america high risk", "na churn risk", "at risk accounts in na"],
        "sql": """
            SELECT c.account_id, d.plan_type, d.contract_type, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.region = 'NA' AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    "high_risk_eu": {
        "keywords": ["high risk accounts in eu", "high risk eu accounts",
                     "europe high risk", "eu churn risk", "at risk accounts in eu"],
        "sql": """
            SELECT c.account_id, d.plan_type, d.contract_type, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.region = 'EU' AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    "high_risk_apac": {
        "keywords": ["high risk accounts in apac", "high risk apac accounts",
                     "apac high risk", "apac churn risk", "at risk accounts in apac"],
        "sql": """
            SELECT c.account_id, d.plan_type, d.contract_type, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.region = 'APAC' AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    # ── By Contract ──────────────────────────────────────────────────────────

    "avg_risk_by_contract": {
        "keywords": ["risk by contract", "average risk by contract type",
                     "churn by contract", "monthly vs annual risk",
                     "contract type risk", "risk by contract type"],
        "sql": """
            SELECT d.contract_type,
                   COUNT(*)                                            AS total_accounts,
                   SUM(CASE WHEN c.risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_accounts,
                   ROUND(AVG(c.risk_percentile)::numeric, 1)          AS avg_risk_percentile,
                   ROUND(AVG(c.churn_risk_calibrated)::numeric, 4)    AS avg_churn_probability
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            GROUP BY d.contract_type
            ORDER BY avg_risk_percentile DESC;
        """
    },

    "monthly_high_risk": {
        "keywords": ["monthly contract high risk", "high risk monthly accounts",
                     "monthly accounts at risk", "monthly churn risk",
                     "which monthly accounts are high risk"],
        "sql": """
            SELECT c.account_id, d.plan_type, d.region, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.contract_type = 'Monthly' AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    "annual_high_risk": {
        "keywords": ["annual contract high risk", "high risk annual accounts",
                     "annual accounts at risk", "annual churn risk",
                     "which annual accounts are high risk"],
        "sql": """
            SELECT c.account_id, d.plan_type, d.region, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.contract_type = 'Annual' AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    # ── Segments (cross-cuts) ────────────────────────────────────────────────

    "risk_by_plan_and_region": {
        "keywords": ["risk by plan and region", "plan and region breakdown",
                     "churn by plan and region", "plan region risk matrix",
                     "segment breakdown by plan and region"],
        "sql": """
            SELECT d.plan_type, d.region,
                   COUNT(*)                                            AS total_accounts,
                   SUM(CASE WHEN c.risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_accounts,
                   ROUND(AVG(c.risk_percentile)::numeric, 1)          AS avg_risk_percentile
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            GROUP BY d.plan_type, d.region
            ORDER BY avg_risk_percentile DESC;
        """
    },

    "risk_by_plan_and_contract": {
        "keywords": ["risk by plan and contract", "plan and contract breakdown",
                     "churn by plan and contract type", "plan contract risk"],
        "sql": """
            SELECT d.plan_type, d.contract_type,
                   COUNT(*)                                            AS total_accounts,
                   SUM(CASE WHEN c.risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_accounts,
                   ROUND(AVG(c.risk_percentile)::numeric, 1)          AS avg_risk_percentile
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            GROUP BY d.plan_type, d.contract_type
            ORDER BY avg_risk_percentile DESC;
        """
    },

    "enterprise_monthly_high_risk": {
        "keywords": ["enterprise monthly high risk", "high risk enterprise monthly",
                     "enterprise monthly accounts at risk", "enterprise monthly churn"],
        "sql": """
            SELECT c.account_id, d.region, d.seats,
                   ROUND(c.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(c.risk_percentile::numeric, 1)       AS risk_percentile,
                   c.risk_band
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            WHERE d.plan_type = 'Enterprise' AND d.contract_type = 'Monthly'
              AND c.risk_bucket = 'High'
            ORDER BY c.risk_percentile DESC;
        """
    },

    # ── Revenue at Risk ──────────────────────────────────────────────────────

    "revenue_at_risk": {
        "keywords": ["revenue at risk", "total revenue at risk", "how much revenue is at risk",
                     "revenue from high risk accounts", "at risk revenue"],
        "sql": """
            SELECT
                ROUND(SUM(f.revenue)::numeric, 2)  AS total_revenue_at_risk,
                COUNT(DISTINCT c.account_id)        AS high_risk_accounts
            FROM churn_scores_latest_ranked c
            JOIN fact_usage_daily_account f ON c.account_id = f.account_id
            WHERE c.risk_bucket = 'High';
        """
    },

    "revenue_at_risk_by_plan": {
        "keywords": ["revenue at risk by plan", "plan revenue at risk",
                     "which plan has most revenue at risk", "revenue risk by plan"],
        "sql": """
            SELECT d.plan_type,
                   COUNT(DISTINCT c.account_id)        AS high_risk_accounts,
                   ROUND(SUM(f.revenue)::numeric, 2)   AS total_revenue_at_risk
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            JOIN fact_usage_daily_account f ON c.account_id = f.account_id
            WHERE c.risk_bucket = 'High'
            GROUP BY d.plan_type
            ORDER BY total_revenue_at_risk DESC;
        """
    },

    "revenue_at_risk_by_region": {
        "keywords": ["revenue at risk by region", "region revenue at risk",
                     "which region has most revenue at risk", "revenue risk by region"],
        "sql": """
            SELECT d.region,
                   COUNT(DISTINCT c.account_id)        AS high_risk_accounts,
                   ROUND(SUM(f.revenue)::numeric, 2)   AS total_revenue_at_risk
            FROM churn_scores_latest_ranked c
            JOIN dim_account d ON c.account_id = d.account_id
            JOIN fact_usage_daily_account f ON c.account_id = f.account_id
            WHERE c.risk_bucket = 'High'
            GROUP BY d.region
            ORDER BY total_revenue_at_risk DESC;
        """
    },

    # ── Weekly Trends ────────────────────────────────────────────────────────

    "high_risk_trend": {
        "keywords": ["high risk trend", "churn trend over time", "risk trend over time",
                     "high risk over time", "weekly high risk trend", "churn trend by week"],
        "sql": """
            SELECT run_date,
                   COUNT(*) AS high_risk_accounts
            FROM churn_scores_history
            WHERE risk_bucket = 'High'
            GROUP BY run_date
            ORDER BY run_date;
        """
    },

    "avg_risk_trend": {
        "keywords": ["average risk trend over time", "avg risk trend", "risk score trend",
                     "how has average risk changed", "churn probability trend"],
        "sql": """
            SELECT run_date,
                   ROUND(AVG(risk_percentile)::numeric, 1)       AS avg_risk_percentile,
                   ROUND(AVG(churn_risk_calibrated)::numeric, 4) AS avg_churn_probability
            FROM churn_scores_history
            GROUP BY run_date
            ORDER BY run_date;
        """
    },

    "week_over_week_summary": {
        "keywords": ["week over week", "week on week", "compared to last week",
                     "change since last week", "high risk this week vs last week",
                     "weekly comparison", "what changed this week"],
        "sql": """
            WITH runs AS (
                SELECT DISTINCT run_date FROM churn_scores_history
                ORDER BY run_date DESC LIMIT 2
            ),
            this_week AS (
                SELECT COUNT(*) AS count FROM churn_scores_history
                WHERE risk_bucket = 'High'
                  AND run_date = (SELECT MAX(run_date) FROM runs)
            ),
            last_week AS (
                SELECT COUNT(*) AS count FROM churn_scores_history
                WHERE risk_bucket = 'High'
                  AND run_date = (SELECT MIN(run_date) FROM runs)
            )
            SELECT
                (SELECT MAX(run_date) FROM runs) AS this_week,
                (SELECT MIN(run_date) FROM runs) AS last_week,
                this_week.count                 AS high_risk_this_week,
                last_week.count                 AS high_risk_last_week,
                this_week.count - last_week.count AS change
            FROM this_week, last_week;
        """
    },

    "new_high_risk_accounts": {
        "keywords": ["new high risk accounts", "newly high risk", "new at risk accounts",
                     "accounts that became high risk", "which accounts are newly at risk",
                     "newly at risk this week", "new churn risk accounts"],
        "sql": """
            WITH latest_run AS (SELECT MAX(run_date) AS run_date FROM churn_scores_history),
            prev_run AS (
                SELECT MAX(run_date) AS run_date FROM churn_scores_history
                WHERE run_date < (SELECT run_date FROM latest_run)
            )
            SELECT h.account_id, d.plan_type, d.region, d.contract_type,
                   ROUND(h.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(h.risk_percentile::numeric, 1)       AS risk_percentile,
                   h.risk_band
            FROM churn_scores_history h
            JOIN dim_account d ON h.account_id = d.account_id
            JOIN latest_run lr ON h.run_date = lr.run_date
            WHERE h.risk_bucket = 'High'
              AND h.account_id NOT IN (
                SELECT account_id FROM churn_scores_history
                WHERE run_date = (SELECT run_date FROM prev_run)
                  AND risk_bucket = 'High'
              )
            ORDER BY h.risk_percentile DESC;
        """
    },

    # ── Account Explainability ───────────────────────────────────────────────

    "explain_account": {
        "keywords": ["explain", "why is", "why high risk", "what makes", "risk drivers for",
                     "drivers for", "why is this account", "explain account"],
        "sql_template": """
            SELECT feature_label AS driver, feature_name, shap_value,
                   feature_value AS value, pop_avg, driver_rank
            FROM account_shap_drivers
            WHERE account_id = '{account_id}'
            ORDER BY driver_rank;
        """
    },

    "recovered_accounts": {
        "keywords": ["recovered accounts", "accounts no longer high risk",
                     "accounts that left high risk", "improved accounts",
                     "accounts that recovered", "accounts dropped from high risk"],
        "sql": """
            WITH latest_run AS (SELECT MAX(run_date) AS run_date FROM churn_scores_history),
            prev_run AS (
                SELECT MAX(run_date) AS run_date FROM churn_scores_history
                WHERE run_date < (SELECT run_date FROM latest_run)
            )
            SELECT h.account_id, d.plan_type, d.region,
                   ROUND(h.churn_risk_calibrated::numeric, 4) AS churn_probability,
                   ROUND(h.risk_percentile::numeric, 1)       AS risk_percentile,
                   h.risk_bucket AS current_bucket
            FROM churn_scores_history h
            JOIN dim_account d ON h.account_id = d.account_id
            JOIN latest_run lr ON h.run_date = lr.run_date
            WHERE h.risk_bucket != 'High'
              AND h.account_id IN (
                SELECT account_id FROM churn_scores_history
                WHERE run_date = (SELECT run_date FROM prev_run)
                  AND risk_bucket = 'High'
              )
            ORDER BY h.risk_percentile DESC;
        """
    },
}

# ── Routing ──────────────────────────────────────────────────────────────────

def extract_limit(question: str, default_limit: int = 10) -> int:
    words = question.lower().replace("%", " % ").split()
    for w in words:
        if w.isdigit():
            val = int(w)
            if 1 <= val <= 200:
                return val
    return default_limit

def route_question(question: str):
    q = question.lower().strip()
    best_match = None
    best_score = 0

    for query_name, config in QUERY_MAP.items():
        score = 0
        for kw in config["keywords"]:
            if kw in q:
                score += len(kw.split())
        if score > best_score:
            best_score = score
            best_match = query_name

    return best_match if best_score > 0 else None

def answer_question(question: str) -> dict:
    match = route_question(question)

    if match is None:
        return {
            "matched_query": None,
            "sql": None,
            "result": None,
            "message": "Sorry, I couldn't match that question. Try one of the preset questions above."
        }

    config = QUERY_MAP[match]

    if "sql_template" in config:
        limit = extract_limit(question, config.get("default_limit", 10))
        sql = config["sql_template"].format(limit=limit)
    else:
        sql = config["sql"]

    df = run_query(sql)

    return {
        "matched_query": match,
        "sql": sql,
        "result": df,
        "message": "Success"
    }

def answer_direct(query_name: str, params: dict = {}) -> dict:
    """Called by buttons — bypasses keyword matching, goes straight to the query."""
    if query_name not in QUERY_MAP:
        return {"matched_query": None, "sql": None, "result": None,
                "message": f"Unknown query: {query_name}"}

    config = QUERY_MAP[query_name]

    if "sql_template" in config:
        fmt = {"limit": params.get("limit", config.get("default_limit", 10))}
        # Pass through any extra string params (e.g. account_id) — strip to prevent injection
        for k, v in params.items():
            if k != "limit":
                fmt[k] = str(v).strip().replace("'", "")
        sql = config["sql_template"].format(**fmt)
    else:
        sql = config["sql"]

    df = run_query(sql)
    return {"matched_query": query_name, "sql": sql, "result": df, "message": "Success"}
