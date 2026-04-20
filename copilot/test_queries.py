from query_db import run_query

queries = {
    "top_10_high_risk": """
        SELECT account_id, snapshot_date, churn_risk_calibrated, risk_percentile, risk_band, risk_bucket
        FROM churn_scores_latest_ranked
        ORDER BY risk_percentile DESC
        LIMIT 10;
    """,

    "avg_risk_by_region": """
        SELECT d.region,
               COUNT(*) AS account_count,
               ROUND(AVG(c.risk_percentile)::numeric, 1) AS avg_risk_percentile,
               ROUND(AVG(c.churn_risk_calibrated)::numeric, 4) AS avg_churn_probability
        FROM churn_scores_latest_ranked c
        JOIN dim_account d
          ON c.account_id = d.account_id
        GROUP BY d.region
        ORDER BY avg_risk_percentile DESC;
    """,

    "high_risk_by_plan": """
        SELECT d.plan_type,
               COUNT(*) AS high_risk_accounts
        FROM churn_scores_latest_ranked c
        JOIN dim_account d
          ON c.account_id = d.account_id
        WHERE c.risk_bucket = 'High'
        GROUP BY d.plan_type
        ORDER BY high_risk_accounts DESC;
    """,

    "top_5_percent_count": """
        SELECT COUNT(*) AS top_5_percent_accounts
        FROM churn_scores_latest_ranked
        WHERE risk_band IN ('Top 1%', 'Top 5%');
    """
}

if __name__ == "__main__":
    for name, sql in queries.items():
        print(f"\n===== {name} =====")
        df = run_query(sql)
        print(df)