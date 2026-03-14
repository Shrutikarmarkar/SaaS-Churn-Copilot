from query_db import run_query

QUERY_MAP = {
    "top_10_high_risk": {
        "keywords": ["top", "high risk", "risky accounts", "top accounts", "highest risk"],
        "sql": """
            SELECT account_id, snapshot_date, churn_risk_score, risk_percentile, risk_band, risk_bucket
            FROM churn_scores_latest_ranked
            ORDER BY churn_risk_score DESC
            LIMIT 10;
        """
    },
    "avg_risk_by_region": {
        "keywords": ["region", "by region", "average risk", "avg risk by region"],
        "sql": """
            SELECT d.region,
                   COUNT(*) AS account_count,
                   AVG(c.churn_risk_score) AS avg_churn_risk
            FROM churn_scores_latest_ranked c
            JOIN dim_account d
              ON c.account_id = d.account_id
            GROUP BY d.region
            ORDER BY avg_churn_risk DESC;
        """
    },
    "high_risk_by_plan": {
        "keywords": ["plan", "plan type", "high risk by plan", "which plan"],
        "sql": """
            SELECT d.plan_type,
                   COUNT(*) AS high_risk_accounts
            FROM churn_scores_latest_ranked c
            JOIN dim_account d
              ON c.account_id = d.account_id
            WHERE c.risk_bucket = 'High'
            GROUP BY d.plan_type
            ORDER BY high_risk_accounts DESC;
        """
    },
    "top_5_percent_count": {
        "keywords": ["top 5", "top 5%", "top five percent", "how many top 5"],
        "sql": """
            SELECT COUNT(*) AS top_5_percent_accounts
            FROM churn_scores_latest_ranked
            WHERE risk_band IN ('Top 1%', 'Top 5%');
        """
    }
}

def route_question(question: str):
    q = question.lower()

    best_match = None
    best_score = 0

    for query_name, config in QUERY_MAP.items():
        score = 0
        for kw in config["keywords"]:
            if kw in q:
                score += 1
        if score > best_score:
            best_score = score
            best_match = query_name

    return best_match

def answer_question(question: str):
    match = route_question(question)

    if match is None:
        return {
            "matched_query": None,
            "sql": None,
            "result": None,
            "message": "Sorry, I couldn't match that question to a known analytics query yet."
        }

    sql = QUERY_MAP[match]["sql"]
    df = run_query(sql)

    return {
        "matched_query": match,
        "sql": sql,
        "result": df,
        "message": "Success"
    }

if __name__ == "__main__":
    while True:
        question = input("\nAsk a churn analytics question (or type 'exit'): ").strip()
        if question.lower() == "exit":
            break

        output = answer_question(question)
        print("\nMatched query:", output["matched_query"])
        print("\nSQL:")
        print(output["sql"])
        print("\nResult:")
        print(output["result"])