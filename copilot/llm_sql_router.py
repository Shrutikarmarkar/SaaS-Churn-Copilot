import os
import re
import anthropic
from query_db import run_query

def _get_api_key() -> str | None:
    # Local: set via export ANTHROPIC_API_KEY=...
    # Streamlit Cloud: set via the Secrets dashboard
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None

# ── Schema context (cached in the system prompt) ─────────────────────────────
SCHEMA = """
TABLE: churn_scores_latest_ranked
  account_id             TEXT       -- unique account identifier
  snapshot_date          TIMESTAMP  -- date of the account's latest data snapshot
  churn_risk_calibrated  FLOAT      -- calibrated churn probability (0 to 1); e.g. 0.08 means ~8% chance of churning in 30 days
  risk_percentile        FLOAT      -- percentile rank 0-100; higher = riskier relative to all accounts
  risk_band              TEXT       -- 'Top 1%', 'Top 5%', 'Top 10%', 'Top 25%', 'Rest'
  risk_bucket            TEXT       -- 'High', 'Medium', 'Low'  (High = top 5%, Medium = top 25%, Low = rest)

TABLE: churn_scores_history
  run_date               DATE       -- date the weekly pipeline ran
  account_id             TEXT
  snapshot_date          TIMESTAMP
  churn_risk_calibrated  FLOAT
  risk_percentile        FLOAT
  risk_band              TEXT
  risk_bucket            TEXT

TABLE: dim_account
  account_id      TEXT    -- joins to churn_scores tables
  plan_type       TEXT    -- 'Basic', 'Pro', 'Enterprise'
  contract_type   TEXT    -- 'Monthly', 'Annual'
  region          TEXT    -- 'NA', 'EU', 'APAC'
  seats           INT     -- number of licensed seats

TABLE: dim_user
  user_id     INT
  account_id  TEXT  -- joins to dim_account

TABLE: fact_usage_daily_account
  account_id    TEXT
  event_day     TIMESTAMP
  active_users  INT
  sessions      INT
  events        INT
  revenue       FLOAT

TABLE: fact_churn_account
  account_id     TEXT
  last_active_day  TIMESTAMP
  churn_date     TIMESTAMP
  churn_flag     INT        -- 1 = churned, 0 = active
  plan_type      TEXT
  contract_type  TEXT
  region         TEXT
  seats          INT
"""

SYSTEM_PROMPT = f"""You are a SQL analyst for a B2B SaaS churn analytics platform.
Your job is to translate a business question into a single valid PostgreSQL SELECT query.

{SCHEMA}

RULES:
1. Return ONLY the SQL query — no explanation, no markdown fences, no commentary.
2. Only generate SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, TRUNCATE, or any mutation.
3. Use exact column and table names from the schema above.
4. "High risk" accounts means risk_bucket = 'High'.
5. "Top N accounts" means ORDER BY risk_percentile DESC LIMIT N.
6. To filter or group by plan, region, or contract_type, JOIN churn_scores_latest_ranked with dim_account ON account_id.
7. For week-over-week or trend questions, use churn_scores_history (has run_date column).
8. For "new high risk this week" — accounts in High bucket this week that were not High last week.
9. Round floats to 4 decimal places where appropriate using ROUND(x::numeric, 4).
10. If the question cannot be answered from the available schema, return:
    SELECT 'I cannot answer this question from the available data.' AS message;"""

def _extract_sql(text: str) -> str:
    # Strip markdown fences if Claude adds them despite instructions
    text = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text

def _is_safe(sql: str) -> bool:
    forbidden = r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke)\b"
    return not re.search(forbidden, sql, re.IGNORECASE)

def answer_question_llm(question: str) -> dict:
    api_key = _get_api_key()
    if not api_key:
        return {
            "matched_query": None,
            "sql": None,
            "result": None,
            "message": "ANTHROPIC_API_KEY is not set. Set it as an environment variable to enable the LLM copilot."
        }

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": question}
            ],
        )
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower():
            return {
                "matched_query": None,
                "sql": None,
                "result": None,
                "message": "Anthropic account has no credits. Go to console.anthropic.com → Plans & Billing to add credits."
            }
        return {"matched_query": None, "sql": None, "result": None, "message": f"API error: {e}"}
    except anthropic.APIError as e:
        return {"matched_query": None, "sql": None, "result": None, "message": f"API error: {e}"}

    raw = response.content[0].text.strip()
    sql = _extract_sql(raw)

    if not _is_safe(sql):
        return {
            "matched_query": "llm_generated",
            "sql": sql,
            "result": None,
            "message": "Generated query contains unsafe operations and was blocked."
        }

    try:
        df = run_query(sql)
        return {
            "matched_query": "llm_generated",
            "sql": sql,
            "result": df,
            "message": "Success"
        }
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower():
            return {
                "matched_query": "llm_generated",
                "sql": None,
                "result": None,
                "message": "Anthropic account has no credits. Go to console.anthropic.com → Plans & Billing to add credits."
            }
        return {"matched_query": "llm_generated", "sql": sql, "result": None, "message": f"API error: {e}"}
    except Exception as e:
        return {
            "matched_query": "llm_generated",
            "sql": sql,
            "result": None,
            "message": f"SQL error: {e}"
        }
