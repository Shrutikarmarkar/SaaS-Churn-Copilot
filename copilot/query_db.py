from sqlalchemy import create_engine, text
import pandas as pd
import os

try:
    import streamlit as st
    DATABASE_URL = st.secrets.get("DATABASE_URL") or os.environ.get("DATABASE_URL")
except Exception:
    DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add it to .streamlit/secrets.toml or as an environment variable.")

engine = create_engine(DATABASE_URL)

def run_query(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

if __name__ == "__main__":
    sql = """
    SELECT account_id, snapshot_date, churn_risk_calibrated, risk_percentile, risk_band, risk_bucket
    FROM churn_scores_latest_ranked
    ORDER BY risk_percentile DESC
    LIMIT 10;
    """
    df = run_query(sql)
    print(df)