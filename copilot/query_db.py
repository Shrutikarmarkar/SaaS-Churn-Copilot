from sqlalchemy import create_engine, text
import pandas as pd
import os

_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            try:
                import streamlit as st
                url = st.secrets.get("DATABASE_URL")
            except Exception:
                pass
        if not url:
            raise RuntimeError("DATABASE_URL is not set. Add it to .streamlit/secrets.toml or as an environment variable.")
        _engine = create_engine(url)
    return _engine

def run_query(sql: str) -> pd.DataFrame:
    with _get_engine().connect() as conn:
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