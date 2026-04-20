from sqlalchemy import create_engine, text
import pandas as pd
import os

_engine = None

def _get_url():
    # Option 1: full URL from environment
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return "".join(url.split())

    try:
        import streamlit as st
        s = st.secrets

        # Option 2: full URL stored as one secret
        if "DATABASE_URL" in s:
            return "".join(str(s["DATABASE_URL"]).split())

        # Option 3: individual parts (avoids long-line wrapping in Streamlit UI)
        if "NEON_HOST" in s:
            host = str(s["NEON_HOST"]).strip()
            user = str(s["NEON_USER"]).strip()
            pw   = str(s["NEON_PASS"]).strip()
            db   = str(s["NEON_DB"]).strip()
            return f"postgresql+psycopg2://{user}:{pw}@{host}/{db}?sslmode=require"

    except Exception as e:
        raise RuntimeError(f"Failed to read secrets: {e}")

    raise RuntimeError("DATABASE_URL is not set. Add NEON_HOST/NEON_USER/NEON_PASS/NEON_DB to Streamlit Cloud secrets.")

def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_get_url())
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