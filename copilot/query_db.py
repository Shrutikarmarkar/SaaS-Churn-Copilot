from sqlalchemy import create_engine, text
import pandas as pd
import os

_engine = None

def _get_url():
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return "".join(url.split())
    try:
        import streamlit as st
        raw = st.secrets["DATABASE_URL"]
        return "".join(str(raw).split())
    except KeyError:
        raise RuntimeError("DATABASE_URL not found in st.secrets. Check the key name in Streamlit Cloud secrets.")
    except Exception:
        pass
    raise RuntimeError("DATABASE_URL is not set. Add it to Streamlit Cloud secrets or as an environment variable.")

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