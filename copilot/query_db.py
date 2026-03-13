from sqlalchemy import create_engine, text
import pandas as pd

DB_USER = "churn"
DB_PASSWORD = "churn"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "churn_db"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def run_query(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

if __name__ == "__main__":
    sql = """
    SELECT account_id, snapshot_date, churn_risk_score, risk_percentile, risk_band, risk_bucket
    FROM churn_scores_latest_ranked
    ORDER BY churn_risk_score DESC
    LIMIT 10;
    """
    df = run_query(sql)
    print(df)