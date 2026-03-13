import os
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MART_DIR = os.path.join(BASE_DIR, "data", "mart")

DB_USER = "churn"
DB_PASSWORD = "churn"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "churn_db"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

FILES_TO_TABLES = {
    "dim_account.csv": "dim_account",
    "dim_user.csv": "dim_user",
    "fact_usage_daily_account.csv": "fact_usage_daily_account",
    "fact_churn_account.csv": "fact_churn_account",
    "churn_scores_latest_ranked.csv": "churn_scores_latest_ranked",
}

def main():
    print("Connecting to Postgres...")
    engine = create_engine(DATABASE_URL)

    for filename, table_name in FILES_TO_TABLES.items():
        path = os.path.join(DATA_MART_DIR, filename)
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        print(f"\nLoading {filename} -> {table_name}")
        df = pd.read_csv(path)

        # Try parsing likely date columns
        for col in df.columns:
            if "date" in col.lower() or "day" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        print("Shape:", df.shape)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Loaded table: {table_name}")

    print("\nAll available marts loaded into Postgres.")

if __name__ == "__main__":
    main()