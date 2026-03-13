from sqlalchemy import create_engine, text

DB_USER = "churn"
DB_PASSWORD = "churn"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "churn_db"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """))
    print("Tables:")
    for row in result:
        print("-", row[0])