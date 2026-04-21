# SaaS Churn Copilot

A full-stack B2B SaaS churn analytics platform — from raw event data to a production ML model to an AI-powered natural language interface.

Built as a portfolio project to demonstrate end-to-end data/ML engineering: feature engineering, model training and calibration, a weekly scoring pipeline, SHAP explainability, and a Claude-powered text-to-SQL copilot.

---

## What it does

Customer Success and RevOps teams use the app to answer one question every week: **which accounts are about to churn, and why?**

- **Dashboard** — KPI cards (high-risk count, revenue at risk, newly at-risk this week), trend charts, and a top-5 accounts preview
- **Ask Copilot** — preset analytical queries (risk by plan, region, contract) plus a free-form AI interface: type any business question in plain English and Claude writes the SQL
- **Account Explainability** — click any account to see a plain-English breakdown of the top 5 model drivers (sessions, active users, revenue, tenure, etc.) with a HIGH / MEDIUM / LOW risk verdict

---

## Tech stack

| Layer | Tools |
|---|---|
| Data ingestion & ETL | Python, Pandas |
| Feature engineering | Rolling 7/14/30-day aggregates, trend features, session variability |
| ML model | XGBoost with early stopping, Platt sigmoid calibration |
| Explainability | SHAP TreeExplainer — top-5 drivers per account |
| Database | PostgreSQL (Neon serverless) via SQLAlchemy |
| LLM copilot | Anthropic Claude (Haiku) — text-to-SQL with schema-scoped system prompt |
| Frontend | Streamlit with Plotly charts |
| Deployment | Streamlit Cloud |

---

## Project structure

```
etl/                     Raw data ingestion and B2B mart construction
ml/
  train_churn_30d.py     XGBoost model training with time-split validation
  production_score_30d.py  Score all accounts + compute SHAP values
  production_score_percentiles.py  Assign risk percentile, band, bucket
pipeline/
  run_weekly_refresh.py  End-to-end weekly pipeline (score → rank → load)
db/
  load_marts_to_postgres.py  Load all CSVs into Neon Postgres
copilot/
  query_router.py        Preset SQL query library (30+ queries)
  llm_sql_router.py      Claude text-to-SQL router with safety checks
  query_db.py            SQLAlchemy connection layer
app/
  Home.py                Dashboard with KPI cards and charts
  pages/02_Ask_Copilot.py  Copilot UI with SHAP explainability cards
```

---

## ML approach

**Problem:** Predict which accounts will churn in the next 30 days.

**Features:** 18 engineered features including rolling session counts (7/14/30d), active user counts, in-product events, revenue, days since last activity, week-over-week session drop, and session trend (7d vs 30d baseline).

**Model:** XGBoost classifier with `scale_pos_weight` for class imbalance, early stopping on validation PR-AUC, and Platt sigmoid calibration to map overconfident raw scores to honest probabilities.

**Output:** Each account gets a calibrated churn probability, a risk percentile (0–100), a risk band (Top 1%/5%/10%/25%/Rest), and a risk bucket (High/Medium/Low). High = top 5% of accounts by risk.

**Explainability:** SHAP TreeExplainer computes per-account feature attributions. The top 5 positive drivers are stored and surfaced in the UI as plain-English cards — e.g. "~1 active user in the past 30 days, 94% below average."

---

## Running locally

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your database credentials to `.streamlit/secrets.toml`:
   ```toml
   DATABASE_URL = "postgresql+psycopg2://..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

3. Run the weekly pipeline to score accounts:
   ```bash
   python pipeline/run_weekly_refresh.py
   ```

4. Launch the app:
   ```bash
   streamlit run app/Home.py
   ```

---

## StreamLit Cloud link: https://saas-churn-copilot-4zercogks3sgf8xvdwzvbc.streamlit.app/
