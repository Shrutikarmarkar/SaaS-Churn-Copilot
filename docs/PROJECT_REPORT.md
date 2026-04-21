# SaaS Churn Copilot — End-to-End Project Report

**Live Application:** https://saas-churn-copilot-4zercogks3sgf8xvdwzvbc.streamlit.app/
**GitHub Repository:** SaaS-Churn-Copilot
**Author:** Shruti Karmarkar — NYU

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Problem](#2-business-problem)
3. [Dataset and Data Design](#3-dataset-and-data-design)
4. [Data Pipeline and ETL](#4-data-pipeline-and-etl)
5. [Feature Engineering](#5-feature-engineering)
6. [Machine Learning Model](#6-machine-learning-model)
7. [Model Calibration](#7-model-calibration)
8. [SHAP Explainability](#8-shap-explainability)
9. [Production Scoring Pipeline](#9-production-scoring-pipeline)
10. [Database Layer](#10-database-layer)
11. [Application and UI](#11-application-and-ui)
12. [LLM Copilot](#12-llm-copilot)
13. [Tech Stack](#13-tech-stack)
14. [Key Design Decisions and Trade-offs](#14-key-design-decisions-and-trade-offs)
15. [Deliverables Summary](#15-deliverables-summary)

---

## 1. Executive Summary

SaaS Churn Copilot is a full-stack, end-to-end churn intelligence platform built for a B2B SaaS company. It takes raw transactional event data, transforms it into a structured account-level data model, trains and calibrates a machine learning model to predict 30-day churn risk, and surfaces the results through a production-grade analytics dashboard powered by an AI natural language interface.

The platform answers the core question that every Customer Success and Revenue Operations team faces each week: **which accounts are most likely to cancel, and why?** Rather than returning a raw probability score, the system assigns each account a calibrated churn probability, a relative risk percentile, and a plain-English explanation of the top five model drivers — enabling teams to prioritize outreach with confidence.

---

## 2. Business Problem

### Context

In subscription-based B2B SaaS businesses, customer churn is the primary threat to revenue growth. Unlike B2C churn (which is often impulsive), B2B churn follows a predictable pattern: engagement declines weeks before a cancellation decision is made. A team that can detect disengagement signals early — and act before the renewal conversation — can meaningfully reduce churn.

### Problem Statement

> **Given the usage and engagement history of every account in the portfolio, predict which accounts will churn in the next 30 days — and explain the specific behavioral signals driving each account's risk.**

### Why 30 Days

A 30-day prediction horizon is the operational sweet spot for a B2B SaaS CS team:
- Short enough that the signal is still actionable (the account is still engaged enough to reach)
- Long enough to run a proper intervention (executive sponsor outreach, QBR, product training)
- Aligns with monthly renewal and billing cycles

### Success Metrics

The system is evaluated on two levels:

**Model performance:** PR-AUC (Precision-Recall AUC) is the primary metric, not ROC-AUC, because churn is a minority class (~15–20% of accounts). PR-AUC directly measures how well the model separates churners from non-churners when positives are rare.

**Operational utility:** A CS manager using the platform should be able to open the dashboard on Monday morning, identify the top 10 accounts to call this week, understand why each one is at risk, and export that list — all in under five minutes.

---

## 3. Dataset and Data Design

### 3.1 Source Dataset

The raw data originates from the **UCI Machine Learning Repository Online Retail dataset** (also available on Kaggle). This is a real transactional dataset from a UK-based non-store online retailer spanning December 2010 to December 2011.

| Attribute | Value |
|---|---|
| Total records | 541,909 transaction line items |
| Unique customers | ~4,372 |
| Date range | 01 Dec 2010 — 09 Dec 2011 (~13 months) |
| Key fields | InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country |

**Important:** This is not native SaaS telemetry. It is a retail transaction log. The core contribution of this project's data design is a principled transformation that maps retail customer purchasing behavior to a credible B2B SaaS account engagement model.

### 3.2 Conceptual Mapping: Retail → SaaS

The following table defines how each retail concept was re-interpreted as a SaaS equivalent:

| Retail Concept | SaaS Equivalent | Rationale |
|---|---|---|
| Customer (CustomerID) | User | An individual who logs into and uses the product |
| Invoice (InvoiceNo) | Session | A discrete visit or login to the product |
| Line item (StockCode row) | In-product event | A specific action taken within a session |
| Revenue (Quantity × UnitPrice) | Subscription revenue | Payment from the account for using the product |
| Period of no purchases | Inactivity / disengagement | Precursor to churn |

### 3.3 B2B Account Abstraction

The most important transformation is the elevation from individual customers to **company accounts**. In B2B SaaS, the billing and churn unit is the company (account), not the individual user. A company pays for a seat license; multiple employees (users) share that account.

The `etl/build_b2b_marts.py` script performs this grouping using a realistic company-size distribution:

```
75% of accounts: 2–6 seats    (small companies, most common)
20% of accounts: 7–15 seats   (mid-size companies)
 5% of accounts: 16–30 seats  (large enterprise accounts)
```

This yields a long-tail distribution consistent with real B2B SaaS portfolios where small customers dominate by count and large customers dominate by revenue. Users are randomly assigned to accounts with a fixed random seed (42) for reproducibility.

Each account is then assigned synthetic but realistic attributes:

| Attribute | Distribution |
|---|---|
| Plan type | Basic 55%, Pro 35%, Enterprise 10% |
| Contract type | Monthly 70%, Annual 30% |
| Region | NA 55%, EU 30%, APAC 15% |
| Seats | Derived from user count (see above) |

### 3.4 Churn Label Definition

Churn is defined **behaviorally** rather than contractually (since the retail dataset has no cancellation events):

> An account is considered to have churned if it records **zero activity for 30 consecutive days** after its last active day.

The churn date is set as `last_active_day + 30 days`. An account's `churn_flag = 1` only if its derived churn date falls within the observable dataset window (before December 2011). Accounts that stop appearing near the dataset end are treated as right-censored (not labeled as churned) to avoid false positives at the boundary.

### 3.5 Final Data Model

The transformation produces four structured mart tables:

#### `dim_account`
One row per account. Contains account metadata.

| Column | Type | Description |
|---|---|---|
| account_id | TEXT | Unique identifier (e.g., ACC_0364) |
| plan_type | TEXT | Basic / Pro / Enterprise |
| contract_type | TEXT | Monthly / Annual |
| region | TEXT | NA / EU / APAC |
| seats | INT | Number of licensed users |

#### `dim_user`
One row per user. Maps individual users to their parent account.

| Column | Type | Description |
|---|---|---|
| user_id | INT | Original CustomerID from retail dataset |
| account_id | TEXT | Foreign key to dim_account |

#### `fact_usage_daily_account`
One row per account per active day. Daily engagement metrics at account level.

| Column | Type | Description |
|---|---|---|
| account_id | TEXT | Foreign key to dim_account |
| event_day | TIMESTAMP | Calendar date |
| active_users | INT | Unique users who had at least one event |
| sessions | INT | Unique invoices (sessions) that day |
| events | INT | Total event count (line items) |
| revenue | FLOAT | Total revenue (sum of Quantity × UnitPrice) |

#### `fact_churn_account`
One row per account. Ground truth churn labels.

| Column | Type | Description |
|---|---|---|
| account_id | TEXT | Foreign key to dim_account |
| last_active_day | TIMESTAMP | Last day with recorded activity |
| churn_date | TIMESTAMP | Inferred churn date (last_active + 30d) |
| churn_flag | INT | 1 = churned (observed within dataset), 0 = active |
| plan_type, contract_type, region, seats | — | Denormalized account attributes |

---

## 4. Data Pipeline and ETL

The ETL pipeline (`etl/`) processes the raw dataset through four sequential stages:

### Stage 1 — Download and Inspect (`download_dataset.py`, `inspect_dataset.py`)
Downloads the raw UCI dataset and produces a data quality report: null rates, data types, date ranges, customer counts.

### Stage 2 — Clean Events (`clean_events.py`)
Applies cleaning rules to the raw transaction log:
- Remove rows with null CustomerID (guest transactions, not account activity)
- Remove cancelled invoices (InvoiceNo starting with 'C')
- Remove rows with zero or negative quantity/price (data entry errors)
- Parse InvoiceDate as datetime and extract `event_day`
- Compute `line_revenue = Quantity × UnitPrice`
- Output: `data/processed/events_clean.parquet`

### Stage 3 — Build B2B Marts (`build_b2b_marts.py`)
Transforms cleaned events into the four mart tables described in Section 3.5. Key operations:
- Assign users to accounts using the company-size distribution
- Aggregate daily usage metrics per account (sessions, active users, events, revenue)
- Label churn using the 30-day inactivity rule

### Stage 4 — Build Feature Table (`build_feature_table_30d.py`)
Constructs the ML feature table from daily account usage. Described in detail in Section 5.

---

## 5. Feature Engineering

Feature engineering is the most important step between raw data and model performance. The feature table is built as a **full daily panel** — one row per account per day — using a rolling window approach that simulates real-time scoring.

### 5.1 Full Daily Panel

Because most accounts are not active every single day, the daily usage table has gaps. Missing days are filled with zeros before computing rolling features. This is critical: a gap is not missing data — it is a signal of inactivity.

### 5.2 Feature Categories

**Rolling usage averages** (7, 14, 30-day windows)

These capture how active the account has been at different timescales. Three windows are used to capture both recent behavior and the longer-term baseline.

| Feature | Description |
|---|---|
| `active_users_mean_7d` | Average daily unique users, last 7 days |
| `active_users_mean_14d` | Average daily unique users, last 14 days |
| `active_users_mean_30d` | Average daily unique users, last 30 days |
| `sessions_mean_7d` | Average daily sessions, last 7 days |
| `sessions_mean_14d` | Average daily sessions, last 14 days |
| `sessions_mean_30d` | Average daily sessions, last 30 days |
| `events_mean_7d` | Average daily in-product events, last 7 days |
| `events_mean_14d` | Average daily in-product events, last 14 days |
| `events_mean_30d` | Average daily in-product events, last 30 days |

**Rolling revenue sums** (7, 14, 30-day windows)

| Feature | Description |
|---|---|
| `revenue_sum_7d` | Total revenue, last 7 days |
| `revenue_sum_14d` | Total revenue, last 14 days |
| `revenue_sum_30d` | Total revenue, last 30 days |

**Trend and momentum features**

These capture *change* in behavior, which is often more predictive than the level.

| Feature | Formula | Interpretation |
|---|---|---|
| `sessions_drop_7v7` | (prev_7d_sessions − last_7d_sessions) / prev_7d_sessions | Week-over-week session decline |
| `sessions_trend_7_minus_30` | sessions_mean_7d − sessions_mean_30d | Whether recent activity is above or below the 30d baseline |
| `sessions_std_30d` | Rolling std of daily sessions over 30d | Usage variability — erratic usage is a risk signal |

**Recency feature**

| Feature | Description |
|---|---|
| `days_since_last_activity` | Days elapsed since the account last had any session |

**Account attributes**

| Feature | Description |
|---|---|
| `tenure_days` | Days since the account's first recorded activity |
| `seats` | Number of licensed users |
| `plan_type` | Basic / Pro / Enterprise (one-hot encoded at model training) |
| `contract_type` | Monthly / Annual (one-hot encoded) |
| `region` | NA / EU / APAC (one-hot encoded) |

**Total: 18 engineered features** (plus 3 categorical attributes encoded at training time).

### 5.3 Label Construction

The binary label `will_churn_30d` is set to 1 if the account's derived churn date falls within the 30-day window after the snapshot date:

```
will_churn_30d = 1  if  snapshot_date < churn_date ≤ snapshot_date + 30 days
```

**Censoring fix:** Snapshot rows within 30 days of the dataset end date are removed. Without this, the model would incorrectly learn that accounts active near the dataset boundary are non-churners — when in fact their churn outcome is simply unobservable.

---

## 6. Machine Learning Model

### 6.1 Model Choice: XGBoost

XGBoost was selected for the following reasons:
- Strong performance on tabular data with mixed feature types
- Native handling of missing values (important for early-tenure accounts where some rolling windows are unavailable)
- Built-in feature importance and compatibility with SHAP for explainability
- Supports `scale_pos_weight` for class imbalance without resampling

### 6.2 Train / Validation / Test Split

A **time-based split** is used — not a random split. This is essential for time-series prediction: randomly shuffling rows would allow the model to "see the future" during training, inflating performance estimates.

| Split | Proportion | Purpose |
|---|---|---|
| Train | 70% (earliest snapshots) | Model fitting |
| Validation | 15% (middle period) | Early stopping, calibration |
| Test | 15% (most recent snapshots) | Final held-out evaluation |

### 6.3 Class Imbalance

The overall churn rate in the feature table is approximately 15–20%. To prevent the model from ignoring the minority class, `scale_pos_weight` is set to the ratio of negative to positive examples in the training set:

```
scale_pos_weight = count(non-churners) / count(churners)
```

This effectively upweights churn examples in the loss function without oversampling.

### 6.4 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| n_estimators | 5000 (with early stopping) | Upper bound; early stopping finds the optimal number |
| max_depth | 4 | Shallow trees reduce overfitting on small dataset |
| learning_rate | 0.05 | Low learning rate with many trees is more regularized |
| subsample | 0.8 | Row subsampling reduces variance |
| colsample_bytree | 0.8 | Feature subsampling reduces correlation between trees |
| reg_lambda | 1.0 | L2 regularization |
| eval_metric | aucpr | PR-AUC is the right metric for imbalanced classes |
| early_stopping_rounds | 50 | Stop if validation PR-AUC doesn't improve for 50 rounds |

### 6.5 Performance

The model is evaluated on the held-out test set using both the raw (uncalibrated) scores and the calibrated scores:

| Metric | Raw Score | Calibrated Score |
|---|---|---|
| ROC-AUC | Reported at training time | Preserved post-calibration |
| PR-AUC | Primary metric | Preserved post-calibration |
| Mean score | Overconfident (>> true churn rate) | Close to true churn rate |

---

## 7. Model Calibration

### 7.1 The Problem with Raw XGBoost Scores

XGBoost is optimized to **rank** accounts (high-risk vs. low-risk) rather than to produce **calibrated probabilities**. The raw output scores from `predict_proba` are systematically overconfident — for example, a raw score of 0.45 does not mean a 45% chance of churn. It just means the model thinks this account is very high-risk relative to others.

Showing these overconfident raw scores to a CS team is harmful: a manager who sees "45% churn probability" and later finds out only 12% of such accounts actually churned will stop trusting the model.

### 7.2 Solution: Platt Sigmoid Calibration

**Platt calibration** fits a logistic regression on top of the model's raw output scores using the held-out validation set. The logistic regression learns a sigmoid mapping from raw scores to true probabilities:

```
calibrated_probability = sigmoid(a × raw_score + b)
```

where `a` and `b` are fitted on the validation set. The result is a calibrated probability where:
- The mean calibrated score ≈ the true churn rate in the population
- A score of 0.12 can be interpreted as "approximately 12% probability of churning in 30 days"

This is the `churn_risk_calibrated` column stored in the database. The raw score is stored as `model_score_raw` and is not surfaced to users.

---

## 8. SHAP Explainability

### 8.1 Why Explainability Matters

A churn risk score alone is not actionable. A CS manager needs to know: *why* is this account flagged? What specific behavior changed? This is what makes the difference between a black-box model and a tool people actually trust and use.

### 8.2 SHAP (SHapley Additive exPlanations)

SHAP uses game-theoretic Shapley values to assign each feature a contribution to the model's prediction for a specific account. A positive SHAP value means the feature increased the predicted churn probability; a negative value means it reduced it.

For each account, the top 5 features with the largest positive SHAP values are stored as the account's **risk drivers**.

### 8.3 Implementation

`shap.TreeExplainer` is used for efficiency with tree-based models. It computes exact Shapley values using the tree structure rather than sampling.

```python
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_dense)  # shape: (n_accounts, n_features)
```

For each account, the top 5 positive drivers are stored with:
- `feature_name` — technical feature identifier
- `feature_label` — human-readable label
- `shap_value` — the SHAP value (magnitude = contribution to risk)
- `feature_value` — the account's actual value for that feature
- `pop_avg` — the population average for that feature (for comparison)

### 8.4 Plain-English Rendering

Raw feature values (e.g., `active_users_mean_30d = 0.0333`) are converted to interpretable totals before display. A 30-day average of 0.0333 active users/day is shown as "~1 active user in the past 30 days" — a statement a non-technical user immediately understands.

Each driver card shows:
1. What the metric means (definition)
2. What the account's value is (in plain units)
3. How it compares to the portfolio average
4. Why it signals churn risk

---

## 9. Production Scoring Pipeline

### 9.1 Weekly Refresh Cycle

The pipeline is designed to run weekly. Each run:

1. **Score** — Re-train the model on the latest feature table and score every account's most recent snapshot (`production_score_30d.py`)
2. **Rank** — Assign risk percentiles, bands, and buckets (`production_score_percentiles.py`)
3. **Load** — Push all updated CSVs to the Neon Postgres database (`db/load_marts_to_postgres.py`)

### 9.2 Risk Classification

After scoring, each account receives three complementary risk signals:

| Signal | Definition |
|---|---|
| `churn_risk_calibrated` | Calibrated 30-day churn probability (0–1) |
| `risk_percentile` | Percentile rank 0–100 relative to all accounts (100 = highest risk) |
| `risk_band` | Top 1% / Top 5% / Top 10% / Top 25% / Rest |
| `risk_bucket` | High (top 5%) / Medium (top 25%) / Low (remaining) |

**Why three signals?** The calibrated probability is honest but requires statistical literacy. The percentile is intuitive ("this account is in the top 2% most at-risk"). The bucket is the operational trigger: "High = call this week."

### 9.3 History Tracking

Every pipeline run appends to `churn_scores_history` with a `run_date` column. This enables:
- Week-over-week comparison (how many accounts moved into/out of High risk?)
- Trend charts on the dashboard
- "Newly at-risk" detection (accounts in High this week that were not High last week)

---

## 10. Database Layer

### 10.1 Database: Neon Serverless Postgres

All mart tables are loaded into a Neon serverless PostgreSQL instance. Neon provides a serverless Postgres endpoint that scales to zero when idle — appropriate for a portfolio project with intermittent traffic.

### 10.2 Schema

Six tables are loaded into the database:

| Table | Rows (approx) | Purpose |
|---|---|---|
| `dim_account` | ~900 | Account master data |
| `dim_user` | ~4,372 | User-to-account mapping |
| `fact_usage_daily_account` | ~300,000+ | Daily usage metrics |
| `fact_churn_account` | ~900 | Churn labels |
| `churn_scores_latest_ranked` | ~900 | Current risk scores |
| `churn_scores_history` | ~900 × weeks | Historical scores per run |
| `account_shap_drivers` | ~4,500 | Top-5 SHAP drivers per account |

### 10.3 Connection Layer

`copilot/query_db.py` manages the database connection using SQLAlchemy with connection pooling. Credentials are read from Streamlit secrets (Streamlit Cloud) or environment variables (local development), never hardcoded.

---

## 11. Application and UI

The application is built in **Streamlit** and deployed on **Streamlit Cloud**.

### 11.1 Home Dashboard (`app/Home.py`)

The dashboard provides an immediate portfolio-level view on load:

**KPI Cards (4 metrics)**
- High-Risk Accounts — count with week-over-week delta (▲/▼)
- Total Accounts — with percentage flagged high-risk
- Newly At-Risk — accounts that entered High risk since last week
- Revenue at Risk — total revenue from high-risk accounts (formatted as $K/$M)

**Charts**
- High-Risk Account Trend — weekly line chart from `churn_scores_history`
- Risk Distribution — donut chart of High/Medium/Low breakdown
- High-Risk by Region — bar chart
- Avg Risk Percentile by Plan — bar chart

**Sections**
- Top 5 High-Risk Accounts — preview table
- Newly High-Risk This Week — alert banner with account count

### 11.2 Ask Copilot (`app/pages/02_Ask_Copilot.py`)

The copilot page provides two modes of interaction:

**Preset queries** (left panel, organized by category):
- Risk Overview (9 queries): top accounts, distribution, percentile bands
- By Plan (5 queries)
- By Region (5 queries)
- By Contract (3 queries)
- Segments (3 queries): cross-cuts like plan × region
- Revenue at Risk (3 queries)
- Weekly Trends (5 queries)

**Free-form LLM input** (text box at top): any natural language question is routed through the Claude LLM copilot (see Section 12).

**Explain Account**: enter any account ID to see a full SHAP explainability breakdown with a HIGH/MEDIUM/LOW risk verdict banner.

**Click-to-Explain**: every account list result (from both preset and LLM queries) has an "Explain →" button on each row, providing a one-click path from the risk list to the account explanation.

### 11.3 Chart Rendering

Charts are rendered with Plotly. The renderer automatically selects the appropriate chart type based on the query result shape:
- SHAP results → verdict banner + plain-English driver cards
- Account list with `risk_percentile` → horizontal bar chart colored by plan type, with churn probability labels
- Account list without `risk_percentile` (LLM results) → horizontal bar chart by primary numeric column
- Time series (`run_date`) → line chart with area fill
- Distribution → donut chart
- Two categorical columns → grouped bar chart

---

## 12. LLM Copilot

### 12.1 Architecture

The LLM copilot (`copilot/llm_sql_router.py`) uses **Anthropic Claude Haiku** to translate free-form business questions into valid PostgreSQL queries against the project's schema.

The system prompt provides:
- Full schema definition (all 7 tables with column descriptions and types)
- Business rules (e.g., "High risk" = risk_bucket = 'High', how to join tables)
- Output format rules (SELECT only, exact column names, round floats to 4dp)
- Fallback instruction for unanswerable questions

### 12.2 Safety Layer

Every LLM-generated query passes through a safety check before execution:

```python
def _is_safe(sql: str) -> bool:
    forbidden = r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke)\b"
    return not re.search(forbidden, sql, re.IGNORECASE)
```

Only SELECT statements reach the database. Mutation queries are blocked regardless of what the LLM generates.

### 12.3 Prompt Caching

The system prompt is marked with `"cache_control": {"type": "ephemeral"}` to use Anthropic's prompt caching. Since the schema and rules are static and constitute most of the token count, caching significantly reduces latency and API cost for repeated questions.

### 12.4 User Experience

- LLM-generated results display an **"AI Generated"** badge
- A **"View generated SQL"** expander shows the exact SQL produced — useful for demos and building trust
- The fallback message for unanswerable questions returns a single-row result with an explanatory message rather than an error

---

## 13. Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| Machine learning | XGBoost 3.2 |
| Calibration | Scikit-learn LogisticRegression (Platt) |
| Explainability | SHAP 0.45 (TreeExplainer) |
| Database | Neon serverless PostgreSQL |
| ORM / query | SQLAlchemy 2.0 |
| LLM | Anthropic Claude Haiku (claude-haiku-4-5) |
| Frontend | Streamlit 1.35 |
| Charts | Plotly 5.22 |
| Deployment | Streamlit Cloud |
| Version control | Git / GitHub |

---

## 14. Key Design Decisions and Trade-offs

### Calibration over raw scores
Raw XGBoost scores are systematically overconfident. Platt calibration was chosen over isotonic regression because the validation set is small and isotonic regression can overfit with limited data. The calibrated mean score closely tracks the true population churn rate.

### Percentile-based risk classification over probability thresholds
Classifying accounts as High/Medium/Low using probability thresholds (e.g., p > 0.3 = High) is fragile — it depends on the model's calibration being perfect and the churn rate being stable. Percentile-based classification ("High = top 5% of accounts") is self-normalizing: it always identifies a fixed proportion of the portfolio as needing attention, regardless of overall churn rate changes.

### Time-based train/test split
A random split would leak future information into training. Time-based splitting ensures the model is evaluated on a strictly future period, giving a realistic estimate of production performance.

### LLM for free-form queries, not for all queries
Preset SQL queries are used for the 30+ most common analytical questions. The LLM is reserved for free-form input. This is the right architecture: preset queries are faster, cheaper, and have guaranteed correct SQL. The LLM handles the long tail of questions that can't be anticipated.

### Schema-scoped system prompt
The LLM is given only the tables and columns that exist in the database. This prevents hallucinated table names and column references, which is the most common failure mode in text-to-SQL systems.

---

## 15. Deliverables Summary

| Deliverable | Description | Location |
|---|---|---|
| Raw data pipeline | Download, clean, and validate the UCI retail dataset | `etl/` |
| B2B data model | Transform retail transactions into account-level SaaS schema | `etl/build_b2b_marts.py` |
| Feature table | 18 engineered rolling features with churn labels | `etl/build_feature_table_30d.py` |
| XGBoost model | Time-split trained churn classifier with early stopping | `ml/production_score_30d.py` |
| Platt calibration | Sigmoid calibrator mapping raw scores to honest probabilities | `ml/production_score_30d.py` |
| SHAP drivers | Per-account top-5 feature attributions stored to database | `ml/production_score_30d.py` |
| Risk ranking | Percentile, band, and bucket assignment with history tracking | `ml/production_score_percentiles.py` |
| Weekly pipeline | End-to-end refresh script (score → rank → load) | `pipeline/run_weekly_refresh.py` |
| Neon Postgres | Production database with 7 tables | `db/load_marts_to_postgres.py` |
| Home dashboard | KPI cards, trend charts, risk distribution | `app/Home.py` |
| Ask Copilot UI | 30+ preset queries + SHAP explainability + click-to-explain | `app/pages/02_Ask_Copilot.py` |
| LLM copilot | Claude text-to-SQL with schema prompt and safety layer | `copilot/llm_sql_router.py` |
| Live deployment | Streamlit Cloud app, publicly accessible | https://saas-churn-copilot-4zercogks3sgf8xvdwzvbc.streamlit.app/ |

---

*Report prepared April 2026. Project by Shruti Karmarkar, NYU.*
