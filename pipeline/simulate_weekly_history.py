"""
Simulates 5 weeks of historical churn scores for demo purposes.

Starting from the current production scores, each week applies realistic drift:
  - 75% of accounts: small random walk (stable)
  - 15% of accounts: meaningful deterioration (riskier)
  -  10% of accounts: meaningful improvement (recovering)

Bucketing uses fixed score thresholds (not percentile rank) so accounts
genuinely move in and out of High risk as their scores drift week to week.
"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import create_engine

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_PATH  = os.path.join(BASE_DIR, "data", "mart", "churn_scores_latest_ranked.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "mart", "churn_scores_history.csv")

DB_URL = "postgresql+psycopg2://churn:churn@localhost:5432/churn_db"

N_WEEKS     = 5
RANDOM_SEED = 42

# Fixed thresholds derived from current data (95th and 75th percentile scores)
# Using fixed thresholds means accounts can cross them as scores drift
HIGH_THRESHOLD   = 0.0143
MEDIUM_THRESHOLD = 0.0032

def assign_bucket(score: float) -> str:
    if score >= HIGH_THRESHOLD:   return "High"
    elif score >= MEDIUM_THRESHOLD: return "Medium"
    else:                           return "Low"

def assign_band(percentile: float) -> str:
    if percentile >= 99:   return "Top 1%"
    elif percentile >= 95: return "Top 5%"
    elif percentile >= 90: return "Top 10%"
    elif percentile >= 75: return "Top 25%"
    else:                  return "Rest"

def build_week(base_df: pd.DataFrame, scores: np.ndarray, run_date: str) -> pd.DataFrame:
    df = base_df[["account_id", "snapshot_date"]].copy()
    df["churn_risk_calibrated"] = scores
    df["risk_percentile"] = pd.Series(scores).rank(pct=True) * 100
    df["risk_band"]       = df["risk_percentile"].apply(assign_band)
    df["risk_bucket"]     = df["churn_risk_calibrated"].apply(assign_bucket)
    df.insert(0, "run_date", run_date)
    return df[["run_date", "account_id", "snapshot_date",
               "churn_risk_calibrated", "risk_percentile", "risk_band", "risk_bucket"]]

def perturb(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n     = len(scores)
    noise = np.zeros(n)
    idx   = np.arange(n)

    # 75% stable — tiny noise relative to score magnitude
    stable = rng.choice(idx, size=int(n * 0.75), replace=False)
    noise[stable] = rng.normal(0, 0.0005, size=len(stable))

    rest = np.setdiff1d(idx, stable)

    # 15% deteriorating — push toward and past the High threshold
    worsen = rng.choice(rest, size=int(n * 0.15), replace=False)
    noise[worsen] = rng.uniform(0.002, 0.008, size=len(worsen))

    # 10% recovering — pull away from High threshold
    recover = np.setdiff1d(rest, worsen)
    noise[recover] = rng.uniform(-0.006, -0.001, size=len(recover))

    return np.clip(scores + noise, 0.001, 0.999)

def main():
    rng  = np.random.default_rng(RANDOM_SEED)
    base = pd.read_csv(SCORES_PATH, parse_dates=["snapshot_date"])
    print(f"Loaded base scores: {len(base)} accounts")
    print(f"Fixed High threshold: >= {HIGH_THRESHOLD}")

    today      = date.today()
    run_dates  = [today - timedelta(weeks=(N_WEEKS - 1 - i)) for i in range(N_WEEKS)]

    print(f"\nSimulating {N_WEEKS} weeks:")
    for d in run_dates:
        print(f"  {d}")

    all_weeks     = []
    current_scores = base["churn_risk_calibrated"].values.copy()

    for i, run_date in enumerate(run_dates):
        week_df    = build_week(base, current_scores, run_date.isoformat())
        high_count = (week_df["risk_bucket"] == "High").sum()
        print(f"\nWeek {i+1} ({run_date})  |  High: {high_count}  "
              f"Medium: {(week_df['risk_bucket']=='Medium').sum()}  "
              f"Low: {(week_df['risk_bucket']=='Low').sum()}")
        all_weeks.append(week_df)

        if i < N_WEEKS - 1:
            current_scores = perturb(current_scores, rng)

    history = pd.concat(all_weeks, ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)
    print(f"\nSaved {len(history)} rows to churn_scores_history.csv")

    print("\nLoading into Postgres...")
    engine = create_engine(DB_URL)
    history.to_sql("churn_scores_history", engine, if_exists="replace", index=False)
    print("Done.")

    print("\nHigh-risk count by week:")
    summary = (
        history[history["risk_bucket"] == "High"]
        .groupby("run_date").size()
        .reset_index(name="high_risk_accounts")
    )
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
