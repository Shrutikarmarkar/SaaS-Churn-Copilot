import os
import pandas as pd
import numpy as np
from datetime import date

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH      = os.path.join(BASE_DIR, "data", "mart", "churn_scores_latest.csv")
OUT_PATH     = os.path.join(BASE_DIR, "data", "mart", "churn_scores_latest_ranked.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "mart", "churn_scores_history.csv")

def band_from_percentile(p):
    if p >= 99:
        return "Top 1%"
    elif p >= 95:
        return "Top 5%"
    elif p >= 90:
        return "Top 10%"
    elif p >= 75:
        return "Top 25%"
    else:
        return "Rest"

def bucket_from_percentile(p):
    if p >= 95:
        return "High"
    elif p >= 75:
        return "Medium"
    else:
        return "Low"

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["snapshot_date"])

    df = df.sort_values("churn_risk_calibrated", ascending=True).reset_index(drop=True)

    # percentile rank: lowest risk -> 0, highest risk -> 100
    df["risk_percentile"] = df["churn_risk_calibrated"].rank(pct=True) * 100

    df["risk_band"]   = df["risk_percentile"].apply(band_from_percentile)
    df["risk_bucket"] = df["risk_percentile"].apply(bucket_from_percentile)

    df = df.sort_values("risk_percentile", ascending=False)

    # model_score_raw stays out of the DB-facing table — it is not a calibrated probability
    out = df[["account_id", "snapshot_date", "churn_risk_calibrated", "risk_percentile", "risk_band", "risk_bucket"]]

    out.to_csv(OUT_PATH, index=False)
    print("Saved ranked scores to:", OUT_PATH)

    # Append this run to history — one row per account per run_date
    run_date = date.today().isoformat()
    history_row = out.copy()
    history_row.insert(0, "run_date", run_date)

    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        # Drop any rows from today if the pipeline is re-run on the same day
        history = history[history["run_date"] != run_date]
        history = pd.concat([history, history_row], ignore_index=True)
    else:
        history = history_row

    history.to_csv(HISTORY_PATH, index=False)
    print(f"Appended run_date={run_date} to history ({len(history_row)} accounts). Total history rows: {len(history)}")

    print("\nTop 10:")
    print(out.head(10))

if __name__ == "__main__":
    main()
