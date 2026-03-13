import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(BASE_DIR, "data", "mart", "churn_scores_latest.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "mart", "churn_scores_latest_ranked.csv")

def band_from_percentile(p):
    # p is 0..100, higher means riskier
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
    df = df.sort_values("churn_risk_score", ascending=True).reset_index(drop=True)

    # percentile rank: lowest risk -> 0, highest risk -> 100
    df["risk_percentile"] = df["churn_risk_score"].rank(pct=True) * 100

    df["risk_band"] = df["risk_percentile"].apply(band_from_percentile)
    df["risk_bucket"] = df["risk_percentile"].apply(bucket_from_percentile)

    df = df.sort_values("churn_risk_score", ascending=False)

    df.to_csv(OUT_PATH, index=False)

    print("Saved ranked scores to:", OUT_PATH)
    print("\nTop 10:")
    print(df.head(10)[["account_id", "snapshot_date", "churn_risk_score", "risk_percentile", "risk_band", "risk_bucket"]])

if __name__ == "__main__":
    main()