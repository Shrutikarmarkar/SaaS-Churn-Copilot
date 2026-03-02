import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DAILY_PATH = os.path.join(BASE_DIR, "data", "mart", "fact_usage_daily_account.csv")
CHURN_PATH = os.path.join(BASE_DIR, "data", "mart", "fact_churn_account.csv")
ACCOUNT_PATH = os.path.join(BASE_DIR, "data", "mart", "dim_account.csv")

OUT_DIR = os.path.join(BASE_DIR, "data", "mart")
OUT_FEATURES = os.path.join(OUT_DIR, "feature_table_30d.csv")

HORIZON_DAYS = 30
MIN_HISTORY_DAYS = 30   # require at least 30 days history for stable rolling features

def make_full_daily_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Expand to full calendar per account so rolling windows are correct (missing days -> 0 activity).
    """
    daily = daily.copy()
    daily["event_day"] = pd.to_datetime(daily["event_day"])

    # Build full date range across dataset
    min_day = daily["event_day"].min()
    max_day = daily["event_day"].max()
    all_days = pd.date_range(min_day, max_day, freq="D")

    accounts = daily["account_id"].unique()
    idx = pd.MultiIndex.from_product([accounts, all_days], names=["account_id", "event_day"])
    panel = pd.DataFrame(index=idx).reset_index()

    panel = panel.merge(daily, on=["account_id", "event_day"], how="left")

    # Fill missing usage with zeros
    for col in ["active_users", "sessions", "events", "revenue"]:
        panel[col] = panel[col].fillna(0)

    return panel

def add_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window features per account.
    All features are computed using data up to snapshot_date (event_day).
    """
    df = panel.sort_values(["account_id", "event_day"]).copy()

    g = df.groupby("account_id", group_keys=False)

    # Rolling means
    for w in [7, 14, 30]:
        df[f"sessions_mean_{w}d"] = g["sessions"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
        df[f"events_mean_{w}d"] = g["events"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
        df[f"active_users_mean_{w}d"] = g["active_users"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
        df[f"revenue_sum_{w}d"] = g["revenue"].rolling(w, min_periods=w).sum().reset_index(level=0, drop=True)

    # Rolling volatility (std) for sessions
    df["sessions_std_30d"] = g["sessions"].rolling(30, min_periods=30).std().reset_index(level=0, drop=True)

    # Usage drop: compare last 7d vs previous 7d (14d window split)
    # drop_ratio = (prev7 - last7) / (prev7 + eps)
    last7 = g["sessions"].rolling(7, min_periods=7).sum().reset_index(level=0, drop=True)
    prev7 = g["sessions"].shift(7).rolling(7, min_periods=7).sum().reset_index(level=0, drop=True)
    eps = 1e-6
    df["sessions_drop_7v7"] = (prev7 - last7) / (prev7 + eps)

    # Simple trend slope over last 30 days (approx):
    # slope = (today_sessions_mean_7d - sessions_mean_30d)
    df["sessions_trend_7_minus_30"] = df["sessions_mean_7d"] - df["sessions_mean_30d"]

    # Days since last activity (sessions>0) at snapshot date
    def days_since_last_active(sessions_series: pd.Series) -> pd.Series:
        last_active_idx = np.where(sessions_series.values > 0, np.arange(len(sessions_series)), -1)
        last_active = np.maximum.accumulate(last_active_idx)
        days_since = (np.arange(len(sessions_series)) - last_active).astype(float)
        days_since[last_active == -1] = np.nan
        return pd.Series(days_since, index=sessions_series.index)

    df["days_since_last_activity"] = g["sessions"].apply(days_since_last_active).reset_index(level=0, drop=True)

    return df

def build_labels(features: pd.DataFrame, churn: pd.DataFrame) -> pd.DataFrame:
    """
    Label = churn in next 30 days after snapshot_date (event_day).
    """
    df = features.copy()
    df = df.rename(columns={"event_day": "snapshot_date"})
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    churn = churn.copy()
    churn["churn_date"] = pd.to_datetime(churn["churn_date"])
    churn["last_active_day"] = pd.to_datetime(churn["last_active_day"])

    df = df.merge(churn[["account_id", "churn_date", "churn_flag"]], on="account_id", how="left")

    # Only accounts with observed churn_flag=1 have a meaningful churn_date inside dataset window.
    # For churn_flag=0, churn_date is after dataset end; label will naturally be 0.
    df["will_churn_30d"] = (
        (df["churn_date"] > df["snapshot_date"]) &
        (df["churn_date"] <= df["snapshot_date"] + pd.Timedelta(days=HORIZON_DAYS))
    ).astype(int)

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    daily = pd.read_csv(DAILY_PATH)
    churn = pd.read_csv(CHURN_PATH)
    dim_account = pd.read_csv(ACCOUNT_PATH)

    print("Daily rows:", daily.shape)
    print("Churn rows:", churn.shape)

    print("Building full daily panel...")
    panel = make_full_daily_panel(daily)
    print("Panel shape:", panel.shape)

    print("Adding rolling features...")
    feat = add_rolling_features(panel)

    # Keep only snapshot rows where we have enough history to compute 30d features
    feat["snapshot_date"] = feat["event_day"]
    # Tenure proxy: days since first day in dataset for that account
    first_day = feat.groupby("account_id")["event_day"].transform("min")
    feat["tenure_days"] = (feat["event_day"] - first_day).dt.days

    # Filter: need at least MIN_HISTORY_DAYS days to compute stable rolling features
    feat = feat[feat["tenure_days"] >= MIN_HISTORY_DAYS].copy()

    # Add labels
    feat = build_labels(feat.drop(columns=["snapshot_date"]), churn)

    # Add account attributes
    feat = feat.merge(dim_account, on="account_id", how="left")

    # Select final columns (clean set)
    keep_cols = [
        "account_id", "snapshot_date",
        "plan_type", "contract_type", "region", "seats",
        "tenure_days",
        "active_users_mean_7d", "active_users_mean_14d", "active_users_mean_30d",
        "sessions_mean_7d", "sessions_mean_14d", "sessions_mean_30d",
        "events_mean_7d", "events_mean_14d", "events_mean_30d",
        "revenue_sum_7d", "revenue_sum_14d", "revenue_sum_30d",
        "sessions_std_30d",
        "sessions_drop_7v7",
        "sessions_trend_7_minus_30",
        "days_since_last_activity",
        "will_churn_30d",
    ]

    feat_out = feat[keep_cols].copy()
    # ---- CENSORING FIX: remove snapshots too close to dataset end ----
    max_snapshot = feat_out["snapshot_date"].max()   # get the maximum snapshot date
    cutoff = max_snapshot - pd.Timedelta(days=HORIZON_DAYS)   # get the cutoff date
    feat_out = feat_out[feat_out["snapshot_date"] <= cutoff].copy()   # remove the snapshots too close to the dataset end

    # Basic report
    print("\nAfter censoring fix:")
    print("Feature table shape:", feat_out.shape)
    print("Label rate (will_churn_30d):", feat_out["will_churn_30d"].mean())
    print("Date range:", feat_out["snapshot_date"].min(), "→", feat_out["snapshot_date"].max())

    feat_out.to_csv(OUT_FEATURES, index=False)
    print("\nSaved feature table to:", OUT_FEATURES)

if __name__ == "__main__":
    main()