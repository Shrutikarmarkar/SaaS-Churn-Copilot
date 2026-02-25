import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "events_clean.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "mart")

# Outputs
DIM_USER_PATH = os.path.join(OUT_DIR, "dim_user.csv")
DIM_ACCOUNT_PATH = os.path.join(OUT_DIR, "dim_account.csv")
FACT_ACCOUNT_DAILY_PATH = os.path.join(OUT_DIR, "fact_usage_daily_account.csv")
FACT_CHURN_PATH = os.path.join(OUT_DIR, "fact_churn_account.csv")

INACTIVITY_DAYS_FOR_CHURN = 30

def assign_accounts(users: pd.Series, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic B2B accounts by grouping users into companies.
    We simulate company sizes (seats) using a realistic skew:
      many small companies, fewer large.
    """
    rng = np.random.default_rng(seed)
    user_ids = users.drop_duplicates().sort_values().to_list()

    # Company sizes: mostly 2-6 seats, some 7-15, few 16-30
    # This yields a realistic long-tail distribution
    sizes = []
    while sum(sizes) < len(user_ids):
        r = rng.random()
        if r < 0.75:
            sizes.append(int(rng.integers(2, 7)))   # 2-6
        elif r < 0.95:
            sizes.append(int(rng.integers(7, 16)))  # 7-15
        else:
            sizes.append(int(rng.integers(16, 31))) # 16-30

    # Trim sizes to match exactly number of users
    total = 0
    trimmed = []
    for s in sizes:
        if total + s >= len(user_ids):
            trimmed.append(len(user_ids) - total)
            break
        trimmed.append(s)
        total += s

    # Assign users to accounts
    assignments = []
    idx = 0
    for acc_num, s in enumerate(trimmed, start=1):
        acc_id = f"ACC_{acc_num:04d}"
        for _ in range(s):
            assignments.append((user_ids[idx], acc_id))
            idx += 1

    dim_user = pd.DataFrame(assignments, columns=["user_id", "account_id"])

    # Add some account attributes (simulated but realistic)
    account_ids = dim_user["account_id"].unique()
    plans = rng.choice(["Basic", "Pro", "Enterprise"], size=len(account_ids), p=[0.55, 0.35, 0.10])
    contracts = rng.choice(["Monthly", "Annual"], size=len(account_ids), p=[0.70, 0.30])
    regions = rng.choice(["NA", "EU", "APAC"], size=len(account_ids), p=[0.55, 0.30, 0.15])

    dim_account = pd.DataFrame({
        "account_id": account_ids,
        "plan_type": plans,
        "contract_type": contracts,
        "region": regions,
    })

    # Company size = seat count
    seat_counts = dim_user.groupby("account_id")["user_id"].nunique().rename("seats").reset_index()
    dim_account = dim_account.merge(seat_counts, on="account_id", how="left")

    return dim_user, dim_account

def build_daily_account_usage(events: pd.DataFrame, dim_user: pd.DataFrame) -> pd.DataFrame:
    """
    Convert events to daily usage at account level.
    We'll treat each Invoice as a 'session' and each line item as an 'event'.
    """
    df = events.copy()
    df = df.rename(columns={"CustomerID": "user_id"})
    df["user_id"] = df["user_id"].astype(int)
    df["event_day"] = pd.to_datetime(df["event_day"])

    # Join user -> account
    df = df.merge(dim_user, on="user_id", how="inner")

    # Metrics per account per day
    daily = df.groupby(["account_id", "event_day"]).agg(
        active_users=("user_id", "nunique"),
        sessions=("Invoice", "nunique"),
        events=("StockCode", "count"),
        revenue=("line_revenue", "sum"),
    ).reset_index()

    return daily

def label_account_churn(daily: pd.DataFrame, dim_account: pd.DataFrame) -> pd.DataFrame:
    """
    Define churn dynamically:
    An account churns if it has NO activity for 30 consecutive days.
    We approximate churn_date as (last_active_day + 30).
    """
    # last active day per account
    last_active = daily.groupby("account_id")["event_day"].max().reset_index()
    last_active = last_active.rename(columns={"event_day": "last_active_day"})

    churn = last_active.copy()
    churn["churn_date"] = churn["last_active_day"] + pd.Timedelta(days=INACTIVITY_DAYS_FOR_CHURN)

    # churn_flag: churn_date must be within dataset window to be "observed"
    dataset_max_day = daily["event_day"].max()
    churn["churn_flag"] = (churn["churn_date"] <= dataset_max_day).astype(int)

    # Merge in account attributes (handy later)
    churn = churn.merge(dim_account, on="account_id", how="left")

    return churn[["account_id", "last_active_day", "churn_date", "churn_flag", "plan_type", "contract_type", "region", "seats"]]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading cleaned events parquet...")
    events = pd.read_parquet(EVENTS_PATH)
    print("Events shape:", events.shape)

    print("Building dim_user + dim_account...")
    users = events["CustomerID"].dropna().astype(int)
    dim_user, dim_account = assign_accounts(users, seed=42)

    print("Users:", dim_user["user_id"].nunique(), "Accounts:", dim_account["account_id"].nunique())

    print("Building daily account usage...")
    daily = build_daily_account_usage(events, dim_user)
    print("Daily account usage shape:", daily.shape)
    print("Date range:", daily["event_day"].min(), "→", daily["event_day"].max())

    print("Labelling churn...")
    churn = label_account_churn(daily, dim_account)
    print("Churned accounts (flag=1):", churn["churn_flag"].sum(), "out of", churn.shape[0])

    print("Saving marts to data/mart/ ...")
    dim_user.to_csv(DIM_USER_PATH, index=False)
    dim_account.to_csv(DIM_ACCOUNT_PATH, index=False)
    daily.to_csv(FACT_ACCOUNT_DAILY_PATH, index=False)
    churn.to_csv(FACT_CHURN_PATH, index=False)

    print("Saved:")
    print(" -", DIM_USER_PATH)
    print(" -", DIM_ACCOUNT_PATH)
    print(" -", FACT_ACCOUNT_DAILY_PATH)
    print(" -", FACT_CHURN_PATH)

if __name__ == "__main__":
    main()