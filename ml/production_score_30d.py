import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "mart", "feature_table_30d.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "mart", "churn_scores_latest.csv")

def time_split(df, date_col="snapshot_date"):
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    i2 = int(n * 0.85)
    return df.iloc[:i2], df.iloc[i2:]

def split_xy(d):
    y = d["will_churn_30d"].astype(int)
    X = d.drop(columns=["will_churn_30d", "account_id", "snapshot_date"])
    return X, y

def build_preprocessor(X_train):
    cat_cols = ["plan_type", "contract_type", "region"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

def assign_risk_bucket(score):
    if score >= 0.25:
        return "High"
    elif score >= 0.10:
        return "Medium"
    else:
        return "Low"

def main():

    df = pd.read_csv(FEATURES_PATH, parse_dates=["snapshot_date"])
    print("Loaded feature table:", df.shape)

    # Split into train+val and test
    train_val_df, test_df = time_split(df)

    X_train_val, y_train_val = split_xy(train_val_df)

    # Fit preprocessing on train+val
    pre = build_preprocessor(X_train_val)
    X_train_val_t = pre.fit_transform(X_train_val)

    # Class imbalance weight
    pos = y_train_val.sum()
    neg = len(y_train_val) - pos
    spw = (neg / max(1, pos))

    print("Training final production model...")

    model = XGBClassifier(
        n_estimators=149,   # from early stopping
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        scale_pos_weight=spw,
        eval_metric="aucpr"
    )

    model.fit(X_train_val_t, y_train_val)

    # Identify latest snapshot per account
    latest_idx = df.groupby("account_id")["snapshot_date"].idxmax()
    latest_df = df.loc[latest_idx].copy()

    X_latest = latest_df.drop(columns=["will_churn_30d", "account_id", "snapshot_date"])
    X_latest_t = pre.transform(X_latest)

    latest_df["churn_risk_score"] = model.predict_proba(X_latest_t)[:, 1]
    latest_df["risk_bucket"] = latest_df["churn_risk_score"].apply(assign_risk_bucket)

    out = latest_df[["account_id", "snapshot_date", "churn_risk_score", "risk_bucket"]]

    out = out.sort_values("churn_risk_score", ascending=False)

    out.to_csv(OUT_PATH, index=False)

    print("\nSaved production churn scores to:")
    print(OUT_PATH)

    print("\nTop 10 high-risk accounts:")
    print(out.head(10))

if __name__ == "__main__":
    main()