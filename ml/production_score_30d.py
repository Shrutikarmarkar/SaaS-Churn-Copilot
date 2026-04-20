import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "mart", "feature_table_30d.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "mart", "churn_scores_latest.csv")

def time_split(df, date_col="snapshot_date"):
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]

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

def fit_platt_calibrator(p_val, y_val):
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(p_val.reshape(-1, 1), y_val)
    return lr

def apply_calibration(calibrator, p):
    return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]

def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["snapshot_date"])
    print("Loaded feature table:", df.shape)
    print("Overall churn rate:", round(df["will_churn_30d"].mean(), 4))

    train_df, val_df, test_df = time_split(df)

    X_train, y_train = split_xy(train_df)
    X_val,   y_val   = split_xy(val_df)
    X_test,  y_test  = split_xy(test_df)

    print(f"\nSplit sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Churn rate — train: {y_train.mean():.4f}, val: {y_val.mean():.4f}, test: {y_test.mean():.4f}")

    pre = build_preprocessor(X_train)
    X_train_t = pre.fit_transform(X_train)
    X_val_t   = pre.transform(X_val)
    X_test_t  = pre.transform(X_test)

    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / max(1, pos)

    print("\nTraining XGBoost with early stopping...")
    model = XGBClassifier(
        n_estimators=5000,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
    )
    model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
    print("Best iteration:", model.best_iteration)

    p_val_raw  = model.predict_proba(X_val_t)[:, 1]
    p_test_raw = model.predict_proba(X_test_t)[:, 1]

    # Fit Platt sigmoid calibrator on val — maps overconfident raw scores to honest probabilities
    print("\nFitting Platt calibrator on validation set...")
    calibrator = fit_platt_calibrator(p_val_raw, y_val)
    p_test_cal = apply_calibration(calibrator, p_test_raw)

    print("\n=== Calibration diagnostics (test set) ===")
    print(f"True churn rate:          {y_test.mean():.4f}")
    print(f"Raw score mean:           {p_test_raw.mean():.4f}  (should be >> true rate if overconfident)")
    print(f"Calibrated score mean:    {p_test_cal.mean():.4f}  (should be close to true rate)")
    print(f"ROC-AUC  raw / cal:       {roc_auc_score(y_test, p_test_raw):.4f} / {roc_auc_score(y_test, p_test_cal):.4f}")
    print(f"PR-AUC   raw / cal:       {average_precision_score(y_test, p_test_raw):.4f} / {average_precision_score(y_test, p_test_cal):.4f}")

    # Score the latest snapshot for every account
    latest_idx = df.groupby("account_id")["snapshot_date"].idxmax()
    latest_df = df.loc[latest_idx].copy()

    X_latest   = latest_df.drop(columns=["will_churn_30d", "account_id", "snapshot_date"])
    X_latest_t = pre.transform(X_latest)

    p_latest_raw = model.predict_proba(X_latest_t)[:, 1]
    p_latest_cal = apply_calibration(calibrator, p_latest_raw)

    latest_df["model_score_raw"]      = p_latest_raw   # internal — overconfident, not shown to users
    latest_df["churn_risk_calibrated"] = p_latest_cal   # calibrated probability, honest

    out = latest_df[["account_id", "snapshot_date", "model_score_raw", "churn_risk_calibrated"]]
    out = out.sort_values("churn_risk_calibrated", ascending=False)

    out.to_csv(OUT_PATH, index=False)

    print("\nSaved production scores to:", OUT_PATH)
    print("\nTop 10 accounts by calibrated risk:")
    print(out[["account_id", "snapshot_date", "churn_risk_calibrated", "model_score_raw"]].head(10))

if __name__ == "__main__":
    main()
