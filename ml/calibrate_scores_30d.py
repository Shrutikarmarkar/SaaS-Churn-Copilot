import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "mart", "feature_table_30d.csv")

OUT_CALIBRATED = os.path.join(BASE_DIR, "data", "mart", "churn_scores_latest_calibrated.csv")

HOLDOUT_FRACTION = 0.15  # test is last 15%

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
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ]
    )

def train_xgb_earlystop(X_train_t, y_train, X_val_t, y_val):
    # imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / max(1, pos)

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

    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )
    return model

def platt_calibrator_fit(p_val, y_val):
    """
    Fit sigmoid calibration: y ~ logistic(a * p + b)
    We fit logistic regression on a single feature = raw probability.
    """
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(p_val.reshape(-1, 1), y_val)
    return lr

def platt_calibrate(calibrator, p):
    return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]

def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["snapshot_date"])
    print("Loaded:", df.shape, "Label rate:", df["will_churn_30d"].mean())

    train_df, val_df, test_df = time_split(df)

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    print("\nLabel rates:")
    print("Train:", y_train.mean(), "Val:", y_val.mean(), "Test:", y_test.mean())

    # Preprocess
    pre = build_preprocessor(X_train)
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    print("\nTraining XGB (early stopping)...")
    xgb = train_xgb_earlystop(X_train_t, y_train, X_val_t, y_val)
    print("Best iteration:", xgb.best_iteration)

    # Raw probs
    p_val = xgb.predict_proba(X_val_t)[:, 1]
    p_test = xgb.predict_proba(X_test_t)[:, 1]

    # Fit calibrator on validation
    print("\nFitting Platt (sigmoid) calibrator on validation...")
    calibrator = platt_calibrator_fit(p_val, y_val)

    p_test_cal = platt_calibrate(calibrator, p_test)

    # Compare metrics (ranking should stay similar; calibration improves probability meaning)
    print("\n=== TEST METRICS (RAW) ===")
    print("ROC-AUC:", roc_auc_score(y_test, p_test))
    print("PR-AUC :", average_precision_score(y_test, p_test))
    print("Mean prob:", float(np.mean(p_test)), " True rate:", float(np.mean(y_test)))

    print("\n=== TEST METRICS (CALIBRATED) ===")
    print("ROC-AUC:", roc_auc_score(y_test, p_test_cal))
    print("PR-AUC :", average_precision_score(y_test, p_test_cal))
    print("Mean prob:", float(np.mean(p_test_cal)), " True rate:", float(np.mean(y_test)))

    # Check the overconfidence issue
    mask_raw = p_test > 0.9
    if mask_raw.sum() > 0:
        print("\nRAW:  P>0.9 count:", int(mask_raw.sum()), " actual rate:", float(np.mean(y_test[mask_raw])))
    else:
        print("\nRAW:  No samples with P>0.9")

    mask_cal = p_test_cal > 0.9
    if mask_cal.sum() > 0:
        print("CAL:  P>0.9 count:", int(mask_cal.sum()), " actual rate:", float(np.mean(y_test[mask_cal])))
    else:
        print("CAL:  No samples with calibrated P>0.9 (this is normal)")

    # ---- Produce calibrated latest scores for production ----
    latest_idx = df.groupby("account_id")["snapshot_date"].idxmax()
    latest_df = df.loc[latest_idx].copy()

    X_latest = latest_df.drop(columns=["will_churn_30d", "account_id", "snapshot_date"])
    X_latest_t = pre.transform(X_latest)

    p_latest_raw = xgb.predict_proba(X_latest_t)[:, 1]
    p_latest_cal = platt_calibrate(calibrator, p_latest_raw)

    latest_df["churn_risk_score_raw"] = p_latest_raw
    latest_df["churn_risk_score"] = p_latest_cal  # calibrated score

    # Risk buckets based on calibrated score (more meaningful)
    def bucket(p):
        if p >= 0.25:
            return "High"
        elif p >= 0.10:
            return "Medium"
        else:
            return "Low"

    latest_df["risk_bucket"] = latest_df["churn_risk_score"].apply(bucket)

    out = latest_df[["account_id", "snapshot_date", "churn_risk_score_raw", "churn_risk_score", "risk_bucket"]]
    out = out.sort_values("churn_risk_score", ascending=False)

    out.to_csv(OUT_CALIBRATED, index=False)

    print("\nSaved calibrated latest scores to:")
    print(OUT_CALIBRATED)

    print("\nTop 10 calibrated high-risk accounts:")
    print(out.head(10))

if __name__ == "__main__":
    main()