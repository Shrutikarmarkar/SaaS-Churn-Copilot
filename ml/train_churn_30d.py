import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.compose import ColumnTransformer   # lets us treat numeric and categorical features differently
from sklearn.preprocessing import OneHotEncoder   # converts categorical features to dummy variables
from sklearn.pipeline import Pipeline   # lets us chain together multiple preprocessing steps
from sklearn.linear_model import LogisticRegression   # the model we're using
from sklearn.impute import SimpleImputer   # fills in missing values

from xgboost import XGBClassifier   # the model we're using

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "mart", "feature_table_30d.csv")

ART_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

def precision_at_k(y_true, y_prob, k_frac=0.01):   # precision at k%
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_prob)[:k]
    return y_true.iloc[idx].mean()

def recall_at_k(y_true, y_prob, k_frac=0.01):   # recall at k%
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_prob)[:k]
    return y_true.iloc[idx].sum() / max(1, y_true.sum())

def time_split(df, date_col="snapshot_date"):   # time-based split: 70% train, 15% val, 15% test
    """
    Time-based split:
      Train: earliest 70%
      Val: next 15%
      Test: last 15%
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    train = df.iloc[:i1]
    val = df.iloc[i1:i2]
    test = df.iloc[i2:]
    return train, val, test

def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["snapshot_date"])   # load the feature table and parse the snapshot_date column as a datetime object
    print("Loaded feature table:", df.shape)   # print the shape of the feature table
    print("Label rate:", df["will_churn_30d"].mean())   # print the label rate (proportion of churned accounts)

    # Drop identifiers we shouldn't feed directly
    y = df["will_churn_30d"].astype(int)   # convert the will_churn_30d column to an integer column
    X = df.drop(columns=["will_churn_30d", "account_id", "snapshot_date"])   # drop the will_churn_30d, account_id, and snapshot_date columns

    train_df, val_df, test_df = time_split(df)   # split the data into train, validation, and test sets

    def split_xy(d):
        y_ = d["will_churn_30d"].astype(int)   # convert the will_churn_30d column to an integer column
        X_ = d.drop(columns=["will_churn_30d", "account_id", "snapshot_date"])
        return X_, y_   # return the X and y values

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)   # split the validation set into X and y values
    X_test, y_test = split_xy(test_df)   # split the test set into X and y values

    print("\nLabel rate by split:")
    print("Train:", y_train.mean())
    print("Val  :", y_val.mean())
    print("Test :", y_test.mean())

    cat_cols = ["plan_type", "contract_type", "region"]   # categorical columns
    num_cols = [c for c in X_train.columns if c not in cat_cols]   # numerical columns

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),   # fill in missing values with the median                           
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ]
    )

    # ---------- Baseline: Logistic Regression ----------
    logreg = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])

    print("\nTraining Logistic Regression...")
    logreg.fit(X_train, y_train)
    val_prob_lr = logreg.predict_proba(X_val)[:, 1]
    test_prob_lr = logreg.predict_proba(X_test)[:, 1]

    print("\n[LogReg] Validation:")
    print("  ROC-AUC:", roc_auc_score(y_val, val_prob_lr))
    print("  PR-AUC :", average_precision_score(y_val, val_prob_lr))
    print("  P@1%   :", precision_at_k(y_val, pd.Series(val_prob_lr), 0.01))
    print("  R@1%   :", recall_at_k(y_val, pd.Series(val_prob_lr), 0.01))
    print("  P@5%   :", precision_at_k(y_val, pd.Series(val_prob_lr), 0.05))
    print("  R@5%   :", recall_at_k(y_val, pd.Series(val_prob_lr), 0.05))

    print("\n[LogReg] Test:")
    print("  ROC-AUC:", roc_auc_score(y_test, test_prob_lr))
    print("  PR-AUC :", average_precision_score(y_test, test_prob_lr))
    print("  P@1%   :", precision_at_k(y_test, pd.Series(test_prob_lr), 0.01))
    print("  R@1%   :", recall_at_k(y_test, pd.Series(test_prob_lr), 0.01))
    print("  P@5%   :", precision_at_k(y_test, pd.Series(test_prob_lr), 0.05))
    print("  R@5%   :", recall_at_k(y_test, pd.Series(test_prob_lr), 0.05))

    # ---------- Strong model: XGBoost ----------
    # scale_pos_weight helps with imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = (neg / max(1, pos))

    xgb = Pipeline(steps=[
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            scale_pos_weight=spw,
            eval_metric="logloss",
        ))
    ])

    print("\nTraining XGBoost...")
    xgb.fit(X_train, y_train)
    val_prob_xgb = xgb.predict_proba(X_val)[:, 1]
    test_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    print("\n[XGB] Validation:")
    print("  ROC-AUC:", roc_auc_score(y_val, val_prob_xgb))
    print("  PR-AUC :", average_precision_score(y_val, val_prob_xgb))
    print("  P@1%   :", precision_at_k(y_val, pd.Series(val_prob_xgb), 0.01))
    print("  R@1%   :", recall_at_k(y_val, pd.Series(val_prob_xgb), 0.01))
    print("  P@5%   :", precision_at_k(y_val, pd.Series(val_prob_xgb), 0.05))
    print("  R@5%   :", recall_at_k(y_val, pd.Series(val_prob_xgb), 0.05))

    print("\n[XGB] Test:")
    print("  ROC-AUC:", roc_auc_score(y_test, test_prob_xgb))
    print("  PR-AUC :", average_precision_score(y_test, test_prob_xgb))
    print("  P@1%   :", precision_at_k(y_test, pd.Series(test_prob_xgb), 0.01))
    print("  R@1%   :", recall_at_k(y_test, pd.Series(test_prob_xgb), 0.01))
    print("  P@5%   :", precision_at_k(y_test, pd.Series(test_prob_xgb), 0.05))
    print("  R@5%   :", recall_at_k(y_test, pd.Series(test_prob_xgb), 0.05))

    print("\nDone.")

if __name__ == "__main__":
    main()