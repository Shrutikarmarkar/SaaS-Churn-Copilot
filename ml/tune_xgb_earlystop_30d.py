import os
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "mart", "feature_table_30d.csv")

def precision_at_k(y_true, y_prob, k_frac=0.05):
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_prob)[:k]
    return y_true.iloc[idx].mean()

def recall_at_k(y_true, y_prob, k_frac=0.05):
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_prob)[:k]
    return y_true.iloc[idx].sum() / max(1, y_true.sum())

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

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ]
    )
    return pre

def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["snapshot_date"])
    print("Loaded:", df.shape, "Label rate:", df["will_churn_30d"].mean())

    train_df, val_df, test_df = time_split(df)
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    print("\nLabel rates:")
    print("Train:", y_train.mean(), "Val:", y_val.mean(), "Test:", y_test.mean())

    # Fit preprocessing ONLY on train (no leakage)
    pre = build_preprocessor(X_train)
    print("\nFitting preprocessor on train...")
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    # imbalance weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = (neg / max(1, pos))
    print("scale_pos_weight:", spw)

    # A small, sensible grid for early-stopped models
    configs = [
        {"name": "es_base", "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"name": "es_low_lr", "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"name": "es_more_reg", "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0},
        {"name": "es_shallow", "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    ]

    results = []
    best_model = None
    best_cfg = None
    best_val_pr = -1

    for cfg in configs:
        print(f"\nTraining {cfg['name']} with early stopping...")

        model = XGBClassifier(
            n_estimators=5000,               # huge cap; early stopping will stop earlier
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_lambda=cfg["reg_lambda"],
            random_state=42,
            n_jobs=4,
            scale_pos_weight=spw,
            eval_metric="aucpr",             # optimize PR-AUC during training
            early_stopping_rounds=50,
        )

        model.fit(
            X_train_t, y_train,
            eval_set=[(X_val_t, y_val)],
            verbose=False,
        )

        val_prob = model.predict_proba(X_val_t)[:, 1]
        val_roc = roc_auc_score(y_val, val_prob)
        val_pr = average_precision_score(y_val, val_prob)
        val_p5 = precision_at_k(y_val, pd.Series(val_prob), 0.05)
        val_r5 = recall_at_k(y_val, pd.Series(val_prob), 0.05)

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(model, "best_ntree_limit", None)

        print("  best_iteration:", model.best_iteration)
        print("  Val ROC-AUC:", val_roc)
        print("  Val PR-AUC :", val_pr)
        print("  Val P@5%   :", val_p5)
        print("  Val R@5%   :", val_r5)

        results.append({
            "config": cfg["name"],
            "best_iteration": model.best_iteration,
            "val_roc_auc": val_roc,
            "val_pr_auc": val_pr,
            "val_p@5%": val_p5,
            "val_r@5%": val_r5
        })

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_model = model
            best_cfg = cfg

    leaderboard = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False)
    print("\n====================")
    print("Leaderboard (by Val PR-AUC)")
    print("====================")
    print(leaderboard.to_string(index=False))

    print("\n🏆 Best config:", best_cfg)
    print("Best best_iteration:", best_model.best_iteration)

    # Evaluate best model on TEST once
    test_prob = best_model.predict_proba(X_test_t)[:, 1]
    test_roc = roc_auc_score(y_test, test_prob)
    test_pr = average_precision_score(y_test, test_prob)
    test_p1 = precision_at_k(y_test, pd.Series(test_prob), 0.01)
    test_r1 = recall_at_k(y_test, pd.Series(test_prob), 0.01)
    test_p5 = precision_at_k(y_test, pd.Series(test_prob), 0.05)
    test_r5 = recall_at_k(y_test, pd.Series(test_prob), 0.05)

    print("\n====================")
    print("Best Early-Stopped Model Test Metrics")
    print("====================")
    print("Test ROC-AUC:", test_roc)
    print("Test PR-AUC :", test_pr)
    print("Test P@1%   :", test_p1)
    print("Test R@1%   :", test_r1)
    print("Test P@5%   :", test_p5)
    print("Test R@5%   :", test_r5)

if __name__ == "__main__":
    main()