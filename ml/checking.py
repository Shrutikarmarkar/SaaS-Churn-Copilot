import pandas as pd

scores = pd.read_csv("data/mart/churn_scores_latest.csv")

print(scores["churn_risk_score"].describe())
print("\nTop 20 scores:")
print(scores["churn_risk_score"].head(20))

print("\nScore distribution buckets:")
print(pd.cut(scores["churn_risk_score"], bins=[0,0.1,0.25,0.5,0.75,1]).value_counts())

print("Accounts with score > 0.9:", (scores["churn_risk_score"] > 0.9).sum())
print("Accounts with score > 0.75:", (scores["churn_risk_score"] > 0.75).sum())
print("Total accounts:", len(scores))

