"""
train_model.py — Generates synthetic creditcard data and trains the model.
Run this once to create fraud_model.pkl and scaler.pkl
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

print("Generating synthetic credit card dataset...")

np.random.seed(42)
n_legit = 9800
n_fraud = 200
n_total = n_legit + n_fraud

# 28 PCA-like features V1-V28, Time, Amount, Class
legit_data = np.random.randn(n_legit, 28) * np.array(
    [1.5, 1.2, 2.0, 1.1, 1.3, 0.9, 1.4, 1.0, 1.2, 0.8,
     1.1, 1.3, 0.9, 1.5, 1.2, 1.0, 1.1, 1.4, 0.8, 1.2,
     1.0, 0.9, 1.3, 1.1, 0.8, 1.0, 0.9, 1.2]
)

# Fraud transactions have shifted distributions (make them distinguishable)
fraud_data = np.random.randn(n_fraud, 28) * 1.5 + np.array(
    [-2.5, 1.8, -3.2, 1.0, -0.5, -0.7, 2.1, -0.3, 0.8, -2.1,
      0.5, 1.1, -0.4, 0.9, 0.3, 0.2, -1.2, 0.4, 0.6, -0.8,
      0.3, -0.5, 0.7, 0.2, -0.3, 0.4, -0.6, 0.9]
)

features = np.vstack([legit_data, fraud_data])
time_vals = np.random.uniform(0, 172800, n_total)
legit_amounts = np.abs(np.random.exponential(80, n_legit))
fraud_amounts = np.abs(np.random.exponential(200, n_fraud))
amounts = np.concatenate([legit_amounts, fraud_amounts])
labels = np.array([0]*n_legit + [1]*n_fraud)

cols = [f"V{i}" for i in range(1, 29)]
df = pd.DataFrame(features, columns=cols)
df.insert(0, "Time", time_vals)
df["Amount"] = amounts
df["Class"] = labels

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Dataset: {df.shape[0]} rows | Fraud: {df['Class'].sum()} | Legit: {(df['Class']==0).sum()}")

# Save dataset
df.to_csv("/home/claude/shieldpay_app/creditcard_synthetic.csv", index=False)

# ─── IQR Outlier Removal ───
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
df_clean = df[(df['Amount'] >= lower_limit) & (df['Amount'] <= upper_limit)].copy()
print(f"After outlier removal: {df_clean.shape[0]} rows")

X = df_clean.drop("Class", axis=1)
y = df_clean["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── Logistic Regression ───
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_metrics = {
    "accuracy":  round(accuracy_score(y_test, lr_pred), 4),
    "precision": round(precision_score(y_test, lr_pred, zero_division=0), 4),
    "recall":    round(recall_score(y_test, lr_pred, zero_division=0), 4),
    "f1":        round(f1_score(y_test, lr_pred, zero_division=0), 4),
    "cm":        confusion_matrix(y_test, lr_pred).tolist(),
    "name":      "Logistic Regression"
}

# ─── Naive Bayes ───
nb = GaussianNB()
nb.fit(X_train_sc, y_train)
nb_pred = nb.predict(X_test_sc)
nb_metrics = {
    "accuracy":  round(accuracy_score(y_test, nb_pred), 4),
    "precision": round(precision_score(y_test, nb_pred, zero_division=0), 4),
    "recall":    round(recall_score(y_test, nb_pred, zero_division=0), 4),
    "f1":        round(f1_score(y_test, nb_pred, zero_division=0), 4),
    "cm":        confusion_matrix(y_test, nb_pred).tolist(),
    "name":      "Gaussian Naive Bayes"
}

print("\n── Logistic Regression ──")
for k, v in lr_metrics.items():
    if k != "cm": print(f"  {k}: {v}")
print("\n── Naive Bayes ──")
for k, v in nb_metrics.items():
    if k != "cm": print(f"  {k}: {v}")

# ─── Save artefacts ───
pickle.dump(lr,      open("/home/claude/shieldpay_app/fraud_model.pkl", "wb"))
pickle.dump(nb,      open("/home/claude/shieldpay_app/nb_model.pkl",    "wb"))
pickle.dump(scaler,  open("/home/claude/shieldpay_app/scaler.pkl",      "wb"))
pickle.dump({"lr": lr_metrics, "nb": nb_metrics},
            open("/home/claude/shieldpay_app/metrics.pkl",  "wb"))
pickle.dump({"Q1": Q1, "Q3": Q3, "IQR": IQR,
             "lower": lower_limit, "upper": upper_limit,
             "n_total": n_total, "n_fraud": df['Class'].sum(),
             "n_clean": df_clean.shape[0],
             "feature_cols": list(X.columns)},
            open("/home/claude/shieldpay_app/meta.pkl", "wb"))
# Save a small sample for the scatter/distribution charts
df_clean.head(500).to_csv("/home/claude/shieldpay_app/sample.csv", index=False)

print("\nAll artefacts saved successfully!")
