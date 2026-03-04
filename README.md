<<<<<<< HEAD
# ShieldPay — Credit Card Fraud Detection Web App

A full-stack ML-powered fraud detection web application using Flask + Scikit-Learn.

## Features
- 🔍 **Live Fraud Detection** — Input 30 transaction features (Time, V1–V28, Amount) and get instant verdict
- 🤖 **Dual ML Models** — Logistic Regression + Gaussian Naive Bayes with ensemble scoring
- 📊 **Full EDA Analysis** — IQR outlier removal, scatter plots, boxplots, confusion matrices
- 📈 **Model Metrics** — Accuracy, Precision, Recall, F1 Score side-by-side comparison
- ⚡ **Quick Fill** — One-click random fraud/legit sample population

## Setup

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn
```

### 2. Add your dataset (optional)
Place `creditcard.csv` in the app folder.
If no CSV is found, a synthetic dataset is auto-generated.

### 3. Train the model
```bash
python train_model.py
```
This generates:
- `fraud_model.pkl` — Logistic Regression model
- `nb_model.pkl` — Gaussian Naive Bayes model
- `scaler.pkl` — StandardScaler
- `metrics.pkl` — Evaluation metrics
- `meta.pkl` — Dataset metadata

### 4. Run the app
```bash
python app.py
```
Open **http://localhost:5050** in your browser.

## Pages
| Page | Description |
|------|-------------|
| **Home** | Overview, model stats, live detection log |
| **Detect Fraud** | Input form → instant fraud verdict with risk score |
| **Analysis** | Full EDA: plots, boxplots, confusion matrices, metrics |

## Input Features
The form accepts the standard creditcard.csv columns:
- `Time` — Seconds since first transaction
- `V1–V28` — PCA-transformed features
- `Amount` — Transaction amount in USD

## Using Your Own creditcard.csv
Edit `train_model.py` line:
```python
df = pd.read_csv("creditcard.csv")  # your file path here
```
Then retrain with `python train_model.py`.
=======
# Credit_Card_Fraud_Detection
A machine learning project to identify fraudulent credit card transactions by analyzing transaction data and handling class imbalance for accurate detection.
>>>>>>> 54815a55ed0ce7b04f0d17d5698dadb2027bc5e1
