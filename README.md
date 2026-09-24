# Customer Churn Predictor (Streamlit)

An interactive Streamlit app that predicts whether a customer will churn using
your trained Random Forest model, and shows the probability of **Yes** (churn)
vs **No** (stay).

## What's included
- `app.py` — the Streamlit app (single prediction + batch CSV scoring + model insights)
- `model.pkl` — your trained RandomForestClassifier (+ feature list)
- `encoders.pkl` — the LabelEncoders used for each categorical column
- `data.csv` — the Telco churn dataset, used for reference stats/charts
- `requirements.txt` — Python dependencies

## How to run locally

1. Install dependencies (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. Your browser will open to `http://localhost:8501`.

## Features

**🔮 Predict tab**
- Fill in a customer's profile (demographics, account, services, billing)
- Get a churn probability gauge, Yes/No probability split, a risk badge
  (Low / Medium / High), and plain-English notes on the key risk drivers
  for that profile

**📁 Batch Prediction tab**
- Upload a CSV of many customers (same columns as the training data)
- Get churn probability + predicted label + risk level for every row
- Download the scored results as CSV

**ℹ️ About the Model tab**
- Model details, feature importance chart, and churn-rate breakdowns by
  contract type and internet service, based on the reference dataset

## Notes
- Keep `model.pkl`, `encoders.pkl`, `data.csv`, and `app.py` in the same folder.
- The model was trained with an older scikit-learn version; you may see an
  `InconsistentVersionWarning` on load — this is harmless but for best
  reliability, try to use a scikit-learn version close to 1.6.x.
