import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0rem;
        background: linear-gradient(90deg, #6366F1, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(127,127,127,0.07);
        border: 1px solid rgba(127,127,127,0.15);
        padding: 1rem 1.2rem;
        border-radius: 12px;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load model / encoders / reference data
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    model = bundle["model"]
    feature_names = bundle["features_names"]
    return model, feature_names, encoders


@st.cache_data
def load_reference_data():
    df = pd.read_csv("data.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


model, FEATURE_NAMES, encoders = load_artifacts()
ref_df = load_reference_data()

CATEGORICAL_COLS = list(encoders.keys())
NUMERIC_COLS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]

FRIENDLY_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract Type",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
}


def encode_row(raw: dict) -> pd.DataFrame:
    """Turn a dict of raw human-readable inputs into the encoded feature row the model expects."""
    row = {}
    for col in FEATURE_NAMES:
        val = raw[col]
        if col in encoders:
            le = encoders[col]
            val = le.transform([val])[0]
        elif col == "SeniorCitizen":
            val = 1 if val == "Yes" else 0
        row[col] = val
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def risk_bucket(prob):
    if prob >= 0.65:
        return "High Risk", "#EF4444", "🔴"
    elif prob >= 0.35:
        return "Medium Risk", "#F59E0B", "🟠"
    else:
        return "Low Risk", "#22C55E", "🟢"


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="main-header">📉 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Enter a customer\'s profile to estimate the probability they will churn, '
    'powered by a Random Forest model trained on the Telco Customer Churn dataset.</div>',
    unsafe_allow_html=True,
)

tab_predict, tab_batch, tab_about = st.tabs(["🔮 Predict", "📁 Batch Prediction", "ℹ️ About the Model"])

# ============================================================================
# TAB 1: SINGLE PREDICTION
# ============================================================================
with tab_predict:
    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
        st.subheader("Customer Profile")

        with st.form("churn_form"):
            st.markdown("**👤 Demographics**")
            c1, c2 = st.columns(2)
            gender = c1.selectbox("Gender", encoders["gender"].classes_)
            senior = c2.selectbox("Senior Citizen", ["No", "Yes"])
            c3, c4 = st.columns(2)
            partner = c3.selectbox("Has Partner", encoders["Partner"].classes_)
            dependents = c4.selectbox("Has Dependents", encoders["Dependents"].classes_)

            st.markdown("**📅 Account**")
            c5, c6 = st.columns(2)
            tenure = c5.slider("Tenure (months)", 0, 72, 12)
            contract = c6.selectbox("Contract Type", encoders["Contract"].classes_)
            c7, c8 = st.columns(2)
            paperless = c7.selectbox("Paperless Billing", encoders["PaperlessBilling"].classes_)
            payment = c8.selectbox("Payment Method", encoders["PaymentMethod"].classes_)

            st.markdown("**📞 Phone & Internet**")
            c9, c10 = st.columns(2)
            phone_service = c9.selectbox("Phone Service", encoders["PhoneService"].classes_)
            internet_service = c10.selectbox("Internet Service", encoders["InternetService"].classes_)

            if phone_service == "No":
                multiple_lines = "No phone service"
                st.caption("Multiple Lines: No phone service")
            else:
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

            st.markdown("**🛡️ Add-on Services**")
            if internet_service == "No":
                st.caption("No internet service — add-ons default to 'No internet service'")
                online_security = online_backup = device_protection = "No internet service"
                tech_support = streaming_tv = streaming_movies = "No internet service"
            else:
                c11, c12, c13 = st.columns(3)
                online_security = c11.selectbox("Online Security", ["No", "Yes"])
                online_backup = c12.selectbox("Online Backup", ["No", "Yes"])
                device_protection = c13.selectbox("Device Protection", ["No", "Yes"])
                c14, c15 = st.columns(2)
                tech_support = c14.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = c15.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

            st.markdown("**💵 Billing**")
            c16, c17 = st.columns(2)
            monthly_charges = c16.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
            total_charges = c17.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(round(monthly_charges * max(tenure, 1), 2)), step=10.0)

            submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    with right:
        st.subheader("Prediction Result")

        if submitted:
            raw = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            X = encode_row(raw)
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            # figure out which index is "Yes"/1 (churn)
            churn_idx = classes.index("Yes") if "Yes" in classes else (classes.index(1) if 1 in classes else 1)
            no_idx = 1 - churn_idx if len(classes) == 2 else 0

            p_churn = proba[churn_idx]
            p_stay = proba[no_idx]

            label, color, emoji = risk_bucket(p_churn)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_churn * 100,
                number={"suffix": "%", "font": {"size": 44}},
                title={"text": "Churn Probability", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(34,197,94,0.18)"},
                        {"range": [35, 65], "color": "rgba(245,158,11,0.18)"},
                        {"range": [65, 100], "color": "rgba(239,68,68,0.18)"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 4},
                        "thickness": 0.85,
                        "value": p_churn * 100,
                    },
                },
            ))
            fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f'<div style="text-align:center; margin-top:-1rem;">'
                f'<span class="risk-badge" style="background:{color}22; color:{color}; border:1.5px solid {color};">'
                f'{emoji} {label}</span></div>',
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Probability: Churn (Yes)", f"{p_churn*100:.1f}%")
            m2.metric("Probability: Stay (No)", f"{p_stay*100:.1f}%")

            st.progress(float(p_churn))

            st.markdown("---")
            st.markdown("**📌 Key factors typically driving this outlook:**")
            notes = []
            if contract == "Month-to-month":
                notes.append("Month-to-month contracts have the highest churn rate — no lock-in.")
            if tenure <= 6:
                notes.append("Low tenure (≤6 months) customers churn far more often.")
            if internet_service == "Fiber optic":
                notes.append("Fiber optic customers historically churn more than DSL customers.")
            if payment == "Electronic check":
                notes.append("Electronic check payers tend to have higher churn rates.")
            if online_security == "No" and internet_service != "No":
                notes.append("No online security add-on correlates with higher churn.")
            if tech_support == "No" and internet_service != "No":
                notes.append("No tech support add-on correlates with higher churn.")
            if monthly_charges > 80:
                notes.append("High monthly charges relative to the base tier can raise churn risk.")
            if not notes:
                notes.append("Profile aligns with typically low-churn customer segments (longer tenure, longer contract).")
            for n in notes:
                st.write(f"- {n}")

            with st.expander("View encoded feature vector sent to the model"):
                st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)

        else:
            st.info("Fill in the customer profile on the left and click **Predict Churn** to see results.")
            st.markdown("##### Dataset churn rate (reference)")
            churn_rate = (ref_df["Churn"] == "Yes").mean()
            st.metric("Overall churn rate in training data", f"{churn_rate*100:.1f}%")

# ============================================================================
# TAB 2: BATCH PREDICTION
# ============================================================================
with tab_batch:
    st.subheader("Batch Prediction from CSV")
    st.write(
        "Upload a CSV with the same columns as the training data "
        f"(`{', '.join(FEATURE_NAMES)}`) to score many customers at once."
    )

    up = st.file_uploader("Upload customer CSV", type=["csv"])

    if up is not None:
        try:
            batch_df = pd.read_csv(up)
            missing = [c for c in FEATURE_NAMES if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                work = batch_df.copy()
                work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce")
                work["TotalCharges"] = work["TotalCharges"].fillna(work["TotalCharges"].median())

                for col in CATEGORICAL_COLS:
                    le = encoders[col]
                    if col == "SeniorCitizen":
                        continue
                    known = set(le.classes_)
                    unknown_mask = ~work[col].astype(str).isin(known)
                    if unknown_mask.any():
                        st.warning(f"{unknown_mask.sum()} rows had unrecognized values in '{col}' and were set to the most common category.")
                        work.loc[unknown_mask, col] = le.classes_[0]
                    work[col] = le.transform(work[col].astype(str))

                if "SeniorCitizen" in work.columns and work["SeniorCitizen"].dtype == object:
                    work["SeniorCitizen"] = work["SeniorCitizen"].map({"Yes": 1, "No": 0}).fillna(work["SeniorCitizen"])

                X_batch = work[FEATURE_NAMES]
                proba_batch = model.predict_proba(X_batch)
                classes = list(model.classes_)
                churn_idx = classes.index("Yes") if "Yes" in classes else (classes.index(1) if 1 in classes else 1)

                result = batch_df.copy()
                result["Churn_Probability"] = proba_batch[:, churn_idx]
                result["Stay_Probability"] = 1 - result["Churn_Probability"]
                result["Predicted"] = np.where(result["Churn_Probability"] >= 0.5, "Yes", "No")
                result["Risk_Level"] = result["Churn_Probability"].apply(lambda p: risk_bucket(p)[0])

                st.success(f"Scored {len(result)} customers.")

                colA, colB, colC = st.columns(3)
                colA.metric("Predicted to Churn", int((result["Predicted"] == "Yes").sum()))
                colB.metric("Predicted to Stay", int((result["Predicted"] == "No").sum()))
                colC.metric("Avg. Churn Probability", f"{result['Churn_Probability'].mean()*100:.1f}%")

                st.dataframe(
                    result.sort_values("Churn_Probability", ascending=False),
                    use_container_width=True,
                    height=420,
                )

                csv_out = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download scored results as CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")
    else:
        st.caption("No file uploaded yet. You can also try the built-in sample dataset below.")
        if st.button("Use sample Telco dataset (first 25 rows)"):
            st.session_state["use_sample"] = True

        if st.session_state.get("use_sample"):
            sample = ref_df.drop(columns=["customerID", "Churn"], errors="ignore").head(25)
            st.dataframe(sample, use_container_width=True)

# ============================================================================
# TAB 3: ABOUT
# ============================================================================
with tab_about:
    st.subheader("About This Model")
    st.write(
        "This app uses a **Random Forest Classifier** trained on the IBM Telco Customer Churn dataset "
        "(7,043 customers, 19 input features) to estimate the probability that a customer will churn."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model type:** `RandomForestClassifier` (scikit-learn)")
        st.markdown(f"**Number of features:** {len(FEATURE_NAMES)}")
        st.markdown(f"**Training rows (reference data):** {len(ref_df):,}")
    with c2:
        churn_rate = (ref_df["Churn"] == "Yes").mean()
        st.markdown(f"**Historical churn rate:** {churn_rate*100:.1f}%")
        st.markdown(f"**Classes:** {', '.join(map(str, model.classes_))}")

    st.markdown("---")
    st.markdown("**Feature importance (from the trained Random Forest)**")
    try:
        importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=importances.values,
            y=[FRIENDLY_LABELS.get(f, f) for f in importances.index],
            orientation="h",
            marker=dict(color=importances.values, colorscale="Viridis"),
        ))
        fig_imp.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Relative Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Feature importances not available for this model.")

    st.markdown("---")
    st.markdown("**Churn rate by key segments (reference data)**")
    seg1, seg2 = st.columns(2)
    with seg1:
        by_contract = ref_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
        fig_c = go.Figure(go.Bar(x=by_contract.index, y=by_contract.values,
                                  marker_color=["#EF4444", "#F59E0B", "#22C55E"]))
        fig_c.update_layout(title="Churn % by Contract Type", height=320, yaxis_title="Churn %",
                             margin=dict(t=40, b=10))
        st.plotly_chart(fig_c, use_container_width=True)
    with seg2:
        by_internet = ref_df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
        fig_i = go.Figure(go.Bar(x=by_internet.index, y=by_internet.values,
                                  marker_color=["#6366F1", "#EC4899", "#22C55E"]))
        fig_i.update_layout(title="Churn % by Internet Service", height=320, yaxis_title="Churn %",
                             margin=dict(t=40, b=10))
        st.plotly_chart(fig_i, use_container_width=True)

    st.markdown("---")
    st.caption(
        "⚠️ This tool provides a statistical estimate based on historical patterns and should be used "
        "as one input among several when making retention decisions — not as a sole determinant."
    )
