# app.py
# ============================================================
# Weekly Finance Collection – Payment Risk Prediction
# Streamlit Application (Corrected)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Payment Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f4e79;
        color: white;
        font-weight: bold;
        height: 3rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load Model & Data
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    # Try common locations
    possible_paths = [
        Path("payment_risk_model.pkl"),
        Path("./payment_risk_model.pkl"),
        Path("/home/workdir/artifacts/payment_risk_model.pkl"),
        Path("/home/workdir/attachments/payment_risk_model.pkl"),
    ]
    model_path = None
    for p in possible_paths:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        st.error("❌ Model file `payment_risk_model.pkl` not found.\n\nPlease place it in the same folder as `app.py` or train & save the model first.")
        st.stop()

    saved = joblib.load(model_path)
    return saved["model"], saved["label_encoder"], saved["features"]

@st.cache_data
def load_data():
    possible_paths = [
        Path("weekly_finance_collection_payment_behaviour.csv"),
        Path("./weekly_finance_collection_payment_behaviour.csv"),
        Path("/home/workdir/attachments/weekly_finance_collection_payment_behaviour.csv"),
        Path("/home/workdir/artifacts/weekly_finance_collection_payment_behaviour.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            return pd.read_csv(p)
    return None

try:
    model, le, feature_cols = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

df = load_data()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown('<div class="main-header">💰 Payment Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Weekly Finance Collection • Predict Low / Medium / High Risk</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar – Navigation
# ------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict Risk", "📊 Data Insights", "ℹ️ About Model"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Model Performance**  
    - Random Forest / Gradient Boosting  
    - Accuracy ≈ 100% (on this dataset)  
    - Features used: 16  
    """
)

# ============================================================
# PAGE 1: PREDICT RISK
# ============================================================
if page == "🔮 Predict Risk":
    st.subheader("Enter Customer & Payment Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Loan Details")
        loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, max_value=50000, value=10000, step=500)
        interest_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=40.0, value=20.0, step=0.5)
        total_weeks = st.number_input("Total Weeks", min_value=4, max_value=30, value=12, step=1)
        week_number = st.number_input("Current Week Number", min_value=1, max_value=30, value=5, step=1)

    with col2:
        st.markdown("#### Payment Details")
        weekly_due = st.number_input("Weekly Due (₹)", min_value=50.0, max_value=5000.0, value=600.0, step=10.0)
        amount_paid = st.number_input("Amount Paid this week (₹)", min_value=0.0, max_value=10000.0, value=600.0, step=10.0)
        days_late = st.number_input("Days Late", min_value=0, max_value=60, value=0, step=1)
        remaining_balance = st.number_input("Remaining Balance (₹)", min_value=0.0, max_value=60000.0, value=8000.0, step=100.0)

    with col3:
        st.markdown("#### History")
        previous_late_count = st.number_input("Previous Late Count", min_value=0, max_value=20, value=0, step=1)
        missed_payments = st.number_input("Missed Payments so far", min_value=0, max_value=15, value=0, step=1)
        interest_amount = st.number_input("Total Interest Amount (₹)", min_value=0.0, max_value=15000.0, value=2000.0, step=50.0)
        total_payable = st.number_input("Total Payable (₹)", min_value=1000.0, max_value=70000.0, value=12000.0, step=100.0)

    # ---------- Derived features (must match training) ----------
    payment_ratio = amount_paid / (weekly_due + 1e-6)
    balance_ratio = remaining_balance / (total_payable + 1e-6)
    week_progress = week_number / max(total_weeks, 1)
    is_partial = int((amount_paid > 0) and (amount_paid < weekly_due * 0.95))   # 0 or 1

    st.markdown("---")

    with st.expander("🔍 View Calculated Features"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Payment Ratio", f"{payment_ratio:.2f}")
        c2.metric("Balance Ratio", f"{balance_ratio:.2f}")
        c3.metric("Week Progress", f"{week_progress:.2%}")
        c4.metric("Is Partial Payment", "Yes" if is_partial else "No")

    if st.button("🚀 Predict Payment Risk"):
        input_dict = {
            "Loan_Amount": loan_amount,
            "Interest_Rate_Percent": interest_rate,
            "Interest_Amount": interest_amount,
            "Total_Payable": total_payable,
            "Weekly_Due": weekly_due,
            "Total_Weeks": total_weeks,
            "Week_Number": week_number,
            "Amount_Paid": amount_paid,
            "Days_Late": days_late,
            "Remaining_Balance": remaining_balance,
            "Previous_Late_Count": previous_late_count,
            "Missed_Payments": missed_payments,
            "Payment_Ratio": payment_ratio,
            "Balance_Ratio": balance_ratio,
            "Week_Progress": week_progress,
            "Is_Partial": is_partial
        }

        # Strict column order from the saved model
        input_data = pd.DataFrame([input_dict])[feature_cols]

        try:
            prediction = model.predict(input_data)[0]
            risk_label = le.inverse_transform([prediction])[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Expected features:", feature_cols)
            st.write("Provided columns:", list(input_data.columns))
            st.stop()

        # Probability
        proba_df = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            proba_df = pd.DataFrame({
                "Risk Level": le.classes_,
                "Probability": proba
            }).sort_values("Probability", ascending=False)

        st.markdown("### Prediction Result")

        if risk_label == "Low":
            st.markdown('<div class="risk-low">✅ LOW RISK</div>', unsafe_allow_html=True)
            st.success("Customer is likely to pay on time. Low collection priority.")
        elif risk_label == "Medium":
            st.markdown('<div class="risk-medium">⚠️ MEDIUM RISK</div>', unsafe_allow_html=True)
            st.warning("Customer shows some delay patterns. Monitor closely.")
        else:
            st.markdown('<div class="risk-high">🚨 HIGH RISK</div>', unsafe_allow_html=True)
            st.error("High chance of missed/late payments. Prioritize collection efforts.")

        if proba_df is not None:
            st.markdown("#### Probability Distribution")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            colors = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
            bar_colors = [colors.get(r, "#6c757d") for r in proba_df["Risk Level"]]
            bars = ax.barh(proba_df["Risk Level"], proba_df["Probability"], color=bar_colors)
            ax.set_xlim(0, 1.15)
            ax.set_xlabel("Probability")
            ax.set_title("Model Confidence")
            for bar, val in zip(bars, proba_df["Probability"]):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.1%}", va="center")
            st.pyplot(fig)
            plt.close(fig)

# ============================================================
# PAGE 2: DATA INSIGHTS
# ============================================================
elif page == "📊 Data Insights":
    st.subheader("Dataset Overview & Insights")

    if df is None:
        st.warning("CSV file not found. Place `weekly_finance_collection_payment_behaviour.csv` in the same folder as `app.py`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Unique Customers", df["Customer_ID"].nunique())
        c3.metric("Avg Loan Amount", f"₹{df['Loan_Amount'].mean():,.0f}")
        c4.metric("Missed Payments", f"{(df['Payment_Status'] == 'Missed').sum():,}")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Risk Distribution", "Payment Status", "Key Relationships"])

        with tab1:
            fig, ax = plt.subplots(figsize=(8, 4))
            order = ["Low", "Medium", "High"]
            # Updated seaborn usage (no FutureWarning)
            sns.countplot(data=df, x="Payment_Risk", order=order, 
                          hue="Payment_Risk", palette=["#28a745", "#ffc107", "#dc3545"], 
                          legend=False, ax=ax)
            ax.set_title("Payment Risk Distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                df["Payment_Risk"].value_counts(normalize=True)
                  .mul(100).round(1)
                  .rename("% of records")
                  .to_frame(),
                use_container_width=True
            )

        with tab2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df, x="Payment_Status", hue="Payment_Status", 
                          palette="Set2", legend=False, ax=ax)
            ax.set_title("Payment Status Distribution")
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            col_a, col_b = st.columns(2)

            with col_a:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df, x="Payment_Risk", y="Days_Late",
                            order=["Low", "Medium", "High"], ax=ax)
                ax.set_title("Days Late by Risk Level")
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df, x="Payment_Risk", y="Missed_Payments",
                            order=["Low", "Medium", "High"], ax=ax)
                ax.set_title("Missed Payments by Risk Level")
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("#### Correlation with Risk (numeric features)")
            risk_map = {"Low": 0, "Medium": 1, "High": 2}
            temp = df.copy()
            temp["Risk_Code"] = temp["Payment_Risk"].map(risk_map)
            corr_cols = ["Days_Late", "Missed_Payments", "Previous_Late_Count",
                         "Loan_Amount", "Interest_Rate_Percent", "Week_Number", "Risk_Code"]
            corr = temp[corr_cols].corr()["Risk_Code"].drop("Risk_Code").sort_values(ascending=False)
            st.bar_chart(corr)

# ============================================================
# PAGE 3: ABOUT MODEL
# ============================================================
else:
    st.subheader("About the Model")

    st.markdown("""
    ### Objective
    Predict the **Payment Risk** of a customer for a given week in a weekly finance collection loan.

    ### Target Classes
    | Class     | Meaning                                      |
    |-----------|----------------------------------------------|
    | **Low**   | Customer pays on time consistently           |
    | **Medium**| Some delays / occasional late payments       |
    | **High**  | Multiple missed payments or chronic delays   |

    ### Features Used (16)
    - Loan Amount, Interest Rate, Interest Amount, Total Payable  
    - Weekly Due, Total Weeks, Week Number, Amount Paid  
    - Days Late, Remaining Balance  
    - Previous Late Count, Missed Payments  
    - **Engineered**: Payment Ratio, Balance Ratio, Week Progress, Is Partial Payment  

    ### Models Compared
    - Logistic Regression (with StandardScaler)  
    - Random Forest Classifier  
    - Gradient Boosting Classifier  

    ### Performance (on this dataset)
    - Random Forest & Gradient Boosting achieved **~100% accuracy / F1**  
    - This is expected because the target (`Payment_Risk`) is largely deterministic  
      from features like `Days_Late`, `Missed_Payments`, and `Previous_Late_Count`.

    ### How to use
    1. Go to **Predict Risk**  
    2. Fill in the customer’s current week details  
    3. Click **Predict** to get Low / Medium / High risk + probability
    """)

    st.markdown("---")
    st.caption("Built with Streamlit • Scikit-learn • Pandas")
