import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ───────────────────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────
# Custom CSS for a Premium Look
# ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1.05rem;
    }

    .prediction-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }
    .prediction-safe {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    .prediction-risk {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    .prediction-card h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .prediction-card .probability {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .safe-color { color: #155724; }
    .risk-color { color: #721c24; }

    .metric-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1rem 0;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        min-width: 120px;
    }
    .metric-box .label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-box .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1f2937;
    }

    .sidebar .sidebar-content {
        background: #f8f9fa;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #9ca3af;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────
# Load Model & Scaler
# ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best_churn_model.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model files not found! Please ensure `models/best_churn_model.joblib` "
                 "and `models/scaler.joblib` are present.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()


# ───────────────────────────────────────────────────────────
# Header
# ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Customer Churn Predictor</h1>
    <p>Enter customer details to predict whether they will leave the bank</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ───────────────────────────────────────────────────────────
# Sidebar — Input Form
# ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Customer Details")
    st.markdown("Fill in the customer's information below:")

    st.markdown("#### Demographics")
    age = st.slider("Age", min_value=18, max_value=92, value=35, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    st.markdown("---")
    st.markdown("#### Banking Info")
    credit_score = st.slider("Credit Score", min_value=350, max_value=850, value=650, step=1)
    tenure = st.slider("Tenure (years with bank)", min_value=0, max_value=10, value=5, step=1)
    balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0,
                               value=75000.0, step=1000.0, format="%.2f")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=200000.0,
                                        value=100000.0, step=1000.0, format="%.2f")

    st.markdown("---")
    st.markdown("#### Products & Activity")
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])


# ───────────────────────────────────────────────────────────
# Prepare Features
# ───────────────────────────────────────────────────────────
def prepare_input(credit_score, gender, age, tenure, balance, num_products,
                  has_cr_card, is_active, estimated_salary, geography):
    """Build a feature DataFrame matching the training schema."""
    gender_encoded = 1 if gender == "Male" else 0
    has_card = 1 if has_cr_card == "Yes" else 0
    active = 1 if is_active == "Yes" else 0

    # One-hot encoding for Geography (drop_first=True → France is baseline)
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    features = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_card],
        "IsActiveMember": [active],
        "EstimatedSalary": [estimated_salary],
        "Geography_Germany": [geo_germany],
        "Geography_Spain": [geo_spain],
    })
    return features


# ───────────────────────────────────────────────────────────
# Prediction
# ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

if predict_btn:
    # Prepare and scale
    input_df = prepare_input(credit_score, gender, age, tenure, balance,
                              num_products, has_cr_card, is_active,
                              estimated_salary, geography)
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    churn_prob = probability[1] * 100
    stay_prob = probability[0] * 100

    # Display Result
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-card prediction-risk">
            <h2 class="risk-color">⚠️ HIGH CHURN RISK</h2>
            <div class="probability risk-color">{churn_prob:.1f}%</div>
            <p style="color: #721c24; font-size: 1.1rem;">
                This customer is likely to <strong>leave the bank</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-card prediction-safe">
            <h2 class="safe-color">✅ LOW CHURN RISK</h2>
            <div class="probability safe-color">{stay_prob:.1f}%</div>
            <p style="color: #155724; font-size: 1.1rem;">
                This customer is likely to <strong>stay with the bank</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Metrics Row
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="label">Stay Probability</div>
            <div class="value" style="color: #28a745;">{stay_prob:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="label">Churn Probability</div>
            <div class="value" style="color: #dc3545;">{churn_prob:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show input summary
    st.markdown("---")
    st.markdown("#### 📊 Input Summary")

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown(f"""
        | Feature | Value |
        |---------|-------|
        | **Age** | {age} |
        | **Gender** | {gender} |
        | **Geography** | {geography} |
        | **Credit Score** | {credit_score} |
        | **Tenure** | {tenure} years |
        """)
    with summary_col2:
        st.markdown(f"""
        | Feature | Value |
        |---------|-------|
        | **Balance** | ${balance:,.2f} |
        | **Salary** | ${estimated_salary:,.2f} |
        | **Products** | {num_products} |
        | **Credit Card** | {has_cr_card} |
        | **Active** | {is_active} |
        """)

else:
    # Default state — show instructions
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #6b7280;">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">👈</p>
        <p style="font-size: 1.1rem;">
            Fill in the customer details in the <strong>sidebar</strong>,<br>
            then click <strong>"Predict Churn"</strong> to see the result.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show some example scenarios
    st.markdown("---")
    st.markdown("#### 🧪 Try These Example Scenarios")

    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("""
        **👴 High-Risk Customer**
        - Age: 55, Germany, Female
        - Balance: $120,000
        - 1 Product, Inactive
        """)
    with ex_col2:
        st.markdown("""
        **👩 Low-Risk Customer**
        - Age: 30, France, Male
        - Balance: $50,000
        - 2 Products, Active
        """)


# ───────────────────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    Built by <strong>Ashish Singh</strong> | Powered by XGBoost & Streamlit<br>
    <a href="https://github.com/yourusername/Customer-Churn-Prediction" target="_blank">
        View Source Code on GitHub
    </a>
</div>
""", unsafe_allow_html=True)
