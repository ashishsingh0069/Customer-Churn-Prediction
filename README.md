# Bank Customer Churn Prediction 🏦

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-red?logo=streamlit)

## 🚀 Live Demo

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://customer-churn-ashish.streamlit.app/)

## 📌 Project Overview
Customer churn is one of the most critical metrics for any business, especially in the banking industry where acquiring a new customer can cost **5–7x more** than retaining an existing one.

This project predicts whether a bank customer will churn (close their account) using **Machine Learning**, and provides a **live web app** where anyone can input customer details and get a real-time prediction.

## 📊 Dataset
- **Source**: [Kaggle - Bank Customer Churn](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- **Size**: 10,000 customers from France, Germany, and Spain
- **Target**: `Exited` (1 = Churned, 0 = Stayed)

## 🚀 Key Features
- **Comprehensive EDA** with 6+ visualizations (boxplots, heatmaps, bar charts)
- **4 Models Compared**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Class Imbalance Handling** via `class_weight='balanced'` and `scale_pos_weight`
- **Hyperparameter Tuning** with `RandomizedSearchCV` (50 iterations, 5-fold CV)
- **SHAP Interpretability** to explain predictions to business stakeholders
- **Live Web App** built with Streamlit for real-time predictions

## 📈 Key Findings
| Finding | Detail |
|---------|--------|
| **Age** | Older customers (40+) are significantly more likely to churn |
| **Geography** | Germany has ~2x the churn rate of France or Spain |
| **Activity** | Inactive members churn at a much higher rate |
| **Products** | Customers with 3+ products almost always churn |
| **Gender** | Female customers churn more than males |

## 🛠️ Installation & Usage

### Run the Notebook
```bash
git clone https://github.com/ashishsingh0069/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
jupyter notebook CUSTOMER_CHURN__PREDICTION.ipynb
```

### Run the Web App Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository Structure
```
Customer-Churn-Prediction/
├── app.py                                # Streamlit web app (live demo)
├── CUSTOMER_CHURN__PREDICTION.ipynb      # Analysis & modeling notebook
├── Churn_Modelling.csv                   # Dataset
├── requirements.txt                      # Dependencies
├── runtime.txt                           # Python version for Streamlit Cloud
├── README.md                             # This file
└── models/
    ├── best_churn_model.joblib           # Trained XGBoost model
    └── scaler.joblib                     # Feature scaler
```

## 🧠 Tech Stack
`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `XGBoost` · `SHAP` · `Streamlit`

## 👨‍💻 Author
**Ashish Singh** — [GitHub](https://github.com/ashishsingh0069)



