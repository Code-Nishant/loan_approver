import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Loan Approval ML App", layout="wide")

st.title("🏦 Loan Approval Classification App")

st.markdown("Upload dataset → Select model → View performance")

# -------------------- LOAD MODELS --------------------
lr_model = joblib.load("model/logistic_regression.pkl")
dt_model = joblib.load("model/decision_tree.pkl")
knn_model = joblib.load("model/knn.pkl")
nb_model = joblib.load("model/naive_bayes.pkl")
rf_model = joblib.load("model/random_forest.pkl")
xgb_model = joblib.load("model/xgboost.pkl")

# -------------------- DATA UPLOAD --------------------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------- PREPROCESSING --------------------
    df = df.dropna()

    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    if "loan_status" not in df.columns:
        st.error("❌ 'loan_status' target column missing.")
    else:

        X = df.drop("loan_status", axis=1)
        y = df["loan_status"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scaling (for LR & kNN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -------------------- MODEL SELECTION --------------------
        model_name = st.selectbox(
            "🤖 Select Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            ]
        )

        # -------------------- PREDICTIONS --------------------
        if model_name == "Logistic Regression":
            model = lr_model
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        elif model_name == "Decision Tree":
            model = dt_model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        elif model_name == "KNN":
            model = knn_model
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        elif model_name == "Naive Bayes":
            model = nb_model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        elif model_name == "Random Forest":
            model = rf_model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        elif model_name == "XGBoost":
            model = xgb_model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # -------------------- METRICS --------------------
        st.subheader("📈 Evaluation Metrics")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("AUC", f"{auc:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # -------------------- CONFUSION MATRIX --------------------
        st.subheader("🔲 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # -------------------- CLASSIFICATION REPORT --------------------
        st.subheader("🧾 Classification Report")

        report = classification_report(y_test, y_pred)
        st.text(report)

else:
    st.info("👆 Upload dataset to begin")
