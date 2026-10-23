# ============================================================
# BANK CUSTOMER CHURN PREDICTION AND RISK SCORING
# ============================================================

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction", page_icon="🏦", layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction")
st.subheader("Predictive Modeling and Risk Scoring for Bank Customer Churn")

st.write(
    """
    This application predicts whether a bank customer is likely to leave
    the bank and calculates the customer's churn risk probability.
    """
)

# ============================================================
# LOAD DATASET
# ============================================================


@st.cache_data
def load_data():
    file_name = "Churn_Modelling.csv"
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv"

    # Download file if missing or corrupted (0 bytes)
    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
        df_download = pd.read_csv(url)
        df_download.to_csv(file_name, index=False)
        return df_download

    return pd.read_csv(file_name)


df = load_data()
st.success("Dataset loaded successfully!")

# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric("Churned Customers", int(df["Exited"].sum()))

st.write("### First 10 Records")
st.dataframe(df.head(10))

# ============================================================
# DATA CLEANING
# ============================================================

st.header("2. Data Cleaning")

columns_to_remove = []
if "RowNumber" in df.columns:
    columns_to_remove.append("RowNumber")
if "CustomerId" in df.columns:
    columns_to_remove.append("CustomerId")
if "Surname" in df.columns:
    columns_to_remove.append("Surname")

df_model = df.drop(columns=columns_to_remove, errors="ignore")

missing_values = df_model.isnull().sum()
st.write("### Missing Values")
st.dataframe(missing_values[missing_values > 0])

df_model = df_model.dropna()
st.write("After cleaning, dataset contains", len(df_model), "records.")

# ============================================================
# TARGET AND FEATURES
# ============================================================

X = df_model.drop("Exited", axis=1)
y = df_model["Exited"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

st.write("### Numerical Features")
st.write(numerical_features)

st.write("### Categorical Features")
st.write(categorical_features)

# ============================================================
# PREPROCESSING & PIPELINE SETUP
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        )
    ],
    remainder="passthrough",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
)

pipeline = Pipeline(
    steps=[("preprocessing", preprocessor), ("model", model)]
)

with st.spinner("Training machine learning model..."):
    pipeline.fit(X_train, y_train)

st.success("Machine learning model trained successfully!")

# ============================================================
# MODEL EVALUATION
# ============================================================

st.header("3. Model Evaluation")

y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
with col2:
    st.metric("Precision", f"{precision * 100:.2f}%")
with col3:
    st.metric("Recall", f"{recall * 100:.2f}%")
with col4:
    st.metric("F1 Score", f"{f1 * 100:.2f}%")
with col5:
    st.metric("ROC-AUC", f"{roc_auc:.3f}")

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.write("### Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# ============================================================
# CHURN DISTRIBUTION
# ============================================================

st.header("4. Customer Churn Analysis")
churn_count = df["Exited"].value_counts()
fig, ax = plt.subplots()
ax.bar(
    ["Stayed", "Churned"], [churn_count.get(0, 0), churn_count.get(1, 0)]
)
ax.set_ylabel("Number of Customers")
ax.set_title("Customer Churn Distribution")
st.pyplot(fig)

# ============================================================
# CUSTOMER PREDICTION
# ============================================================

st.header("5. Customer Churn Risk Prediction")
st.write(
    """
    Enter customer information below to calculate the probability
    that the customer will leave the bank.
    """
)

input_data = {}

if "CreditScore" in X.columns:
    input_data["CreditScore"] = st.number_input(
        "Credit Score", min_value=300, max_value=850, value=650
    )

if "Geography" in X.columns:
    geography_values = df["Geography"].dropna().unique()
    input_data["Geography"] = st.selectbox("Geography", geography_values)

if "Gender" in X.columns:
    gender_values = df["Gender"].dropna().unique()
    input_data["Gender"] = st.selectbox("Gender", gender_values)

if "Age" in X.columns:
    input_data["Age"] = st.number_input(
        "Age", min_value=18, max_value=100, value=35
    )

if "Tenure" in X.columns:
    input_data["Tenure"] = st.number_input(
        "Tenure", min_value=0, max_value=20, value=5
    )

if "Balance" in X.columns:
    input_data["Balance"] = st.number_input(
        "Balance", min_value=0.0, value=50000.0
    )

if "NumOfProducts" in X.columns:
    input_data["NumOfProducts"] = st.number_input(
        "Number of Products", min_value=1, max_value=10, value=2
    )

if "HasCrCard" in X.columns:
    input_data["HasCrCard"] = st.selectbox(
        "Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
    )

if "IsActiveMember" in X.columns:
    input_data["IsActiveMember"] = st.selectbox(
        "Is Active Member?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

if "EstimatedSalary" in X.columns:
    input_data["EstimatedSalary"] = st.number_input(
        "Estimated Salary", min_value=0.0, value=50000.0
    )

if st.button("🔮 Predict Churn Risk"):
    customer_df = pd.DataFrame([input_data])
    customer_df = customer_df.reindex(columns=X.columns)

    prediction = pipeline.predict(customer_df)[0]
    probability = pipeline.predict_proba(customer_df)[0][1]
    risk_percentage = probability * 100

    if risk_percentage < 30:
        risk_level = "LOW RISK"
    elif risk_percentage < 60:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "HIGH RISK"

    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("Customer is likely to CHURN")
        else:
            st.success("Customer is likely to STAY")

    with col2:
        st.metric("Churn Probability", f"{risk_percentage:.2f}%")

    with col3:
        st.metric("Risk Level", risk_level)

    st.write("### Risk Score")
    st.progress(float(probability))

    st.write("### Recommended Action")
    if risk_level == "HIGH RISK":
        st.warning(
            """
            Recommended actions:
            • Contact the customer proactively
            • Provide personalized offers
            • Check customer satisfaction
            • Offer suitable banking products
            • Provide retention incentives
            """
        )
    elif risk_level == "MEDIUM RISK":
        st.info(
            """
            Recommended actions:
            • Monitor customer activity
            • Send personalized communication
            • Offer relevant products
            • Improve customer engagement
            """
        )
    else:
        st.success(
            """
            Customer currently has low churn risk.
            Continue normal customer engagement and relationship management.
            """
        )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("6. Churn Driver Analysis")
st.write(
    """
    These features show which customer characteristics are
    important for the Random Forest model.
    """
)

trained_model = pipeline.named_steps["model"]
trained_preprocessor = pipeline.named_steps["preprocessing"]

feature_names = trained_preprocessor.get_feature_names_out()
importance = trained_model.feature_importances_

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importance}
)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

st.dataframe(importance_df.head(15))

top_features = importance_df.head(10)
fig, ax = plt.subplots()
ax.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
ax.set_xlabel("Importance")
ax.set_title("Top 10 Churn Drivers")
st.pyplot(fig)

# ============================================================
# SCENARIO ANALYSIS
# ============================================================

st.header("7. Scenario-Based Risk Analysis")
st.write(
    """
    Change the customer's information and observe how the
    predicted churn probability changes.
    """
)

scenario_age = st.slider("Scenario Age", 18, 80, 40)
scenario_active = st.selectbox(
    "Scenario Active Member",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
scenario_products = st.slider("Scenario Number of Products", 1, 4, 2)

if st.button("Run Scenario Analysis"):
    scenario = input_data.copy()

    if "Age" in X.columns:
        scenario["Age"] = scenario_age
    if "IsActiveMember" in X.columns:
        scenario["IsActiveMember"] = scenario_active
    if "NumOfProducts" in X.columns:
        scenario["NumOfProducts"] = scenario_products

    scenario_df = pd.DataFrame([scenario])
    scenario_df = scenario_df.reindex(columns=X.columns)

    scenario_probability = pipeline.predict_proba(scenario_df)[0][1]

    st.metric(
        "Scenario Churn Probability", f"{scenario_probability * 100:.2f}%"
    )

# ============================================================
# PROJECT CONCLUSION
# ============================================================

st.header("8. Conclusion")
st.write(
    """
    The developed machine learning system predicts the likelihood
    of customer churn and assigns a risk score to each customer.

    The system can help banks identify high-risk customers before
    they leave and take proactive retention actions such as
    personalized offers, targeted communication and improved
    customer engagement.
    """
)