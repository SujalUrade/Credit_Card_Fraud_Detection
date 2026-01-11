import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Machine Learning based Fraud Detection using Logistic Regression")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

data = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Options")
show_raw = st.sidebar.checkbox("Show Raw Dataset")
train_model = st.sidebar.button("Train Model")

# ---------------- DATA PREVIEW ----------------
if show_raw:
    st.subheader("📊 Raw Dataset")
    st.dataframe(data.head(100))

# ---------------- DATA INFO ----------------
st.subheader("📈 Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", data.shape[0])

with col2:
    st.metric("Fraud Transactions", data["Class"].value_counts()[1])

with col3:
    st.metric("Legit Transactions", data["Class"].value_counts()[0])

# ---------------- CLASS DISTRIBUTION ----------------
st.subheader("⚖️ Class Distribution")
st.bar_chart(data["Class"].value_counts())

# ---------------- MODEL TRAINING ----------------
if train_model:
    st.subheader("🧠 Model Training & Evaluation")

    # ---------------- UNDER SAMPLING ----------------
    fraud = data[data["Class"] == 1]
    legit = data[data["Class"] == 0]

    legit_sampled = legit.sample(n=len(fraud), random_state=2)

    balanced_data = pd.concat([fraud, legit_sampled]) \
                        .sample(frac=1, random_state=2)

    X = balanced_data.drop(columns="Class", axis=1)
    Y = balanced_data["Class"]

    # ---------------- FEATURE SCALING ----------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled,
        Y,
        test_size=0.2,
        stratify=Y,
        random_state=2
    )

    # ---------------- MODEL ----------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # ---------------- EVALUATION ----------------
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)

    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{train_acc:.4f}")
    col2.metric("Test Accuracy", f"{test_acc:.4f}")

    st.subheader("📄 Classification Report")
    st.text(classification_report(Y_test, test_pred))

    st.success("✅ Model trained successfully on balanced data!")

    # ---------------- SAVE MODEL ----------------
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler

# ---------------- PREDICTION SECTION ----------------
st.subheader("🔍 Predict New Transaction")

uploaded_file = st.file_uploader(
    "Upload a CSV file (same format as dataset, without Class column)",
    type=["csv"]
)

if uploaded_file and "model" in st.session_state:
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview")
    st.dataframe(new_data.head())

    scaled_data = st.session_state["scaler"].transform(new_data)
    predictions = st.session_state["model"].predict(scaled_data)

    new_data["Prediction"] = np.where(predictions == 1, "Fraud", "Legit")

    st.subheader("🧾 Prediction Results")
    st.dataframe(new_data)

    st.download_button(
        "⬇️ Download Results",
        data=new_data.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

elif uploaded_file:
    st.warning("⚠️ Please train the model first.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit | Logistic Regression Fraud Detection")
