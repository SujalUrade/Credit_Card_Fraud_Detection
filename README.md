💳 Credit Card Fraud Detection System (Streamlit + ML)

A machine learning–based Credit Card Fraud Detection System built using Logistic Regression and deployed with an interactive Streamlit frontend.
The system handles highly imbalanced data using under-sampling to improve fraud detection performance.

🚀 Project Overview

Credit card fraud datasets are extremely imbalanced, where fraudulent transactions form a very small fraction of total data.
This project focuses on:

Handling class imbalance correctly

Building a reliable fraud detection model

Providing an interactive web interface for prediction

Making predictions on unseen transaction data

🧩 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Streamlit

Logistic Regression

📂 Project Structure
├── app.py                     # Streamlit application
├── creditcard.csv             # Dataset
├── requirements.txt
└── README.md

📊 Dataset Description

Source: Public Credit Card Fraud Dataset

Total Features: 30

Time

V1 – V28 (PCA-transformed features)

Amount

Class (Target)

0 → Legit transaction

1 → Fraud transaction

⚠️ Note: Due to confidentiality, original feature meanings are hidden using PCA.

🔄 Project Workflow
1️⃣ Data Gathering

Dataset Source:  
👉 [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset loaded from creditcard.csv

Cached using Streamlit for performance optimization

pd.read_csv("creditcard.csv")

2️⃣ Exploratory Data Analysis (EDA)

Checked dataset size and class distribution

Observed extreme imbalance:

Legit transactions ≫ Fraud transactions

Visualized class distribution using Streamlit charts

3️⃣ Data Preprocessing

Separated features (X) and target (Y)

Applied StandardScaler for feature normalization

PCA features already provided in dataset

4️⃣ Handling Imbalanced Data (Key Step)

To address class imbalance:

✅ Under-sampling was applied

All fraud transactions retained

Legit transactions randomly sampled to match fraud count

Resulted in a balanced dataset

This step significantly improves fraud detection performance.

5️⃣ Model Training

Algorithm used: Logistic Regression

Reason:

Simple

Interpretable

Effective baseline for binary classification

Stratified train-test split (80–20)

6️⃣ Model Evaluation

Evaluation metrics used:

Training Accuracy

Test Accuracy

Classification Report

Precision

Recall

F1-score

Accuracy alone was not trusted due to imbalance; emphasis was placed on fraud recall and precision.

7️⃣ Prediction System

Users can upload a CSV file (without Class column)

Model predicts:

Fraud

Legit

Results are downloadable as a CSV file

🖥️ Streamlit Frontend Features

Dataset preview

Class distribution visualization

One-click model training

CSV upload for prediction

Downloadable prediction results

Clean, responsive UI

⚠️ Challenges Faced
1️⃣ Highly Imbalanced Dataset

Fraud cases were extremely rare

Initial models showed high accuracy but poor fraud detection

Solution:
Applied under-sampling to balance the dataset before training.

2️⃣ Misleading Accuracy Metric

Accuracy was not reliable due to class imbalance

Solution:
Focused on classification report, especially recall for fraud class.

3️⃣ PCA-Transformed Features

Features were not interpretable

Limited feature-level explanations

Solution:
Focused on model performance rather than feature interpretability.

4️⃣ Deployment Constraints

Ensuring uploaded data matches training format

Preventing prediction errors in Streamlit

Solution:
Strict input validation and clear upload instructions.

📈 Results & Impact

Improved fraud detection reliability

Reduced false negatives

Robust prediction pipeline

Real-world applicable fraud detection workflow

🔮 Future Improvements

Add SMOTE for oversampling comparison

Introduce Random Forest / XGBoost

ROC-AUC and confusion matrix visualization

Threshold tuning for fraud sensitivity

Model persistence using joblib

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py

👨‍💻 Author

Sujal Urade
Machine Learning & Data Science Enthusiast
