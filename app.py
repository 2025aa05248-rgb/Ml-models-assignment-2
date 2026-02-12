import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Breast Cancer Survival Prediction")

# ==========================
# 1️⃣ Model Selection
# ==========================

model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", 
     "Decision Tree", 
     "KNN", 
     "Naive Bayes", 
     "Random Forest", 
     "XGBoost"]
)

model_files = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgb.pkl"
}

model = pickle.load(open(model_files[model_option], "rb"))

# ==========================
# 2️⃣ Upload Test Dataset
# ==========================

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file,encoding="ISO-8859-1",on_bad_lines="skip")
    test_data['Grade'] = test_data['Grade'].replace(' anaplastic; Grade IV', 4)

    st.write("Uploaded Dataset Preview:")
    st.write(test_data.head())

    if "Status" not in test_data.columns:
        st.error("CSV must contain 'Status' column for evaluation.")
    else:
        X_test = test_data.drop("Status", axis=1)
        y_test = test_data["Status"]

        # Convert labels same as training
        y_test = y_test.map({"Alive": 0, "Dead": 1})

        # ==========================
        # 3️⃣ Predictions
        # ==========================

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        # ==========================
        # 4️⃣ Evaluation Metrics
        # ==========================

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")

        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"F1 Score: {f1:.3f}")
        st.write(f"AUC Score: {auc:.3f}")
        st.write(f"MCC Score: {mcc:.3f}")

        # ==========================
        # 5️⃣ Confusion Matrix
        # ==========================

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Alive", "Dead"],
                    yticklabels=["Alive", "Dead"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)
