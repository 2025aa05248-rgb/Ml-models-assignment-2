import streamlit as st
import pickle
import pandas as pd

st.title("Breast Cancer Survival Prediction")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", 
     "Decision Tree", 
     "KNN", 
     "Naive Bayes", 
     "Random Forest", 
     "XGBoost"]
)

# Load selected model
model_files = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgb.pkl"
}

model = pickle.load(open(model_files[model_option], "rb"))

# Example inputs (you must match your dataset columns)
age = st.number_input("Age", min_value=0)
tumor_size = st.number_input("Tumor Size", min_value=0)
survival_months = st.number_input("Survival Months", min_value=0)
grade = st.number_input("Grade",min_value = 0)

if st.button("Predict"):
    input_df = pd.DataFrame([[age, tumor_size, survival_months]],
                            columns=["Age", "Tumor Size","Grade", "Survival Months"])
    
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("Prediction: Alive")
    else:
        st.error("Prediction: Dead")
