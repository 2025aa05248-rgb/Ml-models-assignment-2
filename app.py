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
race = st.selectbox("Race", ["White","Black","Other"])
marital_status = st.selectbox("Marital Status", ["Married","Single","Divorced","Widowed","Separated"])
t_stage = st.selectbox("T Stage", ["T1","T2","T3","T4"])
n_stage = st.selectbox("N Stage", ["N1","N2","N3"])
grade = st.number_input("Grade",min_value = 0)
tumor_size = st.number_input("Tumor Size", min_value=0)
survival_months = st.number_input("Survival Months", min_value=0)
estrogen = st.selectbox("Estrogen Status", ["Negative","Positive"])
progesterone = st.selectbox("Progesterone Status", ["Negative","Positive"])
regional_node_examined = st.number_input("Regional Node Examined", min_value=0)
regional_node_positive = st.number_input("Reginol Node Positive", min_value=0)

if st.button("Predict"):
    input_df = pd.DataFrame([[age,race, tumor_size,grade,marital_status,t_stage,
                              n_stage, survival_months,estrogen,progesterone,
                              regional_node_examined,regional_node_positive]],
                            columns=["Age","Race","Tumor Size","Grade","Marital Status",
                                     'T Stage','N Stage',"Survival Months",'Estrogen Status',
                                     'Progesterone Status','Regional Node Examined','Reginol Node Positive'])
    
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("Prediction: Alive")
    else:
        st.error("Prediction: Dead")
