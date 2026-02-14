# Ml-models-assignment-2

4.a.Problem Statement:
Breast cancer survival prediction plays an important role in improving treatment planning and patient care. The goal of this project is to build machine learning models that predict whether a patient will survive or not based on clinical and demographic features such as tumor stage, grade, hormone receptor status, and lymph node involvement.

This task is formulated as a binary classification problem using patient medical data. Multiple models including Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, Random Forest, and XGBoost are trained and evaluated to identify the most accurate and reliable model for survival prediction.

Model performance is assessed using metrics such as Accuracy, Precision, Recall, F1-score, AUC, and MCC to ensure dependable predictions.

b.Dataset Description:
The dataset used in this project contains clinical and demographic information of breast cancer patients and is designed to support survival prediction. Each record represents an individual patient along with medical attributes that influence prognosis.

Target Variable
--------------------------
Status → Patient outcome
Alive/Dead
This is the binary class label the models aim to predict.
Feature Categories

1.Demographic Features
-------------------------------------
Age – Age of the patient
Race – Patient’s racial background
Marital Status – Marital status of the patient

2.Tumor Characteristics
-------------------------------------
Grade – Severity/aggressiveness of the tumor
Tumor Size – Size of the tumor
T Stage – Size and extent of the primary tumor
N Stage – Lymph node involvement stage

3.Hormone Receptor Status
-------------------------------------------
Estrogen Status – Positive or Negative
Progesterone Status – Positive or Negative

4.Lymph Node Information
----------------------------------------
Regional Node Examined – Number of lymph nodes examined
Regional Node Positive – Number of lymph nodes affected

5.Survival Information
-------------------------
Survival Months – Duration of survival after diagnosis

c.Evaluation Metrics

| Model               | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.8189   | 0.8750 | 0.8302    | 0.7586 | 0.7928   | 0.6346 |
| Decision Tree       | 0.7638   | 0.8751 | 0.8111    | 0.6293 | 0.7087   | 0.5270 |
| KNN                 | 0.7323   | 0.7996 | 0.7264    | 0.6638 | 0.6937   | 0.4583 |
| Naive Bayes         | 0.7323   | 0.8166 | 0.7500    | 0.6207 | 0.6792   | 0.4590 |
| Random Forest       | 0.7992   | 0.8657 | 0.8494    | 0.6810 | 0.7560   | 0.5992 |
| XGBoost             | 0.7953   | 0.8659 | 0.8077    | 0.7241 | 0.7636   | 0.5867 |


## Model Performance Observations

| ML Model                      | Observation About Model Performance                                                                                                                                                                                                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**       | 1. Achieved the highest accuracy (0.8189) and best MCC (0.6346). <br> 2. Highest F1-score (0.7928) indicates strong balance between precision and recall. <br> 3. AUC = 0.875 shows excellent class separation. <br> 4. Performs well because the dataset likely has linearly separable patterns. <br> 5. **Best overall performer.** |
| **Decision Tree**             | 1. Moderate accuracy (0.7638) but low recall (0.6293). <br> 2. High precision indicates careful positive predictions but misses many true positives. <br> 3. Trees can overfit and may not generalize well. <br> 4. Good interpretability but weaker generalization.                                                                  |
| **K-Nearest Neighbors (KNN)** | 1. Lower accuracy (0.7323) and MCC (0.4583). <br> 2. Performance affected by feature scaling sensitivity and high dimensionality. <br> 3. Works better with simpler feature spaces. <br> 4. Struggles with complex or high-dimension medical data.                                                                                    |
| **Naive Bayes**               | 1. Similar performance to KNN. <br> 2. Assumes feature independence, which is rare in medical datasets. <br> 3. Lower recall indicates missed positive cases. <br> 4. Useful baseline model but oversimplifies relationships.                                                                                                         |
| **Random Forest (Ensemble)**  | 1. Strong performance (Accuracy 0.799, MCC 0.599). <br> 2. High precision (0.849) → fewer false positives. <br> 3. Handles nonlinear relationships and feature interactions well. <br> 4. Reliable and robust model.                                                                                                                  |
| **XGBoost (Ensemble)**        | 1. Strong balanced performance. <br> 2. Best recall among ensemble models (0.724) → detects more true positives. <br> 3. Boosting improves difficult cases by focusing on misclassified samples. <br> 4. Excellent trade-off between recall and precision.                                                                            |


Recommended Model

Primary choice: Logistic Regression<br>
Best alternative: XGBoost<br>
Most robust: Random Forest<br>

