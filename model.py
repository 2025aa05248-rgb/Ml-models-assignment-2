import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score,precision_score,recall_score,roc_auc_score,matthews_corrcoef
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

data = pd.read_csv("data.csv")   # your dataset
# print(data.head())
# print(data.isnull().sum())
# print(data.info())
# print(data.columns)
data['Grade'] = data['Grade'].replace(' anaplastic; Grade IV', 4)

alive_df = data[data['Status'] == 'Alive']
dead_df  = data[data['Status'] == 'Dead']

# Randomly sample 650 from Alive
alive_sampled = alive_df.sample(n=650, random_state=42)

# Combine back
data = pd.concat([alive_sampled, dead_df])

X = data.drop(["differentiate","Status","6th Stage","A Stage"],axis=1)
y = data['Status']

# nominal columns
# Race,Marital status,
# Ordinal columns
# T Stage, N Stage,Estrogen Status,Progesterone Status
# Numerical columns
# Age,Grade,Tumor Size,Regional Node Examined,Reginol Node Positive,Survival Months

numerical_cols = ['Age','Grade','Tumor Size','Regional Node Examined','Reginol Node Positive','Survival Months']
nominal_cols = ['Race', 'Marital Status']
ordinal_cols = [
    'T Stage',
    'N Stage',
    'Estrogen Status',
    'Progesterone Status'
]
ordinal_categories = [
    ['T1', 'T2', 'T3', 'T4'],     # T Stage
    ['N1', 'N2', 'N3'],           # N Stage
    ['Negative', 'Positive'],           # Estrogen Status
    ['Negative', 'Positive']            # Progesterone Status
]

# Encode nominal target
le = LabelEncoder()
y = le.fit_transform(data["Status"])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        # Ordinal → then scale
        ('ord', Pipeline([
            ('ord_enc', OrdinalEncoder(categories=ordinal_categories,
                                       handle_unknown='use_encoded_value',
                                       unknown_value=-1)),
            # ('scale', StandardScaler())
        ]), ordinal_cols),

        # One-hot → then scale
        ('nom', Pipeline([
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')),
            # ('scale', StandardScaler(with_mean=False))  # with_mean=False for sparse data
        ]), nominal_cols),

        # Numerical → scale
        # ('num', StandardScaler(), numerical_cols)
    ],
    remainder='passthrough'
)

# X_train_processed = preprocessor.fit_transform(X_train)
# X_test_processed = preprocessor.transform(X_test)
pipeline_logistic = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('scaler', StandardScaler()),
    ('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))
])

print("------------------------------")
print("Logistic regression Classifier")
print("---------------------------")

# pipeline_logistic = LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)
pipeline_logistic.fit(X_train, y_train)
y_pred = pipeline_logistic.predict(X_test)
y_prob = pipeline_logistic.predict_proba(X_test)[:,1]
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
aoc = roc_auc_score(y_test,y_prob)
mcc = matthews_corrcoef(y_test,y_pred)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Confusion matrix:",conf_mat)
print("AUC Score: ",aoc)
print("MCC Score :",mcc)

print("------------------------------")
print("Decision tree Classifier")
print("---------------------------")
# model2 = DecisionTreeClassifier(max_depth=4, random_state=42,min_samples_split=20,
    # min_samples_leaf=10)
pipeline_decision_tree = Pipeline(steps=[
    ('preprocessing', preprocessor),
    # ('scaler', StandardScaler()),
    ('model',DecisionTreeClassifier(max_depth=4, random_state=42,min_samples_split=20,
    min_samples_leaf=10))
])

pipeline_decision_tree.fit(X_train,y_train)
y_pred_tree = pipeline_decision_tree.predict(X_test)
y_prob_tree = pipeline_logistic.predict_proba(X_test)[:,1]
accuracy_tree = accuracy_score(y_test,y_pred_tree)
# print("Accuracy :",accuracy)
recall_tree = recall_score(y_test,y_pred_tree)
precision_tree = precision_score(y_test,y_pred_tree)
f1_tree = f1_score(y_test,y_pred_tree)
confusion_matrix_tree = confusion_matrix(y_test, y_pred_tree)
aoc_tree = roc_auc_score(y_test,y_prob_tree)
mcc_tree = matthews_corrcoef(y_test,y_pred_tree)
print("Accuracy:", accuracy_tree)
print("Recall:", recall_tree)
print("Precision:", precision_tree)
print("F1-score:", f1_tree)
print("Confusion matrix:",confusion_matrix_tree)
print("AUC Score: ",aoc_tree)
print("MCC Score :",mcc_tree)


print("------------------------------")
print("K-NN")
print("---------------------------")
# model3 = KNeighborsClassifier(n_neighbors=7,weights='distance',metric='minkowski') 
pipeline_knn = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('scaler', StandardScaler()),
    ('model',KNeighborsClassifier(n_neighbors=7,weights='distance',metric='minkowski'))
])

pipeline_knn.fit(X_train,y_train)
y_pred_nn = pipeline_knn.predict(X_test)
y_prob_nn = pipeline_knn.predict_proba(X_test)[:,1]
accuracy_knn = accuracy_score(y_test,y_pred_nn)
# print("Accuracy :",accuracy)
recall_knn = recall_score(y_test,y_pred_nn)
precision_knn = precision_score(y_test,y_pred_nn)
f1_knn = f1_score(y_test,y_pred_nn)
conf_mat_knn = confusion_matrix(y_test, y_pred_nn)
aoc_nn = roc_auc_score(y_test,y_prob_nn)
mcc_nn = matthews_corrcoef(y_test,y_pred_nn)
print("Accuracy:", accuracy_knn)
print("Recall:", recall_knn)
print("Precision:", precision_knn)
print("F1-score:", f1_knn)
print("Confusion matrix:",conf_mat_knn)
print("AUC Score: ",aoc_nn)
print("MCC Score :",mcc_nn)

print("------------------------------")
print("Navie Bayes Gaussian")
print("---------------------------")
# model4 = GaussianNB()
pipeline_bayes = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('scaler', StandardScaler()),
    ('model',GaussianNB())
])
pipeline_bayes.fit(X_train,y_train)
y_pred_nb = pipeline_bayes.predict(X_test)
y_prob_nb = pipeline_bayes.predict_proba(X_test)[:,1]
accuracy_nb = accuracy_score(y_test,y_pred_nb)
recall_nb = recall_score(y_test,y_pred_nb)
precision_nb = precision_score(y_test,y_pred_nb)
f1_nb = f1_score(y_test,y_pred_nb)
confusion_nb = confusion_matrix(y_test, y_pred_nb)
aoc_nb = roc_auc_score(y_test,y_prob_nb)
mcc_nb = matthews_corrcoef(y_test,y_pred_nb)
print("Accuracy:", accuracy_nb)
print("Recall:", recall_nb)
print("Precision:", precision_nb)
print("F1-score:", f1_nb)
print("Confusion matrix:",confusion_nb)
print("AUC Score: ",aoc_nb)
print("MCC Score :",mcc_nb) 


print("------------------------------")
print("Random Forest Classifier")
print("---------------------------")
# model5 = RandomForestClassifier(n_estimators=300,
#     max_depth=6,
#     min_samples_leaf=10,
#     min_samples_split=20,
#     random_state=42)

pipeline_random_forest = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model',RandomForestClassifier(n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=42))
])
pipeline_random_forest.fit(X_train,y_train)
y_pred_5 = pipeline_random_forest.predict(X_test)
y_prob_5 = pipeline_random_forest.predict_proba(X_test)[:,1]
accuracy_rf = accuracy_score(y_test,y_pred_5)
recall_rf = recall_score(y_test,y_pred_5)
precision_rf = precision_score(y_test,y_pred_5)
f1_rf = f1_score(y_test,y_pred_5)
confusion_rf = confusion_matrix(y_test, y_pred_5)
aoc_rf = roc_auc_score(y_test,y_prob_5)
mcc_rf = matthews_corrcoef(y_test,y_pred_5)
print("Accuracy:", accuracy_rf)
print("Recall:", recall_rf)
print("Precision:", precision_rf)
print("F1-score:", f1_rf)
print("Confusion matrix:",confusion_rf)
print("AUC Score: ",aoc_rf)
print("MCC Score :",mcc_rf) 


print("------------------------------")
print("XGBoost")
print("---------------------------")
# model6 = XGBClassifier(n_estimators=300,
#     learning_rate=0.05,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric='logloss')

pipeline_xgb = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model',XGBClassifier(n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'))
])
pipeline_xgb.fit(X_train,y_train)
y_pred_6 = pipeline_xgb.predict(X_test)
y_prob_6 = pipeline_xgb.predict_proba(X_test)[:,1]
accuracy_xgb = accuracy_score(y_test,y_pred_6)
recall_xgb = recall_score(y_test,y_pred_6)
precision_xgb = precision_score(y_test,y_pred_6)
f1_xgb = f1_score(y_test,y_pred_6)
confusion_xgb = confusion_matrix(y_test, y_pred_6)
aoc_xgb = roc_auc_score(y_test,y_prob_6)
mcc_xgb = matthews_corrcoef(y_test,y_pred_6)
print("Accuracy:", accuracy_xgb)
print("Recall:", recall_xgb)
print("Precision:", precision_xgb)
print("F1-score:", f1_xgb)
print("Confusion matrix:",confusion_xgb)
print("AUC Score: ",aoc_xgb)
print("MCC Score :",mcc_xgb) 


import pickle
pickle.dump(pipeline_logistic, open("logistic.pkl", "wb"))
pickle.dump(pipeline_decision_tree,open("decision_tree.pkl","wb"))
pickle.dump(pipeline_knn,open("knn.pkl","wb"))
pickle.dump(pipeline_bayes,open("naive_bayes.pkl","wb"))
pickle.dump(pipeline_random_forest,open("random_forest.pkl","wb"))
pickle.dump(pipeline_xgb,open("xgb.pkl",'wb'))