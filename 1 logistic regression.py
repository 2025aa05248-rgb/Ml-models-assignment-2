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
            ('scale', StandardScaler())
        ]), ordinal_cols),

        # One-hot → then scale
        ('nom', Pipeline([
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')),
            ('scale', StandardScaler(with_mean=False))  # with_mean=False for sparse data
        ]), nominal_cols),

        # Numerical → scale
        ('num', StandardScaler(), numerical_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
# pipeline = Pipeline(steps=[
#     ('preprocess', preprocessor),
#     ('scaler', StandardScaler())
# ])

print("------------------------------")
print("Logistic regression Classifier")
print("---------------------------")

model = LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)
model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Confusion matrix:",confusion_matrix(y_test, y_pred))
print("AUC Score: ",roc_auc_score(y_test,y_pred))
print("MCC Score :",matthews_corrcoef(y_test,y_pred))

print("------------------------------")
print("Decision tree Classifier")
print("---------------------------")
model2 = DecisionTreeClassifier(max_depth=4, random_state=42,min_samples_split=20,
    min_samples_leaf=10)
model2.fit(X_train_processed,y_train)
y_pred_tree = model2.predict(X_test_processed)
accuracy = accuracy_score(y_test,y_pred_tree)
# print("Accuracy :",accuracy)
recall = recall_score(y_test,y_pred_tree)
precision = precision_score(y_test,y_pred_tree)
f1 = f1_score(y_test,y_pred_tree)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Confusion matrix:",confusion_matrix(y_test, y_pred_tree))
print("AUC Score: ",roc_auc_score(y_test,y_pred_tree))
print("MCC Score :",matthews_corrcoef(y_test,y_pred_tree))


print("------------------------------")
print("K-NN")
print("---------------------------")
model3 = KNeighborsClassifier(n_neighbors=7,weights='distance',metric='minkowski') 
model3.fit(X_train_processed,y_train)
y_pred_nn = model3.predict(X_test_processed)
y_prob_nn = model3.predict_proba(X_test_processed)[:,1]
accuracy = accuracy_score(y_test,y_pred_nn)
print("Accuracy :",accuracy)
recall = recall_score(y_test,y_pred_nn)
precision = precision_score(y_test,y_pred_nn)
f1 = f1_score(y_test,y_pred_nn)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Confusion matrix:",confusion_matrix(y_test, y_pred_nn))
print("AUC Score: ",roc_auc_score(y_test,y_prob_nn))
print("MCC Score :",matthews_corrcoef(y_test,y_pred_nn))

print("------------------------------")
print("Navie Bayes Gaussian")
print("---------------------------")
model4 = GaussianNB()
model4.fit(X_train_processed,y_train)
y_pred_nb = model4.predict(X_test_processed)
y_prob_nb = model4.predict_proba(X_test_processed)[:,1]
accuracy = accuracy_score(y_test,y_pred_nb)
recall = recall_score(y_test,y_pred_nb)
precision = precision_score(y_test,y_pred_nb)
f1 = f1_score(y_test,y_pred_nb)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Confusion matrix:",confusion_matrix(y_test, y_pred_nb))
print("AUC Score: ",roc_auc_score(y_test,y_prob_nb))
print("MCC Score :",matthews_corrcoef(y_test,y_pred_nb)) 


print("------------------------------")
print("Random Forest Classifier")
print("---------------------------")
model5 = RandomForestClassifier(n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=42)
model5.fit(X_train_processed,y_train)
y_pred_5 = model5.predict(X_test_processed)
y_prob_5 = model5.predict_proba(X_test_processed)[:,1]
accuracy = accuracy_score(y_test,y_pred_5)
recall = recall_score(y_test,y_pred_5)
precision = precision_score(y_test,y_pred_5)
f1 = f1_score(y_test,y_pred_5)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Confusion matrix:",confusion_matrix(y_test, y_pred_5))
print("AUC Score: ",roc_auc_score(y_test,y_prob_5))
print("MCC Score :",matthews_corrcoef(y_test,y_pred_5)) 


print("------------------------------")
print("XGBoost")
print("---------------------------")
model6 = XGBClassifier(n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss')
model6.fit(X_train_processed,y_train)
y_pred_6 = model6.predict(X_test_processed)
y_prob_6 = model6.predict_proba(X_test_processed)[:,1]
accuracy = accuracy_score(y_test,y_pred_6)
recall = recall_score(y_test,y_pred_6)
precision = precision_score(y_test,y_pred_6)
f1 = f1_score(y_test,y_pred_6)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Confusion matrix:",confusion_matrix(y_test, y_pred_6))
print("AUC Score: ",roc_auc_score(y_test,y_prob_6))
print("MCC Score :",matthews_corrcoef(y_test,y_pred_6)) 
