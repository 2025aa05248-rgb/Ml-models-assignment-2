import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

# -----------------------
# Target Encoding
# -----------------------
data = pd.read_csv("data.csv") 
data['Grade'] = data['Grade'].replace(' anaplastic; Grade IV', 4)

alive_df = data[data['Status'] == 'Alive']
dead_df  = data[data['Status'] == 'Dead']

# Randomly sample 650 from Alive
alive_sampled = alive_df.sample(n=650, random_state=42)

# Combine back
data = pd.concat([alive_sampled, dead_df])
le = LabelEncoder()
y = le.fit_transform(data["Status"])

X = data.drop(["differentiate","Status","6th Stage","A Stage"], axis=1)

# -----------------------
# Column Definitions
# -----------------------
nominal_cols = ['Race', 'Marital Status']

ordinal_cols = [
    'T Stage',
    'N Stage',
    'Estrogen Status',
    'Progesterone Status'
]

ordinal_categories = [
    ['T1', 'T2', 'T3', 'T4'],
    ['N1', 'N2', 'N3'],
    ['Negative', 'Positive'],
    ['Negative', 'Positive']
]

num_cols = [
    'Age',
    'Grade',
    'Tumor Size',
    'Regional Node Examined',
    'Reginol Node Positive',
    'Survival Months'
]

# -----------------------
# Column Transformer
# -----------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('ord',
         OrdinalEncoder(
             categories=ordinal_categories,
             handle_unknown='use_encoded_value',
             unknown_value=-1
         ),
         ordinal_cols),

        ('nom',
         OneHotEncoder(
             drop='first',
             handle_unknown='ignore'
         ),
         nominal_cols),

        ('num',
         'passthrough',
         num_cols)
    ]
)

# -----------------------
# Full Pipeline
# -----------------------
model = Pipeline([
    ('preprocessing', preprocessor),
    ('scaler', StandardScaler()),   # 🔥 Scales ALL columns
    ('knn', KNeighborsClassifier(n_neighbors=7))
])

# -----------------------
# Train Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Train
# -----------------------
model.fit(X_train, y_train)

# -----------------------
# Predict
# -----------------------
y_pred = model.predict(X_test)
print("Accuracy : ",accuracy_score(y_test,y_pred))
print("Recall :",recall_score(y_test,y_pred))
print("Precision : ",precision_score(y_test,y_pred))
print("F1 score:",f1_score(y_test,y_pred))