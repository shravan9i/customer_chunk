import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train_df = pd.read_csv('../dataset/customer_churn_dataset-training-master.csv')
test_df = pd.read_csv('../dataset/customer_churn_dataset-testing-master.csv')

train_df['Churn'] = pd.to_numeric(train_df['Churn'], errors='coerce')
test_df['Churn'] = pd.to_numeric(test_df['Churn'], errors='coerce')


train_df = train_df.dropna(subset=['Churn'])
test_df = test_df.dropna(subset=['Churn'])


train_df['Churn'] = train_df['Churn'].astype(int)
test_df['Churn'] = test_df['Churn'].astype(int)

if 'CustomerID' in train_df.columns:
    train_df.drop('CustomerID', axis=1, inplace=True)
    test_df.drop('CustomerID', axis=1, inplace=True)

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)


cat_cols = ['Gender', 'Subscription Type', 'Contract Length']

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    encoders[col] = le


X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']

X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(" Model Accuracy:", accuracy)


import os

os.makedirs('../model', exist_ok=True)

pickle.dump(model, open('../model/model.pkl', 'wb'))

print(" Model saved successfully!")