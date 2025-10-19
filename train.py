# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json

df = pd.read_csv('data/iris.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

with open('metrics.json', 'w') as f:
    json.dump({"accuracy": accuracy}, f)

joblib.dump(model, 'model.pkl')
print(f"Model trained with accuracy: {accuracy:.4f}")
