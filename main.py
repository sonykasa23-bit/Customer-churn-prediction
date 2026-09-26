import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data Cleaning
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Label Encoding
encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

# Display first 5 rows
print(df.head())

# Display data types
print("\nData Types After Encoding")
print(df.dtypes)

# Separate input and output
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert target into numbers
y = y.map({"Yes": 1, "No": 0})

# Convert all text columns in X into numbers
X = pd.get_dummies(X, drop_first=True)

# Make sure all values are numeric
X = X.astype(float)

print(X.head())
print(X.dtypes)

# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, "churn_model.pkl")

# Save the column names used by the model
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model saved successfully!")