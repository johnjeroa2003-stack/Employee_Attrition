import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/cleaned_employee_data.csv")

# Features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Logistic Regression
# -----------------------------

print("Training Logistic Regression...")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))


# -----------------------------
# Decision Tree
# -----------------------------

print("\nTraining Decision Tree...")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall:", recall_score(y_test, dt_pred))


# -----------------------------
# Random Forest
# -----------------------------

print("\nTraining Random Forest...")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))


# -----------------------------
# Save Model
# -----------------------------

print("\nSaving model...")

# Save trained model
with open("model/attrition_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Save feature columns (VERY IMPORTANT for Streamlit)
model_columns = X_train.columns.tolist()

with open("model/model_columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)

print("✅ Model and columns saved successfully!")

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

print("F1 Score:", f1_score(y_test, rf_pred))
print("ROC AUC:", roc_auc_score(y_test, rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


