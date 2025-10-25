import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

TRAIN = "data/processed/train.csv"
TEST = "data/processed/test.csv"

# Load processed data
train_df = pd.read_csv(TRAIN)
test_df  = pd.read_csv(TEST)

target = "Survived"
X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_test  = test_df.drop(columns=[target])
y_test  = test_df[target]

# Train
model = LogisticRegression(max_iter=200, solver="liblinear", random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Save metrics
with open("metrics.json", "w") as f:
    json.dump({"accuracy": float(acc)}, f)

# Confusion matrix
cm = confusion_matrix(y_test, preds, labels=model.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print(f"✅ Accuracy: {acc:.3f}")
