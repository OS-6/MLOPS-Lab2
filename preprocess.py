import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = "data/titanic_raw.csv"
PROC_DIR = "data/processed"

def preprocess():
    os.makedirs(PROC_DIR, exist_ok=True)

    df = pd.read_csv(RAW)

    # Drop leakage/IDs/text columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Fill missing
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Survived"])

    train_df.to_csv(f"{PROC_DIR}/train.csv", index=False)
    test_df.to_csv(f"{PROC_DIR}/test.csv", index=False)
    print("✅ Preprocessing complete. Files saved to data/processed/")

if __name__ == "__main__":
    preprocess()
