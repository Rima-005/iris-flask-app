"""
train_model.py
Author: Rima

This script trains a Logistic Regression model on the Iris dataset
and saves the trained model for deployment using Flask.
"""

import os
import joblib

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 40)
    print("Iris Model Training Completed")
    print(f"Model Accuracy : {accuracy:.2%}")
    print("=" * 40)

    # Create model directory
    os.makedirs("model", exist_ok=True)

    # Save trained model
    joblib.dump(model, "model/model.pkl")

    # Save class labels
    joblib.dump(iris.target_names, "model/class_names.pkl")

    print("Model saved successfully!")


if __name__ == "__main__":
    train()