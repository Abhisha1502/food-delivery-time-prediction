from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


def train_and_evaluate_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[LogisticRegression, Dict[str, float], pd.DataFrame]:
    """
    Train a Logistic Regression model and compute evaluation metrics.
    Returns model, metric dict, and confusion matrix as DataFrame.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Fast(0)", "Actual_Delayed(1)"],
        columns=["Predicted_Fast(0)", "Predicted_Delayed(1)"],
    )

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }

    return model, metrics, cm_df
