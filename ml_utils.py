"""
ml_utils.py
-----------
Shared helpers used by BOTH the training script (train_models.py) and the
Streamlit app (app.py). Keeping the metric logic in one place guarantees that
the numbers reported in the README are computed exactly the same way as the
numbers shown in the web app.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Absolute paths so the code works no matter which folder it is launched from
# (Streamlit Community Cloud runs the app from the repository root).
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
DATA_DIR = PROJECT_DIR / "data"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Name of the label column in the Dry Bean dataset.
TARGET_COL = "Class"

# Display name -> file name of the saved pipeline inside model/.
# The order here is the order used in every table and dropdown.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# Column order of the comparison table required by the assignment.
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def _multiclass_auc(y_true, y_proba, classes):
    """One-vs-Rest ROC-AUC, weighted by class support.

    When every class is present in y_true this is identical to
    roc_auc_score(..., multi_class="ovr", average="weighted").
    If a user uploads a subset of the test data in which some classes are
    missing, sklearn raises an error, so we instead average only over the
    classes that are actually present (a class with zero support would get a
    weight of zero anyway in the "weighted" average).
    """
    classes = list(classes)
    present = [c for c in classes if np.any(y_true == c)]

    if len(present) == len(classes):
        return roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted", labels=classes
        )
    if len(present) < 2:
        # AUC is undefined when only one class exists in the ground truth.
        return float("nan")

    scores, weights = [], []
    for cls in present:
        y_bin = (y_true == cls).astype(int)
        scores.append(roc_auc_score(y_bin, y_proba[:, classes.index(cls)]))
        weights.append(y_bin.sum())
    return float(np.average(scores, weights=weights))


def compute_metrics(y_true, y_pred, y_proba, classes):
    """Return the six evaluation metrics required by the assignment.

    Parameters
    ----------
    y_true  : true labels
    y_pred  : predicted labels (model.predict)
    y_proba : class probabilities (model.predict_proba), columns ordered as
              `classes`
    classes : model.classes_

    Averaging strategy
    ------------------
    * Binary problem      -> precision/recall/F1 for the positive class
                             (classes[1]); AUC from P(positive class).
    * Multi-class problem -> "weighted" average, i.e. each class's score is
                             weighted by its number of true samples. This is
                             the right choice for Dry Bean because the classes
                             are imbalanced (BOMBAY ~4% vs DERMASON ~26%).
    MCC is computed natively for multi-class by sklearn (no averaging needed).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = list(classes)

    if len(classes) == 2:
        positive = classes[1]
        avg_kwargs = {"average": "binary", "pos_label": positive}
        if len(np.unique(y_true)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score((y_true == positive).astype(int), y_proba[:, 1])
    else:
        avg_kwargs = {"average": "weighted", "labels": classes}
        auc = _multiclass_auc(y_true, y_proba, classes)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        # zero_division=0 avoids a crash/warning if a class is never predicted.
        "Precision": precision_score(y_true, y_pred, zero_division=0, **avg_kwargs),
        "Recall": recall_score(y_true, y_pred, zero_division=0, **avg_kwargs),
        "F1": f1_score(y_true, y_pred, zero_division=0, **avg_kwargs),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
