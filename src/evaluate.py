"""Metrics with a fixed ten-class order; explicit population of each summary."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from .dataset import CATEGORIES


def metrics(y_true, y_pred):
    if not len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Nonempty equally sized labels required")
    if not set(y_true).issubset(CATEGORIES) or not set(y_pred).issubset(CATEGORIES):
        raise ValueError("Unknown class label")
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0)),
            "recall": dict(zip(CATEGORIES, recall_score(y_true, y_pred, labels=CATEGORIES, average=None, zero_division=0).tolist())),
            "confusion_matrix": cm.tolist(),
            "confusion_normalized": (cm / np.maximum(cm.sum(1, keepdims=True), 1)).tolist()}
