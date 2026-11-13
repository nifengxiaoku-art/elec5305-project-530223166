"""Validation-selected Logistic Regression; test predictions made only after selection."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from .evaluate import metrics


def fit_classifier(x, y, x_val, y_val, c_grid, sample_weight=None):
    best_model, best_f1, best_c = None, -1, None
    validation = []
    for c in sorted(c_grid):
        # penalty defaults to L2 in the installed sklearn version; avoid the
        # deprecated penalty argument while retaining the documented objective.
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, tol=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x, y, sample_weight=sample_weight)
        score = metrics(y_val, model.predict(x_val))["macro_f1"]
        validation.append({"C": c, "validation_macro_f1": score, "iterations": int(np.max(model.n_iter_))})
        if score > best_f1 + 1e-12:
            best_model, best_f1, best_c = model, score, c
    return best_model, best_c, validation
