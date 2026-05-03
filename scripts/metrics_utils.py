from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> Dict:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }

    # Handle degenerate cases when only one class exists in y_true
    if len(np.unique(y_true_arr)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        metrics["auprc"] = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    # Confusion matrix: [[tn, fp], [fn, tp]]
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    metrics["confusion_matrix"] = cm
    metrics["recall_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    precision, recall, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    metrics["precision_curve"] = precision.tolist()
    metrics["recall_curve"] = recall.tolist()

    return metrics
