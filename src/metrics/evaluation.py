"""Evaluation metrics shared by the experiment runners."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    probabilities: ArrayLike,
    classes: ArrayLike,
) -> dict[str, float]:
    """Return accuracy, macro F1, and binary/macro-OVR AUC."""

    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    probability_array = np.asarray(probabilities, dtype=float)
    class_array = np.asarray(classes)
    if probability_array.shape != (truth.size, class_array.size):
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1_macro": float(f1_score(truth, predicted, average="macro")),
    }
    try:
        if class_array.size == 2:
            encoded = (truth == class_array[1]).astype(int)
            auc = roc_auc_score(encoded, probability_array[:, 1])
        else:
            auc = roc_auc_score(
                truth,
                probability_array,
                labels=class_array,
                multi_class="ovr",
                average="macro",
            )
        metrics["auc_roc"] = float(auc)
    except ValueError:
        metrics["auc_roc"] = float("nan")
    return metrics


def class_distribution(y: ArrayLike) -> dict[str, Any]:
    values = np.asarray(y)
    classes, counts = np.unique(values, return_counts=True)
    total = counts.sum()
    return {
        "total": int(total),
        "classes": {
            str(class_label): {
                "count": int(count),
                "fraction": float(count / total),
            }
            for class_label, count in zip(classes, counts, strict=True)
        },
        "minority_fraction": float(counts.min() / total),
    }


def brier_bias_variance(
    probability_predictions: ArrayLike,
    y_true: ArrayLike,
    classes: ArrayLike,
) -> dict[str, float]:
    """Brier bias-variance terms across bootstrap-fitted classifiers."""

    predictions = np.asarray(probability_predictions, dtype=float)
    truth = np.asarray(y_true)
    class_array = np.asarray(classes)
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape (B, n_samples, n_classes)")
    if predictions.shape[1:] != (truth.size, class_array.size):
        raise ValueError("prediction shape does not match targets/classes")
    one_hot = np.zeros((truth.size, class_array.size), dtype=float)
    encoded = np.searchsorted(class_array, truth)
    one_hot[np.arange(truth.size), encoded] = 1.0
    mean_prediction = predictions.mean(axis=0)
    bias_squared = float(np.mean(np.sum((mean_prediction - one_hot) ** 2, axis=1)))
    variance = float(
        np.mean(np.sum((predictions - mean_prediction[None, :, :]) ** 2, axis=2))
    )
    expected_loss = float(
        np.mean(np.sum((predictions - one_hot[None, :, :]) ** 2, axis=2))
    )
    return {
        "bias_squared": bias_squared,
        "variance": variance,
        "expected_brier_loss": expected_loss,
        "decomposition_residual": expected_loss - bias_squared - variance,
    }


def align_probabilities(
    probabilities: ArrayLike,
    local_classes: ArrayLike,
    global_classes: ArrayLike,
) -> NDArray[np.float64]:
    local = np.asarray(probabilities, dtype=float)
    local_class_array = np.asarray(local_classes)
    global_class_array = np.asarray(global_classes)
    aligned = np.zeros((local.shape[0], global_class_array.size), dtype=float)
    indices = np.searchsorted(global_class_array, local_class_array)
    aligned[:, indices] = local
    return aligned


__all__ = [
    "align_probabilities",
    "brier_bias_variance",
    "class_distribution",
    "classification_metrics",
]
