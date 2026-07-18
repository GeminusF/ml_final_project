"""Leakage, encoding, resampling, and evaluation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.evaluation import (
    align_probabilities,
    brier_bias_variance,
    class_distribution,
    classification_metrics,
)
from src.utils.preprocessing import LeakageSafePreprocessor, random_oversample_minority


def test_preprocessor_uses_training_statistics_and_handles_unknown_category() -> None:
    train = pd.DataFrame(
        {"number": [0.0, 2.0, np.nan], "kind": ["a", "b", "?"]}
    )
    test = pd.DataFrame({"number": [100.0], "kind": ["unseen"]})
    preprocessor = LeakageSafePreprocessor(["kind"]).fit(train)
    transformed_train = preprocessor.transform(train)
    transformed_test = preprocessor.transform(test)

    assert preprocessor.numeric_means_["number"] == pytest.approx(1.0)  # type: ignore[index]
    assert transformed_train.shape[1] == transformed_test.shape[1]
    np.testing.assert_allclose(transformed_test[0, 1:], 0.0)
    assert transformed_test[0, 0] > 50.0


def test_oversampling_is_deterministic_and_never_downsamples() -> None:
    X = np.arange(200, dtype=float).reshape(100, 2)
    y = np.array(["major"] * 98 + ["minor"] * 2)
    first_X, first_y = random_oversample_minority(X, y, 0.05, 42)
    second_X, second_y = random_oversample_minority(X, y, 0.05, 42)

    np.testing.assert_array_equal(first_X, second_X)
    np.testing.assert_array_equal(first_y, second_y)
    _, counts = np.unique(first_y, return_counts=True)
    assert sorted(counts.tolist()) == [5, 98]


def test_metrics_class_order_auc_and_probability_alignment() -> None:
    y = np.array([10, 10, 30, 30])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    metrics = classification_metrics(y, y, probabilities, [10, 30])
    assert metrics == {"accuracy": 1.0, "f1_macro": 1.0, "auc_roc": 1.0}
    aligned = align_probabilities([[1.0], [1.0]], [30], [10, 30])
    np.testing.assert_allclose(aligned, [[0.0, 1.0], [0.0, 1.0]])
    assert class_distribution(y)["minority_fraction"] == pytest.approx(0.5)


def test_brier_decomposition_reconstructs_expected_loss() -> None:
    predictions = np.array(
        [
            [[0.8, 0.2], [0.1, 0.9]],
            [[0.6, 0.4], [0.2, 0.8]],
        ]
    )
    terms = brier_bias_variance(predictions, [0, 1], [0, 1])
    assert terms["decomposition_residual"] == pytest.approx(0.0, abs=1e-12)
    assert terms["expected_brier_loss"] == pytest.approx(
        terms["bias_squared"] + terms["variance"]
    )
