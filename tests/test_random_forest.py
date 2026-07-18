"""Tests for bootstrap aggregation, OOB estimates, and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from src.trees.random_forest import RandomForestClassifier


def _classification_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, "positive", "negative")
    return X, y


def test_bootstrap_and_oob_indices_are_complements() -> None:
    X, y = _classification_data()
    forest = RandomForestClassifier(
        n_estimators=7, oob_score=True, random_state=42
    ).fit(X, y)

    for sampled, oob in zip(
        forest.bootstrap_indices_, forest.oob_indices_, strict=True
    ):
        assert sampled.size == X.shape[0]
        expected = np.setdiff1d(np.arange(X.shape[0]), np.unique(sampled))
        np.testing.assert_array_equal(oob, expected)
    assert forest.oob_coverage_ is not None and 0.0 < forest.oob_coverage_ <= 1.0
    assert 0.0 <= forest.oob_score_ <= 1.0


def test_class_alignment_probability_normalization_and_importance() -> None:
    X, y = _classification_data()
    forest = RandomForestClassifier(
        n_estimators=9, max_depth=3, random_state=3
    ).fit(X, y)

    probabilities = forest.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(forest.predict(X)) <= set(y)
    assert forest.feature_importances_.shape == (X.shape[1],)
    np.testing.assert_allclose(forest.feature_importances_.sum(), 1.0)


def test_sequential_and_parallel_runs_are_identical() -> None:
    X, y = _classification_data()
    sequential_updates: list[int] = []
    parallel_updates: list[int] = []
    sequential = RandomForestClassifier(
        n_estimators=5, max_depth=3, n_jobs=1, random_state=11
    ).fit(
        X,
        y,
        progress_callback=lambda completed, total, detail: (
            sequential_updates.append(completed)
        ),
    )
    parallel = RandomForestClassifier(
        n_estimators=5, max_depth=3, n_jobs=2, random_state=11
    ).fit(
        X,
        y,
        progress_callback=lambda completed, total, detail: (
            parallel_updates.append(completed)
        ),
    )

    assert sequential_updates == [1, 2, 3, 4, 5]
    assert parallel_updates == [1, 2, 3, 4, 5]
    np.testing.assert_array_equal(sequential.predict(X), parallel.predict(X))
    np.testing.assert_allclose(sequential.predict_proba(X), parallel.predict_proba(X))
    for left, right in zip(
        sequential.bootstrap_indices_, parallel.bootstrap_indices_, strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_bootstrap_false_has_no_oob_samples() -> None:
    X, y = _classification_data()
    forest = RandomForestClassifier(
        n_estimators=3, bootstrap=False, random_state=42
    ).fit(X, y)
    assert all(indices.size == 0 for indices in forest.oob_indices_)
    with pytest.raises(AttributeError, match="oob_score"):
        _ = forest.oob_score_


def test_invalid_oob_combination_and_single_class() -> None:
    X, y = _classification_data()
    with pytest.raises(ValueError, match="requires bootstrap"):
        RandomForestClassifier(bootstrap=False, oob_score=True).fit(X, y)
    with pytest.raises(ValueError, match="at least two"):
        RandomForestClassifier().fit(X, np.zeros(X.shape[0]))


def test_voting_ties_follow_class_order() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1, 2, 1, 2])
    forest = RandomForestClassifier(
        n_estimators=2, max_depth=0, bootstrap=False, random_state=42
    ).fit(X, y)
    assert set(forest.predict(X)) == {1}
