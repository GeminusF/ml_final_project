"""Tests for the optional binary gradient-boosting implementation."""

from __future__ import annotations

import numpy as np
import pytest

from src.boosting.gradient_boosting import GradientBoostingClassifier


def test_loss_decreases_probabilities_are_stable_and_stages_match() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] - 0.3 * X[:, 1] > 0).astype(int)
    model = GradientBoostingClassifier(
        n_estimators=20, learning_rate=0.2, max_depth=2, random_state=42
    ).fit(X, y)

    assert model.train_score_ is not None
    assert model.train_score_[-1] < model.train_score_[0]
    probabilities = model.predict_proba(X)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    stages = list(model.staged_predict(X))
    assert len(stages) == 20
    np.testing.assert_array_equal(stages[-1], model.predict(X))
    assert np.mean(model.predict(X) == y) > 0.9


def test_binary_only_and_overfit_controls_are_validated() -> None:
    X = np.arange(9, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError, match="binary"):
        GradientBoostingClassifier().fit(X, np.repeat([0, 1, 2], 3))
    with pytest.raises(ValueError, match="max_depth"):
        GradientBoostingClassifier(max_depth=0).fit(X[:4], [0, 0, 1, 1])
