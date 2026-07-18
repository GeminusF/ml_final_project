"""Tests for discrete SAMME and bonus SAMME.R."""

from __future__ import annotations

import numpy as np
import pytest

from src.trees.adaboost import AdaBoostClassifier, DecisionStump


def test_decision_stump_has_one_split_at_most() -> None:
    X = np.arange(6, dtype=float).reshape(-1, 1)
    stump = DecisionStump().fit(X, [0, 0, 0, 1, 1, 1])
    assert stump.depth == 1


def test_hand_calculated_first_samme_round_and_learning_rate() -> None:
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.array([0, 1, 0, 1])
    model = AdaBoostClassifier(n_estimators=1, random_state=42).fit(X, y)
    slower = AdaBoostClassifier(
        n_estimators=1, learning_rate=0.5, random_state=42
    ).fit(X, y)

    assert model.estimator_errors[0] == pytest.approx(0.25)
    assert model.estimator_weights[0] == pytest.approx(np.log(3.0))
    assert slower.estimator_weights[0] == pytest.approx(0.5 * np.log(3.0))


def test_perfect_learner_stops_early_and_stages_match() -> None:
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.repeat(["no", "yes"], 4)
    model = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X, y)
    stages = list(model.staged_predict(X))

    assert len(model.estimators_) == 1
    assert len(stages) == 1
    np.testing.assert_array_equal(stages[-1], model.predict(X))
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_chance_first_learner_is_rejected() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="no better"):
        AdaBoostClassifier(n_estimators=3, random_state=42).fit(X, y)


def test_multiclass_samme_and_samme_r_probabilities() -> None:
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.repeat(np.array(["a", "b", "c"]), 4)
    for algorithm in ("SAMME", "SAMME.R"):
        model = AdaBoostClassifier(
            n_estimators=8,
            algorithm=algorithm,
            learning_rate=0.5,
            random_state=7,
        ).fit(X, y)
        probabilities = model.predict_proba(X)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert np.isfinite(probabilities).all()
        assert len(list(model.staged_predict(X))) == len(model.estimators_)


def test_numerical_clipping_keeps_weights_finite() -> None:
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.repeat([0, 1], 5)
    model = AdaBoostClassifier(
        n_estimators=5, algorithm="SAMME.R", random_state=42
    ).fit(X, y)
    assert np.isfinite(model.estimator_weights).all()
    assert np.isfinite(model.predict_proba(X)).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"learning_rate": 0.0},
        {"criterion": "invalid"},
        {"algorithm": "invalid"},
    ],
)
def test_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AdaBoostClassifier(**kwargs).fit([[0.0], [1.0]], [0, 1])
