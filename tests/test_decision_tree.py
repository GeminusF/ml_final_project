"""Unit tests for the weighted CART classifier."""

from __future__ import annotations

import numpy as np
import pytest

from src.trees.decision_tree import DecisionTree


def test_known_midpoint_split_and_tree_statistics() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["left", "left", "right", "right"])
    tree = DecisionTree(max_depth=1, random_state=42).fit(X, y)
    assert tree._root is not None
    assert tree._root.threshold == pytest.approx(1.5)
    assert tree.depth == 1 and tree.n_leaves == 2
    assert tree.predict(X).tolist() == y.tolist()
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1.0)
    np.testing.assert_allclose(tree.feature_importances(), [1.0])


def test_xor_documents_greedy_zero_gain_behavior() -> None:
    """CART correctly rejects XOR's zero-gain root split."""

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    tree = DecisionTree(max_depth=2, random_state=42).fit(X, y)
    assert tree.depth == 0
    assert np.mean(tree.predict(X) == y) == pytest.approx(0.5)


def test_two_levels_learn_a_nonzero_gain_hierarchy() -> None:
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 2, 2])
    tree = DecisionTree(max_depth=2, random_state=42).fit(X, y)
    assert tree.depth == 2
    np.testing.assert_array_equal(tree.predict(X), y)


def test_depth_zero_constant_features_and_one_class_stop_at_root() -> None:
    X = np.ones((5, 1))
    tree = DecisionTree(max_depth=0).fit(X, np.array([0, 1, 1, 1, 0]))
    pure = DecisionTree().fit(X, np.array(["same"] * 5))
    assert tree.depth == 0 and tree.n_leaves == 1
    assert pure.depth == 0 and pure.n_leaves == 1
    assert set(pure.predict(X)) == {"same"}


def test_weighted_split_and_zero_weight_sample() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array(["a", "b", "a"])
    tree = DecisionTree(max_depth=1).fit(
        X, y, sample_weight=np.array([1.0, 8.0, 0.0])
    )
    assert tree._root is not None
    assert tree._root.threshold == pytest.approx(0.5)
    assert tree.predict([[2.0]])[0] == "b"


def test_multiclass_noncontiguous_labels_and_entropy() -> None:
    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.array([10, 10, 10, 30, 30, 30, 90, 90, 90])
    tree = DecisionTree(max_depth=3, criterion="entropy").fit(X, y)
    np.testing.assert_array_equal(tree.classes_, [10, 30, 90])
    np.testing.assert_array_equal(tree.predict(X), y)
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1.0)


def test_feature_subsampling_and_ties_are_reproducible() -> None:
    X = np.column_stack((np.arange(20), np.arange(20), np.arange(20))).astype(float)
    y = np.repeat([0, 1], 10)
    first = DecisionTree(max_depth=2, max_features=1, random_state=7).fit(X, y)
    second = DecisionTree(max_depth=2, max_features=1, random_state=7).fit(X, y)
    deterministic = DecisionTree(max_depth=1, random_state=7).fit(X[:, :2], y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert deterministic._root is not None
    assert deterministic._root.feature_index == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": -1}, "max_depth"),
        ({"min_samples_split": 0}, "min_samples_split"),
        ({"criterion": "mse"}, "criterion"),
        ({"max_features": 0}, "max_features"),
    ],
)
def test_parameter_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DecisionTree(**kwargs).fit([[0.0], [1.0]], [0, 1])


def test_weight_validation_and_unfitted_errors() -> None:
    with pytest.raises(ValueError, match="positive total"):
        DecisionTree().fit([[0.0], [1.0]], [0, 1], [0.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        DecisionTree().fit([[0.0], [1.0]], [0, 1], [1.0, -1.0])
    with pytest.raises(RuntimeError, match="not fitted"):
        DecisionTree().predict([[0.0]])


def test_min_samples_split_one_remains_safe() -> None:
    tree = DecisionTree(min_samples_split=1).fit([[0.0]], ["only"])
    assert tree.depth == 0
    assert tree.predict([[3.0]])[0] == "only"
