"""A compact NumPy implementation of a weighted CART classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


_EPS = 1e-12


@dataclass(slots=True)
class Node:
    """One node in a classification tree."""

    feature_index: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None
    value: NDArray[np.float64] | None = None
    samples: int = 0
    weighted_samples: float = 0.0
    impurity: float = 0.0
    impurity_decrease: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTree:
    """Binary-split CART classifier supporting weighted multiclass targets.

    Depth is counted from zero: ``max_depth=0`` creates a root-only tree and
    ``max_depth=1`` creates a decision stump with at most one split.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.classes_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None
        self._root: Node | None = None
        self._feature_importance_totals: NDArray[np.float64] | None = None
        self._rng: np.random.Generator | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> DecisionTree:
        """Fit the tree; sample weights are optional and scale invariant."""

        X_array, y_array, weights = self._validate_training_data(
            X, y, sample_weight
        )
        self._validate_parameters(X_array.shape[1])
        self.classes_, encoded_y = np.unique(y_array, return_inverse=True)
        self.n_features_in_ = X_array.shape[1]
        self._feature_importance_totals = np.zeros(
            self.n_features_in_, dtype=float
        )
        self._rng = np.random.default_rng(self.random_state)
        self._root = self._grow_tree(
            X_array, encoded_y.astype(np.int64, copy=False), weights, depth=0
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict original class labels."""

        probabilities = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return empirical weighted class probabilities at reached leaves."""

        X_array = self._validate_prediction_data(X)
        assert self._root is not None and self.classes_ is not None
        output = np.empty((X_array.shape[0], self.classes_.size), dtype=float)
        for row_index, row in enumerate(X_array):
            node = self._root
            while not node.is_leaf:
                assert node.feature_index is not None and node.threshold is not None
                next_node = (
                    node.left
                    if row[node.feature_index] <= node.threshold
                    else node.right
                )
                assert next_node is not None
                node = next_node
            assert node.value is not None
            total = float(node.value.sum())
            output[row_index] = (
                node.value / total
                if total > _EPS
                else np.full(self.classes_.size, 1.0 / self.classes_.size)
            )
        return output

    @property
    def depth(self) -> int:
        """Maximum number of edges from root to a leaf."""

        self._check_is_fitted()
        assert self._root is not None

        def visit(node: Node) -> int:
            if node.is_leaf:
                return 0
            assert node.left is not None and node.right is not None
            return 1 + max(visit(node.left), visit(node.right))

        return visit(self._root)

    @property
    def n_leaves(self) -> int:
        """Number of terminal nodes."""

        self._check_is_fitted()
        assert self._root is not None

        def visit(node: Node) -> int:
            if node.is_leaf:
                return 1
            assert node.left is not None and node.right is not None
            return visit(node.left) + visit(node.right)

        return visit(self._root)

    def feature_importances(self) -> NDArray[np.float64]:
        """Return normalised weighted impurity decrease by feature."""

        self._check_is_fitted()
        assert self._feature_importance_totals is not None
        importances = self._feature_importance_totals.copy()
        total = float(importances.sum())
        if total > _EPS:
            importances /= total
        return importances

    def __repr__(self) -> str:
        if self._root is None:
            return (
                "DecisionTree("
                f"max_depth={self.max_depth!r}, criterion={self.criterion!r}, "
                "fitted=False)"
            )
        if self.depth > 4:
            return (
                f"DecisionTree(depth={self.depth}, n_leaves={self.n_leaves}, "
                f"criterion={self.criterion!r})"
            )
        lines: list[str] = []

        def visit(node: Node, indent: str, branch: str) -> None:
            distribution = np.array2string(
                node.value if node.value is not None else np.array([]),
                precision=3,
                suppress_small=True,
            )
            prefix = f"{indent}{branch}" if branch else indent
            if node.is_leaf:
                lines.append(
                    f"{prefix}leaf impurity={node.impurity:.4f} "
                    f"samples={node.samples} value={distribution}"
                )
                return
            lines.append(
                f"{prefix}X[{node.feature_index}] <= {node.threshold:.6g} "
                f"{self.criterion}={node.impurity:.4f} "
                f"samples={node.samples} value={distribution}"
            )
            assert node.left is not None and node.right is not None
            visit(node.left, indent + "  ", "L: ")
            visit(node.right, indent + "  ", "R: ")

        visit(self._root, "", "")
        return "\n".join(lines)

    def _grow_tree(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        weights: NDArray[np.float64],
        depth: int,
    ) -> Node:
        assert self.classes_ is not None
        class_weights = np.bincount(
            y, weights=weights, minlength=self.classes_.size
        ).astype(float)
        weighted_samples = float(weights.sum())
        impurity = self._impurity(class_weights)
        node = Node(
            value=class_weights,
            samples=X.shape[0],
            weighted_samples=weighted_samples,
            impurity=impurity,
        )
        depth_limit = self.max_depth is not None and depth >= self.max_depth
        too_small = X.shape[0] < max(2, self.min_samples_split)
        pure = np.count_nonzero(class_weights > _EPS) <= 1
        if depth_limit or too_small or pure:
            return node
        split = self._best_split(X, y, weights, class_weights, impurity)
        if split is None:
            return node
        feature_index, threshold, gain = split
        left_mask = X[:, feature_index] <= threshold
        if left_mask.all() or (~left_mask).all():
            return node
        node.feature_index = feature_index
        node.threshold = threshold
        node.impurity_decrease = gain
        assert self._feature_importance_totals is not None
        self._feature_importance_totals[feature_index] += weighted_samples * gain
        node.left = self._grow_tree(
            X[left_mask], y[left_mask], weights[left_mask], depth + 1
        )
        node.right = self._grow_tree(
            X[~left_mask], y[~left_mask], weights[~left_mask], depth + 1
        )
        return node

    def _best_split(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        weights: NDArray[np.float64],
        parent_class_weights: NDArray[np.float64],
        parent_impurity: float,
    ) -> tuple[int, float, float] | None:
        assert self.n_features_in_ is not None and self.classes_ is not None
        parent_weight = float(parent_class_weights.sum())
        if parent_weight <= _EPS:
            return None
        best_gain = _EPS
        best_feature = -1
        best_threshold = 0.0
        for feature_index in self._candidate_features(self.n_features_in_):
            order = np.argsort(X[:, feature_index], kind="mergesort")
            values = X[order, feature_index]
            labels = y[order]
            ordered_weights = weights[order]
            left_class_weights = np.zeros(self.classes_.size, dtype=float)
            for split_position in range(values.size - 1):
                left_class_weights[labels[split_position]] += ordered_weights[
                    split_position
                ]
                if values[split_position] == values[split_position + 1]:
                    continue
                left_weight = float(left_class_weights.sum())
                right_weight = parent_weight - left_weight
                if left_weight <= _EPS or right_weight <= _EPS:
                    continue
                right_class_weights = parent_class_weights - left_class_weights
                children_impurity = (
                    left_weight * self._impurity(left_class_weights)
                    + right_weight * self._impurity(right_class_weights)
                ) / parent_weight
                gain = parent_impurity - children_impurity
                if gain > best_gain + _EPS:
                    lower = float(values[split_position])
                    upper = float(values[split_position + 1])
                    best_gain = float(gain)
                    best_feature = int(feature_index)
                    best_threshold = lower + (upper - lower) / 2.0
        if best_feature < 0:
            return None
        return best_feature, best_threshold, best_gain

    def _candidate_features(self, n_features: int) -> NDArray[np.int64]:
        count = self._resolved_max_features(n_features)
        if count == n_features:
            return np.arange(n_features, dtype=np.int64)
        assert self._rng is not None
        return np.sort(
            self._rng.choice(n_features, size=count, replace=False)
        ).astype(np.int64)

    def _resolved_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be an int, 'sqrt', 'log2', or None")

    def _impurity(self, class_weights: NDArray[np.float64]) -> float:
        total = float(class_weights.sum())
        if total <= _EPS:
            return 0.0
        probabilities = class_weights / total
        if self.criterion == "gini":
            return float(1.0 - np.dot(probabilities, probabilities))
        positive = probabilities > 0.0
        return float(
            -np.sum(probabilities[positive] * np.log2(probabilities[positive]))
        )

    def _validate_parameters(self, n_features: int) -> None:
        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or self.max_depth < 0
        ):
            raise ValueError("max_depth must be a non-negative int or None")
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 1:
            raise ValueError("min_samples_split must be an integer >= 1")
        if self.criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if isinstance(self.max_features, int) and self.max_features < 1:
            raise ValueError("integer max_features must be >= 1")
        self._resolved_max_features(n_features)

    @staticmethod
    def _validate_training_data(
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None,
    ) -> tuple[NDArray[np.float64], NDArray[Any], NDArray[np.float64]]:
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric array")
        if y_array.ndim != 1:
            raise ValueError("y must be a one-dimensional array")
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X must contain at least one row and one feature")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        if sample_weight is None:
            weights = np.ones(X_array.shape[0], dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.ndim != 1 or weights.shape[0] != X_array.shape[0]:
                raise ValueError("sample_weight must have shape (n_samples,)")
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError("sample_weight must be finite and non-negative")
            if float(weights.sum()) <= _EPS:
                raise ValueError("sample_weight must have a positive total")
        return X_array, y_array, weights

    def _validate_prediction_data(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric array")
        assert self.n_features_in_ is not None
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features; expected {self.n_features_in_}"
            )
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        return X_array

    def _check_is_fitted(self) -> None:
        if self._root is None or self.classes_ is None:
            raise RuntimeError("DecisionTree is not fitted")


__all__ = ["DecisionTree", "Node"]
