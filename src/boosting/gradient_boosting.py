"""Bonus binary gradient boosting classifier using shallow regression trees."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


_EPS = 1e-12


@dataclass(slots=True)
class _RegressionNode:
    value: float
    feature_index: int | None = None
    threshold: float | None = None
    left: _RegressionNode | None = None
    right: _RegressionNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class _RegressionTree:
    """Least-squares tree used only for fitting pseudo-residuals."""

    def __init__(self, max_depth: int, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_: _RegressionNode | None = None

    def fit(
        self, X: NDArray[np.float64], target: NDArray[np.float64]
    ) -> _RegressionTree:
        self.root_ = self._grow(X, target, 0)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.root_ is None:
            raise RuntimeError("Regression tree is not fitted")
        output = np.empty(X.shape[0], dtype=float)
        for index, row in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                assert node.feature_index is not None and node.threshold is not None
                next_node = (
                    node.left
                    if row[node.feature_index] <= node.threshold
                    else node.right
                )
                assert next_node is not None
                node = next_node
            output[index] = node.value
        return output

    def _grow(
        self,
        X: NDArray[np.float64],
        target: NDArray[np.float64],
        depth: int,
    ) -> _RegressionNode:
        node = _RegressionNode(value=float(np.mean(target)))
        if (
            depth >= self.max_depth
            or target.size < max(2, self.min_samples_split)
            or np.allclose(target, target[0])
        ):
            return node
        split = self._best_split(X, target)
        if split is None:
            return node
        feature_index, threshold = split
        left_mask = X[:, feature_index] <= threshold
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow(X[left_mask], target[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], target[~left_mask], depth + 1)
        return node

    @staticmethod
    def _best_split(
        X: NDArray[np.float64], target: NDArray[np.float64]
    ) -> tuple[int, float] | None:
        best_loss = float("inf")
        best_feature = -1
        best_threshold = 0.0
        for feature_index in range(X.shape[1]):
            order = np.argsort(X[:, feature_index], kind="mergesort")
            values = X[order, feature_index]
            ordered_target = target[order]
            prefix = np.cumsum(ordered_target)
            prefix_squared = np.cumsum(ordered_target**2)
            total = float(prefix[-1])
            total_squared = float(prefix_squared[-1])
            for position in range(values.size - 1):
                if values[position] == values[position + 1]:
                    continue
                left_count = position + 1
                right_count = values.size - left_count
                left_sum = float(prefix[position])
                right_sum = total - left_sum
                left_sse = float(prefix_squared[position]) - left_sum**2 / left_count
                right_sse = (
                    total_squared
                    - float(prefix_squared[position])
                    - right_sum**2 / right_count
                )
                loss = max(0.0, left_sse) + max(0.0, right_sse)
                if loss < best_loss - _EPS:
                    lower = float(values[position])
                    upper = float(values[position + 1])
                    best_loss = loss
                    best_feature = feature_index
                    best_threshold = lower + (upper - lower) / 2.0
        if best_feature < 0:
            return None
        return best_feature, best_threshold


class GradientBoostingClassifier:
    """Binary gradient boosting with log-loss and residual regression trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 1,
        min_samples_split: int = 2,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.classes_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None
        self.estimators_: list[_RegressionTree] = []
        self.initial_log_odds_: float | None = None
        self.train_score_: NDArray[np.float64] | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> GradientBoostingClassifier:
        X_array, y_array = self._validate_training_data(X, y)
        self._validate_parameters()
        self.classes_ = np.unique(y_array)
        if self.classes_.size != 2:
            raise ValueError("GradientBoostingClassifier supports binary targets")
        self.n_features_in_ = X_array.shape[1]
        encoded_y = (y_array == self.classes_[1]).astype(float)
        positive_rate = float(
            np.clip(encoded_y.mean(), _EPS, 1.0 - _EPS)
        )
        self.initial_log_odds_ = float(
            np.log(positive_rate / (1.0 - positive_rate))
        )
        raw_scores = np.full(X_array.shape[0], self.initial_log_odds_)
        self.estimators_ = []
        losses: list[float] = []

        for estimator_index in range(1, self.n_estimators + 1):
            probabilities = self._sigmoid(raw_scores)
            residuals = encoded_y - probabilities
            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            ).fit(X_array, residuals)
            raw_scores += self.learning_rate * tree.predict(X_array)
            self.estimators_.append(tree)
            losses.append(self._log_loss(encoded_y, self._sigmoid(raw_scores)))
            if progress_callback is not None:
                progress_callback(
                    estimator_index,
                    self.n_estimators,
                    f"GBM tree {estimator_index}/{self.n_estimators}",
                )
        self.train_score_ = np.asarray(losses, dtype=float)
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        probabilities = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[(probabilities[:, 1] >= 0.5).astype(int)]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        X_array = self._validate_prediction_data(X)
        positive = self._sigmoid(self._raw_predict(X_array))
        return np.column_stack((1.0 - positive, positive))

    def staged_predict(self, X: ArrayLike) -> Iterator[NDArray[Any]]:
        X_array = self._validate_prediction_data(X)
        assert self.classes_ is not None and self.initial_log_odds_ is not None
        raw_scores = np.full(X_array.shape[0], self.initial_log_odds_)
        for tree in self.estimators_:
            raw_scores += self.learning_rate * tree.predict(X_array)
            yield self.classes_[(self._sigmoid(raw_scores) >= 0.5).astype(int)]

    def _raw_predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.initial_log_odds_ is not None
        raw_scores = np.full(X.shape[0], self.initial_log_odds_)
        for tree in self.estimators_:
            raw_scores += self.learning_rate * tree.predict(X)
        return raw_scores

    @staticmethod
    def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
        output = np.empty_like(values, dtype=float)
        positive = values >= 0.0
        output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponentiated = np.exp(values[~positive])
        output[~positive] = exponentiated / (1.0 + exponentiated)
        return output

    @staticmethod
    def _log_loss(
        y: NDArray[np.float64], probabilities: NDArray[np.float64]
    ) -> float:
        clipped = np.clip(probabilities, _EPS, 1.0 - _EPS)
        return float(
            -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
        )

    def _validate_parameters(self) -> None:
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise ValueError("max_depth must be an integer >= 1")
        if (
            not isinstance(self.min_samples_split, int)
            or self.min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer >= 2")

    @staticmethod
    def _validate_training_data(
        X: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[Any]]:
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)
        if X_array.ndim != 2 or y_array.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")
        if X_array.shape[0] == 0 or X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must contain the same non-zero row count")
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        return X_array, y_array

    def _validate_prediction_data(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError("GradientBoostingClassifier is not fitted")
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected shape")
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        return X_array


__all__ = ["GradientBoostingClassifier"]
