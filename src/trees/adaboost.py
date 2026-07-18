"""AdaBoost with weighted decision stumps (SAMME and SAMME.R)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.trees.decision_tree import DecisionTree


_ERROR_EPS = 1e-10
_PROBABILITY_EPS = 1e-12


class DecisionStump(DecisionTree):
    """Convenience tree restricted to one binary split."""

    def __init__(
        self,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            max_depth=1,
            criterion=criterion,
            random_state=random_state,
        )


class AdaBoostClassifier:
    """Discrete SAMME or real-valued SAMME.R boosting.

    ``algorithm`` is a backward-compatible bonus extension; the default remains
    the discrete SAMME algorithm required by the brief.  Probabilities are
    derived from ensemble scores and should not be treated as calibrated.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        random_state: int | None = None,
        algorithm: Literal["SAMME", "SAMME.R"] = "SAMME",
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state
        self.algorithm = algorithm

        self.classes_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None
        self.estimators_: list[DecisionStump] = []
        self._estimator_weights: list[float] = []
        self._estimator_errors: list[float] = []

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> AdaBoostClassifier:
        X_array, y_array = self._validate_training_data(X, y)
        self._validate_parameters()
        self.classes_ = np.unique(y_array)
        if self.classes_.size < 2:
            raise ValueError("AdaBoost requires at least two classes")
        self.n_features_in_ = X_array.shape[1]
        self.estimators_ = []
        self._estimator_weights = []
        self._estimator_errors = []

        n_samples = X_array.shape[0]
        n_classes = self.classes_.size
        weights = np.full(n_samples, 1.0 / n_samples, dtype=float)
        encoded_y = np.searchsorted(self.classes_, y_array)
        seed_sequence = np.random.SeedSequence(self.random_state)
        child_sequences = seed_sequence.spawn(self.n_estimators)

        for round_index, child_sequence in enumerate(child_sequences, start=1):
            stump_seed = int(child_sequence.generate_state(1, dtype=np.uint32)[0])
            stump = DecisionStump(self.criterion, stump_seed)
            stump.fit(X_array, y_array, sample_weight=weights)
            predictions = stump.predict(X_array)
            incorrect = predictions != y_array
            error = float(np.dot(weights, incorrect))
            if progress_callback is not None:
                progress_callback(
                    round_index,
                    self.n_estimators,
                    f"{self.algorithm} round {round_index}/{self.n_estimators}",
                )
            chance_error = 1.0 - 1.0 / n_classes

            if error >= chance_error - _ERROR_EPS:
                if not self.estimators_:
                    raise ValueError(
                        "The first stump is no better than random guessing"
                    )
                break

            clipped_error = float(np.clip(error, _ERROR_EPS, 1.0 - _ERROR_EPS))
            if self.algorithm == "SAMME":
                estimator_weight = self.learning_rate * (
                    np.log((1.0 - clipped_error) / clipped_error)
                    + np.log(n_classes - 1.0)
                )
            else:
                estimator_weight = self.learning_rate

            self.estimators_.append(stump)
            self._estimator_weights.append(float(estimator_weight))
            self._estimator_errors.append(error)

            if error <= _ERROR_EPS:
                break

            if self.algorithm == "SAMME":
                weights *= np.exp(estimator_weight * incorrect)
            else:
                probabilities = self._aligned_probabilities(stump, X_array)
                log_probabilities = np.log(
                    np.clip(probabilities, _PROBABILITY_EPS, 1.0)
                )
                class_coding = np.full(
                    (n_samples, n_classes), -1.0 / (n_classes - 1.0)
                )
                class_coding[np.arange(n_samples), encoded_y] = 1.0
                exponent = -self.learning_rate * (n_classes - 1.0) / n_classes
                sample_exponents = exponent * np.sum(
                    class_coding * log_probabilities, axis=1
                )
                weights *= np.exp(np.clip(sample_exponents, -700.0, 700.0))

            weight_sum = float(weights.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                raise FloatingPointError("AdaBoost sample weights became invalid")
            weights /= weight_sum

        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        scores = self.decision_function(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Convert aggregate votes/log-probabilities through a stable softmax."""

        scores = self.decision_function(X)
        assert self.classes_ is not None
        scale = max(1, self.classes_.size - 1)
        scaled = scores / scale
        scaled -= scaled.max(axis=1, keepdims=True)
        exponentiated = np.exp(scaled)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        X_array = self._validate_prediction_data(X)
        scores = np.zeros((X_array.shape[0], self._n_classes), dtype=float)
        normalizer = 0.0
        for estimator, estimator_weight in zip(
            self.estimators_, self._estimator_weights, strict=True
        ):
            self._add_estimator_scores(
                scores, estimator, estimator_weight, X_array
            )
            normalizer += abs(estimator_weight)
        if normalizer > 0.0:
            scores /= normalizer
        return scores

    @property
    def estimator_weights(self) -> NDArray[np.float64]:
        self._check_is_fitted()
        return np.asarray(self._estimator_weights, dtype=float)

    @property
    def estimator_errors(self) -> NDArray[np.float64]:
        self._check_is_fitted()
        return np.asarray(self._estimator_errors, dtype=float)

    def staged_predict(self, X: ArrayLike) -> Iterator[NDArray[Any]]:
        """Yield predictions after each accepted boosting round."""

        X_array = self._validate_prediction_data(X)
        assert self.classes_ is not None
        scores = np.zeros((X_array.shape[0], self._n_classes), dtype=float)
        for estimator, estimator_weight in zip(
            self.estimators_, self._estimator_weights, strict=True
        ):
            self._add_estimator_scores(
                scores, estimator, estimator_weight, X_array
            )
            yield self.classes_[np.argmax(scores, axis=1)]

    def _add_estimator_scores(
        self,
        scores: NDArray[np.float64],
        estimator: DecisionStump,
        estimator_weight: float,
        X: NDArray[np.float64],
    ) -> None:
        assert self.classes_ is not None
        if self.algorithm == "SAMME":
            predictions = estimator.predict(X)
            indices = np.searchsorted(self.classes_, predictions)
            scores[np.arange(X.shape[0]), indices] += estimator_weight
            return
        probabilities = self._aligned_probabilities(estimator, X)
        log_probabilities = np.log(
            np.clip(probabilities, _PROBABILITY_EPS, 1.0)
        )
        centered = log_probabilities - log_probabilities.mean(
            axis=1, keepdims=True
        )
        scores += estimator_weight * (self._n_classes - 1.0) * centered

    def _aligned_probabilities(
        self, estimator: DecisionStump, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        assert self.classes_ is not None and estimator.classes_ is not None
        local = estimator.predict_proba(X)
        aligned = np.full(
            (X.shape[0], self.classes_.size), _PROBABILITY_EPS, dtype=float
        )
        indices = np.searchsorted(self.classes_, estimator.classes_)
        aligned[:, indices] = local
        aligned /= aligned.sum(axis=1, keepdims=True)
        return aligned

    @property
    def _n_classes(self) -> int:
        assert self.classes_ is not None
        return int(self.classes_.size)

    def _validate_parameters(self) -> None:
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if self.criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if self.algorithm not in {"SAMME", "SAMME.R"}:
            raise ValueError("algorithm must be 'SAMME' or 'SAMME.R'")

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
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected number of features")
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        return X_array

    def _check_is_fitted(self) -> None:
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError("AdaBoostClassifier is not fitted")


__all__ = ["AdaBoostClassifier", "DecisionStump"]
