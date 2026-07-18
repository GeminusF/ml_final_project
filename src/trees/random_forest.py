"""Random Forest built from the project's NumPy DecisionTree."""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.trees.decision_tree import DecisionTree


@dataclass(slots=True)
class _TreeTask:
    index: int
    X: NDArray[np.float64]
    y: NDArray[Any]
    sample_indices: NDArray[np.int64]
    oob_indices: NDArray[np.int64]
    max_depth: int | None
    max_features: int | str | None
    min_samples_split: int
    random_state: int


def _fit_tree_task(task: _TreeTask) -> tuple[int, DecisionTree, NDArray[np.int64]]:
    """Fit one tree in a top-level, spawn-pickleable worker function."""

    tree = DecisionTree(
        max_depth=task.max_depth,
        min_samples_split=task.min_samples_split,
        criterion="gini",
        max_features=task.max_features,
        random_state=task.random_state,
    )
    tree.fit(task.X[task.sample_indices], task.y[task.sample_indices])
    return task.index, tree, task.oob_indices


class RandomForestClassifier:
    """Bootstrap aggregation with per-node random feature sub-sampling."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | str = "sqrt",
        min_samples_split: int = 2,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classes_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None
        self.estimators_: list[DecisionTree] = []
        self.bootstrap_indices_: list[NDArray[np.int64]] = []
        self.oob_indices_: list[NDArray[np.int64]] = []
        self.oob_coverage_: float | None = None
        self._oob_score: float | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> RandomForestClassifier:
        """Fit deterministic bootstrap trees, optionally in spawned workers."""

        X_array, y_array = self._validate_training_data(X, y)
        self._validate_parameters(X_array.shape[1])
        self.classes_ = np.unique(y_array)
        self.n_features_in_ = X_array.shape[1]
        tasks = self._build_tasks(X_array, y_array)

        workers = self._resolved_workers()
        if workers == 1:
            fitted = []
            for completed, task in enumerate(tasks, start=1):
                fitted.append(_fit_tree_task(task))
                if progress_callback is not None:
                    progress_callback(
                        completed,
                        self.n_estimators,
                        f"forest tree {completed}/{self.n_estimators}",
                    )
        else:
            context = mp.get_context("spawn")
            try:
                with context.Pool(processes=workers) as pool:
                    fitted = []
                    results = pool.imap_unordered(_fit_tree_task, tasks)
                    for completed, result in enumerate(results, start=1):
                        fitted.append(result)
                        if progress_callback is not None:
                            progress_callback(
                                completed,
                                self.n_estimators,
                                f"forest tree {completed}/{self.n_estimators}",
                            )
            except (OSError, PermissionError) as error:
                warnings.warn(
                    "Multiprocessing is unavailable in this environment; "
                    "using the deterministic sequential fallback "
                    f"({error}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                fitted = []
                for completed, task in enumerate(tasks, start=1):
                    fitted.append(_fit_tree_task(task))
                    if progress_callback is not None:
                        progress_callback(
                            completed,
                            self.n_estimators,
                            f"forest tree {completed}/{self.n_estimators}",
                        )

        fitted.sort(key=lambda item: item[0])
        self.estimators_ = [tree for _, tree, _ in fitted]
        self.oob_indices_ = [indices for _, _, indices in fitted]
        if self.oob_score:
            self._compute_oob_score(X_array, y_array)
        else:
            self._oob_score = None
            self.oob_coverage_ = None
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        probabilities = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        X_array = self._validate_prediction_data(X)
        assert self.classes_ is not None
        probabilities = np.zeros(
            (X_array.shape[0], self.classes_.size), dtype=float
        )
        for tree in self.estimators_:
            probabilities += self._aligned_probabilities(tree, X_array)
        probabilities /= len(self.estimators_)
        return probabilities

    @property
    def oob_score_(self) -> float:
        self._check_is_fitted()
        if not self.oob_score or self._oob_score is None:
            raise AttributeError("oob_score_ is available only when oob_score=True")
        return self._oob_score

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        self._check_is_fitted()
        importances = np.mean(
            [tree.feature_importances() for tree in self.estimators_], axis=0
        )
        total = float(importances.sum())
        if total > 0.0:
            importances /= total
        return np.asarray(importances, dtype=float)

    def _build_tasks(
        self, X: NDArray[np.float64], y: NDArray[Any]
    ) -> list[_TreeTask]:
        sample_count = X.shape[0]
        seed_sequence = np.random.SeedSequence(self.random_state)
        children = seed_sequence.spawn(self.n_estimators * 2)
        tasks: list[_TreeTask] = []
        self.bootstrap_indices_ = []
        for tree_index in range(self.n_estimators):
            bootstrap_seed = int(
                children[2 * tree_index].generate_state(1, dtype=np.uint32)[0]
            )
            tree_seed = int(
                children[2 * tree_index + 1].generate_state(
                    1, dtype=np.uint32
                )[0]
            )
            if self.bootstrap:
                rng = np.random.default_rng(bootstrap_seed)
                sample_indices = rng.integers(
                    0, sample_count, size=sample_count, dtype=np.int64
                )
                in_bag = np.zeros(sample_count, dtype=bool)
                in_bag[sample_indices] = True
                oob_indices = np.flatnonzero(~in_bag).astype(np.int64)
            else:
                sample_indices = np.arange(sample_count, dtype=np.int64)
                oob_indices = np.empty(0, dtype=np.int64)
            self.bootstrap_indices_.append(sample_indices.copy())
            tasks.append(
                _TreeTask(
                    index=tree_index,
                    X=X,
                    y=y,
                    sample_indices=sample_indices,
                    oob_indices=oob_indices,
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split,
                    random_state=tree_seed,
                )
            )
        return tasks

    def _compute_oob_score(
        self, X: NDArray[np.float64], y: NDArray[Any]
    ) -> None:
        assert self.classes_ is not None
        vote_totals = np.zeros((X.shape[0], self.classes_.size), dtype=float)
        voter_counts = np.zeros(X.shape[0], dtype=np.int64)
        for tree, oob_indices in zip(
            self.estimators_, self.oob_indices_, strict=True
        ):
            if oob_indices.size == 0:
                continue
            vote_totals[oob_indices] += self._aligned_probabilities(
                tree, X[oob_indices]
            )
            voter_counts[oob_indices] += 1
        valid = voter_counts > 0
        self.oob_coverage_ = float(valid.mean())
        if not valid.any():
            self._oob_score = float("nan")
            return
        predictions = self.classes_[np.argmax(vote_totals[valid], axis=1)]
        self._oob_score = float(np.mean(predictions == y[valid]))

    def _aligned_probabilities(
        self, tree: DecisionTree, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        assert self.classes_ is not None and tree.classes_ is not None
        local = tree.predict_proba(X)
        aligned = np.zeros((X.shape[0], self.classes_.size), dtype=float)
        indices = np.searchsorted(self.classes_, tree.classes_)
        aligned[:, indices] = local
        return aligned

    def _resolved_workers(self) -> int:
        if self.n_jobs == -1:
            return max(1, os.cpu_count() or 1)
        return self.n_jobs

    def _validate_parameters(self, n_features: int) -> None:
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or self.max_depth < 0
        ):
            raise ValueError("max_depth must be a non-negative int or None")
        if (
            not isinstance(self.min_samples_split, int)
            or self.min_samples_split < 1
        ):
            raise ValueError("min_samples_split must be an integer >= 1")
        if self.n_jobs != -1 and (
            not isinstance(self.n_jobs, int) or self.n_jobs < 1
        ):
            raise ValueError("n_jobs must be -1 or an integer >= 1")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True")
        probe = DecisionTree(max_features=self.max_features)
        probe._resolved_max_features(n_features)

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
        if np.unique(y_array).size < 2:
            raise ValueError("RandomForestClassifier requires at least two classes")
        return X_array, y_array

    def _validate_prediction_data(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected shape")
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        return X_array

    def _check_is_fitted(self) -> None:
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError("RandomForestClassifier is not fitted")


__all__ = ["RandomForestClassifier"]
