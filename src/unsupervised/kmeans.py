"""K-Means clustering with deterministic k-means++ initialisation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class KMeans:
    """Lloyd's algorithm for Euclidean K-Means clustering."""

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: NDArray[np.float64] | None = None
        self.labels_: NDArray[np.int64] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def fit(self, X: ArrayLike) -> KMeans:
        X_array = self._validate_input(X)
        self._validate_parameters(X_array.shape[0])
        rng = np.random.default_rng(self.random_state)
        centroids = self._initialise_centroids(X_array, rng)
        previous_labels: NDArray[np.int64] | None = None

        for iteration in range(1, self.max_iter + 1):
            distances = self._squared_distances(X_array, centroids)
            labels = np.argmin(distances, axis=1).astype(np.int64)
            new_centroids = self._updated_centroids(
                X_array, labels, distances, centroids
            )
            shift = float(np.linalg.norm(new_centroids - centroids))
            converged = (
                previous_labels is not None
                and np.array_equal(labels, previous_labels)
            ) or shift <= self.tol
            centroids = new_centroids
            previous_labels = labels
            if converged:
                break

        final_distances = self._squared_distances(X_array, centroids)
        final_labels = np.argmin(final_distances, axis=1).astype(np.int64)
        centroids = self._updated_centroids(
            X_array, final_labels, final_distances, centroids
        )
        final_distances = self._squared_distances(X_array, centroids)
        final_labels = np.argmin(final_distances, axis=1).astype(np.int64)

        self.centroids_ = centroids
        self.labels_ = final_labels
        self.inertia_ = float(
            final_distances[np.arange(X_array.shape[0]), final_labels].sum()
        )
        self.n_iter_ = iteration
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        if self.centroids_ is None:
            raise RuntimeError("KMeans is not fitted")
        X_array = self._validate_input(X)
        if X_array.shape[1] != self.centroids_.shape[1]:
            raise ValueError("X has an unexpected number of features")
        return np.argmin(
            self._squared_distances(X_array, self.centroids_), axis=1
        ).astype(np.int64)

    def _initialise_centroids(
        self, X: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first_index = int(rng.integers(0, X.shape[0]))
        centroids[0] = X[first_index]
        closest = np.sum((X - centroids[0]) ** 2, axis=1)
        for cluster_index in range(1, self.n_clusters):
            total = float(closest.sum())
            if total <= 0.0:
                chosen_index = next(
                    (
                        index
                        for index in range(X.shape[0])
                        if not np.any(
                            np.all(
                                np.isclose(centroids[:cluster_index], X[index]),
                                axis=1,
                            )
                        )
                    ),
                    0,
                )
            else:
                chosen_index = int(rng.choice(X.shape[0], p=closest / total))
            centroids[cluster_index] = X[chosen_index]
            distance = np.sum((X - centroids[cluster_index]) ** 2, axis=1)
            closest = np.minimum(closest, distance)
        return centroids

    def _updated_centroids(
        self,
        X: NDArray[np.float64],
        labels: NDArray[np.int64],
        distances: NDArray[np.float64],
        previous: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        centroids = np.empty_like(previous)
        point_errors = distances[np.arange(X.shape[0]), labels]
        available = np.ones(X.shape[0], dtype=bool)
        for cluster_index in range(self.n_clusters):
            members = labels == cluster_index
            if members.any():
                centroids[cluster_index] = X[members].mean(axis=0)
                continue
            candidate_errors = np.where(available, point_errors, -np.inf)
            farthest = int(np.argmax(candidate_errors))
            centroids[cluster_index] = X[farthest]
            available[farthest] = False
        return centroids

    @staticmethod
    def _squared_distances(
        X: NDArray[np.float64], centroids: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    def _validate_parameters(self, n_samples: int) -> None:
        if (
            not isinstance(self.n_clusters, int)
            or not 1 <= self.n_clusters <= n_samples
        ):
            raise ValueError("n_clusters must be between 1 and n_samples")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1")
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("tol must be non-negative and finite")

    @staticmethod
    def _validate_input(X: ArrayLike) -> NDArray[np.float64]:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        return X_array


__all__ = ["KMeans"]
