"""Density-based spatial clustering from first principles."""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import ArrayLike, NDArray


_UNVISITED = -2
_NOISE = -1


class DBSCAN:
    """DBSCAN using an inclusive Euclidean epsilon neighbourhood."""

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: NDArray[np.int64] | None = None
        self.core_sample_indices_: NDArray[np.int64] | None = None

    def fit(self, X: ArrayLike) -> DBSCAN:
        X_array = self._validate_input(X)
        self._validate_parameters()
        squared_distances = np.sum(
            (X_array[:, None, :] - X_array[None, :, :]) ** 2, axis=2
        )
        neighbourhoods = [
            np.flatnonzero(row <= self.eps**2).astype(np.int64)
            for row in squared_distances
        ]
        core = np.asarray(
            [indices.size >= self.min_samples for indices in neighbourhoods],
            dtype=bool,
        )
        labels = np.full(X_array.shape[0], _UNVISITED, dtype=np.int64)
        visited = np.zeros(X_array.shape[0], dtype=bool)
        cluster_id = 0

        for point in range(X_array.shape[0]):
            if visited[point]:
                continue
            visited[point] = True
            if not core[point]:
                labels[point] = _NOISE
                continue
            self._expand_cluster(
                point,
                cluster_id,
                labels,
                visited,
                core,
                neighbourhoods,
            )
            cluster_id += 1

        labels[labels == _UNVISITED] = _NOISE
        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core).astype(np.int64)
        return self

    @staticmethod
    def _expand_cluster(
        seed: int,
        cluster_id: int,
        labels: NDArray[np.int64],
        visited: NDArray[np.bool_],
        core: NDArray[np.bool_],
        neighbourhoods: list[NDArray[np.int64]],
    ) -> None:
        labels[seed] = cluster_id
        queue: deque[int] = deque(int(value) for value in neighbourhoods[seed])
        queued = np.zeros(labels.size, dtype=bool)
        queued[neighbourhoods[seed]] = True
        while queue:
            point = queue.popleft()
            if not visited[point]:
                visited[point] = True
                if core[point]:
                    for neighbour in neighbourhoods[point]:
                        neighbour_index = int(neighbour)
                        if not queued[neighbour_index]:
                            queue.append(neighbour_index)
                            queued[neighbour_index] = True
            if labels[point] in {_UNVISITED, _NOISE}:
                labels[point] = cluster_id

    def _validate_parameters(self) -> None:
        if not np.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError("min_samples must be an integer >= 1")

    @staticmethod
    def _validate_input(X: ArrayLike) -> NDArray[np.float64]:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        return X_array


__all__ = ["DBSCAN"]
