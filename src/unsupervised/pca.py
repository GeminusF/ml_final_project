"""Principal Component Analysis implemented with NumPy eigendecomposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class PCA:
    """Project centered data onto the leading covariance eigenvectors."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.explained_variance_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> PCA:
        X_array = self._validate_input(X)
        if X_array.shape[0] < 2:
            raise ValueError("PCA requires at least two samples")
        upper_bound = min(X_array.shape)
        if (
            not isinstance(self.n_components, int)
            or not 1 <= self.n_components <= upper_bound
        ):
            raise ValueError(
                f"n_components must be between 1 and {upper_bound}"
            )
        self.mean_ = X_array.mean(axis=0)
        self.n_features_in_ = X_array.shape[1]
        centered = X_array - self.mean_
        covariance = centered.T @ centered / (X_array.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        total_variance = float(eigenvalues.sum())
        if total_variance == 0.0:
            self.explained_variance_ratio_ = np.zeros(
                self.n_components, dtype=float
            )
        else:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA is not fitted")
        X_array = self._validate_input(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected number of features")
        return (X_array - self.mean_) @ self.components_.T

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        return self.fit(X).transform(X)

    @staticmethod
    def _validate_input(X: ArrayLike) -> NDArray[np.float64]:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")
        return X_array


__all__ = ["PCA"]
