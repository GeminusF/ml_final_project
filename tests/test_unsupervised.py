"""Tests for PCA, K-Means, and DBSCAN."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from src.unsupervised.dbscan import DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA


def test_pca_centering_orthonormality_and_reference_variance() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 4)) @ np.diag([4.0, 2.0, 1.0, 0.2]) + 10.0
    model = PCA(3).fit(X)
    reference = SklearnPCA(3).fit(X)

    assert model.mean_ is not None and model.components_ is not None
    np.testing.assert_allclose(model.mean_, X.mean(axis=0))
    np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(
        model.explained_variance_ratio_, reference.explained_variance_ratio_
    )
    transformed = model.transform(X)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)


def test_pca_constant_features_have_zero_explained_variance() -> None:
    model = PCA(2).fit(np.ones((5, 3)))
    np.testing.assert_allclose(model.explained_variance_ratio_, 0.0)
    np.testing.assert_allclose(model.transform(np.ones((2, 3))), 0.0)


def test_kmeans_obvious_clusters_inertia_and_determinism() -> None:
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
    first = KMeans(2, random_state=42).fit(X)
    second = KMeans(2, random_state=42).fit(X)

    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_allclose(first.centroids_, second.centroids_)
    assert first.inertia_ == pytest.approx(1.0)
    np.testing.assert_array_equal(first.predict(X), first.labels_)


@pytest.mark.parametrize("k", [1, 5])
def test_kmeans_k_one_and_k_n(k: int) -> None:
    X = np.arange(10, dtype=float).reshape(5, 2)
    model = KMeans(k, random_state=1).fit(X)
    assert model.labels_ is not None
    assert np.unique(model.labels_).size == k
    if k == 5:
        assert model.inertia_ == pytest.approx(0.0)


def test_kmeans_duplicates_and_empty_cluster_reseeding_are_finite() -> None:
    X = np.array([[0, 0], [0, 0], [0, 0], [5, 5], [10, 10]], dtype=float)
    model = KMeans(4, max_iter=20, random_state=4).fit(X)
    assert model.centroids_ is not None and np.isfinite(model.centroids_).all()
    assert model.inertia_ is not None and model.inertia_ >= 0.0


def test_dbscan_core_border_noise_and_boundary() -> None:
    X = np.array([[0.0], [0.1], [0.2], [0.3], [1.0]])
    model = DBSCAN(eps=0.1, min_samples=3).fit(X)
    assert model.labels_ is not None and model.core_sample_indices_ is not None
    assert model.labels_[0] == model.labels_[1] == model.labels_[2] == model.labels_[3]
    assert model.labels_[4] == -1
    assert 0 not in model.core_sample_indices_
    assert 1 in model.core_sample_indices_


def test_dbscan_all_noise_all_cluster_duplicates_and_noise_reassignment() -> None:
    spaced = np.array([[0.0], [5.0], [10.0]])
    assert np.all(DBSCAN(0.1, 2).fit(spaced).labels_ == -1)
    assert np.all(DBSCAN(20.0, 1).fit(spaced).labels_ == 0)
    duplicated = np.array([[0.0], [0.0], [0.05]])
    assert np.all(DBSCAN(0.05, 2).fit(duplicated).labels_ == 0)
