"""Named experiment constants; no experimental value is hidden in code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int = 42
    test_size: float = 0.20
    covertype_size: int = 10_000
    oversample_min_fraction: float = 0.05
    ada_estimators: tuple[int, ...] = (1, *range(5, 201, 5))
    rf_estimators: tuple[int, ...] = (1, *range(10, 201, 10))
    rf_depths: tuple[int, ...] = tuple(range(1, 21))
    fixed_estimators: int = 100
    cv_folds: int = 5
    noise_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    noise_replicates: int = 5
    bootstrap_replicates: int = 100
    pca_variance_target: float = 0.90
    k_values: tuple[int, ...] = tuple(range(1, 11))
    kmeans_restarts: int = 10
    dbscan_min_samples: int = 5
    tsne_sample_size: int = 2_000

    @classmethod
    def quick(cls, seed: int = 42) -> ExperimentConfig:
        """Fast architecture smoke profile; never used for final claims."""

        return cls(
            seed=seed,
            covertype_size=1_000,
            ada_estimators=(1, 5, 10),
            rf_estimators=(1, 5, 10),
            rf_depths=(1, 3, 5),
            fixed_estimators=5,
            cv_folds=2,
            noise_levels=(0.0, 0.10),
            noise_replicates=1,
            bootstrap_replicates=3,
            k_values=(1, 2, 3),
            kmeans_restarts=2,
            tsne_sample_size=250,
        )


__all__ = ["ExperimentConfig"]
