"""Reproducible data, result, and plotting helpers for all experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from src.experiments.config import ExperimentConfig
from src.experiments.datasets import DatasetBundle
from src.unsupervised.kmeans import KMeans
from src.utils.preprocessing import (
    LeakageSafePreprocessor,
    random_oversample_minority,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
_BLUE = "#2457A7"
_ORANGE = "#D97706"
_GOLD = "#C4A000"
_INK = "#20242A"


@dataclass(slots=True)
class PreparedSplit:
    X_train_clean: NDArray[np.float64]
    y_train_clean: NDArray[Any]
    X_train_fit: NDArray[np.float64]
    y_train_fit: NDArray[Any]
    X_test: NDArray[np.float64]
    y_test: NDArray[Any]
    preprocessor: LeakageSafePreprocessor
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]


def prepare_holdout(
    bundle: DatasetBundle,
    config: ExperimentConfig,
) -> PreparedSplit:
    indices = np.arange(bundle.X.shape[0])
    train_indices, test_indices = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=bundle.y,
    )
    preprocessor = LeakageSafePreprocessor(bundle.categorical_columns)
    X_train_clean = preprocessor.fit_transform(bundle.X.iloc[train_indices])
    X_test = preprocessor.transform(bundle.X.iloc[test_indices])
    y_train_clean = np.asarray(bundle.y)[train_indices]
    y_test = np.asarray(bundle.y)[test_indices]
    X_train_fit, y_train_fit = random_oversample_minority(
        X_train_clean,
        y_train_clean,
        min_fraction=config.oversample_min_fraction,
        random_state=config.seed,
    )
    return PreparedSplit(
        X_train_clean=X_train_clean,
        y_train_clean=y_train_clean,
        X_train_fit=np.asarray(X_train_fit, dtype=float),
        y_train_fit=y_train_fit,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
    )


def fit_best_kmeans(
    X: NDArray[np.float64],
    n_clusters: int,
    restarts: int,
    seed: int,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> KMeans:
    best: KMeans | None = None
    seed_sequence = np.random.SeedSequence(seed)
    for restart, child in enumerate(seed_sequence.spawn(restarts), start=1):
        child_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        candidate = KMeans(n_clusters=n_clusters, random_state=child_seed).fit(X)
        if best is None or candidate.inertia_ < best.inertia_:
            best = candidate
        if progress_callback is not None:
            progress_callback(
                restart,
                restarts,
                f"K-Means k={n_clusters}; restart {restart}/{restarts}",
            )
    assert best is not None
    return best


def k_distance_curve(
    X: NDArray[np.float64], min_samples: int
) -> tuple[NDArray[np.float64], float]:
    if not 1 <= min_samples <= X.shape[0]:
        raise ValueError("min_samples must be between 1 and n_samples")
    distances = np.sqrt(
        np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    )
    kth = np.partition(distances, min_samples - 1, axis=1)[:, min_samples - 1]
    sorted_kth = np.sort(kth)
    if np.isclose(sorted_kth[-1], sorted_kth[0]):
        return sorted_kth, max(float(sorted_kth[-1]), 1e-6)
    x = np.linspace(0.0, 1.0, sorted_kth.size)
    y = (sorted_kth - sorted_kth[0]) / (sorted_kth[-1] - sorted_kth[0])
    knee_index = int(np.argmax(x - y))
    epsilon = max(float(sorted_kth[knee_index]), 1e-6)
    return sorted_kth, epsilon


def corrupt_labels_nested(
    y: NDArray[Any],
    levels: tuple[float, ...],
    seed: int,
) -> dict[float, NDArray[Any]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(y.size)
    classes = np.unique(y)
    corrupted: dict[float, NDArray[Any]] = {}
    for level in sorted(levels):
        output = y.copy()
        count = int(round(level * y.size))
        for index in order[:count]:
            alternatives = classes[classes != output[index]]
            output[index] = rng.choice(alternatives)
        corrupted[level] = output
    return corrupted


def save_records(
    records: list[dict[str, Any]], experiment: str, name: str
) -> Path:
    directory = RESULTS_DIR / experiment
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


def save_json(payload: Any, relative_path: str | Path) -> Path:
    path = RESULTS_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return path


def line_plot(
    records: list[dict[str, Any]],
    x: str,
    series: tuple[str, ...],
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    output: Path,
) -> None:
    frame = pd.DataFrame.from_records(records).sort_values(x)
    colors = (_BLUE, _ORANGE, _GOLD)
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for index, column in enumerate(series):
        axis.plot(
            frame[x],
            frame[column],
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=colors[index % len(colors)],
            label=column.replace("_", " ").title(),
        )
    axis.set_title(f"{title}\n{subtitle}", loc="left", color=_INK)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(axis="y", color="#D8DCE2", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=max(1, len(series)), loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def scatter_plot(
    coordinates: NDArray[np.float64],
    labels: NDArray[Any],
    title: str,
    subtitle: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    unique = np.unique(labels)
    palette = plt.get_cmap("tab10")
    for index, label in enumerate(unique):
        mask = labels == label
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=12,
            alpha=0.70,
            color=palette(index % 10),
            label=str(label),
        )
    axis.set_title(f"{title}\n{subtitle}", loc="left", color=_INK)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, markerscale=1.4, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _git_commit() -> str | None:
    """Return the checked-out commit without making Git a runtime requirement."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def write_manifest(
    config: ExperimentConfig,
    profile: str,
    datasets: dict[str, DatasetBundle],
    artifacts: list[Path],
) -> Path:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "sklearn", "matplotlib", "tqdm"):
        module = __import__(package)
        versions[package] = getattr(module, "__version__", "unknown")
    payload = {
        "profile": profile,
        "seed": config.seed,
        "git_commit": _git_commit(),
        "config": asdict(config),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "datasets": {name: bundle.metadata() for name, bundle in datasets.items()},
        "artifacts": [str(path.relative_to(PROJECT_ROOT)) for path in artifacts],
        "determinism": (
            "Numeric outputs are deterministic for the recorded inputs and seed; "
            "runtime metadata may differ."
        ),
    }
    return save_json(payload, "manifest.json")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


__all__ = [
    "FIGURES_DIR",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "PreparedSplit",
    "corrupt_labels_nested",
    "fit_best_kmeans",
    "k_distance_curve",
    "line_plot",
    "prepare_holdout",
    "save_json",
    "save_records",
    "scatter_plot",
    "write_manifest",
]
