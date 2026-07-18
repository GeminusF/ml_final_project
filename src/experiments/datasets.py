"""Versioned dataset loading and compliance checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import fetch_covtype, fetch_openml, load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit

from src.metrics.evaluation import class_distribution


BREAST_CANCER_SOURCE = (
    "https://scikit-learn.org/stable/modules/generated/"
    "sklearn.datasets.load_breast_cancer.html"
)
ADULT_SOURCE = "https://archive.ics.uci.edu/dataset/2/adult"
COVERTYPE_SOURCE = "https://archive.ics.uci.edu/dataset/31/covertype"


@dataclass(slots=True)
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: NDArray[Any]
    categorical_columns: list[str]
    source: str
    version: str
    description: str
    local_hashes: dict[str, str]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "version": self.version,
            "description": self.description,
            "shape": [int(self.X.shape[0]), int(self.X.shape[1])],
            "categorical_columns": self.categorical_columns,
            "missing_values": int(self.X.isna().sum().sum()),
            "class_distribution": class_distribution(self.y),
            "local_hashes": self.local_hashes,
        }


def load_project_datasets(
    data_dir: str | Path = "data",
    covertype_size: int = 10_000,
    random_state: int = 42,
    names: tuple[str, ...] = ("breast_cancer", "adult", "covertype"),
) -> dict[str, DatasetBundle]:
    """Load the three locked datasets, using local files before downloads."""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    loaders = {
        "breast_cancer": lambda: load_breast_cancer_bundle(),
        "adult": lambda: load_adult_bundle(data_path),
        "covertype": lambda: load_covertype_bundle(
            data_path, covertype_size, random_state
        ),
    }
    unknown = set(names) - set(loaders)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    bundles = {name: loaders[name]() for name in names}
    verify_dataset_requirements(bundles)
    return bundles


def load_breast_cancer_bundle() -> DatasetBundle:
    dataset = load_breast_cancer(as_frame=True)
    assert dataset.frame is not None
    X = dataset.frame.drop(columns=["target"])
    y = dataset.frame["target"].to_numpy()
    return DatasetBundle(
        name="breast_cancer",
        X=X,
        y=y,
        categorical_columns=[],
        source=BREAST_CANCER_SOURCE,
        version="scikit-learn bundled Wisconsin Diagnostic Breast Cancer",
        description="Balanced binary medical diagnosis dataset with 30 features.",
        local_hashes={},
    )


def load_adult_bundle(data_dir: Path) -> DatasetBundle:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    categorical = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    local_files = [data_dir / "adult.data", data_dir / "adult.test"]
    hashes: dict[str, str] = {}
    if local_files[0].exists():
        frames = [
            pd.read_csv(
                local_files[0], names=columns, skipinitialspace=True, na_values="?"
            )
        ]
        hashes[local_files[0].name] = _sha256(local_files[0])
        if local_files[1].exists():
            frames.append(
                pd.read_csv(
                    local_files[1],
                    names=columns,
                    skipinitialspace=True,
                    na_values="?",
                    comment="|",
                )
            )
            hashes[local_files[1].name] = _sha256(local_files[1])
        frame = pd.concat(frames, ignore_index=True)
    else:
        dataset = fetch_openml(
            "adult",
            version=2,
            as_frame=True,
            parser="auto",
            data_home=str(data_dir / "sklearn_cache"),
        )
        assert dataset.frame is not None
        frame = dataset.frame.copy()
        target_name = dataset.target.name if dataset.target is not None else "class"
        if target_name not in frame:
            frame[target_name] = dataset.target
        rename = {
            "education-num": "education_num",
            "marital-status": "marital_status",
            "capital-gain": "capital_gain",
            "capital-loss": "capital_loss",
            "hours-per-week": "hours_per_week",
            "native-country": "native_country",
            target_name: "income",
        }
        frame = frame.rename(columns=rename)
        for cache_file in sorted((data_dir / "sklearn_cache").rglob("adult.arff.gz")):
            hashes[str(cache_file.relative_to(data_dir))] = _sha256(cache_file)
    frame["income"] = (
        frame["income"].astype(str).str.strip().str.rstrip(".")
    )
    X = frame.drop(columns=["income"])
    y = frame["income"].to_numpy()
    for column in categorical:
        X[column] = X[column].astype("string").replace("?", pd.NA)
    return DatasetBundle(
        name="adult",
        X=X,
        y=y,
        categorical_columns=categorical,
        source=ADULT_SOURCE,
        version="UCI Adult / Census Income, OpenML version 2 fallback",
        description="Binary income task with categorical data and missing values.",
        local_hashes=hashes,
    )


def load_covertype_bundle(
    data_dir: Path, requested_size: int, random_state: int
) -> DatasetBundle:
    local_candidates = [data_dir / "covtype.data", data_dir / "covtype.data.gz"]
    local_path = next((path for path in local_candidates if path.exists()), None)
    hashes: dict[str, str] = {}
    if local_path is not None:
        frame = pd.read_csv(local_path, header=None)
        X_array = frame.iloc[:, :-1].to_numpy(dtype=float)
        y = frame.iloc[:, -1].to_numpy()
        hashes[local_path.name] = _sha256(local_path)
    else:
        X_array, y = fetch_covtype(
            data_home=str(data_dir / "sklearn_cache"),
            return_X_y=True,
            download_if_missing=True,
        )
        cache_dir = data_dir / "sklearn_cache" / "covertype"
        for cache_name in ("samples_py3", "targets_py3"):
            cache_file = cache_dir / cache_name
            if cache_file.exists():
                hashes[str(cache_file.relative_to(data_dir))] = _sha256(cache_file)
    feature_names = [f"feature_{index + 1}" for index in range(X_array.shape[1])]
    X = pd.DataFrame(X_array, columns=feature_names)

    size = min(requested_size, X.shape[0])
    while True:
        if size >= X.shape[0]:
            selected = np.arange(X.shape[0])
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=size, random_state=random_state
            )
            selected, _ = next(splitter.split(X, y))
        subset_y = np.asarray(y)[selected]
        minority_fraction = class_distribution(subset_y)["minority_fraction"]
        if minority_fraction <= 0.01 or size >= X.shape[0]:
            break
        size = min(X.shape[0], size * 2)
    if minority_fraction > 0.01:
        raise ValueError(
            "The selected Covertype version does not satisfy minority <= 1%"
        )
    return DatasetBundle(
        name="covertype",
        X=X.iloc[selected].reset_index(drop=True),
        y=subset_y,
        categorical_columns=[],
        source=COVERTYPE_SOURCE,
        version=f"UCI Covertype deterministic stratified subset, n={len(selected)}",
        description="Seven-class, 54-feature forest cover type dataset.",
        local_hashes=hashes,
    )


def verify_dataset_requirements(bundles: dict[str, DatasetBundle]) -> None:
    if len(bundles) < 1:
        raise ValueError("At least one dataset is required")
    metadata = [bundle.metadata() for bundle in bundles.values()]
    if len(bundles) >= 3:
        if not any(item["shape"][1] > 20 for item in metadata):
            raise AssertionError("At least one dataset must have more than 20 features")
        if not any(
            item["class_distribution"]["minority_fraction"] <= 0.01
            for item in metadata
        ):
            raise AssertionError("At least one dataset must have minority <= 1%")
        class_counts = [len(item["class_distribution"]["classes"]) for item in metadata]
        if not any(count == 2 for count in class_counts):
            raise AssertionError("At least one binary dataset is required")
        if not any(count > 2 for count in class_counts):
            raise AssertionError("At least one multiclass dataset is required")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DatasetBundle",
    "load_adult_bundle",
    "load_breast_cancer_bundle",
    "load_covertype_bundle",
    "load_project_datasets",
    "verify_dataset_requirements",
]
