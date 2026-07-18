"""Leakage-safe preprocessing and deterministic minority oversampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


class LeakageSafePreprocessor:
    """Median/most-frequent imputation, one-hot encoding, and scaling.

    The object deliberately mirrors the fit/transform discipline of an sklearn
    pipeline while keeping the learned statistics visible for oral defence.
    """

    def __init__(self, categorical_columns: list[str] | None = None) -> None:
        self.categorical_columns = categorical_columns or []
        self.feature_names_in_: list[str] | None = None
        self.feature_names_out_: list[str] | None = None
        self.numeric_columns_: list[str] | None = None
        self.numeric_medians_: dict[str, float] | None = None
        self.numeric_means_: dict[str, float] | None = None
        self.numeric_scales_: dict[str, float] | None = None
        self.category_fill_: dict[str, str] | None = None
        self.categories_: dict[str, list[str]] | None = None

    def fit(self, X: pd.DataFrame | ArrayLike) -> LeakageSafePreprocessor:
        frame = self._to_frame(X)
        missing = set(self.categorical_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Unknown categorical columns: {sorted(missing)}")
        self.feature_names_in_ = [str(column) for column in frame.columns]
        self.numeric_columns_ = [
            column
            for column in self.feature_names_in_
            if column not in self.categorical_columns
        ]
        self.numeric_medians_ = {}
        self.numeric_means_ = {}
        self.numeric_scales_ = {}
        for column in self.numeric_columns_:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            median = float(numeric.median())
            if not np.isfinite(median):
                median = 0.0
            filled = numeric.fillna(median).to_numpy(dtype=float)
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            self.numeric_medians_[column] = median
            self.numeric_means_[column] = mean
            self.numeric_scales_[column] = scale if scale > 0.0 else 1.0

        self.category_fill_ = {}
        self.categories_ = {}
        for column in self.categorical_columns:
            values = frame[column].astype("string").replace("?", pd.NA)
            modes = values.dropna().mode()
            fill = str(modes.iloc[0]) if not modes.empty else "<missing>"
            filled = values.fillna(fill).astype(str)
            self.category_fill_[column] = fill
            self.categories_[column] = sorted(filled.unique().tolist())

        self.feature_names_out_ = list(self.numeric_columns_)
        for column in self.categorical_columns:
            self.feature_names_out_.extend(
                f"{column}={category}" for category in self.categories_[column]
            )
        return self

    def transform(self, X: pd.DataFrame | ArrayLike) -> NDArray[np.float64]:
        self._check_is_fitted()
        frame = self._to_frame(X)
        assert self.feature_names_in_ is not None
        if [str(column) for column in frame.columns] != self.feature_names_in_:
            raise ValueError("Input columns do not match the fitted schema")
        assert self.numeric_columns_ is not None
        assert self.numeric_medians_ is not None
        assert self.numeric_means_ is not None
        assert self.numeric_scales_ is not None
        assert self.category_fill_ is not None
        assert self.categories_ is not None
        blocks: list[NDArray[np.float64]] = []
        if self.numeric_columns_:
            numeric_output = np.empty(
                (frame.shape[0], len(self.numeric_columns_)), dtype=float
            )
            for output_index, column in enumerate(self.numeric_columns_):
                numeric = pd.to_numeric(frame[column], errors="coerce")
                filled = numeric.fillna(self.numeric_medians_[column]).to_numpy(
                    dtype=float
                )
                numeric_output[:, output_index] = (
                    filled - self.numeric_means_[column]
                ) / self.numeric_scales_[column]
            blocks.append(numeric_output)
        for column in self.categorical_columns:
            values = (
                frame[column]
                .astype("string")
                .replace("?", pd.NA)
                .fillna(self.category_fill_[column])
                .astype(str)
                .to_numpy()
            )
            category_to_index = {
                category: index
                for index, category in enumerate(self.categories_[column])
            }
            encoded = np.zeros(
                (frame.shape[0], len(category_to_index)), dtype=float
            )
            for row_index, value in enumerate(values):
                category_index = category_to_index.get(value)
                if category_index is not None:
                    encoded[row_index, category_index] = 1.0
            blocks.append(encoded)
        return np.hstack(blocks) if blocks else np.empty((frame.shape[0], 0))

    def fit_transform(self, X: pd.DataFrame | ArrayLike) -> NDArray[np.float64]:
        return self.fit(X).transform(X)

    @staticmethod
    def _to_frame(X: pd.DataFrame | ArrayLike) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("X must be non-empty")
            return X.copy()
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        return pd.DataFrame(array, columns=[f"x{index}" for index in range(array.shape[1])])

    def _check_is_fitted(self) -> None:
        if self.feature_names_out_ is None:
            raise RuntimeError("LeakageSafePreprocessor is not fitted")


def random_oversample_minority(
    X: ArrayLike,
    y: ArrayLike,
    min_fraction: float = 0.05,
    random_state: int | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Raise under-represented classes to a fraction of the original fold.

    Majority classes are never downsampled.  The function must be called only
    on a training split or training fold.
    """

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim != 2 or y_array.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X_array.shape[0] != y_array.shape[0] or X_array.shape[0] == 0:
        raise ValueError("X and y must contain the same non-zero row count")
    if not np.isfinite(min_fraction) or not 0.0 < min_fraction <= 1.0:
        raise ValueError("min_fraction must lie in (0, 1]")
    target = int(np.ceil(min_fraction * X_array.shape[0]))
    classes, counts = np.unique(y_array, return_counts=True)
    rng = np.random.default_rng(random_state)
    selected = [np.arange(X_array.shape[0], dtype=np.int64)]
    for class_label, count in zip(classes, counts, strict=True):
        if count >= target:
            continue
        candidates = np.flatnonzero(y_array == class_label)
        selected.append(
            rng.choice(candidates, size=target - count, replace=True).astype(np.int64)
        )
    combined = np.concatenate(selected)
    rng.shuffle(combined)
    return X_array[combined], y_array[combined]


__all__ = ["LeakageSafePreprocessor", "random_oversample_minority"]
