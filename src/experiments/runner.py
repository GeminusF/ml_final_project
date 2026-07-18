"""The seven required experiments plus the three computational bonuses."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from src.boosting.gradient_boosting import GradientBoostingClassifier
from src.experiments.config import ExperimentConfig
from src.experiments.datasets import DatasetBundle
from src.experiments.progress import ProgressCallback, ProgressReporter
from src.experiments.utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    corrupt_labels_nested,
    fit_best_kmeans,
    k_distance_curve,
    line_plot,
    prepare_holdout,
    save_json,
    save_records,
    scatter_plot,
)
from src.metrics.evaluation import (
    align_probabilities,
    brier_bias_variance,
    classification_metrics,
)
from src.trees.adaboost import AdaBoostClassifier, DecisionStump
from src.trees.decision_tree import DecisionTree
from src.trees.random_forest import RandomForestClassifier
from src.unsupervised.dbscan import DBSCAN
from src.unsupervised.pca import PCA
from src.utils.preprocessing import (
    LeakageSafePreprocessor,
    random_oversample_minority,
)


@contextmanager
def _progress_phase(
    progress: ProgressReporter | None,
    dataset: str,
    experiment: str,
    units: int,
) -> Iterator[None]:
    """Open a reporting phase while keeping direct experiment calls valid."""

    if progress is None:
        yield
        return
    with progress.phase(dataset, experiment, units):
        yield


@contextmanager
def _progress_task(
    progress: ProgressReporter | None,
    units: int,
    detail: str,
) -> Iterator[ProgressCallback | None]:
    """Yield an optional model callback and reconcile early termination."""

    if progress is None:
        yield None
        return
    with progress.task(units, detail) as callback:
        yield callback


def _fit_with_progress(
    model: Any,
    X: NDArray[np.float64],
    y: NDArray[Any],
    callback: ProgressCallback | None,
) -> Any:
    """Fit custom iterative models with their backward-compatible callback."""

    if isinstance(
        model,
        (AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier),
    ):
        return model.fit(X, y, progress_callback=callback)
    return model.fit(X, y)


def _model_work_units(model: Any) -> int:
    """Return planned fit units for a head-to-head model."""

    return int(getattr(model, "n_estimators", 1))


def _prefixed_callback(
    callback: ProgressCallback | None, prefix: str
) -> ProgressCallback | None:
    """Preserve fold/configuration context in iterative model updates."""

    if callback is None:
        return None

    def wrapped(completed: int, total: int, detail: str) -> None:
        callback(completed, total, f"{prefix} | {detail}")

    return wrapped


def run_dataset_experiments(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    """Run all dataset-level required and bonus experiments."""

    artifacts: list[Path] = []
    with _progress_phase(progress, bundle.name, "Preprocessing", 1):
        with _progress_task(progress, 1, "prepare leakage-safe holdout"):
            split = prepare_holdout(bundle, config)

    phases: list[tuple[str, int, Callable[[], list[Path]]]] = [
        (
            "Baseline",
            3,
            lambda: run_baseline(bundle, split, config, progress),
        ),
        (
            "AdaBoost scaling",
            max(config.ada_estimators),
            lambda: run_adaboost_scaling(bundle, split, config, progress),
        ),
        (
            "Random Forest scaling",
            max(config.rf_estimators)
            + len(config.rf_depths) * config.fixed_estimators,
            lambda: run_rf_scaling(bundle, split, config, progress),
        ),
        (
            "Head-to-head comparison",
            config.cv_folds * (1 + 3 * config.fixed_estimators),
            lambda: run_head_to_head(bundle, config, progress),
        ),
        (
            "Noise robustness",
            config.noise_replicates
            * len(config.noise_levels)
            * 2
            * config.fixed_estimators,
            lambda: run_noise_robustness(bundle, split, config, progress),
        ),
        (
            "Unsupervised analysis",
            len(config.k_values) * config.kmeans_restarts + 5,
            lambda: run_unsupervised(bundle, config, progress),
        ),
        (
            "SAMME.R bonus",
            2 * config.fixed_estimators,
            lambda: run_bonus_samme_r(bundle, split, config, progress),
        ),
    ]
    if np.unique(bundle.y).size == 2:
        phases.append(
            (
                "Gradient Boosting bonus",
                2 * config.fixed_estimators,
                lambda: run_bonus_gradient_boosting(
                    bundle, split, config, progress
                ),
            )
        )
    for name, units, runner in phases:
        with _progress_phase(progress, bundle.name, name, units):
            artifacts.extend(runner())
    return artifacts

def run_baseline(
    bundle: DatasetBundle,
    split: Any,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    models: dict[str, Any] = {
        "custom_tree": DecisionTree(random_state=config.seed),
        "decision_stump": DecisionStump(random_state=config.seed),
        "sklearn_tree_reference": SklearnDecisionTree(random_state=config.seed),
    }
    records: list[dict[str, Any]] = []
    custom_accuracy = float("nan")
    reference_accuracy = float("nan")
    for name, model in models.items():
        with _progress_task(progress, 1, f"fit {name}") as callback:
            _fit_with_progress(
                model, split.X_train_fit, split.y_train_fit, callback
            )
        prediction = model.predict(split.X_test)
        local_probabilities = model.predict_proba(split.X_test)
        probabilities = align_probabilities(
            local_probabilities, model.classes_, np.unique(bundle.y)
        )
        metrics = classification_metrics(
            split.y_test, prediction, probabilities, np.unique(bundle.y)
        )
        records.append(
            {
                "dataset": bundle.name,
                "model": name,
                **metrics,
                "depth": getattr(model, "depth", getattr(model, "get_depth", lambda: 1)()),
            }
        )
        if name == "custom_tree":
            custom_accuracy = metrics["accuracy"]
        if name == "sklearn_tree_reference":
            reference_accuracy = metrics["accuracy"]
    difference = abs(custom_accuracy - reference_accuracy)
    records.append(
        {
            "dataset": bundle.name,
            "model": "custom_reference_check",
            "accuracy": custom_accuracy,
            "f1_macro": float("nan"),
            "auc_roc": float("nan"),
            "depth": float("nan"),
            "absolute_accuracy_difference": difference,
            "within_two_percentage_points": difference <= 0.02,
        }
    )
    return [save_records(records, "01_baseline", bundle.name)]


def run_adaboost_scaling(
    bundle: DatasetBundle,
    split: Any,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    maximum = max(config.ada_estimators)
    model = AdaBoostClassifier(
        n_estimators=maximum,
        random_state=config.seed,
        algorithm="SAMME",
    )
    with _progress_task(
        progress, maximum, f"AdaBoost round 0/{maximum}"
    ) as callback:
        model.fit(
            split.X_train_fit,
            split.y_train_fit,
            progress_callback=callback,
        )
    requested = set(config.ada_estimators)
    train_stages = model.staged_predict(split.X_train_clean)
    test_stages = model.staged_predict(split.X_test)
    records: list[dict[str, Any]] = []
    for stage, (train_prediction, test_prediction) in enumerate(
        zip(train_stages, test_stages, strict=True), start=1
    ):
        if stage not in requested:
            continue
        records.append(
            {
                "dataset": bundle.name,
                "n_estimators": stage,
                "train_error": 1.0
                - float(np.mean(train_prediction == split.y_train_clean)),
                "test_error": 1.0 - float(np.mean(test_prediction == split.y_test)),
            }
        )
    csv_path = save_records(records, "02_adaboost_scaling", bundle.name)
    figure_path = FIGURES_DIR / "02_adaboost_scaling" / f"{bundle.name}.png"
    line_plot(
        records,
        "n_estimators",
        ("train_error", "test_error"),
        "AdaBoost scaling",
        f"{bundle.name}; discrete SAMME decision stumps",
        "Number of estimators",
        "Classification error",
        figure_path,
    )
    return [csv_path, figure_path]


def run_rf_scaling(
    bundle: DatasetBundle,
    split: Any,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    maximum = max(config.rf_estimators)
    forest = RandomForestClassifier(
        n_estimators=maximum,
        max_depth=None,
        oob_score=False,
        random_state=config.seed,
    )
    with _progress_task(
        progress, maximum, f"Forest tree 0/{maximum}; depth unrestricted"
    ) as callback:
        forest.fit(
            split.X_train_fit,
            split.y_train_fit,
            progress_callback=callback,
        )
    global_classes = np.unique(bundle.y)
    cumulative = np.zeros((split.X_test.shape[0], global_classes.size))
    oob_votes = np.zeros((split.X_train_fit.shape[0], global_classes.size))
    oob_counts = np.zeros(split.X_train_fit.shape[0], dtype=int)
    requested = set(config.rf_estimators)
    estimator_records: list[dict[str, Any]] = []
    for index, (tree, oob_indices) in enumerate(
        zip(forest.estimators_, forest.oob_indices_, strict=True), start=1
    ):
        cumulative += align_probabilities(
            tree.predict_proba(split.X_test), tree.classes_, global_classes
        )
        if oob_indices.size:
            oob_votes[oob_indices] += align_probabilities(
                tree.predict_proba(split.X_train_fit[oob_indices]),
                tree.classes_,
                global_classes,
            )
            oob_counts[oob_indices] += 1
        if index not in requested:
            continue
        prediction = global_classes[np.argmax(cumulative, axis=1)]
        valid = oob_counts > 0
        oob_prediction = global_classes[np.argmax(oob_votes[valid], axis=1)]
        estimator_records.append(
            {
                "dataset": bundle.name,
                "n_estimators": index,
                "test_accuracy": float(np.mean(prediction == split.y_test)),
                "oob_accuracy": float(
                    np.mean(oob_prediction == split.y_train_fit[valid])
                ),
                "oob_coverage": float(valid.mean()),
            }
        )
    estimator_csv = save_records(
        estimator_records, "03_rf_scaling", f"{bundle.name}_estimators"
    )
    estimator_figure = (
        FIGURES_DIR / "03_rf_scaling" / f"{bundle.name}_estimators.png"
    )
    line_plot(
        estimator_records,
        "n_estimators",
        ("test_accuracy", "oob_accuracy"),
        "Random Forest estimator scaling",
        f"{bundle.name}; unrestricted tree depth",
        "Number of trees",
        "Accuracy",
        estimator_figure,
    )

    depth_records: list[dict[str, Any]] = []
    for depth_position, depth in enumerate(config.rf_depths, start=1):
        candidate = RandomForestClassifier(
            n_estimators=config.fixed_estimators,
            max_depth=depth,
            oob_score=True,
            random_state=config.seed,
        )
        with _progress_task(
            progress,
            config.fixed_estimators,
            (
                f"Forest depth {depth_position}/{len(config.rf_depths)}; "
                f"max_depth={depth}; tree 0/{config.fixed_estimators}"
            ),
        ) as callback:
            candidate.fit(
                split.X_train_fit,
                split.y_train_fit,
                progress_callback=_prefixed_callback(
                    callback,
                    (
                        f"Forest depth {depth_position}/{len(config.rf_depths)}; "
                        f"max_depth={depth}"
                    ),
                ),
            )
        depth_records.append(
            {
                "dataset": bundle.name,
                "max_depth": depth,
                "test_accuracy": float(
                    np.mean(candidate.predict(split.X_test) == split.y_test)
                ),
                "oob_accuracy": candidate.oob_score_,
                "oob_coverage": candidate.oob_coverage_,
            }
        )
    depth_csv = save_records(
        depth_records, "03_rf_scaling", f"{bundle.name}_depth"
    )
    depth_figure = FIGURES_DIR / "03_rf_scaling" / f"{bundle.name}_depth.png"
    line_plot(
        depth_records,
        "max_depth",
        ("test_accuracy", "oob_accuracy"),
        "Random Forest depth scaling",
        f"{bundle.name}; {config.fixed_estimators} trees",
        "Maximum depth",
        "Accuracy",
        depth_figure,
    )
    return [estimator_csv, estimator_figure, depth_csv, depth_figure]


def run_head_to_head(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    splitter = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.seed
    )
    records: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(bundle.X, bundle.y), start=1
    ):
        processor = LeakageSafePreprocessor(bundle.categorical_columns)
        X_train = processor.fit_transform(bundle.X.iloc[train_indices])
        X_test = processor.transform(bundle.X.iloc[test_indices])
        y_train = np.asarray(bundle.y)[train_indices]
        y_test = np.asarray(bundle.y)[test_indices]
        X_fit, y_fit = random_oversample_minority(
            X_train,
            y_train,
            config.oversample_min_fraction,
            config.seed + fold,
        )
        models: dict[str, Any] = {
            "single_tree": DecisionTree(random_state=config.seed + fold),
            "adaboost": AdaBoostClassifier(
                n_estimators=config.fixed_estimators,
                random_state=config.seed + fold,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=config.fixed_estimators,
                random_state=config.seed + fold,
            ),
            "sklearn_rf_reference": SklearnRandomForest(
                n_estimators=config.fixed_estimators,
                random_state=config.seed + fold,
                max_features="sqrt",
            ),
        }
        global_classes = np.unique(bundle.y)
        for model_name, model in models.items():
            units = _model_work_units(model)
            with _progress_task(
                progress, units, f"Fold {fold}/{config.cv_folds}; {model_name}"
            ) as callback:
                _fit_with_progress(
                    model,
                    X_fit,
                    y_fit,
                    _prefixed_callback(
                        callback,
                        f"Fold {fold}/{config.cv_folds}; {model_name}",
                    ),
                )
            prediction = model.predict(X_test)
            probabilities = align_probabilities(
                model.predict_proba(X_test), model.classes_, global_classes
            )
            records.append(
                {
                    "dataset": bundle.name,
                    "fold": fold,
                    "model": model_name,
                    **classification_metrics(
                        y_test, prediction, probabilities, global_classes
                    ),
                }
            )
    csv_path = save_records(records, "04_head_to_head", bundle.name)
    frame = pd.DataFrame.from_records(records)
    summary = (
        frame.groupby("model")[["accuracy", "f1_macro", "auc_roc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_path = RESULTS_DIR / "04_head_to_head" / f"{bundle.name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    figure_path = FIGURES_DIR / "04_head_to_head" / f"{bundle.name}.png"
    _box_plot(frame, figure_path, f"Head-to-head comparison - {bundle.name}")
    return [csv_path, summary_path, figure_path]


def run_noise_robustness(
    bundle: DatasetBundle,
    split: Any,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    records: list[dict[str, Any]] = []
    for replicate in range(config.noise_replicates):
        corrupted_by_level = corrupt_labels_nested(
            split.y_train_fit,
            config.noise_levels,
            config.seed + 10_000 + replicate,
        )
        for level, corrupted in corrupted_by_level.items():
            models: dict[str, Any] = {
                "AdaBoost": AdaBoostClassifier(
                    n_estimators=config.fixed_estimators,
                    random_state=config.seed + replicate,
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=config.fixed_estimators,
                    random_state=config.seed + replicate,
                ),
            }
            for model_name, model in models.items():
                try:
                    with _progress_task(
                        progress,
                        config.fixed_estimators,
                        (
                            f"Noise {level:.0%}; replicate "
                            f"{replicate + 1}/{config.noise_replicates}; {model_name}"
                        ),
                    ) as callback:
                        _fit_with_progress(
                            model,
                            split.X_train_fit,
                            corrupted,
                            _prefixed_callback(
                                callback,
                                (
                                    f"Noise {level:.0%}; replicate "
                                    f"{replicate + 1}/{config.noise_replicates}; "
                                    f"{model_name}"
                                ),
                            ),
                        )
                    accuracy = float(
                        np.mean(model.predict(split.X_test) == split.y_test)
                    )
                except ValueError:
                    accuracy = float("nan")
                records.append(
                    {
                        "dataset": bundle.name,
                        "replicate": replicate,
                        "noise_fraction": level,
                        "model": model_name,
                        "accuracy": accuracy,
                    }
                )
    csv_path = save_records(records, "05_noise_robustness", bundle.name)
    frame = pd.DataFrame.from_records(records)
    summary = (
        frame.groupby(["noise_fraction", "model"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna(0.0)
    )
    summary_path = RESULTS_DIR / "05_noise_robustness" / f"{bundle.name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    figure_path = FIGURES_DIR / "05_noise_robustness" / f"{bundle.name}.png"
    _noise_plot(summary, figure_path, bundle.name)
    return [csv_path, summary_path, figure_path]


def run_bias_variance(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    split = prepare_holdout(bundle, config)
    classes = np.unique(bundle.y)
    model_factories: dict[str, Callable[[], Any]] = {
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=config.fixed_estimators,
            random_state=config.seed,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=config.fixed_estimators,
            random_state=config.seed,
        ),
    }
    rng = np.random.default_rng(config.seed)
    predictions: dict[str, list[NDArray[np.float64]]] = {
        name: [] for name in model_factories
    }
    for replicate in range(config.bootstrap_replicates):
        indices = rng.integers(
            0,
            split.X_train_fit.shape[0],
            size=split.X_train_fit.shape[0],
        )
        for model_name, factory in model_factories.items():
            model = factory()
            model.random_state = config.seed + replicate
            with _progress_task(
                progress,
                config.fixed_estimators,
                (
                    f"Bootstrap {replicate + 1}/{config.bootstrap_replicates}; "
                    f"{model_name}"
                ),
            ) as callback:
                _fit_with_progress(
                    model,
                    split.X_train_fit[indices],
                    split.y_train_fit[indices],
                    _prefixed_callback(
                        callback,
                        (
                            f"Bootstrap {replicate + 1}/"
                            f"{config.bootstrap_replicates}; {model_name}"
                        ),
                    ),
                )
            predictions[model_name].append(
                align_probabilities(
                    model.predict_proba(split.X_test), model.classes_, classes
                )
            )
    records = []
    archive: dict[str, NDArray[np.float64]] = {}
    for model_name, values in predictions.items():
        stacked = np.stack(values)
        archive[model_name.replace(" ", "_").lower()] = stacked
        records.append(
            {
                "dataset": bundle.name,
                "model": model_name,
                "bootstrap_replicates": config.bootstrap_replicates,
                **brier_bias_variance(stacked, split.y_test, classes),
            }
        )
    result_dir = RESULTS_DIR / "06_bias_variance"
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = result_dir / f"{bundle.name}_probabilities.npz"
    np.savez_compressed(prediction_path, **archive, y_test=split.y_test)
    csv_path = save_records(records, "06_bias_variance", bundle.name)
    figure_path = FIGURES_DIR / "06_bias_variance" / f"{bundle.name}.png"
    _bias_variance_plot(pd.DataFrame.from_records(records), figure_path)
    return [csv_path, prediction_path, figure_path]


def run_unsupervised(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    selected = _stratified_sample_indices(
        bundle.y, min(config.tsne_sample_size, bundle.X.shape[0]), config.seed
    )
    processor = LeakageSafePreprocessor(bundle.categorical_columns)
    if progress is not None:
        progress.set_detail("PCA preprocessing and decomposition")
    X = processor.fit_transform(bundle.X.iloc[selected])
    y = np.asarray(bundle.y)[selected]
    component_count = min(X.shape)
    full_pca = PCA(component_count).fit(X)
    assert full_pca.explained_variance_ratio_ is not None
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    retained = int(np.searchsorted(cumulative, config.pca_variance_target) + 1)
    retained = max(2, min(retained, component_count))
    reduced = PCA(retained).fit_transform(X)
    coordinates = reduced[:, :2]
    if progress is not None:
        progress.advance(2, f"PCA retained {retained} components")

    scree_records = [
        {
            "component": index + 1,
            "cumulative_explained_variance": value,
        }
        for index, value in enumerate(cumulative)
    ]
    scree_csv = save_records(scree_records, "07_unsupervised", f"{bundle.name}_scree")
    scree_figure = FIGURES_DIR / "07_unsupervised" / f"{bundle.name}_scree.png"
    line_plot(
        scree_records,
        "component",
        ("cumulative_explained_variance",),
        "PCA cumulative explained variance",
        f"{bundle.name}; {retained} components reach {config.pca_variance_target:.0%}",
        "Principal component",
        "Cumulative variance ratio",
        scree_figure,
    )

    inertia_records: list[dict[str, Any]] = []
    fitted_kmeans: dict[int, Any] = {}
    for k in config.k_values:
        if k > reduced.shape[0]:
            continue
        with _progress_task(
            progress,
            config.kmeans_restarts,
            f"K-Means k={k}; restart 0/{config.kmeans_restarts}",
        ) as callback:
            fitted = fit_best_kmeans(
                reduced,
                k,
                config.kmeans_restarts,
                config.seed + k,
                progress_callback=callback,
            )
        fitted_kmeans[k] = fitted
        inertia_records.append({"k": k, "inertia": fitted.inertia_})
    best_k = _elbow_k(inertia_records)
    best_kmeans = fitted_kmeans[best_k]
    assert best_kmeans.labels_ is not None
    inertia_csv = save_records(
        inertia_records, "07_unsupervised", f"{bundle.name}_elbow"
    )
    elbow_figure = FIGURES_DIR / "07_unsupervised" / f"{bundle.name}_elbow.png"
    line_plot(
        inertia_records,
        "k",
        ("inertia",),
        "K-Means elbow curve",
        f"{bundle.name}; best of {config.kmeans_restarts} restarts",
        "Number of clusters k",
        "Inertia",
        elbow_figure,
    )

    k_distances, epsilon = k_distance_curve(reduced, config.dbscan_min_samples)
    if progress is not None:
        progress.set_detail("DBSCAN neighborhood expansion")
    dbscan = DBSCAN(epsilon, config.dbscan_min_samples).fit(reduced)
    assert dbscan.labels_ is not None
    if progress is not None:
        progress.advance(1, f"DBSCAN eps={epsilon:.3g}")
    k_distance_records = [
        {"point_rank": index + 1, "k_distance": value}
        for index, value in enumerate(k_distances)
    ]
    kdist_csv = save_records(
        k_distance_records, "07_unsupervised", f"{bundle.name}_k_distance"
    )
    kdist_figure = (
        FIGURES_DIR / "07_unsupervised" / f"{bundle.name}_k_distance.png"
    )
    line_plot(
        k_distance_records,
        "point_rank",
        ("k_distance",),
        "DBSCAN k-distance curve",
        f"{bundle.name}; k={config.dbscan_min_samples}, selected eps={epsilon:.3g}",
        "Sorted point rank",
        "Distance to k-th neighbour",
        kdist_figure,
    )

    metrics = {
        "dataset": bundle.name,
        "analysis_sample_size": int(selected.size),
        "retained_components": retained,
        "variance_target": config.pca_variance_target,
        "best_k": best_k,
        "kmeans_ari": float(adjusted_rand_score(y, best_kmeans.labels_)),
        "dbscan_eps": epsilon,
        "dbscan_ari": float(adjusted_rand_score(y, dbscan.labels_)),
        "dbscan_noise_fraction": float(np.mean(dbscan.labels_ == -1)),
    }
    metrics_path = save_json(metrics, f"07_unsupervised/{bundle.name}_metrics.json")
    true_figure = FIGURES_DIR / "07_unsupervised" / f"{bundle.name}_true.png"
    kmeans_figure = FIGURES_DIR / "07_unsupervised" / f"{bundle.name}_kmeans.png"
    dbscan_figure = FIGURES_DIR / "07_unsupervised" / f"{bundle.name}_dbscan.png"
    scatter_plot(coordinates, y, "PCA embedding - true labels", bundle.name, true_figure)
    scatter_plot(
        coordinates,
        best_kmeans.labels_,
        "PCA embedding - K-Means labels",
        f"{bundle.name}; k={best_k}",
        kmeans_figure,
    )
    scatter_plot(
        coordinates,
        dbscan.labels_,
        "PCA embedding - DBSCAN labels",
        f"{bundle.name}; eps={epsilon:.3g}",
        dbscan_figure,
    )
    if progress is not None:
        progress.advance(1, "PCA, K-Means, and DBSCAN plots saved")

    perplexity = min(30.0, max(5.0, (selected.size - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=config.seed,
    ).fit_transform(X)
    tsne_figure = FIGURES_DIR / "bonus_tsne" / f"{bundle.name}.png"
    scatter_plot(
        tsne,
        y,
        "t-SNE embedding - true labels",
        f"{bundle.name}; n={selected.size}, perplexity={perplexity:.1f}",
        tsne_figure,
    )
    if progress is not None:
        progress.advance(1, "t-SNE embedding and plot saved")
    return [
        scree_csv,
        scree_figure,
        inertia_csv,
        elbow_figure,
        kdist_csv,
        kdist_figure,
        metrics_path,
        true_figure,
        kmeans_figure,
        dbscan_figure,
        tsne_figure,
    ]


def run_bonus_samme_r(
    bundle: DatasetBundle,
    split: Any,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    records: list[dict[str, Any]] = []
    classes = np.unique(bundle.y)
    for algorithm in ("SAMME", "SAMME.R"):
        try:
            model = AdaBoostClassifier(
                n_estimators=config.fixed_estimators,
                algorithm=algorithm,
                random_state=config.seed,
            )
            with _progress_task(
                progress,
                config.fixed_estimators,
                f"{algorithm} round 0/{config.fixed_estimators}",
            ) as callback:
                model.fit(
                    split.X_train_fit,
                    split.y_train_fit,
                    progress_callback=callback,
                )
            prediction = model.predict(split.X_test)
            metrics = classification_metrics(
                split.y_test,
                prediction,
                model.predict_proba(split.X_test),
                classes,
            )
        except ValueError:
            metrics = {"accuracy": float("nan"), "f1_macro": float("nan"), "auc_roc": float("nan")}
        records.append({"dataset": bundle.name, "algorithm": algorithm, **metrics})
    return [save_records(records, "bonus_samme_r", bundle.name)]


def run_bonus_gradient_boosting(
    bundle: DatasetBundle,
    split: Any,
    config: ExperimentConfig,
    progress: ProgressReporter | None = None,
) -> list[Path]:
    records: list[dict[str, Any]] = []
    models = {
        "AdaBoost": AdaBoostClassifier(
            n_estimators=config.fixed_estimators, random_state=config.seed
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=config.fixed_estimators, random_state=config.seed
        ),
    }
    classes = np.unique(bundle.y)
    for name, model in models.items():
        with _progress_task(
            progress,
            config.fixed_estimators,
            f"{name} round 0/{config.fixed_estimators}",
        ) as callback:
            _fit_with_progress(
                model, split.X_train_fit, split.y_train_fit, callback
            )
        prediction = model.predict(split.X_test)
        records.append(
            {
                "dataset": bundle.name,
                "model": name,
                **classification_metrics(
                    split.y_test,
                    prediction,
                    model.predict_proba(split.X_test),
                    classes,
                ),
            }
        )
    return [save_records(records, "bonus_gradient_boosting", bundle.name)]


def _stratified_sample_indices(
    y: NDArray[Any], size: int, seed: int
) -> NDArray[np.int64]:
    if size >= y.size:
        return np.arange(y.size, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=max(2, y.size // size), shuffle=True, random_state=seed)
    _, selected = next(splitter.split(np.zeros(y.size), y))
    if selected.size > size:
        rng = np.random.default_rng(seed)
        selected = rng.choice(selected, size=size, replace=False)
    return np.asarray(selected, dtype=np.int64)


def _elbow_k(records: list[dict[str, Any]]) -> int:
    if len(records) <= 2:
        return int(records[-1]["k"])
    x = np.asarray([record["k"] for record in records], dtype=float)
    y = np.asarray([record["inertia"] for record in records], dtype=float)
    x_scaled = (x - x[0]) / (x[-1] - x[0])
    y_scaled = (y - y[-1]) / (y[0] - y[-1] + 1e-12)
    distance = (1.0 - x_scaled) - y_scaled
    return int(x[np.argmax(np.abs(distance))])


def _box_plot(frame: pd.DataFrame, output: Path, title: str) -> None:
    models = frame["model"].unique().tolist()
    values = [frame.loc[frame["model"] == model, "accuracy"] for model in models]
    fig, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    axis.boxplot(values, tick_labels=models, patch_artist=True)
    axis.set_title(f"{title}\nFive-fold accuracy distributions", loc="left")
    axis.set_ylabel("Accuracy")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", color="#D8DCE2", linewidth=0.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _noise_plot(frame: pd.DataFrame, output: Path, dataset: str) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    colors = {"AdaBoost": "#2457A7", "Random Forest": "#D97706"}
    for model, group in frame.groupby("model"):
        axis.errorbar(
            group["noise_fraction"],
            group["mean"],
            yerr=group["std"],
            marker="o",
            capsize=3,
            color=colors[str(model)],
            label=str(model),
        )
    axis.set_title(
        f"Noise robustness\n{dataset}; mean accuracy with replicate variability",
        loc="left",
    )
    axis.set_xlabel("Training-label noise fraction")
    axis.set_ylabel("Clean-test accuracy")
    axis.grid(axis="y", color="#D8DCE2", linewidth=0.7)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _bias_variance_plot(frame: pd.DataFrame, output: Path) -> None:
    models = frame["model"].tolist()
    positions = np.arange(len(models))
    width = 0.34
    fig, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    axis.bar(positions - width / 2, frame["bias_squared"], width, label="Bias squared", color="#2457A7")
    axis.bar(positions + width / 2, frame["variance"], width, label="Variance", color="#D97706")
    axis.set_xticks(positions, models)
    axis.set_ylabel("Mean Brier component")
    axis.set_title("Classification bias-variance decomposition\nShared bootstrap replicates and fixed clean test set", loc="left")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#D8DCE2", linewidth=0.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


__all__ = ["run_bias_variance", "run_dataset_experiments"]
