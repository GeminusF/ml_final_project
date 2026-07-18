"""One-command reproduction for every required experiment and bonus."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

# The PDF requires direct execution as ``python src/experiments/run_all.py``.
# Direct script execution exposes only this directory on sys.path, so add the
# repository root before importing the ``src`` package.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLBACKEND", "Agg")

from src.experiments.config import ExperimentConfig
from src.experiments.datasets import load_project_datasets
from src.experiments.progress import ProgressReporter, planned_suite_units
from src.experiments.runner import run_bias_variance, run_dataset_experiments
from src.experiments.utils import PROJECT_ROOT, save_json, write_manifest


def run(
    profile: str = "full",
    datasets: tuple[str, ...] | None = None,
    *,
    show_progress: bool = False,
    use_color: bool = True,
) -> list[Path]:
    """Run the selected reproducibility profile and return all artifact paths."""

    config = ExperimentConfig.quick() if profile == "quick" else ExperimentConfig()
    requested = datasets or (
        ("breast_cancer",)
        if profile == "quick"
        else ("breast_cancer", "adult", "covertype")
    )
    progress = ProgressReporter(
        planned_suite_units(config, requested),
        enabled=show_progress,
        use_color=use_color,
    )
    artifacts: list[Path] = []
    started = perf_counter()
    status = "failed"
    try:
        with progress.phase("all", "Dataset loading", len(requested)):
            with progress.task(len(requested), "load and verify datasets"):
                bundles = load_project_datasets(
                    PROJECT_ROOT / "data",
                    covertype_size=config.covertype_size,
                    random_state=config.seed,
                    names=requested,
                )
        for bundle in bundles.values():
            artifacts.extend(run_dataset_experiments(bundle, config, progress))
        if "breast_cancer" in bundles:
            bias_units = (
                config.bootstrap_replicates * 2 * config.fixed_estimators
            )
            with progress.phase(
                "breast_cancer", "Bias-variance decomposition", bias_units
            ):
                artifacts.extend(
                    run_bias_variance(
                        bundles["breast_cancer"], config, progress
                    )
                )
        elapsed = perf_counter() - started
        runtime_path = save_json(
            {
                "profile": profile,
                "elapsed_seconds": elapsed,
                "datasets": list(bundles),
                "artifact_count": len(artifacts),
                "phases": progress.timing_records,
            },
            "runtime.json",
        )
        artifacts.append(runtime_path)
        artifacts.append(write_manifest(config, profile, bundles, artifacts))
        status = "completed"
        return artifacts
    finally:
        progress.close(status)


def main() -> None:
    """Parse command-line arguments and run the configured experiment suite."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "quick"), default="full")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=("breast_cancer", "adult", "covertype"),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress and ETA output.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Keep slash bars but disable ANSI colors.",
    )
    arguments = parser.parse_args()
    artifacts = run(
        profile=arguments.profile,
        datasets=tuple(arguments.datasets) if arguments.datasets else None,
        show_progress=not arguments.no_progress,
        use_color=not arguments.no_color,
    )
    print(f"Generated {len(artifacts)} artifacts under results/ and figures/")


if __name__ == "__main__":
    main()
