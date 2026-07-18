"""End-to-end smoke test for the configuration-driven experiment suite."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.experiments import runner as experiment_runner
from src.experiments import utils as experiment_utils
from src.experiments.run_all import run


def test_quick_profile_generates_all_experiment_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Smoke validation must never overwrite the submission's full-run evidence.
    output_root = (
        Path(__file__).resolve().parents[1]
        / "tmp"
        / "test_runs"
        / f"experiment-smoke-{uuid4().hex}"
    )
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    monkeypatch.setattr(experiment_utils, "PROJECT_ROOT", output_root)
    monkeypatch.setattr(experiment_utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(experiment_utils, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(experiment_runner, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(experiment_runner, "FIGURES_DIR", figures_dir)

    artifacts = run(profile="quick", datasets=("breast_cancer",))
    relative = {path.as_posix() for path in artifacts}
    expected_fragments = {
        "01_baseline",
        "02_adaboost_scaling",
        "03_rf_scaling",
        "04_head_to_head",
        "05_noise_robustness",
        "06_bias_variance",
        "07_unsupervised",
        "bonus_samme_r",
        "bonus_gradient_boosting",
        "bonus_tsne",
    }
    for fragment in expected_fragments:
        assert any(fragment in path for path in relative), fragment
    assert all(Path(path).exists() for path in artifacts)

    baseline_path = next(
        path
        for path in artifacts
        if "01_baseline" in path.as_posix() and path.suffix == ".csv"
    )
    baseline = pd.read_csv(baseline_path)
    assert {"dataset", "model", "accuracy", "f1_macro", "auc_roc"} <= set(
        baseline.columns
    )

    manifest_path = next(path for path in artifacts if path.name == "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["profile"] == "quick"
    assert "tqdm" in manifest["package_versions"]

    runtime_path = next(path for path in artifacts if path.name == "runtime.json")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["profile"] == "quick"
    assert runtime["phases"]
    assert all(phase["status"] == "completed" for phase in runtime["phases"])
    assert all(phase["elapsed_seconds"] >= 0.0 for phase in runtime["phases"])
