"""Tests for terminal progress rendering and model progress callbacks."""

from __future__ import annotations

from io import StringIO
from time import sleep

import numpy as np
import pytest

from src.boosting.gradient_boosting import GradientBoostingClassifier
from src.experiments.config import ExperimentConfig
from src.experiments.progress import (
    _SPINNER,
    ProgressReporter,
    planned_suite_units,
)
from src.trees.adaboost import AdaBoostClassifier
from src.trees.random_forest import RandomForestClassifier


def test_green_slash_bar_contains_required_interactive_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = StringIO()
    reporter = ProgressReporter(
        4,
        enabled=True,
        use_color=True,
        stream=stream,
        dynamic=True,
        refresh_interval=60.0,
    )
    with reporter.phase("adult", "Random Forest scaling", 4):
        with reporter.task(4, "Forest tree 0/4") as callback:
            for completed in range(1, 5):
                callback(completed, 4, f"Forest tree {completed}/4")
    reporter.close("completed")

    output = stream.getvalue()
    assert "\x1b[32m" in output
    assert "/" * 4 in output
    assert "Overall" in output and "Current" in output
    assert "adult / Random Forest scaling" in output
    assert "elapsed" in output and "ETA" in output and "finish" in output
    assert "100%" in output
    assert _SPINNER == ("|", "/", "-", "\\")


def test_heartbeat_cycles_activity_indicator() -> None:
    stream = StringIO()
    reporter = ProgressReporter(
        1,
        enabled=True,
        use_color=False,
        stream=stream,
        dynamic=True,
        refresh_interval=0.01,
    )
    with reporter.phase("data", "long operation", 1):
        reporter.set_detail("indivisible work")
        sleep(0.08)
        reporter.advance(1, "done")
    reporter.close("completed")

    output = stream.getvalue()
    for symbol in _SPINNER:
        assert f"{symbol} data / long operation" in output


def test_no_color_environment_removes_ansi_but_keeps_slashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    stream = StringIO()
    reporter = ProgressReporter(
        2,
        enabled=True,
        use_color=True,
        stream=stream,
        dynamic=True,
    )
    with reporter.phase("data", "phase", 2):
        reporter.advance(2, "done")
    reporter.close("completed")

    output = stream.getvalue()
    assert "\x1b[32m" not in output
    assert "/" in output


def test_redirected_output_is_plain_monotonic_and_reconciles_early_stop() -> None:
    stream = StringIO()
    reporter = ProgressReporter(
        5,
        enabled=True,
        stream=stream,
        dynamic=False,
        refresh_interval=60.0,
    )
    with reporter.phase("breast_cancer", "AdaBoost scaling", 5):
        with reporter.task(5, "AdaBoost round 0/5") as callback:
            callback(1, 5, "AdaBoost round 1/5")
            callback(2, 5, "perfect learner; early stop")
    reporter.close("completed")

    output = stream.getvalue()
    assert "[progress] START" in output
    assert "[progress] COMPLETED" in output
    assert "\x1b" not in output and "\r" not in output
    assert "elapsed" in output and "ETA" in output
    assert "nan" not in output.lower()
    assert reporter.timing_records[0]["planned_units"] == 5
    assert reporter.timing_records[0]["observed_units"] == 2
    assert reporter.timing_records[0]["status"] == "completed"


def test_exception_and_keyboard_interrupt_cleanup_are_recorded() -> None:
    for exception in (RuntimeError("failure"), KeyboardInterrupt()):
        reporter = ProgressReporter(3, enabled=False)
        with pytest.raises(type(exception)):
            with reporter.phase("data", "fragile phase", 3):
                reporter.advance(1)
                raise exception
        reporter.close("failed")
        assert reporter.timing_records[0]["status"] == "failed"
        assert reporter.timing_records[0]["observed_units"] == 1


def test_progress_callbacks_do_not_change_model_predictions() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)
    factories = (
        lambda: AdaBoostClassifier(n_estimators=4, random_state=4),
        lambda: RandomForestClassifier(
            n_estimators=4, max_depth=3, random_state=4
        ),
        lambda: GradientBoostingClassifier(
            n_estimators=4, max_depth=2, random_state=4
        ),
    )
    for factory in factories:
        expected = factory().fit(X, y)
        updates: list[tuple[int, int, str]] = []
        actual = factory().fit(
            X,
            y,
            progress_callback=lambda completed, total, detail: updates.append(
                (completed, total, detail)
            ),
        )
        np.testing.assert_array_equal(actual.predict(X), expected.predict(X))
        np.testing.assert_allclose(
            actual.predict_proba(X), expected.predict_proba(X)
        )
        assert updates
        assert [item[0] for item in updates] == list(
            range(1, len(updates) + 1)
        )
        assert all(item[1] == 4 for item in updates)


def test_planned_suite_units_reflects_profile_and_selected_datasets() -> None:
    config = ExperimentConfig.quick()
    one_dataset = planned_suite_units(config, ("breast_cancer",))
    two_datasets = planned_suite_units(config, ("breast_cancer", "adult"))
    assert one_dataset > 0
    assert two_datasets > one_dataset
