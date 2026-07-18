"""Terminal progress reporting for the experiment orchestration layer."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import TextIO

from tqdm import tqdm

from src.experiments.config import ExperimentConfig


ProgressCallback = Callable[[int, int, str], None]
_SPINNER = ("|", "/", "-", "\\")
_BAR_FORMAT = "{desc:<8} [{bar:28}] {percentage:3.0f}% | {postfix}"


@dataclass(frozen=True, slots=True)
class PhaseTiming:
    """Machine-readable timing metadata for one completed phase."""

    dataset: str
    experiment: str
    planned_units: int
    observed_units: int
    elapsed_seconds: float
    status: str


def planned_suite_units(
    config: ExperimentConfig, dataset_names: tuple[str, ...]
) -> int:
    """Return deterministic work units for the selected experiment configuration."""

    total = len(dataset_names)  # dataset loading
    for name in dataset_names:
        binary = name in {"breast_cancer", "adult"}
        total += 1  # holdout preprocessing
        total += 3  # baseline models
        total += max(config.ada_estimators)
        total += max(config.rf_estimators)
        total += len(config.rf_depths) * config.fixed_estimators
        total += config.cv_folds * (1 + 3 * config.fixed_estimators)
        total += (
            config.noise_replicates
            * len(config.noise_levels)
            * 2
            * config.fixed_estimators
        )
        total += len(config.k_values) * config.kmeans_restarts + 5
        total += 2 * config.fixed_estimators  # SAMME and SAMME.R
        if binary:
            total += 2 * config.fixed_estimators  # AdaBoost and GBM bonus
    if "breast_cancer" in dataset_names:
        total += (
            config.bootstrap_replicates * 2 * config.fixed_estimators
        )
    return total


class ProgressReporter:
    """Maintain one overall bar and one current-operation bar.

    Interactive terminals receive green slash-filled tqdm bars. Redirected
    output receives stable, ANSI-free log lines instead.
    """

    def __init__(
        self,
        total_units: int,
        *,
        enabled: bool,
        use_color: bool = True,
        stream: TextIO | None = None,
        dynamic: bool | None = None,
        refresh_interval: float = 10.0,
    ) -> None:
        if total_units < 1:
            raise ValueError("total_units must be at least 1")
        if refresh_interval <= 0.0:
            raise ValueError("refresh_interval must be positive")
        self.total_units = total_units
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.dynamic = (
            bool(getattr(self.stream, "isatty", lambda: False)())
            if dynamic is None
            else dynamic
        )
        self.use_color = (
            use_color
            and self.dynamic
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM", "").lower() != "dumb"
        )
        self.refresh_interval = refresh_interval
        self.timings: list[PhaseTiming] = []

        self._started = perf_counter()
        self._overall_accounted = 0
        self._phase_started: float | None = None
        self._phase_dataset = ""
        self._phase_name = ""
        self._phase_planned = 0
        self._phase_observed = 0
        self._phase_accounted = 0
        self._detail = ""
        self._spinner_index = 0
        self._last_log_time = self._started
        self._last_log_percent = -1
        self._closed = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._overall: tqdm[object] | None = None
        self._current: tqdm[object] | None = None
        self._heartbeat: threading.Thread | None = None

        if self.enabled:
            if self.dynamic:
                self._overall = self._new_bar("Overall", total_units, 0)
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                name="experiment-progress-heartbeat",
                daemon=True,
            )
            self._heartbeat.start()

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self._started

    @property
    def timing_records(self) -> list[dict[str, object]]:
        return [asdict(item) for item in self.timings]

    @contextmanager
    def phase(
        self, dataset: str, experiment: str, planned_units: int
    ) -> Iterator[ProgressReporter]:
        """Start a phase and always close its display and timing record."""

        self.start_phase(dataset, experiment, planned_units)
        try:
            yield self
        except BaseException:
            self.finish_phase("failed")
            raise
        else:
            self.finish_phase("completed")

    @contextmanager
    def task(
        self, planned_units: int, detail: str
    ) -> Iterator[ProgressCallback]:
        """Yield a callback and reconcile unused units on task completion."""

        if planned_units < 1:
            raise ValueError("planned_units must be at least 1")
        before = self._phase_accounted
        self.set_detail(detail)
        last_completed = 0

        def callback(completed: int, total: int, callback_detail: str) -> None:
            nonlocal last_completed
            del total
            delta = max(0, completed - last_completed)
            last_completed = max(last_completed, completed)
            self.advance(delta, callback_detail or detail)

        try:
            yield callback
        except BaseException:
            raise
        else:
            consumed = self._phase_accounted - before
            remaining = max(0, planned_units - consumed)
            if remaining:
                self._account(remaining, observed=False)

    def start_phase(
        self, dataset: str, experiment: str, planned_units: int
    ) -> None:
        if planned_units < 1:
            raise ValueError("planned_units must be at least 1")
        with self._lock:
            if self._phase_started is not None:
                raise RuntimeError("A progress phase is already active")
            self._phase_started = perf_counter()
            self._phase_dataset = dataset
            self._phase_name = experiment
            self._phase_planned = planned_units
            self._phase_observed = 0
            self._phase_accounted = 0
            self._detail = "starting"
            self._last_log_percent = -1
            if self.enabled and self.dynamic:
                self._current = self._new_bar("Current", planned_units, 1)
                self._refresh_dynamic()
            elif self.enabled:
                self._write_line("START")

    def set_detail(self, detail: str) -> None:
        with self._lock:
            self._detail = detail
            if self.enabled and self.dynamic:
                self._refresh_dynamic()

    def advance(self, units: int = 1, detail: str | None = None) -> None:
        if units < 0:
            raise ValueError("Progress units cannot be negative")
        with self._lock:
            if detail is not None:
                self._detail = detail
            self._account(units, observed=True)
            if self.enabled and self.dynamic:
                self._refresh_dynamic()
            elif self.enabled:
                self._maybe_write_progress_line()

    def finish_phase(self, status: str) -> None:
        with self._lock:
            if self._phase_started is None:
                return
            remaining = max(0, self._phase_planned - self._phase_accounted)
            if status == "completed" and remaining:
                self._account(remaining, observed=False)
            elapsed = perf_counter() - self._phase_started
            self.timings.append(
                PhaseTiming(
                    dataset=self._phase_dataset,
                    experiment=self._phase_name,
                    planned_units=self._phase_planned,
                    observed_units=self._phase_observed,
                    elapsed_seconds=elapsed,
                    status=status,
                )
            )
            self._detail = status
            if self.enabled and self.dynamic and self._current is not None:
                self._refresh_dynamic()
                self._current.close()
                self._current = None
            elif self.enabled:
                self._write_line(status.upper())
            self._phase_started = None

    def close(self, status: str) -> None:
        heartbeat = self._heartbeat
        with self._lock:
            if self._closed:
                return
            if self._phase_started is not None:
                self.finish_phase(status)
            self._stop_event.set()
            if self.enabled and self.dynamic and self._overall is not None:
                if status == "completed":
                    remaining = max(0, self.total_units - int(self._overall.n))
                    if remaining:
                        self._overall.update(remaining)
                    self._overall_accounted = self.total_units
                self._detail = status
                self._refresh_dynamic()
                self._overall.close()
            elif self.enabled:
                if status == "completed":
                    self._overall_accounted = self.total_units
                overall_percent = int(
                    100 * self._overall_accounted / self.total_units
                )
                self.stream.write(
                    f"[progress] {status.upper()} | Overall "
                    f"{overall_percent:3d}% | elapsed "
                    f"{_format_duration(self.elapsed_seconds)}\n"
                )
                self.stream.flush()
            self._closed = True
        if heartbeat is not None and heartbeat is not threading.current_thread():
            heartbeat.join(timeout=min(1.0, self.refresh_interval))

    def _new_bar(self, description: str, total: int, position: int) -> tqdm[object]:
        return tqdm(
            total=total,
            desc=description,
            ascii="-/",
            colour="green" if self.use_color else None,
            bar_format=_BAR_FORMAT,
            position=position,
            leave=position == 0,
            dynamic_ncols=True,
            mininterval=0.1,
            file=self.stream,
        )

    def _account(self, units: int, *, observed: bool) -> None:
        if self._phase_started is None:
            raise RuntimeError("No progress phase is active")
        allowed = min(units, self._phase_planned - self._phase_accounted)
        if allowed <= 0:
            return
        self._phase_accounted += allowed
        self._overall_accounted = min(
            self.total_units, self._overall_accounted + allowed
        )
        if observed:
            self._phase_observed += allowed
        if self._current is not None:
            self._current.update(allowed)
        if self._overall is not None:
            self._overall.update(allowed)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.refresh_interval):
            with self._lock:
                if self._closed:
                    return
                self._spinner_index = (self._spinner_index + 1) % len(_SPINNER)
                if self.dynamic:
                    self._refresh_dynamic()
                elif self._phase_started is not None:
                    self._write_line("WORKING")

    def _refresh_dynamic(self) -> None:
        spinner = _SPINNER[self._spinner_index]
        if self._overall is not None:
            eta = _estimate_remaining(self._overall)
            finish = (
                (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M")
                if eta is not None
                else "estimating..."
            )
            self._overall.set_postfix_str(
                f"{spinner} {self._phase_dataset} / {self._phase_name} | "
                f"elapsed {_format_duration(self.elapsed_seconds)} | "
                f"ETA est. {_format_eta(eta)} | finish ~{finish}",
                refresh=False,
            )
            self._overall.refresh()
        if self._current is not None and self._phase_started is not None:
            eta = _estimate_remaining(self._current)
            elapsed = perf_counter() - self._phase_started
            self._current.set_postfix_str(
                f"{spinner} {self._detail} | elapsed {_format_duration(elapsed)} | "
                f"ETA {_format_eta(eta)}",
                refresh=False,
            )
            self._current.refresh()

    def _maybe_write_progress_line(self) -> None:
        if self._phase_planned <= 0:
            return
        percent = int(100 * self._phase_accounted / self._phase_planned)
        now = perf_counter()
        if percent >= self._last_log_percent + 10 or (
            now - self._last_log_time >= self.refresh_interval
        ):
            self._write_line("PROGRESS")
            self._last_log_percent = percent

    def _write_line(self, state: str) -> None:
        if self._phase_started is None:
            return
        elapsed = perf_counter() - self._phase_started
        eta = _estimate_from_counts(
            self._phase_accounted, self._phase_planned, elapsed
        )
        percent = int(100 * self._phase_accounted / self._phase_planned)
        overall_percent = int(
            100 * self._overall_accounted / self.total_units
        )
        overall_eta = _estimate_from_counts(
            self._overall_accounted, self.total_units, self.elapsed_seconds
        )
        finish = (
            (datetime.now() + timedelta(seconds=overall_eta)).strftime("%H:%M")
            if overall_eta is not None
            else "estimating..."
        )
        spinner = _SPINNER[self._spinner_index]
        self.stream.write(
            f"[progress] {state} | Overall {overall_percent:3d}% "
            f"ETA est. {_format_eta(overall_eta)} finish ~{finish} | "
            f"Current {percent:3d}% {spinner} | "
            f"{self._phase_dataset} / {self._phase_name} | {self._detail} | "
            f"elapsed {_format_duration(elapsed)} | ETA {_format_eta(eta)}\n"
        )
        self.stream.flush()
        self._last_log_time = perf_counter()


def _estimate_remaining(bar: tqdm[object]) -> float | None:
    elapsed = float(bar.format_dict.get("elapsed", 0.0) or 0.0)
    return _estimate_from_counts(int(bar.n), int(bar.total or 0), elapsed)


def _estimate_from_counts(
    completed: int, total: int, elapsed: float
) -> float | None:
    if completed < 2 or total <= completed or elapsed <= 0.0:
        return None
    rate = completed / elapsed
    if rate <= 0.0:
        return None
    return max(0.0, (total - completed) / rate)


def _format_eta(seconds: float | None) -> str:
    return "estimating..." if seconds is None else _format_duration(seconds)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


__all__ = [
    "PhaseTiming",
    "ProgressCallback",
    "ProgressReporter",
    "planned_suite_units",
]
