"""Grading against ground truth and the tune-then-grade parameter sweep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .cluster import ClusterConfig, ClusterResult, cluster_batch
from .metrics import Scores, score
from .model import HOUR, Alert, day_index, sort_alerts
from .weights import WeightConfig, flat_table, weight_table

TAU_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
WINDOW_GRID = (1 * HOUR, 2 * HOUR, 4 * HOUR, 8 * HOUR)
TUNE_DAYS = (1, 7)
TEST_DAYS = (8, 14)
NAIVE_TAU = 1.0
TOLERANCE = 0.002


def naive_config(base: ClusterConfig) -> ClusterConfig:
    """The one naive baseline: any shared entity links, no weighting, 24 h link window.

    The window is the whole incident gap (24 h), so the baseline is as permissive in time as the
    incident rule allows. It is fixed, not tuned: ``sweep --mode naive`` and ``eval`` grade this
    same configuration.
    """
    return replace(base, tau=NAIVE_TAU, naive=True, link_window=base.max_gap)


def ids_in_days(alerts: Sequence[Alert], days: tuple[int, int]) -> list[str]:
    origin = min(alert.time for alert in alerts)
    return [a.id for a in alerts if days[0] <= day_index(a.time, origin) <= days[1]]


def grade(
    alerts: Sequence[Alert],
    truth: Mapping[str, str],
    config: ClusterConfig,
    days: tuple[int, int] = TEST_DAYS,
) -> tuple[ClusterResult, Scores]:
    """Cluster every alert (history included) and score only the alerts in ``days``."""
    result = cluster_batch(alerts, config)
    return result, score(result.labels, truth, ids_in_days(alerts, days))


@dataclass(frozen=True)
class SweepCell:
    tau: float
    window: int
    f1: float


@dataclass(frozen=True)
class SweepResult:
    mode: str
    cells: tuple[SweepCell, ...]
    best: ClusterConfig
    tune: Scores
    held_out: Scores


def sweep(
    alerts: Sequence[Alert],
    truth: Mapping[str, str],
    base: ClusterConfig,
    mode: str = "weighted",
    taus: Sequence[float] = TAU_GRID,
    windows: Sequence[int] = WINDOW_GRID,
) -> SweepResult:
    """Pick tau and the link window on days 1-7 only, then grade them on days 8-14.

    The tuning run clusters just the alerts of days 1-7, so nothing from the held-out days can
    influence the choice. Every setting within ``TOLERANCE`` B-cubed F1 of the best counts as
    tied, and the lowest tau then the shortest window wins: differences that small are noise on
    a 7-day sample, so the tie-break is a fixed, total rule (stable choice) rather than a
    judgement made with the held-out days in view.

    ``mode="naive"`` is not tuned: the baseline is the single configuration of ``naive_config``
    (one grid cell), so the sweep and ``eval`` report the same naive numbers.
    """
    ordered = sort_alerts(alerts)
    tune_ids = set(ids_in_days(ordered, TUNE_DAYS))
    tune_alerts = [a for a in ordered if a.id in tune_ids]
    naive = mode == "naive"
    table = flat_table(tune_alerts) if naive else weight_table(tune_alerts, base.weights)
    grid_taus = (NAIVE_TAU,) if naive else tuple(taus)
    if naive:
        windows = (naive_config(base).link_window,)
    cells = []
    for tau in grid_taus:
        for window in windows:
            config = replace(base, tau=tau, link_window=window, naive=naive)
            result = cluster_batch(tune_alerts, config, table)
            f1 = score(result.labels, truth, tune_ids).bcubed_f1
            cells.append(SweepCell(tau, window, f1))
    top = max(cell.f1 for cell in cells)
    eligible = [cell for cell in cells if cell.f1 >= top - TOLERANCE]
    best_cell = min(eligible, key=lambda c: (c.tau, c.window))
    best = replace(base, tau=best_cell.tau, link_window=best_cell.window, naive=naive)
    tuned = cluster_batch(tune_alerts, best, table)
    tune_scores = score(tuned.labels, truth, tune_ids)
    _, held_out = grade(ordered, truth, best, TEST_DAYS)
    return SweepResult(mode, tuple(cells), best, tune_scores, held_out)


def with_allowlist(config: ClusterConfig, allowlist: frozenset[str]) -> ClusterConfig:
    weights = replace(config.weights, allowlist=allowlist)
    return replace(config, weights=weights)


__all__ = [
    "TAU_GRID",
    "WINDOW_GRID",
    "WeightConfig",
    "SweepCell",
    "SweepResult",
    "grade",
    "ids_in_days",
    "naive_config",
    "sweep",
    "with_allowlist",
]
