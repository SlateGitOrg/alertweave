"""Trailing-window entity rarity weights.

For the alert being observed, over the alerts in the trailing window plus the alert itself:

    weight(entity) = prior[type] * ln(N / df(entity))

``N`` counts the window alerts and ``df`` those that carry the entity. An entity whose ``df``
is above the chosen percentile of all entity document frequencies in the window, or that is on
the infrastructure allowlist, gets weight 0. Only alerts earlier in (time, id) order are ever
read, so a weight never changes when later alerts arrive.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from .model import DAY, Alert, entity_type, sort_alerts

DEFAULT_PRIORS: Mapping[str, float] = {
    "host": 1.1,
    "user": 1.0,
    "ip": 0.7,
    "domain": 0.8,
    "hash": 1.5,
}


@dataclass(frozen=True)
class WeightConfig:
    window: int = 7 * DAY
    priors: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PRIORS))
    hub_percentile: float = 99.0
    allowlist: frozenset[str] = frozenset()


class TrailingWeights:
    """Incremental weight computation over a stream of alerts in (time, id) order."""

    def __init__(self, config: WeightConfig) -> None:
        self._config = config
        self._window: deque[Alert] = deque()
        self._df: dict[str, int] = {}

    def observe(self, alert: Alert) -> dict[str, float]:
        """Add ``alert`` to the window and return its entities' weights at this moment."""
        config = self._config
        cutoff = alert.time - config.window
        while self._window and self._window[0].time < cutoff:
            for key in self._window.popleft().entities:
                self._df[key] -= 1
                if self._df[key] == 0:
                    del self._df[key]
        self._window.append(alert)
        for key in alert.entities:
            self._df[key] = self._df.get(key, 0) + 1

        total = len(self._window)
        threshold = math.inf
        if config.hub_percentile < 100.0:
            counts = np.fromiter(self._df.values(), dtype=np.int64, count=len(self._df))
            threshold = float(np.percentile(counts, config.hub_percentile, method="higher"))
        weights: dict[str, float] = {}
        for key in alert.entities:
            df = self._df[key]
            if key in config.allowlist or df > threshold:
                weights[key] = 0.0
            else:
                prior = config.priors.get(entity_type(key), 1.0)
                weights[key] = prior * math.log(total / df)
        return weights


def weight_table(alerts: Iterable[Alert], config: WeightConfig) -> dict[str, dict[str, float]]:
    """Per-alert entity weights (alert id -> entity -> weight) for the whole stream."""
    tracker = TrailingWeights(config)
    return {alert.id: tracker.observe(alert) for alert in sort_alerts(alerts)}


def flat_table(alerts: Iterable[Alert]) -> dict[str, dict[str, float]]:
    """Weight 1 for every entity: the naive baseline where any shared entity is evidence."""
    return {alert.id: dict.fromkeys(alert.entities, 1.0) for alert in sort_alerts(alerts)}
