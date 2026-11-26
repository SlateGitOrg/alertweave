"""Candidate links between alerts.

The inverted index maps an entity to the recent alerts that carry it, so a new alert is only
compared with alerts it shares at least one positively weighted entity with. Zero-weight
entities (hubs, allowlisted infrastructure) are never walked, which is also what keeps the
cost proportional to real overlaps instead of to all pairs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

from .model import Alert


class RecentIndex:
    """Inverted index entity -> alerts inside the link window, in stream order."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._by_entity: dict[str, deque[tuple[int, str]]] = {}
        self._seq: dict[str, int] = {}

    def add(self, alert: Alert) -> None:
        seq = len(self._seq)
        self._seq[alert.id] = seq
        horizon = alert.time - self._window
        for key in alert.entities:
            recent = self._by_entity.setdefault(key, deque())
            while recent and recent[0][0] < horizon:
                recent.popleft()
            recent.append((alert.time, alert.id))

    def candidates(self, alert: Alert, weights: Mapping[str, float]) -> list[tuple[str, float]]:
        """Earlier alerts within the window, with the summed weight of shared entities.

        Call before ``add(alert)``. Results are in stream order of the earlier alert.
        """
        horizon = alert.time - self._window
        scores: dict[str, float] = {}
        for key in alert.entities:
            weight = weights[key]
            if weight <= 0.0:
                continue
            recent = self._by_entity.get(key)
            if not recent:
                continue
            while recent and recent[0][0] < horizon:
                recent.popleft()
            for _, prior_id in recent:
                scores[prior_id] = scores.get(prior_id, 0.0) + weight
        return sorted(scores.items(), key=lambda item: self._seq[item[0]])


def indexed_links(
    ordered: Sequence[Alert],
    table: Mapping[str, Mapping[str, float]],
    window: int,
    tau: float,
) -> dict[tuple[str, str], float]:
    """All (earlier, later) pairs with score >= tau, found through the inverted index."""
    index = RecentIndex(window)
    links: dict[tuple[str, str], float] = {}
    for alert in ordered:
        for prior_id, score in index.candidates(alert, table[alert.id]):
            if score >= tau:
                links[(prior_id, alert.id)] = score
        index.add(alert)
    return links


def brute_force_links(
    ordered: Sequence[Alert],
    table: Mapping[str, Mapping[str, float]],
    window: int,
    tau: float,
) -> dict[tuple[str, str], float]:
    """Reference implementation: score every earlier alert inside the window."""
    links: dict[tuple[str, str], float] = {}
    for position, alert in enumerate(ordered):
        weights = table[alert.id]
        for earlier in ordered[:position]:
            if alert.time - earlier.time > window:
                continue
            shared = set(earlier.entities)
            score = 0.0
            for key in alert.entities:
                if key in shared:
                    score += weights[key]
            if score >= tau:
                links[(earlier.id, alert.id)] = score
    return links
