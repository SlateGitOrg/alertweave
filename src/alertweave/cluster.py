"""Time-bounded incident clustering with union-find.

Alerts are processed in (time, id) order. A link between a new alert and an earlier one is
accepted when the shared-entity score reaches ``tau`` and the earlier alert's incident was
active within ``max_gap`` of the new alert. The streaming engine decides links as alerts
arrive; the batch engine computes every link first through the inverted index and applies them
in the same order, so both return the same partition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .link import RecentIndex, indexed_links
from .model import DAY, Alert, sort_alerts
from .weights import TrailingWeights, WeightConfig, flat_table, weight_table

DEFAULT_TAU = 6.0
DEFAULT_LINK_WINDOW = 4 * 3600
DEFAULT_MAX_GAP = DAY

TACTIC_ORDER = (
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
)


class UnionFind:
    """Disjoint sets with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: list[int] = []
        self._rank: list[int] = []

    def __len__(self) -> int:
        return len(self._parent)

    def add(self) -> int:
        self._parent.append(len(self._parent))
        self._rank.append(0)
        return len(self._parent) - 1

    def parent(self, item: int) -> int:
        """The item's current parent (its root once compressed); does not compress."""
        return self._parent[item]

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, first: int, second: int) -> int | None:
        """Merge the sets; return the surviving root, or None if already together."""
        a, b = self.find(first), self.find(second)
        if a == b:
            return None
        if self._rank[a] < self._rank[b]:
            a, b = b, a
        self._parent[b] = a
        if self._rank[a] == self._rank[b]:
            self._rank[a] += 1
        return a


@dataclass(frozen=True)
class ClusterConfig:
    tau: float = DEFAULT_TAU
    link_window: int = DEFAULT_LINK_WINDOW
    max_gap: int = DEFAULT_MAX_GAP
    weights: WeightConfig = field(default_factory=WeightConfig)
    naive: bool = False


@dataclass(frozen=True)
class Link:
    earlier: str
    later: str
    score: float


@dataclass(frozen=True)
class Incident:
    id: str
    alert_ids: tuple[str, ...]
    first_seen: int
    last_seen: int
    max_severity: int
    tactics: tuple[str, ...]
    entities: tuple[tuple[str, float], ...]

    @property
    def count(self) -> int:
        return len(self.alert_ids)


@dataclass(frozen=True)
class ClusterResult:
    alerts: tuple[Alert, ...]
    labels: dict[str, str]
    links: tuple[Link, ...]
    table: Mapping[str, Mapping[str, float]]
    config: ClusterConfig

    def partition(self) -> frozenset[frozenset[str]]:
        groups: dict[str, set[str]] = {}
        for alert_id, label in self.labels.items():
            groups.setdefault(label, set()).add(alert_id)
        return frozenset(frozenset(group) for group in groups.values())

    def incidents(self) -> list[Incident]:
        """Incidents, largest first (ties by incident id)."""
        by_label: dict[str, list[Alert]] = {}
        for alert in self.alerts:
            by_label.setdefault(self.labels[alert.id], []).append(alert)
        incidents = [self._summarise(label, members) for label, members in by_label.items()]
        return sorted(incidents, key=lambda inc: (-inc.count, inc.id))

    def _summarise(self, label: str, members: Sequence[Alert]) -> Incident:
        best: dict[str, float] = {}
        for alert in members:
            for key, weight in self.table[alert.id].items():
                best[key] = max(best.get(key, 0.0), weight)
        tactics = {a.tactic for a in members}
        rank = {name: i for i, name in enumerate(TACTIC_ORDER)}
        return Incident(
            id=label,
            alert_ids=tuple(a.id for a in members),
            first_seen=members[0].time,
            last_seen=max(a.time for a in members),
            max_severity=max(a.severity for a in members),
            tactics=tuple(sorted(tactics, key=lambda t: (rank.get(t, len(rank)), t))),
            entities=tuple(sorted(best.items(), key=lambda item: (-item[1], item[0]))),
        )


class _Incidents:
    """Union-find over alert ids that also tracks each incident's last activity."""

    def __init__(self, max_gap: int) -> None:
        self._uf = UnionFind()
        self._index: dict[str, int] = {}
        self._ids: list[str] = []
        self._last: list[int] = []
        self._max_gap = max_gap

    def add(self, alert: Alert) -> None:
        self._index[alert.id] = self._uf.add()
        self._ids.append(alert.id)
        self._last.append(alert.time)

    def try_link(self, earlier: str, later: Alert) -> bool:
        first = self._uf.find(self._index[earlier])
        second = self._uf.find(self._index[later.id])
        if first == second or later.time - self._last[first] > self._max_gap:
            return False
        newest = max(self._last[first], self._last[second])
        merged = self._uf.union(first, second)
        self._last[merged] = newest
        return True

    def labels(self) -> dict[str, str]:
        first_of_root: dict[int, str] = {}
        labels: dict[str, str] = {}
        for position, alert_id in enumerate(self._ids):
            root = self._uf.find(position)
            first_of_root.setdefault(root, alert_id)
            labels[alert_id] = f"INC-{first_of_root[root]}"
        return labels


def _make_result(ordered, incidents, links, table, config) -> ClusterResult:
    return ClusterResult(tuple(ordered), incidents.labels(), tuple(links), table, config)


def cluster_streaming(alerts: Iterable[Alert], config: ClusterConfig) -> ClusterResult:
    """Process alerts one at a time in (time, id) order; weights see only the past."""
    ordered = sort_alerts(alerts)
    tracker = TrailingWeights(config.weights)
    index = RecentIndex(config.link_window)
    incidents = _Incidents(config.max_gap)
    table: dict[str, dict[str, float]] = {}
    links: list[Link] = []
    for alert in ordered:
        weights = dict.fromkeys(alert.entities, 1.0) if config.naive else tracker.observe(alert)
        table[alert.id] = weights
        incidents.add(alert)
        for prior_id, score in index.candidates(alert, weights):
            if score >= config.tau and incidents.try_link(prior_id, alert):
                links.append(Link(prior_id, alert.id, score))
        index.add(alert)
    return _make_result(ordered, incidents, links, table, config)


def cluster_batch(
    alerts: Iterable[Alert],
    config: ClusterConfig,
    table: Mapping[str, Mapping[str, float]] | None = None,
) -> ClusterResult:
    """Compute every weight and every candidate link first, then merge in stream order.

    A precomputed ``table`` (from ``weight_table``) can be passed in; it depends only on the
    weight configuration, so parameter sweeps reuse it.
    """
    ordered = sort_alerts(alerts)
    if table is None:
        table = flat_table(ordered) if config.naive else weight_table(ordered, config.weights)
    wanted = indexed_links(ordered, table, config.link_window, config.tau)
    position = {alert.id: i for i, alert in enumerate(ordered)}
    by_later: dict[str, list[str]] = {}
    for earlier, later in wanted:
        by_later.setdefault(later, []).append(earlier)
    incidents = _Incidents(config.max_gap)
    links: list[Link] = []
    for alert in ordered:
        incidents.add(alert)
        for earlier in sorted(by_later.get(alert.id, ()), key=position.__getitem__):
            if incidents.try_link(earlier, alert):
                links.append(Link(earlier, alert.id, wanted[(earlier, alert.id)]))
    return _make_result(ordered, incidents, links, table, config)


def cluster_alerts(
    alerts: Iterable[Alert], config: ClusterConfig, engine: str = "streaming"
) -> ClusterResult:
    if engine == "streaming":
        return cluster_streaming(alerts, config)
    if engine == "batch":
        return cluster_batch(alerts, config)
    raise ValueError(f"unknown engine: {engine!r}")
