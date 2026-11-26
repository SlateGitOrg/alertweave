"""Explain why an incident hangs together: the links that merged its alerts."""

from __future__ import annotations

from dataclasses import dataclass

from .cluster import ClusterResult, Incident
from .model import format_time


@dataclass(frozen=True)
class LinkExplanation:
    earlier: str
    later: str
    score: float
    shared: tuple[tuple[str, float], ...]


def explain_incident(result: ClusterResult, incident: Incident) -> list[LinkExplanation]:
    """Spanning links of ``incident`` in merge order.

    Each link lists every entity the two alerts share with its weight at link time (the later
    alert's trailing-window weight), heaviest first, ties by entity text. Hubs show as 0.
    """
    members = set(incident.alert_ids)
    by_id = {alert.id: alert for alert in result.alerts}
    lines = []
    for link in result.links:
        if link.later not in members:
            continue
        earlier_entities = set(by_id[link.earlier].entities)
        weights = result.table[link.later]
        shared = [
            (key, weights[key]) for key in by_id[link.later].entities if key in earlier_entities
        ]
        shared.sort(key=lambda item: (-item[1], item[0]))
        lines.append(LinkExplanation(link.earlier, link.later, link.score, tuple(shared)))
    return lines


def format_explanation(result: ClusterResult, incident: Incident) -> str:
    by_id = {alert.id: alert for alert in result.alerts}
    lines = [
        f"{incident.id}: {incident.count} alerts, "
        f"{format_time(incident.first_seen)} to {format_time(incident.last_seen)}, "
        f"max severity {incident.max_severity}",
        f"tactics: {', '.join(incident.tactics)}",
        f"spanning links: {len(explain_incident(result, incident))}",
    ]
    for link in explain_incident(result, incident):
        earlier, later = by_id[link.earlier], by_id[link.later]
        lines.append(
            f"{link.earlier} -> {link.later}  score {link.score:.2f}  "
            f"[{earlier.rule} -> {later.rule}]"
        )
        for key, weight in link.shared:
            lines.append(f"    {key}  {weight:.2f}")
    return "\n".join(lines)
