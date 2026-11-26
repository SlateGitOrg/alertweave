"""Alert record and entity normalisation.

An alert carries typed entities. Internally every entity is a key ``"<type>:<value>"`` with the
value normalised, so the same host or account written two ways collapses to one key.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ENTITY_TYPES = ("host", "user", "ip", "hash", "domain")
HOUR = 3600
DAY = 86400
_USER_DOMAIN = "corp\\"
_USER_SUFFIX = "@corp.example"
_IPV4_LIKE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def normalise_host(value: str) -> str:
    """Lowercase short hostname: ``WS-001.corp.example`` becomes ``ws-001``."""
    text = value.strip().lower().rstrip(".")
    if _IPV4_LIKE.match(text):
        return text
    return text.split(".")[0]


def normalise_user(value: str) -> str:
    """Lowercase account name without the ``CORP\\`` prefix or ``@corp.example`` suffix."""
    text = value.strip().lower()
    if text.startswith(_USER_DOMAIN):
        text = text[len(_USER_DOMAIN) :]
    if text.endswith(_USER_SUFFIX):
        text = text[: -len(_USER_SUFFIX)]
    return text


def normalise_ip(value: str) -> str:
    """Canonical IP text; zero-padded IPv4 octets are accepted."""
    text = value.strip().lower()
    if _IPV4_LIKE.match(text):
        text = ".".join(str(int(octet)) for octet in text.split("."))
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return text


def normalise_hash(value: str) -> str:
    return value.strip().lower()


def normalise_domain(value: str) -> str:
    return value.strip().lower().rstrip(".")


_NORMALISERS = {
    "host": normalise_host,
    "user": normalise_user,
    "ip": normalise_ip,
    "hash": normalise_hash,
    "domain": normalise_domain,
}


def entity_key(entity_type: str, value: str) -> str:
    """Return the normalised ``type:value`` key (empty string for an empty value)."""
    if entity_type not in _NORMALISERS:
        raise ValueError(f"unknown entity type: {entity_type!r}")
    normalised = _NORMALISERS[entity_type](value)
    return f"{entity_type}:{normalised}" if normalised else ""


def entity_type(key: str) -> str:
    return key.split(":", 1)[0]


def parse_time(text: str) -> int:
    """ISO-8601 timestamp (``Z`` or offset) to integer UTC epoch seconds."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def format_time(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class Alert:
    id: str
    time: int
    product: str
    rule: str
    tactic: str
    severity: int
    entities: tuple[str, ...]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Alert:
        keys: set[str] = set()
        for kind in ENTITY_TYPES:
            for value in record.get(kind, ()):
                key = entity_key(kind, str(value))
                if key:
                    keys.add(key)
        time = record["time"]
        return cls(
            id=str(record["id"]),
            time=parse_time(time) if isinstance(time, str) else int(time),
            product=str(record["product"]),
            rule=str(record["rule"]),
            tactic=str(record["tactic"]),
            severity=int(record["severity"]),
            entities=tuple(sorted(keys)),
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "time": format_time(self.time),
            "product": self.product,
            "rule": self.rule,
            "tactic": self.tactic,
            "severity": self.severity,
        }
        for kind in ENTITY_TYPES:
            values = [key.split(":", 1)[1] for key in self.entities if entity_type(key) == kind]
            if values:
                record[kind] = values
        return record


def sort_alerts(alerts: Iterable[Alert]) -> list[Alert]:
    """Stream order: by time, ties broken by alert id. Rejects duplicate ids."""
    ordered = sorted(alerts, key=lambda a: (a.time, a.id))
    if len({a.id for a in ordered}) != len(ordered):
        raise ValueError("duplicate alert ids")
    return ordered


def load_alerts(path: Path) -> list[Alert]:
    alerts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                alerts.append(Alert.from_record(json.loads(line)))
    return alerts


def day_index(time: int, origin: int) -> int:
    """1-based day number of ``time`` counted from the UTC midnight at or before ``origin``."""
    return (time - (origin // DAY) * DAY) // DAY + 1
