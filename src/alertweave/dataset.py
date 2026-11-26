"""Reading and writing a scenario directory: alerts.jsonl, truth.jsonl, allowlist.txt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .gen import Scenario
from .model import Alert, load_alerts

ALERTS_FILE = "alerts.jsonl"
TRUTH_FILE = "truth.jsonl"
ALLOWLIST_FILE = "allowlist.txt"
PLANTED_FILE = "planted.json"


@dataclass(frozen=True)
class Dataset:
    alerts: list[Alert]
    truth: dict[str, str]
    allowlist: frozenset[str]


def write_scenario(scenario: Scenario, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ALERTS_FILE).open("w", encoding="utf-8") as handle:
        for record in scenario.records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (directory / TRUTH_FILE).open("w", encoding="utf-8") as handle:
        for alert_id in sorted(scenario.truth):
            row = {"alert_id": alert_id, "incident": scenario.truth[alert_id]}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (directory / ALLOWLIST_FILE).write_text(
        "".join(f"{key}\n" for key in sorted(scenario.allowlist)), encoding="utf-8"
    )
    (directory / PLANTED_FILE).write_text(
        json.dumps(scenario.planted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_dataset(directory: Path, need_truth: bool = True) -> Dataset:
    alerts = load_alerts(directory / ALERTS_FILE)
    truth: dict[str, str] = {}
    truth_path = directory / TRUTH_FILE
    if truth_path.exists():
        with truth_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    truth[row["alert_id"]] = row["incident"]
    elif need_truth:
        raise FileNotFoundError(f"missing {truth_path}")
    allowlist_path = directory / ALLOWLIST_FILE
    allowlist: frozenset[str] = frozenset()
    if allowlist_path.exists():
        lines = allowlist_path.read_text(encoding="utf-8").splitlines()
        allowlist = frozenset(line.strip() for line in lines if line.strip())
    return Dataset(alerts, truth, allowlist)
