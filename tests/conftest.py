from __future__ import annotations

import re

import pytest

from alertweave.cluster import ClusterConfig
from alertweave.evaluate import with_allowlist
from alertweave.gen import Scenario, generate
from alertweave.model import Alert

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean(text: str) -> str:
    """Strip ANSI codes and collapse whitespace before asserting on CLI output."""
    return re.sub(r"\s+", " ", ANSI.sub("", text)).strip()


def make_alert(alert_id: str, time: int, *entities: str, tactic: str = "execution", severity=1):
    return Alert(alert_id, time, "EDR", "rule", tactic, severity, tuple(sorted(entities)))


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    return generate(3)


@pytest.fixture(scope="session")
def config(scenario: Scenario) -> ClusterConfig:
    return with_allowlist(ClusterConfig(), scenario.allowlist)
