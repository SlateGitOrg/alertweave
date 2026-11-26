import json

import pytest
from typer.testing import CliRunner

from alertweave.cli import app
from alertweave.cluster import cluster_batch
from alertweave.explain import explain_incident
from conftest import clean

runner = CliRunner()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("scenario")
    result = runner.invoke(app, ["gen", "--out", str(path)])
    assert result.exit_code == 0, result.output
    return path


def test_explain_orders_entities_by_weight_then_text(scenario, config):
    result = cluster_batch(scenario.alerts, config)
    incident = next(i for i in result.incidents() if 10 <= i.count <= 30)
    lines = explain_incident(result, incident)
    assert len(lines) >= incident.count - 1 - 5
    for line in lines:
        keys = [(-weight, key) for key, weight in line.shared]
        assert keys == sorted(keys)
        assert line.score >= result.config.tau - 1e-9


def test_gen_writes_the_documented_scenario(data_dir):
    assert len((data_dir / "alerts.jsonl").read_text().splitlines()) == 1989
    truth = [json.loads(x) for x in (data_dir / "truth.jsonl").read_text().splitlines()]
    assert len({row["incident"] for row in truth}) < len(truth)
    assert "host:vulnscan01" in (data_dir / "allowlist.txt").read_text()


def test_eval_prints_both_modes(data_dir):
    result = runner.invoke(app, ["eval", "--data", str(data_dir)])
    text = clean(result.output)
    assert result.exit_code == 0
    assert "naive 1039" in text and "weighted 1039" in text


def test_cluster_lists_incidents(data_dir):
    result = runner.invoke(app, ["cluster", "--data", str(data_dir), "--top", "2"])
    text = clean(result.output)
    assert result.exit_code == 0
    assert "1989 alerts ->" in text and "top entities:" in text


def test_cluster_rejects_unknown_mode(data_dir):
    result = runner.invoke(app, ["cluster", "--data", str(data_dir), "--mode", "bogus"])
    assert result.exit_code != 0


def test_explain_requires_exactly_one_selector(data_dir):
    result = runner.invoke(app, ["explain", "--data", str(data_dir)])
    assert result.exit_code != 0


def test_explain_prints_links_for_the_largest_incident(data_dir):
    result = runner.invoke(app, ["explain", "--data", str(data_dir), "--incident", "1"])
    text = clean(result.output)
    assert result.exit_code == 0
    assert "spanning links:" in text and "->" in text


def test_sweep_reports_the_chosen_setting(data_dir):
    result = runner.invoke(app, ["sweep", "--data", str(data_dir)])
    text = clean(result.output)
    assert result.exit_code == 0
    assert "chosen: tau=" in text and "held-out days 8-14" in text
