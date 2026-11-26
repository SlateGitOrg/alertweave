import math

from alertweave.model import DAY
from alertweave.weights import TrailingWeights, WeightConfig, flat_table, weight_table
from conftest import make_alert


def _stream(count: int, rare_last: bool = True):
    alerts = []
    for i in range(count):
        entities = [f"host:filler{i}"]
        if i % 2 == 1:
            entities.append("host:common")
        if rare_last and i == count - 1:
            entities.append("host:rare")
        alerts.append(make_alert(f"a{i:04d}", i, *entities))
    return alerts


def test_entity_in_one_of_a_thousand_outweighs_one_in_five_hundred():
    config = WeightConfig(hub_percentile=100.0)
    table = weight_table(_stream(1000), config)
    last = table["a0999"]
    assert math.isclose(last["host:rare"], 1.1 * math.log(1000 / 1))
    assert math.isclose(last["host:common"], 1.1 * math.log(1000 / 500))
    assert last["host:rare"] > last["host:common"]


def test_hubs_above_the_percentile_get_zero_weight():
    table = weight_table(_stream(1000), WeightConfig(hub_percentile=99.0))
    last = table["a0999"]
    assert last["host:common"] == 0.0
    assert last["host:rare"] > 0.0


def test_domain_controllers_in_the_scenario_get_zero_weight(scenario, config):
    ordered = sorted(scenario.alerts, key=lambda a: (a.time, a.id))
    table = weight_table(ordered, config.weights)
    # Start after the first 600 alerts: early in the stream the window holds few distinct
    # entities, and the 99th-percentile cutoff does not exceed the controllers' counts (ties at
    # the cutoff are not above it), so they only reach weight 0 once the window has filled.
    seen = 0
    for alert in ordered[600:]:
        for key in alert.entities:
            if key in ("host:dc01", "host:dc02"):
                assert table[alert.id][key] == 0.0
                seen += 1
    assert seen > 100


def test_appending_a_future_alert_never_changes_earlier_weights():
    alerts = _stream(300)
    config = WeightConfig(hub_percentile=95.0)
    before = weight_table(alerts, config)
    future = make_alert("z9999", 10_000, "host:common", "host:rare", "host:new")
    after = weight_table([*alerts, future], config)
    assert {key: after[key] for key in before} == before


def test_allowlisted_entities_have_no_weight():
    config = WeightConfig(allowlist=frozenset({"host:scanner"}), hub_percentile=100.0)
    tracker = TrailingWeights(config)
    tracker.observe(make_alert("a1", 0, "host:x"))
    weights = tracker.observe(make_alert("a2", 1, "host:scanner", "host:y"))
    assert weights["host:scanner"] == 0.0
    assert weights["host:y"] > 0.0


def test_alerts_outside_the_window_do_not_count():
    config = WeightConfig(window=DAY, hub_percentile=100.0)
    tracker = TrailingWeights(config)
    tracker.observe(make_alert("a1", 0, "host:x"))
    tracker.observe(make_alert("a2", 10, "host:y"))
    weights = tracker.observe(make_alert("a3", DAY + 5, "host:x"))
    # a1 (t=0) is older than the one-day window at a3's time, so it has expired; a2 (t=10) is
    # still inside. The window holds a2 and a3 (N=2) and host:x is only in a3 (df=1).
    assert math.isclose(weights["host:x"], 1.1 * math.log(2 / 1))


def test_type_priors_scale_the_weight():
    config = WeightConfig(hub_percentile=100.0)
    tracker = TrailingWeights(config)
    tracker.observe(make_alert("a1", 0, "host:a", "hash:a"))
    weights = tracker.observe(make_alert("a2", 1, "host:b", "hash:b", "user:b"))
    assert math.isclose(weights["hash:b"] / weights["user:b"], 1.5)
    assert math.isclose(weights["host:b"] / weights["user:b"], 1.1)


def test_flat_table_weights_everything_one():
    table = flat_table([make_alert("a1", 0, "host:a", "user:b")])
    assert table == {"a1": {"host:a": 1.0, "user:b": 1.0}}
