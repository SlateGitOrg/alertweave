from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from alertweave.cluster import (
    DEFAULT_LINK_WINDOW,
    DEFAULT_MAX_GAP,
    DEFAULT_TAU,
    ClusterConfig,
    cluster_batch,
    cluster_streaming,
)
from alertweave.evaluate import TEST_DAYS, grade, ids_in_days, naive_config, sweep
from alertweave.metrics import score
from conftest import make_alert


def _members(result, label):
    return [alert_id for alert_id, lab in result.labels.items() if lab == label]


def _by_truth(scenario, result):
    groups: dict[str, list[str]] = {}
    for alert_id, truth in scenario.truth.items():
        groups.setdefault(truth, []).append(result.labels[alert_id])
    return groups


def test_naive_mode_collapses_most_alerts_into_one_component(scenario, config):
    result = cluster_batch(scenario.alerts, naive_config(config))
    assert result.incidents()[0].count >= 0.6 * len(scenario.alerts)


def test_weighted_mode_keeps_the_largest_incident_small(scenario, config):
    result = cluster_batch(scenario.alerts, config)
    assert result.incidents()[0].count <= 30


def test_held_out_scores_beat_naive_by_a_wide_margin(scenario, config):
    _, weighted = grade(scenario.alerts, scenario.truth, config, TEST_DAYS)
    _, naive = grade(scenario.alerts, scenario.truth, naive_config(config), TEST_DAYS)
    assert weighted.bcubed_f1 >= 0.90
    assert naive.bcubed_f1 <= 0.20


def test_each_intrusion_is_mostly_in_one_predicted_incident(scenario, config):
    result = cluster_batch(scenario.alerts, config)
    groups = _by_truth(scenario, result)
    for name in scenario.planted["intrusions"]:
        labels = groups[name]
        top = max(labels.count(label) for label in set(labels))
        assert top >= 0.9 * len(labels), name


def test_proxy_only_intrusions_stay_separate(scenario, config):
    result = cluster_batch(scenario.alerts, config)
    groups = _by_truth(scenario, result)
    first, second = scenario.planted["proxy_pair"]
    assert not set(groups[first]) & set(groups[second])


def test_pivot_intrusion_is_a_single_incident(scenario, config):
    result = cluster_batch(scenario.alerts, config)
    assert len(set(_by_truth(scenario, result)[scenario.planted["pivot"]])) == 1


def test_alerts_three_days_apart_never_share_an_incident(scenario, config):
    result = cluster_batch(scenario.alerts, config)
    first, second = scenario.planted["gap_pair"]
    assert result.labels[first] != result.labels[second]


def test_gap_rule_blocks_a_link_to_a_stale_incident():
    config = ClusterConfig(tau=1.0, link_window=10**9, max_gap=100)
    alerts = [
        make_alert("a1", 0, "host:x"),
        make_alert("a2", 50, "host:y"),
        make_alert("a3", 200, "host:x", "host:y"),
    ]
    table = {a.id: dict.fromkeys(a.entities, 2.0) for a in alerts}
    result = cluster_batch(alerts, config, table)
    assert result.labels["a3"] != result.labels["a1"]


def test_streaming_equals_batch(scenario, config):
    stream = cluster_streaming(scenario.alerts, config)
    batch = cluster_batch(scenario.alerts, config)
    assert stream.partition() == batch.partition()
    assert [(link.earlier, link.later) for link in stream.links] == [
        (link.earlier, link.later) for link in batch.links
    ]


def test_streaming_equals_batch_in_naive_mode(scenario, config):
    naive = naive_config(config)
    one = cluster_streaming(scenario.alerts, naive)
    two = cluster_batch(scenario.alerts, naive)
    assert one.partition() == two.partition()


ENTITIES = ["host:a", "host:b", "user:u", "user:v", "ip:1.1.1.1", "hash:" + "0" * 64]


@settings(max_examples=60, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(0, 40),
            st.lists(st.sampled_from(ENTITIES), min_size=1, max_size=3, unique=True),
        ),
        min_size=2,
        max_size=25,
    ),
    seed=st.randoms(use_true_random=False),
)
def test_partition_is_independent_of_input_order(rows, seed):
    alerts = [make_alert(f"a{i:03d}", t * 60, *ents) for i, (t, ents) in enumerate(rows)]
    config = ClusterConfig(tau=0.5, link_window=900, max_gap=1800)
    shuffled = list(alerts)
    seed.shuffle(shuffled)
    for engine in (cluster_streaming, cluster_batch):
        assert engine(alerts, config).partition() == engine(shuffled, config).partition()


def test_incident_summary_fields(scenario, config):
    result = cluster_batch(scenario.alerts, config)
    biggest = result.incidents()[0]
    by_id = {a.id: a for a in result.alerts}
    members = [by_id[i] for i in biggest.alert_ids]
    assert biggest.first_seen == min(a.time for a in members)
    assert biggest.last_seen == max(a.time for a in members)
    assert biggest.max_severity == max(a.severity for a in members)
    weights = [w for _, w in biggest.entities]
    assert weights == sorted(weights, reverse=True)


def test_sweep_tunes_on_early_days_and_holds_out_the_rest(scenario, config):
    result = sweep(scenario.alerts, scenario.truth, config)
    assert result.held_out.bcubed_f1 >= 0.90
    assert result.best.tau in {cell.tau for cell in result.cells}
    top = max(cell.f1 for cell in result.cells)
    chosen = next(
        c for c in result.cells if (c.tau, c.window) == (result.best.tau, result.best.link_window)
    )
    assert chosen.f1 >= top - 0.002


def test_sweep_picks_the_defaults_that_eval_uses(scenario, config):
    result = sweep(scenario.alerts, scenario.truth, config)
    assert result.best.tau == DEFAULT_TAU
    assert result.best.link_window == DEFAULT_LINK_WINDOW


def test_naive_sweep_is_the_single_untuned_baseline(scenario, config):
    swept = sweep(scenario.alerts, scenario.truth, config, mode="naive")
    assert len(swept.cells) == 1
    assert swept.best == naive_config(config)
    _, graded = grade(scenario.alerts, scenario.truth, naive_config(config), TEST_DAYS)
    assert swept.held_out == graded
    assert swept.held_out.largest_share >= 0.6
    assert swept.held_out.bcubed_f1 <= 0.20


def test_default_window_is_inside_the_gap_so_the_gap_rule_is_a_safeguard():
    assert DEFAULT_LINK_WINDOW < DEFAULT_MAX_GAP


def test_gap_rule_alone_separates_the_three_day_pair(scenario, config):
    wide = replace(config, link_window=10**9)
    result = cluster_batch(scenario.alerts, wide)
    first, second = scenario.planted["gap_pair"]
    assert result.labels[first] != result.labels[second]


def test_grading_scores_only_the_requested_days(scenario, config):
    result, scores = grade(scenario.alerts, scenario.truth, config, (8, 14))
    ids = ids_in_days(scenario.alerts, (8, 14))
    assert scores == score(result.labels, scenario.truth, ids)
    assert 0 < len(ids) < len(scenario.alerts)
