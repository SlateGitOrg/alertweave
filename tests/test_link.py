import pytest

from alertweave.link import RecentIndex, brute_force_links, indexed_links
from alertweave.model import sort_alerts
from alertweave.weights import weight_table
from conftest import make_alert


def test_inverted_index_links_equal_brute_force_on_300_alerts(scenario, config):
    subset = sort_alerts(scenario.alerts)[600:900]
    table = weight_table(subset, config.weights)
    for tau, window in ((2.0, 4 * 3600), (6.0, 4 * 3600), (6.0, 20 * 3600)):
        fast = indexed_links(subset, table, window, tau)
        slow = brute_force_links(subset, table, window, tau)
        assert set(fast) == set(slow)
        assert all(abs(fast[pair] - slow[pair]) < 1e-9 for pair in fast)
    assert len(indexed_links(subset, table, 4 * 3600, 2.0)) > 5


def test_window_boundary_is_inclusive():
    first = make_alert("a1", 0, "host:x")
    on_edge = make_alert("a2", 100, "host:x")
    late = make_alert("a3", 101, "host:x")
    index = RecentIndex(100)
    index.add(first)
    assert index.candidates(on_edge, {"host:x": 2.0}) == [("a1", 2.0)]
    assert index.candidates(late, {"host:x": 2.0}) == []


def test_zero_weight_entities_never_produce_candidates():
    index = RecentIndex(1000)
    index.add(make_alert("a1", 0, "host:dc", "user:u"))
    candidates = index.candidates(
        make_alert("a2", 5, "host:dc", "user:v"), {"host:dc": 0.0, "user:v": 0.0}
    )
    assert candidates == []


def test_scores_sum_shared_weights_and_follow_stream_order():
    index = RecentIndex(1000)
    index.add(make_alert("a1", 0, "host:x", "user:u"))
    index.add(make_alert("a2", 1, "user:u"))
    weights = {"host:x": 2.0, "user:u": 3.0}
    result = index.candidates(make_alert("a3", 2, "host:x", "user:u"), weights)
    assert result == [("a1", 5.0), ("a2", 3.0)]


@pytest.mark.parametrize("tau", [3.0, 5.0])
def test_threshold_is_inclusive(tau):
    alerts = [make_alert("a1", 0, "host:x"), make_alert("a2", 1, "host:x")]
    table = {"a1": {"host:x": 5.0}, "a2": {"host:x": 5.0}}
    assert indexed_links(alerts, table, 10, tau) == {("a1", "a2"): 5.0}
    assert indexed_links(alerts, table, 10, 5.0001) == {}
