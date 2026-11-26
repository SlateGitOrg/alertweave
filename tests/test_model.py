import pytest

from alertweave.model import (
    Alert,
    day_index,
    entity_key,
    format_time,
    normalise_host,
    normalise_ip,
    normalise_user,
    parse_time,
    sort_alerts,
)
from conftest import make_alert


def test_host_becomes_short_lowercase_name():
    assert normalise_host("WS-001.corp.example") == "ws-001"
    assert normalise_host(" srv-017. ") == "srv-017"
    assert normalise_host("10.0.0.5") == "10.0.0.5"


def test_user_forms_collapse_to_one_account():
    forms = ["CORP\\Jane.Smith", "jane.smith@corp.example", "JANE.SMITH@CORP.EXAMPLE", "jane.smith"]
    assert {normalise_user(form) for form in forms} == {"jane.smith"}
    assert normalise_user("jane@other.example") == "jane@other.example"


def test_ip_text_is_canonical():
    assert normalise_ip("010.000.000.053") == "10.0.0.53"
    assert normalise_ip(" 203.0.113.10 ") == "203.0.113.10"
    assert normalise_ip("2001:DB8:0:0:0:0:0:1") == "2001:db8::1"


def test_entity_key_rejects_unknown_type_and_drops_empty_values():
    assert entity_key("hash", " ABCD ") == "hash:abcd"
    assert entity_key("domain", "Evil.Example.") == "domain:evil.example"
    assert entity_key("user", "   ") == ""
    with pytest.raises(ValueError):
        entity_key("mac", "aa:bb")


def test_from_record_merges_raw_variants_and_sorts_entities():
    record = {
        "id": "A1",
        "time": "2026-03-02T01:00:00Z",
        "product": "EDR",
        "rule": "r",
        "tactic": "execution",
        "severity": 3,
        "host": ["WS-001.corp.example", "ws-001"],
        "user": ["CORP\\Bob"],
        "ip": ["010.000.000.001"],
    }
    alert = Alert.from_record(record)
    assert alert.entities == ("host:ws-001", "ip:10.0.0.1", "user:bob")


def test_record_round_trip_is_lossless():
    alert = make_alert("A9", 1772413200, "host:ws-1", "user:bob", "hash:ab", tactic="discovery")
    assert Alert.from_record(alert.to_record()) == alert


def test_time_parsing_and_formatting_are_utc():
    assert parse_time("2026-03-02T00:00:00Z") == 1772409600
    assert parse_time("2026-03-02T02:00:00+02:00") == 1772409600
    assert format_time(1772409600) == "2026-03-02T00:00:00Z"


def test_sort_breaks_time_ties_by_id_and_rejects_duplicates():
    a, b, c = make_alert("b", 5), make_alert("a", 5), make_alert("c", 1)
    assert [x.id for x in sort_alerts([a, b, c])] == ["c", "a", "b"]
    with pytest.raises(ValueError):
        sort_alerts([a, a])


def test_day_index_counts_from_the_origin_midnight():
    origin = 1772409600 + 3600
    assert day_index(origin, origin) == 1
    assert day_index(origin + 86400 * 7 + 5, origin) == 8
