"""Seeded scenario generator: 14 days of synthetic SOC alerts with ground-truth incidents.

Properties (seed 3 unless noted):
  * 300 hosts and 250 users; background noise picks them with a Zipf-like popularity, so a few
    busy machines and accounts recur and most of the population is quiet.
  * 12 multi-stage intrusions (3 to 25 alerts), six in days 1-7 and six in days 8-14:
    phish -> execution on host A -> credential access -> lateral movement to host B -> exfiltration.
  * About 1,520 benign single alerts plus 40 benign pairs (an admin tool with one rare file hash
    flagged on two hosts) and 268 vulnerability-scanner alerts, each its own ground-truth incident.
  * Hub entities appear in every kind of alert: two domain controllers and two DNS resolvers in
    about 40% of alerts, the proxy egress IP in 25%, the account svc_backup in 30%.
  * Planted cases: two day-9 intrusions that share only the proxy egress IP, one intrusion that
    pivots to a new user and stays connected only through host B, and two alerts that share a
    rare user three days apart.

Entity values are written in mixed raw forms (case, domain suffixes, zero-padded IPs); the
loader normalises them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from .model import DAY, ENTITY_TYPES, Alert, format_time

T0 = int(datetime(2026, 3, 2, tzinfo=UTC).timestamp())
DAYS = 14
NOISE_SINGLES = 1520
BENIGN_PAIRS = 40
SCANNER_ALERTS = 268

DOMAIN_CONTROLLERS = ("dc01", "dc02")
DNS_RESOLVERS = ("10.0.0.53", "10.0.0.54")
PROXY_IP = "203.0.113.10"
SERVICE_ACCOUNT = "svc_backup"
SCANNER_HOST = "vulnscan01"
SCANNER_IP = "10.20.0.5"
ALLOWLIST = (f"host:{SCANNER_HOST}", f"ip:{SCANNER_IP}")

_FIRST = (
    "alex blake casey drew ellis finley gray harper indigo jordan kai logan morgan nico oakley "
    "parker quinn riley sage tatum uma val wren xan yael"
).split()
_LAST = "adler bishop castillo dumont ekstrom farrow garza holm ibarra jansen".split()

# (product, rule, tactic, severity, p_user, p_ip, p_domain, p_hash)
BENIGN_RULES = (
    ("NIDS", "Outdated TLS version negotiated", "defense-evasion", 1, 0.2, 0.5, 0.4, 0.0),
    ("NIDS", "Port sweep from internal host", "discovery", 2, 0.1, 0.4, 0.0, 0.0),
    ("EDR", "Unsigned binary executed", "execution", 2, 0.7, 0.0, 0.0, 0.5),
    ("IdP", "Failed login burst", "credential-access", 2, 0.9, 0.6, 0.0, 0.0),
    ("IdP", "Login from new device", "initial-access", 1, 0.9, 0.5, 0.0, 0.0),
    ("EmailGW", "Bulk marketing mail delivered", "initial-access", 1, 0.8, 0.2, 0.7, 0.0),
    ("EDR", "Removable media mounted", "collection", 1, 0.8, 0.0, 0.0, 0.1),
    ("NIDS", "Suspicious DNS query volume", "command-and-control", 2, 0.1, 0.3, 0.8, 0.0),
)

# stage -> (product, rule, tactic, severity)
STAGES = {
    "phish": ("EmailGW", "Malicious attachment delivered", "initial-access", 3),
    "exec": ("EDR", "Script host spawned shell", "execution", 3),
    "cred": ("EDR", "LSASS memory access", "credential-access", 5),
    "c2": ("NIDS", "Periodic beacon to rare destination", "command-and-control", 3),
    "persist": ("EDR", "Scheduled task created by script", "persistence", 3),
    "discovery": ("EDR", "Bulk account enumeration", "discovery", 2),
    "lateral": ("EDR", "Remote service creation", "lateral-movement", 4),
    "pivot": ("IdP", "First-time admin logon", "privilege-escalation", 4),
    "staging": ("EDR", "Archive created in temp directory", "collection", 3),
    "exfil": ("NIDS", "Large outbound transfer to rare domain", "exfiltration", 5),
}

# (day, alerts, start hour, kind)
INTRUSIONS = (
    (2, 7, 9, "plain"),
    (3, 25, 5, "plain"),
    (4, 4, 10, "plain"),
    (5, 12, 8, "plain"),
    (6, 3, 11, "plain"),
    (7, 9, 8, "plain"),
    (8, 16, 7, "plain"),
    (9, 10, 8, "proxy"),
    (9, 8, 9, "proxy"),
    (12, 14, 7, "pivot"),
    (13, 5, 9, "plain"),
    (14, 6, 10, "plain"),
)
GAP_PAIR_DAYS = (10, 13)


@dataclass
class _Draft:
    time: int
    product: str
    rule: str
    tactic: str
    severity: int
    entities: dict[str, list[str]]
    label: str


@dataclass(frozen=True)
class Scenario:
    records: list[dict]
    alerts: list[Alert]
    truth: dict[str, str]
    allowlist: frozenset[str]
    planted: dict = field(default_factory=dict)


class _Zipf:
    def __init__(self, items: list[str], offset: float = 10.0, exponent: float = 1.0) -> None:
        weights = 1.0 / (np.arange(len(items)) + offset) ** exponent
        self.items = items
        self.cdf = np.cumsum(weights) / weights.sum()

    def draw(self, rng: np.random.Generator) -> str:
        position = int(np.searchsorted(self.cdf, rng.random(), side="right"))
        return self.items[min(position, len(self.items) - 1)]


class _Builder:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.drafts: list[_Draft] = []
        users = [f"{f}.{last}" for f in _FIRST for last in _LAST][:249] + [SERVICE_ACCOUNT]
        hosts = [f"ws-{i:03d}" for i in range(1, 231)] + [f"srv-{i:03d}" for i in range(1, 66)]
        self.users = self._shuffled(users[:-1])
        self.hosts = self._shuffled(hosts)
        # Intrusions and planted cases use the quiet half of each population.
        self.quiet_hosts = self.hosts[150:]
        self.quiet_users = self.users[120:]
        self._taken: set[str] = set()
        self.free_hosts: list[str] = self.hosts
        self.zipf_hosts = _Zipf(self.hosts)
        self.zipf_users = _Zipf(self.users)

    def release_background(self) -> None:
        """After the planted cases took their entities, background alerts use only the rest."""
        self.free_hosts = [h for h in self.hosts if h not in self._taken]
        self.zipf_hosts = _Zipf(self.free_hosts)
        self.zipf_users = _Zipf([u for u in self.users if u not in self._taken])

    def _shuffled(self, items: list[str]) -> list[str]:
        order = np.argsort(self.rng.random(len(items)), kind="stable")
        return [items[i] for i in order]

    def take(self, pool: list[str], count: int = 1) -> list[str]:
        picked = [item for item in pool if item not in self._taken][:count]
        self._taken.update(picked)
        return picked

    def at(self, day: int, hour: float) -> int:
        return T0 + (day - 1) * DAY + int(hour * 3600)

    def business_time(self, day: int) -> int:
        while True:
            second = int(self.rng.integers(0, DAY))
            night = second < 7 * 3600 or second >= 20 * 3600
            if self.rng.random() < (0.3 if night else 1.0):
                return T0 + (day - 1) * DAY + second

    def external_ip(self) -> str:
        octets = self.rng.integers(11, 223, size=4)
        return ".".join(str(int(o)) for o in octets)

    def new_hash(self) -> str:
        return self.rng.integers(0, 256, size=32, dtype=np.uint8).tobytes().hex()

    def external_domain(self) -> str:
        return f"cdn-{int(self.rng.integers(0, 600))}.example.net"

    def add_hubs(self, entities: dict[str, list[str]], proxy: bool = False) -> None:
        if self.rng.random() < 0.40:
            hub = int(self.rng.integers(0, 4))
            if hub < 2:
                entities.setdefault("host", []).append(DOMAIN_CONTROLLERS[hub])
            else:
                entities.setdefault("ip", []).append(DNS_RESOLVERS[hub - 2])
        if proxy or self.rng.random() < 0.25:
            entities.setdefault("ip", []).append(PROXY_IP)
        if self.rng.random() < 0.30:
            entities.setdefault("user", []).append(SERVICE_ACCOUNT)

    def emit(self, time, spec, entities, label, proxy=False) -> None:
        product, rule, tactic, severity = spec
        self.add_hubs(entities, proxy)
        self.drafts.append(_Draft(time, product, rule, tactic, severity, entities, label))

    # -- background -------------------------------------------------------------------------

    def noise(self) -> None:
        for _ in range(NOISE_SINGLES):
            day = int(self.rng.integers(1, DAYS + 1))
            rule = BENIGN_RULES[int(self.rng.integers(0, len(BENIGN_RULES)))]
            self.benign_alert(self.business_time(day), rule, f"SINGLE-{len(self.drafts):05d}")

    def benign_alert(self, time, rule, label, extra: dict[str, list[str]] | None = None) -> None:
        product, name, tactic, severity, p_user, p_ip, p_domain, p_hash = rule
        entities: dict[str, list[str]] = {"host": [self.zipf_hosts.draw(self.rng)]}
        if self.rng.random() < p_user:
            entities["user"] = [self.zipf_users.draw(self.rng)]
        if self.rng.random() < p_ip:
            entities["ip"] = [self.external_ip()]
        if self.rng.random() < p_domain:
            entities["domain"] = [self.external_domain()]
        if self.rng.random() < p_hash:
            entities["hash"] = [self.new_hash()]
        for kind, values in (extra or {}).items():
            entities.setdefault(kind, []).extend(values)
        self.emit(time, (product, name, tactic, severity), entities, label)

    def benign_pairs(self) -> None:
        rule = BENIGN_RULES[2]
        for number in range(1, BENIGN_PAIRS + 1):
            day = int(self.rng.integers(1, DAYS + 1))
            start = self.at(day, 8 + float(self.rng.random()) * 9)
            tool = self.new_hash()
            label = f"PAIR-{number:02d}"
            first, second = self.take_pair_hosts()
            self.benign_alert(start, rule, label, {"hash": [tool]})
            self.drafts[-1].entities["host"] = [first]
            later = start + int(self.rng.integers(600, 5400))
            self.benign_alert(later, rule, label, {"hash": [tool]})
            self.drafts[-1].entities["host"] = [second]

    def take_pair_hosts(self) -> tuple[str, str]:
        count = len(self.free_hosts)
        picks = self.rng.integers(0, count, size=2)
        first = self.free_hosts[int(picks[0])]
        second = self.free_hosts[int(picks[1])]
        if first == second:
            second = self.free_hosts[(int(picks[1]) + 1) % count]
        return first, second

    def scanner(self) -> None:
        spec = ("NIDS", "Vulnerable service probe", "discovery", 2)
        per_day = [
            SCANNER_ALERTS // DAYS + (1 if d < SCANNER_ALERTS % DAYS else 0) for d in range(DAYS)
        ]
        for day, count in enumerate(per_day, start=1):
            order = np.argsort(self.rng.random(len(self.free_hosts)), kind="stable")[:count]
            moment = self.at(day, 2.0)
            for position in order:
                moment += int(self.rng.integers(30, 150))
                entities = {
                    "host": [SCANNER_HOST, self.free_hosts[int(position)]],
                    "ip": [SCANNER_IP],
                }
                self.emit(moment, spec, entities, f"SINGLE-{len(self.drafts):05d}")

    # -- intrusions -------------------------------------------------------------------------

    def intrusions(self) -> dict:
        planted: dict = {"intrusions": [], "proxy_pair": [], "pivot": None}
        for number, (day, size, hour, kind) in enumerate(INTRUSIONS, start=1):
            label = f"INTRUSION-{number:02d}"
            self.intrusion(label, day, size, hour, pivot=kind == "pivot", proxy=kind == "proxy")
            planted["intrusions"].append(label)
            if kind == "proxy":
                planted["proxy_pair"].append(label)
            if kind == "pivot":
                planted["pivot"] = label
        return planted

    def plan(self, size: int, pivot: bool) -> list[tuple[str, str]]:
        """Ordered (stage, phase) steps; phase A runs on host A, phase B on host B."""
        if pivot:
            extra = size - 6
            head = [("phish", "A"), ("exec", "A"), ("cred", "A")]
            steps = head + self.fillers(extra // 2, "A") + [("lateral", "A"), ("pivot", "B")]
            return steps + self.fillers(extra - extra // 2, "B") + [("exfil", "B")]
        core = [("phish", "A"), ("exec", "A"), ("cred", "A"), ("lateral", "A"), ("exfil", "B")]
        if size <= 5:
            return core[:size]
        extra = size - 5
        steps = core[:3] + self.fillers(extra // 2, "A") + [core[3]]
        return steps + self.fillers(extra - extra // 2, "B") + [core[4]]

    def fillers(self, count: int, phase: str) -> list[tuple[str, str]]:
        kinds = ("c2", "discovery", "persist", "staging")
        return [(kinds[int(self.rng.integers(0, len(kinds)))], phase) for _ in range(count)]

    def intrusion(self, label, day, size, hour, pivot: bool, proxy: bool) -> None:
        victim, second_user = self.take(self.quiet_users, 2)
        host_a, host_b = self.take(self.quiet_hosts, 2)
        hashes = tuple(self.new_hash() for _ in range(3))
        domains = tuple(self.external_domain() for _ in range(3))
        ips = tuple(self.external_ip() for _ in range(3))
        clock = self.at(day, hour + float(self.rng.random()))
        pivoted = False
        for position, (stage, phase) in enumerate(self.plan(size, pivot)):
            if position:
                dwell = stage in ("lateral", "exfil")
                clock += int(
                    self.rng.integers(7500, 12600) if dwell else self.rng.integers(300, 2400)
                )
            pivoted = pivoted or stage == "pivot"
            user = second_user if pivoted else victim
            host = host_a if phase == "A" else host_b
            entities = self.stage_entities(
                stage, user, host, (host_a, host_b), hashes, domains, ips
            )
            self.emit(clock, STAGES[stage], entities, label, proxy=proxy)

    @staticmethod
    def stage_entities(stage, user, host, hosts, hashes, domains, ips):
        host_a, host_b = hosts
        hash1, hash2, hash3 = hashes
        sender, c2_domain, exfil_domain = domains
        relay, c2_ip, exfil_ip = ips
        if stage == "phish":
            return {"user": [user], "domain": [sender], "hash": [hash1], "ip": [relay]}
        if stage == "exec":
            return {"host": [host_a], "user": [user], "hash": [hash1]}
        if stage == "cred":
            return {"host": [host_a], "user": [user], "hash": [hash2]}
        if stage == "lateral":
            return {"host": [host_a, host_b], "user": [user], "hash": [hash2]}
        if stage == "pivot":
            return {"host": [host_b], "user": [user]}
        if stage == "exfil":
            return {"host": [host_b], "user": [user], "ip": [exfil_ip], "domain": [exfil_domain]}
        if stage == "c2":
            return {"host": [host], "user": [user], "ip": [c2_ip], "domain": [c2_domain]}
        if stage == "persist":
            return {"host": [host], "user": [user], "hash": [hash3]}
        return {"host": [host], "user": [user]}

    # -- planted gap pair -------------------------------------------------------------------

    def gap_pair(self) -> list[int]:
        (rare_user,) = self.take(self.quiet_users, 1)
        host_one, host_two = self.take(self.quiet_hosts, 2)
        first = self.at(GAP_PAIR_DAYS[0], 11)
        second = self.at(GAP_PAIR_DAYS[1], 11)
        spec_one = ("IdP", "Failed login burst", "credential-access", 2)
        spec_two = ("EDR", "Unsigned binary executed", "execution", 2)
        self.emit(first, spec_one, {"host": [host_one], "user": [rare_user]}, "GAP-1")
        self.emit(second, spec_two, {"host": [host_two], "user": [rare_user]}, "GAP-2")
        return [first, second]


def _raw_host(rng, host: str) -> str:
    roll = rng.random()
    if roll < 0.6:
        return host
    if roll < 0.8:
        return host.upper()
    return f"{host}.corp.example"


def _raw_user(rng, user: str) -> str:
    roll = rng.random()
    if roll < 0.4:
        return user
    if roll < 0.65:
        return f"CORP\\{user.title()}"
    if roll < 0.9:
        return f"{user}@corp.example"
    return f"{user.upper()}@CORP.EXAMPLE"


def _raw_ip(rng, ip: str) -> str:
    roll = rng.random()
    if roll < 0.85:
        return ip
    if roll < 0.95:
        return ".".join(f"{int(o):03d}" for o in ip.split("."))
    return f" {ip} "


def _raw_value(rng, kind: str, value: str) -> str:
    if kind == "host":
        return _raw_host(rng, value)
    if kind == "user":
        return _raw_user(rng, value)
    if kind == "ip":
        return _raw_ip(rng, value)
    roll = rng.random()
    if kind == "hash":
        return value.upper() if roll < 0.3 else value
    return value.upper() if roll < 0.2 else (value + "." if roll < 0.4 else value)


def generate(seed: int = 3) -> Scenario:
    """Build the whole scenario deterministically from ``seed``."""
    builder = _Builder(seed)
    planted = builder.intrusions()
    gap_times = builder.gap_pair()
    builder.release_background()
    builder.noise()
    builder.benign_pairs()
    builder.scanner()

    order = sorted(range(len(builder.drafts)), key=lambda i: (builder.drafts[i].time, i))
    raw_rng = np.random.default_rng(seed + 1)
    records: list[dict] = []
    truth: dict[str, str] = {}
    gap_ids: dict[str, str] = {}
    for rank, index in enumerate(order, start=1):
        draft = builder.drafts[index]
        alert_id = f"A{rank:04d}"
        record: dict = {
            "id": alert_id,
            "time": format_time(draft.time),
            "product": draft.product,
            "rule": draft.rule,
            "tactic": draft.tactic,
            "severity": draft.severity,
        }
        for kind in ENTITY_TYPES:
            values = list(dict.fromkeys(draft.entities.get(kind, ())))
            if values:
                record[kind] = [_raw_value(raw_rng, kind, v) for v in values]
        records.append(record)
        if draft.label.startswith("GAP-"):
            gap_ids[draft.label] = alert_id
            truth[alert_id] = f"SINGLE-{alert_id}"
        elif draft.label.startswith("SINGLE-"):
            truth[alert_id] = f"SINGLE-{alert_id}"
        else:
            truth[alert_id] = draft.label
    planted["gap_pair"] = [gap_ids["GAP-1"], gap_ids["GAP-2"]]
    planted["gap_times"] = gap_times
    planted["benign_pairs"] = BENIGN_PAIRS
    alerts = [Alert.from_record(record) for record in records]
    return Scenario(records, alerts, truth, frozenset(ALLOWLIST), planted)
