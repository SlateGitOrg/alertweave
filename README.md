# alertweave

[![CI](https://github.com/SlateGitOrg/alertweave/actions/workflows/ci.yml/badge.svg)](https://github.com/SlateGitOrg/alertweave/actions/workflows/ci.yml)

![alertweave](assets/hero.png)

alertweave groups SOC alerts into incidents through the entities they share (hosts, users, IPs,
file hashes, domains), weighting each shared entity by how rare it has been over a trailing
window. Domain controllers, DNS resolvers, the proxy egress IP and a backup service account that
appear in a large fraction of alerts get weight zero, so they cannot chain unrelated intrusions
into one incident. It is a Python and NumPy library with a `typer` command line, run here on a
synthetic scenario with ground truth. All data is synthetic; there is no live SIEM.

## The problem

A SOC receives about 150 alerts a day from EDR, network IDS, the identity provider and the email
gateway, but analysts work incidents, not alerts. Correlating on "any shared host, user or IP"
lets hub entities connect nearly everything into one giant incident, while strict rules split a
single intrusion into a dozen tickets that different analysts pick up without seeing the chain.
The question is which shared entities are actually evidence that two alerts belong together.

## How it works

```mermaid
flowchart LR
    A[Alerts JSONL] --> B[Normalise<br/>hosts, users, IPs, hashes, domains]
    B --> C[Trailing-window weights<br/>prior x ln N/df, hubs = 0]
    C --> D[Inverted index<br/>entity -> recent alerts]
    D --> E[Pair score >= tau]
    E --> F[Union-find incidents<br/>time-bounded]
    F --> G[Grade against ground truth<br/>B-cubed, ARI]
```

1. `alertweave.model` lowercases entities, strips `CORP\` and `@corp.example` user forms, shortens
   hostnames and canonicalises IP text.
2. `alertweave.weights` computes, for each alert, `weight = prior[type] x ln(N / df)` over the
   alerts of the previous 7 days plus that alert. Entities whose document frequency is above the
   99th percentile of all entity frequencies, or that are on the infrastructure allowlist, get 0.
   Only earlier alerts in (time, id) order are read.
3. `alertweave.link` keeps an inverted index from entity to alerts inside the link window
   (default 4 hours) and only compares alerts that share a positively weighted entity. A pair is
   linked when the summed weight of its shared entities reaches `tau` (default 6).
4. `alertweave.cluster` merges linked alerts with union-find (path compression, union by rank).
   An incident stops accepting links once its last alert is more than 24 hours old (a safeguard:
   see the design decisions for when it can fire). A streaming engine and a batch engine give
   the same partition.
5. `alertweave.explain` lists each spanning link with the shared entities and their weights at
   link time; `alertweave.metrics` and `alertweave.evaluate` grade and tune.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
alertweave gen --out data            # 14 days, seed 3
alertweave cluster --data data       # largest incidents
alertweave eval --data data          # naive vs weighted on days 8-14
alertweave sweep --data data         # choose tau and window on days 1-7
alertweave explain --data data --incident 1
pytest -q
```

## Example output

Generated with `alertweave gen --out data` (seed 3): 1,989 alerts over 14 days, 12 multi-stage
intrusions, 40 benign pairs sharing a rare hash, and hubs in a large share of alerts.
`alertweave eval --data data` grades days 8-14 with weights computed from all earlier history:

```text
days 8-14: 1039 alerts, history before the range is used for weights
mode      alerts incidents   largest    B3-P    B3-R   B3-F1  pair-P  pair-R     ARI
naive       1039        48     95.4%   0.047   1.000   0.089   0.001   1.000   0.000
weighted    1039       942      1.5%   0.980   1.000   0.990   0.935   1.000   0.967
```

The naive baseline is one fixed configuration: any shared entity links, no weighting, and a 24 hour
link window (the incident gap). It is not tuned, and `alertweave sweep --mode naive` grades this
same configuration. Given a 1 hour window instead, naive would keep its largest component at 8.0%
but still reach only B-cubed F1 0.496 (precision 0.331), because hubs still link unrelated alerts
inside each hour; the 95.4% collapse is the behaviour of the 24 hour baseline.

`alertweave eval` uses the defaults (`tau` 6, 4 hour window). `alertweave sweep --data data`
chooses those same values on days 1-7 only and then grades them on days 8-14 (the `weighted`
row below matches the eval row above):

```text
B-cubed F1 on days 1-7 (weighted)
tau          1h     2h     4h     8h
2.0       0.921  0.892  0.821  0.720
3.0       0.954  0.937  0.900  0.821
4.0       0.961  0.946  0.919  0.853
5.0       0.971  0.969  0.971  0.946
6.0       0.979  0.981  0.996  0.990
7.0       0.978  0.982  0.997  0.997
8.0       0.975  0.978  0.992  0.992
10.0      0.967  0.968  0.974  0.975
chosen: tau=6 window=4h
held-out days 8-14:
mode      alerts incidents   largest    B3-P    B3-R   B3-F1  pair-P  pair-R     ARI
weighted    1039       942      1.5%   0.980   1.000   0.990   0.935   1.000   0.967
```

`alertweave explain --data data --alert A1749` shows one intrusion; the proxy IP
`203.0.113.10` is listed with weight 0.00 and never contributes to a link:

```text
INC-A1749: 5 alerts, 2026-03-14T09:11:43Z to 2026-03-14T16:18:25Z, max severity 5
tactics: initial-access, execution, credential-access, lateral-movement, exfiltration
spanning links: 4
A1749 -> A1750  score 15.59  [Malicious attachment delivered -> Script host spawned shell]
    hash:d72d722d32a450130cbafdf074703a1d7ad469c2d0df46f3c854e015851141e5  9.35
    user:morgan.jansen  6.24
    ip:203.0.113.10  0.00
A1750 -> A1751  score 12.69  [Script host spawned shell -> LSASS memory access]
    host:ws-027  6.86
    user:morgan.jansen  5.83
A1750 -> A1781  score 11.97  [Script host spawned shell -> Remote service creation]
    host:ws-027  6.42
    user:morgan.jansen  5.55
    ip:203.0.113.10  0.00
A1781 -> A1805  score 12.20  [Remote service creation -> Large outbound transfer to rare domain]
    host:ws-205  6.87
    user:morgan.jansen  5.33
    ip:203.0.113.10  0.00
```

## Design decisions

- **Trailing window, never the future.** Weights use only alerts at or before the one being
  scored, so a live deployment would produce the same weights. The cost is a noisy start: in the
  first day or two there are too few alerts for a stable rarity estimate. The alert being scored
  counts in its own `N` and `df` (a smoothing choice, so a never-seen entity has `df` 1 and a
  finite weight); the 99th-percentile cutoff is recomputed over every distinct entity in the
  window for each alert, which is fine at the scenario's 2,000 alerts but is not built for very
  large streams.
- **Percentile hub cutoff plus an allowlist.** A frequency percentile finds hubs without a
  curated list, but with few distinct entities in the window it lets some hubs through (ties at
  the cutoff are not above it). The allowlist covers known infrastructure such as the scanner.
- **Time-bounded union-find.** Unioning forever would let a rare user seen days apart merge two
  incidents. An incident that has been quiet for more than 24 hours stops accepting links. The
  cost is that a very slow intrusion is split. At the defaults this rule never fires: a link needs
  both alerts inside the 4 hour link window, which is shorter than the 24 hour gap, so the
  window already keeps the three-days-apart pair separate. The gap rule only matters when the
  link window is widened beyond 24 hours, and a test widens it that way to check the rule on the
  scenario.
- **Streaming and batch agree.** Both decide links in (time, id) order and apply the same gap
  rule; the batch engine computes candidate links first so parameter sweeps can reuse weights.
- **Tune on days 1-7, grade on days 8-14.** `sweep` picks `tau` and the window by B-cubed F1 on
  the first week only. Settings within 0.002 of the best are treated as ties and resolved to the
  lowest `tau`, then the shortest window, because differences that small are noise. On seed 3
  that picks `tau` 6 (0.996) over 7 (0.997). Two caveats: the planted pivot intrusion sits on day
  12, so it is held out, and its weakest spanning link scores 6.10, only 0.10 above `tau` 6
  (a single shared host weighs about 1.1 x ln(N/2)); the pivot test therefore checks this setting
  rather than offering independent evidence, and `tau` 7 would split that intrusion while scoring
  0.996 B-cubed F1 on days 8-14. The 0.002 tolerance and the
  lowest-`tau` tie-break were chosen with that pivot case in mind, so read the held-out 0.990 as
  mildly optimistic.
- **Sum of weights, not a count of entities.** One very rare hash can link two alerts, while two
  moderately common hosts may not. The threshold `tau` is therefore in weight units, not a count.

## Testing

60 tests, run with `pytest -q`: union-find behaviour, weights (a 1-in-1,000 entity outweighs a
500-in-1,000 one, hubs get zero, appended future alerts change nothing earlier), inverted-index
links equal to brute force on a 300-alert subset, input-order independence (hypothesis),
streaming equal to batch, hand-computed B-cubed, pairwise and ARI values, the planted cases
(proxy-only pair separate, pivot intrusion together, alerts three days apart apart), held-out
B-cubed F1 of the weighted and naive modes, the sweep choosing the defaults and grading the
fixed naive baseline, the gap rule on its own, and the command line.

## Project layout

```text
src/alertweave/
  model.py      alerts and entity normalisation
  weights.py    trailing-window rarity weights
  link.py       inverted index and brute-force reference
  cluster.py    union-find, streaming and batch engines, incident summaries
  explain.py    spanning links with shared entities
  metrics.py    B-cubed, pairwise, adjusted Rand index
  evaluate.py   grading and the tau/window sweep
  gen.py        seeded scenario generator
  dataset.py    scenario files on disk
  cli.py        gen, cluster, eval, explain, sweep
tests/          pytest suite
```

## License

MIT
