"""Command line interface: gen, cluster, eval, explain, sweep."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from .cluster import ClusterConfig, ClusterResult, cluster_alerts
from .dataset import Dataset, read_dataset, write_scenario
from .evaluate import (
    TEST_DAYS,
    TUNE_DAYS,
    SweepResult,
    grade,
    ids_in_days,
    naive_config,
    sweep,
    with_allowlist,
)
from .explain import format_explanation
from .gen import generate
from .metrics import Scores
from .model import HOUR, format_time

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False, no_args_is_help=True)

DataOpt = Annotated[Path, typer.Option("--data", help="Directory with the scenario files.")]
TauOpt = Annotated[float | None, typer.Option(help="Link threshold (default from the config).")]
WindowOpt = Annotated[float | None, typer.Option("--window-hours", help="Link window in hours.")]


def _config(dataset: Dataset, tau: float | None, window_hours: float | None) -> ClusterConfig:
    config = with_allowlist(ClusterConfig(), dataset.allowlist)
    if tau is not None:
        config = replace(config, tau=tau)
    if window_hours is not None:
        config = replace(config, link_window=int(window_hours * HOUR))
    return config


def _parse_days(text: str) -> tuple[int, int]:
    first, _, last = text.partition("-")
    try:
        low, high = int(first), int(last or first)
    except ValueError as error:
        raise typer.BadParameter("use a day range such as 8-14") from error
    if low < 1 or high < low:
        raise typer.BadParameter("use a day range such as 8-14")
    return low, high


def _score_row(label: str, scores: Scores, count: int) -> str:
    return (
        f"{label:<9} {count:>6} {scores.incidents:>9} {scores.largest_share * 100:>8.1f}% "
        f"{scores.bcubed_precision:>7.3f} {scores.bcubed_recall:>7.3f} {scores.bcubed_f1:>7.3f} "
        f"{scores.pair_precision:>7.3f} {scores.pair_recall:>7.3f} {scores.ari:>7.3f}"
    )


SCORE_HEADER = (
    f"{'mode':<9} {'alerts':>6} {'incidents':>9} {'largest':>9} "
    f"{'B3-P':>7} {'B3-R':>7} {'B3-F1':>7} {'pair-P':>7} {'pair-R':>7} {'ARI':>7}"
)


@app.command()
def gen(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("data"),
    seed: Annotated[int, typer.Option(help="Generator seed.")] = 3,
) -> None:
    """Write a synthetic scenario: alerts.jsonl plus ground-truth incident labels."""
    scenario = generate(seed)
    write_scenario(scenario, out)
    incidents = len(set(scenario.truth.values()))
    typer.echo(f"wrote {len(scenario.alerts)} alerts and {incidents} true incidents to {out}")


@app.command()
def cluster(
    data: DataOpt = Path("data"),
    mode: Annotated[str, typer.Option(help="weighted or naive.")] = "weighted",
    engine: Annotated[str, typer.Option(help="streaming or batch.")] = "streaming",
    tau: TauOpt = None,
    window_hours: WindowOpt = None,
    top: Annotated[int, typer.Option(help="Incidents to list.")] = 10,
    out: Annotated[Path | None, typer.Option(help="Write all incidents as JSON.")] = None,
) -> None:
    """Group alerts into incidents and list the largest."""
    dataset = read_dataset(data, need_truth=False)
    result = _run(dataset, mode, engine, tau, window_hours)
    incidents = result.incidents()
    typer.echo(f"{len(result.alerts)} alerts -> {len(incidents)} incidents ({mode}, {engine})")
    for incident in incidents[:top]:
        typer.echo(
            f"{incident.id}  {incident.count:>4} alerts  "
            f"{format_time(incident.first_seen)} .. {format_time(incident.last_seen)}  "
            f"sev {incident.max_severity}  {', '.join(incident.tactics)}"
        )
        ranked = [f"{key} ({weight:.1f})" for key, weight in incident.entities[:4]]
        typer.echo(f"      top entities: {', '.join(ranked)}")
    if out is not None:
        payload = [
            {
                "id": inc.id,
                "alerts": list(inc.alert_ids),
                "first_seen": format_time(inc.first_seen),
                "last_seen": format_time(inc.last_seen),
                "max_severity": inc.max_severity,
                "tactics": list(inc.tactics),
                "entities": [[key, round(weight, 4)] for key, weight in inc.entities],
            }
            for inc in incidents
        ]
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(
    dataset: Dataset, mode: str, engine: str, tau: float | None, window_hours: float | None
) -> ClusterResult:
    if mode not in ("weighted", "naive"):
        raise typer.BadParameter("mode must be weighted or naive")
    config = _config(dataset, tau, window_hours)
    if mode == "naive":
        config = naive_config(config)
        if window_hours is not None:
            config = replace(config, link_window=int(window_hours * HOUR))
    return cluster_alerts(dataset.alerts, config, engine)


@app.command("eval")
def evaluate_command(
    data: DataOpt = Path("data"),
    days: Annotated[str, typer.Option(help="Days to grade, e.g. 8-14.")] = "8-14",
    tau: TauOpt = None,
    window_hours: WindowOpt = None,
) -> None:
    """Grade naive and weighted clustering against the ground truth."""
    dataset = read_dataset(data)
    low, high = _parse_days(days)
    config = _config(dataset, tau, window_hours)
    rows = []
    for label, cfg in (("naive", naive_config(config)), ("weighted", config)):
        _, scores = grade(dataset.alerts, dataset.truth, cfg, (low, high))
        rows.append((label, scores))
    count = len(ids_in_days(dataset.alerts, (low, high)))
    typer.echo(f"days {low}-{high}: {count} alerts, history before the range is used for weights")
    typer.echo(SCORE_HEADER)
    for label, scores in rows:
        typer.echo(_score_row(label, scores, count))


@app.command()
def explain(
    data: DataOpt = Path("data"),
    alert: Annotated[
        str | None, typer.Option(help="Explain the incident holding this alert.")
    ] = None,
    incident: Annotated[int | None, typer.Option(help="Rank by size, 1 = largest.")] = None,
    tau: TauOpt = None,
    window_hours: WindowOpt = None,
) -> None:
    """Show the links, shared entities and weights that hold one incident together."""
    if (alert is None) == (incident is None):
        raise typer.BadParameter("give exactly one of --alert and --incident")
    dataset = read_dataset(data, need_truth=False)
    result = _run(dataset, "weighted", "streaming", tau, window_hours)
    incidents = result.incidents()
    if alert is not None:
        if alert not in result.labels:
            raise typer.BadParameter(f"unknown alert id {alert}")
        chosen = next(inc for inc in incidents if inc.id == result.labels[alert])
    else:
        assert incident is not None
        if not 1 <= incident <= len(incidents):
            raise typer.BadParameter(f"incident must be between 1 and {len(incidents)}")
        chosen = incidents[incident - 1]
    typer.echo(format_explanation(result, chosen))


@app.command("sweep")
def sweep_command(
    data: DataOpt = Path("data"),
    mode: Annotated[str, typer.Option(help="weighted or naive.")] = "weighted",
) -> None:
    """Choose tau and the link window on days 1-7, then grade them on days 8-14."""
    if mode not in ("weighted", "naive"):
        raise typer.BadParameter("mode must be weighted or naive")
    dataset = read_dataset(data)
    base = with_allowlist(ClusterConfig(), dataset.allowlist)
    result = sweep(dataset.alerts, dataset.truth, base, mode)
    _print_sweep(result, len(ids_in_days(dataset.alerts, TEST_DAYS)))


def _print_sweep(result: SweepResult, held_out_count: int) -> None:
    taus = sorted({cell.tau for cell in result.cells})
    windows = sorted({cell.window for cell in result.cells})
    typer.echo(f"B-cubed F1 on days {TUNE_DAYS[0]}-{TUNE_DAYS[1]} ({result.mode})")
    typer.echo("tau      " + " ".join(f"{w // HOUR:>5}h" for w in windows))
    for tau in taus:
        values = {c.window: c.f1 for c in result.cells if c.tau == tau}
        typer.echo(f"{tau:<8.1f} " + " ".join(f"{values[w]:>6.3f}" for w in windows))
    best = result.best
    typer.echo(f"chosen: tau={best.tau:g} window={best.link_window // HOUR}h")
    typer.echo(f"held-out days {TEST_DAYS[0]}-{TEST_DAYS[1]}:")
    typer.echo(SCORE_HEADER)
    typer.echo(_score_row(result.mode, result.held_out, held_out_count))


if __name__ == "__main__":
    app()
