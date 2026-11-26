"""Clustering metrics against ground-truth incident labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scores:
    bcubed_precision: float
    bcubed_recall: float
    bcubed_f1: float
    pair_precision: float
    pair_recall: float
    ari: float
    largest_share: float
    incidents: int


def _contingency(pred: Mapping[str, object], truth: Mapping[str, object], ids: list[str]):
    pred_codes: dict[object, int] = {}
    truth_codes: dict[object, int] = {}
    for item in ids:
        pred_codes.setdefault(pred[item], len(pred_codes))
        truth_codes.setdefault(truth[item], len(truth_codes))
    table = np.zeros((len(truth_codes), len(pred_codes)), dtype=np.int64)
    for item in ids:
        table[truth_codes[truth[item]], pred_codes[pred[item]]] += 1
    return table


def _pairs(counts: np.ndarray) -> int:
    return int((counts * (counts - 1) // 2).sum())


def bcubed(pred: Mapping[str, object], truth: Mapping[str, object], ids: Iterable[str]):
    """B-cubed (precision, recall, F1), averaged over items."""
    items = sorted(ids)
    if not items:
        return 1.0, 1.0, 1.0
    table = _contingency(pred, truth, items)
    sq = table.astype(np.float64) ** 2
    precision = float((sq / table.sum(axis=0, keepdims=True)).sum() / len(items))
    recall = float((sq / table.sum(axis=1, keepdims=True)).sum() / len(items))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def pairwise(pred: Mapping[str, object], truth: Mapping[str, object], ids: Iterable[str]):
    """Pairwise (precision, recall) over same-incident pairs; empty denominators count as 1."""
    items = sorted(ids)
    if not items:
        return 1.0, 1.0
    table = _contingency(pred, truth, items)
    both = _pairs(table)
    pred_pairs = _pairs(table.sum(axis=0))
    true_pairs = _pairs(table.sum(axis=1))
    return (both / pred_pairs if pred_pairs else 1.0), (both / true_pairs if true_pairs else 1.0)


def adjusted_rand_index(pred: Mapping[str, object], truth: Mapping[str, object], ids):
    items = sorted(ids)
    total = len(items) * (len(items) - 1) // 2
    if total == 0:
        return 1.0
    table = _contingency(pred, truth, items)
    both = _pairs(table)
    pred_pairs = _pairs(table.sum(axis=0))
    true_pairs = _pairs(table.sum(axis=1))
    expected = pred_pairs * true_pairs / total
    maximum = (pred_pairs + true_pairs) / 2
    if maximum == expected:
        return 1.0
    return (both - expected) / (maximum - expected)


def largest_share(pred: Mapping[str, object], ids: Iterable[str]) -> float:
    counts = Counter(pred[item] for item in ids)
    if not counts:
        return 0.0
    return max(counts.values()) / sum(counts.values())


def score(pred: Mapping[str, object], truth: Mapping[str, object], ids: Iterable[str]) -> Scores:
    items = sorted(ids)
    precision, recall, f1 = bcubed(pred, truth, items)
    pair_p, pair_r = pairwise(pred, truth, items)
    return Scores(
        bcubed_precision=precision,
        bcubed_recall=recall,
        bcubed_f1=f1,
        pair_precision=pair_p,
        pair_recall=pair_r,
        ari=adjusted_rand_index(pred, truth, items),
        largest_share=largest_share(pred, items),
        incidents=len({pred[item] for item in items}),
    )
