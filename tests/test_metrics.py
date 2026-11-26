import math

from alertweave.metrics import adjusted_rand_index, bcubed, largest_share, pairwise, score

# Six alerts. Truth: {1,2,3} {4,5} {6}. Predicted: {1,2} {3,4,5} {6}.
TRUTH = {"1": "x", "2": "x", "3": "x", "4": "y", "5": "y", "6": "z"}
PRED = {"1": "p", "2": "p", "3": "q", "4": "q", "5": "q", "6": "r"}
IDS = list(TRUTH)


def test_bcubed_matches_hand_computation():
    # Precision per item: 1,1 (cluster p is pure), 3 in q holds {3,4,5} -> 1/3, 4,5 -> 2/3, 6 -> 1.
    precision = (1 + 1 + 1 / 3 + 2 / 3 + 2 / 3 + 1) / 6
    # Recall per item: 1,2 -> 2/3, 3 -> 1/3, 4,5 -> 1, 6 -> 1.
    recall = (2 / 3 + 2 / 3 + 1 / 3 + 1 + 1 + 1) / 6
    p, r, f1 = bcubed(PRED, TRUTH, IDS)
    assert math.isclose(p, precision, abs_tol=1e-12)
    assert math.isclose(r, recall, abs_tol=1e-12)
    assert math.isclose(f1, 2 * precision * recall / (precision + recall), abs_tol=1e-12)


def test_pairwise_matches_hand_computation():
    # Same-incident pairs: truth 3+1 = 4, predicted 1+3 = 4, in both: (1,2) and (4,5) = 2.
    p, r = pairwise(PRED, TRUTH, IDS)
    assert math.isclose(p, 0.5, abs_tol=1e-12)
    assert math.isclose(r, 0.5, abs_tol=1e-12)


def test_adjusted_rand_index_matches_hand_computation():
    # n=6 -> 15 pairs. index 2, predicted pairs 4, true pairs 4: expected 16/15, max 4.
    expected = 4 * 4 / 15
    ari = (2 - expected) / (4 - expected)
    assert math.isclose(adjusted_rand_index(PRED, TRUTH, IDS), ari, abs_tol=1e-12)
    assert math.isclose(ari, (2 - 16 / 15) / (4 - 16 / 15), abs_tol=1e-12)


def test_identical_partitions_score_one():
    s = score(TRUTH, TRUTH, IDS)
    assert (s.bcubed_f1, s.ari, s.pair_precision, s.pair_recall) == (1.0, 1.0, 1.0, 1.0)
    assert s.incidents == 3


def test_everything_in_one_cluster_has_zero_ari_and_full_share():
    one = dict.fromkeys(IDS, "all")
    assert largest_share(one, IDS) == 1.0
    assert math.isclose(adjusted_rand_index(one, TRUTH, IDS), 0.0, abs_tol=1e-12)


def test_scoring_a_subset_ignores_other_alerts():
    s = score(PRED, TRUTH, ["1", "2", "6"])
    assert s.bcubed_f1 == 1.0
    assert s.incidents == 2


def test_empty_selection_is_vacuously_perfect():
    assert bcubed(PRED, TRUTH, []) == (1.0, 1.0, 1.0)
