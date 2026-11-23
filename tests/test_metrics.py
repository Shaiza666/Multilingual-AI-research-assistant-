import math

from mira.evaluate import tune_threshold
from mira.metrics import bootstrap_mean_ci, exact_match, ndcg_at_k, recall_at_k, reciprocal_rank, token_f1


def test_page_recall_mrr_and_ndcg():
    pages = [("d", 1), ("d", 1), ("d", 2), ("d", 9)]
    gold = {("d", 1), ("d", 3)}
    assert recall_at_k(pages, gold, 4) == 0.5
    assert reciprocal_rank(pages, gold) == 1.0
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(pages, gold, 4) == pytest_approx(expected)


def test_mrr_at_rank_three():
    pages = [("d", 2), ("d", 4), ("d", 1)]
    gold = {("d", 1)}
    assert reciprocal_rank(pages, gold) == pytest_approx(1 / 3)
    assert ndcg_at_k(pages, gold, 3) == pytest_approx(0.5)


def test_token_f1_and_exact_match():
    assert token_f1("the cat sat", "cat sat on", "en") == pytest_approx(0.8)
    assert exact_match("The Cat", "cat", "en") == 1.0
    assert token_f1("", "cat", "en") == 0.0


def test_gate_uses_the_gap_midpoint_and_stays_off_when_scores_overlap():
    separated = tune_threshold([(0.9, True), (0.8, True), (0.2, False), (0.1, False)])
    assert separated["enabled"] is True
    assert separated["rule"] == "gap_midpoint"
    assert separated["threshold"] == pytest_approx(0.5)
    overlapping = tune_threshold([(0.9, True), (0.4, True), (0.2, True), (0.5, False)])
    assert overlapping["majority_baseline"] == 0.75
    assert overlapping["enabled"] is False


def test_bootstrap_interval_contains_the_mean():
    bounds = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_resamples=200, seed=0)
    assert bounds["n"] == 4
    assert bounds["low"] <= bounds["mean"] <= bounds["high"]


def pytest_approx(value: float):
    from pytest import approx

    return approx(value)
