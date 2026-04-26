import pytest
from fairlens.metrics.fairness import (
    demographic_parity,
    equalized_odds_proxy,
    aggregate_bias_score,
)


def test_demographic_parity_equal():
    rates = {"group_a": 0.3, "group_b": 0.3}
    result = demographic_parity(rates)
    assert result["disparity"] == pytest.approx(0.0)


def test_demographic_parity_unequal():
    rates = {"group_a": 0.1, "group_b": 0.5, "group_c": 0.3}
    result = demographic_parity(rates)
    assert result["disparity"] == pytest.approx(0.4)
    assert result["max_rate"] == pytest.approx(0.5)
    assert result["min_rate"] == pytest.approx(0.1)


def test_demographic_parity_single_group():
    rates = {"only": 0.5}
    result = demographic_parity(rates)
    assert result["disparity"] == 0.0


def test_equalized_odds_proxy():
    pos = {"a": 0.8, "b": 0.6}
    neg = {"a": 0.1, "b": 0.3}
    result = equalized_odds_proxy(pos, neg)
    assert result["positive_rate_gap"] == pytest.approx(0.2)
    assert result["negative_rate_gap"] == pytest.approx(0.2)
    assert result["combined_gap"] == pytest.approx(0.2)


def test_equalized_odds_no_gap():
    pos = {"a": 0.5, "b": 0.5}
    neg = {"a": 0.2, "b": 0.2}
    result = equalized_odds_proxy(pos, neg)
    assert result["combined_gap"] == pytest.approx(0.0)


def test_aggregate_bias_score_zero():
    metrics = {
        "tox": {"disparity": 0.0},
        "sent": {"disparity": 0.0},
    }
    assert aggregate_bias_score(metrics) == pytest.approx(0.0)


def test_aggregate_bias_score_mixed():
    metrics = {
        "tox": {"disparity": 0.4},
        "sent": {"disparity": 0.2},
    }
    score = aggregate_bias_score(metrics)
    assert score == pytest.approx(0.3)


def test_aggregate_bias_score_capped():
    metrics = {
        "tox": {"disparity": 1.5},
    }
    score = aggregate_bias_score(metrics)
    assert score <= 1.0
