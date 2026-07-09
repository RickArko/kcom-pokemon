"""Tests for pokemon.stats (Wilson CI, gauntlet aggregation, verdict logic)."""

from __future__ import annotations

from pokemon.stats import (
    Z95,
    GauntletStats,
    MatchResult,
    Verdict,
    decide,
    wilson_high,
    wilson_interval,
    wilson_low,
)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0.5, 100)
    assert 0.4 < lo < 0.5 < hi < 0.6
    # A proportion of 0 with 0 games is undefined -> full range
    assert wilson_interval(0.0, 0) == (0.0, 1.0)


def test_wilson_interval_extremes_robust():
    # 0 wins out of 10 — normal approx would give negative lower bound.
    lo, hi = wilson_interval(0.0, 10)
    assert lo >= 0.0
    assert hi > 0.0
    lo, hi = wilson_interval(1.0, 10)
    assert hi <= 1.0
    assert lo < 1.0


def test_wilson_high_low_consistent():
    p, n = 0.7, 50
    lo, hi = wilson_interval(p, n)
    assert abs(wilson_low(p, n) - lo) < 1e-12
    assert abs(wilson_high(p, n) - hi) < 1e-12


def test_wilson_shrinks_with_more_samples():
    p = 0.6
    w20 = wilson_interval(p, 20)
    w200 = wilson_interval(p, 200)
    assert (w200[1] - w200[0]) < (w20[1] - w20[0])


def test_match_result_record():
    mr = MatchResult("x")
    for w in (0, 1, 1, -1):
        mr.record(w)
    assert mr.wins == 3
    assert mr.draws == 1
    assert mr.decisive == 3
    assert abs(mr.win_rate - 1.0) < 1e-9


def test_gauntlet_stats_wr_and_ci():
    gs = GauntletStats("a", wins=70, losses=30)
    assert abs(gs.win_rate - 0.7) < 1e-9
    lo, hi = gs.ci_low(), gs.ci_high()
    assert lo < 0.7 < hi
    assert gs.margin_of_error() > 0


def test_decide_pass_on_improvement():
    cand = GauntletStats("cand", wins=80, losses=20)
    base = GauntletStats("base", wins=60, losses=40)
    h2h = MatchResult("h2h", wins=30, losses=10, draws=0)
    v = decide(cand, base, h2h, min_head_to_head_wr=0.52, regression_margin=0.03)
    assert isinstance(v, Verdict)
    assert v.passed
    assert v.head_to_head_wr == 0.75


def test_decide_fail_on_head_to_head_regression():
    cand = GauntletStats("cand", wins=70, losses=30)
    base = GauntletStats("base", wins=60, losses=40)
    # Candidate loses the mirror.
    h2h = MatchResult("h2h", wins=20, losses=30, draws=0)
    v = decide(cand, base, h2h, min_head_to_head_wr=0.52)
    assert not v.passed
    assert "head-to-head" in v.reason


def test_decide_fail_on_meta_regression():
    # Candidate wins the mirror but tanks vs the field (overfitting to mirror).
    cand = GauntletStats("cand", wins=40, losses=60)
    base = GauntletStats("base", wins=60, losses=40)
    h2h = MatchResult("h2h", wins=30, losses=20, draws=0)
    v = decide(cand, base, h2h, min_head_to_head_wr=0.52, regression_margin=0.03)
    assert not v.passed
    assert "regression" in v.reason.lower()


def test_decide_no_meta_gate_allows_mirror_only_win():
    cand = GauntletStats("cand", wins=40, losses=60)
    base = GauntletStats("base", wins=60, losses=40)
    h2h = MatchResult("h2h", wins=30, losses=20, draws=0)
    v = decide(cand, base, h2h, min_head_to_head_wr=0.52, require_meta_improvement=False)
    assert v.passed


def test_z95_is_standard():
    assert abs(Z95 - 1.96) < 1e-2
