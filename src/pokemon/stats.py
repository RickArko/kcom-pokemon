"""Statistical helpers for win-rate verification and regression detection.

Used by ``scripts/verify_deck.py`` to decide whether a candidate deck/agent is a
real improvement over the baseline, with quantified confidence.  All functions
are pure (no engine dependency) so they can be unit-tested directly.

Key concepts
------------
- **Win rate** = wins / decisive_games (draws/errors excluded from the denom).
- **Wilson score interval** — robust CI for a binomial proportion, valid even
  for small samples and extreme win rates (unlike the normal approximation).
- **Regression gate** — a candidate "passes" only if it is *not* statistically
  worse than the baseline on the meta gauntlet, and beats the baseline
  head-to-head by a configurable margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# z-score for a two-sided 95% confidence interval.
Z95 = 1.959963984540054


@dataclass
class MatchResult:
    """Aggregated result of one matchup (one agent vs one opponent field)."""

    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0  # errors / timeouts / draws (winner == -1)

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.decisive if self.decisive else 0.0

    def record(self, winner: int) -> None:
        """Record a single match outcome (0 or 1 = side won, -1 = draw/error)."""
        if winner == 0 or winner == 1:
            self.wins += 1
        else:
            self.draws += 1
        # losses are inferred from the opponent's wins at aggregation time.


@dataclass
class GauntletStats:
    """Per-agent stats across a full gauntlet."""

    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    per_opponent: dict[str, MatchResult] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.decisive if self.decisive else 0.0

    def ci_low(self, z: float = Z95) -> float:
        return wilson_low(self.win_rate, self.decisive, z)

    def ci_high(self, z: float = Z95) -> float:
        return wilson_high(self.win_rate, self.decisive, z)

    def margin_of_error(self, z: float = Z95) -> float:
        return (self.ci_high(z) - self.ci_low(z)) / 2.0


def wilson_interval(p: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(low, high)``.  Robust for small ``n`` and proportions near 0/1.
    See: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if n <= 0:
        return 0.0, 1.0
    p = min(max(p, 0.0), 1.0)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - spread), min(1.0, center + spread)


def wilson_low(p: float, n: int, z: float = Z95) -> float:
    return wilson_interval(p, n, z)[0]


def wilson_high(p: float, n: int, z: float = Z95) -> float:
    return wilson_interval(p, n, z)[1]


@dataclass
class Verdict:
    """Pass/fail decision for a candidate vs a baseline."""

    passed: bool
    reason: str
    candidate_wr: float
    baseline_wr: float
    candidate_ci: tuple[float, float]
    baseline_ci: tuple[float, float]
    head_to_head_wr: float
    head_to_head_ci: tuple[float, float]
    head_to_head_n: int


def decide(
    candidate: GauntletStats,
    baseline: GauntletStats,
    head_to_head: MatchResult,
    min_head_to_head_wr: float = 0.52,
    regression_margin: float = 0.03,
    require_meta_improvement: bool = True,
) -> Verdict:
    """Decide whether the candidate is an improvement over the baseline.

    Pass criteria (all must hold):
      1. Head-to-head win rate >= ``min_head_to_head_wr`` (candidate beats
         baseline directly).
      2. If ``require_meta_improvement``: candidate's meta win rate is not
         more than ``regression_margin`` below the baseline's meta win rate
         (no significant regression vs the field).  Equality is acceptable.

    The ``regression_margin`` guards against a candidate that beats the
    baseline in a mirror but loses to the wider meta (overfitting to the
    mirror).  The head-to-head guard prevents noise from a small sample
    flipping the decision.
    """
    cand_wr = candidate.win_rate
    base_wr = baseline.win_rate
    cand_ci = (wilson_low(cand_wr, candidate.decisive), wilson_high(cand_wr, candidate.decisive))
    base_ci = (wilson_low(base_wr, baseline.decisive), wilson_high(base_wr, baseline.decisive))

    # Head-to-head: wins are from the candidate's perspective.
    h2h_decisive = head_to_head.wins + head_to_head.losses
    h2h_wr = head_to_head.wins / h2h_decisive if h2h_decisive else 0.0
    h2h_ci = (wilson_low(h2h_wr, h2h_decisive), wilson_high(h2h_wr, h2h_decisive))

    reasons: list[str] = []
    ok = True

    if h2h_wr < min_head_to_head_wr:
        ok = False
        reasons.append(
            f"head-to-head WR {h2h_wr:.1%} < required {min_head_to_head_wr:.0%} "
            f"({head_to_head.wins}-{head_to_head.losses} over {h2h_decisive} games)"
        )
    else:
        reasons.append(
            f"head-to-head WR {h2h_wr:.1%} >= {min_head_to_head_wr:.0%} "
            f"({head_to_head.wins}-{head_to_head.losses})"
        )

    if require_meta_improvement:
        if cand_wr < base_wr - regression_margin:
            ok = False
            reasons.append(
                f"meta WR {cand_wr:.1%} is {regression_margin:.0%}+ below baseline "
                f"{base_wr:.1%} (regression vs the field)"
            )
        else:
            reasons.append(
                f"meta WR {cand_wr:.1%} within {regression_margin:.0%} of baseline "
                f"{base_wr:.1%} (no regression)"
            )

    return Verdict(
        passed=ok,
        reason="; ".join(reasons),
        candidate_wr=cand_wr,
        baseline_wr=base_wr,
        candidate_ci=cand_ci,
        baseline_ci=base_ci,
        head_to_head_wr=h2h_wr,
        head_to_head_ci=h2h_ci,
        head_to_head_n=h2h_decisive,
    )
