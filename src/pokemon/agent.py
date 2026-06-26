from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """Abstract base for Pokemon TCG AI agents.

    The agent receives an observation dict at each decision point and returns a
    list of choice indices.

    The first call will have ``obs["select"] is None`` — the agent must return
    a 60-card deck as a list of Card IDs.  Subsequent calls require choosing from
    the available actions described in the observation.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed

    def __call__(self, obs: dict) -> list[int]:
        if obs.get("select") is None:
            return self._select_deck(obs)
        return self._act(obs)

    @abstractmethod
    def _select_deck(self, obs: dict) -> list[int]:
        """Return a 60-card deck as a list of Card IDs."""

    @abstractmethod
    def _act(self, obs: dict) -> list[int]:
        """Return chosen action indices for this decision point."""


class RandomAgent(Agent):
    """Randomly picks valid actions. Useful as a baseline."""

    def __init__(self, deck: list[int] | None = None, random_seed: int = 42):
        super().__init__(random_seed=random_seed)
        self._deck = deck or [1] * 60

    def _select_deck(self, obs: dict) -> list[int]:
        return self._deck

    def _act(self, obs: dict) -> list[int]:
        min_count = obs.get("minCount", 0)
        max_count = obs.get("maxCount", len(obs.get("options", [])))
        choices = obs.get("options", [])
        if not choices:
            return []
        count = self.rng.integers(min_count, max_count + 1)
        selected = self.rng.choice(choices, size=min(count, len(choices)), replace=False)
        return [int(x) for x in sorted(selected)]


class RuleBasedAgent(Agent):
    """Rule-based agent using heuristic priorities.

    Subclass and override ``_select_deck`` and ``_act`` with domain-specific
    heuristics (attack if lethal, retreat if threatened, etc.).  This base
    implementation delegates deck selection and falls back to a simple
    priority-ordered action picker.
    """

    def __init__(self, deck: list[int] | None = None, random_seed: int = 42):
        super().__init__(random_seed=random_seed)
        self._deck = deck or [1] * 60

    def _select_deck(self, obs: dict) -> list[int]:
        return self._deck

    def _act(self, obs: dict) -> list[int]:
        options = obs.get("options", [])
        if not options:
            return []
        min_count = int(obs.get("minCount", 1) or 1)
        max_count = int(obs.get("maxCount", len(options)) or len(options))
        # Pick the minimum required count (first options) so the baseline always
        # returns a valid selection and never forfeits on multi-select prompts.
        n = min_count if min_count > 0 else (1 if max_count > 0 else 0)
        n = min(n, len(options))
        return [options[i] for i in range(n)]
