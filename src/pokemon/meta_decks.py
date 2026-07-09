"""Canonical meta decklists and proxy opponents for local meta gauntlet testing.

The real Kaggle Simulation Arena does not expose individual match replays — only
aggregate TrueSkill scores.  To gauge our agent's *real-meta* performance
locally, we extract the most-common valid 60-card decklist per archetype from
aggregated Kaggle episode data (see ``scripts/meta_deck_extract.py``) and pilot
each one with a generic heuristic agent.  This lets the local gauntlet measure
our deck/agent against the actual meta instead of only against our own prior
experiments.

The proxy pilot is :class:`pokemon.heuristic.LucarioHeuristicAgent` — it is
deck-agnostic (it picks attacks by damage/weakness, evolves toward highest-HP
Stage 2/Mega, plays supporters when low on cards) so it can reasonably pilot
any deck.  Using the *same* pilot for both our deck and the meta decks isolates
the deck-vs-deck comparison, while using our real MCTS agent vs heuristic meta
proxies approximates the live arena.

Example
-------
>>> from pokemon.meta_decks import META_DECKS, make_meta_proxies
>>> proxies = make_meta_proxies()  # {archetype: heuristic-piloted agent}
"""

from __future__ import annotations

import logging
from pathlib import Path

from pokemon.agent import Agent
from pokemon.heuristic import LucarioHeuristicAgent

logger = logging.getLogger(__name__)

_META_DIR = Path("data/meta_decks")

# Archetype display names mapped to CSV file basenames in data/meta_decks/.
# These are produced by `scripts/meta_deck_extract.py` from Kaggle episode data.
# The 4 official sample decks (always available) come from pokemon.sample_decks.
META_ARCHETYPES: dict[str, str] = {
    "cynthia_garchomp": "fighting_toolbox.csv",  # 55.4% WR — top Fighting deck
    "grimmsnarl": "unknown.csv",  # Marnie's Grimmsnarl ex (Darkness)
    "dragapult_ex": "dragapult_ex.csv",
    "meta_lucario": "mega_lucario_ex.csv",  # other players' Lucario build
}


def _load_csv(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().split("\n") if line.strip()]


def load_meta_decks(meta_dir: str | Path = _META_DIR) -> dict[str, list[int]]:
    """Load all available meta decks from ``meta_dir``.

    Returns a dict ``{archetype_name: [card_ids]}``.  Missing CSVs are silently
    skipped (callers fall back to the 4 sample decks).  Decks are validated
    lazily by the harness when ``battle_start`` is called.
    """
    base = Path(meta_dir)
    decks: dict[str, list[int]] = {}
    for name, filename in META_ARCHETYPES.items():
        path = base / filename
        if path.exists():
            try:
                decks[name] = _load_csv(path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load meta deck %s: %s", path, e)
    return decks


# Lazy-loaded cache of meta decks discovered on disk.
_META_CACHE: dict[str, list[int]] | None = None


def meta_decks() -> dict[str, list[int]]:
    """Return the cached set of meta decks, loading from disk on first use."""
    global _META_CACHE
    if _META_CACHE is None:
        _META_CACHE = load_meta_decks()
    return _META_CACHE


def make_meta_proxy(
    archetype: str,
    deck: list[int] | None = None,
    random_seed: int = 42,
    pilot_cls: type[Agent] = LucarioHeuristicAgent,
) -> Agent:
    """Build a proxy agent piloting a meta deck with a generic heuristic.

    Parameters
    ----------
    archetype:
        Name used for display/logging (does not need to match a CSV).
    deck:
        Card list.  If omitted, pulled from :func:`meta_decks` by ``archetype``.
    pilot_cls:
        Agent class to pilot the deck.  Defaults to the deck-agnostic heuristic.
    """
    if deck is None:
        md = meta_decks()
        if archetype not in md:
            raise KeyError(f"No meta deck for archetype '{archetype}'. Available: {list(md)}")
        deck = md[archetype]
    return pilot_cls(deck=deck, random_seed=random_seed)


def make_meta_proxies(
    random_seed: int = 42,
    pilot_cls: type[Agent] = LucarioHeuristicAgent,
    include_sample: bool = False,
) -> list[tuple[str, Agent]]:
    """Build all meta proxy agents as ``(name, agent)`` tuples for the gauntlet.

    Set ``include_sample=True`` to also include the 4 official sample decks
    (piloted by the same heuristic), giving a broader opponent field.
    """
    proxies: list[tuple[str, Agent]] = []
    for name, deck in meta_decks().items():
        proxies.append((name, pilot_cls(deck=deck, random_seed=random_seed)))
    if include_sample:
        from pokemon.sample_decks import SAMPLE_DECKS

        for name, deck in SAMPLE_DECKS.items():
            proxies.append((f"sample_{name}", pilot_cls(deck=deck, random_seed=random_seed)))
    return proxies
