"""Opponent archetype classifier.

Identifies the opponent's deck archetype from the cards they play in the early
game (observed via the ``logs`` field in each observation).  Each log entry of
type ``PLAY`` (10), ``EVOLVE`` (12), or ``ATTACH`` (11) with
``playerIndex != yourIndex`` reveals an opponent card ID.  We match accumulated
IDs against known archetype signatures (the 4 official sample decks) and return
a confidence-ranked classification.

Once classified, the agent can adjust its strategy:
- vs Iono (Lightning, weak to Fighting): aggressive attacks (we hit weakness x2)
- vs Abomasnow (Water): race prizes, avoid leaving low-HP basics on bench
- vs Dragapult (Dragon, no weakness): protect bench from Phantom Dive spread
- vs Lucario (Fighting mirror): tempo race, prioritize Mega Brave (270)

Example
-------
>>> clf = OpponentClassifier()
>>> clf.update(state)              # feed each observation's logs
>>> clf.classify()                 # -> ("iono", 0.85)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokemon.state import GameState

# LogType ints (mirror cg.api.LogType)
LOG_PLAY = 10
LOG_EVOLVE = 12
LOG_ATTACH = 11
LOG_CHANGE = 9


# --- archetype signatures ----------------------------------------------------
# Each archetype is defined by its signature card IDs — the Pokemon and key
# trainers that uniquely identify the deck.  Basic energy IDs (1-8) are excluded
# since they appear in every deck.

_ARCHETYPES: dict[str, dict] = {
    "mega_lucario_ex": {
        "signature": {677, 333, 678, 1145, 1158, 1205, 1227, 1235},
        "key_pokemon": {678},  # Mega Lucario ex
        "energy_type": 6,  # Fighting
    },
    "mega_abomasnow_ex": {
        "signature": {721, 722, 723, 1145, 1158, 1205, 1227, 1235},
        "key_pokemon": {723},  # Mega Abomasnow ex
        "energy_type": 3,  # Water
    },
    "dragapult_ex": {
        "signature": {119, 120, 121, 1145, 1205, 1227, 1235},
        "key_pokemon": {121},  # Dragapult ex
        "energy_type": 9,  # Dragon
    },
    "ionos_deck": {
        "signature": {265, 266, 268, 269, 270, 271, 1145, 1205, 1227, 1235},
        "key_pokemon": {269},  # Bellibolt ex
        "energy_type": 4,  # Lightning
    },
}


@dataclass
class Classification:
    """Result of classifying the opponent's deck."""

    archetype: str | None
    confidence: float
    seen_cards: set[int]
    matched_cards: set[int]
    all_scores: dict[str, float] = field(default_factory=dict)

    @property
    def identified(self) -> bool:
        return self.archetype is not None and self.confidence >= 0.5


class OpponentClassifier:
    """Classify the opponent's deck archetype from observed card plays."""

    def __init__(self, your_index: int = 0):
        self.your_index = your_index
        self._seen_cards: set[int] = set()
        self._last_turn_processed: int = -1

    def update(self, state: GameState) -> None:
        """Extract opponent card IDs from the observation's logs.

        Call this with each new ``GameState``.  Logs are incremental (only
        events since the last selection), so each call adds new information.
        """
        for log in state.logs:
            player = log.get("playerIndex")
            if player is None or player == self.your_index:
                continue
            ltype = log.get("type")
            if ltype in (LOG_PLAY, LOG_EVOLVE, LOG_ATTACH, LOG_CHANGE):
                card_id = log.get("cardId")
                if card_id is not None and card_id > 8:  # skip basic energy
                    self._seen_cards.add(int(card_id))
                # EVOLVE also has cardIdTarget (the pre-evolution)
                if ltype == LOG_EVOLVE:
                    target = log.get("cardIdTarget")
                    if target is not None and target > 8:
                        self._seen_cards.add(int(target))
        self._last_turn_processed = state.turn

    def classify(self) -> Classification:
        """Classify the opponent based on accumulated card observations."""
        if not self._seen_cards:
            return Classification(
                archetype=None,
                confidence=0.0,
                seen_cards=set(),
                matched_cards=set(),
            )

        scores: dict[str, float] = {}
        matched: dict[str, set[int]] = {}
        for name, info in _ARCHETYPES.items():
            sig = info["signature"]
            hits = self._seen_cards & sig
            # Confidence = fraction of signature cards seen, weighted by
            # key Pokemon (which are stronger identifiers).
            key_hits = hits & info["key_pokemon"]
            base = len(hits) / len(sig) if sig else 0.0
            key_bonus = (
                0.3 * (len(key_hits) / len(info["key_pokemon"])) if info["key_pokemon"] else 0.0
            )
            scores[name] = min(1.0, base + key_bonus)
            matched[name] = hits

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]
        # Normalize confidence relative to the second-best to measure separation.
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        confidence = best_score * (0.5 + 0.5 * min(1.0, margin * 3))

        return Classification(
            archetype=best_name if best_score > 0 else None,
            confidence=confidence,
            seen_cards=set(self._seen_cards),
            matched_cards=matched.get(best_name, set()),
            all_scores=scores,
        )

    @property
    def seen_cards(self) -> set[int]:
        return set(self._seen_cards)


# --- counter-strategy hints --------------------------------------------------


def counter_strategy(archetype: str | None) -> dict:
    """Return strategy hints for playing against the identified archetype.

    The hints are consumed by the agent to adjust priorities (aggression,
    bench protection, energy targets).  Defaults are for unknown opponents.
    """
    if archetype == "ionos_deck":
        # Iono plays Lightning (weak to Fighting). Our Fighting hits x2.
        return {
            "aggression": 1.3,  # prioritize attacks over setup
            "protect_bench": False,
            "type_advantage": True,
            "note": "Iono: Lightning weak to Fighting — attack aggressively",
        }
    if archetype == "mega_abomasnow_ex":
        # Abomasnow plays Water (weak to Metal, not Fighting). No type advantage.
        # Race prizes; avoid leaving low-HP basics on bench (Hammer-lanche spread).
        return {
            "aggression": 1.0,
            "protect_bench": True,
            "type_advantage": False,
            "note": "Abomasnow: race prizes, protect bench from spread",
        }
    if archetype == "dragapult_ex":
        # Dragon has no weakness. Phantom Dive does 200 + 6 counters on bench.
        return {
            "aggression": 1.1,
            "protect_bench": True,
            "type_advantage": False,
            "note": "Dragapult: protect bench from Phantom Dive spread",
        }
    if archetype == "mega_lucario_ex":
        # Mirror match: Fighting vs Fighting (weak to Psychic, not Fighting).
        # Tempo race — prioritize Mega Brave (270) for OHKOs.
        return {
            "aggression": 1.2,
            "protect_bench": False,
            "type_advantage": False,
            "note": "Lucario mirror: tempo race, prioritize Mega Brave",
        }
    return {
        "aggression": 1.0,
        "protect_bench": False,
        "type_advantage": False,
        "note": "Unknown opponent — balanced play",
    }


# --- archetype deck lists (for opponent_deck_hint in MCTS) -------------------


def archetype_deck_list(archetype: str | None) -> list[int] | None:
    """Return the deck card list for a known archetype (for MCTS opponent hint).

    Returns None for unknown archetypes (the MCTS will use the mirror prior).
    """
    from pokemon.sample_decks import SAMPLE_DECKS

    if archetype and archetype in SAMPLE_DECKS:
        return SAMPLE_DECKS[archetype]
    return None
