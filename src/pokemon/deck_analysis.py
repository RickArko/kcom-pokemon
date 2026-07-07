"""Static deck analysis: archetype classification, composition stats, card frequency.

Provides functions to classify a 60-card deck list into known archetypes,
compute composition summaries (energy/pokemon/trainer counts), and aggregate
card frequency across multiple decks.

Example
-------
>>> from pokemon.deck_analysis import classify_deck, deck_summary
>>> deck = [677, 677, 677, 677, 678, 678, 678, 678, 6] + [6] * 51
>>> classify_deck(deck, db)
'mega_lucario_ex'
>>> deck_summary(deck, db)["energy_pct"]
61.7
"""

from __future__ import annotations

from typing import Any

from pokemon.card_db import CT_BASIC_ENERGY, CT_POKEMON, CT_SPECIAL_ENERGY, CardDB

# ── Archetype signatures ─────────────────────────────────────────────────────
# Key Pokemon and energy type for each known archetype.

_ARCHETYPE_SIGNATURES: dict[str, dict[str, Any]] = {
    "mega_lucario_ex": {
        "key_pokemon": {678},  # Mega Lucario ex
        "signature_pokemon": {677, 678},  # Riolu, Mega Lucario ex
        "energy_type": 6,  # Fighting
        "energy_id": 6,
    },
    "mega_abomasnow_ex": {
        "key_pokemon": {723},  # Mega Abomasnow ex
        "signature_pokemon": {721, 722, 723},  # Kyogre, Snover, Mega Abomasnow ex
        "energy_type": 3,  # Water
        "energy_id": 3,
    },
    "dragapult_ex": {
        "key_pokemon": {121},  # Dragapult ex
        "signature_pokemon": {119, 120, 121},  # Dreepy, Drakloak, Dragapult ex
        "energy_type": 9,  # Dragon
        "energy_id": 2,  # Psychic energy (Dragapult uses psychic + darkness)
    },
    "ionos_deck": {
        "key_pokemon": {269},  # Bellibolt ex
        "signature_pokemon": {265, 266, 268, 269, 270, 271},
        "energy_type": 4,  # Lightning
        "energy_id": 4,
    },
}


def classify_deck(card_ids: list[int], db: CardDB) -> str:
    """Classify a 60-card deck list into a known archetype.

    Uses key Pokemon presence and energy type to match against known
    archetype signatures.  Falls back to ``"unknown"`` if no signature matches.

    Parameters
    ----------
    card_ids:
        60-card deck as Card ID list.
    db:
        Card database for name/id lookups.

    Returns
    -------
    str:
        Archetype name (e.g. ``"mega_lucario_ex"``) or ``"unknown"``.
    """
    card_set = set(card_ids)
    energy_ids = _energy_ids(card_ids, db)

    best_match: str | None = None
    best_score = 0.0

    for archetype, sig in _ARCHETYPE_SIGNATURES.items():
        key_hits = len(card_set & sig["key_pokemon"])
        sig_hits = len(card_set & sig["signature_pokemon"])

        has_energy = sig.get("energy_id") in energy_ids

        # Score: key Pokemon presence is weighted heavily
        score = 0.0
        if key_hits > 0:
            score += 0.6 * key_hits
        if sig_hits > 0:
            score += 0.2 * (sig_hits / len(sig["signature_pokemon"]))
        if has_energy:
            score += 0.2

        if score > best_score:
            best_score = score
            best_match = archetype

    # Require a minimum score to avoid false positives
    if best_score >= 0.5:
        return best_match  # type: ignore[return-value]

    # Fallback: generic "fighting_toolbox" if Fighting energy + basic Pokemon
    if 6 in energy_ids:
        return "fighting_toolbox"

    return "unknown"


def _energy_ids(card_ids: list[int], db: CardDB) -> set[int]:
    """Return the set of distinct basic energy Card IDs in a deck."""
    seen: set[int] = set()
    for cid in card_ids:
        info = db.get(cid)
        if info and info.card_type == CT_BASIC_ENERGY:
            seen.add(cid)
    return seen


# ── Deck composition stats ───────────────────────────────────────────────────


def deck_summary(card_ids: list[int], db: CardDB) -> dict[str, Any]:
    """Compute composition statistics for a deck.

    Returns
    -------
    dict with keys:
        total_cards, n_pokemon, n_trainer, n_energy,
        energy_pct, trainer_pct, pokemon_pct,
        n_basic, n_stage1, n_stage2, n_ex, n_mega_ex,
        energy_types (list of energy type names),
        unique_pokemon_ids, archetype
    """
    n_total = len(card_ids)
    n_pokemon = 0
    n_trainer = 0
    n_energy = 0
    n_basic = 0
    n_stage1 = 0
    n_stage2 = 0
    n_ex = 0
    n_mega_ex = 0
    energy_types: set[int] = set()
    pokemon_ids: set[int] = set()

    for cid in card_ids:
        info = db.get(cid)
        if info is None:
            continue
        if info.card_type == CT_POKEMON:
            n_pokemon += 1
            pokemon_ids.add(cid)
            if info.is_basic:
                n_basic += 1
            if info.is_stage1:
                n_stage1 += 1
            if info.is_stage2:
                n_stage2 += 1
            if info.is_ex:
                n_ex += 1
            if info.is_mega_ex:
                n_mega_ex += 1
            if info.energy_type is not None:
                energy_types.add(info.energy_type)
        elif info.card_type in (CT_BASIC_ENERGY, CT_SPECIAL_ENERGY):
            n_energy += 1
            if info.energy_type is not None:
                energy_types.add(info.energy_type)
        else:
            n_trainer += 1

    return {
        "total_cards": n_total,
        "n_pokemon": n_pokemon,
        "n_trainer": n_trainer,
        "n_energy": n_energy,
        "pokemon_pct": round(n_pokemon / n_total * 100, 1) if n_total else 0.0,
        "trainer_pct": round(n_trainer / n_total * 100, 1) if n_total else 0.0,
        "energy_pct": round(n_energy / n_total * 100, 1) if n_total else 0.0,
        "n_basic": n_basic,
        "n_stage1": n_stage1,
        "n_stage2": n_stage2,
        "n_ex": n_ex,
        "n_mega_ex": n_mega_ex,
        "energy_types": sorted(energy_types),
        "unique_pokemon_ids": sorted(pokemon_ids),
        "archetype": classify_deck(card_ids, db),
    }


# ── Card frequency across decks ──────────────────────────────────────────────


def card_frequencies(
    deck_lists: list[list[int]],
    db: CardDB,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Compute aggregate card frequency across multiple deck lists.

    Returns
    -------
    list of dicts: ``{card_id, name, card_type, total_copies, n_decks, pct_decks}``
    sorted by total copies descending.
    """
    from collections import Counter

    total_decks = len(deck_lists)
    copy_counter: Counter[int] = Counter()
    deck_counter: Counter[int] = Counter()

    for deck in deck_lists:
        seen = set()
        for cid in deck:
            copy_counter[cid] += 1
            if cid not in seen:
                deck_counter[cid] += 1
                seen.add(cid)

    rows: list[dict[str, Any]] = []
    for cid, total_copies in copy_counter.most_common(top_n):
        info = db.get(cid)
        rows.append(
            {
                "card_id": cid,
                "name": info.name if info else f"#{cid}",
                "card_type": info.card_type_name if info else "unknown",
                "stage": info.stage if info else "unknown",
                "total_copies": total_copies,
                "n_decks": deck_counter[cid],
                "pct_decks": round(deck_counter[cid] / total_decks * 100, 1)
                if total_decks
                else 0.0,
            }
        )

    return rows
