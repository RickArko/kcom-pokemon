"""Card database for the Pokemon TCG AI agent.

Provides :class:`CardDB`, a lookup from Card ID to parsed card properties.  It
prefers the authoritative engine data (``cg.api.all_card_data`` /
``all_attack``) when the ``cg`` package is importable, and falls back to parsing
``EN_Card_Data.csv`` so the module stays usable in offline tests.

The engine source is required to map ``Option.attackId`` (seen in observations)
to concrete attack damage/cost — the CSV has no engine attack IDs.  The CSV
source still exposes card properties (HP, type, weakness, retreat, stage) and
parsed move text, which is enough for static deck analysis.

Example
-------
>>> db = CardDB.load()                 # engine if available, else CSV
>>> db.get(678).name                   # Mega Lucario ex
'Mega Lucario ex'
>>> db.attack(983).damage              # Mega Brave
270
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# --- EnergyType int constants (mirror cg.api.EnergyType) ---------------------
COLORLESS = 0
GRASS = 1
FIRE = 2
WATER = 3
LIGHTNING = 4
PSYCHIC = 5
FIGHTING = 6
DARKNESS = 7
METAL = 8
DRAGON = 9
RAINBOW = 10
TEAM_ROCKET = 11

ENERGY_NAME: dict[int, str] = {
    COLORLESS: "Colorless",
    GRASS: "Grass",
    FIRE: "Fire",
    WATER: "Water",
    LIGHTNING: "Lightning",
    PSYCHIC: "Psychic",
    FIGHTING: "Fighting",
    DARKNESS: "Darkness",
    METAL: "Metal",
    DRAGON: "Dragon",
    RAINBOW: "Rainbow",
    TEAM_ROCKET: "Team Rocket",
}

# CSV type-code token -> EnergyType int.  ``●`` is the colorless cost marker.
_TYPE_CODE: dict[str, int | None] = {
    "{C}": COLORLESS,
    "{G}": GRASS,
    "{R}": FIRE,
    "{W}": WATER,
    "{L}": LIGHTNING,
    "{P}": PSYCHIC,
    "{F}": FIGHTING,
    "{D}": DARKNESS,
    "{M}": METAL,
    "竜": DRAGON,  # Dragon glyph used in the EN card CSV
    "{Team Rocket}": TEAM_ROCKET,
    "{A}": None,  # Ancient special-energy token (treated as unknown)
}
_COLORLESS_DOT = "●"

# --- CardType int constants (mirror cg.api.CardType) -------------------------
CT_POKEMON = 0
CT_ITEM = 1
CT_TOOL = 2
CT_SUPPORTER = 3
CT_STADIUM = 4
CT_BASIC_ENERGY = 5
CT_SPECIAL_ENERGY = 6

CARD_TYPE_NAME: dict[int, str] = {
    CT_POKEMON: "pokemon",
    CT_ITEM: "item",
    CT_TOOL: "tool",
    CT_SUPPORTER: "supporter",
    CT_STADIUM: "stadium",
    CT_BASIC_ENERGY: "basic_energy",
    CT_SPECIAL_ENERGY: "special_energy",
}


@dataclass
class AttackInfo:
    """A single attack usable by a Pokemon."""

    attack_id: int
    name: str
    text: str
    damage: int
    energies: list[int]  # EnergyType ints required to use the attack

    @property
    def cost_count(self) -> int:
        return len(self.energies)


@dataclass
class MoveInfo:
    """A move parsed from the CSV (no engine attack ID available)."""

    name: str
    cost: list[int]
    damage: int
    text: str


@dataclass
class CardInfo:
    """Parsed properties of a single card."""

    id: int
    name: str
    card_type: int
    card_type_name: str
    hp: int
    energy_type: int | None
    weakness: int | None
    resistance: int | None
    retreat_cost: int
    is_basic: bool
    is_stage1: bool
    is_stage2: bool
    is_ex: bool
    is_mega_ex: bool
    is_tera: bool
    is_ace_spec: bool
    evolves_from: str | None
    attack_ids: list[int] = field(default_factory=list)
    moves: list[MoveInfo] = field(default_factory=list)

    @property
    def is_pokemon(self) -> bool:
        return self.card_type == CT_POKEMON

    @property
    def is_trainer(self) -> bool:
        return self.card_type in (CT_ITEM, CT_TOOL, CT_SUPPORTER, CT_STADIUM)

    @property
    def is_energy(self) -> bool:
        return self.card_type in (CT_BASIC_ENERGY, CT_SPECIAL_ENERGY)

    @property
    def is_basic_energy(self) -> bool:
        return self.card_type == CT_BASIC_ENERGY

    @property
    def stage(self) -> str:
        if self.is_basic_energy:
            return "basic_energy"
        if self.card_type == CT_SPECIAL_ENERGY:
            return "special_energy"
        if not self.is_pokemon:
            return self.card_type_name
        if self.is_basic:
            return "basic"
        if self.is_stage1:
            return "stage1"
        if self.is_stage2:
            return "stage2"
        return "pokemon"

    @property
    def energy_name(self) -> str | None:
        return ENERGY_NAME.get(self.energy_type) if self.energy_type is not None else None


class CardDB:
    """Card / attack lookup, indexed by Card ID and Attack ID."""

    def __init__(
        self,
        cards: dict[int, CardInfo],
        attacks: dict[int, AttackInfo] | None = None,
        source: str = "unknown",
    ):
        self._cards = cards
        self._attacks = attacks or {}
        self.source = source

    def __contains__(self, card_id: int) -> bool:
        return card_id in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def get(self, card_id: int) -> CardInfo | None:
        return self._cards.get(card_id)

    def __getitem__(self, card_id: int) -> CardInfo:
        try:
            return self._cards[card_id]
        except KeyError as e:
            raise KeyError(f"Unknown Card ID: {card_id}") from e

    def attack(self, attack_id: int) -> AttackInfo | None:
        return self._attacks.get(attack_id)

    @property
    def all_ids(self) -> list[int]:
        return list(self._cards)

    def find_by_name(self, name: str) -> list[CardInfo]:
        lower = name.lower()
        return [c for c in self._cards.values() if lower in c.name.lower()]

    # --- constructors --------------------------------------------------------

    @classmethod
    def load(cls, data_dir: str = "data/raw", use_engine: bool = True) -> CardDB:
        """Load card data, preferring the engine source and falling back to CSV."""
        if use_engine:
            try:
                return cls.from_engine()
            except Exception as e:  # noqa: BLE001 - broad fallback is intentional
                logger.info("Engine card data unavailable (%s); falling back to CSV.", e)
        return cls.from_csv(data_dir)

    @classmethod
    def from_engine(cls) -> CardDB:
        """Build from the authoritative ``cg.api`` data (requires the cg engine)."""
        from cg.api import all_attack, all_card_data  # local import: engine optional

        attacks: dict[int, AttackInfo] = {}
        for a in all_attack():
            energies = [int(e) for e in (a.energies or [])]
            attacks[int(a.attackId)] = AttackInfo(
                attack_id=int(a.attackId),
                name=a.name,
                text=a.text,
                damage=int(a.damage),
                energies=energies,
            )

        cards: dict[int, CardInfo] = {}
        for c in all_card_data():
            ct = int(c.cardType)
            cards[int(c.cardId)] = CardInfo(
                id=int(c.cardId),
                name=c.name,
                card_type=ct,
                card_type_name=CARD_TYPE_NAME.get(ct, "unknown"),
                hp=int(c.hp),
                energy_type=int(c.energyType) if c.energyType is not None else None,
                weakness=int(c.weakness) if c.weakness is not None else None,
                resistance=int(c.resistance) if c.resistance is not None else None,
                retreat_cost=int(c.retreatCost),
                is_basic=bool(c.basic),
                is_stage1=bool(c.stage1),
                is_stage2=bool(c.stage2),
                is_ex=bool(c.ex),
                is_mega_ex=bool(c.megaEx),
                is_tera=bool(c.tera),
                is_ace_spec=bool(c.aceSpec),
                evolves_from=c.evolvesFrom,
                attack_ids=[int(x) for x in (c.attacks or [])],
            )
        logger.info("Loaded %d cards / %d attacks from engine.", len(cards), len(attacks))
        return cls(cards=cards, attacks=attacks, source="engine")

    @classmethod
    def from_csv(cls, data_dir: str = "data/raw") -> CardDB:
        """Build from ``EN_Card_Data.csv`` (no engine attack IDs)."""
        path = Path(data_dir) / "EN_Card_Data.csv"
        if not path.exists():
            # The sim_sample mirror also carries a copy of the card CSV.
            path = Path("data/sim_sample") / "EN_Card_Data.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"EN_Card_Data.csv not found under {data_dir} or data/sim_sample"
            )

        cards: dict[int, CardInfo] = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = int(row["Card ID"])
                stage_col = row["Stage (Pokémon)/Type (Energy and Trainer)"].strip()
                move = _parse_csv_move(row)
                existing = cards.get(cid)
                if existing is None:
                    cards[cid] = _card_info_from_csv_row(cid, row, stage_col)
                if move is not None:
                    cards[cid].moves.append(move)
        logger.info("Loaded %d cards from CSV (%s).", len(cards), path)
        return cls(cards=cards, attacks={}, source="csv")


# --- CSV parsing helpers -----------------------------------------------------


def _parse_type_token(token: str) -> int | None:
    token = token.strip()
    if not token or token == "n/a":
        return None
    return _TYPE_CODE.get(token)


def _parse_cost(cost_str: str) -> list[int]:
    """Parse a CSV cost string like ``"{W}{W}●"`` into EnergyType ints."""
    if not cost_str or cost_str.strip() == "n/a":
        return []
    cost: list[int] = []
    i = 0
    while i < len(cost_str):
        ch = cost_str[i]
        if ch == "{":
            j = cost_str.find("}", i)
            if j == -1:
                break
            e = _TYPE_CODE.get(cost_str[i : j + 1])
            if e is not None:
                cost.append(e)
            i = j + 1
        elif ch == _COLORLESS_DOT:
            cost.append(COLORLESS)
            i += 1
        else:
            i += 1
    return cost


def _parse_damage(damage_str: str) -> int:
    if not damage_str or damage_str.strip() == "n/a":
        return 0
    m = re.match(r"\d+", damage_str.strip())
    return int(m.group()) if m else 0


def _parse_csv_move(row: dict) -> MoveInfo | None:
    name = (row.get("Move Name") or "").strip()
    if not name or name == "n/a":
        return None
    return MoveInfo(
        name=name,
        cost=_parse_cost(row.get("Cost", "")),
        damage=_parse_damage(row.get("Damage", "")),
        text=(row.get("Effect Explanation") or "").strip(),
    )


def _card_info_from_csv_row(cid: int, row: dict, stage_col: str) -> CardInfo:
    is_pokemon = "Pokémon" in stage_col
    is_basic_energy = stage_col == "Basic Energy"
    is_special_energy = stage_col == "Special Energy"
    if is_basic_energy:
        ct = CT_BASIC_ENERGY
    elif is_special_energy:
        ct = CT_SPECIAL_ENERGY
    elif stage_col == "Item":
        ct = CT_ITEM
    elif stage_col == "Supporter":
        ct = CT_SUPPORTER
    elif stage_col == "Pokémon Tool":
        ct = CT_TOOL
    elif stage_col == "Stadium":
        ct = CT_STADIUM
    else:
        ct = CT_POKEMON

    return CardInfo(
        id=cid,
        name=(row["Card Name"] or "").strip(),
        card_type=ct,
        card_type_name=CARD_TYPE_NAME.get(ct, "unknown"),
        hp=_parse_damage(row.get("HP", "")),
        energy_type=_parse_type_token(row.get("Type", "")),
        weakness=_parse_type_token(row.get("Weakness", "")),
        resistance=_parse_type_token(row.get("Resistance (Type)", "")),
        retreat_cost=_parse_damage(row.get("Retreat", "")),
        is_basic=is_pokemon and stage_col == "Basic Pokémon",
        is_stage1=is_pokemon and stage_col == "Stage 1 Pokémon",
        is_stage2=is_pokemon and stage_col == "Stage 2 Pokémon",
        is_ex="ex" in (row["Card Name"] or "").lower(),
        is_mega_ex="Mega" in (row["Card Name"] or "") and "ex" in (row["Card Name"] or "").lower(),
        is_tera="Tera" in (row.get("Category") or ""),
        is_ace_spec=False,  # ACE SPEC is not encoded in the CSV columns reliably
        evolves_from=(row.get("Previous stage") or "").strip() or None
        if (row.get("Previous stage") or "").strip() not in ("", "n/a")
        else None,
    )


# --- deck validation ---------------------------------------------------------


def validate_deck(deck: list[int], db: CardDB) -> tuple[bool, list[str]]:
    """Validate a 60-card deck against the engine's deck-building rules.

    The engine enforces the 4-copy limit by **card name** (basic energy is
    exempt) and allows at most one ACE SPEC card.

    Returns ``(ok, errors)``.
    """
    errors: list[str] = []
    if len(deck) != 60:
        errors.append(f"Deck must have exactly 60 cards, got {len(deck)}.")

    name_counts: dict[str, int] = {}
    ace_spec_count = 0
    for cid in deck:
        info = db.get(cid)
        if info is None:
            errors.append(f"Unknown Card ID {cid}.")
            continue
        if info.is_ace_spec:
            ace_spec_count += 1
        if not info.is_basic_energy:
            name_counts[info.name] = name_counts.get(info.name, 0) + 1

    for name, count in name_counts.items():
        if count > 4:
            errors.append(f"More than 4 copies of '{name}' ({count}).")
    if ace_spec_count > 1:
        errors.append(f"At most 1 ACE SPEC card allowed, got {ace_spec_count}.")

    return (len(errors) == 0, errors)


# --- cached singleton --------------------------------------------------------

_DEFAULT_DB: CardDB | None = None


def get_card_db(data_dir: str = "data/raw", use_engine: bool = True) -> CardDB:
    """Return a process-wide cached :class:`CardDB`."""
    global _DEFAULT_DB
    if _DEFAULT_DB is None:
        _DEFAULT_DB = CardDB.load(data_dir=data_dir, use_engine=use_engine)
    return _DEFAULT_DB
