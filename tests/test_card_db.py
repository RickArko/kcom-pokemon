"""Tests for pokemon.card_db (CSV parsing, CardInfo, deck validation)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pokemon.card_db import (
    CARD_TYPE_NAME,
    COLORLESS,
    CT_BASIC_ENERGY,
    CT_POKEMON,
    CT_SUPPORTER,
    FIGHTING,
    PSYCHIC,
    WATER,
    AttackInfo,
    CardDB,
    CardInfo,
    _parse_cost,
    _parse_damage,
    validate_deck,
)


def _csv(text: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    f.write(text)
    f.close()
    return f.name


_HEADER = (
    "Card ID,Card Name,Expansion,Collection No.,Stage (Pokémon)/Type (Energy and Trainer),"
    "Rule,Category,Previous stage,HP,Type,Weakness,Resistance (Type),Retreat,Move Name,"
    "Cost,Damage,Effect Explanation\n"
)


class TestCSVParsing:
    def test_parse_basic_energy(self):
        path = _csv(
            _HEADER + "3,Basic {W} Energy,SVE,3,Basic Energy,n/a,n/a,n/a,n/a,{W},,,n/a,,n/a,n/a,\n"
        )
        try:
            db = CardDB.from_csv(Path(path).parent)
        finally:
            Path(path).unlink()
        c = db.get(3)
        assert c is not None
        assert c.is_basic_energy
        assert c.card_type == CT_BASIC_ENERGY
        assert c.energy_type == WATER
        assert c.is_energy

    def test_parse_pokemon_with_two_moves(self):
        row1 = (
            "678,Mega Lucario ex,SVE,678,Stage 1 Pokémon,n/a,n/a,Riolu,340,{F},{P},n/a,2,"
            "Aura Jab,{F},130,Attach energy\n"
        )
        row2 = (
            "678,Mega Lucario ex,SVE,678,Stage 1 Pokémon,n/a,n/a,Riolu,340,{F},{P},n/a,2,"
            "Mega Brave,{F}{F},270,Can't reuse next turn\n"
        )
        path = _csv(_HEADER + row1 + row2)
        try:
            db = CardDB.from_csv(Path(path).parent)
        finally:
            Path(path).unlink()
        c = db.get(678)
        assert c is not None
        assert c.is_pokemon
        assert c.is_stage1
        assert c.is_mega_ex
        assert c.is_ex
        assert c.hp == 340
        assert c.energy_type == FIGHTING
        assert c.weakness == PSYCHIC
        assert c.retreat_cost == 2
        assert c.evolves_from == "Riolu"
        assert len(c.moves) == 2
        assert c.moves[0].name == "Aura Jab"
        assert c.moves[0].damage == 130
        assert c.moves[0].cost == [FIGHTING]
        assert c.moves[1].cost == [FIGHTING, FIGHTING]
        assert c.moves[1].damage == 270

    def test_parse_supporter(self):
        path = _csv(
            _HEADER + "1235,Waitress,SVE,1235,Supporter,n/a,n/a,n/a,n/a,n/a,,n/a,n/a,,n/a,n/a,\n"
        )
        try:
            db = CardDB.from_csv(Path(path).parent)
        finally:
            Path(path).unlink()
        c = db.get(1235)
        assert c is not None
        assert c.card_type == CT_SUPPORTER
        assert c.is_trainer
        assert not c.is_pokemon


class TestCostDamageParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("{W}", [WATER]),
            ("{W}{W}", [WATER, WATER]),
            ("{W}{W}●", [WATER, WATER, COLORLESS]),
            ("●●●", [COLORLESS, COLORLESS, COLORLESS]),
            ("{R}{P}", [2, 5]),
            ("n/a", []),
            ("", []),
        ],
    )
    def test_parse_cost(self, text, expected):
        assert _parse_cost(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("130", 130),
            ("20×", 20),
            ("n/a", 0),
            ("", 0),
            ("100+", 100),
        ],
    )
    def test_parse_damage(self, text, expected):
        assert _parse_damage(text) == expected


class TestCardInfoProperties:
    def _card(self, **kw):
        defaults = dict(
            id=1,
            name="Test",
            card_type=CT_POKEMON,
            card_type_name="pokemon",
            hp=100,
            energy_type=FIGHTING,
            weakness=PSYCHIC,
            resistance=None,
            retreat_cost=2,
            is_basic=False,
            is_stage1=True,
            is_stage2=False,
            is_ex=True,
            is_mega_ex=True,
            is_tera=False,
            is_ace_spec=False,
            evolves_from="Riolu",
        )
        defaults.update(kw)
        return CardInfo(**defaults)

    def test_stage_labels(self):
        assert self._card().stage == "stage1"
        assert self._card(is_basic=True, is_stage1=False).stage == "basic"
        assert self._card(is_stage1=False, is_stage2=True).stage == "stage2"
        assert (
            self._card(card_type=CT_BASIC_ENERGY, is_basic=False, is_stage1=False).stage
            == "basic_energy"
        )

    def test_attack_info(self):
        a = AttackInfo(
            attack_id=983, name="Mega Brave", text="x", damage=270, energies=[FIGHTING, FIGHTING]
        )
        assert a.damage == 270
        assert a.cost_count == 2


class TestValidateDeck:
    def _db_with(self, cards: dict[int, CardInfo]) -> CardDB:
        return CardDB(cards=cards, attacks={}, source="test")

    def _info(self, cid, name, card_type=CT_POKEMON, is_basic_energy=False, ace=False):
        return CardInfo(
            id=cid,
            name=name,
            card_type=CT_BASIC_ENERGY if is_basic_energy else card_type,
            card_type_name=CARD_TYPE_NAME[CT_BASIC_ENERGY if is_basic_energy else card_type],
            hp=0,
            energy_type=None,
            weakness=None,
            resistance=None,
            retreat_cost=0,
            is_basic=False,
            is_stage1=False,
            is_stage2=False,
            is_ex=False,
            is_mega_ex=False,
            is_tera=False,
            is_ace_spec=ace,
            evolves_from=None,
        )

    def test_valid_60_card_deck(self):
        db = self._db_with(
            {1: self._info(1, "Riolu"), 6: self._info(6, "F Energy", is_basic_energy=True)}
        )
        deck = [1] * 4 + [6] * 56
        ok, errors = validate_deck(deck, db)
        assert ok, errors

    def test_rejects_wrong_size(self):
        db = self._db_with({1: self._info(1, "Riolu")})
        ok, errors = validate_deck([1] * 59, db)
        assert not ok
        assert any("60" in e for e in errors)

    def test_rejects_over_4_by_name(self):
        db = self._db_with({1: self._info(1, "Riolu"), 2: self._info(2, "Riolu")})
        # 4 of ID 1 + 2 of ID 2 = 6 cards named "Riolu" -> illegal by name
        deck = [1] * 4 + [2] * 2 + [6] * 54
        # need a basic energy id 6 in db
        db = self._db_with(
            {
                1: self._info(1, "Riolu"),
                2: self._info(2, "Riolu"),
                6: self._info(6, "F", is_basic_energy=True),
            }
        )
        ok, errors = validate_deck(deck, db)
        assert not ok
        assert any("Riolu" in e for e in errors)

    def test_rejects_two_ace_spec(self):
        db = self._db_with(
            {
                1: self._info(1, "Riolu"),
                1158: self._info(1158, "Belt", ace=True),
                1159: self._info(1159, "OtherBelt", ace=True),
                6: self._info(6, "F", is_basic_energy=True),
            }
        )
        deck = [1] * 4 + [1158] * 1 + [1159] * 1 + [6] * 54
        ok, errors = validate_deck(deck, db)
        assert not ok
        assert any("ACE SPEC" in e for e in errors)

    def test_basic_energy_exempt_from_4_limit(self):
        db = self._db_with({6: self._info(6, "F Energy", is_basic_energy=True)})
        ok, errors = validate_deck([6] * 60, db)
        assert ok, errors
