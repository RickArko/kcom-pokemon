"""Tests for the deck-validation gate in scripts/build_submission.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_submission.py"


def _load_build_submission_module():
    spec = importlib.util.spec_from_file_location("_build_submission_test", str(_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_build_submission_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_card_db():
    """Build a minimal CardDB that knows about ACE SPEC + basic energy."""
    from pokemon.card_db import CardDB, CardInfo

    def info(cid, name, ace=False, energy=False):
        return CardInfo(
            id=cid,
            name=name,
            card_type=5 if energy else 0,
            card_type_name="basic_energy" if energy else "pokemon",
            hp=0,
            energy_type=6 if energy else None,
            weakness=None,
            resistance=None,
            retreat_cost=0,
            is_basic=not energy,
            is_stage1=False,
            is_stage2=False,
            is_ex=False,
            is_mega_ex=False,
            is_tera=False,
            is_ace_spec=ace,
            evolves_from=None,
        )

    cards = {
        677: info(677, "Riolu"),
        678: info(678, "Mega Lucario ex"),
        1158: info(1158, "Maximum Belt", ace=True),
        1159: info(1159, "Hero's Cape", ace=True),
        6: info(6, "F Energy", energy=True),
    }
    return CardDB(cards=cards, attacks={}, source="fake")


def _patch_db(monkeypatch, mod):
    """Patch CardDB.load so validation uses the fake db (engine-independent).

    ``build_submission`` imports ``CardDB`` locally inside the function, so we
    patch the source module's attribute (resolved at call time).
    """
    import pokemon.card_db as card_db_mod

    fake = _fake_card_db()
    monkeypatch.setattr(
        card_db_mod, "CardDB", type("C", (), {"load": staticmethod(lambda *a, **k: fake)})
    )


def test_valid_deck_passes(tmp_path, monkeypatch):
    mod = _load_build_submission_module()
    _patch_db(monkeypatch, mod)
    deck_path = tmp_path / "deck.csv"
    deck_path.write_text("\n".join(["677"] * 4 + ["678"] * 4 + ["6"] * 52) + "\n")
    deck = mod._validate_deck_or_exit(deck_path, tmp_path)
    assert len(deck) == 60


def test_invalid_ace_spec_exits(tmp_path, monkeypatch):
    mod = _load_build_submission_module()
    _patch_db(monkeypatch, mod)
    deck_path = tmp_path / "deck.csv"
    # 2 ACE SPEC cards -> illegal.
    deck_path.write_text("\n".join(["677"] * 4 + ["1158", "1159"] + ["6"] * 54) + "\n")
    with pytest.raises(SystemExit) as exc:
        mod._validate_deck_or_exit(deck_path, tmp_path)
    assert exc.value.code == 1


def test_wrong_size_exits(tmp_path, monkeypatch):
    mod = _load_build_submission_module()
    _patch_db(monkeypatch, mod)
    deck_path = tmp_path / "deck.csv"
    deck_path.write_text("\n".join(["677"] * 59) + "\n")
    with pytest.raises(SystemExit):
        mod._validate_deck_or_exit(deck_path, tmp_path)
