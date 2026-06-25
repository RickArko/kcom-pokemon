from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pokemon.deck import Deck, build_deck


class TestDeck:
    def test_build_deck_60_cards(self):
        cards = list(range(60))
        deck = build_deck(cards)
        assert len(deck) == 60
        assert deck.cards == sorted(cards)

    def test_build_deck_rejects_wrong_size(self):
        with pytest.raises(ValueError, match="exactly 60"):
            build_deck([1])

    def test_deck_repr(self):
        deck = build_deck(list(range(60)))
        assert "60" in repr(deck)

    def test_deck_csv_roundtrip(self):
        cards = list(range(1, 61))
        deck = build_deck(cards)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp = f.name
        try:
            deck.to_csv(tmp)
            loaded = Deck.from_csv(tmp)
            assert loaded.cards == deck.cards
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_deck_from_csv_missing_id_column(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("wrong_column\n1\n2\n")
            tmp = f.name
        try:
            with pytest.raises(KeyError):
                Deck.from_csv(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)


class TestLoadCardData:
    def test_load_en_card_data(self, tmp_path):
        csv_path = tmp_path / "EN_Card_Data.csv"
        pd.DataFrame(
            {
                "Card ID": [1, 2, 3],
                "Card Name": ["Pikachu", "Charizard", "Mewtwo"],
            }
        ).to_csv(csv_path, index=False)

        from pokemon.data import load_card_data

        result = load_card_data(data_dir=str(tmp_path))
        assert len(result["en"]) == 3
        assert result["jp"] is None
