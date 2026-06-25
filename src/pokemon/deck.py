from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Deck:
    cards: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"Deck({len(self.cards)} cards)"

    def to_csv(self, path: str) -> None:
        pd.DataFrame({"Card ID": self.cards}).to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str) -> Deck:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Empty deck file: {path}")
        if lines[0] == "Card ID":
            df = pd.read_csv(path)
            return cls(cards=df["Card ID"].tolist())
        # Header-less raw list (Kaggle submission format)
        if lines[0].isdigit():
            return cls(cards=[int(x) for x in lines])
        # Treat first line as header and validate column name
        df = pd.read_csv(path)
        return cls(cards=df["Card ID"].tolist())


def build_deck(card_ids: list[int]) -> Deck:
    if len(card_ids) != 60:
        raise ValueError(f"Deck must have exactly 60 cards, got {len(card_ids)}")
    return Deck(cards=sorted(card_ids))
