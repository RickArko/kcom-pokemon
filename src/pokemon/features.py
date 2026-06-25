from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Columns that carry no predictive signal (IDs, timestamps, metadata).
_DEFAULT_DROP = [
    "id",
    "match_id",
    "game_id",
    "timestamp",
    "round",
    "event",
]

# Default pairwise differences: (col_a, col_b) → col_a minus col_b.
# Captures player vs. opponent advantages in key game-state dimensions.
_DEFAULT_DIFF_PAIRS = [
    ("player_active_pokemon_hp", "opponent_active_pokemon_hp"),
    ("player_bench_count", "opponent_bench_count"),
    ("player_hand_count", "opponent_hand_count"),
    ("player_deck_count", "opponent_deck_count"),
    # prizes_taken: higher is better → opponent_prize_count - player_prize_count
    # gives a positive value when the player is ahead on prizes taken
    ("opponent_prize_count", "player_prize_count"),
]

# Default categorical columns to encode.
_DEFAULT_CAT_COLS: list[str] = []


class StrategyFeatureEngineer(BaseEstimator, TransformerMixin):
    """Drop low-signal columns, derive pairwise difference features, encode categoricals.

    Parameters
    ----------
    drop_cols:
        Column names to remove.  Only columns actually present are dropped.
    diff_pairs:
        Column pairs to subtract: each entry ``(col_a, col_b)`` creates a
        new feature ``"{col_a}_{col_b}"`` equal to ``col_a - col_b``.
        Pairs where either column is absent are skipped silently.
    cat_cols:
        Column names to encode.  If ``None`` (default), any object-dtype
        column not in *drop_cols* is automatically encoded.
    encoding:
        How to encode categorical columns.
        ``"ohe"`` (default) — one-hot encode via ``OneHotEncoder``.
        ``"label"`` — ordinal label encode via ``LabelEncoder``.
        ``"passthrough"`` — keep raw string values unchanged.
    interaction_pairs:
        Pairs of column names to multiply as interaction features.  Each
        entry is ``(col_a, col_b)`` producing column ``"{col_a}_x_{col_b}"``.
        A pair is valid if both names survive *drop_cols* or are produced
        by *diff_pairs*.  Invalid pairs are dropped with a warning.
        Created after diff features and before encoding.
    """

    def __init__(
        self,
        drop_cols: list[str] | None = None,
        diff_pairs: list[tuple[str, str]] | None = None,
        cat_cols: list[str] | None = None,
        encoding: str = "ohe",
        interaction_pairs: list[tuple[str, str]] | None = None,
    ):
        self.drop_cols = drop_cols if drop_cols is not None else _DEFAULT_DROP
        self.diff_pairs = diff_pairs if diff_pairs is not None else _DEFAULT_DIFF_PAIRS
        self.cat_cols = cat_cols
        self.encoding = encoding
        self.interaction_pairs = interaction_pairs

    def fit(self, X: pd.DataFrame, y=None) -> StrategyFeatureEngineer:
        self._drop_cols_ = [c for c in self.drop_cols if c in X.columns]
        self._diff_pairs_ = [
            (a, b) for a, b in self.diff_pairs if a in X.columns and b in X.columns
        ]

        available = {c for c in X.columns if c not in self._drop_cols_}
        available |= {f"{a}_{b}" for a, b in self._diff_pairs_}
        self._interaction_pairs_ = []
        for a, b in self.interaction_pairs or []:
            if a in available and b in available:
                self._interaction_pairs_.append((a, b))
            else:
                warnings.warn(
                    f"interaction_pair ({a!r}, {b!r}) references unknown or "
                    f"dropped columns — skipping.",
                    stacklevel=2,
                )

        if self.cat_cols is not None:
            self._cat_cols_ = [c for c in self.cat_cols if c in X.columns]
        else:
            self._cat_cols_ = [
                c for c in X.columns if X[c].dtype == "object" and c not in self._drop_cols_
            ]

        if self.encoding == "ohe":
            self._encoders_: dict[str, OneHotEncoder | LabelEncoder] = {}
            for c in self._cat_cols_:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                enc.fit(X[[c]])
                self._encoders_[c] = enc
        elif self.encoding == "label":
            self._encoders_ = {}
            for c in self._cat_cols_:
                enc = LabelEncoder()
                enc.fit(X[c])
                self._encoders_[c] = enc
        elif self.encoding == "passthrough":
            self._encoders_ = {}
        else:
            raise ValueError(f"Unknown encoding: {self.encoding!r}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(columns=self._drop_cols_, errors="ignore")
        for a, b in self._diff_pairs_:
            X[f"{a}_{b}"] = X[a] - X[b]
        for a, b in self._interaction_pairs_:
            X[f"{a}_x_{b}"] = X[a] * X[b]

        if self.encoding == "ohe":
            for c in self._cat_cols_:
                enc = self._encoders_[c]
                encoded = enc.transform(X[[c]])
                col_names = [f"{c}_{v}" for v in enc.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=col_names, index=X.index).astype(np.int8)
                X = pd.concat([X.drop(columns=[c]), encoded_df], axis=1)
        elif self.encoding == "label":
            for c in self._cat_cols_:
                enc = self._encoders_[c]
                X[c] = enc.transform(X[c])
                X[c] = X[c].astype(np.int32)
        elif self.encoding == "passthrough":
            pass

        return X


def make_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "winner",
    drop_cols: list[str] | None = None,
    diff_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    engineer = StrategyFeatureEngineer(drop_cols=drop_cols, diff_pairs=diff_pairs)
    y_train = train[target_col].copy()
    X_train = engineer.fit_transform(train.drop(columns=[target_col]))
    X_test = engineer.transform(test.copy())
    return X_train, X_test, y_train
