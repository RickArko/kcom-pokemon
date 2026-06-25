"""End-to-end integration test with structured synthetic battle data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from pokemon.features import StrategyFeatureEngineer
from pokemon.models import StackingEnsemble, save_submission


@pytest.fixture
def synthetic_battle_data():
    """Generate separable synthetic Pokemon TCG battle-state data.

    The winner is made partially separable by HP advantage and prize progress
    so the pipeline has signal to learn from.
    """
    n_train, n_test = 300, 100
    rng = np.random.default_rng(42)

    deck_names = ["Charizard", "Mewtwo", "Pikachu", "Gardevoir", "Lugia"]
    pokemon_names = ["Charizard", "Mewtwo", "Pikachu", "Gardevoir", "Lugia", "Blastoise"]

    def _make_df(n: int, has_target: bool = False) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "id": range(n),
                "first_player": rng.integers(0, 2, n),
                "turns": rng.integers(1, 20, n),
                "prizes_left": rng.integers(0, 6, n),
                "deck_name": rng.choice(deck_names, n),
                "opponent_deck_name": rng.choice(deck_names, n),
                "player_active_pokemon_name": rng.choice(pokemon_names, n),
                "opponent_active_pokemon_name": rng.choice(pokemon_names, n),
                "player_active_pokemon_hp": rng.integers(10, 200, n),
                "opponent_active_pokemon_hp": rng.integers(10, 200, n),
                "player_bench_count": rng.integers(0, 5, n),
                "opponent_bench_count": rng.integers(0, 5, n),
                "player_hand_count": rng.integers(0, 10, n),
                "opponent_hand_count": rng.integers(0, 10, n),
                "player_deck_count": rng.integers(0, 60, n),
                "opponent_deck_count": rng.integers(0, 60, n),
                "player_prize_count": rng.integers(0, 6, n),
                "opponent_prize_count": rng.integers(0, 6, n),
            }
        )
        if has_target:
            # Winner partially determined by HP advantage and prizes taken
            hp_adv = df["player_active_pokemon_hp"] > df["opponent_active_pokemon_hp"]
            prize_adv = df["player_prize_count"] >= df["opponent_prize_count"]
            df["winner"] = ((hp_adv | prize_adv) & rng.random(n) > 0.3).astype(int)
        return df

    train = _make_df(n_train, has_target=True)
    test = _make_df(n_test, has_target=False)
    return train, test


class TestPipeline:
    def test_feature_engineering_drops_and_diffs(self, synthetic_battle_data):
        train, test = synthetic_battle_data
        engineer = StrategyFeatureEngineer(
            diff_pairs=[
                ("player_active_pokemon_hp", "opponent_active_pokemon_hp"),
                ("player_bench_count", "opponent_bench_count"),
            ],
            cat_cols=[],
        )
        X_train = engineer.fit_transform(train.drop(columns=["winner"]))
        X_test = engineer.transform(test)

        # Diff features are created
        assert "player_active_pokemon_hp_opponent_active_pokemon_hp" in X_train.columns
        assert "player_bench_count_opponent_bench_count" in X_train.columns
        # ID column is dropped
        assert "id" not in X_train.columns
        assert "id" not in X_test.columns
        # Shape and column consistency
        assert len(X_train) == len(train)
        assert len(X_test) == len(test)
        assert X_train.columns.tolist() == X_test.columns.tolist()

    def test_feature_engineering_with_ohe_categoricals(self, synthetic_battle_data):
        train, test = synthetic_battle_data
        engineer = StrategyFeatureEngineer(
            diff_pairs=[],
            cat_cols=["deck_name"],
            encoding="ohe",
        )
        X_train = engineer.fit_transform(train.drop(columns=["winner"]))
        X_test = engineer.transform(test)

        assert "deck_name" not in X_train.columns
        # At least one OHE column created
        ohe_cols = [c for c in X_train.columns if c.startswith("deck_name_")]
        assert len(ohe_cols) > 0
        assert X_train.columns.tolist() == X_test.columns.tolist()

    def test_interaction_features(self, synthetic_battle_data):
        train, test = synthetic_battle_data
        _str_cols = [
            "deck_name",
            "opponent_deck_name",
            "player_active_pokemon_name",
            "opponent_active_pokemon_name",
        ]
        engineer = StrategyFeatureEngineer(
            drop_cols=["id", "match_id", "game_id", "timestamp", "round", "event"] + _str_cols,
            diff_pairs=[
                ("player_active_pokemon_hp", "opponent_active_pokemon_hp"),
            ],
            cat_cols=[],
            interaction_pairs=[
                ("turns", "player_active_pokemon_hp_opponent_active_pokemon_hp"),
            ],
        )
        X_train = engineer.fit_transform(train.drop(columns=["winner"]))
        X_test = engineer.transform(test)

        assert "turns_x_player_active_pokemon_hp_opponent_active_pokemon_hp" in X_train.columns
        assert X_train.columns.tolist() == X_test.columns.tolist()

    def test_ensemble_learns_from_data(self, synthetic_battle_data):
        train, test = synthetic_battle_data
        _str_cols = [
            "deck_name",
            "opponent_deck_name",
            "player_active_pokemon_name",
            "opponent_active_pokemon_name",
        ]
        engineer = StrategyFeatureEngineer(
            drop_cols=["id", "match_id", "game_id", "timestamp", "round", "event"] + _str_cols,
            diff_pairs=[
                ("player_active_pokemon_hp", "opponent_active_pokemon_hp"),
                ("player_bench_count", "opponent_bench_count"),
            ],
            cat_cols=[],
        )
        y_train = train["winner"]
        X_train = engineer.fit_transform(train.drop(columns=["winner"]))
        X_test = engineer.transform(test)

        base_models = [
            ("lr1", LogisticRegression(max_iter=1000, random_state=42)),
            ("lr2", LogisticRegression(max_iter=1000, random_state=42, C=0.5)),
        ]
        ensemble = StackingEnsemble(base_models)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ensemble.fit(X_train, y_train, cv, X_test)

        assert len(ensemble.valid_scores_) == 3
        assert ensemble.overall_oof_score_ is not None
        assert len(ensemble.fold_models_) == 3

        preds = ensemble.predict(X_test)
        assert len(preds) == len(test)
        assert set(preds) <= {0, 1}

    def test_full_pipeline_with_submission(self, synthetic_battle_data, tmp_path):
        train, test = synthetic_battle_data
        _str_cols = [
            "deck_name",
            "opponent_deck_name",
            "player_active_pokemon_name",
            "opponent_active_pokemon_name",
        ]
        engineer = StrategyFeatureEngineer(
            drop_cols=["id", "match_id", "game_id", "timestamp", "round", "event"] + _str_cols,
            diff_pairs=[("player_active_pokemon_hp", "opponent_active_pokemon_hp")],
            cat_cols=[],
        )
        y_train = train["winner"]
        X_train = engineer.fit_transform(train.drop(columns=["winner"]))
        X_test = engineer.transform(test)

        base_models = [
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
        ensemble = StackingEnsemble(base_models)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ensemble.fit(X_train, y_train, cv, X_test)

        preds = ensemble.predict(X_test)
        out = tmp_path / "sub.csv"
        save_submission(test["id"], preds, output_path=str(out))
        df = pd.read_csv(out)
        assert list(df.columns) == ["id", "winner"]
        assert len(df) == len(test)
        assert all(c in {0, 1} for c in df["winner"])

    def test_save_load_and_predict_consistent(self, synthetic_battle_data, tmp_path):
        train, test = synthetic_battle_data
        _str_cols = [
            "deck_name",
            "opponent_deck_name",
            "player_active_pokemon_name",
            "opponent_active_pokemon_name",
        ]
        engineer = StrategyFeatureEngineer(
            drop_cols=["id", "match_id", "game_id", "timestamp", "round", "event"] + _str_cols,
            diff_pairs=[("player_active_pokemon_hp", "opponent_active_pokemon_hp")],
            cat_cols=[],
        )
        y_train = train["winner"]
        X_train = engineer.fit_transform(train.drop(columns=["winner"]))
        X_test = engineer.transform(test)

        base_models = [
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
        ensemble = StackingEnsemble(base_models)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ensemble.fit(X_train, y_train, cv, X_test)

        model_path = tmp_path / "ensemble.joblib"
        ensemble.save(model_path)

        loaded = StackingEnsemble.load(model_path)
        preds_orig = ensemble.predict(X_test)
        preds_loaded = loaded.predict(X_test)
        np.testing.assert_array_equal(preds_orig, preds_loaded)
