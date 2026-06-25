"""Unit tests for core components using synthetic battle data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from pokemon.features import StrategyFeatureEngineer, make_features
from pokemon.models import StackingEnsemble, save_submission


class TestStrategyFeatureEngineer:
    def test_drops_id_columns(self):
        X = pd.DataFrame(
            {
                "id": [1, 2],
                "match_id": [100, 200],
                "game_id": [1, 2],
                "player_active_pokemon_hp": [100, 80],
                "opponent_active_pokemon_hp": [60, 90],
            }
        )
        engineer = StrategyFeatureEngineer()
        out = engineer.fit_transform(X)
        assert "id" not in out.columns
        assert "match_id" not in out.columns
        assert "game_id" not in out.columns
        assert "player_active_pokemon_hp" in out.columns

    def test_adds_diff_features(self):
        X = pd.DataFrame(
            {
                "player_active_pokemon_hp": [100.0, 80.0],
                "opponent_active_pokemon_hp": [60.0, 90.0],
                "player_bench_count": [3.0, 2.0],
                "opponent_bench_count": [2.0, 4.0],
            }
        )
        engineer = StrategyFeatureEngineer(
            diff_pairs=[
                ("player_active_pokemon_hp", "opponent_active_pokemon_hp"),
                ("player_bench_count", "opponent_bench_count"),
            ]
        )
        out = engineer.fit_transform(X)
        assert "player_active_pokemon_hp_opponent_active_pokemon_hp" in out.columns
        assert "player_bench_count_opponent_bench_count" in out.columns
        np.testing.assert_array_almost_equal(
            out["player_active_pokemon_hp_opponent_active_pokemon_hp"], [40.0, -10.0]
        )

    def test_adds_interaction_features(self):
        X = pd.DataFrame(
            {
                "player_active_pokemon_hp": [100.0, 80.0],
                "opponent_active_pokemon_hp": [60.0, 90.0],
                "turns": [5.0, 10.0],
            }
        )
        engineer = StrategyFeatureEngineer(
            diff_pairs=[("player_active_pokemon_hp", "opponent_active_pokemon_hp")],
            interaction_pairs=[("turns", "player_active_pokemon_hp_opponent_active_pokemon_hp")],
        )
        out = engineer.fit_transform(X)
        assert "turns_x_player_active_pokemon_hp_opponent_active_pokemon_hp" in out.columns
        np.testing.assert_array_almost_equal(
            out["turns_x_player_active_pokemon_hp_opponent_active_pokemon_hp"],
            [200.0, -100.0],
        )

    def test_invalid_interaction_pair_dropped(self):
        X = pd.DataFrame(
            {
                "player_active_pokemon_hp": [100.0, 80.0],
                "opponent_active_pokemon_hp": [60.0, 90.0],
            }
        )
        engineer = StrategyFeatureEngineer(
            interaction_pairs=[("nonexistent_col", "player_active_pokemon_hp")]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = engineer.fit_transform(X)
        assert "nonexistent_col_x_player_active_pokemon_hp" not in out.columns

    def test_skips_missing_diff_pair_columns(self):
        X = pd.DataFrame({"player_active_pokemon_hp": [100.0, 80.0], "turns": [5.0, 3.0]})
        # opponent_active_pokemon_hp is missing — pair should be silently skipped
        engineer = StrategyFeatureEngineer(
            diff_pairs=[("player_active_pokemon_hp", "opponent_active_pokemon_hp")]
        )
        out = engineer.fit_transform(X)
        assert "player_active_pokemon_hp_opponent_active_pokemon_hp" not in out.columns

    def test_consistent_columns_train_test(self):
        train = pd.DataFrame(
            {
                "id": [1, 2],
                "player_active_pokemon_hp": [100.0, 80.0],
                "opponent_active_pokemon_hp": [60.0, 90.0],
                "player_bench_count": [3.0, 2.0],
                "opponent_bench_count": [2.0, 4.0],
            }
        )
        test = pd.DataFrame(
            {
                "id": [3, 4],
                "player_active_pokemon_hp": [70.0, 110.0],
                "opponent_active_pokemon_hp": [80.0, 50.0],
                "player_bench_count": [1.0, 3.0],
                "opponent_bench_count": [3.0, 2.0],
            }
        )
        engineer = StrategyFeatureEngineer(
            diff_pairs=[
                ("player_active_pokemon_hp", "opponent_active_pokemon_hp"),
                ("player_bench_count", "opponent_bench_count"),
            ]
        )
        X_train = engineer.fit_transform(train)
        X_test = engineer.transform(test)
        assert list(X_train.columns) == list(X_test.columns)

    def test_ohe_encoding(self):
        X = pd.DataFrame(
            {
                "deck_name": ["Charizard", "Mewtwo", "Pikachu"],
                "player_active_pokemon_hp": [100.0, 80.0, 90.0],
            }
        )
        engineer = StrategyFeatureEngineer(cat_cols=["deck_name"], encoding="ohe")
        out = engineer.fit_transform(X)
        assert "deck_name" not in out.columns
        assert "deck_name_Charizard" in out.columns
        assert "deck_name_Mewtwo" in out.columns

    def test_label_encoding(self):
        X = pd.DataFrame(
            {
                "deck_name": ["Charizard", "Mewtwo", "Pikachu"],
                "player_active_pokemon_hp": [100.0, 80.0, 90.0],
            }
        )
        engineer = StrategyFeatureEngineer(cat_cols=["deck_name"], encoding="label")
        out = engineer.fit_transform(X)
        assert "deck_name" in out.columns
        assert out["deck_name"].dtype == np.int32

    def test_make_features_wrapper(self):
        train = pd.DataFrame(
            {
                "id": [1, 2],
                "player_active_pokemon_hp": [100.0, 80.0],
                "opponent_active_pokemon_hp": [60.0, 90.0],
                "winner": [1, 0],
            }
        )
        test = pd.DataFrame(
            {
                "id": [3],
                "player_active_pokemon_hp": [70.0],
                "opponent_active_pokemon_hp": [80.0],
            }
        )
        X_train, X_test, y_train = make_features(
            train,
            test,
            diff_pairs=[("player_active_pokemon_hp", "opponent_active_pokemon_hp")],
        )
        assert "winner" not in X_train.columns
        assert len(y_train) == 2
        assert list(y_train) == [1, 0]
        assert "player_active_pokemon_hp_opponent_active_pokemon_hp" in X_train.columns


class TestStackingEnsemble:
    def test_fit_and_predict_on_synthetic_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame(
            {
                "hp_diff": rng.uniform(-100, 100, n),
                "bench_diff": rng.integers(-4, 5, n).astype(float),
                "hand_diff": rng.integers(-9, 10, n).astype(float),
                "turns": rng.integers(1, 20, n).astype(float),
                "prizes_left": rng.integers(0, 6, n).astype(float),
            }
        )
        y = pd.Series(rng.integers(0, 2, n))

        base_models = [
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ]
        ensemble = StackingEnsemble(base_models)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ensemble.fit(X, y, cv)

        assert len(ensemble.valid_scores_) == 3
        assert ensemble.overall_oof_score_ is not None
        assert all(0 <= s <= 1 for s in ensemble.valid_scores_)

        preds = ensemble.predict(X)
        assert len(preds) == n
        assert set(preds) <= {0, 1}

    def test_save_load_roundtrip(self, tmp_path):
        n = 50
        X = pd.DataFrame(
            {
                "hp_diff": [20.0] * n,
                "bench_diff": [1.0] * n,
                "hand_diff": [-2.0] * n,
            }
        )
        y = pd.Series([0, 1] * (n // 2))

        ensemble = StackingEnsemble([("lr", LogisticRegression(max_iter=500, random_state=42))])
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        ensemble.fit(X, y, cv)

        model_path = tmp_path / "ensemble.joblib"
        ensemble.save(model_path)
        assert model_path.exists()

        loaded = StackingEnsemble.load(model_path)
        preds_orig = ensemble.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestSaveSubmission:
    def test_saves_correct_format(self, tmp_path):
        ids = pd.Series([0, 1, 2])
        preds = np.array([1, 0, 1])
        out = tmp_path / "sub.csv"
        save_submission(ids, preds, output_path=str(out))
        df = pd.read_csv(out)
        assert list(df.columns) == ["id", "winner"]
        assert df["winner"].tolist() == [1, 0, 1]
        assert df["id"].tolist() == [0, 1, 2]

    def test_saves_with_custom_target_col(self, tmp_path):
        ids = pd.Series([0, 1])
        preds = np.array(["WIN", "LOSS"])
        out = tmp_path / "sub2.csv"
        save_submission(ids, preds, output_path=str(out), target_col="outcome")
        df = pd.read_csv(out)
        assert list(df.columns) == ["id", "outcome"]


class TestEncodeConsistency:
    def test_label_encoder_binary(self):
        le = LabelEncoder()
        classes = le.fit_transform([0, 1, 0, 1, 1])
        assert list(classes) == [0, 1, 0, 1, 1]
        assert list(le.classes_) == [0, 1]

    def test_roundtrip_with_model_prediction(self):
        le = LabelEncoder()
        y = [0, 1]
        le.fit(y)
        inverse = le.inverse_transform([0, 1])
        assert list(inverse) == [0, 1]
