"""Integration tests for the agent-building pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pokemon.agent import RandomAgent, RuleBasedAgent
from pokemon.data import load_card_data
from pokemon.deck import Deck, build_deck
from pokemon.tracking import track_experiment


class TestPipeline:
    def test_card_data_loading(self, tmp_path):
        csv_path = tmp_path / "EN_Card_Data.csv"
        pd.DataFrame(
            {
                "Card ID": range(1, 11),
                "Card Name": [f"Card_{i}" for i in range(1, 11)],
            }
        ).to_csv(csv_path, index=False)

        result = load_card_data(data_dir=str(tmp_path))
        assert len(result["en"]) == 10

    def test_deck_60_card_roundtrip(self):
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

    def test_agent_baseline_interface(self):
        deck_cards = list(range(1, 61))
        agent = RuleBasedAgent(deck=deck_cards, random_seed=42)

        deck_result = agent({"select": None, "current_player": 0})
        assert len(deck_result) == 60

        act_result = agent(
            {
                "select": "options",
                "options": [10, 20, 30],
                "minCount": 1,
                "maxCount": 1,
            }
        )
        assert len(act_result) == 1
        assert act_result[0] in [10, 20, 30]

    def test_random_agent_reproducibility(self):
        a1 = RandomAgent(random_seed=42)
        a2 = RandomAgent(random_seed=42)
        obs = {"select": "options", "options": list(range(100)), "minCount": 1, "maxCount": 1}

        np.testing.assert_array_equal(a1(obs), a2(obs))

    def test_tracking_experiment(self, tmp_path):
        runs_dir = tmp_path / "runs"
        config = {"agent": {"type": "random"}, "seed": 42}

        with track_experiment(config, run_name="test_run", base_dir=str(runs_dir)) as run:
            run.log_metrics({"score": 0.5})
            run.log_params({"n_games": 100})

        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1
        assert "test_run" in str(run_dirs[0])

        metrics_path = run_dirs[0] / "metrics.json"
        assert metrics_path.exists()
        import json

        with open(metrics_path) as f:
            data = json.load(f)
        assert data["metrics"]["score"] == 0.5

        config_path = run_dirs[0] / "config.yaml"
        assert config_path.exists()
