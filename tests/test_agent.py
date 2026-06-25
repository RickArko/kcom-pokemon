from __future__ import annotations

import pytest

from pokemon.agent import Agent, RandomAgent, RuleBasedAgent


class TestAgentInterface:
    def test_abstract_agent_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Agent()  # type: ignore[abstract]

    def test_agent_delegates_to_select_deck_on_first_call(self):
        class DeckAgent(Agent):
            def _select_deck(self, obs):
                return [42] * 60

            def _act(self, obs):
                return [0]

        agent = DeckAgent()
        result = agent({"select": None, "current_player": 0})
        assert result == [42] * 60

    def test_agent_delegates_to_act_on_play_call(self):
        class PlayAgent(Agent):
            def _select_deck(self, obs):
                return [1] * 60

            def _act(self, obs):
                return [obs["best"]]

        agent = PlayAgent()
        result = agent({"select": "options", "best": 7, "options": [3, 5, 7]})
        assert result == [7]


class TestRandomAgent:
    @pytest.fixture
    def dummy_obs(self):
        return {"select": "options", "options": [0, 1, 2, 3, 4], "minCount": 1, "maxCount": 1}

    def test_random_agent_returns_deck(self):
        agent = RandomAgent(deck=list(range(60)))
        result = agent({"select": None})
        assert len(result) == 60
        assert result == list(range(60))

    def test_random_agent_picks_valid_option(self, dummy_obs):
        agent = RandomAgent(random_seed=42)
        result = agent(dummy_obs)
        assert len(result) == 1
        assert result[0] in dummy_obs["options"]

    def test_random_agent_repicks_with_different_seed(self, dummy_obs):
        a1 = RandomAgent(random_seed=1)
        a2 = RandomAgent(random_seed=9999)
        results = [a1(dummy_obs) for _ in range(10)] + [a2(dummy_obs) for _ in range(10)]
        assert len(set(tuple(r) for r in results)) > 1

    def test_random_agent_returns_empty_on_no_options(self):
        agent = RandomAgent()
        result = agent({"select": "options", "options": [], "minCount": 0, "maxCount": 0})
        assert result == []

    def test_random_agent_default_deck(self):
        agent = RandomAgent()
        result = agent({"select": None})
        assert len(result) == 60


class TestRuleBasedAgent:
    def test_rule_based_returns_deck(self):
        agent = RuleBasedAgent(deck=list(range(60)))
        result = agent({"select": None})
        assert len(result) == 60

    def test_rule_based_picks_first_option(self):
        agent = RuleBasedAgent()
        result = agent({"select": "options", "options": [99, 88, 77]})
        assert result == [99]

    def test_rule_based_empty_options(self):
        agent = RuleBasedAgent()
        result = agent({"select": "options", "options": []})
        assert result == []
