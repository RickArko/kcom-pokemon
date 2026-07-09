"""Tests for pokemon.meta_decks (loading + proxy agent factory)."""

from __future__ import annotations

import pytest

from pokemon.agent import Agent
from pokemon.meta_decks import (
    META_ARCHETYPES,
    load_meta_decks,
    make_meta_proxies,
    make_meta_proxy,
)


def _write_deck(dir_, name, cards):
    path = dir_ / name
    path.write_text("\n".join(str(c) for c in cards) + "\n")
    return path


def test_load_meta_decks_skips_missing(tmp_path):
    _write_deck(tmp_path, "fighting_toolbox.csv", [1] * 60)
    decks = load_meta_decks(tmp_path)
    assert "cynthia_garchomp" in decks
    assert len(decks["cynthia_garchomp"]) == 60
    assert "grimmsnarl" not in decks  # missing CSV skipped


def test_load_meta_decks_empty_dir(tmp_path):
    assert load_meta_decks(tmp_path) == {}


def test_make_meta_proxy_builds_agent():
    # Pass an explicit deck so the test doesn't depend on on-disk meta data.
    proxy = make_meta_proxy("cynthia_garchomp", deck=[677] * 4 + [6] * 56, random_seed=3)
    assert isinstance(proxy, Agent)
    deck = proxy({"select": None})
    assert len(deck) == 60


def test_make_meta_proxy_unknown_archetype_raises():
    with pytest.raises(KeyError):
        make_meta_proxy("nonexistent", deck=None)


def test_make_meta_proxies_returns_tuples(monkeypatch, tmp_path):
    _write_deck(tmp_path, "fighting_toolbox.csv", [1] * 60)
    _write_deck(tmp_path, "unknown.csv", [2] * 60)
    # Point the module-level loader at tmp_path and bust the cache.
    import pokemon.meta_decks as md

    monkeypatch.setattr(md, "_META_CACHE", None)
    monkeypatch.setattr(md, "_META_DIR", tmp_path)
    proxies = make_meta_proxies(random_seed=1)
    names = [n for n, _ in proxies]
    assert "cynthia_garchomp" in names
    assert "grimmsnarl" in names
    for _, agent in proxies:
        assert isinstance(agent, Agent)


def test_make_meta_proxies_include_sample():
    proxies = make_meta_proxies(random_seed=1, include_sample=True)
    names = [n for n, _ in proxies]
    assert any(n.startswith("sample_") for n in names)


def test_meta_archetypes_mapping_has_expected_names():
    assert "cynthia_garchomp" in META_ARCHETYPES
    assert "dragapult_ex" in META_ARCHETYPES
