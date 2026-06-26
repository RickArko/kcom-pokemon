"""Forward search via the ``cg`` search API (MCTS).

Wraps ``cg.api.search_begin`` / ``search_step`` / ``search_end`` /
``search_release`` to provide a Monte Carlo Tree Search that plans ahead from
the current observation.  The search API simulates the game forward with
*predicted* hidden information (opponent deck/hand/prizes, our face-down
prizes), so an :class:`MCTSResult` is a distribution over the **real**
observation's option indices.

The module imports the ``cg`` engine lazily (inside functions), so it imports
cleanly in offline tests; only :func:`mcts_search` requires the engine.

Design
------
- ``MCTSNode`` stores a ``searchId`` (a persistent forward-sim state).  Tree
  edges are created once via ``search_step`` and re-traversed by their stored
  id in later simulations (states persist until released).
- Selection: UCB1. Expansion: one untried action per simulation.
- Rollout: a fast inline policy (prefer ATTACK, else END, else first option)
  to a depth limit; the same policy is used for both players (symmetric prior).
- Leaf value: from the **root player's** perspective — terminal result if the
  rollout ended the game, otherwise a heuristic utility (prize lead + HP lead +
  bench lead) clipped to [-1, 1].
- Time budget + simulation cap; releases all states in ``finally``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from pokemon.card_db import CardDB
from pokemon.state import OPT_ATTACK, OPT_END

logger = logging.getLogger(__name__)

# A rollout policy maps a search observation dict (the asdict of a search
# Observation) to a valid action list for that state's select.
RolloutPolicy = Callable[[dict], list[int]]

# --- result ------------------------------------------------------------------


@dataclass
class MCTSResult:
    """Outcome of an MCTS search over the real observation's options."""

    action: list[int]
    visits: dict[int, int]  # option index -> simulation visits
    win_rates: dict[int, float]  # option index -> mean value (root player perspective)
    simulations: int
    elapsed: float
    fell_back: bool = False


# --- predictions for hidden information --------------------------------------


@dataclass
class SearchPredictions:
    """Predicted hidden card IDs required by ``search_begin``."""

    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]


def _observed_my_card_ids(state_current: dict, your_index: int) -> list[int]:
    """Card IDs we can see on our side (hand, in-play, discard) — to subtract
    from our known deck when predicting our face-down deck/prizes."""
    me = (state_current.get("players") or [None, None])[your_index] or {}
    seen: list[int] = []
    for c in me.get("hand") or []:
        if c:
            seen.append(c.get("id"))
    for p in me.get("active") or []:
        if p:
            seen.append(p.get("id"))
            seen += [c.get("id") for c in (p.get("energyCards") or []) if c]
            seen += [c.get("id") for c in (p.get("tools") or []) if c]
            seen += [c.get("id") for c in (p.get("preEvolution") or []) if c]
    for p in me.get("bench") or []:
        if p:
            seen.append(p.get("id"))
            seen += [c.get("id") for c in (p.get("energyCards") or []) if c]
            seen += [c.get("id") for c in (p.get("tools") or []) if c]
            seen += [c.get("id") for c in (p.get("preEvolution") or []) if c]
    seen += [c.get("id") for c in (me.get("discard") or []) if c]
    return [x for x in seen if x is not None]


def _sample_pool(pool: list[int], counts_by_player: dict, rng) -> dict:
    """Deal `counts_by_player` cards from `pool` without replacement.

    Returns a dict mapping the same keys to sampled lists.  If the pool is too
    small, samples with replacement (degrades gracefully for prediction).
    """
    out: dict = {}
    remaining = list(pool)
    for key, n in counts_by_player.items():
        if n <= 0:
            out[key] = []
            continue
        if len(remaining) >= n:
            pick = rng.choice(len(remaining), size=n, replace=False)
            picked = [int(remaining[i]) for i in pick]
            for x in picked:
                remaining.remove(x)
        else:
            base = pool or [1]
            pick = rng.choice(base, size=n)
            picked = [int(x) for x in pick]
        out[key] = picked
    return out


def build_predictions(
    obs_dict: dict,
    my_deck: list[int],
    opponent_deck_hint: list[int] | None = None,
    card_db: CardDB | None = None,
    rng=None,
) -> SearchPredictions:
    """Construct hidden-info predictions matching the engine's required counts.

    Parameters
    ----------
    obs_dict:
        The raw (or flattened) observation dict — must contain ``current``.
    my_deck:
        Our known 60-card deck list (used to predict our face-down deck/prizes
        by subtracting observed cards, and as the default opponent prior).
    opponent_deck_hint:
        Optional prior for the opponent's deck.  Defaults to a mirror
        assumption (``my_deck``) so the agent stays submission-honest — on
        Kaggle we do not know the opponent's list.
    card_db:
        Used to pick a Basic Pokémon ID when the opponent's active is face-down.
    rng:
        A ``numpy`` ``Generator`` for reproducibility.
    """
    import numpy as np

    rng = rng or np.random.default_rng()
    current = obs_dict.get("current") or {}
    yi = int(current.get("yourIndex", 0))
    players = current.get("players") or [None, None]
    me = players[yi] or {}
    opp = players[1 - yi] or {}

    opp_hint = list(opponent_deck_hint or my_deck)

    # Our face-down cards = known deck minus everything we can see on our side.
    seen = _observed_my_card_ids(current, yi)
    seen_set = list(seen)
    my_pool = list(my_deck)
    for x in seen_set:
        if x in my_pool:
            my_pool.remove(x)
    my_counts = {
        "deck": int(me.get("deckCount", 0)),
        "prize": len(me.get("prize") or []),
    }
    my_sampled = _sample_pool(my_pool, my_counts, rng)

    opp_counts = {
        "deck": int(opp.get("deckCount", 0)),
        "prize": len(opp.get("prize") or []),
        "hand": int(opp.get("handCount", 0)),
    }
    opp_sampled = _sample_pool(opp_hint, opp_counts, rng)

    # Opponent active: only needed when their active is face-down.
    opp_active_pred: list[int] = []
    opp_active_list = opp.get("active") or []
    if opp_active_list and opp_active_list[0] is None:
        basic_id = _first_basic_pokemon_id(opp_hint, card_db)
        if basic_id is not None:
            opp_active_pred = [basic_id]

    return SearchPredictions(
        your_deck=my_sampled["deck"],
        your_prize=my_sampled["prize"],
        opponent_deck=opp_sampled["deck"],
        opponent_prize=opp_sampled["prize"],
        opponent_hand=opp_sampled["hand"],
        opponent_active=opp_active_pred,
    )


def _first_basic_pokemon_id(deck_ids: list[int], card_db: CardDB | None) -> int | None:
    if card_db is None:
        return deck_ids[0] if deck_ids else None
    for cid in deck_ids:
        info = card_db.get(cid)
        if info is not None and info.is_pokemon and info.is_basic:
            return cid
    return deck_ids[0] if deck_ids else None


# --- MCTS tree ---------------------------------------------------------------


@dataclass
class _Node:
    search_id: int
    parent: _Node | None
    action: list[int]  # action that led here (option indices at the parent)
    option_index: int  # the parent option index this action corresponds to
    obs_dict: dict  # observation dict cached at creation (carries this state's select)
    children: dict[int, _Node] = field(default_factory=dict)
    untried: list[int] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0


def _ucb1(node: _Node, parent_visits: int, c: float) -> float:
    if node.visits == 0:
        return float("inf")
    exploit = node.total_value / node.visits
    explore = c * ((2 * (parent_visits + 1)) ** 0.5) / (node.visits + 1)
    return exploit + explore


def _select_child(node: _Node, c: float) -> _Node:
    return max(node.children.values(), key=lambda ch: _ucb1(ch, node.visits, c))


# --- leaf evaluation (root player perspective) -------------------------------


def _prizes_taken(side: dict) -> int:
    return max(0, 6 - len(side.get("prize") or []))


def _active_hp_ratio(side: dict) -> float:
    act = side.get("active") or []
    if not act or act[0] is None:
        return 0.0
    p = act[0]
    hp = int(p.get("hp", 0))
    max_hp = int(p.get("maxHp", 1)) or 1
    return hp / max_hp


def _leaf_value(obs_dict: dict, root_player: int) -> float:
    current = obs_dict.get("current") or {}
    result = int(current.get("result", -1))
    if result != -1:
        if result == root_player:
            return 1.0
        if result == 2:
            return 0.0
        return -1.0
    players = current.get("players") or [None, None]
    me = players[root_player] or {}
    opp = players[1 - root_player] or {}
    prize_lead = _prizes_taken(me) - _prizes_taken(opp)
    hp_lead = _active_hp_ratio(me) - _active_hp_ratio(opp)
    bench_lead = len(me.get("bench") or []) - len(opp.get("bench") or [])
    val = prize_lead * 0.15 + hp_lead * 0.5 + bench_lead * 0.05
    return max(-1.0, min(1.0, val))


def _root_player(obs_dict: dict) -> int:
    return int((obs_dict.get("current") or {}).get("yourIndex", 0))


# --- fast rollout policy (operates on search Observation dicts) --------------


def _fast_rollout_action(select_dict: dict, rng) -> list[int]:
    """Pick a quick action: prefer ATTACK, else END, else first option.

    The engine only offers affordable ATTACK options, so attacking is always a
    legal progress move.  Uses the select dict (converted from the search
    Observation) so it needs no card lookups.
    """
    opts = select_dict.get("option") or []
    if not opts:
        return []
    n = int(select_dict.get("maxCount", 1) or 1)
    n = max(1, n)
    # Prefer ATTACK
    for o in opts:
        if int(o.get("type", -1)) == OPT_ATTACK:
            return [opts.index(o)]
    # Then END
    for o in opts:
        if int(o.get("type", -1)) == OPT_END:
            return [opts.index(o)]
    # Otherwise a random valid selection of size n
    indices = list(range(len(opts)))
    n = min(n, len(indices))
    return [int(x) for x in rng.choice(indices, size=n, replace=False)]


def _obs_to_dict(obs_obj) -> dict:
    """Convert a search Observation dataclass back to a plain dict for rollout."""
    from dataclasses import asdict

    return asdict(obs_obj)


# --- main entry point --------------------------------------------------------


def mcts_search(
    obs_dict: dict,
    my_deck: list[int],
    opponent_deck_hint: list[int] | None = None,
    card_db: CardDB | None = None,
    simulations: int = 50,
    time_budget: float = 0.8,
    rollout_depth: int = 20,
    exploration_c: float = 1.4,
    rollout_policy: RolloutPolicy | None = None,
    rng=None,
) -> MCTSResult:
    """Run MCTS from the current observation and return the best action.

    The returned ``action`` is a list of option indices valid for the **real**
    observation's ``select`` (the search root mirrors the real state, so root
    option indices map 1:1).

    ``rollout_policy`` defaults to the fast inline policy (ATTACK > END >
    random).  Pass a richer policy (e.g. an exp002 heuristic ``_act``) for more
    realistic playouts at higher CPU cost.
    """
    import numpy as np
    from cg.api import search_begin, search_end, search_release, search_step, to_observation_class

    rng = rng or np.random.default_rng()
    if rollout_policy is not None:
        rollout_fn: RolloutPolicy = rollout_policy
    else:
        rollout_fn = lambda sel_obs: _fast_rollout_action_from_obs(sel_obs, rng)  # noqa: E731
    t0 = time.time()
    root_player = _root_player(obs_dict)

    preds = build_predictions(
        obs_dict, my_deck, opponent_deck_hint=opponent_deck_hint, card_db=card_db, rng=rng
    )
    obs_obj = to_observation_class(obs_dict)
    root_select = obs_dict.get("select") or {}
    root_options = root_select.get("option") or []
    n_root = len(root_options)
    if n_root == 0:
        return MCTSResult(
            action=[], visits={}, win_rates={}, simulations=0, elapsed=0.0, fell_back=True
        )

    root_state = search_begin(
        obs_obj,
        your_deck=preds.your_deck,
        your_prize=preds.your_prize,
        opponent_deck=preds.opponent_deck,
        opponent_prize=preds.opponent_prize,
        opponent_hand=preds.opponent_hand,
        opponent_active=preds.opponent_active,
        manual_coin=False,
    )
    root_obs_dict = _obs_to_dict(root_state.observation)

    root_node = _Node(
        search_id=root_state.searchId,
        parent=None,
        action=[],
        option_index=-1,
        obs_dict=root_obs_dict,
        untried=list(range(n_root)),
    )

    visits = {i: 0 for i in range(n_root)}
    totals = {i: 0.0 for i in range(n_root)}
    sim = 0

    try:
        while sim < simulations and (time.time() - t0) < time_budget:
            # --- selection ---
            node = root_node
            while not node.untried and node.children:
                node = _select_child(node, exploration_c)

            # --- expansion ---
            if node.untried:
                action_idx = node.untried.pop()
                action = _resolve_action(node, action_idx, root_select)
                try:
                    child_state = search_step(node.search_id, action)
                except Exception as e:  # noqa: BLE001
                    logger.debug("search_step expand failed: %s", e)
                    continue
                child_obs_dict = _obs_to_dict(child_state.observation)
                child_node = _Node(
                    search_id=child_state.searchId,
                    parent=node,
                    action=action,
                    option_index=action_idx,
                    obs_dict=child_obs_dict,
                    untried=_child_untried(child_obs_dict),
                )
                node.children[action_idx] = child_node
                node = child_node

            # --- rollout + backprop ---
            value = _rollout(node, root_player, rollout_depth, rng, rollout_fn)
            bn = node
            while bn is not None:
                bn.visits += 1
                bn.total_value += value
                # Only root children map to real-observation option indices.
                if bn.parent is root_node:
                    visits[bn.option_index] = visits.get(bn.option_index, 0) + 1
                    totals[bn.option_index] = totals.get(bn.option_index, 0.0) + value
                bn = bn.parent

            sim += 1

        # --- choose: robust child (most visits), tie-break by value ---
        best_idx, best_v = -1, -1
        for i in range(n_root):
            v = visits.get(i, 0)
            if v > best_v or (v == best_v and totals.get(i, 0.0) > totals.get(best_idx, 0.0)):
                best_idx, best_v = i, v
        if best_idx < 0:
            best_idx = 0
        chosen = _resolve_action(root_node, best_idx, root_select)
        win_rates = {i: (totals[i] / visits[i] if visits[i] else 0.0) for i in range(n_root)}
        return MCTSResult(
            action=chosen,
            visits=visits,
            win_rates=win_rates,
            simulations=sim,
            elapsed=time.time() - t0,
            fell_back=False,
        )
    finally:
        _release_tree(root_node, search_release)
        search_end()


def _resolve_action(node: _Node, option_index: int, root_select: dict) -> list[int]:
    """Resolve an option index into a valid action list for ``search_step``.

    For MAIN selects ``minCount``/``maxCount`` are usually 1, so a single-index
    action is correct.  When ``maxCount > 1`` we take the first ``maxCount``
    indices (a conservative valid selection).
    """
    sel = node.obs_dict.get("select") if node.parent is not None else root_select
    mx = int((sel or {}).get("maxCount", 1) or 1)
    mn = int((sel or {}).get("minCount", 1) or 1)
    n_opts = len((sel or {}).get("option") or [])
    n = max(mn, 1)
    n = min(n, mx, n_opts) if n_opts else 1
    if n <= 1:
        return [option_index]
    return list(range(n))


def _child_untried(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select")
    if sel is None:
        return []
    return list(range(len(sel.get("option") or [])))


def _rollout(node: _Node, root_player: int, depth: int, rng, rollout_fn: RolloutPolicy) -> float:
    """Play a rollout from `node`'s state; return leaf value (root view).

    Uses ``rollout_fn`` to pick actions.  Transient search states created here
    are released inline (they are not part of the tree).
    """
    from cg.api import search_release, search_step

    if _is_terminal(node.obs_dict):
        return _leaf_value(node.obs_dict, root_player)

    cur_sid = node.search_id
    cur_obs = node.obs_dict
    transient: list[int] = []
    try:
        for _ in range(depth):
            sel = cur_obs.get("select")
            if sel is None:
                break
            try:
                action = rollout_fn(cur_obs)
            except Exception:  # noqa: BLE001 - rollout policy must not crash the search
                action = _fast_rollout_action_from_obs(cur_obs, rng)
            if not action:
                break
            try:
                nxt = search_step(cur_sid, action)
            except Exception:  # noqa: BLE001
                break
            transient.append(nxt.searchId)
            cur_sid = nxt.searchId
            cur_obs = _obs_to_dict(nxt.observation)
            if _is_terminal(cur_obs):
                return _leaf_value(cur_obs, root_player)
        return _leaf_value(cur_obs, root_player)
    finally:
        for sid in transient:
            try:
                search_release(sid)
            except Exception:  # noqa: BLE001
                pass


def _fast_rollout_action_from_obs(obs_dict: dict, rng) -> list[int]:
    """Fast inline rollout policy operating on a search obs dict."""
    sel = obs_dict.get("select") or {}
    return _fast_rollout_action(sel, rng)


def _is_terminal(obs_dict: dict) -> bool:
    return int((obs_dict.get("current") or {}).get("result", -1)) != -1


def _release_tree(node: _Node, release_fn) -> None:
    """Recursively release all searchIds in the tree."""
    stack = [node]
    while stack:
        n = stack.pop()
        for ch in n.children.values():
            stack.append(ch)
        try:
            release_fn(n.search_id)
        except Exception:  # noqa: BLE001
            pass
