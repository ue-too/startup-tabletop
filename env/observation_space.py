"""Observation encoding: game state -> fixed-size float array for RL.

Layout (for max 4 players):
  - Global features (turn, phase, event, deck sizes)
  - Self player features (cash, equity, AP, etc.)
  - Self talent cards (bench + board, padded to MAX_TALENT)
  - Self products (backlog + dev + ops, padded to MAX_PRODUCTS)
  - Self strategy hand (padded)
  - Per-opponent public features (cash, board talent, products)
  - Market features (agency row, product market, pools)
  - Action mask (separate, not in obs)
"""

from __future__ import annotations

import numpy as np

from startup_simulator.card_registry import get_registry
from startup_simulator.state import GameState, TalentInstance, ProductInstance
from startup_simulator.types import (
    CubeType, Phase, Sector, SubPhase, Tag, TalentType, Tier, Zone,
)

# Sizing constants
MAX_PLAYERS = 4
MAX_TALENT_PER_PLAYER = 20   # bench(5) + dev(~9) + ops(~6)
MAX_PRODUCTS_PER_PLAYER = 10  # backlog(3) + dev(3) + ops(~4)
MAX_STRATEGY_HAND = 3
MAX_AGENCY = 4
MAX_OPEN_MARKET = 5
MAX_PRODUCT_MARKET = 4
MAX_IDEA_POOL = 5

# Feature sizes per card
TALENT_FEATURES = 18
PRODUCT_FEATURES = 22
STRATEGY_FEATURES = 5

# Total sizes
GLOBAL_FEATURES = 12
PLAYER_SCALAR_FEATURES = 12
TALENT_BLOCK = MAX_TALENT_PER_PLAYER * TALENT_FEATURES  # 360
PRODUCT_BLOCK = MAX_PRODUCTS_PER_PLAYER * PRODUCT_FEATURES  # 220
STRATEGY_BLOCK = MAX_STRATEGY_HAND * STRATEGY_FEATURES  # 15
PLAYER_BLOCK = PLAYER_SCALAR_FEATURES + TALENT_BLOCK + PRODUCT_BLOCK + STRATEGY_BLOCK  # 607

MARKET_BLOCK = (
    MAX_AGENCY * TALENT_FEATURES          # 72
    + MAX_OPEN_MARKET * TALENT_FEATURES   # 90
    + MAX_PRODUCT_MARKET * PRODUCT_FEATURES  # 88
    + MAX_IDEA_POOL * PRODUCT_FEATURES    # 110
)  # 360

OBS_SIZE = GLOBAL_FEATURES + PLAYER_BLOCK * MAX_PLAYERS + MARKET_BLOCK
# 12 + 607*4 + 360 = 2800


def encode_observation(state: GameState, player_id: int) -> np.ndarray:
    """Encode game state from a player's perspective into a flat float array.

    Hidden information (opponent hands, face-down products) is masked with zeros.
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    registry = get_registry()
    offset = 0

    # --- Global features ---
    obs[offset] = state.turn_number / 30.0  # Normalized
    obs[offset + 1] = state.phase / 5.0
    obs[offset + 2] = state.sub_phase / 32.0
    obs[offset + 3] = state.current_player / MAX_PLAYERS
    obs[offset + 4] = state.num_players / MAX_PLAYERS
    obs[offset + 5] = 1.0 if state.market_crash_drawn else 0.0
    obs[offset + 6] = len(state.markets.seed_deck) / 30.0
    obs[offset + 7] = len(state.markets.growth_deck) / 35.0
    obs[offset + 8] = len(state.markets.talent_deck) / 22.0
    obs[offset + 9] = len(state.markets.strategy_deck) / 30.0
    obs[offset + 10] = len(state.markets.event_deck) / 18.0
    obs[offset + 11] = _encode_event(state)
    offset += GLOBAL_FEATURES

    # --- Player blocks (self first, then opponents in order) ---
    player_order = [player_id] + [
        i for i in range(state.num_players) if i != player_id
    ]
    # Pad to MAX_PLAYERS
    while len(player_order) < MAX_PLAYERS:
        player_order.append(-1)

    for idx, pid in enumerate(player_order):
        is_self = (pid == player_id)
        if pid < 0 or pid >= state.num_players:
            offset += PLAYER_BLOCK
            continue

        player = state.get_player(pid)

        # Scalar features
        obs[offset] = player.cash / 50.0
        obs[offset + 1] = player.equity_tokens_own / 3.0
        obs[offset + 2] = player.action_points / 3.0 if is_self else 0.0
        obs[offset + 3] = player.market_share_tokens / 10.0
        obs[offset + 4] = player.debt_tokens / 5.0
        obs[offset + 5] = len(player.bench) / 5.0
        obs[offset + 6] = len(player.dev_products) / 3.0
        obs[offset + 7] = len(player.ops_products) / 8.0
        obs[offset + 8] = len(player.product_backlog) / 3.0 if is_self else 0.0
        obs[offset + 9] = len(player.strategy_hand) / 3.0 if is_self else 0.0
        # Equity held in others
        total_equity = sum(player.equity_held.values())
        obs[offset + 10] = total_equity / 6.0
        obs[offset + 11] = 1.0  # Player exists flag
        offset += PLAYER_SCALAR_FEATURES

        # Talent cards
        talent_ids = _get_all_talent_ids(state, pid)
        for t_idx in range(MAX_TALENT_PER_PLAYER):
            if t_idx < len(talent_ids):
                tid = talent_ids[t_idx]
                talent = state.talent_instances[tid]
                # Hide opponent bench cards
                if not is_self and talent.zone == Zone.BENCH:
                    offset += TALENT_FEATURES
                    continue
                _encode_talent(obs, offset, talent, registry)
            offset += TALENT_FEATURES

        # Product cards
        product_ids = _get_all_product_ids(state, pid, is_self)
        for p_idx in range(MAX_PRODUCTS_PER_PLAYER):
            if p_idx < len(product_ids):
                pid_prod = product_ids[p_idx]
                prod = state.product_instances[pid_prod]
                pdef = registry.get_product(prod.card_def_id)
                # Hide opponent face-down products (only show that slot exists)
                if not is_self and prod.is_face_down:
                    obs[offset] = 1.0  # Exists
                    obs[offset + 1] = prod.zone / 2.0
                    offset += PRODUCT_FEATURES
                    continue
                _encode_product(obs, offset, prod, pdef)
            offset += PRODUCT_FEATURES

        # Strategy hand (self only)
        if is_self:
            for s_idx in range(MAX_STRATEGY_HAND):
                if s_idx < len(player.strategy_hand):
                    card_id = player.strategy_hand[s_idx]
                    sdef = registry.get_strategy(card_id)
                    obs[offset] = 1.0  # Exists
                    obs[offset + 1] = sdef.cost / 10.0
                    cat_map = {"training": 0.25, "warfare": 0.5, "attribute": 0.75, "utility": 1.0}
                    obs[offset + 2] = cat_map.get(sdef.category, 0.0)
                    obs[offset + 3] = hash(sdef.effect_id) % 100 / 100.0
                    obs[offset + 4] = sdef.count / 4.0
                offset += STRATEGY_FEATURES
        else:
            offset += STRATEGY_BLOCK

    # --- Market features ---
    # Agency row
    for a_idx in range(MAX_AGENCY):
        if a_idx < len(state.markets.agency_row):
            tid = state.markets.agency_row[a_idx]
            talent = state.talent_instances[tid]
            _encode_talent(obs, offset, talent, registry)
        offset += TALENT_FEATURES

    # Open job market
    for m_idx in range(MAX_OPEN_MARKET):
        market_list = list(state.markets.open_job_market)
        if m_idx < len(market_list):
            tid = market_list[m_idx]
            talent = state.talent_instances[tid]
            _encode_talent(obs, offset, talent, registry)
        offset += TALENT_FEATURES

    # Product market (seed + growth)
    all_market_products = (
        list(state.markets.product_market_seed) +
        list(state.markets.product_market_growth)
    )
    for p_idx in range(MAX_PRODUCT_MARKET):
        if p_idx < len(all_market_products):
            pid_prod = all_market_products[p_idx]
            prod = state.product_instances[pid_prod]
            pdef = registry.get_product(prod.card_def_id)
            _encode_product(obs, offset, prod, pdef)
        offset += PRODUCT_FEATURES

    # Open idea pool
    for i_idx in range(MAX_IDEA_POOL):
        pool_list = list(state.markets.open_idea_pool)
        if i_idx < len(pool_list):
            pid_prod = pool_list[i_idx]
            prod = state.product_instances[pid_prod]
            pdef = registry.get_product(prod.card_def_id)
            _encode_product(obs, offset, prod, pdef)
        offset += PRODUCT_FEATURES

    return obs


def _encode_talent(obs: np.ndarray, offset: int, talent: TalentInstance, registry) -> None:
    """Encode a talent instance into the obs array at offset."""
    tdef = registry.get_talent(talent.card_def_id)
    obs[offset] = 1.0  # Exists
    obs[offset + 1] = tdef.talent_type / 11.0
    obs[offset + 2] = tdef.cost / 10.0
    obs[offset + 3] = tdef.salary / 3.0
    obs[offset + 4] = tdef.base_output / 5.0
    obs[offset + 5] = 1.0 if tdef.is_junior else 0.0
    obs[offset + 6] = 1.0 if tdef.is_cross_functional else 0.0
    obs[offset + 7] = 1.0 if tdef.is_specialist else 0.0
    obs[offset + 8] = talent.zone / 2.0
    obs[offset + 9] = talent.total_xp / 4.0
    obs[offset + 10] = len(talent.xp_pending) / 3.0
    obs[offset + 11] = len(talent.skills) / 3.0
    obs[offset + 12] = talent.rank_badges / 1.0
    obs[offset + 13] = 1.0 if talent.onboarding else 0.0
    obs[offset + 14] = (talent.declared_mode or 0) / 2.0
    obs[offset + 15] = len(talent.attributes) / 3.0
    obs[offset + 16] = 1.0 if talent.equity_vested is not None else 0.0
    obs[offset + 17] = talent.get_output(tdef) / 7.0


def _encode_product(obs: np.ndarray, offset: int, prod: ProductInstance, pdef) -> None:
    """Encode a product instance into the obs array at offset."""
    obs[offset] = 1.0  # Exists
    obs[offset + 1] = prod.zone / 2.0
    obs[offset + 2] = pdef.tier / 3.0
    obs[offset + 3] = (pdef.sector or 0) / 5.0
    obs[offset + 4] = pdef.cost_software / 20.0
    obs[offset + 5] = pdef.cost_hardware / 20.0
    obs[offset + 6] = pdef.revenue / 15.0
    obs[offset + 7] = pdef.valuation / 28.0
    obs[offset + 8] = pdef.maint_software / 3.0
    obs[offset + 9] = pdef.maint_hardware / 3.0
    # Progress
    sw_cost, hw_cost = prod.get_effective_cost(pdef)
    obs[offset + 10] = prod.cubes_software / max(sw_cost, 1)
    obs[offset + 11] = prod.cubes_hardware / max(hw_cost, 1)
    obs[offset + 12] = prod.bugs / 5.0
    obs[offset + 13] = prod.hype / 3.0
    obs[offset + 14] = prod.scandal / 3.0
    obs[offset + 15] = 1.0 if prod.is_face_down else 0.0
    obs[offset + 16] = 1.0 if prod.is_feature_complete else 0.0
    obs[offset + 17] = 1.0 if prod.is_online else 0.0
    obs[offset + 18] = 1.0 if prod.integrated_with is not None else 0.0
    obs[offset + 19] = 1.0 if prod.is_host else 0.0
    # Tags (simplified: encode provides as a single value)
    obs[offset + 20] = (pdef.provides or 0) / 21.0
    obs[offset + 21] = len(pdef.requires) / 3.0


def _encode_event(state: GameState) -> float:
    """Encode active event as a normalized index."""
    if state.markets.active_event is None:
        return 0.0
    registry = get_registry()
    event_ids = [e.card_def_id for e in registry.event_cards]
    try:
        idx = event_ids.index(state.markets.active_event)
        return (idx + 1) / len(event_ids)
    except ValueError:
        return 0.0


def _get_all_talent_ids(state: GameState, player_id: int) -> list[int]:
    """Get all talent for a player: bench + board."""
    player = state.get_player(player_id)
    board = state.get_board_talent(player_id)
    return list(player.bench) + board


def _get_all_product_ids(state: GameState, player_id: int, include_hidden: bool) -> list[int]:
    """Get all product ids for a player."""
    player = state.get_player(player_id)
    result = []
    if include_hidden:
        result.extend(player.product_backlog)
    result.extend(player.dev_products)
    result.extend(player.ops_products)
    return result
