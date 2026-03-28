"""Reward shaping functions for RL training."""

from __future__ import annotations

from startup_simulator.state import GameState
from startup_simulator.phases.scoring import calculate_final_scores
from startup_simulator.phases.income_phase import calculate_operational_revenue


def sparse_reward(state: GameState, player_id: int) -> float:
    """Sparse reward: only at game end.

    +1.0 for winner, -1.0 for loser, 0.0 for tie, 0.0 during game.
    """
    if not state.game_over:
        return 0.0

    scores = state.final_scores
    if not scores:
        return 0.0

    my_score = scores[player_id]
    max_score = max(scores)

    if my_score == max_score:
        # Check for ties
        num_winners = sum(1 for s in scores if s == max_score)
        if num_winners > 1:
            return 0.5  # Tie
        return 1.0
    else:
        return -1.0


def shaped_reward(state: GameState, player_id: int, prev_valuation: float) -> tuple[float, float]:
    """Shaped reward: delta in estimated valuation + time pressure + game-end bonus.

    Returns (reward, current_valuation) so caller can track prev_valuation.

    Components:
    - Delta valuation: reward for increasing VP estimate
    - Time pressure: small negative per step to discourage stalling
    - Game-end: +1/-1 for win/loss
    """
    curr_val = estimate_valuation(state, player_id)

    if state.game_over:
        end_reward = sparse_reward(state, player_id)
        delta = (curr_val - prev_valuation) / 100.0
        return delta + end_reward, curr_val
    else:
        delta = (curr_val - prev_valuation) / 100.0
        # Time pressure: small negative per step to discourage passing/stalling
        # -0.001 per step means ~-0.3 over a 300-step game (small but consistent)
        time_penalty = -0.001
        return delta + time_penalty, curr_val


def estimate_valuation(state: GameState, player_id: int) -> float:
    """Estimate current valuation for a player (same formula as final scoring).

    This can be called mid-game as a progress signal.
    """
    from startup_simulator.card_registry import get_registry
    from startup_simulator.types import Tier

    registry = get_registry()
    player = state.get_player(player_id)
    vp = 0.0

    # 1. Product Portfolio (ops)
    for pid in player.ops_products:
        prod = state.product_instances[pid]
        if not prod.is_online:
            continue
        pdef = registry.get_product(prod.card_def_id)
        product_vp = pdef.valuation + prod.hype - prod.scandal + prod.stealth_launch_bonus
        if prod.integrated_with is not None and prod.is_host:
            partner = state.product_instances.get(prod.integrated_with)
            if partner and partner.is_online:
                product_vp += 5
        vp += max(0, product_vp)

    # 2. Cash
    vp += player.cash / 5.0

    # 3. Market Share
    vp += player.market_share_tokens * 2

    # 4. Portfolio (equity held)
    for other_id, count in player.equity_held.items():
        vp += count * 5

    # 5. Human Capital
    for tid in state.get_all_talent_for_player(player_id):
        talent = state.talent_instances[tid]
        vp += len(talent.xp_permanent)
        vp += len(talent.skills)
        vp += talent.rank_badges * 2

    # 6. Penalties
    for pid in player.product_backlog:
        prod = state.product_instances[pid]
        pdef = registry.get_product(prod.card_def_id)
        if pdef.tier == Tier.TIER1:
            vp -= 2
        elif pdef.tier == Tier.TIER2:
            vp -= 5
        elif pdef.tier == Tier.TIER3:
            vp -= 10
    vp -= player.debt_tokens * 5

    # 7. Dev products (partial credit for progress)
    for pid in player.dev_products:
        prod = state.product_instances[pid]
        pdef = registry.get_product(prod.card_def_id)
        sw_cost, hw_cost = prod.get_effective_cost(pdef)
        total_cost = sw_cost + hw_cost
        if total_cost > 0:
            progress = (prod.cubes_software + prod.cubes_hardware) / total_cost
            vp += pdef.valuation * progress * 0.3  # 30% credit for in-progress

    return vp
