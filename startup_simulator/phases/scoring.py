"""End-game scoring calculation."""

from __future__ import annotations

from ..card_registry import get_registry
from ..state import GameState
from ..types import Tier


def calculate_final_scores(state: GameState) -> list[int]:
    """Calculate final valuation for all players.

    Scoring categories:
    1. Product Portfolio: Active maintenance products (VP + Hype - Scandal + Integration)
    2. Cash: $1 VP per $5
    3. Market Share: 2 VP per token
    4. Portfolio: 5 VP per opponent equity token
    5. Milestones: face value (Phase 4)
    6. Human Capital: 1 VP per XP/Skill, 2 VP per Rank Badge
    7. Penalties: Vaporware -2/-5/-10, Debt -5 each
    """
    registry = get_registry()
    scores = []

    for player in state.players:
        vp = 0

        # 1. Product Portfolio
        for pid in player.ops_products:
            prod = state.product_instances[pid]
            if not prod.is_online:
                continue
            pdef = registry.get_product(prod.card_def_id)

            product_vp = pdef.valuation
            product_vp += prod.hype * 1  # +1 VP per hype (simplified from rulebook's +VP)
            product_vp -= prod.scandal * 1  # -1 VP per scandal
            product_vp += prod.stealth_launch_bonus

            # Integration bonus: Host gets +5 VP
            if prod.integrated_with is not None and prod.is_host:
                partner = state.product_instances.get(prod.integrated_with)
                if partner and partner.is_online:
                    product_vp += 5

            vp += max(0, product_vp)

        # 2. Cash conversion
        vp += player.cash // 5

        # 3. Market Share
        vp += player.market_share_tokens * 2

        # 4. Portfolio (equity in opponents)
        for other_id, count in player.equity_held.items():
            vp += count * 5

        # 5. Milestones (Phase 4 - skip for now)

        # 6. Human Capital
        for tid in state.get_all_talent_for_player(player.player_id):
            talent = state.talent_instances[tid]
            vp += len(talent.xp_permanent)  # 1 VP per XP
            vp += len(talent.skills)         # 1 VP per Skill
            vp += talent.rank_badges * 2     # 2 VP per Rank Badge

        # 7. Penalties
        # Vaporware: backlog cards
        for pid in player.product_backlog:
            prod = state.product_instances[pid]
            pdef = registry.get_product(prod.card_def_id)
            if pdef.tier == Tier.TIER1:
                vp -= 2
            elif pdef.tier == Tier.TIER2:
                vp -= 5
            elif pdef.tier == Tier.TIER3:
                vp -= 10

        # Debt penalty
        vp -= player.debt_tokens * 5

        scores.append(vp)

    return scores
