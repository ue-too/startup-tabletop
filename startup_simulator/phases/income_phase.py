"""Phase B: Income Phase - bandwidth check, revenue, dividends, salaries."""

from __future__ import annotations

from ..card_registry import get_registry
from ..state import GameState, PlayerState
from ..types import CubeType, SubPhase, TalentType, Zone


def calculate_bandwidth(state: GameState, player_id: int) -> tuple[int, int]:
    """Calculate total ops bandwidth for a player (software, hardware)."""
    registry = get_registry()
    player = state.get_player(player_id)
    bw_software = 0
    bw_hardware = 0

    for tid, talent in state.talent_instances.items():
        if talent.owner != player_id or talent.zone != Zone.OPS:
            continue
        if talent.onboarding:
            continue
        cdef = registry.get_talent(talent.card_def_id)
        output = talent.get_output(cdef)
        if output <= 0:
            continue

        if cdef.is_flex or cdef.is_cross_functional:
            # Flex: use declared mode or default to software
            mode = talent.declared_mode or CubeType.SOFTWARE
            if mode == CubeType.SOFTWARE:
                bw_software += output
            else:
                bw_hardware += output
        elif cdef.output_type == CubeType.SOFTWARE:
            bw_software += output
        elif cdef.output_type == CubeType.HARDWARE:
            bw_hardware += output

    return bw_software, bw_hardware


def calculate_maintenance_cost(state: GameState, player_id: int) -> tuple[int, int]:
    """Calculate total maintenance cost for a player's ops products (software, hardware)."""
    registry = get_registry()
    player = state.get_player(player_id)
    maint_sw = 0
    maint_hw = 0

    for pid in player.ops_products:
        prod = state.product_instances[pid]
        if not prod.is_online:
            continue
        pdef = registry.get_product(prod.card_def_id)
        maint_sw += pdef.maint_software
        maint_hw += pdef.maint_hardware

    return maint_sw, maint_hw


def check_bandwidth(state: GameState, player_id: int) -> bool:
    """Check if bandwidth covers maintenance. Returns True if OK, False if deficit."""
    bw_sw, bw_hw = calculate_bandwidth(state, player_id)
    maint_sw, maint_hw = calculate_maintenance_cost(state, player_id)
    return bw_sw >= maint_sw and bw_hw >= maint_hw


def calculate_operational_revenue(state: GameState, player_id: int) -> int:
    """Calculate operational revenue for a player.

    Revenue = Sum of active product revenue + staff bonuses - bug decay.
    """
    registry = get_registry()
    player = state.get_player(player_id)
    total_revenue = 0

    for pid in player.ops_products:
        prod = state.product_instances[pid]
        if not prod.is_online:
            continue
        pdef = registry.get_product(prod.card_def_id)

        product_rev = pdef.revenue

        # Staff bonuses
        team = state.get_talent_on_product(pid)
        has_vp_sales = False
        for tid in team:
            talent = state.talent_instances[tid]
            tdef = registry.get_talent(talent.card_def_id)
            if talent.zone != Zone.OPS or talent.onboarding:
                continue

            # Sales Rep: +$2/3/4 based on XP
            if tdef.talent_type == TalentType.SALES:
                from .engine_phase import _get_sales_bonus
                product_rev += _get_sales_bonus(talent, tdef)
                # VP of Sales (rank badge): ignores bug decay
                if talent.rank_badges > 0:
                    has_vp_sales = True

            # Growth Hacker: +$3
            elif tdef.talent_type == TalentType.GROWTH_HACKER:
                product_rev += 3

            # Workaholic attribute: +$2 revenue (attached to product staff)
            if "workaholic" in talent.attributes:
                product_rev += 2

        # Integration bonus: Client gets +$2
        if prod.integrated_with is not None and not prod.is_host:
            host = state.product_instances.get(prod.integrated_with)
            if host and host.is_online:
                product_rev += 2

        # Bug decay: -$1 per bug
        bug_decay = prod.bugs

        # VP of Sales: ignores bug decay
        if has_vp_sales:
            bug_decay = 0

        # QA with 2+ XP: prevents revenue decay on this product
        for tid in team:
            talent = state.talent_instances[tid]
            tdef = registry.get_talent(talent.card_def_id)
            if tdef.talent_type == TalentType.QA and talent.total_xp >= 2 and talent.zone == Zone.OPS:
                bug_decay = 0
                break

        # Stickiness: Client in integration ignores bug decay
        if prod.integrated_with is not None and not prod.is_host:
            host = state.product_instances.get(prod.integrated_with)
            if host and host.is_online:
                bug_decay = 0  # Stickiness

        product_rev = max(0, product_rev - bug_decay)
        total_revenue += product_rev

    return total_revenue


def calculate_dividend_tier(operational_revenue: int) -> int:
    """Calculate dividend payout per token based on operational revenue."""
    if operational_revenue <= 0:
        return 0
    elif operational_revenue <= 10:
        return 1
    elif operational_revenue <= 20:
        return 2
    else:
        return 3


def calculate_salary_cost(state: GameState, player_id: int) -> int:
    """Calculate total salary cost for a player's board + bench talent."""
    from ..types import Trait
    registry = get_registry()
    total = 0

    for tid, talent in state.talent_instances.items():
        if talent.owner != player_id:
            continue
        if talent.zone not in (Zone.DEV, Zone.OPS):
            continue
        cdef = registry.get_talent(talent.card_def_id)
        if cdef.is_junior:
            sal = talent.salary  # Dynamic based on XP
        else:
            sal = cdef.salary
            # Efficient trait: salary $1 in Ops (instead of $2)
            if cdef.trait == Trait.EFFICIENT and talent.zone == Zone.OPS:
                sal = 1

        # Workaholic attribute: +$2 salary
        if "workaholic" in talent.attributes:
            sal += 2

        total += sal

    return total


def process_income(state: GameState, player_id: int) -> dict:
    """Process the full income phase for a player. Returns summary dict."""
    player = state.get_player(player_id)
    registry = get_registry()
    summary = {}

    # 1. Bandwidth check (auto for Phase 1 - take products offline if needed)
    # For now, simple auto-resolution: take offline the lowest revenue products first
    bw_sw, bw_hw = calculate_bandwidth(state, player_id)
    maint_sw, maint_hw = calculate_maintenance_cost(state, player_id)

    if bw_sw < maint_sw or bw_hw < maint_hw:
        # Auto-resolve: take offline products until bandwidth is met
        # Sort by revenue (lowest first) to minimize loss
        online_products = [
            pid for pid in player.ops_products
            if state.product_instances[pid].is_online
        ]
        online_products.sort(
            key=lambda pid: registry.get_product(
                state.product_instances[pid].card_def_id
            ).revenue
        )
        for pid in online_products:
            if bw_sw >= maint_sw and bw_hw >= maint_hw:
                break
            prod = state.product_instances[pid]
            pdef = registry.get_product(prod.card_def_id)
            prod.is_online = False
            maint_sw -= pdef.maint_software
            maint_hw -= pdef.maint_hardware

    summary["bandwidth"] = {"sw": bw_sw, "hw": bw_hw, "maint_sw": maint_sw, "maint_hw": maint_hw}

    # 2. Operational Revenue
    op_rev = calculate_operational_revenue(state, player_id)
    summary["operational_revenue"] = op_rev

    # 3. Dividends (pay opponents who hold your equity)
    div_per_token = calculate_dividend_tier(op_rev)
    for other_id in range(state.num_players):
        if other_id == player_id:
            continue
        other = state.get_player(other_id)
        tokens_held = other.equity_held.get(player_id, 0)
        if tokens_held > 0 and div_per_token > 0:
            payout = tokens_held * div_per_token
            other.cash += payout  # Bank pays

    # 4. Collect dividends from equity we hold
    for founder_id, count in player.equity_held.items():
        if count > 0:
            founder_rev = calculate_operational_revenue(state, founder_id)
            div = calculate_dividend_tier(founder_rev)
            player.cash += count * div

    summary["dividends_received"] = sum(
        player.equity_held.get(fid, 0) * calculate_dividend_tier(
            calculate_operational_revenue(state, fid)
        )
        for fid in range(state.num_players) if fid != player_id
    )

    # 5. Collect operational revenue
    player.cash += op_rev
    summary["cash_from_revenue"] = op_rev

    # 6. Pay salaries
    salary_cost = calculate_salary_cost(state, player_id)
    summary["salary_cost"] = salary_cost

    if player.cash >= salary_cost:
        player.cash -= salary_cost
    else:
        # Cannot pay all salaries - fire cheapest first (auto for Phase 1)
        player.cash -= salary_cost
        if player.cash < 0:
            # Add debt tokens
            debt = -player.cash
            player.cash = 0
            player.debt_tokens += (debt + 4) // 5  # Each debt token = $5 penalty

    return summary
