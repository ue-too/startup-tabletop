"""Phase D: Engine Phase - cube generation, QA, audit, commit, complete, refill, cleanup."""

from __future__ import annotations

from ..card_registry import get_registry
from ..state import GameState, TalentInstance
from ..cards import TalentCardDef
from ..types import CubeType, Trait, TalentType, Tier, Zone
from ..actions.product_actions import _refill_product_market_seed, _refill_product_market_growth


# ---------------------------------------------------------------------------
# Trait helpers
# ---------------------------------------------------------------------------

def _has_trait(talent: TalentInstance, card_def: TalentCardDef, trait: Trait) -> bool:
    return card_def.trait == trait


def _has_clean_code(talent: TalentInstance, card_def: TalentCardDef) -> bool:
    """Check if unit has Clean Code trait (never generates bugs)."""
    if card_def.trait == Trait.CLEAN_CODE:
        return True
    # Also check attached attribute cards
    return "clean_coder" in talent.attributes


def _get_qa_bug_removal(talent: TalentInstance, card_def: TalentCardDef) -> int:
    """How many bugs can this QA remove per turn?"""
    if card_def.talent_type != TalentType.QA:
        return 0
    base = 1
    # +1 XP = removes 2 bugs, +2 XP = removes 2 bugs (same, but adds decay prevention)
    xp_count = talent.total_xp
    if xp_count >= 1:
        base = 2
    return base


def _get_sales_bonus(talent: TalentInstance, card_def: TalentCardDef) -> int:
    """Revenue bonus from Sales Rep."""
    if card_def.talent_type != TalentType.SALES:
        return 0
    base = 2
    xp_count = talent.total_xp
    if xp_count >= 2:
        return 4
    elif xp_count >= 1:
        return 3
    return base


def _get_pm_synergy(talent: TalentInstance, card_def: TalentCardDef, team_size: int) -> int:
    """PM synergy bonus: +1 output per other teammate."""
    if not card_def.is_pm:
        return 0
    if talent.onboarding:
        return 0
    return max(0, team_size - 1)


# ---------------------------------------------------------------------------
# Cube generation
# ---------------------------------------------------------------------------

def generate_cubes(state: GameState, player_id: int) -> None:
    """Generate work cubes for all dev teams of a player.

    Cubes go into the transient zone first (committed later).
    Also handles Pending XP for juniors.
    """
    registry = get_registry()
    player = state.get_player(player_id)

    for pid in player.dev_products:
        prod = state.product_instances[pid]
        pdef = registry.get_product(prod.card_def_id)
        team = state.get_talent_on_product(pid)
        team_size = len(team)

        # Check for stalled project (Tier 2/3 without matching lead)
        is_stalled = False
        if pdef.tier >= Tier.TIER2:
            is_stalled = not _has_matching_lead(state, pid, pdef)

        # First pass: compute PM synergy bonus
        pm_bonus_sw = 0
        pm_bonus_hw = 0
        for tid in team:
            talent = state.talent_instances[tid]
            tdef = registry.get_talent(talent.card_def_id)
            synergy = _get_pm_synergy(talent, tdef, team_size)
            if synergy > 0 and not is_stalled:
                # PM bonus goes to the product's majority cube type
                if pdef.cost_software >= pdef.cost_hardware:
                    pm_bonus_sw += synergy
                else:
                    pm_bonus_hw += synergy

        # Second pass: generate cubes from each team member
        for tid in team:
            talent = state.talent_instances[tid]
            if talent.onboarding:
                continue
            tdef = registry.get_talent(talent.card_def_id)

            # Specialists
            if tdef.is_specialist:
                if tdef.talent_type == TalentType.QA:
                    # QA in dev: remove bugs
                    removals = _get_qa_bug_removal(talent, tdef)
                    for _ in range(removals):
                        if prod.bugs > 0:
                            prod.bugs -= 1
                    # Pending XP for junior QA (QA specialists aren't junior, but
                    # a junior with QA skill acting as QA earns green pending XP)
                elif tdef.talent_type == TalentType.GROWTH_HACKER:
                    pass  # Growth hacker in dev does nothing
                continue

            # Stalled projects produce 0 output
            if is_stalled:
                continue

            output = talent.get_output(tdef)
            if output <= 0:
                continue

            # Determine cube type via effective mode
            mode = talent.get_effective_mode(tdef)
            if mode is None:
                continue

            # Spaghetti Code trait: add 1 bug on first cube generation
            # (handled at assign time, not here)

            # Place cubes in transient zone
            if mode == CubeType.SOFTWARE:
                prod.transient_software += output
            elif mode == CubeType.HARDWARE:
                prod.transient_hardware += output

            # Pending XP for juniors
            if tdef.is_junior and not talent.has_pending_xp_of_type(mode):
                if len(talent.xp_pending) < 3 and talent.total_xp < 4:
                    talent.xp_pending.append(mode)

        # Apply PM synergy bonus
        if not is_stalled:
            prod.transient_software += pm_bonus_sw
            prod.transient_hardware += pm_bonus_hw

        # Handle juniors acting as QA (green skill token)
        for tid in team:
            talent = state.talent_instances[tid]
            if talent.onboarding:
                continue
            tdef = registry.get_talent(talent.card_def_id)
            if tdef.is_junior and talent.declared_mode == CubeType.QA:
                # Junior in QA mode removes 1 bug (base) + XP bonus
                xp_bonus = sum(1 for x in talent.xp_permanent if x == CubeType.QA)
                removals = 1 + xp_bonus
                for _ in range(removals):
                    if prod.bugs > 0:
                        prod.bugs -= 1
                # Pending QA XP
                if not talent.has_pending_xp_of_type(CubeType.QA):
                    if len(talent.xp_pending) < 3 and talent.total_xp < 4:
                        talent.xp_pending.append(CubeType.QA)

        # Senior Hardware "The Fixer" - can switch to QA mode
        for tid in team:
            talent = state.talent_instances[tid]
            if talent.onboarding:
                continue
            tdef = registry.get_talent(talent.card_def_id)
            if tdef.trait == Trait.QA_SKILL and talent.declared_mode == CubeType.QA:
                if prod.bugs > 0:
                    prod.bugs -= 1


def _has_matching_lead(state: GameState, product_id: int, pdef) -> bool:
    """Check if a product has a matching lead for Tier 2/3."""
    registry = get_registry()
    team = state.get_talent_on_product(product_id)

    needs_sw_lead = pdef.cost_software > 0
    needs_hw_lead = pdef.cost_hardware > 0
    has_sw_lead = False
    has_hw_lead = False

    for tid in team:
        talent = state.talent_instances[tid]
        tdef = registry.get_talent(talent.card_def_id)

        # Check if this talent is Tier 2+
        is_tier2 = tdef.is_senior_dev or (tdef.is_junior and talent.rank_badges > 0)
        # Visionary attribute: counts as lead for any tier
        if "visionary" in talent.attributes:
            is_tier2 = True

        if not is_tier2:
            continue

        if talent.can_lead_software(tdef):
            has_sw_lead = True
        if talent.can_lead_hardware(tdef):
            has_hw_lead = True

    if pdef.is_hybrid:
        return (not needs_sw_lead or has_sw_lead) and (not needs_hw_lead or has_hw_lead)
    elif pdef.is_software_only:
        return has_sw_lead
    elif pdef.is_hardware_only:
        return has_hw_lead
    return True


# ---------------------------------------------------------------------------
# QA in Ops
# ---------------------------------------------------------------------------

def process_qa_ops(state: GameState, player_id: int) -> None:
    """QA staff in Ops remove bugs from maintenance products."""
    registry = get_registry()
    player = state.get_player(player_id)

    for pid in player.ops_products:
        prod = state.product_instances[pid]
        team = state.get_talent_on_product(pid)

        for tid in team:
            talent = state.talent_instances[tid]
            if talent.onboarding:
                continue
            tdef = registry.get_talent(talent.card_def_id)

            if tdef.talent_type == TalentType.QA:
                removals = _get_qa_bug_removal(talent, tdef)
                for _ in range(removals):
                    if prod.bugs > 0:
                        prod.bugs -= 1
                # Pending XP for junior QA in ops
                if tdef.is_junior and not talent.has_pending_xp_of_type(CubeType.QA):
                    if len(talent.xp_pending) < 3 and talent.total_xp < 4:
                        talent.xp_pending.append(CubeType.QA)


# ---------------------------------------------------------------------------
# Commit cubes
# ---------------------------------------------------------------------------

def commit_cubes(state: GameState, player_id: int) -> None:
    """Move legal cubes from transient zone to product tracks."""
    registry = get_registry()
    player = state.get_player(player_id)

    for pid in player.dev_products:
        prod = state.product_instances[pid]
        pdef = registry.get_product(prod.card_def_id)
        sw_cost, hw_cost = prod.get_effective_cost(pdef)

        # Commit software cubes (cap at effective cost)
        sw_needed = max(0, sw_cost - prod.cubes_software)
        sw_commit = min(prod.transient_software, sw_needed)
        prod.cubes_software += sw_commit

        # Commit hardware cubes
        hw_needed = max(0, hw_cost - prod.cubes_hardware)
        hw_commit = min(prod.transient_hardware, hw_needed)
        prod.cubes_hardware += hw_commit

        # Clear transient zone
        prod.transient_software = 0
        prod.transient_hardware = 0

        # Check if feature complete
        if prod.is_development_complete(pdef):
            prod.is_feature_complete = True


# ---------------------------------------------------------------------------
# Growth Hacker bug generation
# ---------------------------------------------------------------------------

def process_growth_hacker_bugs(state: GameState, player_id: int) -> None:
    """Growth Hacker adds 1 bug to their attached product at end of turn."""
    registry = get_registry()
    for tid, talent in state.talent_instances.items():
        if talent.owner != player_id or talent.zone != Zone.OPS:
            continue
        tdef = registry.get_talent(talent.card_def_id)
        if tdef.talent_type == TalentType.GROWTH_HACKER and talent.assigned_product is not None:
            prod = state.product_instances.get(talent.assigned_product)
            if prod:
                prod.bugs += 1


# ---------------------------------------------------------------------------
# Market refill
# ---------------------------------------------------------------------------

def refill_markets(state: GameState) -> None:
    """Refill agency row and product market."""
    _refill_agency_row(state)
    _refill_product_market_seed(state)
    _refill_product_market_growth(state)


def _refill_agency_row(state: GameState) -> None:
    """Refill agency row to 4 cards."""
    while len(state.markets.agency_row) < 4 and state.markets.talent_deck:
        card_def_id = state.markets.talent_deck.pop()
        inst = state.create_talent_instance(card_def_id, -1, Zone.BENCH)
        state.markets.agency_row.append(inst.instance_id)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_hand_limits(state: GameState, player_id: int) -> bool:
    """Check and enforce hand limits. Returns True if discards needed."""
    player = state.get_player(player_id)
    return (
        len(player.bench) > 5
        or len(player.strategy_hand) > 3
        or len(player.product_backlog) > 3
    )


def auto_cleanup_excess(state: GameState, player_id: int) -> None:
    """Auto-discard excess cards."""
    player = state.get_player(player_id)

    # Bench overflow: discard to open job market
    while len(player.bench) > 5:
        tid = player.bench.pop()
        talent = state.talent_instances[tid]
        talent.owner = -1
        talent.zone = Zone.BENCH
        state.markets.open_job_market.append(tid)
        while len(state.markets.open_job_market) > 5:
            state.markets.open_job_market.popleft()

    # Strategy hand overflow: discard to strategy discard pile
    while len(player.strategy_hand) > 3:
        card_id = player.strategy_hand.pop()
        state.markets.strategy_discard.append(card_id)

    # Backlog overflow: discard to open idea pool
    while len(player.product_backlog) > 3:
        pid = player.product_backlog.pop()
        prod = state.product_instances[pid]
        prod.owner = -1
        state.markets.open_idea_pool.append(pid)
        while len(state.markets.open_idea_pool) > 5:
            state.markets.open_idea_pool.popleft()


def clear_onboarding(state: GameState, player_id: int) -> None:
    """Clear onboarding tokens at end of turn."""
    for tid, talent in state.talent_instances.items():
        if talent.owner == player_id:
            talent.onboarding = False


def reset_online_status(state: GameState, player_id: int) -> None:
    """Reset all products to online at start of income phase."""
    player = state.get_player(player_id)
    for pid in player.ops_products:
        state.product_instances[pid].is_online = True


# ---------------------------------------------------------------------------
# Domain expertise
# ---------------------------------------------------------------------------

def get_domain_expertise_sectors(state: GameState, player_id: int) -> set:
    """Get sectors where the player has active maintenance products.

    Infrastructure sector does NOT generate synergy.
    """
    from ..types import Sector
    registry = get_registry()
    player = state.get_player(player_id)
    sectors = set()

    for pid in player.ops_products:
        prod = state.product_instances[pid]
        if not prod.is_online:
            continue
        pdef = registry.get_product(prod.card_def_id)
        if pdef.sector is not None and pdef.sector != Sector.INFRA:
            sectors.add(pdef.sector)

    return sectors


def apply_domain_expertise(state: GameState, player_id: int, product_def) -> tuple[int, int]:
    """Calculate effective cost after domain expertise reduction.

    Returns (effective_sw_cost, effective_hw_cost).
    -2 cubes total for matching sector. Player can split between sw/hw for hybrid.
    Minimum 1 cube total.
    """
    from ..types import Sector
    sectors = get_domain_expertise_sectors(state, player_id)

    sw_cost = product_def.cost_software
    hw_cost = product_def.cost_hardware

    if product_def.sector in sectors and product_def.sector != Sector.INFRA:
        total = sw_cost + hw_cost
        reduction = min(2, total - 1)  # Can't reduce below 1 total

        if product_def.is_hybrid:
            # For hybrid: reduce proportionally, bias toward larger cost
            if sw_cost >= hw_cost:
                sw_reduce = min(reduction, sw_cost)
                hw_reduce = min(reduction - sw_reduce, hw_cost)
            else:
                hw_reduce = min(reduction, hw_cost)
                sw_reduce = min(reduction - hw_reduce, sw_cost)
            sw_cost -= sw_reduce
            hw_cost -= hw_reduce
        elif sw_cost > 0:
            sw_cost = max(1, sw_cost - reduction) if hw_cost == 0 else sw_cost - min(reduction, sw_cost)
        else:
            hw_cost = max(1, hw_cost - reduction) if sw_cost == 0 else hw_cost - min(reduction, hw_cost)

        # Ensure minimum 1 total
        if sw_cost + hw_cost < 1:
            if product_def.cost_software > 0:
                sw_cost = 1
            else:
                hw_cost = 1

    return sw_cost, hw_cost
