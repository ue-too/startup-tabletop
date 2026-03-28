"""Product-related actions: Ideation, Greenlight, Launch, Pivot."""

from __future__ import annotations

from ..card_registry import get_registry
from ..state import GameState, ProductInstance
from ..types import ActionType, CubeType, Tier, Zone
from .base import Action, ActionResult


def execute_ideation(state: GameState, action: Action) -> ActionResult:
    """Draft a product card to backlog.

    source_type: "seed_market", "growth_market", "seed_deck", "growth_deck", "idea_pool"
    source_index: index in the relevant market row or pool
    """
    player = state.get_player(state.current_player)
    registry = get_registry()

    # Check backlog limit (allow overflow to 4, must discard during cleanup)
    if len(player.product_backlog) >= 4:
        return ActionResult(False, "Backlog overflow already at max")

    if action.source_type == "seed_market":
        idx = action.source_index
        if idx < 0 or idx >= len(state.markets.product_market_seed):
            return ActionResult(False, "Invalid seed market index")
        prod_id = state.markets.product_market_seed.pop(idx)
        prod_inst = state.product_instances[prod_id]
        prod_inst.owner = state.current_player
        player.product_backlog.append(prod_id)
        _refill_product_market_seed(state)
        return ActionResult(True, f"Drafted {registry.get_product(prod_inst.card_def_id).name}")

    elif action.source_type == "growth_market":
        idx = action.source_index
        if idx < 0 or idx >= len(state.markets.product_market_growth):
            return ActionResult(False, "Invalid growth market index")
        prod_id = state.markets.product_market_growth.pop(idx)
        prod_inst = state.product_instances[prod_id]
        pdef = registry.get_product(prod_inst.card_def_id)
        if pdef.is_market_crash:
            state.market_crash_drawn = True
            state.finish_round = True
            return ActionResult(True, "MARKET CRASH DRAWN!", game_over=False)
        prod_inst.owner = state.current_player
        player.product_backlog.append(prod_id)
        _refill_product_market_growth(state)
        return ActionResult(True, f"Drafted {pdef.name}")

    elif action.source_type == "seed_deck":
        # Blind draft: draw 3 cards, keep 1 (by choice index), discard 2 to Idea Pool
        if not state.markets.seed_deck:
            return ActionResult(False, "Seed deck empty")
        drawn = []
        for _ in range(3):
            if state.markets.seed_deck:
                drawn.append(state.markets.seed_deck.pop())
        if not drawn:
            return ActionResult(False, "Seed deck empty")
        keep_idx = max(0, min(action.choice, len(drawn) - 1)) if action.choice >= 0 else 0
        kept_id = drawn[keep_idx]
        pdef = registry.get_product(kept_id)
        prod_inst = state.create_product_instance(kept_id, state.current_player, Zone.BENCH)
        player.product_backlog.append(prod_inst.instance_id)
        for i, cid in enumerate(drawn):
            if i != keep_idx:
                discard_inst = state.create_product_instance(cid, -1, Zone.BENCH)
                _add_to_idea_pool(state, discard_inst.instance_id)
        return ActionResult(True, f"Drafted {pdef.name} from Seed Deck (blind)")

    elif action.source_type == "growth_deck":
        # Blind draft: draw 3, keep 1, discard 2
        if not state.markets.growth_deck:
            return ActionResult(False, "Growth deck empty")
        drawn = []
        for _ in range(3):
            if state.markets.growth_deck:
                cid = state.markets.growth_deck.pop()
                pdef_check = registry.get_product(cid)
                if pdef_check.is_market_crash:
                    state.market_crash_drawn = True
                    state.finish_round = True
                    continue
                drawn.append(cid)
        if not drawn:
            if state.market_crash_drawn:
                return ActionResult(True, "MARKET CRASH DRAWN!", game_over=False)
            return ActionResult(False, "Growth deck empty")
        keep_idx = max(0, min(action.choice, len(drawn) - 1)) if action.choice >= 0 else 0
        kept_id = drawn[keep_idx]
        pdef = registry.get_product(kept_id)
        prod_inst = state.create_product_instance(kept_id, state.current_player, Zone.BENCH)
        player.product_backlog.append(prod_inst.instance_id)
        for i, cid in enumerate(drawn):
            if i != keep_idx:
                discard_inst = state.create_product_instance(cid, -1, Zone.BENCH)
                _add_to_idea_pool(state, discard_inst.instance_id)
        return ActionResult(True, f"Drafted {pdef.name} from Growth Deck (blind)")

    elif action.source_type == "idea_pool":
        idx = action.source_index
        pool_list = list(state.markets.open_idea_pool)
        if idx < 0 or idx >= len(pool_list):
            return ActionResult(False, "Invalid idea pool index")
        prod_id = pool_list[idx]
        state.markets.open_idea_pool.remove(prod_id)
        prod_inst = state.product_instances[prod_id]
        prod_inst.owner = state.current_player
        player.product_backlog.append(prod_id)
        return ActionResult(True, "Drafted from Open Idea Pool")

    return ActionResult(False, f"Unknown source_type: {action.source_type}")


def execute_greenlight(state: GameState, action: Action) -> ActionResult:
    """Move product from backlog to development zone (0 AP).

    target_instance: product instance_id in backlog
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    prod_id = action.target_instance

    if prod_id not in player.product_backlog:
        return ActionResult(False, "Product not in backlog")
    if len(player.dev_products) >= 3:
        return ActionResult(False, "Dev zone full (max 3)")

    prod_inst = state.product_instances[prod_id]
    pdef = registry.get_product(prod_inst.card_def_id)

    # Dependency check
    paid_license = False
    if pdef.requires:
        player_tags = state.get_player_tags_with_partners(state.current_player)
        has_all_tags = all(t in player_tags for t in pdef.requires)

        if not has_all_tags:
            missing = [t for t in pdef.requires if t not in player_tags]
            for missing_tag in missing:
                tag_found = False
                for other_p in state.players:
                    if other_p.player_id == state.current_player:
                        continue
                    if missing_tag in state.get_player_tags(other_p.player_id):
                        tag_found = True
                        if player.cash < 3:
                            return ActionResult(False, f"Cannot afford $3 license for {missing_tag.name}")
                        player.cash -= 3
                        other_p.cash += 3
                        paid_license = True
                        break
                if not tag_found:
                    return ActionResult(False, f"No one has tag {missing_tag.name}")

    # Apply domain expertise cost reduction
    from ..phases.engine_phase import apply_domain_expertise
    eff_sw, eff_hw = apply_domain_expertise(state, state.current_player, pdef)
    prod_inst.effective_cost_software = eff_sw
    prod_inst.effective_cost_hardware = eff_hw

    # Move to dev
    player.product_backlog.remove(prod_id)
    player.dev_products.append(prod_id)
    prod_inst.zone = Zone.DEV

    # Stealth: face-down unless license was paid
    prod_inst.is_face_down = not paid_license

    return ActionResult(True, f"Greenlit {pdef.name}")


def execute_launch(state: GameState, action: Action) -> ActionResult:
    """Launch a completed product from Dev to Ops.

    target_instance: product instance_id in dev zone
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    prod_id = action.target_instance

    if prod_id not in player.dev_products:
        return ActionResult(False, "Product not in dev zone")

    prod_inst = state.product_instances[prod_id]
    pdef = registry.get_product(prod_inst.card_def_id)

    # Check feature complete
    if not prod_inst.is_development_complete(pdef):
        return ActionResult(False, "Product not feature complete")

    # Check 0 bugs
    if prod_inst.bugs > 0:
        return ActionResult(False, f"Product has {prod_inst.bugs} bugs")

    # Lead check for Tier 2/3
    if pdef.tier >= Tier.TIER2:
        from ..phases.engine_phase import _has_matching_lead
        if not _has_matching_lead(state, prod_id, pdef):
            if pdef.is_hybrid:
                return ActionResult(False, "Hybrid Tier 2/3 needs both SW and HW leads (or cross-functional)")
            return ActionResult(False, "Tier 2/3 product requires a matching Lead")

    # Move to ops
    player.dev_products.remove(prod_id)
    player.ops_products.append(prod_id)
    prod_inst.zone = Zone.OPS
    prod_inst.is_online = True

    # Move assigned staff to ops
    team = state.get_talent_on_product(prod_id)
    for tid in team:
        talent = state.talent_instances[tid]
        talent.zone = Zone.OPS

    # XP Graduation for juniors
    for tid in team:
        talent = state.talent_instances[tid]
        tdef = registry.get_talent(talent.card_def_id)

        if tdef.is_junior and talent.xp_pending:
            if talent.total_xp < 4:
                # Auto-graduate: pick the first pending XP
                # (In a full implementation, this would yield to the player for choice.
                #  For now, auto-pick matching native type, or first available.)
                chosen = None
                # Prefer XP matching native output type
                if tdef.output_type is not None:
                    for xp in talent.xp_pending:
                        if xp == tdef.output_type:
                            chosen = xp
                            break
                if chosen is None:
                    chosen = talent.xp_pending[0]
                talent.xp_permanent.append(chosen)

                # Mentor trait: extra +1 XP on launch
                for other_tid in team:
                    other = state.talent_instances[other_tid]
                    other_def = registry.get_talent(other.card_def_id)
                    if other_def.trait is not None and other_def.trait.name == "MENTOR" and other_tid != tid:
                        if talent.total_xp < 4 and talent.xp_pending:
                            # Grant one more XP from remaining pending
                            remaining = [x for x in talent.xp_pending if x != chosen]
                            if remaining:
                                talent.xp_permanent.append(remaining[0])
                        break

            talent.xp_pending.clear()
        elif not tdef.is_junior:
            # Seniors don't gain XP from launches
            talent.xp_pending.clear()

    # PM Rank promotion: Junior PM on Tier 2/3 project gets permanent rank
    for tid in team:
        talent = state.talent_instances[tid]
        tdef = registry.get_talent(talent.card_def_id)
        if tdef.is_pm and talent.rank_pending and pdef.tier >= Tier.TIER2:
            talent.rank_badges = 1
            talent.rank_pending = False

    # Stealth launch bonus
    if prod_inst.is_face_down:
        if pdef.tier == Tier.TIER2:
            prod_inst.stealth_launch_bonus = 5
        elif pdef.tier == Tier.TIER3:
            prod_inst.stealth_launch_bonus = 10
        prod_inst.is_face_down = False

    return ActionResult(True, f"Launched {pdef.name}")


def execute_pivot(state: GameState, action: Action) -> ActionResult:
    """Scrap a development project.

    target_instance: product instance_id in dev zone
    """
    player = state.get_player(state.current_player)
    prod_id = action.target_instance

    if prod_id not in player.dev_products:
        return ActionResult(False, "Product not in dev zone")

    prod_inst = state.product_instances[prod_id]

    # Return staff to bench
    team = state.get_talent_on_product(prod_id)
    for tid in team:
        talent = state.talent_instances[tid]
        talent.zone = Zone.BENCH
        talent.assigned_product = None
        talent.onboarding = True  # Benched staff get onboarding penalty
        talent.xp_pending.clear()  # Lose all pending XP
        player.bench.append(tid)

    # Discard product to Open Idea Pool
    player.dev_products.remove(prod_id)
    _add_to_idea_pool(state, prod_id)

    return ActionResult(True, "Pivoted/scrapped project")


def _refill_product_market_seed(state: GameState) -> None:
    """Refill seed market slots."""
    while len(state.markets.product_market_seed) < 2 and state.markets.seed_deck:
        card_def_id = state.markets.seed_deck.pop()
        inst = state.create_product_instance(card_def_id, -1, Zone.BENCH)
        state.markets.product_market_seed.append(inst.instance_id)


def _refill_product_market_growth(state: GameState) -> None:
    """Refill growth market slots. May draw Market Crash."""
    registry = get_registry()
    while len(state.markets.product_market_growth) < 2 and state.markets.growth_deck:
        card_def_id = state.markets.growth_deck.pop()
        pdef = registry.get_product(card_def_id)
        if pdef.is_market_crash:
            state.market_crash_drawn = True
            state.finish_round = True
            continue
        inst = state.create_product_instance(card_def_id, -1, Zone.BENCH)
        state.markets.product_market_growth.append(inst.instance_id)


def _add_to_idea_pool(state: GameState, prod_id: int) -> None:
    """Add product to Open Idea Pool (FIFO, max 5)."""
    state.markets.open_idea_pool.append(prod_id)
    while len(state.markets.open_idea_pool) > 5:
        state.markets.open_idea_pool.popleft()  # Remove oldest
