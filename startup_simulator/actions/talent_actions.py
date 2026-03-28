"""Talent-related actions: Recruit, Assign, Recall, Reassign, Layoff/Source."""

from __future__ import annotations

from ..card_registry import get_registry
from ..state import GameState, TalentInstance
from ..types import ActionType, TalentType, Tier, Zone
from .base import Action, ActionResult


def execute_recruit(state: GameState, action: Action) -> ActionResult:
    """Recruit a talent card to the bench or board.

    source_type: "university_sw", "university_hw", "agency", "open_market"
    source_index: index in the relevant market row
    """
    player = state.get_player(state.current_player)
    registry = get_registry()

    if action.source_type == "university_sw":
        cost = 2
        if player.cash < cost:
            return ActionResult(False, "Not enough cash")
        player.cash -= cost
        inst = state.create_talent_instance("jr_software", state.current_player, Zone.BENCH)
        player.bench.append(inst.instance_id)
        return ActionResult(True, f"Recruited Junior Software Dev (id={inst.instance_id})")

    elif action.source_type == "university_hw":
        cost = 2
        if player.cash < cost:
            return ActionResult(False, "Not enough cash")
        player.cash -= cost
        inst = state.create_talent_instance("jr_hardware", state.current_player, Zone.BENCH)
        player.bench.append(inst.instance_id)
        return ActionResult(True, f"Recruited Junior Hardware Eng (id={inst.instance_id})")

    elif action.source_type in ("agency", "agency_deploy"):
        idx = action.source_index
        if idx < 0 or idx >= len(state.markets.agency_row):
            return ActionResult(False, "Invalid agency index")
        talent_id = state.markets.agency_row[idx]
        talent_inst = state.talent_instances[talent_id]
        card_def = registry.get_talent(talent_inst.card_def_id)
        cost = card_def.cost
        if player.cash < cost:
            return ActionResult(False, "Not enough cash")
        player.cash -= cost
        state.markets.agency_row.pop(idx)
        talent_inst.owner = state.current_player

        if action.source_type == "agency_deploy" and action.target_instance >= 0:
            # Immediate deployment: place directly on board (Dev or Ops product)
            prod_id = action.target_instance
            prod = state.product_instances.get(prod_id)
            if prod and prod.owner == state.current_player:
                talent_inst.zone = prod.zone
                talent_inst.assigned_product = prod_id
                # Spaghetti Code trait on deploy
                if card_def.trait is not None and card_def.trait.name == "SPAGHETTI_CODE":
                    prod.bugs += 1
            else:
                talent_inst.zone = Zone.BENCH
                player.bench.append(talent_id)
        else:
            talent_inst.zone = Zone.BENCH
            player.bench.append(talent_id)

        _refill_agency(state)
        deploy_msg = " (deployed to board)" if action.source_type == "agency_deploy" else ""
        return ActionResult(True, f"Recruited {card_def.name} from Agency{deploy_msg}")

    elif action.source_type == "open_market":
        idx = action.source_index
        market_list = list(state.markets.open_job_market)
        if idx < 0 or idx >= len(market_list):
            return ActionResult(False, "Invalid open market index")
        talent_id = market_list[idx]
        cost = 1
        if player.cash < cost:
            return ActionResult(False, "Not enough cash")
        player.cash -= cost
        # Remove from open market
        state.markets.open_job_market.remove(talent_id)
        talent_inst = state.talent_instances[talent_id]
        talent_inst.owner = state.current_player
        talent_inst.zone = Zone.BENCH
        player.bench.append(talent_id)
        return ActionResult(True, "Recruited from Open Market")

    return ActionResult(False, f"Unknown source_type: {action.source_type}")


def _refill_agency(state: GameState) -> None:
    """Slide agency row left and draw from talent deck."""
    while len(state.markets.agency_row) < 4 and state.markets.talent_deck:
        card_def_id = state.markets.talent_deck.pop()
        inst = state.create_talent_instance(card_def_id, -1, Zone.BENCH)
        state.markets.agency_row.append(inst.instance_id)


def execute_assign_one(state: GameState, action: Action) -> ActionResult:
    """Assign one talent from bench to a product (dev or ops).

    target_instance: talent instance_id
    source_index: product instance_id to assign to
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    talent_id = action.target_instance
    product_id = action.source_index

    if talent_id not in player.bench:
        return ActionResult(False, "Talent not on bench")

    talent_inst = state.talent_instances[talent_id]
    tdef = registry.get_talent(talent_inst.card_def_id)

    # Determine target zone
    prod = state.product_instances[product_id]
    target_zone = prod.zone  # DEV or OPS

    # Move from bench to board
    player.bench.remove(talent_id)
    talent_inst.zone = target_zone
    talent_inst.owner = state.current_player
    talent_inst.assigned_product = product_id

    # Spaghetti Code trait: on entry, add 1 bug to the product
    if tdef.trait is not None and tdef.trait.name == "SPAGHETTI_CODE":
        prod.bugs += 1

    # PM on Tier 2/3 project: place pending rank badge
    pdef = registry.get_product(prod.card_def_id)
    if tdef.is_pm and pdef.tier >= Tier.TIER2 and talent_inst.rank_badges == 0:
        talent_inst.rank_pending = True

    return ActionResult(True, f"Assigned talent {talent_id} to product {product_id}")


def execute_recall(state: GameState, action: Action) -> ActionResult:
    """Recall talent from Ops Zone to Bench.

    Moves ALL talent from ops to bench (mass retreat).
    """
    player = state.get_player(state.current_player)

    # Find all talent in ops zone for this player
    ops_talent = [
        tid for tid, t in state.talent_instances.items()
        if t.owner == state.current_player and t.zone == Zone.OPS
    ]

    if not ops_talent:
        return ActionResult(False, "No talent in Ops Zone")

    for tid in ops_talent:
        talent_inst = state.talent_instances[tid]
        talent_inst.zone = Zone.BENCH
        talent_inst.assigned_product = None
        player.bench.append(tid)

    return ActionResult(True, f"Recalled {len(ops_talent)} talent to bench")


def execute_reassign(state: GameState, action: Action) -> ActionResult:
    """Move 1 talent between board teams.

    target_instance: talent instance_id to move
    source_index: destination product instance_id
    """
    registry = get_registry()
    talent_id = action.target_instance
    dest_product_id = action.source_index

    talent_inst = state.talent_instances[talent_id]
    tdef = registry.get_talent(talent_inst.card_def_id)
    if talent_inst.owner != state.current_player:
        return ActionResult(False, "Not your talent")
    if talent_inst.zone not in (Zone.DEV, Zone.OPS):
        return ActionResult(False, "Talent not on board")

    # Ego trait: cannot be reassigned
    if tdef.trait is not None and tdef.trait.name == "EGO":
        return ActionResult(False, "This talent has the Ego trait and cannot be reassigned")

    old_product = talent_inst.assigned_product
    talent_inst.assigned_product = dest_product_id
    talent_inst.zone = Zone.DEV
    talent_inst.onboarding = True  # Standard penalty

    # Agile PM exception: if destination team has a Senior PM (or promoted PM),
    # the reassigned talent does NOT get onboarding penalty
    dest_team = state.get_talent_on_product(dest_product_id)
    for tid in dest_team:
        if tid == talent_id:
            continue
        other = state.talent_instances[tid]
        other_def = registry.get_talent(other.card_def_id)
        # Senior PM card or promoted Junior PM
        has_agile = (
            other_def.talent_type == TalentType.SENIOR_PM
            or (other_def.is_pm and other.rank_badges > 0)
        )
        if has_agile and not other.onboarding:
            talent_inst.onboarding = False
            break

    return ActionResult(True, f"Reassigned talent {talent_id} to product {dest_product_id}")


def execute_layoff_source(state: GameState, action: Action) -> ActionResult:
    """Discard X cards from bench -> reveal X from talent deck -> hire or pass.

    target_instances: tuple of talent instance_ids to discard from bench
    For simplification: discard the specified talents, reveal same count from deck,
    and add revealed to agency row (pushing excess to open market).
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    to_discard = action.target_instances

    if not to_discard:
        return ActionResult(False, "Must discard at least 1 card")

    # Verify all are on bench
    for tid in to_discard:
        if tid not in player.bench:
            return ActionResult(False, f"Talent {tid} not on bench")

    # Discard to Open Job Market
    for tid in to_discard:
        player.bench.remove(tid)
        talent = state.talent_instances[tid]
        talent.owner = -1
        talent.zone = Zone.BENCH
        talent.assigned_product = None
        state.markets.open_job_market.append(tid)
        while len(state.markets.open_job_market) > 5:
            state.markets.open_job_market.popleft()

    # Reveal X cards from talent deck
    reveal_count = len(to_discard)
    revealed = []
    for _ in range(reveal_count):
        if not state.markets.talent_deck:
            break
        card_def_id = state.markets.talent_deck.pop()
        inst = state.create_talent_instance(card_def_id, -1, Zone.BENCH)
        revealed.append(inst.instance_id)

    # Push revealed cards into agency row from the left
    # Existing cards slide right, excess falls to Open Job Market
    new_agency = revealed + list(state.markets.agency_row)
    state.markets.agency_row = new_agency[:4]
    # Excess goes to open market
    for tid in new_agency[4:]:
        talent = state.talent_instances[tid]
        talent.owner = -1
        state.markets.open_job_market.append(tid)
        while len(state.markets.open_job_market) > 5:
            state.markets.open_job_market.popleft()

    return ActionResult(True, f"Sourced {len(revealed)} new talent cards")
