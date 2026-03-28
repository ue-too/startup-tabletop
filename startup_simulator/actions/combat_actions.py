"""Combat actions: Poaching (Headhunter card play)."""

from __future__ import annotations

import math

from ..card_registry import get_registry
from ..state import GameState
from ..types import ActionType, TalentType, Zone
from .base import Action, ActionResult


def calculate_poach_cost(state: GameState, talent_id: int, multiplier: float = 2.0) -> int:
    """Calculate the poaching fee for a talent.

    Total Cost = (Base Cost + Token Value) x multiplier
    Token values: XP/Skill/Attribute/Pending = +$2 each
    Rank Badge on Senior = +$2, on Junior = +$4
    """
    talent = state.talent_instances[talent_id]
    registry = get_registry()
    tdef = registry.get_talent(talent.card_def_id)

    base_cost = tdef.cost

    # Token value
    token_value = 0
    # XP tokens
    token_value += len(talent.xp_permanent) * 2
    # Skill tokens
    token_value += len(talent.skills) * 2
    # Attributes
    token_value += len(talent.attributes) * 2
    # Pending tokens
    token_value += len(talent.xp_pending) * 2

    # Rank badge
    if talent.rank_badges > 0:
        if tdef.is_junior:
            token_value += 4  # Growth premium
        else:
            token_value += 2

    # Flight Risk attribute: cost is 1x base (ignore tokens)
    if "flight_risk" in talent.attributes:
        return base_cost  # 1x base, no multiplier

    adjusted_base = base_cost + token_value

    # Mercenary trait: 1.5x instead of 2x
    if tdef.trait is not None and tdef.trait.name == "MERCENARY":
        multiplier = min(multiplier, 1.5)

    total = math.ceil(adjusted_base * multiplier)
    return total


def execute_poach(state: GameState, action: Action, multiplier: float = 2.0, bypass_hr: bool = False) -> ActionResult:
    """Execute a poach attempt.

    target_player: victim player_id
    target_instance: talent instance_id to poach
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    victim_id = action.target_player
    talent_id = action.target_instance

    if victim_id < 0 or victim_id == state.current_player:
        return ActionResult(False, "Invalid target player")
    if victim_id >= state.num_players:
        return ActionResult(False, "Invalid target player")

    talent = state.talent_instances.get(talent_id)
    if talent is None or talent.owner != victim_id:
        return ActionResult(False, "Target talent not owned by victim")

    tdef = registry.get_talent(talent.card_def_id)

    # Check targeting restrictions
    if talent.zone == Zone.BENCH:
        return ActionResult(False, "Cannot poach from bench")
    if talent.zone == Zone.OPS:
        return ActionResult(False, "Cannot poach from Ops Zone (Golden Handcuffs)")
    if talent.zone != Zone.DEV:
        return ActionResult(False, "Target must be on board (Dev)")

    # Check defenses
    victim = state.get_player(victim_id)

    # Investor immunity: cannot poach if you hold their equity
    if player.equity_held.get(victim_id, 0) > 0:
        return ActionResult(False, "Investor immunity: you hold their equity")

    # Vested interest: talent has equity token on it
    if talent.equity_vested is not None:
        return ActionResult(False, "Vested interest: talent has equity protection")

    # HR Shield: check if team has HR Manager
    if not bypass_hr and talent.assigned_product is not None:
        team = state.get_talent_on_product(talent.assigned_product)
        for tid in team:
            t = state.talent_instances[tid]
            td = registry.get_talent(t.card_def_id)
            if td.talent_type == TalentType.HR and not t.onboarding:
                return ActionResult(False, "HR Manager protects this team")

    # Tapped talent (just poached) is immune
    if talent.onboarding:
        return ActionResult(False, "Tapped talent is immune to poaching")

    # Calculate cost
    cost = calculate_poach_cost(state, talent_id, multiplier)
    if player.cash < cost:
        return ActionResult(False, f"Cannot afford poach cost ${cost}")

    # Check aggressor has open board slot
    has_slot = len(player.dev_products) > 0 or len(player.ops_products) > 0
    if not has_slot:
        return ActionResult(False, "No board slot available")

    # Pay to bank
    player.cash -= cost

    # Transfer talent
    old_product = talent.assigned_product
    talent.owner = state.current_player
    talent.onboarding = True  # Tapped: 0 output, immune to poaching
    talent.xp_pending.clear()  # Lose pending XP on transfer

    # Place on aggressor's board (first available dev product or ops)
    if player.dev_products:
        talent.assigned_product = player.dev_products[0]
        talent.zone = Zone.DEV
    elif player.ops_products:
        talent.assigned_product = player.ops_products[0]
        talent.zone = Zone.OPS

    # Impact on victim's project
    if old_product is not None:
        prod = state.product_instances.get(old_product)
        if prod and prod.zone == Zone.DEV:
            # Check if stolen talent was a lead - project may stall
            pass  # Stall check happens naturally in generate_cubes

    return ActionResult(True, f"Poached {tdef.name} for ${cost}")
