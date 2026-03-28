"""Audit system: bidding, legality check, fold/settle resolution."""

from __future__ import annotations

from ..card_registry import get_registry
from ..state import GameState, ProductInstance
from ..types import CubeType, Tag, TalentType, Tier, Zone


def check_legality(state: GameState, product_id: int) -> tuple[bool, str]:
    """Check if a face-down product is legal."""
    registry = get_registry()
    prod = state.product_instances[product_id]
    pdef = registry.get_product(prod.card_def_id)

    # 1. Ghost Progress (Tier 2/3 only)
    if pdef.tier >= Tier.TIER2:
        has_transient = prod.transient_software > 0 or prod.transient_hardware > 0
        if has_transient:
            from .engine_phase import _has_matching_lead
            if not _has_matching_lead(state, product_id, pdef):
                return False, "Ghost Progress: transient cubes without matching lead"

    # 2. Missing Dependencies
    if pdef.requires:
        player_tags = state.get_player_tags_with_partners(prod.owner)
        player_tags |= prod.legacy_tags

        for tag in pdef.requires:
            if tag not in player_tags:
                return False, f"Missing dependency: {tag.name}"

    return True, "Legal"


def resolve_legal(state: GameState, product_id: int, whistleblower_id: int, bid: int) -> None:
    """Resolve a legal audit: whistleblower pays bid to owner."""
    prod = state.product_instances[product_id]
    whistleblower = state.get_player(whistleblower_id)
    owner = state.get_player(prod.owner)
    whistleblower.cash -= bid
    owner.cash += bid
    prod.is_face_down = False


def resolve_fold(state: GameState, product_id: int, whistleblower_id: int) -> None:
    """Resolve illegal-fold: project scrapped, owner pays $5 ($4 to WB, $1 to bank)."""
    prod = state.product_instances[product_id]
    owner = state.get_player(prod.owner)
    whistleblower = state.get_player(whistleblower_id)
    fine = min(owner.cash, 5)
    wb_payment = min(fine, 4)
    owner.cash -= fine
    whistleblower.cash += wb_payment
    _scrap_product(state, product_id)


def resolve_settle(state: GameState, product_id: int, whistleblower_id: int) -> None:
    """Resolve illegal-settle: project survives, owner pays $6, WB gets $4 from bank."""
    registry = get_registry()
    prod = state.product_instances[product_id]
    pdef = registry.get_product(prod.card_def_id)
    owner = state.get_player(prod.owner)
    whistleblower = state.get_player(whistleblower_id)

    modifiers = state.round_modifiers
    audit_reward = 4
    if modifiers and hasattr(modifiers, 'audit_reward_multiplier'):
        audit_reward = 4 * modifiers.audit_reward_multiplier

    settlement = min(owner.cash, 6)
    owner.cash -= settlement

    # Find tag owner for $3 payment
    if pdef.requires:
        for tag in pdef.requires:
            for other in state.players:
                if other.player_id == prod.owner:
                    continue
                if tag in state.get_player_tags(other.player_id):
                    tag_payment = min(3, settlement)
                    other.cash += tag_payment
                    break

    whistleblower.cash += audit_reward
    prod.is_face_down = False


def can_settle(state: GameState, product_id: int) -> bool:
    """Check if settling is possible (required tag exists somewhere)."""
    registry = get_registry()
    prod = state.product_instances[product_id]
    pdef = registry.get_product(prod.card_def_id)

    if not pdef.requires:
        return True

    for tag in pdef.requires:
        tag_exists = False
        for player in state.players:
            if player.player_id == prod.owner:
                continue
            if tag in state.get_player_tags(player.player_id):
                tag_exists = True
                break
        if not tag_exists:
            return False
    return True


def _scrap_product(state: GameState, product_id: int) -> None:
    """Scrap a product: discard card, return staff to bench."""
    prod = state.product_instances[product_id]
    owner = state.get_player(prod.owner)
    team = state.get_talent_on_product(product_id)
    for tid in team:
        talent = state.talent_instances[tid]
        talent.zone = Zone.BENCH
        talent.assigned_product = None
        talent.xp_pending.clear()
        talent.onboarding = True
        owner.bench.append(tid)
    if product_id in owner.dev_products:
        owner.dev_products.remove(product_id)
    state.markets.open_idea_pool.append(product_id)
    while len(state.markets.open_idea_pool) > 5:
        state.markets.open_idea_pool.popleft()
    prod.owner = -1
