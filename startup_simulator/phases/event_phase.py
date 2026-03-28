"""Phase A: Event Phase - draw and apply event card."""

from __future__ import annotations

from ..card_registry import get_registry
from ..modifiers import RoundModifiers, parse_event_modifiers
from ..state import GameState
from ..types import Tier, Zone


def draw_event(state: GameState) -> str | None:
    """Draw an event card and set it as active. Returns the card_def_id."""
    # Discard previous event
    if state.markets.active_event is not None:
        state.markets.event_discard.append(state.markets.active_event)
        state.markets.active_event = None

    if not state.markets.event_deck:
        return None

    card_id = state.markets.event_deck.pop()
    state.markets.active_event = card_id
    return card_id


def get_round_modifiers(state: GameState) -> RoundModifiers:
    """Get the RoundModifiers for the current event."""
    if state.markets.active_event is None:
        return RoundModifiers()

    registry = get_registry()
    event_def = registry.get_event(state.markets.active_event)
    return parse_event_modifiers(event_def.effect_id)


def apply_immediate_event_effects(state: GameState) -> None:
    """Apply one-time immediate effects of the current event."""
    if state.markets.active_event is None:
        return

    registry = get_registry()
    event_def = registry.get_event(state.markets.active_event)

    if event_def.effect_id == "tier1_only_bonus_3":
        # Players with only Tier 1 products get $3
        for player in state.players:
            highest = state.get_player_highest_tier(player.player_id)
            has_any = len(player.ops_products) > 0
            if has_any and highest <= Tier.TIER1:
                player.cash += 3

    elif event_def.effect_id == "payroll_tax":
        # Pay $1 per employee with salary > $0
        for player in state.players:
            tax = 0
            for tid, t in state.talent_instances.items():
                if t.owner != player.player_id:
                    continue
                if t.zone not in (Zone.DEV, Zone.OPS):
                    continue
                tdef = registry.get_talent(t.card_def_id)
                if tdef.is_junior:
                    if t.salary > 0:
                        tax += 1
                elif tdef.salary > 0:
                    tax += 1
            if tax > 0:
                if player.cash >= tax:
                    player.cash -= tax
                else:
                    # Can't pay: simplified - just take what they have
                    player.cash = 0
