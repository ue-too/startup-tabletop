"""Management actions: Brainstorm, Invest, Divest, Buyback, SecondaryTrade."""

from __future__ import annotations

import math

from ..card_registry import get_registry
from ..state import GameState
from ..types import ActionType, TalentType, Tier, Zone
from .base import Action, ActionResult


def execute_brainstorm(state: GameState, action: Action) -> ActionResult:
    """Draw strategy cards: flush optional, draw 2, keep 1.

    PM bonus: draw 3 keep 1.
    For simplification: draw N, auto-keep first, discard rest.
    Player will choose which to keep via choice parameter.
    choice: index of card to keep (0-based among drawn cards)
    """
    player = state.get_player(state.current_player)
    registry = get_registry()

    # Flush: discard strategy cards at the indices specified in target_instances
    # Indices are sorted descending to avoid shifting issues
    flush_indices = sorted(action.target_instances, reverse=True)
    for idx in flush_indices:
        if 0 <= idx < len(player.strategy_hand):
            discarded = player.strategy_hand.pop(idx)
            state.markets.strategy_discard.append(discarded)

    # Check for PM bonus (any PM on player's board)
    has_pm = False
    for tid, talent in state.talent_instances.items():
        if talent.owner == state.current_player and talent.zone in (Zone.DEV, Zone.OPS):
            tdef = registry.get_talent(talent.card_def_id)
            if tdef.is_pm and not talent.onboarding:
                has_pm = True
                break

    draw_count = 3 if has_pm else 2

    # Reshuffle if needed
    _maybe_reshuffle_strategy(state)

    # Draw cards
    drawn = []
    for _ in range(draw_count):
        if not state.markets.strategy_deck:
            _maybe_reshuffle_strategy(state)
        if state.markets.strategy_deck:
            drawn.append(state.markets.strategy_deck.pop())

    if not drawn:
        return ActionResult(False, "No strategy cards available")

    # Keep one (by choice index, default 0)
    keep_idx = max(0, min(action.choice, len(drawn) - 1)) if action.choice >= 0 else 0
    kept = drawn[keep_idx]
    player.strategy_hand.append(kept)

    # Discard rest
    for i, card_id in enumerate(drawn):
        if i != keep_idx:
            state.markets.strategy_discard.append(card_id)

    return ActionResult(True, f"Brainstormed: kept {registry.get_strategy(kept).name}")


def _maybe_reshuffle_strategy(state: GameState) -> None:
    """Reshuffle strategy discard into deck if deck is empty."""
    if not state.markets.strategy_deck and state.markets.strategy_discard:
        state.markets.strategy_deck = list(state.markets.strategy_discard)
        state.markets.strategy_discard.clear()
        # Note: we don't have RNG access here, so shuffle won't be random
        # The engine should handle reshuffling with proper RNG


def execute_invest(state: GameState, action: Action) -> ActionResult:
    """Buy equity in an opponent.

    target_player: opponent player_id
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    target_pid = action.target_player

    if target_pid == state.current_player:
        return ActionResult(False, "Cannot invest in yourself")
    if target_pid < 0 or target_pid >= state.num_players:
        return ActionResult(False, "Invalid target player")

    target = state.get_player(target_pid)

    # Check if target has equity to sell
    if target.equity_tokens_own <= 1:
        return ActionResult(False, "Target has no equity to sell (must keep 1)")

    # Determine share price based on target's highest active tier
    highest_tier = state.get_player_highest_tier(target_pid)
    if highest_tier >= Tier.TIER3:
        price = 15
    elif highest_tier >= Tier.TIER2:
        price = 10
    else:
        price = 5

    if player.cash < price:
        return ActionResult(False, f"Cannot afford ${price}")

    # Execute transaction
    player.cash -= price
    target.cash += price
    target.equity_tokens_own -= 1
    player.equity_held[target_pid] = player.equity_held.get(target_pid, 0) + 1

    return ActionResult(True, f"Invested in Player {target_pid} for ${price}")


def execute_divest(state: GameState, action: Action) -> ActionResult:
    """Sell an equity token to raise cash.

    target_player: the founder whose equity we're selling
    source_index: buyer player_id (must consent - auto-consent for now)
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    founder_id = action.target_player
    buyer_id = action.source_index

    tokens_held = player.equity_held.get(founder_id, 0)
    if tokens_held <= 0:
        return ActionResult(False, "You don't hold this equity")

    if buyer_id < 0 or buyer_id >= state.num_players:
        return ActionResult(False, "Invalid buyer")
    if buyer_id == state.current_player:
        return ActionResult(False, "Cannot sell to yourself")

    buyer = state.get_player(buyer_id)
    founder = state.get_player(founder_id)

    # Price = current share price
    highest_tier = state.get_player_highest_tier(founder_id)
    if highest_tier >= Tier.TIER3:
        price = 15
    elif highest_tier >= Tier.TIER2:
        price = 10
    else:
        price = 5

    if buyer.cash < price:
        return ActionResult(False, "Buyer cannot afford")

    # Transfer
    player.equity_held[founder_id] -= 1
    if player.equity_held[founder_id] == 0:
        del player.equity_held[founder_id]
    buyer.equity_held[founder_id] = buyer.equity_held.get(founder_id, 0) + 1
    buyer.cash -= price
    player.cash += price

    return ActionResult(True, f"Divested equity for ${price}")


def execute_buyback(state: GameState, action: Action) -> ActionResult:
    """Founder buys back their own equity from an investor.

    target_player: the investor holding our equity
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    investor_id = action.target_player

    if investor_id < 0 or investor_id >= state.num_players:
        return ActionResult(False, "Invalid investor")

    investor = state.get_player(investor_id)
    tokens_held = investor.equity_held.get(state.current_player, 0)
    if tokens_held <= 0:
        return ActionResult(False, "Investor doesn't hold your equity")

    # Price = current share price
    highest_tier = state.get_player_highest_tier(state.current_player)
    if highest_tier >= Tier.TIER3:
        price = 15
    elif highest_tier >= Tier.TIER2:
        price = 10
    else:
        price = 5

    if player.cash < price:
        return ActionResult(False, f"Cannot afford buyback at ${price}")

    # Transfer
    investor.equity_held[state.current_player] -= 1
    if investor.equity_held[state.current_player] == 0:
        del investor.equity_held[state.current_player]
    player.equity_tokens_own += 1
    player.cash -= price
    investor.cash += price

    return ActionResult(True, f"Bought back equity for ${price}")


def execute_secondary_trade(state: GameState, action: Action) -> ActionResult:
    """Investor sells equity to another investor.

    target_player: founder_id whose equity
    source_index: buyer player_id
    """
    return execute_divest(state, action)  # Same mechanics


def execute_acquisition(state: GameState, action: Action) -> ActionResult:
    """Buy an opponent's maintenance product.

    target_player: seller player_id
    target_instance: product instance_id
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    seller_id = action.target_player
    prod_id = action.target_instance

    if seller_id == state.current_player:
        return ActionResult(False, "Cannot acquire from yourself")

    seller = state.get_player(seller_id)
    if prod_id not in seller.ops_products:
        return ActionResult(False, "Product not in seller's ops")

    prod = state.product_instances[prod_id]
    pdef = registry.get_product(prod.card_def_id)

    # M&A Price
    base_prices = {Tier.TIER1: 6, Tier.TIER2: 12, Tier.TIER3: 20}
    price = base_prices.get(pdef.tier, 6)
    price += prod.hype * 5
    price -= prod.scandal * 5
    price -= prod.bugs * 1

    # Check for attached specialists
    team = state.get_talent_on_product(prod_id)
    for tid in team:
        tdef = registry.get_talent(state.talent_instances[tid].card_def_id)
        if tdef.is_specialist:
            price += 5

    price = max(1, price)

    if player.cash < price:
        return ActionResult(False, f"Cannot afford ${price}")

    # Transfer product
    player.cash -= price
    seller.cash += price
    seller.ops_products.remove(prod_id)
    player.ops_products.append(prod_id)
    prod.owner = state.current_player

    # Transfer attached staff
    for tid in team:
        talent = state.talent_instances[tid]
        talent.owner = state.current_player

    return ActionResult(True, f"Acquired {pdef.name} for ${price}")
