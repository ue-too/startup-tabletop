"""Legal action enumeration for Startup Simulator."""

from __future__ import annotations

from itertools import combinations

from ..card_registry import get_registry
from ..state import GameState
from ..types import ActionType, CubeType, Phase, SubPhase, TalentType, Tier, Zone, IDEATION_POOL_AP_COST
from .base import Action


def get_legal_actions(state: GameState) -> list[Action]:
    """Return all legal actions for the current decision-maker."""
    if state.game_over:
        return []

    if state.phase == Phase.ACTION:
        if state.sub_phase == SubPhase.ACTION_ASSIGN_BATCH:
            return _get_assign_batch_actions(state)
        if state.sub_phase == SubPhase.ACTION_MAIN:
            return _get_main_actions(state)

    if state.phase == Phase.ENGINE:
        if state.sub_phase == SubPhase.ENGINE_MODE_DECLARE:
            return _get_mode_declare_actions(state)
        if state.sub_phase == SubPhase.ENGINE_AUDIT_BID:
            return _get_audit_bid_actions(state)
        if state.sub_phase == SubPhase.ENGINE_AUDIT_RESOLVE:
            return _get_audit_resolve_actions(state)
        if state.sub_phase == SubPhase.ENGINE_XP_GRADUATE:
            return _get_xp_graduate_actions(state)
        if state.sub_phase == SubPhase.ENGINE_CLEANUP_TALENT:
            return _get_talent_discard_actions(state)
        if state.sub_phase == SubPhase.ENGINE_CLEANUP_BACKLOG:
            return _get_backlog_discard_actions(state)

    if state.phase == Phase.INCOME:
        if state.sub_phase == SubPhase.INCOME_CHOOSE_OFFLINE:
            return _get_choose_offline_actions(state)
        if state.sub_phase == SubPhase.INCOME_FIRE_CHOICE:
            return _get_fire_choice_actions(state)

    # Default: only PASS
    return [Action(ActionType.PASS)]


def _get_main_actions(state: GameState) -> list[Action]:
    """Get all legal actions during the main action phase."""
    actions: list[Action] = []
    player = state.get_player(state.current_player)
    registry = get_registry()
    ap = player.action_points

    # Always can pass (ends action phase)
    actions.append(Action(ActionType.PASS))

    if ap <= 0:
        # Only free actions available
        _add_free_actions(state, actions)
        return actions

    # --- 1 AP Actions ---

    # RECRUIT
    _add_recruit_actions(state, actions)

    # ASSIGN (starts batch)
    if player.bench and (player.dev_products or player.ops_products):
        actions.append(Action(ActionType.ASSIGN))

    # RECALL (from ops)
    ops_talent = [
        tid for tid, t in state.talent_instances.items()
        if t.owner == state.current_player and t.zone == Zone.OPS
    ]
    if ops_talent:
        actions.append(Action(ActionType.RECALL))

    # REASSIGN
    board_talent = state.get_board_talent(state.current_player)
    all_products = player.dev_products + player.ops_products
    for tid in board_talent:
        t = state.talent_instances[tid]
        tdef = registry.get_talent(t.card_def_id)
        # Ego trait: cannot reassign
        if tdef.trait is not None and tdef.trait.name == "EGO":
            continue
        for pid in all_products:
            if pid != t.assigned_product:
                actions.append(Action(
                    ActionType.REASSIGN,
                    target_instance=tid,
                    source_index=pid,
                ))

    # LAYOFF/SOURCE
    _add_layoff_source_actions(state, actions)

    # IDEATION
    _add_ideation_actions(state, actions)

    # LAUNCH
    for pid in player.dev_products:
        prod = state.product_instances[pid]
        pdef = registry.get_product(prod.card_def_id)
        if prod.is_development_complete(pdef) and prod.bugs == 0:
            # Also check lead requirement for Tier 2/3
            if pdef.tier >= Tier.TIER2:
                from ..phases.engine_phase import _has_matching_lead
                if not _has_matching_lead(state, pid, pdef):
                    continue
            actions.append(Action(ActionType.LAUNCH, target_instance=pid))

    # PIVOT
    for pid in player.dev_products:
        actions.append(Action(ActionType.PIVOT, target_instance=pid))

    # BRAINSTORM (draw strategy cards)
    if state.markets.strategy_deck or state.markets.strategy_discard:
        # Offer different keep choices (0, 1, 2 for PM)
        for choice in range(3):
            actions.append(Action(ActionType.BRAINSTORM, choice=choice))

    # INVEST (buy equity in opponent)
    _add_invest_actions(state, actions)

    # BUYBACK (buy back own equity from investor)
    _add_buyback_actions(state, actions)

    # DIVEST / SECONDARY TRADE
    _add_divest_actions(state, actions)

    # ACQUISITION (buy opponent's maintenance product)
    _add_acquisition_actions(state, actions)

    # --- Free Actions (0 AP) ---
    _add_free_actions(state, actions)

    return actions


def _add_recruit_actions(state: GameState, actions: list[Action]) -> None:
    """Add recruit actions from all sources."""
    player = state.get_player(state.current_player)
    registry = get_registry()

    # University (always available, infinite supply)
    if player.cash >= 2:
        actions.append(Action(ActionType.RECRUIT, source_type="university_sw"))
        actions.append(Action(ActionType.RECRUIT, source_type="university_hw"))

    # Agency Row (bench or immediate deploy to board)
    all_products = player.dev_products + player.ops_products
    for i, tid in enumerate(state.markets.agency_row):
        talent = state.talent_instances[tid]
        cdef = registry.get_talent(talent.card_def_id)
        if player.cash >= cdef.cost:
            actions.append(Action(ActionType.RECRUIT, source_index=i, source_type="agency"))
            # Immediate deployment option (to any existing product)
            for pid in all_products:
                actions.append(Action(
                    ActionType.RECRUIT, source_index=i,
                    source_type="agency_deploy", target_instance=pid,
                ))

    # Open Job Market
    for i, tid in enumerate(state.markets.open_job_market):
        if player.cash >= 1:
            actions.append(Action(ActionType.RECRUIT, source_index=i, source_type="open_market"))


def _add_layoff_source_actions(state: GameState, actions: list[Action]) -> None:
    """Add layoff/source actions. Must discard 1+ from bench."""
    player = state.get_player(state.current_player)
    if not player.bench or not state.markets.talent_deck:
        return
    # Offer discarding 1, 2, or 3 bench cards (up to bench size)
    max_discard = min(len(player.bench), 3)
    for count in range(1, max_discard + 1):
        for combo in combinations(player.bench, count):
            actions.append(Action(
                ActionType.LAYOFF_SOURCE,
                target_instances=combo,
            ))


def _add_ideation_actions(state: GameState, actions: list[Action]) -> None:
    """Add ideation actions from all sources."""
    player = state.get_player(state.current_player)

    # Cannot ideate if backlog is at 4 (overflow already happened)
    if len(player.product_backlog) >= 4:
        return

    # Seed market
    for i in range(len(state.markets.product_market_seed)):
        actions.append(Action(ActionType.IDEATION, source_index=i, source_type="seed_market"))

    # Growth market
    for i in range(len(state.markets.product_market_growth)):
        actions.append(Action(ActionType.IDEATION, source_index=i, source_type="growth_market"))

    # Blind draft from decks
    if state.markets.seed_deck:
        actions.append(Action(ActionType.IDEATION, source_type="seed_deck"))
    if state.markets.growth_deck:
        actions.append(Action(ActionType.IDEATION, source_type="growth_deck"))

    # Open Idea Pool (costs 2 AP)
    if player.action_points >= IDEATION_POOL_AP_COST:
        for i in range(len(state.markets.open_idea_pool)):
            actions.append(Action(ActionType.IDEATION, source_index=i, source_type="idea_pool"))


def _tier_to_price(tier: Tier) -> int:
    if tier >= Tier.TIER3:
        return 15
    elif tier >= Tier.TIER2:
        return 10
    return 5


def _add_invest_actions(state: GameState, actions: list[Action]) -> None:
    """Add invest actions for each opponent with equity to sell."""
    player = state.get_player(state.current_player)
    for other in state.players:
        if other.player_id == state.current_player:
            continue
        if other.equity_tokens_own <= 1:
            continue
        price = _tier_to_price(state.get_player_highest_tier(other.player_id))
        if player.cash >= price:
            actions.append(Action(ActionType.INVEST, target_player=other.player_id))


def _add_buyback_actions(state: GameState, actions: list[Action]) -> None:
    """Add buyback actions for each investor holding our equity."""
    player = state.get_player(state.current_player)
    price = _tier_to_price(state.get_player_highest_tier(state.current_player))

    for other in state.players:
        if other.player_id == state.current_player:
            continue
        if other.equity_held.get(state.current_player, 0) > 0 and player.cash >= price:
            actions.append(Action(ActionType.BUYBACK, target_player=other.player_id))


def _add_divest_actions(state: GameState, actions: list[Action]) -> None:
    """Add divest/secondary trade actions."""
    player = state.get_player(state.current_player)

    for founder_id, count in player.equity_held.items():
        if count <= 0:
            continue
        price = _tier_to_price(state.get_player_highest_tier(founder_id))
        for buyer in state.players:
            if buyer.player_id == state.current_player:
                continue
            if buyer.cash >= price:
                actions.append(Action(
                    ActionType.DIVEST,
                    target_player=founder_id,
                    source_index=buyer.player_id,
                ))


def _add_acquisition_actions(state: GameState, actions: list[Action]) -> None:
    """Add acquisition actions for opponent maintenance products."""
    player = state.get_player(state.current_player)
    registry = get_registry()

    for other in state.players:
        if other.player_id == state.current_player:
            continue
        for pid in other.ops_products:
            prod = state.product_instances[pid]
            pdef = registry.get_product(prod.card_def_id)
            base_prices = {Tier.TIER1: 6, Tier.TIER2: 12, Tier.TIER3: 20}
            price = base_prices.get(pdef.tier, 6)
            price += prod.hype * 5 - prod.scandal * 5 - prod.bugs
            price = max(1, price)
            if player.cash >= price:
                actions.append(Action(
                    ActionType.ACQUISITION,
                    target_player=other.player_id,
                    target_instance=pid,
                ))


def _add_free_actions(state: GameState, actions: list[Action]) -> None:
    """Add free (0 AP) actions."""
    player = state.get_player(state.current_player)
    registry = get_registry()

    # GREENLIGHT (backlog -> dev)
    if len(player.dev_products) < 3:
        for pid in player.product_backlog:
            if _can_greenlight(state, pid):
                actions.append(Action(ActionType.GREENLIGHT, target_instance=pid))

    # PLAY STRATEGY CARD
    _add_play_strategy_actions(state, actions)

    # INTEGRATE (stack host/client in ops)
    _add_integrate_actions(state, actions)

    # VOLUNTARY DISCLOSURE (flip face-down card)
    for pid in player.dev_products:
        prod = state.product_instances[pid]
        if prod.is_face_down:
            actions.append(Action(ActionType.VOLUNTARY_DISCLOSURE, target_instance=pid))


def _can_greenlight(state: GameState, prod_id: int) -> bool:
    """Check if a product can be greenlit (dependency tags available)."""
    registry = get_registry()
    prod = state.product_instances[prod_id]
    pdef = registry.get_product(prod.card_def_id)

    if not pdef.requires:
        return True

    player = state.get_player(state.current_player)
    player_tags = state.get_player_tags_with_partners(state.current_player)

    if all(t in player_tags for t in pdef.requires):
        return True

    # Check if missing tags can be licensed ($3 each)
    for tag in pdef.requires:
        if tag in player_tags:
            continue
        found = False
        for other in state.players:
            if other.player_id == state.current_player:
                continue
            if tag in state.get_player_tags(other.player_id):
                found = True
                break
        if not found:
            return False

    missing_count = sum(1 for t in pdef.requires if t not in player_tags)
    return player.cash >= missing_count * 3


def _add_integrate_actions(state: GameState, actions: list[Action]) -> None:
    """Add integration actions for valid host/client pairs in ops."""
    player = state.get_player(state.current_player)
    registry = get_registry()

    ops_products = [(pid, state.product_instances[pid]) for pid in player.ops_products]

    for i, (host_id, host) in enumerate(ops_products):
        if host.integrated_with is not None:
            continue
        host_def = registry.get_product(host.card_def_id)
        for j, (client_id, client) in enumerate(ops_products):
            if i == j or client.integrated_with is not None:
                continue
            client_def = registry.get_product(client.card_def_id)
            # Check compatibility
            from .free_actions import _is_valid_integration
            if _is_valid_integration(host_def, client_def, registry):
                # Same tag check
                if host_def.provides != client_def.provides:
                    actions.append(Action(
                        ActionType.INTEGRATE,
                        target_instance=host_id,
                        source_index=client_id,
                    ))


def _add_play_strategy_actions(state: GameState, actions: list[Action]) -> None:
    """Add play strategy card actions."""
    player = state.get_player(state.current_player)
    registry = get_registry()

    for idx, card_id in enumerate(player.strategy_hand):
        sdef = registry.get_strategy(card_id)
        if player.cash < sdef.cost:
            continue

        # Generate valid targets based on effect
        if sdef.effect_id == "train_software_skill":
            for tid in _get_own_juniors(state):
                talent = state.talent_instances[tid]
                tdef = registry.get_talent(talent.card_def_id)
                if CubeType.SOFTWARE not in talent.skills and tdef.output_type != CubeType.SOFTWARE:
                    actions.append(Action(ActionType.PLAY_STRATEGY, source_index=idx, target_instance=tid))

        elif sdef.effect_id == "train_qa_skill":
            for tid in _get_own_juniors(state):
                talent = state.talent_instances[tid]
                if CubeType.QA not in talent.skills:
                    actions.append(Action(ActionType.PLAY_STRATEGY, source_index=idx, target_instance=tid))

        elif sdef.effect_id == "train_specialist_xp":
            for tid, t in state.talent_instances.items():
                if t.owner != state.current_player:
                    continue
                tdef = registry.get_talent(t.card_def_id)
                if tdef.talent_type in (TalentType.QA, TalentType.SALES) and t.total_xp < 2:
                    actions.append(Action(ActionType.PLAY_STRATEGY, source_index=idx, target_instance=tid))

        elif sdef.effect_id == "add_rank_badge":
            for tid in _get_own_juniors(state):
                talent = state.talent_instances[tid]
                if talent.rank_badges == 0:
                    actions.append(Action(ActionType.PLAY_STRATEGY, source_index=idx, target_instance=tid))

        elif sdef.effect_id == "add_pm_rank_badge":
            for tid, t in state.talent_instances.items():
                if t.owner != state.current_player:
                    continue
                tdef = registry.get_talent(t.card_def_id)
                if tdef.is_pm and t.rank_badges == 0:
                    actions.append(Action(ActionType.PLAY_STRATEGY, source_index=idx, target_instance=tid))

        elif sdef.effect_id in ("poach_2x", "poach_1_5x_bypass_hr"):
            # Poach: target opponent's dev board talent
            from .combat_actions import calculate_poach_cost
            multiplier = 1.5 if "1_5x" in sdef.effect_id else 2.0
            bypass_hr = "bypass_hr" in sdef.effect_id
            for other in state.players:
                if other.player_id == state.current_player:
                    continue
                if player.equity_held.get(other.player_id, 0) > 0:
                    continue  # Investor immunity
                for tid, t in state.talent_instances.items():
                    if t.owner == other.player_id and t.zone == Zone.DEV and not t.onboarding:
                        if t.equity_vested is not None:
                            continue  # Vested
                        # Check HR shield
                        if not bypass_hr and t.assigned_product is not None:
                            team = state.get_talent_on_product(t.assigned_product)
                            has_hr = any(
                                registry.get_talent(state.talent_instances[x].card_def_id).talent_type == TalentType.HR
                                for x in team if not state.talent_instances[x].onboarding
                            )
                            if has_hr:
                                continue
                        # Check cost
                        cost = calculate_poach_cost(state, tid, multiplier) + sdef.cost
                        if player.cash < cost:
                            continue
                        actions.append(Action(
                            ActionType.PLAY_STRATEGY, source_index=idx,
                            target_instance=tid, target_player=other.player_id,
                        ))

        elif sdef.effect_id == "add_scandal":
            for other in state.players:
                if other.player_id == state.current_player:
                    continue
                if player.equity_held.get(other.player_id, 0) > 0:
                    continue
                for pid in other.ops_products:
                    actions.append(Action(
                        ActionType.PLAY_STRATEGY, source_index=idx,
                        target_instance=pid, target_player=other.player_id,
                    ))

        elif sdef.effect_id == "hostile_buyout":
            for other in state.players:
                if other.player_id == state.current_player:
                    continue
                if other.equity_held.get(state.current_player, 0) > 0 and player.cash >= 5:
                    actions.append(Action(
                        ActionType.PLAY_STRATEGY, source_index=idx,
                        target_player=other.player_id,
                    ))

        elif sdef.effect_id.startswith("attr_"):
            attr_name = sdef.effect_id.replace("attr_", "")
            if attr_name in ("workaholic", "clean_coder", "visionary"):
                # Buff own talent
                for tid, t in state.talent_instances.items():
                    if t.owner == state.current_player:
                        actions.append(Action(ActionType.PLAY_STRATEGY, source_index=idx, target_instance=tid))
            else:
                # Debuff opponent talent
                for other in state.players:
                    if other.player_id == state.current_player:
                        continue
                    if player.equity_held.get(other.player_id, 0) > 0:
                        continue
                    for tid, t in state.talent_instances.items():
                        if t.owner == other.player_id:
                            actions.append(Action(
                                ActionType.PLAY_STRATEGY, source_index=idx,
                                target_instance=tid, target_player=other.player_id,
                            ))

        elif sdef.effect_id == "add_hype":
            for pid in player.ops_products:
                actions.append(Action(ActionType.PLAY_STRATEGY, source_index=idx, target_instance=pid))

        elif sdef.effect_id == "draw_5_products":
            if state.markets.seed_deck or state.markets.growth_deck:
                actions.append(Action(ActionType.PLAY_STRATEGY, source_index=idx))

        elif sdef.effect_id == "cancel_attack":
            # Reaction card: don't generate actions for it (held in hand)
            pass


def _get_own_juniors(state: GameState) -> list[int]:
    """Get all junior talent instance_ids owned by current player."""
    registry = get_registry()
    juniors = []
    for tid, t in state.talent_instances.items():
        if t.owner != state.current_player:
            continue
        tdef = registry.get_talent(t.card_def_id)
        if tdef.is_junior:
            juniors.append(tid)
    return juniors


def _get_assign_batch_actions(state: GameState) -> list[Action]:
    """Get legal assign-one actions within an assign batch."""
    actions: list[Action] = []
    player = state.get_player(state.current_player)

    # Can always end the batch
    actions.append(Action(ActionType.END_ASSIGN_BATCH))

    # Assign each bench talent to each dev/ops product
    all_products = player.dev_products + player.ops_products
    for tid in player.bench:
        for pid in all_products:
            actions.append(Action(
                ActionType.ASSIGN_ONE,
                target_instance=tid,
                source_index=pid,
            ))

    return actions


def _get_audit_bid_actions(state: GameState) -> list[Action]:
    """Get audit bid/pass actions for the current opponent."""
    actions: list[Action] = [Action(ActionType.PASS_AUDIT)]
    player = state.get_player(state.current_player)

    # Find face-down products of the active player being audited
    if state.pending_decisions:
        ctx = state.pending_decisions[0].context
        active_pid = ctx.get("active_player")
        if active_pid is not None:
            active = state.get_player(active_pid)
            for pid in active.dev_products:
                prod = state.product_instances[pid]
                if prod.is_face_down:
                    # Bid at $3, $4, $5, $6 increments
                    for bid in range(3, min(player.cash + 1, 10)):
                        actions.append(Action(ActionType.BID_AUDIT, target_instance=pid, amount=bid))

    return actions


def _get_audit_resolve_actions(state: GameState) -> list[Action]:
    """Get fold/settle actions for audit resolution (owner decides)."""
    actions: list[Action] = [Action(ActionType.FOLD)]
    prod_id = state.audit_target_product
    if prod_id is not None:
        from ..phases.audit_phase import can_settle
        if can_settle(state, prod_id):
            owner = state.product_instances[prod_id].owner
            if state.players[owner].cash >= 6:
                actions.append(Action(ActionType.SETTLE))
    return actions


def _get_mode_declare_actions(state: GameState) -> list[Action]:
    """Get mode declaration actions for multi-skilled/flex talent."""
    actions: list[Action] = []
    registry = get_registry()
    player = state.get_player(state.current_player)

    # Find the talent that needs mode declaration (stored in pending_decisions context)
    if state.pending_decisions:
        decision = state.pending_decisions[0]
        tid = decision.context.get("talent_id")
        if tid is not None:
            talent = state.talent_instances[tid]
            tdef = registry.get_talent(talent.card_def_id)

            # Available modes: native output type + trained skills
            modes = set()
            if tdef.output_type is not None and tdef.output_type != CubeType.QA:
                modes.add(tdef.output_type)
            if tdef.is_flex:
                modes.add(CubeType.SOFTWARE)
                modes.add(CubeType.HARDWARE)
            for skill in talent.skills:
                modes.add(skill)
            # Fixer trait: can switch to QA mode
            if tdef.trait is not None and tdef.trait.name == "QA_SKILL":
                modes.add(CubeType.QA)

            for mode in modes:
                actions.append(Action(
                    ActionType.CHOOSE_MODE,
                    target_instance=tid,
                    choice=int(mode),
                ))

    return actions


def _get_xp_graduate_actions(state: GameState) -> list[Action]:
    """Get XP graduation choices for a junior after launch."""
    actions: list[Action] = []
    if state.pending_decisions:
        decision = state.pending_decisions[0]
        tid = decision.context.get("talent_id")
        if tid is not None:
            talent = state.talent_instances[tid]
            # Choose one of the pending XP tokens
            seen = set()
            for xp in talent.xp_pending:
                if xp not in seen:
                    actions.append(Action(
                        ActionType.CHOOSE_XP,
                        target_instance=tid,
                        choice=int(xp),
                    ))
                    seen.add(xp)
    return actions


def _get_talent_discard_actions(state: GameState) -> list[Action]:
    """Get legal discard actions for bench overflow (max 5)."""
    player = state.get_player(state.current_player)
    actions: list[Action] = []
    for tid in player.bench:
        actions.append(Action(ActionType.DISCARD_TALENT, target_instance=tid))
    return actions


def _get_backlog_discard_actions(state: GameState) -> list[Action]:
    """Get legal discard actions for backlog overflow (max 3)."""
    player = state.get_player(state.current_player)
    actions: list[Action] = []
    for pid in player.product_backlog:
        actions.append(Action(ActionType.DISCARD_BACKLOG, target_instance=pid))
    return actions


def _get_choose_offline_actions(state: GameState) -> list[Action]:
    """Choose which products to take offline due to bandwidth deficit."""
    player = state.get_player(state.current_player)
    actions: list[Action] = []
    for pid in player.ops_products:
        prod = state.product_instances[pid]
        if prod.is_online:
            actions.append(Action(ActionType.CHOOSE_OFFLINE, target_instance=pid))
    return actions


def _get_fire_choice_actions(state: GameState) -> list[Action]:
    """Choose which staff to fire when can't pay salary."""
    player = state.get_player(state.current_player)
    actions: list[Action] = []
    board_talent = state.get_board_talent(state.current_player)
    for tid in board_talent:
        actions.append(Action(ActionType.FIRE_STAFF, target_instance=tid))
    for tid in player.bench:
        actions.append(Action(ActionType.FIRE_STAFF, target_instance=tid))
    return actions
