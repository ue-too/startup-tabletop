"""Free actions (0 AP): Play Strategy Card, Integrate."""

from __future__ import annotations

from ..card_registry import get_registry
from ..state import GameState
from ..types import ActionType, CubeType, TalentType, Zone
from .base import Action, ActionResult


def execute_play_strategy(state: GameState, action: Action) -> ActionResult:
    """Play a strategy card from hand (0 AP).

    source_index: index in strategy hand
    target_instance: talent or product instance_id (for targeted effects)
    target_player: opponent player_id (for warfare/debuffs)
    """
    player = state.get_player(state.current_player)
    registry = get_registry()

    idx = action.source_index
    if idx < 0 or idx >= len(player.strategy_hand):
        return ActionResult(False, "Invalid strategy card index")

    card_def_id = player.strategy_hand[idx]
    sdef = registry.get_strategy(card_def_id)

    # Check cost
    if player.cash < sdef.cost:
        return ActionResult(False, f"Cannot afford ${sdef.cost}")

    # Execute effect
    result = _execute_strategy_effect(state, sdef.effect_id, action, sdef)
    if not result.success:
        return result

    # Pay cost and discard card
    player.cash -= sdef.cost
    player.strategy_hand.pop(idx)
    state.markets.strategy_discard.append(card_def_id)

    return result


def _execute_strategy_effect(state: GameState, effect_id: str, action: Action, sdef) -> ActionResult:
    """Execute a strategy card effect."""
    registry = get_registry()
    player = state.get_player(state.current_player)

    # --- TRAINING ---
    if effect_id == "train_software_skill":
        return _train_skill(state, action.target_instance, CubeType.SOFTWARE)

    elif effect_id == "train_qa_skill":
        return _train_skill(state, action.target_instance, CubeType.QA)

    elif effect_id == "train_specialist_xp":
        return _train_specialist(state, action.target_instance)

    elif effect_id == "add_rank_badge":
        return _add_rank_badge(state, action.target_instance, developer=True)

    elif effect_id == "add_pm_rank_badge":
        return _add_rank_badge(state, action.target_instance, developer=False)

    # --- WARFARE ---
    elif effect_id == "poach_2x":
        # Headhunter: poach at 2x cost. Handled by combat_actions.
        return _initiate_poach(state, action, multiplier=2.0, bypass_hr=False)

    elif effect_id == "poach_1_5x_bypass_hr":
        return _initiate_poach(state, action, multiplier=1.5, bypass_hr=True)

    elif effect_id == "add_scandal":
        return _add_scandal(state, action)

    elif effect_id == "hostile_buyout":
        return _hostile_buyout(state, action)

    # --- ATTRIBUTES ---
    elif effect_id.startswith("attr_"):
        return _attach_attribute(state, action, effect_id)

    # --- UTILITY ---
    elif effect_id == "add_hype":
        return _add_hype(state, action)

    elif effect_id == "draw_5_products":
        return _design_sprint(state)

    elif effect_id == "cancel_attack":
        # Cease & Desist: reaction card, handled separately
        return ActionResult(True, "Cease & Desist held (reaction)")

    return ActionResult(False, f"Unknown effect: {effect_id}")


def _train_skill(state: GameState, talent_id: int, skill: CubeType) -> ActionResult:
    """Add a skill token to a junior."""
    if talent_id < 0:
        return ActionResult(False, "No talent target")
    talent = state.talent_instances.get(talent_id)
    if talent is None or talent.owner != state.current_player:
        return ActionResult(False, "Invalid talent")
    registry = get_registry()
    tdef = registry.get_talent(talent.card_def_id)
    if not tdef.is_junior:
        return ActionResult(False, "Only juniors can be trained")
    if skill in talent.skills:
        return ActionResult(False, "Already has this skill")
    # Check native type: can't train native skill
    if tdef.output_type == skill:
        return ActionResult(False, "Already has native skill")
    talent.skills.append(skill)
    return ActionResult(True, f"Trained {skill.name} skill")


def _train_specialist(state: GameState, talent_id: int) -> ActionResult:
    """Add 1 XP to a specialist (QA or Sales)."""
    if talent_id < 0:
        return ActionResult(False, "No talent target")
    talent = state.talent_instances.get(talent_id)
    if talent is None or talent.owner != state.current_player:
        return ActionResult(False, "Invalid talent")
    registry = get_registry()
    tdef = registry.get_talent(talent.card_def_id)
    if tdef.talent_type not in (TalentType.QA, TalentType.SALES):
        return ActionResult(False, "Only QA/Sales specialists can receive training XP")
    if talent.total_xp >= 2:
        return ActionResult(False, "Specialist at max XP (2)")
    talent.xp_permanent.append(CubeType.QA)  # Type doesn't matter for specialists
    return ActionResult(True, "Trained specialist +1 XP")


def _add_rank_badge(state: GameState, talent_id: int, developer: bool) -> ActionResult:
    """Add Gold Rank Badge."""
    if talent_id < 0:
        return ActionResult(False, "No talent target")
    talent = state.talent_instances.get(talent_id)
    if talent is None or talent.owner != state.current_player:
        return ActionResult(False, "Invalid talent")
    registry = get_registry()
    tdef = registry.get_talent(talent.card_def_id)

    if developer:
        if not tdef.is_junior:
            return ActionResult(False, "Rank badge targets junior developers")
        if talent.rank_badges > 0:
            return ActionResult(False, "Already has rank badge")
        talent.rank_badges = 1
        return ActionResult(True, "Added Gold Rank Badge (Tier 2 Lead)")
    else:
        if not tdef.is_pm:
            return ActionResult(False, "PM rank badge targets PMs only")
        if talent.rank_badges > 0:
            return ActionResult(False, "Already has rank badge")
        talent.rank_badges = 1
        return ActionResult(True, "Added PM Gold Rank Badge (Agile)")


def _initiate_poach(state: GameState, action: Action, multiplier: float, bypass_hr: bool) -> ActionResult:
    """Initiate a poach attempt."""
    from .combat_actions import execute_poach
    return execute_poach(state, action, multiplier=multiplier, bypass_hr=bypass_hr)


def _add_scandal(state: GameState, action: Action) -> ActionResult:
    """Add Scandal token to opponent's maintenance product."""
    target_pid = action.target_player
    prod_id = action.target_instance
    if target_pid < 0 or target_pid == state.current_player:
        return ActionResult(False, "Invalid target player")

    # Check investor immunity: cannot attack partner
    player = state.get_player(state.current_player)
    if player.equity_held.get(target_pid, 0) > 0:
        return ActionResult(False, "Cannot attack investment partner")

    target = state.get_player(target_pid)
    if prod_id not in target.ops_products:
        return ActionResult(False, "Invalid target product")

    prod = state.product_instances[prod_id]
    prod.scandal += 1
    return ActionResult(True, "Added Scandal token")


def _hostile_buyout(state: GameState, action: Action) -> ActionResult:
    """Non-Compete Suit: pay investor $5 to force return of equity."""
    player = state.get_player(state.current_player)
    target_pid = action.target_player  # The investor holding our equity

    if target_pid < 0 or target_pid == state.current_player:
        return ActionResult(False, "Invalid target")

    target = state.get_player(target_pid)
    tokens = target.equity_held.get(state.current_player, 0)
    if tokens <= 0:
        return ActionResult(False, "Target doesn't hold your equity")

    # Pay $5 to investor (card cost $4 already deducted)
    if player.cash < 5:
        return ActionResult(False, "Cannot afford $5 buyout payment")

    player.cash -= 5
    target.cash += 5
    target.equity_held[state.current_player] -= 1
    if target.equity_held[state.current_player] == 0:
        del target.equity_held[state.current_player]
    player.equity_tokens_own += 1

    return ActionResult(True, "Forced equity return via Non-Compete Suit")


def _attach_attribute(state: GameState, action: Action, effect_id: str) -> ActionResult:
    """Attach an attribute card to a talent."""
    talent_id = action.target_instance
    if talent_id < 0:
        return ActionResult(False, "No talent target")

    talent = state.talent_instances.get(talent_id)
    if talent is None:
        return ActionResult(False, "Invalid talent")

    player = state.get_player(state.current_player)
    attr_name = effect_id.replace("attr_", "")

    # Buff attributes target own talent
    if attr_name in ("workaholic", "clean_coder", "visionary"):
        if talent.owner != state.current_player:
            return ActionResult(False, "Buff attributes target your own talent")
    # Debuff attributes target opponent talent
    elif attr_name in ("toxic", "burnout", "flight_risk"):
        if talent.owner == state.current_player:
            return ActionResult(False, "Debuff attributes target opponent talent")
        # Check investor immunity
        if player.equity_held.get(talent.owner, 0) > 0:
            return ActionResult(False, "Cannot debuff investment partner")
    else:
        return ActionResult(False, f"Unknown attribute: {attr_name}")

    talent.attributes.append(attr_name)
    return ActionResult(True, f"Attached {attr_name} attribute")


def _add_hype(state: GameState, action: Action) -> ActionResult:
    """Add Hype token to own maintenance product."""
    player = state.get_player(state.current_player)
    prod_id = action.target_instance
    if prod_id not in player.ops_products:
        return ActionResult(False, "Invalid product")
    prod = state.product_instances[prod_id]
    prod.hype += 1
    return ActionResult(True, "Added Hype token")


def _design_sprint(state: GameState) -> ActionResult:
    """Draw 5 products, keep 1, discard rest to Open Idea Pool."""
    player = state.get_player(state.current_player)

    drawn = []
    # Draw from both decks
    for _ in range(5):
        if state.markets.seed_deck:
            drawn.append(("seed", state.markets.seed_deck.pop()))
        elif state.markets.growth_deck:
            card_id = state.markets.growth_deck.pop()
            registry = get_registry()
            pdef = registry.get_product(card_id)
            if pdef.is_market_crash:
                state.market_crash_drawn = True
                state.finish_round = True
                continue
            drawn.append(("growth", card_id))

    if not drawn:
        return ActionResult(False, "No products to draw")

    # Auto-keep first, discard rest (simplified)
    kept_source, kept_id = drawn[0]
    inst = state.create_product_instance(kept_id, state.current_player, Zone.BENCH)
    if len(player.product_backlog) < 4:
        player.product_backlog.append(inst.instance_id)

    for source, card_id in drawn[1:]:
        discard_inst = state.create_product_instance(card_id, -1, Zone.BENCH)
        state.markets.open_idea_pool.append(discard_inst.instance_id)
        while len(state.markets.open_idea_pool) > 5:
            state.markets.open_idea_pool.popleft()

    return ActionResult(True, f"Design Sprint: kept 1 product")


def execute_integrate(state: GameState, action: Action) -> ActionResult:
    """Stack two maintenance products as Host/Client (0 AP).

    target_instance: host product instance_id
    source_index: client product instance_id
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    host_id = action.target_instance
    client_id = action.source_index

    if host_id not in player.ops_products or client_id not in player.ops_products:
        return ActionResult(False, "Both products must be in Ops")

    host = state.product_instances[host_id]
    client = state.product_instances[client_id]
    host_def = registry.get_product(host.card_def_id)
    client_def = registry.get_product(client.card_def_id)

    # Check not already integrated
    if host.integrated_with is not None or client.integrated_with is not None:
        return ActionResult(False, "One of the products is already integrated")

    # Check compatibility
    if not _is_valid_integration(host_def, client_def, registry):
        return ActionResult(False, "Invalid host/client pairing")

    # Same tag restriction
    if host_def.provides is not None and host_def.provides == client_def.provides:
        return ActionResult(False, "Cannot stack same tag")

    # Stack
    host.integrated_with = client_id
    host.is_host = True
    client.integrated_with = host_id
    client.is_host = False

    return ActionResult(True, f"Integrated {host_def.name} (host) + {client_def.name} (client)")


def _is_valid_integration(host_def, client_def, registry) -> bool:
    """Check if host can host client based on integration rules."""
    from ..types import Tag
    rules = registry.integration_rules

    host_tag = host_def.provides
    client_tag = client_def.provides

    if host_tag is None or client_tag is None:
        return False

    # Check standard host/client rules
    host_tag_name = host_tag.name
    client_tag_name = client_tag.name

    host_rules = rules.get("host_client_rules", {})
    if host_tag_name in host_rules:
        if client_tag_name in host_rules[host_tag_name]:
            return True

    # Check unicorn hosts
    unicorn_rules = rules.get("unicorn_hosts", {})
    if host_def.card_def_id in unicorn_rules:
        if client_tag_name in unicorn_rules[host_def.card_def_id]:
            return True

    return False


def execute_voluntary_disclosure(state: GameState, action: Action) -> ActionResult:
    """Flip a stealth (face-down) card face-up voluntarily (0 AP).

    target_instance: product instance_id
    """
    player = state.get_player(state.current_player)
    registry = get_registry()
    prod_id = action.target_instance

    if prod_id not in player.dev_products:
        return ActionResult(False, "Product not in dev zone")

    prod = state.product_instances[prod_id]
    if not prod.is_face_down:
        return ActionResult(False, "Product is already face-up")

    pdef = registry.get_product(prod.card_def_id)

    # Check if disclosure requires late license
    if pdef.requires:
        player_tags = state.get_player_tags_with_partners(state.current_player)
        has_secured = all(t in player_tags for t in pdef.requires)
        if not has_secured:
            if player.cash < 4:
                return ActionResult(False, "Cannot afford $4 late license")
            player.cash -= 4
            for tag in pdef.requires:
                if tag not in player_tags:
                    for other in state.players:
                        if other.player_id == state.current_player:
                            continue
                        if tag in state.get_player_tags(other.player_id):
                            other.cash += 3
                            break
                    break

    prod.is_face_down = False
    # Lose stealth launch bonus potential
    return ActionResult(True, f"Voluntarily disclosed {pdef.name}")
