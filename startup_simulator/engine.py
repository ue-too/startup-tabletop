"""Game engine: top-level state machine orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from .card_registry import get_registry
from .rng import GameRng
from .state import GameState, MarketState, PlayerState
from .types import (
    ActionType,
    CubeType,
    Phase,
    SubPhase,
    Tier,
    Zone,
    AP_COST,
    IDEATION_POOL_AP_COST,
)
from .actions.base import Action, ActionResult
from .actions.talent_actions import (
    execute_assign_one,
    execute_layoff_source,
    execute_recall,
    execute_reassign,
    execute_recruit,
)
from .actions.product_actions import (
    execute_greenlight,
    execute_ideation,
    execute_launch,
    execute_pivot,
)
from .actions.management_actions import (
    execute_acquisition,
    execute_brainstorm,
    execute_buyback,
    execute_divest,
    execute_invest,
    execute_secondary_trade,
)
from .actions.free_actions import execute_integrate, execute_play_strategy, execute_voluntary_disclosure
from .actions.validator import get_legal_actions
from .modifiers import RoundModifiers
from .phases.event_phase import apply_immediate_event_effects, draw_event, get_round_modifiers
from .phases.audit_phase import check_legality, resolve_fold, resolve_legal, resolve_settle, can_settle
from .phases.income_phase import process_income
from .phases.engine_phase import (
    auto_cleanup_excess,
    clear_onboarding,
    commit_cubes,
    generate_cubes,
    process_growth_hacker_bugs,
    process_qa_ops,
    refill_markets,
    reset_online_status,
)
from .phases.scoring import calculate_final_scores

# Maximum number of turns before forced game end
MAX_TURNS = 30


@dataclass
class StepResult:
    """Result of a single step in the game."""
    current_player: int
    phase: Phase
    sub_phase: SubPhase
    action_result: ActionResult | None = None
    game_over: bool = False
    scores: list[int] | None = None


class GameEngine:
    """Top-level game engine using a flat state machine."""

    def __init__(self, num_players: int = 2, seed: int = 42) -> None:
        if num_players < 2 or num_players > 4:
            raise ValueError("num_players must be 2-4")
        self.rng = GameRng(seed)
        self._player_passed = False
        self._failed_actions: set[tuple] = set()  # Track failed free actions to avoid loops
        self.state = self._setup(num_players)
        self._advance()  # Auto-advance through EVENT and INCOME to first player decision

    def _setup(self, num_players: int) -> GameState:
        """Initialize game state: shuffle decks, deal starting hands, set up markets."""
        registry = get_registry()
        state = GameState(num_players=num_players)

        # Create players
        for i in range(num_players):
            state.players.append(PlayerState(player_id=i))

        # Build decks from card definitions
        seed_defs = registry.get_seed_deck()
        growth_defs = registry.get_growth_deck()
        agency_defs = registry.agency_deck_defs

        # Seed deck (card_def_ids)
        state.markets.seed_deck = [p.card_def_id for p in seed_defs]
        self.rng.shuffle(state.markets.seed_deck)

        # Growth deck with Market Crash in bottom 20%
        growth_non_crash = [p.card_def_id for p in growth_defs if not p.is_market_crash]
        self.rng.shuffle(growth_non_crash)

        # Insert Market Crash into bottom 20%
        bottom_20_size = max(1, len(growth_non_crash) // 5)
        crash_pos = self.rng.randint(0, bottom_20_size)
        growth_non_crash.insert(crash_pos, "market_crash")
        state.markets.growth_deck = growth_non_crash

        # Talent deck (card_def_ids for agency cards)
        state.markets.talent_deck = [c.card_def_id for c in agency_defs]
        self.rng.shuffle(state.markets.talent_deck)

        # Strategy deck
        state.markets.strategy_deck = [c.card_def_id for c in registry.strategy_deck_defs]
        self.rng.shuffle(state.markets.strategy_deck)

        # Event deck
        state.markets.event_deck = [c.card_def_id for c in registry.event_cards]
        self.rng.shuffle(state.markets.event_deck)

        # Deal Agency Row (4 face-up)
        for _ in range(min(4, len(state.markets.talent_deck))):
            card_def_id = state.markets.talent_deck.pop()
            inst = state.create_talent_instance(card_def_id, -1, Zone.BENCH)
            state.markets.agency_row.append(inst.instance_id)

        # Deal Product Market (2 Seed + 2 Growth)
        for _ in range(2):
            if state.markets.seed_deck:
                cid = state.markets.seed_deck.pop()
                inst = state.create_product_instance(cid, -1, Zone.BENCH)
                state.markets.product_market_seed.append(inst.instance_id)
        for _ in range(2):
            if state.markets.growth_deck:
                cid = state.markets.growth_deck.pop()
                pdef = registry.get_product(cid)
                if pdef.is_market_crash:
                    # Shouldn't happen since it's in bottom 20%, but handle it
                    state.market_crash_drawn = True
                    state.finish_round = True
                    continue
                inst = state.create_product_instance(cid, -1, Zone.BENCH)
                state.markets.product_market_growth.append(inst.instance_id)

        # Player setup: 1 Jr Software, 1 Jr Hardware, 1 Jr QA
        # "Jr QA" = a Junior Software Dev pre-trained with QA skill
        for player in state.players:
            inst = state.create_talent_instance("jr_software", player.player_id, Zone.BENCH)
            player.bench.append(inst.instance_id)
            inst = state.create_talent_instance("jr_hardware", player.player_id, Zone.BENCH)
            player.bench.append(inst.instance_id)
            # Jr QA: a junior with green QA skill token
            inst = state.create_talent_instance("jr_software", player.player_id, Zone.BENCH)
            inst.skills.append(CubeType.QA)
            player.bench.append(inst.instance_id)

        # Seed Round: Draft 1 "Concept" product to Dev Zone
        concept_ids = [
            p.card_def_id for p in seed_defs
            if p.sector is not None and p.sector.name == "CONCEPT"
        ]
        for player in state.players:
            if concept_ids and state.markets.seed_deck:
                # Find a concept in the seed deck
                chosen = None
                for i, cid in enumerate(state.markets.seed_deck):
                    if cid in concept_ids:
                        chosen = i
                        break
                if chosen is not None:
                    cid = state.markets.seed_deck.pop(chosen)
                else:
                    # Fallback: just take top of seed deck
                    cid = state.markets.seed_deck.pop()
                inst = state.create_product_instance(cid, player.player_id, Zone.DEV)
                player.dev_products.append(inst.instance_id)

        # Start the game
        state.phase = Phase.EVENT
        state.sub_phase = SubPhase.NONE
        state.current_player = 0
        state.turn_number = 1

        return state

    def step(self, action: Action) -> StepResult:
        """Apply one action, advance state, return result.

        The engine auto-advances through deterministic phases and
        stops when a player decision is needed.
        """
        state = self.state

        if state.game_over:
            return StepResult(
                current_player=state.current_player,
                phase=Phase.GAME_OVER,
                sub_phase=SubPhase.NONE,
                game_over=True,
                scores=state.final_scores,
            )

        result = self._execute_action(action)

        # Track failed free actions to prevent infinite loops
        if not result.success and action.action_type in (
            ActionType.GREENLIGHT, ActionType.PLAY_STRATEGY,
            ActionType.INTEGRATE, ActionType.VOLUNTARY_DISCLOSURE,
        ):
            key = (action.action_type, action.source_index, action.target_instance, action.target_player)
            self._failed_actions.add(key)
        elif action.action_type == ActionType.PASS:
            self._failed_actions.clear()

        self._advance()

        return StepResult(
            current_player=state.current_player,
            phase=state.phase,
            sub_phase=state.sub_phase,
            action_result=result,
            game_over=state.game_over,
            scores=state.final_scores if state.game_over else None,
        )

    def _execute_action(self, action: Action) -> ActionResult:
        """Execute a single action on the current state."""
        state = self.state
        player = state.get_player(state.current_player)

        atype = action.action_type

        # AP-costing actions
        if atype == ActionType.RECRUIT:
            ap_cost = 1
            if player.action_points < ap_cost:
                return ActionResult(False, "Not enough AP")
            result = execute_recruit(state, action)
            if result.success:
                player.action_points -= ap_cost
            return result

        elif atype == ActionType.ASSIGN:
            # Start assign batch: costs 1 AP, then switch to ASSIGN_BATCH sub-phase
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            player.action_points -= 1
            state.sub_phase = SubPhase.ACTION_ASSIGN_BATCH
            return ActionResult(True, "Started assign batch")

        elif atype == ActionType.ASSIGN_ONE:
            return execute_assign_one(state, action)

        elif atype == ActionType.END_ASSIGN_BATCH:
            state.sub_phase = SubPhase.ACTION_MAIN
            return ActionResult(True, "Ended assign batch")

        elif atype == ActionType.RECALL:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_recall(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.REASSIGN:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_reassign(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.IDEATION:
            ap_cost = IDEATION_POOL_AP_COST if action.source_type == "idea_pool" else 1
            if player.action_points < ap_cost:
                return ActionResult(False, "Not enough AP")
            result = execute_ideation(state, action)
            if result.success:
                player.action_points -= ap_cost
            return result

        elif atype == ActionType.LAUNCH:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_launch(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.PIVOT:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_pivot(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.LAYOFF_SOURCE:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_layoff_source(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.BRAINSTORM:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_brainstorm(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.INVEST:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_invest(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.DIVEST:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_divest(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.BUYBACK:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_buyback(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.SECONDARY_TRADE:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_secondary_trade(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.ACQUISITION:
            if player.action_points < 1:
                return ActionResult(False, "Not enough AP")
            result = execute_acquisition(state, action)
            if result.success:
                player.action_points -= 1
            return result

        elif atype == ActionType.GREENLIGHT:
            # 0 AP
            return execute_greenlight(state, action)

        elif atype == ActionType.PLAY_STRATEGY:
            # 0 AP
            return execute_play_strategy(state, action)

        elif atype == ActionType.INTEGRATE:
            # 0 AP
            return execute_integrate(state, action)

        elif atype == ActionType.VOLUNTARY_DISCLOSURE:
            # 0 AP
            return execute_voluntary_disclosure(state, action)

        elif atype == ActionType.BID_AUDIT:
            # Bid on audit
            prod_id = action.target_instance
            bid_amount = action.amount
            state.audit_bids[state.current_player] = bid_amount
            state.audit_target_product = prod_id
            return ActionResult(True, f"Bid ${bid_amount} on audit")

        elif atype == ActionType.PASS_AUDIT:
            # Pass on audit opportunity
            return ActionResult(True, "Passed on audit")

        elif atype == ActionType.FOLD:
            # Owner folds after illegal audit
            prod_id = state.audit_target_product
            if prod_id is not None:
                # Find whistleblower (highest bidder)
                wb_id = max(state.audit_bids, key=state.audit_bids.get) if state.audit_bids else -1
                if wb_id >= 0:
                    resolve_fold(state, prod_id, wb_id)
                state.audit_bids.clear()
                state.audit_target_product = None
            return ActionResult(True, "Folded")

        elif atype == ActionType.SETTLE:
            prod_id = state.audit_target_product
            if prod_id is not None:
                wb_id = max(state.audit_bids, key=state.audit_bids.get) if state.audit_bids else -1
                if wb_id >= 0:
                    resolve_settle(state, prod_id, wb_id)
                state.audit_bids.clear()
                state.audit_target_product = None
            return ActionResult(True, "Settled")

        elif atype == ActionType.CHOOSE_MODE:
            tid = action.target_instance
            talent = state.talent_instances.get(tid)
            if talent:
                talent.declared_mode = CubeType(action.choice)
                if state.pending_decisions:
                    state.pending_decisions.pop(0)
                return ActionResult(True, f"Declared mode {CubeType(action.choice).name}")
            return ActionResult(False, "Invalid talent")

        elif atype == ActionType.CHOOSE_XP:
            tid = action.target_instance
            talent = state.talent_instances.get(tid)
            if talent and talent.xp_pending:
                chosen = CubeType(action.choice)
                if chosen in talent.xp_pending and talent.total_xp < 4:
                    talent.xp_permanent.append(chosen)
                talent.xp_pending.clear()
                if state.pending_decisions:
                    state.pending_decisions.pop(0)
                return ActionResult(True, f"Graduated XP: {chosen.name}")
            return ActionResult(False, "Invalid talent or no pending XP")

        elif atype == ActionType.PASS:
            # End action phase for this player
            player.action_points = 0
            self._player_passed = True
            return ActionResult(True, "Passed")

        elif atype == ActionType.DISCARD_TALENT:
            tid = action.target_instance
            if tid in player.bench:
                player.bench.remove(tid)
                talent = state.talent_instances[tid]
                talent.owner = -1
                state.markets.open_job_market.append(tid)
                while len(state.markets.open_job_market) > 5:
                    state.markets.open_job_market.popleft()
                return ActionResult(True, "Discarded talent")
            return ActionResult(False, "Talent not on bench")

        elif atype == ActionType.DISCARD_BACKLOG:
            pid = action.target_instance
            if pid in player.product_backlog:
                player.product_backlog.remove(pid)
                state.product_instances[pid].owner = -1
                state.markets.open_idea_pool.append(pid)
                while len(state.markets.open_idea_pool) > 5:
                    state.markets.open_idea_pool.popleft()
                return ActionResult(True, "Discarded backlog product")
            return ActionResult(False, "Product not in backlog")

        elif atype == ActionType.CHOOSE_OFFLINE:
            pid = action.target_instance
            prod = state.product_instances.get(pid)
            if prod and prod.is_online:
                prod.is_online = False
                return ActionResult(True, "Product taken offline")
            return ActionResult(False, "Invalid product")

        elif atype == ActionType.FIRE_STAFF:
            tid = action.target_instance
            talent = state.talent_instances.get(tid)
            if talent and talent.owner == state.current_player:
                if tid in player.bench:
                    player.bench.remove(tid)
                talent.owner = -1
                talent.zone = Zone.BENCH
                talent.assigned_product = None
                state.markets.open_job_market.append(tid)
                while len(state.markets.open_job_market) > 5:
                    state.markets.open_job_market.popleft()
                return ActionResult(True, "Fired staff")
            return ActionResult(False, "Invalid talent")

        return ActionResult(False, f"Unknown action type: {atype}")

    def _advance(self) -> None:
        """Auto-advance through deterministic phases until a player decision is needed."""
        state = self.state

        while True:
            if state.game_over:
                return

            if state.phase == Phase.EVENT:
                self._do_event_phase()
                state.phase = Phase.INCOME
                state.sub_phase = SubPhase.NONE
                state.current_player = 0
                continue

            elif state.phase == Phase.INCOME:
                # Process income for all players (auto for Phase 1)
                for pid in range(state.num_players):
                    reset_online_status(state, pid)
                    process_income(state, pid)

                # Move to action phase
                state.phase = Phase.ACTION
                state.sub_phase = SubPhase.ACTION_MAIN
                state.current_player = 0
                state.players[0].action_points = 3
                return  # Yield: need player action

            elif state.phase == Phase.ACTION:
                player = state.get_player(state.current_player)

                if state.sub_phase == SubPhase.ACTION_ASSIGN_BATCH:
                    # Waiting for assign batch decisions
                    return

                if player.action_points <= 0:
                    if not self._player_passed:
                        # Check for free actions (excluding known-failed ones)
                        legal = get_legal_actions(state)
                        has_viable = False
                        for a in legal:
                            if a.action_type == ActionType.PASS:
                                continue
                            key = (a.action_type, a.source_index, a.target_instance, a.target_player)
                            if key not in self._failed_actions:
                                has_viable = True
                                break
                        if has_viable:
                            return  # Still has free actions

                    self._player_passed = False
                    self._failed_actions.clear()
                    # Move to next player or engine phase
                    next_player = state.current_player + 1
                    if next_player >= state.num_players:
                        # All players done, move to engine phase
                        state.phase = Phase.ENGINE
                        state.sub_phase = SubPhase.NONE
                        state.engine_player_index = 0
                        continue
                    else:
                        state.current_player = next_player
                        state.players[next_player].action_points = 3
                        state.sub_phase = SubPhase.ACTION_MAIN
                        return  # Yield: next player's turn
                else:
                    return  # Yield: player has AP

            elif state.phase == Phase.ENGINE:
                self._do_engine_phase()

                # Check for game over: Market Crash or turn limit
                if state.market_crash_drawn and state.finish_round:
                    self._end_game()
                    return
                if state.turn_number >= MAX_TURNS:
                    self._end_game()
                    return

                # Next turn
                state.turn_number += 1
                state.phase = Phase.EVENT
                state.sub_phase = SubPhase.NONE
                state.current_player = 0
                continue

            else:
                return

    def _do_event_phase(self) -> None:
        """Phase A: Draw and apply event card."""
        state = self.state
        card_id = draw_event(state)
        if card_id is not None:
            state.round_modifiers = get_round_modifiers(state)
            apply_immediate_event_effects(state)
        else:
            state.round_modifiers = RoundModifiers()

    def _do_engine_phase(self) -> None:
        """Phase D: Engine phase for all players.

        Order per rulebook: Generate -> QA -> Audit -> Commit -> Complete -> Refill -> Cleanup
        Growth Hacker bugs happen at end of turn (after commit, part of Complete).
        """
        state = self.state

        for pid in range(state.num_players):
            # 1. Generate & Train: dev teams produce cubes + pending XP
            generate_cubes(state, pid)

            # 2. Clean & Train: QA in Ops removes bugs
            process_qa_ops(state, pid)

            # 3. Audit window (auto-skipped for now; bidding handled via sub-phases)

            # 4. Commit: move transient cubes to tracks
            commit_cubes(state, pid)

            # 5. Complete: mark full products, Growth Hacker decay
            process_growth_hacker_bugs(state, pid)

        # 6. Refill markets (once per round, before cleanup per rulebook)
        refill_markets(state)

        # 7. Cleanup + clear onboarding (per player)
        for pid in range(state.num_players):
            auto_cleanup_excess(state, pid)
            clear_onboarding(state, pid)

    def _end_game(self) -> None:
        """Calculate final scores and end the game."""
        state = self.state
        state.final_scores = calculate_final_scores(state)
        state.game_over = True
        state.phase = Phase.GAME_OVER

    def get_current_agent(self) -> int:
        """Which player must act next."""
        return self.state.current_player

    def get_legal_actions(self) -> list[Action]:
        """Legal actions for current agent."""
        return get_legal_actions(self.state)

    def is_done(self) -> bool:
        return self.state.game_over

    def get_scores(self) -> list[int]:
        return self.state.final_scores
