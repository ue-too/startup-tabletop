"""Tests for Phase 3: Strategy cards, Investment, Poaching."""

import random

from startup_simulator.card_registry import get_registry
from startup_simulator.engine import GameEngine
from startup_simulator.state import GameState, TalentInstance, PlayerState
from startup_simulator.types import (
    ActionType, CubeType, TalentType, Tier, Trait, Zone,
)
from startup_simulator.actions.base import Action
from startup_simulator.actions.combat_actions import calculate_poach_cost


class TestStrategyCards:
    """Tests for strategy card loading and deck."""

    def test_strategy_cards_loaded(self):
        registry = get_registry()
        assert len(registry.strategy_cards) == 30

    def test_strategy_card_categories(self):
        registry = get_registry()
        cats = {}
        for c in registry.strategy_cards:
            cats[c.category] = cats.get(c.category, 0) + 1
        assert cats["training"] == 8
        assert cats["warfare"] == 10
        assert cats["attribute"] == 6
        assert cats["utility"] == 6

    def test_strategy_deck_in_game(self):
        engine = GameEngine(num_players=2, seed=42)
        assert len(engine.state.markets.strategy_deck) == 30

    def test_strategy_lookup(self):
        registry = get_registry()
        hh = registry.get_strategy("headhunter")
        assert hh.name == "Headhunter"
        assert hh.cost == 0
        assert hh.category == "warfare"


class TestBrainstorm:
    """Tests for the Brainstorm action."""

    def test_brainstorm_draws_card(self):
        engine = GameEngine(num_players=2, seed=42)
        state = engine.state
        player = state.players[0]
        assert len(player.strategy_hand) == 0

        # Find brainstorm action
        actions = engine.get_legal_actions()
        brainstorm = [a for a in actions if a.action_type == ActionType.BRAINSTORM]
        assert len(brainstorm) > 0

        initial_deck = len(state.markets.strategy_deck)
        engine.step(brainstorm[0])
        assert len(player.strategy_hand) == 1
        # Drew 2, kept 1, discarded 1
        assert len(state.markets.strategy_deck) == initial_deck - 2


class TestInvestment:
    """Tests for the investment system."""

    def test_invest_basic(self):
        engine = GameEngine(num_players=2, seed=42)
        state = engine.state
        p0 = state.players[0]
        p1 = state.players[1]

        # Give P0 enough cash
        p0.cash = 20
        p1_equity_before = p1.equity_tokens_own

        actions = engine.get_legal_actions()
        invest = [a for a in actions if a.action_type == ActionType.INVEST]
        if invest:
            engine.step(invest[0])
            assert p0.equity_held.get(1, 0) == 1
            assert p1.equity_tokens_own == p1_equity_before - 1

    def test_cannot_invest_in_self(self):
        """Investment in self should not appear in legal actions."""
        engine = GameEngine(num_players=2, seed=42)
        actions = engine.get_legal_actions()
        invest = [a for a in actions if a.action_type == ActionType.INVEST]
        for a in invest:
            assert a.target_player != engine.state.current_player

    def test_equity_limit(self):
        """Target must keep at least 1 equity token."""
        engine = GameEngine(num_players=2, seed=42)
        state = engine.state
        p0 = state.players[0]
        p1 = state.players[1]
        p0.cash = 100

        # Buy 2 equity (P1 has 3, must keep 1)
        for _ in range(3):
            actions = engine.get_legal_actions()
            invest = [a for a in actions if a.action_type == ActionType.INVEST and a.target_player == 1]
            if invest:
                engine.step(invest[0])
            else:
                break

        assert p0.equity_held.get(1, 0) == 2  # Max 2 (P1 keeps 1)
        assert p1.equity_tokens_own == 1


class TestPoaching:
    """Tests for the poaching system."""

    def test_poach_cost_raw_junior(self):
        """Raw Junior: $2 base * 2x = $4."""
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        inst = state.create_talent_instance("jr_software", 1, Zone.DEV)
        cost = calculate_poach_cost(state, inst.instance_id)
        assert cost == 4  # 2 * 2 = 4

    def test_poach_cost_with_xp(self):
        """Junior with 1 XP: (2 + 2) * 2 = 8."""
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        inst = state.create_talent_instance("jr_software", 1, Zone.DEV)
        inst.xp_permanent.append(CubeType.SOFTWARE)
        cost = calculate_poach_cost(state, inst.instance_id)
        assert cost == 8  # (2 + 2) * 2

    def test_poach_cost_with_rank_badge(self):
        """Junior with rank badge: (2 + 4) * 2 = 12."""
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        inst = state.create_talent_instance("jr_software", 1, Zone.DEV)
        inst.rank_badges = 1
        cost = calculate_poach_cost(state, inst.instance_id)
        assert cost == 12  # (2 + 4) * 2

    def test_poach_cost_senior(self):
        """Standard Senior $6: 6 * 2 = 12."""
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        inst = state.create_talent_instance("sr_backend_hacker", 1, Zone.DEV)
        cost = calculate_poach_cost(state, inst.instance_id)
        assert cost == 12  # 6 * 2

    def test_poach_cost_mercenary(self):
        """Full Stack Ninja (mercenary): 1.5x instead of 2x."""
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        inst = state.create_talent_instance("full_stack_ninja", 1, Zone.DEV)
        cost = calculate_poach_cost(state, inst.instance_id)
        assert cost == 12  # ceil(8 * 1.5) = 12

    def test_poach_cost_flight_risk(self):
        """Flight Risk attribute: 1x base only."""
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        inst = state.create_talent_instance("sr_backend_hacker", 1, Zone.DEV)
        inst.attributes.append("flight_risk")
        cost = calculate_poach_cost(state, inst.instance_id)
        assert cost == 6  # 1x base only

    def test_cannot_poach_from_ops(self):
        """Ops zone is safe from poaching."""
        from startup_simulator.actions.combat_actions import execute_poach
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        state.current_player = 0
        state.players[0].cash = 100
        # Put some dev products so aggressor has a slot
        prod = state.create_product_instance("todo_list", 0, Zone.DEV)
        state.players[0].dev_products.append(prod.instance_id)

        inst = state.create_talent_instance("jr_software", 1, Zone.OPS)
        action = Action(ActionType.PLAY_STRATEGY, target_player=1, target_instance=inst.instance_id)
        result = execute_poach(state, action)
        assert not result.success
        assert "Ops Zone" in result.message


class TestPlayStrategy:
    """Tests for playing strategy cards."""

    def test_play_training_card(self):
        """Play Full Stack Bootcamp to train a junior."""
        engine = GameEngine(num_players=2, seed=42)
        state = engine.state
        p0 = state.players[0]
        p0.cash = 20

        # Get a junior on bench
        jr = None
        for tid in p0.bench:
            t = state.talent_instances[tid]
            registry = get_registry()
            tdef = registry.get_talent(t.card_def_id)
            if tdef.is_junior and tdef.output_type == CubeType.HARDWARE:
                jr = tid
                break

        if jr is None:
            return  # Skip if no hardware junior (need SW skill training)

        # Give player a Full Stack Bootcamp card
        p0.strategy_hand.append("full_stack_bootcamp")

        actions = engine.get_legal_actions()
        play = [a for a in actions if a.action_type == ActionType.PLAY_STRATEGY
                and a.target_instance == jr]
        if play:
            result = engine.step(play[0])
            talent = state.talent_instances[jr]
            assert CubeType.SOFTWARE in talent.skills


class TestRandomGamesPhase3:
    """Run random games with all Phase 3 features."""

    def test_random_2p(self):
        engine = GameEngine(num_players=2, seed=3001)
        rng = random.Random(3001)
        steps = 0
        while not engine.is_done() and steps < 5000:
            actions = engine.get_legal_actions()
            assert len(actions) > 0, f"No legal actions at step {steps}"
            engine.step(rng.choice(actions))
            steps += 1
        if engine.is_done():
            assert len(engine.get_scores()) == 2

    def test_random_3p(self):
        engine = GameEngine(num_players=3, seed=3002)
        rng = random.Random(3002)
        steps = 0
        while not engine.is_done() and steps < 6000:
            actions = engine.get_legal_actions()
            assert len(actions) > 0
            engine.step(rng.choice(actions))
            steps += 1
        if engine.is_done():
            assert len(engine.get_scores()) == 3

    def test_random_4p(self):
        engine = GameEngine(num_players=4, seed=3003)
        rng = random.Random(3003)
        steps = 0
        while not engine.is_done() and steps < 8000:
            actions = engine.get_legal_actions()
            assert len(actions) > 0
            engine.step(rng.choice(actions))
            steps += 1
        if engine.is_done():
            assert len(engine.get_scores()) == 4

    def test_determinism_phase3(self):
        def play(seed):
            engine = GameEngine(num_players=2, seed=seed)
            rng = random.Random(seed + 200)
            snaps = []
            for _ in range(300):
                if engine.is_done():
                    break
                actions = engine.get_legal_actions()
                engine.step(rng.choice(actions))
                snaps.append((
                    engine.state.current_player,
                    engine.state.phase,
                    engine.state.players[0].cash,
                    engine.state.players[1].cash,
                    len(engine.state.players[0].strategy_hand),
                ))
            return snaps

        assert play(777) == play(777)

    def test_many_seeds_no_crash(self):
        """Run 20 games with different seeds to find edge cases."""
        for seed in range(4000, 4020):
            engine = GameEngine(num_players=2, seed=seed)
            rng = random.Random(seed)
            steps = 0
            while not engine.is_done() and steps < 2000:
                actions = engine.get_legal_actions()
                assert len(actions) > 0, f"Seed {seed}: no actions at step {steps}"
                engine.step(rng.choice(actions))
                steps += 1
