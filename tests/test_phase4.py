"""Tests for Phase 4: Events, Stealth, Audit, Ecosystem Integration."""

import random

from startup_simulator.card_registry import get_registry
from startup_simulator.engine import GameEngine
from startup_simulator.modifiers import RoundModifiers, parse_event_modifiers
from startup_simulator.state import GameState, PlayerState, ProductInstance
from startup_simulator.types import ActionType, CubeType, Sector, Tag, Tier, Zone
from startup_simulator.phases.audit_phase import check_legality, can_settle
from startup_simulator.phases.event_phase import draw_event, get_round_modifiers
from startup_simulator.phases.engine_phase import apply_domain_expertise
from startup_simulator.actions.free_actions import _is_valid_integration


class TestEvents:
    """Tests for event system."""

    def test_event_deck_loaded(self):
        registry = get_registry()
        assert len(registry.event_cards) == 18

    def test_event_deck_in_game(self):
        engine = GameEngine(num_players=2, seed=42)
        # One event already drawn during first turn setup
        assert len(engine.state.markets.event_deck) == 17
        assert engine.state.markets.active_event is not None

    def test_draw_event(self):
        engine = GameEngine(num_players=2, seed=42)
        state = engine.state
        initial_count = len(state.markets.event_deck)
        # Manually test event draw (normally happens in _do_event_phase)
        card_id = draw_event(state)
        assert card_id is not None
        assert state.markets.active_event == card_id
        assert len(state.markets.event_deck) == initial_count - 1

    def test_round_modifiers_default(self):
        m = RoundModifiers()
        assert m.license_fee == 3
        assert m.revenue_decay_per_bug == 1
        assert m.software_output_bonus == 0
        assert not m.poaching_suspended

    def test_parse_event_modifiers(self):
        m = parse_event_modifiers("license_fee_5")
        assert m.license_fee == 5

        m = parse_event_modifiers("software_output_plus_1")
        assert m.software_output_bonus == 1

        m = parse_event_modifiers("poaching_suspended")
        assert m.poaching_suspended

        m = parse_event_modifiers("decay_2_per_bug")
        assert m.revenue_decay_per_bug == 2

        m = parse_event_modifiers("university_free")
        assert m.university_cost == 0

    def test_event_applied_in_game(self):
        """Events are drawn and modifiers set each turn."""
        engine = GameEngine(num_players=2, seed=42)
        state = engine.state
        # After setup, we've already gone through one event phase
        assert state.round_modifiers is not None


class TestStealth:
    """Tests for stealth mode."""

    def test_greenlight_stealth_no_deps(self):
        """Products with no dependencies can go stealth."""
        engine = GameEngine(num_players=2, seed=42)
        state = engine.state
        p0 = state.players[0]
        # Draft a concept product to backlog
        prod = state.create_product_instance("todo_list", 0, Zone.BENCH)
        p0.product_backlog.append(prod.instance_id)
        # Clear dev zone to have room
        p0.dev_products.clear()

        from startup_simulator.actions.product_actions import execute_greenlight
        from startup_simulator.actions.base import Action
        action = Action(ActionType.GREENLIGHT, target_instance=prod.instance_id)
        state.current_player = 0
        result = execute_greenlight(state, action)
        assert result.success
        assert state.product_instances[prod.instance_id].is_face_down  # Stealth


class TestAudit:
    """Tests for audit legality checking."""

    def test_legal_product(self):
        """Product with proper tags is legal."""
        registry = get_registry()
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]

        # Tier 1 product: no deps, always legal
        prod = state.create_product_instance("todo_list", 0, Zone.DEV)
        state.players[0].dev_products.append(prod.instance_id)
        is_legal, reason = check_legality(state, prod.instance_id)
        assert is_legal

    def test_missing_dependency(self):
        """Product with missing required tag is illegal."""
        registry = get_registry()
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]

        # Streaming Service requires [Social] tag
        prod = state.create_product_instance("streaming_service", 0, Zone.DEV)
        state.players[0].dev_products.append(prod.instance_id)
        is_legal, reason = check_legality(state, prod.instance_id)
        assert not is_legal
        assert "Missing dependency" in reason

    def test_dependency_satisfied_by_ops(self):
        """Product is legal if owner has required tag in maintenance."""
        registry = get_registry()
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]

        # Put a Social product in ops first
        social_prod = state.create_product_instance("dating_app", 0, Zone.OPS)
        state.players[0].ops_products.append(social_prod.instance_id)

        # Now Streaming Service (requires [Social]) should be legal
        prod = state.create_product_instance("streaming_service", 0, Zone.DEV)
        state.players[0].dev_products.append(prod.instance_id)
        is_legal, reason = check_legality(state, prod.instance_id)
        assert is_legal


class TestIntegration:
    """Tests for ecosystem integration."""

    def test_valid_hardware_host(self):
        """IoT device can host an App."""
        registry = get_registry()
        host = registry.get_product("smart_thermostat")  # [IoT]
        client = registry.get_product("todo_list")  # [App]
        assert _is_valid_integration(host, client, registry)

    def test_valid_platform_host(self):
        """Platform can host Media."""
        registry = get_registry()
        host = registry.get_product("app_store")  # [Platform]
        client = registry.get_product("streaming_service")  # [Media]
        assert _is_valid_integration(host, client, registry)

    def test_invalid_pairing(self):
        """Two apps can't stack."""
        registry = get_registry()
        app1 = registry.get_product("todo_list")  # [App]
        app2 = registry.get_product("flashlight_app")  # [App]
        assert not _is_valid_integration(app1, app2, registry)

    def test_same_tag_rejected(self):
        """Same tag products cannot stack."""
        registry = get_registry()
        # Both provide [Platform] - shouldn't integrate even if rules matched
        p1 = registry.get_product("basic_website")
        p2 = registry.get_product("tech_blog")
        # These both provide Platform so same-tag check should catch it
        # (also not a valid host/client anyway)
        assert not _is_valid_integration(p1, p2, registry)


class TestRandomGamesPhase4:
    """Run random games with all Phase 4 features."""

    def test_random_2p(self):
        engine = GameEngine(num_players=2, seed=4001)
        rng = random.Random(4001)
        steps = 0
        while not engine.is_done() and steps < 5000:
            actions = engine.get_legal_actions()
            assert len(actions) > 0, f"No legal actions at step {steps}, phase={engine.state.phase}"
            engine.step(rng.choice(actions))
            steps += 1
        if engine.is_done():
            assert len(engine.get_scores()) == 2

    def test_random_3p(self):
        engine = GameEngine(num_players=3, seed=4002)
        rng = random.Random(4002)
        steps = 0
        while not engine.is_done() and steps < 6000:
            actions = engine.get_legal_actions()
            assert len(actions) > 0
            engine.step(rng.choice(actions))
            steps += 1
        if engine.is_done():
            assert len(engine.get_scores()) == 3

    def test_random_4p(self):
        engine = GameEngine(num_players=4, seed=4003)
        rng = random.Random(4003)
        steps = 0
        while not engine.is_done() and steps < 8000:
            actions = engine.get_legal_actions()
            assert len(actions) > 0
            engine.step(rng.choice(actions))
            steps += 1
        if engine.is_done():
            assert len(engine.get_scores()) == 4

    def test_determinism(self):
        def play(seed):
            engine = GameEngine(num_players=2, seed=seed)
            rng = random.Random(seed + 300)
            snaps = []
            for _ in range(300):
                if engine.is_done():
                    break
                actions = engine.get_legal_actions()
                engine.step(rng.choice(actions))
                s = engine.state
                snaps.append((
                    s.current_player, s.phase, s.players[0].cash, s.players[1].cash,
                    s.markets.active_event, len(s.markets.event_deck),
                ))
            return snaps
        assert play(888) == play(888)

    def test_many_seeds_no_crash(self):
        """Run 30 games with different seeds."""
        for seed in range(5000, 5030):
            engine = GameEngine(num_players=2, seed=seed)
            rng = random.Random(seed)
            steps = 0
            while not engine.is_done() and steps < 2000:
                actions = engine.get_legal_actions()
                assert len(actions) > 0, f"Seed {seed}: no actions at step {steps}, phase={engine.state.phase}"
                engine.step(rng.choice(actions))
                steps += 1
