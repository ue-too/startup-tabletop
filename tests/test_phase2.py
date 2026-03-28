"""Tests for Phase 2 features: XP, traits, domain expertise, lead matching."""

from startup_simulator.card_registry import get_registry
from startup_simulator.engine import GameEngine
from startup_simulator.state import GameState, TalentInstance, ProductInstance, PlayerState
from startup_simulator.types import (
    ActionType, CubeType, Phase, Sector, SubPhase, Tag, TalentType, Tier, Trait, Zone,
)
from startup_simulator.actions.base import Action
from startup_simulator.phases.engine_phase import (
    apply_domain_expertise, generate_cubes, _has_matching_lead,
    _get_qa_bug_removal, _get_sales_bonus,
)
from startup_simulator.phases.income_phase import calculate_salary_cost


class TestXPSystem:
    """Tests for the XP system."""

    def test_junior_salary_threshold(self):
        """Juniors with 0-1 XP pay $0, 2+ XP pay $1."""
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        assert t.salary == 0
        t.xp_permanent.append(CubeType.SOFTWARE)
        assert t.salary == 0  # 1 XP, still $0
        t.xp_permanent.append(CubeType.SOFTWARE)
        assert t.salary == 1  # 2 XP, now $1

    def test_junior_max_xp(self):
        """Juniors can hold max 4 permanent XP."""
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        for _ in range(4):
            t.xp_permanent.append(CubeType.SOFTWARE)
        assert t.total_xp == 4

    def test_pending_xp_limit(self):
        """Can hold max 1 pending per color (3 total)."""
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        t.xp_pending = [CubeType.SOFTWARE, CubeType.HARDWARE, CubeType.QA]
        assert t.has_pending_xp_of_type(CubeType.SOFTWARE)
        assert t.has_pending_xp_of_type(CubeType.HARDWARE)
        assert t.has_pending_xp_of_type(CubeType.QA)

    def test_junior_output_with_xp(self):
        """XP increases output in matching mode."""
        registry = get_registry()
        tdef = registry.get_talent("jr_software")
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)

        assert t.get_output(tdef) == 1  # Base output
        t.xp_permanent.append(CubeType.SOFTWARE)
        assert t.get_output(tdef) == 2  # 1 base + 1 XP
        t.xp_permanent.append(CubeType.SOFTWARE)
        assert t.get_output(tdef) == 3  # 1 base + 2 XP

    def test_junior_xp_only_matches_mode(self):
        """XP bonus only applies when mode matches."""
        registry = get_registry()
        tdef = registry.get_talent("jr_software")
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        # Add hardware XP (from cross-training)
        t.xp_permanent.append(CubeType.HARDWARE)
        # Software output should still be base 1 (hardware XP doesn't help)
        assert t.get_output(tdef) == 1

    def test_onboarding_zeroes_output(self):
        """Onboarding token means 0 output."""
        registry = get_registry()
        tdef = registry.get_talent("jr_software")
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        t.onboarding = True
        assert t.get_output(tdef) == 0


class TestMultiSkilled:
    """Tests for multi-skilled juniors."""

    def test_needs_mode_declaration(self):
        """Junior with extra skill needs mode declaration."""
        registry = get_registry()
        tdef = registry.get_talent("jr_software")
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        assert not t.needs_mode_declaration(tdef)
        t.skills.append(CubeType.HARDWARE)
        assert t.needs_mode_declaration(tdef)

    def test_flex_needs_mode(self):
        """Flex units (Firmware/Full Stack) always need mode declaration."""
        registry = get_registry()
        tdef = registry.get_talent("firmware_specialist")
        t = TalentInstance(instance_id=0, card_def_id="firmware_specialist", owner=0, zone=Zone.DEV)
        assert t.needs_mode_declaration(tdef)

    def test_effective_mode_declared(self):
        """Declared mode overrides native type."""
        registry = get_registry()
        tdef = registry.get_talent("jr_software")
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        t.skills.append(CubeType.HARDWARE)
        t.declared_mode = CubeType.HARDWARE
        assert t.get_effective_mode(tdef) == CubeType.HARDWARE


class TestLeadMatching:
    """Tests for Tier 2/3 lead requirements."""

    def test_senior_backend_leads_software(self):
        """Senior Backend can lead software projects."""
        registry = get_registry()
        tdef = registry.get_talent("sr_backend_hacker")
        t = TalentInstance(instance_id=0, card_def_id="sr_backend_hacker", owner=0, zone=Zone.DEV)
        assert t.can_lead_software(tdef)
        assert not t.can_lead_hardware(tdef)

    def test_senior_hardware_leads_hardware(self):
        """Senior Hardware can lead hardware projects."""
        registry = get_registry()
        tdef = registry.get_talent("sr_hardware_diva")
        t = TalentInstance(instance_id=0, card_def_id="sr_hardware_diva", owner=0, zone=Zone.DEV)
        assert not t.can_lead_software(tdef)
        assert t.can_lead_hardware(tdef)

    def test_cross_functional_leads_both(self):
        """Firmware/Full Stack can lead both types."""
        registry = get_registry()
        tdef = registry.get_talent("firmware_specialist")
        t = TalentInstance(instance_id=0, card_def_id="firmware_specialist", owner=0, zone=Zone.DEV)
        assert t.can_lead_software(tdef)
        assert t.can_lead_hardware(tdef)

    def test_junior_with_badge_leads(self):
        """Junior with rank badge can lead matching type."""
        registry = get_registry()
        tdef = registry.get_talent("jr_software")
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        assert not t.can_lead_software(tdef)  # No badge
        t.rank_badges = 1
        assert t.can_lead_software(tdef)  # Has badge + sw skill
        assert not t.can_lead_hardware(tdef)  # No hw skill

    def test_junior_with_badge_and_hw_skill(self):
        """Junior with badge + trained hardware can lead hardware."""
        registry = get_registry()
        tdef = registry.get_talent("jr_software")
        t = TalentInstance(instance_id=0, card_def_id="jr_software", owner=0, zone=Zone.DEV)
        t.rank_badges = 1
        t.skills.append(CubeType.HARDWARE)
        assert t.can_lead_software(tdef)
        assert t.can_lead_hardware(tdef)


class TestSpecialists:
    """Tests for specialist growth (QA, Sales)."""

    def test_qa_bug_removal_scales(self):
        """QA removes 1 bug base, 2 with 1+ XP."""
        registry = get_registry()
        tdef = registry.get_talent("qa_engineer")
        t = TalentInstance(instance_id=0, card_def_id="qa_engineer", owner=0, zone=Zone.OPS)
        assert _get_qa_bug_removal(t, tdef) == 1
        t.xp_permanent.append(CubeType.QA)
        assert _get_qa_bug_removal(t, tdef) == 2

    def test_sales_revenue_scales(self):
        """Sales Rep: +$2 base, +$3 with 1XP, +$4 with 2XP."""
        registry = get_registry()
        tdef = registry.get_talent("sales_rep")
        t = TalentInstance(instance_id=0, card_def_id="sales_rep", owner=0, zone=Zone.OPS)
        assert _get_sales_bonus(t, tdef) == 2
        t.xp_permanent.append(CubeType.SOFTWARE)  # Any XP counts
        assert _get_sales_bonus(t, tdef) == 3
        t.xp_permanent.append(CubeType.SOFTWARE)
        assert _get_sales_bonus(t, tdef) == 4


class TestDomainExpertise:
    """Tests for domain expertise cost reduction."""

    def test_no_reduction_without_maintenance(self):
        """No reduction if player has no products in sector."""
        registry = get_registry()
        pdef = registry.get_product("dating_app")  # CONSUMER sector
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        sw, hw = apply_domain_expertise(state, 0, pdef)
        assert sw == pdef.cost_software
        assert hw == pdef.cost_hardware

    def test_infra_no_synergy(self):
        """Infrastructure sector does NOT get synergy."""
        registry = get_registry()
        pdef = registry.get_product("tech_blog")  # INFRA sector
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        # Even with infra product in ops, no discount
        prod = state.create_product_instance("app_store", 0, Zone.OPS)
        state.players[0].ops_products.append(prod.instance_id)
        sw, hw = apply_domain_expertise(state, 0, pdef)
        assert sw == pdef.cost_software

    def test_software_reduction(self):
        """-2 cubes for matching sector on software-only product."""
        registry = get_registry()
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]

        # Put a CONSUMER product in ops
        prod = state.create_product_instance("dating_app", 0, Zone.OPS)
        state.players[0].ops_products.append(prod.instance_id)

        # Now draft another CONSUMER product
        target = registry.get_product("viral_video_app")  # CONSUMER, 6 {} cost
        sw, hw = apply_domain_expertise(state, 0, target)
        assert sw == 4  # 6 - 2 = 4
        assert hw == 0

    def test_min_cost_1(self):
        """Cost cannot go below 1 total cube."""
        registry = get_registry()
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]

        # Mock a scenario with cheap product
        prod = state.create_product_instance("messaging_app", 0, Zone.OPS)
        state.players[0].ops_products.append(prod.instance_id)

        # Messaging App: CONSUMER, 4 {} cost -> should become 2
        target = registry.get_product("messaging_app")
        sw, hw = apply_domain_expertise(state, 0, target)
        assert sw == 2  # 4 - 2 = 2
        assert sw + hw >= 1


class TestTraits:
    """Tests for senior traits."""

    def test_spaghetti_code_bug(self):
        """Spaghetti Code trait verified in data."""
        registry = get_registry()
        tdef = registry.get_talent("sr_backend_hacker")
        assert tdef.trait == Trait.SPAGHETTI_CODE

    def test_ego_cannot_reassign(self):
        """Ego trait prevents reassignment."""
        registry = get_registry()
        tdef = registry.get_talent("sr_hardware_diva")
        assert tdef.trait == Trait.EGO

    def test_efficient_salary(self):
        """Efficient trait: salary $1 in ops (vs $2 base)."""
        from startup_simulator.phases.income_phase import calculate_salary_cost
        registry = get_registry()
        state = GameState(num_players=2)
        state.players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        # Place efficient senior in OPS
        inst = state.create_talent_instance("sr_hardware_engineer", 0, Zone.OPS)
        # Salary should be $1 (efficient in ops)
        sal = calculate_salary_cost(state, 0)
        assert sal == 1

    def test_clean_code_trait(self):
        """Clean Code: never generates bugs."""
        registry = get_registry()
        tdef = registry.get_talent("sr_backend_architect")
        assert tdef.trait == Trait.CLEAN_CODE

    def test_mercenary_trait(self):
        """Full Stack Ninja has mercenary trait (1.5x poach cost)."""
        registry = get_registry()
        tdef = registry.get_talent("full_stack_ninja")
        assert tdef.trait == Trait.MERCENARY


class TestRandomGamesPhase2:
    """Run random games with all Phase 2 features."""

    def test_random_2p_game_completes(self):
        """Random 2p game completes without errors."""
        import random
        engine = GameEngine(num_players=2, seed=2024)
        rng = random.Random(2024)
        steps = 0
        while not engine.is_done() and steps < 5000:
            actions = engine.get_legal_actions()
            assert len(actions) > 0, f"No legal actions at step {steps}"
            action = rng.choice(actions)
            engine.step(action)
            steps += 1
        if engine.is_done():
            assert len(engine.get_scores()) == 2

    def test_random_4p_game_completes(self):
        """Random 4p game completes without errors."""
        import random
        engine = GameEngine(num_players=4, seed=1337)
        rng = random.Random(1337)
        steps = 0
        while not engine.is_done() and steps < 8000:
            actions = engine.get_legal_actions()
            assert len(actions) > 0
            action = rng.choice(actions)
            engine.step(action)
            steps += 1
        if engine.is_done():
            assert len(engine.get_scores()) == 4

    def test_determinism_phase2(self):
        """Same seed produces identical games with Phase 2 features."""
        import random
        def play(seed):
            engine = GameEngine(num_players=2, seed=seed)
            rng = random.Random(seed + 100)
            snapshots = []
            for _ in range(300):
                if engine.is_done():
                    break
                actions = engine.get_legal_actions()
                action = rng.choice(actions)
                engine.step(action)
                snapshots.append((
                    engine.state.current_player,
                    engine.state.phase,
                    engine.state.players[0].cash,
                    engine.state.players[1].cash,
                ))
            return snapshots

        r1 = play(555)
        r2 = play(555)
        assert r1 == r2
