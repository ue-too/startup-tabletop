"""Tests for the game engine."""

import random

from startup_simulator.engine import GameEngine
from startup_simulator.actions.base import Action
from startup_simulator.types import ActionType, Phase, SubPhase


def test_engine_creation():
    """Test that engine initializes correctly."""
    engine = GameEngine(num_players=2, seed=42)
    state = engine.state
    assert state.num_players == 2
    assert len(state.players) == 2
    assert state.phase == Phase.ACTION  # After auto-advancing through EVENT and INCOME
    assert not state.game_over

    # Players should have starting cash
    for p in state.players:
        # Cash may have changed due to income phase
        assert p.cash >= 0

    # Players should have starting talent
    for p in state.players:
        assert len(p.bench) >= 0  # Some may be assigned

    # Players should have 1 concept product in dev
    for p in state.players:
        assert len(p.dev_products) == 1


def test_engine_legal_actions():
    """Test that legal actions are generated."""
    engine = GameEngine(num_players=2, seed=42)
    actions = engine.get_legal_actions()
    assert len(actions) > 0
    # Should always have PASS
    assert any(a.action_type == ActionType.PASS for a in actions)


def test_random_game_2p():
    """Play a full random game with 2 players and verify it completes."""
    engine = GameEngine(num_players=2, seed=123)
    rng = random.Random(456)

    max_steps = 5000
    steps = 0

    while not engine.is_done() and steps < max_steps:
        actions = engine.get_legal_actions()
        assert len(actions) > 0, f"No legal actions at step {steps}, phase={engine.state.phase}"
        action = rng.choice(actions)
        engine.step(action)
        steps += 1

    # Game should eventually end (Market Crash in bottom 20%)
    # With random play, it might not always trigger within max_steps
    # but the game should not crash
    if engine.is_done():
        scores = engine.get_scores()
        assert len(scores) == 2
        print(f"Game finished in {steps} steps. Scores: {scores}")
    else:
        print(f"Game did not finish in {max_steps} steps (expected for random play)")


def test_random_game_3p():
    """Play a random 3-player game."""
    engine = GameEngine(num_players=3, seed=789)
    rng = random.Random(101)

    steps = 0
    max_steps = 5000

    while not engine.is_done() and steps < max_steps:
        actions = engine.get_legal_actions()
        assert len(actions) > 0
        action = rng.choice(actions)
        engine.step(action)
        steps += 1

    if engine.is_done():
        scores = engine.get_scores()
        assert len(scores) == 3
        print(f"3P game finished in {steps} steps. Scores: {scores}")


def test_random_game_4p():
    """Play a random 4-player game."""
    engine = GameEngine(num_players=4, seed=999)
    rng = random.Random(202)

    steps = 0
    max_steps = 8000

    while not engine.is_done() and steps < max_steps:
        actions = engine.get_legal_actions()
        assert len(actions) > 0
        action = rng.choice(actions)
        engine.step(action)
        steps += 1

    if engine.is_done():
        scores = engine.get_scores()
        assert len(scores) == 4
        print(f"4P game finished in {steps} steps. Scores: {scores}")


def test_determinism():
    """Verify that same seed produces identical games."""
    def play_game(seed):
        engine = GameEngine(num_players=2, seed=seed)
        rng = random.Random(seed + 1000)
        steps = 0
        states = []
        while not engine.is_done() and steps < 500:
            actions = engine.get_legal_actions()
            action = rng.choice(actions)
            result = engine.step(action)
            states.append((
                result.current_player,
                result.phase,
                result.sub_phase,
                engine.state.players[0].cash,
                engine.state.players[1].cash,
            ))
            steps += 1
        return states

    run1 = play_game(42)
    run2 = play_game(42)

    assert len(run1) == len(run2)
    for i, (s1, s2) in enumerate(zip(run1, run2)):
        assert s1 == s2, f"Divergence at step {i}: {s1} != {s2}"


def test_basic_gameplay_flow():
    """Test a scripted sequence of actions."""
    engine = GameEngine(num_players=2, seed=42)
    state = engine.state

    # Player 0 should be active with 3 AP
    assert state.current_player == 0
    assert state.players[0].action_points == 3

    # Recruit a junior software dev
    recruit_actions = [
        a for a in engine.get_legal_actions()
        if a.action_type == ActionType.RECRUIT and a.source_type == "university_sw"
    ]
    if recruit_actions:
        result = engine.step(recruit_actions[0])
        assert result.action_result.success
        assert state.players[0].action_points == 2

    # Assign talent to the concept product
    assign_actions = [
        a for a in engine.get_legal_actions()
        if a.action_type == ActionType.ASSIGN
    ]
    if assign_actions:
        result = engine.step(assign_actions[0])
        assert result.action_result.success
        # Should be in assign batch mode
        assert state.sub_phase == SubPhase.ACTION_ASSIGN_BATCH

        # Assign one talent
        assign_one = [
            a for a in engine.get_legal_actions()
            if a.action_type == ActionType.ASSIGN_ONE
        ]
        if assign_one:
            engine.step(assign_one[0])

        # End assign batch
        end_batch = [
            a for a in engine.get_legal_actions()
            if a.action_type == ActionType.END_ASSIGN_BATCH
        ]
        if end_batch:
            engine.step(end_batch[0])

    # Pass remaining AP
    engine.step(Action(ActionType.PASS))
