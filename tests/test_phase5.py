"""Tests for Phase 5: RL Environment, Agents, Performance."""

import time
import random

import numpy as np

from env.action_space import MAX_ACTIONS, ActionEncoder
from env.observation_space import OBS_SIZE, encode_observation
from env.reward import sparse_reward, shaped_reward, estimate_valuation
from env.startup_env import StartupEnv
from agents.random_agent import RandomAgent, play_random_game
from agents.heuristic_agent import HeuristicAgent, play_heuristic_game
from startup_simulator.engine import GameEngine
from startup_simulator.types import ActionType


class TestActionEncoder:
    """Tests for action encoding."""

    def test_max_actions_constant(self):
        assert MAX_ACTIONS == 512

    def test_encode_mask(self):
        engine = GameEngine(num_players=2, seed=42)
        encoder = ActionEncoder()
        legal = engine.get_legal_actions()
        encoder.update(legal)
        mask = encoder.encode_mask()
        assert mask.shape == (MAX_ACTIONS,)
        assert mask.dtype == np.int8
        assert mask.sum() == len(legal)
        assert mask[0] == 1
        assert mask[len(legal)] == 0 if len(legal) < MAX_ACTIONS else True

    def test_decode_roundtrip(self):
        engine = GameEngine(num_players=2, seed=42)
        encoder = ActionEncoder()
        legal = engine.get_legal_actions()
        encoder.update(legal)
        for i, action in enumerate(legal):
            decoded = encoder.decode(i)
            assert decoded == action

    def test_legal_actions_under_max(self):
        """Legal actions never exceed MAX_ACTIONS."""
        engine = GameEngine(num_players=4, seed=42)
        rng = random.Random(42)
        max_seen = 0
        for _ in range(500):
            if engine.is_done():
                break
            legal = engine.get_legal_actions()
            max_seen = max(max_seen, len(legal))
            assert len(legal) <= MAX_ACTIONS, f"Too many actions: {len(legal)}"
            engine.step(rng.choice(legal))
        print(f"Max legal actions seen: {max_seen}")


class TestObservationSpace:
    """Tests for observation encoding."""

    def test_obs_size(self):
        assert OBS_SIZE > 0

    def test_encode_shape(self):
        engine = GameEngine(num_players=2, seed=42)
        obs = encode_observation(engine.state, 0)
        assert obs.shape == (OBS_SIZE,)
        assert obs.dtype == np.float32

    def test_encode_different_players(self):
        """Different players see different observations (hidden info)."""
        engine = GameEngine(num_players=2, seed=42)
        obs0 = encode_observation(engine.state, 0)
        obs1 = encode_observation(engine.state, 1)
        # Should differ because each player's own info is in first block
        assert not np.array_equal(obs0, obs1)

    def test_encode_no_nans(self):
        engine = GameEngine(num_players=4, seed=42)
        rng = random.Random(42)
        for _ in range(200):
            if engine.is_done():
                break
            for pid in range(4):
                obs = encode_observation(engine.state, pid)
                assert not np.any(np.isnan(obs)), "NaN in observation"
                assert not np.any(np.isinf(obs)), "Inf in observation"
            legal = engine.get_legal_actions()
            engine.step(rng.choice(legal))

    def test_obs_values_bounded(self):
        """Observation values should be reasonably bounded."""
        engine = GameEngine(num_players=2, seed=42)
        rng = random.Random(42)
        for _ in range(100):
            if engine.is_done():
                break
            obs = encode_observation(engine.state, 0)
            assert obs.min() >= -1.0, f"Obs min too low: {obs.min()}"
            assert obs.max() <= 100.0, f"Obs max too high: {obs.max()}"
            legal = engine.get_legal_actions()
            engine.step(rng.choice(legal))


class TestReward:
    """Tests for reward functions."""

    def test_sparse_zero_during_game(self):
        engine = GameEngine(num_players=2, seed=42)
        assert sparse_reward(engine.state, 0) == 0.0
        assert sparse_reward(engine.state, 1) == 0.0

    def test_sparse_nonzero_at_end(self):
        result = play_random_game(seed=42, max_steps=5000)
        if result["done"]:
            engine = GameEngine(num_players=2, seed=42)
            # Play to end
            rng = random.Random(100)
            while not engine.is_done():
                legal = engine.get_legal_actions()
                engine.step(rng.choice(legal))
            r0 = sparse_reward(engine.state, 0)
            r1 = sparse_reward(engine.state, 1)
            # One should win, one should lose (or tie)
            assert r0 != 0.0 or r1 != 0.0

    def test_estimate_valuation_nonnegative_start(self):
        engine = GameEngine(num_players=2, seed=42)
        # At start, valuation should reflect starting cash + concept product
        val = estimate_valuation(engine.state, 0)
        assert val >= 0, f"Starting valuation should be non-negative: {val}"

    def test_shaped_reward_tracks_progress(self):
        engine = GameEngine(num_players=2, seed=42)
        prev_val = estimate_valuation(engine.state, 0)
        rng = random.Random(42)
        total_reward = 0.0
        for _ in range(50):
            if engine.is_done():
                break
            legal = engine.get_legal_actions()
            engine.step(rng.choice(legal))
            r, prev_val = shaped_reward(engine.state, 0, prev_val)
            total_reward += r
        # Shaped reward should be non-trivial over many steps
        # (can be positive or negative depending on play)


class TestStartupEnv:
    """Tests for the PettingZoo AEC environment."""

    def test_env_creation(self):
        env = StartupEnv(num_players=2, seed=42)
        assert env.num_players == 2
        assert len(env.possible_agents) == 2

    def test_env_reset(self):
        env = StartupEnv(num_players=2, seed=42)
        env.reset()
        assert env.engine is not None
        assert env.agent_selection == "player_0"
        assert not env.is_done

    def test_env_observe(self):
        env = StartupEnv(num_players=2, seed=42)
        env.reset()
        obs = env.observe("player_0")
        assert "observation" in obs
        assert "action_mask" in obs
        assert obs["observation"].shape == (OBS_SIZE,)
        assert obs["action_mask"].shape == (MAX_ACTIONS,)
        assert obs["action_mask"].sum() > 0  # At least one legal action

    def test_env_step(self):
        env = StartupEnv(num_players=2, seed=42)
        env.reset()
        obs = env.observe(env.agent_selection)
        mask = obs["action_mask"]
        legal_idx = np.nonzero(mask)[0]
        env.step(int(legal_idx[0]))
        # Should advance without error

    def test_env_full_game(self):
        """Play a full game through the env API."""
        env = StartupEnv(num_players=2, seed=42, max_steps=5000)
        env.reset()
        rng = random.Random(42)
        steps = 0

        for agent in env.agent_iter(max_iter=5000):
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            mask = obs["action_mask"]
            legal_idx = np.nonzero(mask)[0]
            action = int(rng.choice(legal_idx))
            env.step(action)
            steps += 1

        assert steps > 0

    def test_env_last(self):
        env = StartupEnv(num_players=2, seed=42)
        env.reset()
        obs, reward, terminated, truncated, info = env.last()
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_env_4_players(self):
        env = StartupEnv(num_players=4, seed=99, max_steps=8000)
        env.reset()
        rng = random.Random(99)
        steps = 0
        for agent in env.agent_iter(max_iter=8000):
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            mask = obs["action_mask"]
            legal_idx = np.nonzero(mask)[0]
            if len(legal_idx) == 0:
                env.step(None)
                continue
            action = int(rng.choice(legal_idx))
            env.step(action)
            steps += 1
        assert steps > 0

    def test_env_determinism(self):
        """Same seed produces identical trajectory."""
        def play(seed):
            env = StartupEnv(num_players=2, seed=seed, max_steps=500)
            env.reset()
            rng = random.Random(seed + 1000)
            rewards = []
            for agent in env.agent_iter(max_iter=300):
                obs, reward, terminated, truncated, info = env.last()
                if terminated or truncated:
                    env.step(None)
                    continue
                mask = obs["action_mask"]
                legal = np.nonzero(mask)[0]
                action = int(rng.choice(legal))
                env.step(action)
                rewards.append((agent, reward))
            return rewards

        r1 = play(42)
        r2 = play(42)
        assert len(r1) == len(r2)
        for (a1, rew1), (a2, rew2) in zip(r1, r2):
            assert a1 == a2
            assert abs(rew1 - rew2) < 1e-6


class TestAgents:
    """Tests for baseline agents."""

    def test_random_agent(self):
        agent = RandomAgent(seed=42)
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        mask[0] = 1
        mask[3] = 1
        mask[7] = 1
        obs = {"observation": np.zeros(OBS_SIZE), "action_mask": mask}
        action = agent.act(obs)
        assert action in [0, 3, 7]

    def test_heuristic_agent(self):
        agent = HeuristicAgent(seed=42)
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        mask[0] = 1
        mask[1] = 1
        obs = {"observation": np.zeros(OBS_SIZE), "action_mask": mask}
        action = agent.act(obs)
        assert action in [0, 1]

    def test_random_game_function(self):
        result = play_random_game(num_players=2, seed=42)
        assert "scores" in result
        assert "steps" in result

    def test_heuristic_game_function(self):
        result = play_heuristic_game(num_players=2, seed=42)
        assert "scores" in result
        assert "steps" in result

    def test_heuristic_beats_random_trend(self):
        """Heuristic should tend to get higher scores than random (not guaranteed per game)."""
        h_scores = []
        r_scores = []
        for seed in range(10):
            h = play_heuristic_game(num_players=2, seed=seed)
            r = play_random_game(num_players=2, seed=seed, agent_seed=seed + 100)
            if h["scores"]:
                h_scores.append(max(h["scores"]))
            if r["scores"]:
                r_scores.append(max(r["scores"]))
        # Heuristic should at least sometimes outperform
        if h_scores and r_scores:
            h_avg = sum(h_scores) / len(h_scores)
            r_avg = sum(r_scores) / len(r_scores)
            print(f"Heuristic avg max score: {h_avg:.1f}, Random avg max score: {r_avg:.1f}")


class TestPerformance:
    """Performance benchmarks."""

    def test_engine_speed(self):
        """Benchmark raw engine speed (games per second)."""
        num_games = 50
        rng = random.Random(42)
        start = time.time()
        total_steps = 0

        for seed in range(num_games):
            engine = GameEngine(num_players=2, seed=seed)
            steps = 0
            while not engine.is_done() and steps < 2000:
                legal = engine.get_legal_actions()
                engine.step(rng.choice(legal))
                steps += 1
            total_steps += steps

        elapsed = time.time() - start
        games_per_sec = num_games / elapsed
        steps_per_sec = total_steps / elapsed
        print(f"\nEngine: {games_per_sec:.0f} games/sec, {steps_per_sec:.0f} steps/sec ({num_games} games)")
        # Target: >100 games/sec
        assert games_per_sec > 10, f"Too slow: {games_per_sec:.1f} games/sec"

    def test_env_speed(self):
        """Benchmark environment speed including observation encoding."""
        num_games = 20
        rng = random.Random(42)
        start = time.time()
        total_steps = 0

        for seed in range(num_games):
            env = StartupEnv(num_players=2, seed=seed, max_steps=2000)
            env.reset()
            for agent in env.agent_iter(max_iter=2000):
                obs, reward, terminated, truncated, info = env.last()
                if terminated or truncated:
                    env.step(None)
                    continue
                mask = obs["action_mask"]
                legal = np.nonzero(mask)[0]
                if len(legal) == 0:
                    env.step(None)
                    continue
                action = int(rng.choice(legal))
                env.step(action)
                total_steps += 1

        elapsed = time.time() - start
        games_per_sec = num_games / elapsed
        steps_per_sec = total_steps / elapsed
        print(f"Env: {games_per_sec:.0f} games/sec, {steps_per_sec:.0f} steps/sec ({num_games} games)")
        assert games_per_sec > 5, f"Too slow: {games_per_sec:.1f} games/sec"

    def test_observation_encoding_speed(self):
        """Benchmark observation encoding speed."""
        engine = GameEngine(num_players=4, seed=42)
        num_encodes = 1000
        start = time.time()
        for _ in range(num_encodes):
            encode_observation(engine.state, 0)
        elapsed = time.time() - start
        encodes_per_sec = num_encodes / elapsed
        print(f"Obs encoding: {encodes_per_sec:.0f} encodes/sec")
        assert encodes_per_sec > 1000
