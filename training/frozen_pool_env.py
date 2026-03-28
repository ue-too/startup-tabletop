"""Frozen-pool self-play environment for SB3.

The learning agent plays as one player. The opponent is sampled from a pool
of frozen past checkpoints. This prevents self-play collapse by maintaining
opponent diversity.

Pool is updated periodically by saving the current policy as a new opponent.
"""

from __future__ import annotations

import os
import random as _random
from typing import Any

import gymnasium as gym
import numpy as np

from env.action_space import MAX_ACTIONS
from env.observation_space import OBS_SIZE, encode_observation
from env.reward import estimate_valuation, shaped_reward, sparse_reward
from startup_simulator.engine import GameEngine


class FrozenPoolEnv(gym.Env):
    """Self-play env where the opponent is sampled from a frozen checkpoint pool.

    The learning agent always plays as player 0.
    The opponent (player 1) uses a frozen policy from the pool.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_players: int = 2,
        seed: int | None = None,
        reward_mode: str = "shaped",
        max_steps: int = 3000,
        pool_dir: str = "checkpoints/pool",
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self._seed = seed or np.random.randint(0, 2**31)
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.pool_dir = pool_dir
        self._game_count = 0
        self._rng = _random.Random(seed)

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(
                low=-1.0, high=100.0, shape=(OBS_SIZE,), dtype=np.float32,
            ),
            "action_mask": gym.spaces.Box(
                low=0, high=1, shape=(MAX_ACTIONS,), dtype=np.int8,
            ),
        })
        self.action_space = gym.spaces.Discrete(MAX_ACTIONS)

        self.engine: GameEngine | None = None
        self._legal_actions: list = []
        self._valuations: list[float] = []
        self._step_count = 0

        # Opponent pool
        self._opponent_model = None
        self._pool_paths: list[str] = []
        self._refresh_pool()

    def _refresh_pool(self) -> None:
        """Scan pool directory for checkpoint files."""
        self._pool_paths = []
        if os.path.exists(self.pool_dir):
            for f in sorted(os.listdir(self.pool_dir)):
                if f.endswith(".zip"):
                    self._pool_paths.append(os.path.join(self.pool_dir, f))

    def _load_random_opponent(self) -> None:
        """Load a random opponent from the pool."""
        self._refresh_pool()
        if not self._pool_paths:
            self._opponent_model = None
            return

        import torch
        torch.distributions.Distribution.set_default_validate_args(False)
        from sb3_contrib import MaskablePPO

        path = self._rng.choice(self._pool_paths)
        try:
            self._opponent_model = MaskablePPO.load(path)
        except Exception:
            self._opponent_model = None

    def _opponent_action(self) -> int:
        """Get action from the frozen opponent."""
        if self._opponent_model is None:
            # Random fallback if no pool yet
            mask = self._get_obs()["action_mask"]
            legal = np.nonzero(mask)[0]
            return int(self._rng.choice(legal)) if len(legal) > 0 else 0

        obs = self._get_obs()
        try:
            action, _ = self._opponent_model.predict(obs, action_masks=obs["action_mask"])
            return int(action)
        except (ValueError, Exception):
            mask = obs["action_mask"]
            legal = np.nonzero(mask)[0]
            return int(self._rng.choice(legal)) if len(legal) > 0 else 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._seed = seed
        else:
            self._game_count += 1
            self._seed = (self._seed + self._game_count * 7919) % (2**31)

        self.engine = GameEngine(num_players=self.num_players, seed=self._seed)
        self._step_count = 0
        self._valuations = [
            estimate_valuation(self.engine.state, i)
            for i in range(self.num_players)
        ]

        # Load a new opponent each game
        self._load_random_opponent()

        # If opponent goes first, play their turns
        self._play_opponent_turns()

        self._update_legal()
        obs = self._get_obs()
        return obs, {"legal_actions": len(self._legal_actions)}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self.engine is None or self.engine.is_done():
            return self._get_obs(), 0.0, True, False, {}

        # Play our action (player 0)
        if action < 0 or action >= len(self._legal_actions):
            action = 0
        game_action = self._legal_actions[action]
        self.engine.step(game_action)
        self._step_count += 1

        # Play opponent turns until it's our turn again or game ends
        self._play_opponent_turns()

        # Reward for player 0
        if self.reward_mode == "sparse":
            reward = sparse_reward(self.engine.state, 0)
        else:
            reward, new_val = shaped_reward(self.engine.state, 0, self._valuations[0])
            self._valuations[0] = new_val

        terminated = self.engine.is_done()
        truncated = self._step_count >= self.max_steps and not terminated

        self._update_legal()
        obs = self._get_obs()

        info = {}
        if terminated:
            info["scores"] = self.engine.get_scores()
            info["final_turn"] = self.engine.state.turn_number

        return obs, float(reward), terminated, truncated, info

    def _play_opponent_turns(self) -> None:
        """Play all opponent turns until it's player 0's turn or game ends."""
        while (self.engine and not self.engine.is_done()
               and self.engine.get_current_agent() != 0
               and self._step_count < self.max_steps):
            self._update_legal()
            action = self._opponent_action()
            if action >= len(self._legal_actions):
                action = 0
            game_action = self._legal_actions[action]
            self.engine.step(game_action)
            self._step_count += 1

    def _update_legal(self) -> None:
        if self.engine and not self.engine.is_done():
            self._legal_actions = self.engine.get_legal_actions()
        else:
            self._legal_actions = []

    def _get_obs(self) -> dict:
        if self.engine is None or self.engine.is_done():
            return {
                "observation": np.zeros(OBS_SIZE, dtype=np.float32),
                "action_mask": np.zeros(MAX_ACTIONS, dtype=np.int8),
            }
        obs = encode_observation(self.engine.state, 0)  # Always player 0's perspective
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        n = min(len(self._legal_actions), MAX_ACTIONS)
        mask[:n] = 1
        return {"observation": obs, "action_mask": mask}

    def action_masks(self) -> np.ndarray:
        return self._get_obs()["action_mask"]
