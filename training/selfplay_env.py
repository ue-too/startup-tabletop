"""Self-play Gymnasium wrapper for SB3.

Wraps the multi-agent StartupEnv into a single-agent Gymnasium env where
both (all) players share the same policy. From SB3's perspective, it's
playing a single-player game where each "step" is one player's action.

The opponent's actions are chosen by the same policy (latest weights),
making this a latest-vs-latest self-play setup.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from env.action_space import MAX_ACTIONS
from env.observation_space import OBS_SIZE, encode_observation
from env.reward import estimate_valuation, shaped_reward, sparse_reward
from startup_simulator.engine import GameEngine


class SelfPlayEnv(gym.Env):
    """Single-agent Gymnasium env for self-play training.

    Both players use the same policy. The env alternates between players,
    collecting actions from the external agent for ALL players.

    Observation: dict with "observation" (float array) and "action_mask" (binary).
    Action: int index into legal action list.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_players: int = 2,
        seed: int | None = None,
        reward_mode: str = "shaped",
        max_steps: int = 3000,
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self._seed = seed or np.random.randint(0, 2**31)
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self._game_count = 0

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

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset and start a new game."""
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

        self._update_legal()
        obs = self._get_obs()
        return obs, {"legal_actions": len(self._legal_actions)}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Take one action (for whichever player is current)."""
        if self.engine is None or self.engine.is_done():
            obs = self._get_obs()
            return obs, 0.0, True, False, {}

        # Decode action
        if action < 0 or action >= len(self._legal_actions):
            action = 0  # Fallback to first legal action

        game_action = self._legal_actions[action]
        current_pid = self.engine.get_current_agent()
        self.engine.step(game_action)
        self._step_count += 1

        # Calculate reward for the player who just acted
        if self.reward_mode == "sparse":
            reward = sparse_reward(self.engine.state, current_pid)
        else:
            reward, new_val = shaped_reward(
                self.engine.state, current_pid, self._valuations[current_pid]
            )
            self._valuations[current_pid] = new_val

        terminated = self.engine.is_done()
        truncated = self._step_count >= self.max_steps and not terminated

        if truncated:
            # Force end
            terminated = True

        self._update_legal()
        obs = self._get_obs()

        info = {
            "current_player": self.engine.get_current_agent(),
            "turn": self.engine.state.turn_number,
            "legal_actions": len(self._legal_actions),
        }
        if terminated and self.engine.is_done():
            info["scores"] = self.engine.get_scores()
            info["final_turn"] = self.engine.state.turn_number

        return obs, float(reward), terminated, False, info

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

        pid = self.engine.get_current_agent()
        obs = encode_observation(self.engine.state, pid)

        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        n = min(len(self._legal_actions), MAX_ACTIONS)
        mask[:n] = 1

        return {"observation": obs, "action_mask": mask}

    def action_masks(self) -> np.ndarray:
        """Return action mask (used by MaskablePPO)."""
        return self._get_obs()["action_mask"]
