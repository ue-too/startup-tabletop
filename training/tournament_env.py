"""Tournament environment for multi-agent round-robin training.

4-player game where 1 agent is the learner (controlled by SB3) and
the other 3 are frozen opponents loaded from checkpoints.
The learner always sees the game from their own perspective.
"""

from __future__ import annotations

import os
import random as _random

import gymnasium as gym
import numpy as np

from env.action_space import MAX_ACTIONS
from env.observation_space import OBS_SIZE, encode_observation
from env.reward import estimate_valuation, shaped_reward, sparse_reward
from startup_simulator.engine import GameEngine


class TournamentEnv(gym.Env):
    """4-player env where learner_id is controlled by SB3, others by frozen models."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        learner_id: int = 0,
        num_players: int = 4,
        seed: int | None = None,
        reward_mode: str = "shaped",
        max_steps: int = 3000,
        opponent_dirs: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.learner_id = learner_id
        self.num_players = num_players
        self._seed = seed or np.random.randint(0, 2**31)
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.opponent_dirs = opponent_dirs or []
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

        # Opponent models: one per non-learner player
        self._opponent_models: dict[int, object] = {}

    def _load_opponents(self) -> None:
        """Load random checkpoint for each opponent from their directories."""
        import torch
        torch.distributions.Distribution.set_default_validate_args(False)
        from sb3_contrib import MaskablePPO

        self._opponent_models = {}
        for pid in range(self.num_players):
            if pid == self.learner_id:
                continue

            # Find checkpoint dir for this opponent
            if pid < len(self.opponent_dirs) + (1 if pid > self.learner_id else 0):
                # Map opponent index to directory
                opp_idx = pid if pid < self.learner_id else pid - 1
                if opp_idx < len(self.opponent_dirs):
                    opp_dir = self.opponent_dirs[opp_idx]
                else:
                    opp_dir = None
            else:
                opp_dir = None

            if opp_dir and os.path.exists(opp_dir):
                zips = [f for f in os.listdir(opp_dir) if f.endswith(".zip")]
                if zips:
                    path = os.path.join(opp_dir, self._rng.choice(zips))
                    try:
                        self._opponent_models[pid] = MaskablePPO.load(path)
                        continue
                    except Exception:
                        pass

            self._opponent_models[pid] = None  # Random fallback

    def _opponent_action(self, pid: int) -> int:
        """Get action from a frozen opponent."""
        model = self._opponent_models.get(pid)
        obs = self._get_obs_for(pid)

        if model is not None:
            try:
                action, _ = model.predict(obs, action_masks=obs["action_mask"])
                return int(action)
            except (ValueError, Exception):
                pass

        # Random fallback
        mask = obs["action_mask"]
        legal = np.nonzero(mask)[0]
        return int(self._rng.choice(legal)) if len(legal) > 0 else 0

    def reset(self, seed=None, options=None):
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

        self._load_opponents()
        self._play_opponent_turns()
        self._update_legal()
        obs = self._get_obs()
        return obs, {"legal_actions": len(self._legal_actions)}

    def step(self, action: int):
        if self.engine is None or self.engine.is_done():
            return self._get_obs(), 0.0, True, False, {}

        if action < 0 or action >= len(self._legal_actions):
            action = 0
        game_action = self._legal_actions[action]
        self.engine.step(game_action)
        self._step_count += 1

        self._play_opponent_turns()

        # Reward for the learner
        if self.reward_mode == "sparse":
            reward = sparse_reward(self.engine.state, self.learner_id)
        else:
            reward, new_val = shaped_reward(
                self.engine.state, self.learner_id,
                self._valuations[self.learner_id],
            )
            self._valuations[self.learner_id] = new_val

        terminated = self.engine.is_done()
        truncated = self._step_count >= self.max_steps and not terminated

        self._update_legal()
        obs = self._get_obs()

        info = {}
        if terminated:
            scores = self.engine.get_scores()
            info["scores"] = scores
            info["final_turn"] = self.engine.state.turn_number
            info["learner_score"] = scores[self.learner_id]
            info["learner_rank"] = sorted(scores, reverse=True).index(scores[self.learner_id]) + 1
            info["won"] = scores[self.learner_id] == max(scores)

        return obs, float(reward), terminated, truncated, info

    def _play_opponent_turns(self) -> None:
        while (self.engine and not self.engine.is_done()
               and self.engine.get_current_agent() != self.learner_id
               and self._step_count < self.max_steps):
            self._update_legal()
            pid = self.engine.get_current_agent()
            action = self._opponent_action(pid)
            if action >= len(self._legal_actions):
                action = 0
            game_action = self._legal_actions[action]
            self.engine.step(game_action)
            self._step_count += 1

    def _update_legal(self):
        if self.engine and not self.engine.is_done():
            self._legal_actions = self.engine.get_legal_actions()
        else:
            self._legal_actions = []

    def _get_obs(self) -> dict:
        return self._get_obs_for(self.learner_id)

    def _get_obs_for(self, pid: int) -> dict:
        if self.engine is None or self.engine.is_done():
            return {
                "observation": np.zeros(OBS_SIZE, dtype=np.float32),
                "action_mask": np.zeros(MAX_ACTIONS, dtype=np.int8),
            }
        obs = encode_observation(self.engine.state, pid)
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        n = min(len(self._legal_actions), MAX_ACTIONS)
        mask[:n] = 1
        return {"observation": obs, "action_mask": mask}

    def action_masks(self) -> np.ndarray:
        return self._get_obs()["action_mask"]
