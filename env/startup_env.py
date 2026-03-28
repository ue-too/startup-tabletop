"""PettingZoo AEC Environment for Startup Simulator.

Usage with PettingZoo:
    from env.startup_env import StartupEnv
    env = StartupEnv(num_players=2, seed=42)
    env.reset()

    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            action = env.action_space(agent).sample(mask=obs["action_mask"])
        env.step(action)

Usage without PettingZoo (standalone):
    env = StartupEnv(num_players=2, seed=42)
    env.reset()
    while not env.is_done():
        agent = env.agent_selection
        obs = env.observe(agent)
        action = pick_action(obs)  # Your policy
        env.step(action)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from startup_simulator.engine import GameEngine
from startup_simulator.types import Phase

from .action_space import MAX_ACTIONS, ActionEncoder
from .observation_space import OBS_SIZE, encode_observation
from .reward import estimate_valuation, shaped_reward, sparse_reward

# Try to import PettingZoo; fall back to a minimal duck-typed base
try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import agent_selector
    from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False
    AECEnv = object  # type: ignore


class StartupEnv(AECEnv):
    """Startup Simulator as a PettingZoo AEC environment."""

    metadata = {
        "render_modes": [],
        "name": "startup_simulator_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 2,
        seed: int = 42,
        reward_mode: str = "shaped",
        max_steps: int = 10000,
        render_mode: str | None = None,
    ) -> None:
        if HAS_PETTINGZOO:
            super().__init__()

        self.num_players = num_players
        self._seed = seed
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents: list[str] = []

        # Will be set in reset()
        self.engine: GameEngine | None = None
        self._action_encoders: dict[str, ActionEncoder] = {}
        self._valuations: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self._step_count = 0

        # Spaces
        if HAS_PETTINGZOO:
            self.action_spaces = {
                agent: Discrete(MAX_ACTIONS) for agent in self.possible_agents
            }
            self.observation_spaces = {
                agent: Dict({
                    "observation": Box(
                        low=-1.0, high=100.0,
                        shape=(OBS_SIZE,), dtype=np.float32,
                    ),
                    "action_mask": Box(
                        low=0, high=1,
                        shape=(MAX_ACTIONS,), dtype=np.int8,
                    ),
                })
                for agent in self.possible_agents
            }

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """Reset the environment."""
        if seed is not None:
            self._seed = seed
        self.engine = GameEngine(num_players=self.num_players, seed=self._seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0

        self._action_encoders = {
            agent: ActionEncoder() for agent in self.possible_agents
        }
        self._valuations = {
            agent: estimate_valuation(self.engine.state, i)
            for i, agent in enumerate(self.possible_agents)
        }
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}

        # PettingZoo state
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos: dict[str, dict] = {agent: {} for agent in self.agents}

        self._update_agent_selection()
        self._update_action_encoder()

    def _update_agent_selection(self) -> None:
        """Set agent_selection to the current player."""
        if self.engine is None:
            return
        pid = self.engine.get_current_agent()
        self.agent_selection = f"player_{pid}"

    def _update_action_encoder(self) -> None:
        """Update the action encoder for the current agent."""
        if self.engine is None:
            return
        agent = self.agent_selection
        legal = self.engine.get_legal_actions()
        self._action_encoders[agent].update(legal)

    @property
    def is_done(self) -> bool:
        if self.engine is None:
            return True
        return self.engine.is_done()

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        """Get observation for an agent."""
        if self.engine is None:
            return {
                "observation": np.zeros(OBS_SIZE, dtype=np.float32),
                "action_mask": np.zeros(MAX_ACTIONS, dtype=np.int8),
            }

        pid = int(agent.split("_")[1])
        obs = encode_observation(self.engine.state, pid)

        # Action mask: only valid for current agent
        if agent == self.agent_selection:
            mask = self._action_encoders[agent].encode_mask()
        else:
            mask = np.zeros(MAX_ACTIONS, dtype=np.int8)

        return {"observation": obs, "action_mask": mask}

    def step(self, action: int | None) -> None:
        """Take an action in the environment.

        Args:
            action: Integer index into the legal action list, or None for terminated agents.
        """
        if self.engine is None:
            return

        agent = self.agent_selection
        pid = int(agent.split("_")[1])

        # Handle terminated/truncated agents
        if action is None or self.terminations.get(agent, False) or self.truncations.get(agent, False):
            self._was_dead_step(action)
            return

        # Decode and execute action
        encoder = self._action_encoders[agent]
        game_action = encoder.decode(action)
        result = self.engine.step(game_action)

        self._step_count += 1

        # Calculate rewards
        for i, ag in enumerate(self.possible_agents):
            if self.reward_mode == "sparse":
                self.rewards[ag] = sparse_reward(self.engine.state, i)
            else:
                reward, new_val = shaped_reward(
                    self.engine.state, i, self._valuations[ag]
                )
                self.rewards[ag] = reward
                self._valuations[ag] = new_val
            self._cumulative_rewards[ag] += self.rewards[ag]

        # Check termination
        if self.engine.is_done():
            for ag in self.agents:
                self.terminations[ag] = True
                self.infos[ag] = {
                    "scores": self.engine.get_scores(),
                    "steps": self._step_count,
                }
        elif self._step_count >= self.max_steps:
            for ag in self.agents:
                self.truncations[ag] = True
                self.infos[ag] = {"truncated": True, "steps": self._step_count}

        # Update selection and encoder for next agent
        self._update_agent_selection()
        if not self.engine.is_done():
            self._update_action_encoder()

    def last(self, observe: bool = True) -> tuple[dict | None, float, bool, bool, dict]:
        """Get the last observation, reward, termination, truncation, info for current agent."""
        agent = self.agent_selection
        obs = self.observe(agent) if observe else None
        return (
            obs,
            self.rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def _was_dead_step(self, action: int | None) -> None:
        """Handle step for a dead/terminated agent."""
        # Advance to next live agent
        if self.engine and not self.engine.is_done():
            self._update_agent_selection()
            self._update_action_encoder()

    # --- Gymnasium / PettingZoo API helpers ---

    def action_space(self, agent: str):
        """Return action space for an agent."""
        if HAS_PETTINGZOO:
            return self.action_spaces[agent]
        return None

    def observation_space(self, agent: str):
        """Return observation space for an agent."""
        if HAS_PETTINGZOO:
            return self.observation_spaces[agent]
        return None

    def agent_iter(self, max_iter: int = 2**63):
        """Iterate over agents that need to act."""
        count = 0
        while count < max_iter and self.agents:
            if all(self.terminations.get(a, False) or self.truncations.get(a, False)
                   for a in self.agents):
                break
            yield self.agent_selection
            count += 1

    def close(self) -> None:
        pass

    def __str__(self) -> str:
        return f"StartupEnv(players={self.num_players}, seed={self._seed})"
