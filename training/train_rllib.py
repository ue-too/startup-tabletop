"""RLlib multi-agent self-play training script.

Usage:
    pip install -e ".[rl]"
    python training/train_rllib.py
    python training/train_rllib.py --iterations 100 --players 2
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
    import gymnasium as gym
    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False
    MultiAgentEnv = object  # type: ignore
    print("WARNING: ray[rllib] not installed. Run: pip install -e '.[rl]'")

from env.action_space import MAX_ACTIONS
from env.observation_space import OBS_SIZE, encode_observation
from env.reward import estimate_valuation, shaped_reward
from startup_simulator.engine import GameEngine


class StartupMultiAgentEnv(MultiAgentEnv if HAS_RLLIB else object):
    """RLlib multi-agent env for Startup Simulator.

    Each player is a separate agent. All agents share the same policy
    for self-play (configured via policy mapping).
    """

    def __init__(self, config: dict | None = None):
        if HAS_RLLIB:
            super().__init__()
        config = config or {}
        self.num_players = config.get("num_players", 2)
        self.max_steps = config.get("max_steps", 5000)
        self.reward_mode = config.get("reward_mode", "shaped")
        self._seed = config.get("seed", 42)
        self._game_count = 0

        self._agent_ids = {f"player_{i}" for i in range(self.num_players)}

        obs_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(
                low=-1.0, high=100.0, shape=(OBS_SIZE,), dtype=np.float32,
            ),
            "action_mask": gym.spaces.Box(
                low=0, high=1, shape=(MAX_ACTIONS,), dtype=np.int8,
            ),
        })
        act_space = gym.spaces.Discrete(MAX_ACTIONS)

        self.observation_space = obs_space
        self.action_space = act_space
        self._observation_spaces = {aid: obs_space for aid in self._agent_ids}
        self._action_spaces = {aid: act_space for aid in self._agent_ids}

        self.engine: GameEngine | None = None
        self._legal_actions: list = []
        self._valuations: dict[str, float] = {}
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        self._game_count += 1
        game_seed = (self._seed + self._game_count * 7919) % (2**31)
        self.engine = GameEngine(num_players=self.num_players, seed=game_seed)
        self._step_count = 0
        self._valuations = {
            f"player_{i}": estimate_valuation(self.engine.state, i)
            for i in range(self.num_players)
        }

        self._legal_actions = self.engine.get_legal_actions()
        current_agent = f"player_{self.engine.get_current_agent()}"

        obs = {current_agent: self._get_obs(self.engine.get_current_agent())}
        infos = {current_agent: {}}
        return obs, infos

    def step(self, action_dict):
        if self.engine is None or self.engine.is_done():
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

        # Get the current agent's action
        current_agent = f"player_{self.engine.get_current_agent()}"
        action = action_dict.get(current_agent, 0)
        current_pid = self.engine.get_current_agent()

        # Execute
        if action < 0 or action >= len(self._legal_actions):
            action = 0
        game_action = self._legal_actions[action]
        self.engine.step(game_action)
        self._step_count += 1

        # Calculate rewards
        rewards = {}
        for i in range(self.num_players):
            aid = f"player_{i}"
            if self.reward_mode == "sparse":
                from env.reward import sparse_reward
                rewards[aid] = sparse_reward(self.engine.state, i)
            else:
                r, new_val = shaped_reward(self.engine.state, i, self._valuations[aid])
                rewards[aid] = r
                self._valuations[aid] = new_val

        terminated = self.engine.is_done()
        truncated = self._step_count >= self.max_steps and not terminated

        terminateds = {"__all__": terminated or truncated}
        truncateds = {"__all__": False}

        infos = {}
        if terminated:
            scores = self.engine.get_scores()
            for i in range(self.num_players):
                infos[f"player_{i}"] = {"scores": scores, "final_turn": self.engine.state.turn_number}

        # Next observation
        obs = {}
        if not terminated and not truncated:
            self._legal_actions = self.engine.get_legal_actions()
            next_agent = f"player_{self.engine.get_current_agent()}"
            obs[next_agent] = self._get_obs(self.engine.get_current_agent())
            if next_agent not in infos:
                infos[next_agent] = {}

        return obs, rewards, terminateds, truncateds, infos

    def _get_obs(self, pid: int) -> dict:
        obs = encode_observation(self.engine.state, pid)
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        n = min(len(self._legal_actions), MAX_ACTIONS)
        mask[:n] = 1
        return {"observation": obs, "action_mask": mask}


def train(
    iterations: int = 100,
    num_players: int = 2,
    num_workers: int = 2,
    reward_mode: str = "shaped",
    save_dir: str = "checkpoints/rllib",
    learning_rate: float = 3e-4,
    train_batch_size: int = 2048,
    verbose: bool = True,
) -> str:
    """Train with RLlib PPO multi-agent self-play."""
    if not HAS_RLLIB:
        raise RuntimeError("ray[rllib] required. Run: pip install -e '.[rl]'")

    os.makedirs(save_dir, exist_ok=True)
    ray.init(ignore_reinit_error=True)

    # Register environment
    register_env("startup_simulator", lambda config: StartupMultiAgentEnv(config))

    # All agents share one policy
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(
            env="startup_simulator",
            env_config={
                "num_players": num_players,
                "max_steps": 5000,
                "reward_mode": reward_mode,
            },
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            lr=learning_rate,
            train_batch_size=train_batch_size,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            num_sgd_iter=4,
        )
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length="auto",
        )
        .framework("torch")
    )

    algo = config.build()

    print(f"Training RLlib PPO for {iterations} iterations...")
    print(f"  Players: {num_players}, Workers: {num_workers}")

    best_reward = float("-inf")
    for i in range(iterations):
        result = algo.train()
        mean_reward = result.get("env_runners", {}).get("episode_reward_mean", 0)

        if verbose and (i + 1) % 10 == 0:
            print(
                f"  Iter {i + 1}/{iterations}: "
                f"reward={mean_reward:.2f}, "
                f"episodes={result.get('env_runners', {}).get('num_episodes', 0)}"
            )

        if mean_reward > best_reward:
            best_reward = mean_reward
            checkpoint_path = algo.save(save_dir)
            if verbose:
                print(f"  New best! Saved to {checkpoint_path}")

    final_path = algo.save(save_dir)
    print(f"Training complete. Final checkpoint: {final_path}")

    algo.stop()
    ray.shutdown()
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Train with RLlib multi-agent self-play")
    parser.add_argument("--iterations", "-i", type=int, default=100)
    parser.add_argument("--players", "-p", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--workers", "-w", type=int, default=2)
    parser.add_argument("--reward", type=str, default="shaped", choices=["sparse", "shaped"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints/rllib")
    args = parser.parse_args()

    train(
        iterations=args.iterations,
        num_players=args.players,
        num_workers=args.workers,
        reward_mode=args.reward,
        save_dir=args.save_dir,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
