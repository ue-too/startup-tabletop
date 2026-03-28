"""RLlib multi-agent training with SEPARATE policies per player.

Each player has its own policy network, so they develop distinct strategies.
Trained agents can then play against each other in a league.

Usage:
    python training/train_rllib.py
    python training/train_rllib.py --iterations 200 --players 2
    python training/train_rllib.py --league  # Run league tournament between saved policies
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import gymnasium as gym

from env.observation_space import OBS_SIZE, encode_observation
from env.action_space import MAX_ACTIONS
from env.reward import estimate_valuation, shaped_reward, sparse_reward
from startup_simulator.engine import GameEngine


class StartupMultiAgentEnv(MultiAgentEnv):
    """RLlib multi-agent env with separate policies per player."""

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.num_players = config.get("num_players", 2)
        self.max_steps = config.get("max_steps", 5000)
        self.reward_mode = config.get("reward_mode", "shaped")
        self._seed = config.get("seed", 42)
        self._game_count = 0

        self._agent_ids = {f"player_{i}" for i in range(self.num_players)}

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32,
            ),
            "action_mask": gym.spaces.Box(
                low=0, high=1, shape=(MAX_ACTIONS,), dtype=np.int8,
            ),
        })
        self.action_space = gym.spaces.Discrete(MAX_ACTIONS)

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

        current_agent = f"player_{self.engine.get_current_agent()}"
        action = action_dict.get(current_agent, 0)
        current_pid = self.engine.get_current_agent()

        if action < 0 or action >= len(self._legal_actions):
            action = 0
        game_action = self._legal_actions[action]
        self.engine.step(game_action)
        self._step_count += 1

        # Per-agent rewards
        rewards = {}
        for i in range(self.num_players):
            aid = f"player_{i}"
            if self.reward_mode == "sparse":
                rewards[aid] = sparse_reward(self.engine.state, i)
            else:
                r, new_val = shaped_reward(self.engine.state, i, self._valuations[aid])
                rewards[aid] = r
                self._valuations[aid] = new_val

        terminated = self.engine.is_done()
        truncated = self._step_count >= self.max_steps and not terminated

        terminateds = {"__all__": terminated or truncated}
        truncateds = {"__all__": False}

        obs = {}
        infos = {}
        if terminated or truncated:
            # On terminal step: return dummy obs for all agents so infos keys match
            scores = self.engine.get_scores() if terminated else []
            dummy_obs = {
                "observation": np.zeros(OBS_SIZE, dtype=np.float32),
                "action_mask": np.zeros(MAX_ACTIONS, dtype=np.int8),
            }
            for i in range(self.num_players):
                aid = f"player_{i}"
                obs[aid] = dummy_obs
                infos[aid] = {
                    "scores": scores,
                    "final_turn": self.engine.state.turn_number if terminated else 0,
                    "won": bool(scores and scores[i] == max(scores)),
                }
        else:
            self._legal_actions = self.engine.get_legal_actions()
            next_agent = f"player_{self.engine.get_current_agent()}"
            obs[next_agent] = self._get_obs(self.engine.get_current_agent())
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
    separate_policies: bool = True,
    verbose: bool = True,
) -> str:
    """Train with RLlib PPO, separate policy per player."""

    os.makedirs(save_dir, exist_ok=True)
    ray.init(ignore_reinit_error=True, num_cpus=num_workers + 1)

    register_env("startup_simulator", lambda config: StartupMultiAgentEnv(config))

    # Build policy config: one policy per player, or shared
    policy_ids = [f"policy_{i}" for i in range(num_players)]
    policies = {pid: (None, None, None, {}) for pid in policy_ids}

    if separate_policies:
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            idx = int(agent_id.split("_")[1])
            return f"policy_{idx}"
    else:
        policies = {"shared_policy": (None, None, None, {})}
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
            policies=policies,
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

    mode_str = "separate" if separate_policies else "shared"
    print(f"Training RLlib PPO ({mode_str} policies) for {iterations} iterations...")
    print(f"  Players: {num_players}, Workers: {num_workers}")
    print(f"  Policies: {list(policies.keys())}")

    best_reward = float("-inf")
    start = time.time()

    for i in range(iterations):
        result = algo.train()
        mean_reward = result.get("env_runners", {}).get("episode_reward_mean", 0)

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start
            eps = result.get("env_runners", {}).get("num_episodes_lifetime", 0)

            # Per-policy rewards if available
            policy_rewards = {}
            learner = result.get("learner", {})
            for pid in policies:
                pr = learner.get(pid, {})
                if "learner_stats" in pr:
                    policy_rewards[pid] = pr["learner_stats"].get("total_loss", 0)

            print(
                f"  Iter {i + 1}/{iterations} ({elapsed:.0f}s): "
                f"reward={mean_reward:.2f}, episodes={eps}"
            )

        if mean_reward > best_reward:
            best_reward = mean_reward
            checkpoint_path = algo.save(save_dir)

    final_path = algo.save(save_dir)
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.0f}s. Checkpoint: {final_path}")

    algo.stop()
    ray.shutdown()
    return final_path


def league_tournament(
    checkpoint_dir: str = "checkpoints/rllib",
    num_games: int = 50,
    num_players: int = 2,
) -> None:
    """Run a league tournament between saved RLlib policies.

    Loads policies from checkpoint and pits them against each other
    and against baselines (random, heuristic).
    """
    from training.evaluate import evaluate_agents, make_heuristic_agent, make_random_agent
    import random

    # Find checkpoint: either the dir itself or a nested checkpoint dir
    latest = None
    if os.path.exists(os.path.join(checkpoint_dir, "policies")):
        latest = checkpoint_dir
    else:
        for item in sorted(os.listdir(checkpoint_dir)):
            path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "policies")):
                latest = path

    if latest is None:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Loading checkpoint: {latest}")

    ray.init(ignore_reinit_error=True, num_cpus=2)
    register_env("startup_simulator", lambda config: StartupMultiAgentEnv(config))

    # Rebuild config to load
    policy_ids = [f"policy_{i}" for i in range(num_players)]
    policies = {pid: (None, None, None, {}) for pid in policy_ids}

    config = (
        PPOConfig()
        .environment(
            env="startup_simulator",
            env_config={"num_players": num_players, "max_steps": 5000, "reward_mode": "shaped"},
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"policy_{int(agent_id.split('_')[1])}",
        )
        .framework("torch")
    )

    algo = config.build()
    algo.restore(latest)

    # Extract per-policy agent functions
    def make_rllib_agent(policy_id: str):
        def agent_fn(obs_dict, legal_actions):
            try:
                action = algo.compute_single_action(
                    obs_dict, policy_id=policy_id,
                )
                mask = obs_dict["action_mask"]
                if mask[action] == 0:
                    legal = np.nonzero(mask)[0]
                    action = int(np.random.choice(legal)) if len(legal) > 0 else 0
                return int(action)
            except Exception:
                mask = obs_dict["action_mask"]
                legal = np.nonzero(mask)[0]
                return int(np.random.choice(legal)) if len(legal) > 0 else 0
        return agent_fn

    agents_rllib = {pid: make_rllib_agent(pid) for pid in policy_ids}
    heuristic = make_heuristic_agent(42)
    random_ag = make_random_agent(42)

    print(f"\n=== League Tournament ({num_games} games each) ===\n")

    # Policy 0 vs Policy 1
    r = evaluate_agents([agents_rllib["policy_0"], agents_rllib["policy_1"]], num_games=num_games)
    print(f"Policy_0 vs Policy_1: {r['win_rates'][0]:.0%} vs {r['win_rates'][1]:.0%}")
    print(f"  Avg scores: {r['avg_scores'][0]:.1f} vs {r['avg_scores'][1]:.1f}")

    # Each policy vs random
    for pid in policy_ids:
        r = evaluate_agents([agents_rllib[pid], random_ag], num_games=num_games)
        print(f"{pid} vs Random: {r['win_rates'][0]:.0%} (avg {r['avg_scores'][0]:.1f})")

    # Each policy vs heuristic
    for pid in policy_ids:
        r = evaluate_agents([agents_rllib[pid], heuristic], num_games=num_games)
        print(f"{pid} vs Heuristic: {r['win_rates'][0]:.0%} (avg {r['avg_scores'][0]:.1f})")

    # Baseline
    r = evaluate_agents([heuristic, random_ag], num_games=num_games)
    print(f"Heuristic vs Random (baseline): {r['win_rates'][0]:.0%}")

    algo.stop()
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Train with RLlib per-agent policies")
    parser.add_argument("--iterations", "-i", type=int, default=100)
    parser.add_argument("--players", "-p", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--workers", "-w", type=int, default=2)
    parser.add_argument("--reward", type=str, default="shaped", choices=["sparse", "shaped"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints/rllib")
    parser.add_argument("--shared", action="store_true", help="Use shared policy (self-play)")
    parser.add_argument("--league", action="store_true", help="Run league tournament")
    parser.add_argument("--league-games", type=int, default=50)
    args = parser.parse_args()

    if args.league:
        league_tournament(
            checkpoint_dir=args.save_dir,
            num_games=args.league_games,
            num_players=args.players,
        )
    else:
        train(
            iterations=args.iterations,
            num_players=args.players,
            num_workers=args.workers,
            reward_mode=args.reward,
            save_dir=args.save_dir,
            learning_rate=args.lr,
            separate_policies=not args.shared,
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
