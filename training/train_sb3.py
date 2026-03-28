"""SB3 MaskablePPO self-play training script.

Usage:
    pip install -e ".[rl]"
    python training/train_sb3.py
    python training/train_sb3.py --timesteps 500000 --players 2
    python training/train_sb3.py --eval  # Evaluate saved model vs heuristic
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("WARNING: sb3-contrib not installed. Run: pip install -e '.[rl]'")

from training.selfplay_env import SelfPlayEnv
from training.callbacks import SelfPlayCallback


def make_env(num_players: int = 2, seed: int = 0, reward_mode: str = "shaped"):
    """Factory for creating a self-play environment."""
    def _init():
        env = SelfPlayEnv(
            num_players=num_players,
            seed=seed,
            reward_mode=reward_mode,
            max_steps=5000,
        )
        return env
    return _init


def mask_fn(env: SelfPlayEnv) -> np.ndarray:
    """Extract action mask from the environment."""
    return env.action_masks()


def train(
    timesteps: int = 200_000,
    num_players: int = 2,
    num_envs: int = 4,
    reward_mode: str = "shaped",
    save_dir: str = "checkpoints/sb3",
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    n_steps: int = 1024,
    n_epochs: int = 4,
    gamma: float = 0.99,
    verbose: int = 1,
) -> str:
    """Train a MaskablePPO agent via self-play.

    Returns path to the saved model.
    """
    if not HAS_SB3:
        raise RuntimeError("sb3-contrib required. Run: pip install -e '.[rl]'")

    os.makedirs(save_dir, exist_ok=True)

    # Create vectorized environments
    env_fns = [make_env(num_players, seed=i * 1000, reward_mode=reward_mode) for i in range(num_envs)]

    if num_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Wrap with action masking
    # Note: MaskablePPO handles masking internally via the env's action_masks() method
    # We need to use ActionMasker wrapper for single envs or handle via SubprocVecEnv

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
    )

    callback = SelfPlayCallback(
        save_dir=save_dir,
        save_freq=max(timesteps // 10, 1000),
        eval_freq=max(timesteps // 20, 500),
        eval_games=20,
        verbose=verbose,
    )

    print(f"Training MaskablePPO for {timesteps} timesteps...")
    print(f"  Players: {num_players}, Envs: {num_envs}")
    print(f"  Reward: {reward_mode}, LR: {learning_rate}")
    print(f"  Save dir: {save_dir}")

    model.learn(total_timesteps=timesteps, callback=callback)

    final_path = os.path.join(save_dir, "model_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")

    vec_env.close()
    return final_path


def evaluate_model(model_path: str, num_games: int = 50) -> None:
    """Evaluate a trained model against heuristic and random baselines."""
    from training.evaluate import evaluate_agents, make_heuristic_agent, make_random_agent, make_sb3_agent

    trained = make_sb3_agent(model_path)
    heuristic = make_heuristic_agent(seed=42)
    random_agent = make_random_agent(seed=42)

    print(f"\nEvaluating {model_path} over {num_games} games each:")

    print("\nTrained vs Heuristic:")
    result = evaluate_agents([trained, heuristic], num_games=num_games)
    print(f"  Win rates: P0(trained)={result['win_rates'][0]:.2f} P1(heuristic)={result['win_rates'][1]:.2f}")
    print(f"  Avg scores: {[f'{s:.1f}' for s in result['avg_scores']]}")

    print("\nTrained vs Random:")
    result = evaluate_agents([trained, random_agent], num_games=num_games)
    print(f"  Win rates: P0(trained)={result['win_rates'][0]:.2f} P1(random)={result['win_rates'][1]:.2f}")
    print(f"  Avg scores: {[f'{s:.1f}' for s in result['avg_scores']]}")

    print("\nHeuristic vs Random (baseline):")
    result = evaluate_agents([heuristic, random_agent], num_games=num_games)
    print(f"  Win rates: P0(heuristic)={result['win_rates'][0]:.2f} P1(random)={result['win_rates'][1]:.2f}")
    print(f"  Avg scores: {[f'{s:.1f}' for s in result['avg_scores']]}")


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO via self-play")
    parser.add_argument("--timesteps", "-t", type=int, default=200_000)
    parser.add_argument("--players", "-p", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--envs", "-e", type=int, default=4)
    parser.add_argument("--reward", type=str, default="shaped", choices=["sparse", "shaped"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints/sb3")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model instead of training")
    parser.add_argument("--model", type=str, default="checkpoints/sb3/model_final",
                        help="Model path for --eval")
    args = parser.parse_args()

    if args.eval:
        evaluate_model(args.model)
    else:
        train(
            timesteps=args.timesteps,
            num_players=args.players,
            num_envs=args.envs,
            reward_mode=args.reward,
            save_dir=args.save_dir,
            learning_rate=args.lr,
        )


if __name__ == "__main__":
    main()
