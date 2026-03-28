"""Frozen-pool self-play training.

Trains against a pool of past checkpoints to prevent self-play collapse.
Every N steps, saves the current model as a new opponent in the pool.

Usage:
    python training/train_pool.py
    python training/train_pool.py --timesteps 2000000 --pool-interval 200000
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import warnings

warnings.filterwarnings("ignore")

import torch
torch.distributions.Distribution.set_default_validate_args(False)

import numpy as np
from sb3_contrib import MaskablePPO
from training.frozen_pool_env import FrozenPoolEnv
from training.callbacks import SelfPlayCallback


def train(
    timesteps: int = 2_000_000,
    pool_interval: int = 200_000,
    pool_dir: str = "checkpoints/pool",
    save_dir: str = "checkpoints/sb3_pool",
    seed_model: str | None = None,
    learning_rate: float = 3e-4,
    threads: int = 4,
    verbose: int = 1,
) -> str:
    torch.set_num_threads(threads)
    os.makedirs(pool_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Seed the pool with existing checkpoints or heuristic baseline
    if not any(f.endswith(".zip") for f in os.listdir(pool_dir)):
        print("Pool empty. Seeding with initial random model...")
        env_tmp = FrozenPoolEnv(seed=0, pool_dir=pool_dir)
        init_model = MaskablePPO("MultiInputPolicy", env_tmp, verbose=0)
        init_model.save(os.path.join(pool_dir, "pool_000_random.zip"))
        del init_model, env_tmp

    env = FrozenPoolEnv(seed=42, reward_mode="shaped", pool_dir=pool_dir)

    if seed_model and os.path.exists(seed_model):
        print(f"Loading seed model: {seed_model}")
        model = MaskablePPO.load(seed_model, env=env)
        # Also add seed model to pool
        shutil.copy(seed_model, os.path.join(pool_dir, "pool_001_seed.zip"))
    else:
        model = MaskablePPO(
            "MultiInputPolicy", env,
            n_steps=512, batch_size=128, n_epochs=4,
            learning_rate=learning_rate, gamma=0.99,
            ent_coef=0.01, max_grad_norm=0.5, verbose=0,
        )

    callback = SelfPlayCallback(
        save_dir=save_dir, save_freq=pool_interval,
        eval_freq=pool_interval // 2, eval_games=30, verbose=verbose,
    )

    print(f"Frozen-pool training: {timesteps:,} steps")
    print(f"  Pool dir: {pool_dir} ({len(os.listdir(pool_dir))} models)")
    print(f"  Pool update interval: {pool_interval:,} steps")
    print(f"  Threads: {threads}")

    start = time.time()
    trained_so_far = 0
    pool_counter = 2  # 0=random, 1=seed

    while trained_so_far < timesteps:
        chunk = min(pool_interval, timesteps - trained_so_far)
        model.learn(total_timesteps=chunk, callback=callback, reset_num_timesteps=False)
        trained_so_far += chunk

        # Save current model to pool
        pool_path = os.path.join(pool_dir, f"pool_{pool_counter:03d}_{trained_so_far // 1000}k.zip")
        model.save(pool_path)
        pool_counter += 1

        elapsed = time.time() - start
        pool_size = len([f for f in os.listdir(pool_dir) if f.endswith(".zip")])
        print(f"  [{trained_so_far // 1000}k/{timesteps // 1000}k] "
              f"Pool size: {pool_size}, Elapsed: {elapsed:.0f}s")

    model.save(os.path.join(save_dir, "model_final"))
    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s ({timesteps / elapsed:.0f} steps/sec)")
    return os.path.join(save_dir, "model_final")


def main():
    parser = argparse.ArgumentParser(description="Frozen-pool self-play training")
    parser.add_argument("--timesteps", "-t", type=int, default=2_000_000)
    parser.add_argument("--pool-interval", type=int, default=200_000)
    parser.add_argument("--pool-dir", type=str, default="checkpoints/pool")
    parser.add_argument("--save-dir", type=str, default="checkpoints/sb3_pool")
    parser.add_argument("--seed-model", type=str, default=None,
                        help="Path to model to seed training from")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        pool_interval=args.pool_interval,
        pool_dir=args.pool_dir,
        save_dir=args.save_dir,
        seed_model=args.seed_model,
        learning_rate=args.lr,
        threads=args.threads,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
