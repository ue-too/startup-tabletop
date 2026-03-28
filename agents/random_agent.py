"""Random agent: selects uniformly among legal actions."""

from __future__ import annotations

import random as _random

import numpy as np

from env.startup_env import StartupEnv


class RandomAgent:
    """Uniform random legal-action agent."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = _random.Random(seed)

    def act(self, observation: dict[str, np.ndarray]) -> int:
        """Select a random legal action."""
        mask = observation["action_mask"]
        legal_indices = np.nonzero(mask)[0]
        if len(legal_indices) == 0:
            return 0
        return int(self.rng.choice(legal_indices))


def play_random_game(
    num_players: int = 2,
    seed: int = 42,
    agent_seed: int = 100,
    max_steps: int = 10000,
    verbose: bool = False,
) -> dict:
    """Play a full game with random agents. Returns result dict."""
    env = StartupEnv(num_players=num_players, seed=seed, max_steps=max_steps)
    env.reset()

    agents = {name: RandomAgent(seed=agent_seed + i) for i, name in enumerate(env.possible_agents)}
    step_count = 0

    for agent_name in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue

        action = agents[agent_name].act(obs)
        env.step(action)
        step_count += 1

    scores = env.engine.get_scores() if env.engine and env.engine.is_done() else []
    result = {
        "scores": scores,
        "steps": step_count,
        "done": env.is_done,
        "seed": seed,
    }

    if verbose and scores:
        print(f"Game (seed={seed}): {step_count} steps, scores={scores}")

    return result


if __name__ == "__main__":
    play_random_game(num_players=2, seed=42, verbose=True)
    play_random_game(num_players=4, seed=99, verbose=True)
