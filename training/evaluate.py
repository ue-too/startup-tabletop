"""Evaluate trained models against baselines."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from startup_simulator.engine import GameEngine
from env.action_space import MAX_ACTIONS
from env.observation_space import OBS_SIZE, encode_observation


def evaluate_agents(
    agent_fns: list[Callable],
    num_games: int = 100,
    num_players: int = 2,
    seed_start: int = 0,
    verbose: bool = False,
) -> dict:
    """Evaluate agents against each other over many games.

    Args:
        agent_fns: list of callables, one per player.
            Each takes (obs_dict, legal_actions) -> action_index.
        num_games: number of games to play.
        num_players: players per game.
        seed_start: starting seed.
        verbose: print per-game results.

    Returns:
        dict with win_rates, avg_scores, etc.
    """
    wins = [0] * num_players
    total_scores = [0.0] * num_players
    completed = 0

    for game_idx in range(num_games):
        seed = seed_start + game_idx
        engine = GameEngine(num_players=num_players, seed=seed)
        steps = 0

        while not engine.is_done() and steps < 5000:
            pid = engine.get_current_agent()
            legal = engine.get_legal_actions()

            obs = encode_observation(engine.state, pid)
            mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
            mask[:len(legal)] = 1
            obs_dict = {"observation": obs, "action_mask": mask}

            action_idx = agent_fns[pid % len(agent_fns)](obs_dict, legal)
            action = legal[min(action_idx, len(legal) - 1)]
            engine.step(action)
            steps += 1

        if engine.is_done():
            scores = engine.get_scores()
            completed += 1
            winner = scores.index(max(scores))
            wins[winner] += 1
            for i, s in enumerate(scores):
                total_scores[i] += s
            if verbose:
                print(f"Game {game_idx}: scores={scores} winner=P{winner}")

    win_rates = [w / max(completed, 1) for w in wins]
    avg_scores = [s / max(completed, 1) for s in total_scores]

    return {
        "games_played": num_games,
        "games_completed": completed,
        "wins": wins,
        "win_rates": win_rates,
        "avg_scores": avg_scores,
    }


def make_random_agent(seed: int = 42) -> Callable:
    """Create a random agent function."""
    rng = random.Random(seed)

    def agent_fn(obs_dict, legal_actions):
        mask = obs_dict["action_mask"]
        legal = np.nonzero(mask)[0]
        return int(rng.choice(legal)) if len(legal) > 0 else 0

    return agent_fn


def make_heuristic_agent(seed: int = 42) -> Callable:
    """Create a heuristic agent function."""
    from agents.heuristic_agent import HeuristicAgent
    agent = HeuristicAgent(seed=seed)

    def agent_fn(obs_dict, legal_actions):
        return agent.act(obs_dict, legal_actions=legal_actions)

    return agent_fn


def make_sb3_agent(model_path: str) -> Callable:
    """Create an agent from a saved SB3 MaskablePPO model."""
    import torch
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path)

    def agent_fn(obs_dict, legal_actions):
        try:
            action, _ = model.predict(obs_dict, action_masks=obs_dict["action_mask"])
            return int(action)
        except ValueError:
            # Numerical issue with action masking - fall back to random legal action
            mask = obs_dict["action_mask"]
            legal = np.nonzero(mask)[0]
            return int(np.random.choice(legal)) if len(legal) > 0 else 0

    return agent_fn


if __name__ == "__main__":
    print("Random vs Random:")
    result = evaluate_agents(
        [make_random_agent(1), make_random_agent(2)],
        num_games=50,
    )
    print(f"  Win rates: {result['win_rates']}, Avg scores: {[f'{s:.1f}' for s in result['avg_scores']]}")

    print("\nHeuristic vs Heuristic:")
    result = evaluate_agents(
        [make_heuristic_agent(1), make_heuristic_agent(2)],
        num_games=50,
    )
    print(f"  Win rates: {result['win_rates']}, Avg scores: {[f'{s:.1f}' for s in result['avg_scores']]}")

    print("\nHeuristic vs Random:")
    result = evaluate_agents(
        [make_heuristic_agent(1), make_random_agent(2)],
        num_games=50,
    )
    print(f"  Win rates: {result['win_rates']}, Avg scores: {[f'{s:.1f}' for s in result['avg_scores']]}")
