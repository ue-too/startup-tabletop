"""Round-robin tournament training for 4 AI agents.

Each round:
  1. Train Agent A against frozen B, C, D
  2. Train Agent B against frozen A, C, D
  3. Train Agent C against frozen A, B, D
  4. Train Agent D against frozen A, B, C
  5. Evaluate all agents against each other
  6. Save progress

Usage:
    python training/train_tournament.py
    python training/train_tournament.py --rounds 20 --steps-per-round 200000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import torch
torch.distributions.Distribution.set_default_validate_args(False)

import numpy as np
from sb3_contrib import MaskablePPO

from training.tournament_env import TournamentEnv
from training.evaluate import evaluate_agents, make_random_agent, make_heuristic_agent


NUM_AGENTS = 4


def get_agent_dir(base_dir: str, agent_id: int) -> str:
    return os.path.join(base_dir, f"agent_{agent_id}")


def get_latest_model(agent_dir: str) -> str | None:
    """Get the latest model checkpoint from an agent's directory."""
    if not os.path.exists(agent_dir):
        return None
    zips = sorted([f for f in os.listdir(agent_dir) if f.endswith(".zip")])
    if not zips:
        return None
    return os.path.join(agent_dir, zips[-1])


def make_sb3_agent_fn(model_path: str):
    """Create an agent function from a saved model."""
    model = MaskablePPO.load(model_path)

    def agent_fn(obs_dict, legal_actions):
        try:
            action, _ = model.predict(obs_dict, action_masks=obs_dict["action_mask"])
            return int(action)
        except (ValueError, Exception):
            mask = obs_dict["action_mask"]
            legal = np.nonzero(mask)[0]
            return int(np.random.choice(legal)) if len(legal) > 0 else 0

    return agent_fn


def train_one_agent(
    agent_id: int,
    base_dir: str,
    steps: int,
    round_num: int,
    learning_rate: float = 3e-4,
    threads: int = 4,
) -> str:
    """Train one agent against the other 3 frozen agents."""
    torch.set_num_threads(threads)

    agent_dir = get_agent_dir(base_dir, agent_id)
    os.makedirs(agent_dir, exist_ok=True)

    # Collect opponent directories (all other agents)
    opponent_dirs = []
    for oid in range(NUM_AGENTS):
        if oid == agent_id:
            continue
        odir = get_agent_dir(base_dir, oid)
        opponent_dirs.append(odir)

    # Create environment
    env = TournamentEnv(
        learner_id=agent_id,
        num_players=NUM_AGENTS,
        seed=42 + round_num * 1000 + agent_id,
        reward_mode="shaped",
        opponent_dirs=opponent_dirs,
    )

    # Load existing model or create new
    latest = get_latest_model(agent_dir)
    if latest:
        model = MaskablePPO.load(latest, env=env)
    else:
        model = MaskablePPO(
            "MultiInputPolicy", env,
            n_steps=512, batch_size=128, n_epochs=4,
            learning_rate=learning_rate, gamma=0.99,
            ent_coef=0.01, max_grad_norm=0.5, verbose=0,
        )

    # Train
    model.learn(total_timesteps=steps, reset_num_timesteps=False)

    # Save checkpoint
    save_path = os.path.join(agent_dir, f"model_r{round_num:03d}.zip")
    model.save(save_path)

    del model
    return save_path


def evaluate_tournament(base_dir: str, num_games: int = 50) -> dict:
    """Run a full round-robin evaluation between all agents + baselines."""
    agents = {}
    for aid in range(NUM_AGENTS):
        latest = get_latest_model(get_agent_dir(base_dir, aid))
        if latest:
            agents[f"Agent_{aid}"] = make_sb3_agent_fn(latest)
        else:
            agents[f"Agent_{aid}"] = make_random_agent(aid)

    random_ag = make_random_agent(99)
    heuristic = make_heuristic_agent(99)

    results = {}

    # Each agent vs random
    for name, agent in agents.items():
        r1 = evaluate_agents([agent, random_ag], num_games=num_games)
        r2 = evaluate_agents([random_ag, agent], num_games=num_games)
        wr = (r1["wins"][0] + r2["wins"][1]) / (num_games * 2)
        avg = (r1["avg_scores"][0] + r2["avg_scores"][1]) / 2
        results[f"{name}_vs_random"] = {"win_rate": wr, "avg_score": avg}

    # Each agent vs heuristic
    for name, agent in agents.items():
        r1 = evaluate_agents([agent, heuristic], num_games=num_games)
        r2 = evaluate_agents([heuristic, agent], num_games=num_games)
        wr = (r1["wins"][0] + r2["wins"][1]) / (num_games * 2)
        results[f"{name}_vs_heuristic"] = {"win_rate": wr}

    # Head-to-head between agents (pairwise)
    agent_names = list(agents.keys())
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            a, b = agent_names[i], agent_names[j]
            r = evaluate_agents([agents[a], agents[b]], num_games=num_games)
            results[f"{a}_vs_{b}"] = {
                "a_win_rate": r["win_rates"][0],
                "b_win_rate": r["win_rates"][1],
            }

    return results


def train_tournament(
    rounds: int = 10,
    steps_per_round: int = 200_000,
    base_dir: str = "checkpoints/tournament",
    learning_rate: float = 3e-4,
    threads: int = 4,
    eval_games: int = 50,
    verbose: bool = True,
) -> dict:
    """Run the full round-robin tournament training."""
    os.makedirs(base_dir, exist_ok=True)

    # Progress tracking
    history = {
        "rounds": [],
        "config": {
            "num_agents": NUM_AGENTS,
            "rounds": rounds,
            "steps_per_round": steps_per_round,
        },
    }
    history_path = os.path.join(base_dir, "training_history.json")

    # Load existing history if resuming
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        start_round = len(history["rounds"])
        if verbose:
            print(f"Resuming from round {start_round + 1}")
    else:
        start_round = 0

    total_start = time.time()

    for round_num in range(start_round, rounds):
        round_start = time.time()
        if verbose:
            print(f"\n{'='*60}")
            print(f"ROUND {round_num + 1}/{rounds}")
            print(f"{'='*60}")

        # Train each agent
        for aid in range(NUM_AGENTS):
            t0 = time.time()
            save_path = train_one_agent(
                agent_id=aid,
                base_dir=base_dir,
                steps=steps_per_round,
                round_num=round_num,
                learning_rate=learning_rate,
                threads=threads,
            )
            dt = time.time() - t0
            if verbose:
                print(f"  Agent {aid}: trained {steps_per_round//1000}k steps in {dt:.0f}s -> {save_path}")

        # Evaluate
        if verbose:
            print(f"  Evaluating ({eval_games} games per matchup)...")
        eval_results = evaluate_tournament(base_dir, num_games=eval_games)

        # Print summary
        round_data = {
            "round": round_num + 1,
            "elapsed_s": time.time() - round_start,
            "eval": eval_results,
        }
        history["rounds"].append(round_data)

        if verbose:
            for aid in range(NUM_AGENTS):
                vs_rand = eval_results.get(f"Agent_{aid}_vs_random", {})
                vs_heur = eval_results.get(f"Agent_{aid}_vs_heuristic", {})
                wr = vs_rand.get("win_rate", 0)
                avg = vs_rand.get("avg_score", 0)
                wrh = vs_heur.get("win_rate", 0)
                print(f"  Agent {aid}: vs_random={wr:.0%} (avg {avg:.1f}), vs_heuristic={wrh:.0%}")

            # Head-to-head summary
            print("  Head-to-head:")
            for i in range(NUM_AGENTS):
                for j in range(i + 1, NUM_AGENTS):
                    h2h = eval_results.get(f"Agent_{i}_vs_Agent_{j}", {})
                    a_wr = h2h.get("a_win_rate", 0.5)
                    print(f"    Agent {i} vs Agent {j}: {a_wr:.0%} / {1 - a_wr:.0%}")

        # Save history
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        total_elapsed = time.time() - total_start
        if verbose:
            print(f"  Round time: {round_data['elapsed_s']:.0f}s, Total: {total_elapsed:.0f}s")

    if verbose:
        print(f"\nTournament complete! {rounds} rounds in {time.time() - total_start:.0f}s")
        print(f"History saved to {history_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Round-robin tournament training")
    parser.add_argument("--rounds", "-r", type=int, default=10)
    parser.add_argument("--steps-per-round", "-s", type=int, default=200_000)
    parser.add_argument("--base-dir", type=str, default="checkpoints/tournament")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=50)
    args = parser.parse_args()

    train_tournament(
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        base_dir=args.base_dir,
        learning_rate=args.lr,
        threads=args.threads,
        eval_games=args.eval_games,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
