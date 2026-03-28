"""CLI entry point for game replay viewer.

Usage:
    python -m startup_simulator.replay
    python -m startup_simulator.replay --players 4 --seed 42
    python -m startup_simulator.replay --agent heuristic --seed 123
    python -m startup_simulator.replay --save replay.json
"""

from __future__ import annotations

import argparse
import sys

from .recorder import record_heuristic_game, record_random_game
from .viewer import view_replay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Startup Simulator - Game Replay Viewer",
    )
    parser.add_argument(
        "--players", "-p", type=int, default=2, choices=[2, 3, 4],
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Game seed (default: 42)",
    )
    parser.add_argument(
        "--agent", "-a", type=str, default="heuristic", choices=["random", "heuristic"],
        help="Agent type (default: heuristic)",
    )
    parser.add_argument(
        "--agent-seed", type=int, default=100,
        help="Agent random seed (default: 100)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Max steps before truncation (default: 5000)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save replay to JSON file instead of viewing",
    )

    args = parser.parse_args()

    print(f"Recording {args.players}p game (seed={args.seed}, agent={args.agent})...")

    if args.agent == "heuristic":
        recorder = record_heuristic_game(
            num_players=args.players,
            seed=args.seed,
            agent_seed=args.agent_seed,
            max_steps=args.max_steps,
        )
    else:
        recorder = record_random_game(
            num_players=args.players,
            seed=args.seed,
            agent_seed=args.agent_seed,
            max_steps=args.max_steps,
        )

    frames = len(recorder.frames)
    done = recorder.is_done()
    scores = recorder.engine.get_scores() if done else []
    print(f"Recorded {frames} frames. Done: {done}. Scores: {scores}")

    if args.save:
        with open(args.save, "w") as f:
            f.write(recorder.to_json())
        print(f"Saved to {args.save}")
    else:
        print("Starting replay viewer... (press q to quit)")
        view_replay(recorder)


if __name__ == "__main__":
    main()
