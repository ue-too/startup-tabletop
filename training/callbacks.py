"""Training callbacks for SB3: checkpointing, evaluation, logging."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    BaseCallback = object  # type: ignore


class SelfPlayCallback(BaseCallback if HAS_SB3 else object):
    """Callback that logs self-play training metrics.

    Tracks: win rates, average scores, game lengths, episode rewards.
    Saves checkpoints periodically.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_freq: int = 10000,
        eval_freq: int = 5000,
        eval_games: int = 20,
        verbose: int = 1,
    ) -> None:
        if HAS_SB3:
            super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_games = eval_games
        self._episode_scores: list[list[int]] = []
        self._episode_lengths: list[int] = []

        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Check for episode completion via info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "scores" in info:
                self._episode_scores.append(info["scores"])
            if "final_turn" in info:
                self._episode_lengths.append(info["final_turn"])

        # Periodic checkpoint
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"model_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"[Checkpoint] Saved to {path}")

        # Periodic logging
        if self.num_timesteps % self.eval_freq == 0 and self._episode_scores:
            recent = self._episode_scores[-self.eval_games:]
            if recent:
                p0_scores = [s[0] for s in recent]
                p1_scores = [s[1] for s in recent if len(s) > 1]
                p0_wins = sum(1 for s in recent if s[0] == max(s))
                avg_score = np.mean(p0_scores)
                avg_length = np.mean(self._episode_lengths[-self.eval_games:]) if self._episode_lengths else 0

                self.logger.record("selfplay/p0_avg_score", avg_score)
                self.logger.record("selfplay/p0_win_rate", p0_wins / len(recent))
                self.logger.record("selfplay/avg_game_length", avg_length)
                self.logger.record("selfplay/episodes", len(self._episode_scores))

                if self.verbose:
                    print(
                        f"[Eval @{self.num_timesteps}] "
                        f"P0 avg score: {avg_score:.1f}, "
                        f"win rate: {p0_wins / len(recent):.2f}, "
                        f"avg turns: {avg_length:.0f}, "
                        f"episodes: {len(self._episode_scores)}"
                    )

        return True

    def _on_training_end(self) -> None:
        # Save final model
        path = os.path.join(self.save_dir, "model_final")
        self.model.save(path)
        if self.verbose:
            print(f"[Final] Saved to {path}")
