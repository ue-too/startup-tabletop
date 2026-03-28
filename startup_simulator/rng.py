"""Seeded random number generator for deterministic gameplay."""

from __future__ import annotations

import numpy as np


class GameRng:
    """Wrapper around numpy RandomState for deterministic game logic."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def shuffle(self, lst: list) -> None:
        """Shuffle a list in-place."""
        self._rng.shuffle(lst)

    def randint(self, low: int, high: int) -> int:
        """Return random int in [low, high)."""
        return int(self._rng.randint(low, high))

    def choice(self, lst: list):
        """Pick a random element from a list."""
        idx = self.randint(0, len(lst))
        return lst[idx]

    def get_state(self) -> dict:
        """Get RNG state for serialization."""
        return self._rng.get_state()

    def set_state(self, state: dict) -> None:
        """Restore RNG state from serialization."""
        self._rng.set_state(state)
