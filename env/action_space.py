"""Action encoding/decoding and action masks for RL."""

from __future__ import annotations

import numpy as np

from startup_simulator.actions.base import Action
from startup_simulator.engine import GameEngine

MAX_ACTIONS = 512


class ActionEncoder:
    """Maps between Action objects and integer indices.

    Each step, get_legal_actions() returns N legal Action objects.
    These are stored and assigned indices 0..N-1.
    The action mask is a binary vector of size MAX_ACTIONS.
    """

    def __init__(self) -> None:
        self._current_actions: list[Action] = []

    def update(self, legal_actions: list[Action]) -> None:
        """Update the current legal action list."""
        self._current_actions = legal_actions

    def encode_mask(self) -> np.ndarray:
        """Return binary action mask of size MAX_ACTIONS."""
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        n = min(len(self._current_actions), MAX_ACTIONS)
        mask[:n] = 1
        return mask

    def decode(self, action_index: int) -> Action:
        """Decode integer index to Action object."""
        if action_index < 0 or action_index >= len(self._current_actions):
            raise ValueError(
                f"Action index {action_index} out of range "
                f"(0-{len(self._current_actions) - 1})"
            )
        return self._current_actions[action_index]

    @property
    def num_legal(self) -> int:
        return len(self._current_actions)

    @property
    def actions(self) -> list[Action]:
        return self._current_actions
