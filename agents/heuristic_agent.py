"""Heuristic agent: improved baseline that prioritizes launches and assigns.

Key insight from analysis: in this game, most actions have negative expected value
when played randomly. The best heuristic is to be strategic about what matters
(launching products for VP) and random about everything else. This beats pure
random by consistently converting dev products into VP while random often misses
launch opportunities.
"""

from __future__ import annotations

import random as _random

import numpy as np

from startup_simulator.types import ActionType, CubeType
from startup_simulator.actions.base import Action


class HeuristicAgent:
    """Prioritizes launches and assigns, random for everything else."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = _random.Random(seed)

    def act(self, observation: dict[str, np.ndarray], legal_actions: list | None = None) -> int:
        mask = observation["action_mask"]
        legal_indices = np.nonzero(mask)[0]
        if len(legal_indices) == 0:
            return 0
        if legal_actions is None:
            return int(self.rng.choice(legal_indices))

        buckets: dict[ActionType, list[tuple[int, Action]]] = {}
        for idx in legal_indices:
            if idx < len(legal_actions):
                a = legal_actions[idx]
                buckets.setdefault(a.action_type, []).append((int(idx), a))

        # 1. Forced micro-decisions
        c = self._forced(buckets)
        if c is not None:
            return c

        # 2. ALWAYS launch if possible (this is the #1 value action in the game)
        if ActionType.LAUNCH in buckets:
            return self._pick(buckets[ActionType.LAUNCH])

        # 3. ALWAYS assign talent in batch (progress toward launch)
        if ActionType.ASSIGN_ONE in buckets:
            return self._pick(buckets[ActionType.ASSIGN_ONE])

        # 4. ALWAYS integrate if possible (free +5 VP)
        if ActionType.INTEGRATE in buckets:
            return self._pick(buckets[ActionType.INTEGRATE])

        # 5. Everything else: random (including PASS)
        # This naturally balances spending vs saving because PASS is in the
        # legal action list alongside other options
        return int(self.rng.choice(legal_indices))

    def _pick(self, items: list[tuple[int, Action]]) -> int:
        return self.rng.choice(items)[0]

    def _forced(self, b: dict) -> int | None:
        if ActionType.END_ASSIGN_BATCH in b:
            if ActionType.ASSIGN_ONE not in b:
                return b[ActionType.END_ASSIGN_BATCH][0][0]
        for at in (ActionType.CHOOSE_MODE, ActionType.CHOOSE_XP,
                   ActionType.DISCARD_TALENT, ActionType.DISCARD_BACKLOG,
                   ActionType.DISCARD_STRATEGY, ActionType.CHOOSE_OFFLINE,
                   ActionType.FIRE_STAFF):
            if at in b:
                return self._pick(b[at])
        if ActionType.SETTLE in b:
            return b[ActionType.SETTLE][0][0]
        if ActionType.FOLD in b and ActionType.SETTLE not in b:
            return b[ActionType.FOLD][0][0]
        if ActionType.PASS_AUDIT in b:
            return b[ActionType.PASS_AUDIT][0][0]
        return None


def play_heuristic_game(
    num_players: int = 2,
    seed: int = 42,
    agent_seed: int = 100,
    verbose: bool = False,
) -> dict:
    from startup_simulator.engine import GameEngine
    engine = GameEngine(num_players=num_players, seed=seed)
    agents = [HeuristicAgent(seed=agent_seed + i) for i in range(num_players)]
    step_count = 0
    while not engine.is_done() and step_count < 10000:
        pid = engine.get_current_agent()
        legal = engine.get_legal_actions()
        mask = np.zeros(512, dtype=np.int8)
        mask[:len(legal)] = 1
        obs = {"observation": np.zeros(1), "action_mask": mask}
        action_idx = agents[pid].act(obs, legal_actions=legal)
        action = legal[min(action_idx, len(legal) - 1)]
        engine.step(action)
        step_count += 1
    scores = engine.get_scores() if engine.is_done() else []
    result = {"scores": scores, "steps": step_count, "done": engine.is_done()}
    if verbose and scores:
        print(f"Heuristic game (seed={seed}): {step_count} steps, scores={scores}")
    return result


if __name__ == "__main__":
    play_heuristic_game(num_players=2, seed=42, verbose=True)
