"""Base action types for Startup Simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import ActionType


@dataclass(frozen=True)
class Action:
    """An action a player can take.

    Fields are overloaded depending on action_type.
    Unused fields default to -1 or empty.
    """
    action_type: ActionType

    # Common parameters
    source_index: int = -1          # Index into a market row, deck choice, etc.
    target_player: int = -1         # Target opponent player_id
    target_instance: int = -1       # Target card instance_id
    target_instances: tuple[int, ...] = ()  # For batch operations
    amount: int = -1                # Bid amount, payment, etc.
    choice: int = -1                # XP color choice, mode choice, etc.
    source_type: str = ""           # "university_sw", "university_hw", "agency", "open_market", etc.

    def __repr__(self) -> str:
        parts = [f"Action({self.action_type.name}"]
        if self.source_index >= 0:
            parts.append(f"src={self.source_index}")
        if self.target_player >= 0:
            parts.append(f"tgt_p={self.target_player}")
        if self.target_instance >= 0:
            parts.append(f"tgt={self.target_instance}")
        if self.target_instances:
            parts.append(f"tgts={self.target_instances}")
        if self.amount >= 0:
            parts.append(f"amt={self.amount}")
        if self.choice >= 0:
            parts.append(f"ch={self.choice}")
        if self.source_type:
            parts.append(f"src_type={self.source_type}")
        return ", ".join(parts) + ")"


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    message: str = ""
    game_over: bool = False
