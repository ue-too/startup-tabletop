"""RoundModifiers: per-round state modifiers from active event cards."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoundModifiers:
    """Modifiers active for the current round, parsed from the event card."""
    # Economic
    equity_sale_bonus: int = 0          # Extra cash on equity sales
    revenue_bonus: int = 0              # +$ per active maintenance product
    license_fee: int = 3                # Default $3, Patent Trolls sets to $5
    revenue_decay_per_bug: int = 1      # Default $1, Legacy Code sets to $2

    # Production
    software_output_bonus: int = 0      # +N to all software output
    hardware_output_bonus: int = 0      # +N to all hardware output
    hardware_hiring_extra: int = 0      # +$ on hardware hiring
    crunch_time: bool = False           # +1 cube but +1 bug per dev team
    software_ops_penalty: int = 0       # -N bandwidth for software ops staff
    hardware_output_penalty: int = 0    # -N to all hardware output

    # HR
    poach_cost_multiplier: float | None = None  # Override poach multiplier (1x for frenzy)
    university_cost: int | None = None  # Override university cost (0 for intern season)
    poaching_suspended: bool = False    # No poaching this round
    training_cost: int | None = None    # Override training cost ($1 for course boom)

    # Conflict
    audit_reward_multiplier: int = 1    # 2x for whistleblower protection
    pr_cost_multiplier: float = 1.0     # 0.5 for trade show


def parse_event_modifiers(effect_id: str) -> RoundModifiers:
    """Create RoundModifiers from an event effect_id."""
    m = RoundModifiers()

    if effect_id == "equity_bonus_3":
        m.equity_sale_bonus = 3
    elif effect_id == "revenue_plus_1":
        m.revenue_bonus = 1
    elif effect_id == "license_fee_5":
        m.license_fee = 5
    elif effect_id == "tier1_only_bonus_3":
        pass  # Handled as immediate effect, not a modifier
    elif effect_id == "decay_2_per_bug":
        m.revenue_decay_per_bug = 2
    elif effect_id == "payroll_tax":
        pass  # Handled as immediate effect
    elif effect_id == "software_output_plus_1":
        m.software_output_bonus = 1
    elif effect_id == "hardware_output_plus_1":
        m.hardware_output_bonus = 1
    elif effect_id == "hardware_hiring_plus_3":
        m.hardware_hiring_extra = 3
    elif effect_id == "crunch_time":
        m.crunch_time = True
    elif effect_id == "software_ops_minus_1":
        m.software_ops_penalty = 1
    elif effect_id == "hardware_output_minus_1":
        m.hardware_output_penalty = 1
    elif effect_id == "poach_cost_1x":
        m.poach_cost_multiplier = 1.0
    elif effect_id == "university_free":
        m.university_cost = 0
    elif effect_id == "poaching_suspended":
        m.poaching_suspended = True
    elif effect_id == "training_cost_1":
        m.training_cost = 1
    elif effect_id == "audit_reward_doubled":
        m.audit_reward_multiplier = 2
    elif effect_id == "pr_half_price":
        m.pr_cost_multiplier = 0.5

    return m
