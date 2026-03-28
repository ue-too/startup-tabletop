"""Immutable card definitions for Startup Simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .types import CubeType, Sector, Tag, TalentType, Tier, Trait


@dataclass(frozen=True)
class TalentCardDef:
    """Definition of a talent card (immutable template)."""
    card_def_id: str
    name: str
    talent_type: TalentType
    cost: int
    salary: int
    base_output: int
    output_type: CubeType | None  # None for specialists, FLEX handled via is_cross_functional
    is_junior: bool
    is_cross_functional: bool
    trait: Trait | None
    is_flex: bool = False  # True for Firmware/Full Stack (can choose SW or HW each turn)

    @property
    def is_specialist(self) -> bool:
        return self.talent_type in (
            TalentType.QA, TalentType.SALES, TalentType.HR,
            TalentType.PM, TalentType.SENIOR_PM, TalentType.GROWTH_HACKER,
        )

    @property
    def is_senior_dev(self) -> bool:
        return self.talent_type in (
            TalentType.SENIOR_BACKEND, TalentType.SENIOR_HARDWARE,
            TalentType.FIRMWARE, TalentType.FULL_STACK,
        )

    @property
    def is_pm(self) -> bool:
        return self.talent_type in (TalentType.PM, TalentType.SENIOR_PM)

    @property
    def can_produce_cubes(self) -> bool:
        return self.base_output > 0 or self.is_junior


@dataclass(frozen=True)
class ProductCardDef:
    """Definition of a product card (immutable template)."""
    card_def_id: str
    name: str
    tier: Tier
    sector: Sector | None  # None for Market Crash
    cost_software: int
    cost_hardware: int
    revenue: int
    valuation: int
    maint_software: int
    maint_hardware: int
    requires: tuple[Tag, ...]
    provides: Tag | None
    is_market_crash: bool = False
    is_expansion: bool = False

    @property
    def total_cost(self) -> int:
        return self.cost_software + self.cost_hardware

    @property
    def total_maint(self) -> int:
        return self.maint_software + self.maint_hardware

    @property
    def is_hybrid(self) -> bool:
        return self.cost_software > 0 and self.cost_hardware > 0

    @property
    def is_software_only(self) -> bool:
        return self.cost_software > 0 and self.cost_hardware == 0

    @property
    def is_hardware_only(self) -> bool:
        return self.cost_hardware > 0 and self.cost_software == 0


@dataclass(frozen=True)
class StrategyCardDef:
    """Definition of a strategy card."""
    card_def_id: str
    name: str
    category: str  # "training", "warfare", "attribute", "utility"
    cost: int
    effect_id: str
    count: int = 1
    description: str = ""


@dataclass(frozen=True)
class EventCardDef:
    """Definition of an event card."""
    card_def_id: str
    name: str
    category: str  # "economic", "production", "hr", "conflict"
    effect_id: str
    description: str = ""
