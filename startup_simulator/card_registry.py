"""Load card definitions from JSON data files."""

from __future__ import annotations

import json
from pathlib import Path

from .cards import EventCardDef, ProductCardDef, StrategyCardDef, TalentCardDef
from .types import (
    CUBE_TYPE_MAP,
    SECTOR_MAP,
    TAG_MAP,
    TALENT_TYPE_MAP,
    TRAIT_NAME_MAP,
    CubeType,
    Tier,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _parse_output_type(raw: str | None) -> tuple[CubeType | None, bool]:
    """Returns (output_type, is_flex)."""
    if raw is None:
        return None, False
    if raw == "FLEX":
        return None, True
    return CUBE_TYPE_MAP[raw], False


def load_talent_cards() -> list[TalentCardDef]:
    """Load all talent card definitions from talent.json."""
    with open(DATA_DIR / "talent.json") as f:
        data = json.load(f)

    cards: list[TalentCardDef] = []

    for category in ("juniors", "seniors", "specialists"):
        for entry in data.get(category, []):
            output_type, is_flex = _parse_output_type(entry.get("output_type"))
            trait = TRAIT_NAME_MAP.get(entry["trait"]) if entry.get("trait") else None
            count = entry.get("count", 1)

            card = TalentCardDef(
                card_def_id=entry["id"],
                name=entry["name"],
                talent_type=TALENT_TYPE_MAP[entry["talent_type"]],
                cost=entry["cost"],
                salary=entry["salary"],
                base_output=entry["base_output"],
                output_type=output_type,
                is_junior=entry["is_junior"],
                is_cross_functional=entry.get("is_cross_functional", False),
                trait=trait,
                is_flex=is_flex,
            )
            for _ in range(count):
                cards.append(card)

    return cards


def load_product_cards(filename: str) -> list[ProductCardDef]:
    """Load product card definitions from a JSON file."""
    with open(DATA_DIR / filename) as f:
        data = json.load(f)

    cards: list[ProductCardDef] = []
    for entry in data["products"]:
        sector = SECTOR_MAP.get(entry["sector"]) if entry.get("sector") else None
        requires = tuple(TAG_MAP[t] for t in entry.get("requires", []))
        provides = TAG_MAP.get(entry["provides"]) if entry.get("provides") else None

        card = ProductCardDef(
            card_def_id=entry["id"],
            name=entry["name"],
            tier=Tier(entry["tier"]),
            sector=sector,
            cost_software=entry["cost_software"],
            cost_hardware=entry["cost_hardware"],
            revenue=entry["revenue"],
            valuation=entry["valuation"],
            maint_software=entry["maint_software"],
            maint_hardware=entry["maint_hardware"],
            requires=requires,
            provides=provides,
            is_market_crash=entry.get("is_market_crash", False),
            is_expansion=entry.get("expansion", False),
        )
        cards.append(card)

    return cards


def load_strategy_cards() -> list[StrategyCardDef]:
    """Load strategy card definitions from strategy.json."""
    with open(DATA_DIR / "strategy.json") as f:
        data = json.load(f)

    cards: list[StrategyCardDef] = []
    for entry in data["cards"]:
        card = StrategyCardDef(
            card_def_id=entry["id"],
            name=entry["name"],
            category=entry["category"],
            cost=entry["cost"],
            effect_id=entry["effect_id"],
            count=entry.get("count", 1),
            description=entry.get("description", ""),
        )
        for _ in range(card.count):
            cards.append(card)
    return cards


def load_event_cards() -> list[EventCardDef]:
    """Load event card definitions from events.json."""
    with open(DATA_DIR / "events.json") as f:
        data = json.load(f)

    cards: list[EventCardDef] = []
    for entry in data["events"]:
        card = EventCardDef(
            card_def_id=entry["id"],
            name=entry["name"],
            category=entry["category"],
            effect_id=entry["effect_id"],
            description=entry.get("description", ""),
        )
        cards.append(card)
    return cards


def load_integration_rules() -> dict:
    """Load host/client compatibility rules."""
    with open(DATA_DIR / "integration_rules.json") as f:
        return json.load(f)


def load_seed_products() -> list[ProductCardDef]:
    return load_product_cards("products_seed.json")


def load_growth_products() -> list[ProductCardDef]:
    return load_product_cards("products_growth.json")


class CardRegistry:
    """Central registry for all card definitions. Singleton-like usage."""

    def __init__(self) -> None:
        self.talent_cards: list[TalentCardDef] = load_talent_cards()
        self.seed_products: list[ProductCardDef] = load_seed_products()
        self.growth_products: list[ProductCardDef] = load_growth_products()
        self.strategy_cards: list[StrategyCardDef] = load_strategy_cards()
        self.event_cards: list[EventCardDef] = load_event_cards()
        self.integration_rules: dict = load_integration_rules()

        # Build lookup maps
        self._talent_by_id: dict[str, TalentCardDef] = {}
        for c in self.talent_cards:
            self._talent_by_id[c.card_def_id] = c

        self._product_by_id: dict[str, ProductCardDef] = {}
        for c in self.seed_products + self.growth_products:
            self._product_by_id[c.card_def_id] = c

        self._strategy_by_id: dict[str, StrategyCardDef] = {}
        for c in self.strategy_cards:
            self._strategy_by_id[c.card_def_id] = c

        self._event_by_id: dict[str, EventCardDef] = {}
        for c in self.event_cards:
            self._event_by_id[c.card_def_id] = c

        # Separate junior templates
        self.junior_software = self._talent_by_id["jr_software"]
        self.junior_hardware = self._talent_by_id["jr_hardware"]

        # Agency deck = all non-junior talent cards
        self.agency_deck_defs: list[TalentCardDef] = [
            c for c in self.talent_cards if not c.is_junior
        ]

        # Strategy deck (card_def_ids, respecting count)
        self.strategy_deck_defs: list[StrategyCardDef] = list(self.strategy_cards)

    def get_talent(self, card_def_id: str) -> TalentCardDef:
        return self._talent_by_id[card_def_id]

    def get_product(self, card_def_id: str) -> ProductCardDef:
        return self._product_by_id[card_def_id]

    def get_seed_deck(self) -> list[ProductCardDef]:
        """Get all Tier 1 products (non-Market Crash)."""
        return [p for p in self.seed_products if not p.is_market_crash]

    def get_growth_deck(self) -> list[ProductCardDef]:
        """Get all Tier 2/3 products + Market Crash."""
        return list(self.growth_products)

    def get_event(self, card_def_id: str) -> EventCardDef:
        return self._event_by_id[card_def_id]

    def get_strategy(self, card_def_id: str) -> StrategyCardDef:
        return self._strategy_by_id[card_def_id]

    def get_market_crash(self) -> ProductCardDef:
        for p in self.growth_products:
            if p.is_market_crash:
                return p
        raise ValueError("Market Crash card not found")


# Module-level singleton
_registry: CardRegistry | None = None


def get_registry() -> CardRegistry:
    global _registry
    if _registry is None:
        _registry = CardRegistry()
    return _registry
