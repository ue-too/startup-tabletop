"""Mutable game state for Startup Simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .cards import ProductCardDef, TalentCardDef
from .types import CubeType, Phase, SubPhase, Tag, TalentType, Tier, Zone


@dataclass
class TalentInstance:
    """A talent card instance in play (mutable)."""
    instance_id: int
    card_def_id: str
    owner: int  # player_id, -1 if in market
    zone: Zone
    assigned_product: int | None = None  # product instance_id

    # XP system
    xp_permanent: list[CubeType] = field(default_factory=list)  # max 4
    xp_pending: list[CubeType] = field(default_factory=list)    # max 3 (1 per color)
    skills: list[CubeType] = field(default_factory=list)         # trained skills
    rank_badges: int = 0  # 0 or 1 (gold)
    rank_pending: bool = False

    # Status
    onboarding: bool = False  # tapped/sideways, 0 output this turn
    declared_mode: CubeType | None = None  # for multi-skilled / flex units
    equity_vested: int | None = None  # player_id of vesting equity token

    # Attached strategy cards (attribute card_def_ids)
    attributes: list[str] = field(default_factory=list)

    @property
    def total_xp(self) -> int:
        return len(self.xp_permanent)

    @property
    def salary(self) -> int:
        """Dynamic salary based on XP count (for juniors)."""
        # This is the base; card_def salary is used for non-juniors
        # Caller should check card_def.is_junior
        if self.total_xp >= 2:
            return 1
        return 0

    def has_pending_xp_of_type(self, cube_type: CubeType) -> bool:
        return cube_type in self.xp_pending

    def has_skill(self, cube_type: CubeType) -> bool:
        return cube_type in self.skills

    def can_lead_software(self, card_def: TalentCardDef) -> bool:
        """Can this instance act as a software lead for Tier 2/3?"""
        if card_def.is_cross_functional:
            return True
        if card_def.talent_type == TalentType.SENIOR_BACKEND:
            return True
        if card_def.is_junior and self.rank_badges > 0:
            return self.has_skill(CubeType.SOFTWARE) or card_def.output_type == CubeType.SOFTWARE
        return False

    def can_lead_hardware(self, card_def: TalentCardDef) -> bool:
        """Can this instance act as a hardware lead for Tier 2/3?"""
        if card_def.is_cross_functional:
            return True
        if card_def.talent_type == TalentType.SENIOR_HARDWARE:
            return True
        if card_def.is_junior and self.rank_badges > 0:
            return self.has_skill(CubeType.HARDWARE) or card_def.output_type == CubeType.HARDWARE
        return False

    def get_effective_mode(self, card_def: TalentCardDef) -> CubeType | None:
        """Resolve the effective production mode for this talent.

        Multi-skilled juniors use declared_mode. Single-skilled juniors
        use their native output_type. Flex seniors use declared_mode.
        Specialists return None.
        """
        if card_def.is_specialist:
            return None
        if card_def.is_flex or len(self.skills) > 0:
            # Multi-skilled: must use declared mode or default to native
            if self.declared_mode is not None:
                return self.declared_mode
        if card_def.output_type is not None:
            return card_def.output_type
        return self.declared_mode

    def get_output(self, card_def: TalentCardDef) -> int:
        """Calculate total output for this talent in its current mode."""
        if self.onboarding:
            return 0
        base = card_def.base_output

        if card_def.is_junior:
            mode = self.get_effective_mode(card_def)
            if mode is not None:
                # XP bonus: count permanent XP matching current mode
                xp_bonus = sum(1 for x in self.xp_permanent if x == mode)
                return base + xp_bonus
            return base

        return base

    def needs_mode_declaration(self, card_def: TalentCardDef) -> bool:
        """Does this talent need a mode declaration this turn?"""
        if self.onboarding:
            return False
        if card_def.is_flex:
            return True
        if card_def.is_junior and len(self.skills) > 0:
            return True
        return False

    @property
    def is_tier2_plus(self) -> bool:
        """Is this talent at least Tier 2 (Lead)?"""
        return self.rank_badges > 0

    @property
    def is_tier3(self) -> bool:
        """Is this talent Tier 3 (CTO)?"""
        return self.equity_vested is not None and self.rank_badges > 0


@dataclass
class ProductInstance:
    """A product card instance in play (mutable)."""
    instance_id: int
    card_def_id: str
    owner: int  # player_id
    zone: Zone  # DEV or OPS

    # Development progress
    cubes_software: int = 0     # committed cubes on track
    cubes_hardware: int = 0
    transient_software: int = 0  # this turn's cubes (pre-commit)
    transient_hardware: int = 0

    # Status tokens
    bugs: int = 0
    hype: int = 0
    scandal: int = 0

    # State flags
    is_face_down: bool = False
    is_feature_complete: bool = False
    is_online: bool = True

    # Integration
    integrated_with: int | None = None  # instance_id of partner
    is_host: bool = False

    # Legacy tags from "Come Clean" protocol
    legacy_tags: set[Tag] = field(default_factory=set)

    # Stealth launch bonus tracking
    stealth_launch_bonus: int = 0  # +5 for T2, +10 for T3

    # Effective cost after domain expertise reduction
    effective_cost_software: int | None = None  # Set at greenlight, None = use card_def
    effective_cost_hardware: int | None = None

    def progress_software(self, product_def: ProductCardDef) -> float:
        """Fraction of software cubes placed (0.0 to 1.0)."""
        if product_def.cost_software == 0:
            return 1.0
        return min(self.cubes_software / product_def.cost_software, 1.0)

    def progress_hardware(self, product_def: ProductCardDef) -> float:
        if product_def.cost_hardware == 0:
            return 1.0
        return min(self.cubes_hardware / product_def.cost_hardware, 1.0)

    def get_effective_cost(self, product_def: ProductCardDef) -> tuple[int, int]:
        """Get effective software/hardware cost (after domain expertise reduction)."""
        sw = self.effective_cost_software if self.effective_cost_software is not None else product_def.cost_software
        hw = self.effective_cost_hardware if self.effective_cost_hardware is not None else product_def.cost_hardware
        return sw, hw

    def is_development_complete(self, product_def: ProductCardDef) -> bool:
        sw_cost, hw_cost = self.get_effective_cost(product_def)
        return self.cubes_software >= sw_cost and self.cubes_hardware >= hw_cost


@dataclass
class PlayerState:
    """Per-player mutable state."""
    player_id: int
    cash: int = 7  # seed funding

    # Equity system
    equity_tokens_own: int = 3  # starts with 3 of own color
    equity_held: dict[int, int] = field(default_factory=dict)  # player_id -> count

    # Zones (lists of instance_ids)
    bench: list[int] = field(default_factory=list)          # talent instances, max 5
    dev_products: list[int] = field(default_factory=list)    # product instances, max 3
    ops_products: list[int] = field(default_factory=list)    # product instances

    # Hands
    product_backlog: list[int] = field(default_factory=list)  # product instance_ids, max 3
    strategy_hand: list[str] = field(default_factory=list)    # strategy card_def_ids, max 3

    # Scoring tokens
    market_share_tokens: int = 0
    debt_tokens: int = 0
    milestones: list[str] = field(default_factory=list)

    # Turn state
    action_points: int = 0

    def get_all_board_talent(self) -> list[int]:
        """All talent instance_ids on board (dev + ops)."""
        # Talent assigned to dev products or ops products
        # This will be computed from game state
        return []

    def get_tags_in_maintenance(self, product_instances: dict[int, ProductInstance],
                                 product_defs: dict[str, ProductCardDef]) -> set[Tag]:
        """Tags provided by active maintenance products."""
        tags: set[Tag] = set()
        for pid in self.ops_products:
            prod = product_instances[pid]
            pdef = product_defs[prod.card_def_id]
            if prod.is_online and pdef.provides is not None:
                tags.add(pdef.provides)
        return tags

    def highest_active_tier(self, product_instances: dict[int, ProductInstance],
                            product_defs: dict[str, ProductCardDef]) -> Tier:
        """Highest tier among active maintenance products."""
        max_tier = Tier.TIER0
        for pid in self.ops_products:
            prod = product_instances[pid]
            pdef = product_defs[prod.card_def_id]
            if prod.is_online and pdef.tier > max_tier:
                max_tier = pdef.tier
        return max_tier


@dataclass
class MarketState:
    """Shared market state."""
    # Talent markets
    agency_row: list[int] = field(default_factory=list)           # up to 4 talent instance_ids
    open_job_market: deque[int] = field(default_factory=deque)    # FIFO, max 5 talent instance_ids

    # Product markets
    product_market_seed: list[int] = field(default_factory=list)    # 2 face-up product instance_ids
    product_market_growth: list[int] = field(default_factory=list)  # 2 face-up product instance_ids
    open_idea_pool: deque[int] = field(default_factory=deque)       # FIFO, max 5 product instance_ids

    # Decks (lists of card_def_ids, top = end of list for O(1) pop)
    seed_deck: list[str] = field(default_factory=list)
    growth_deck: list[str] = field(default_factory=list)
    talent_deck: list[str] = field(default_factory=list)
    strategy_deck: list[str] = field(default_factory=list)
    strategy_discard: list[str] = field(default_factory=list)
    event_deck: list[str] = field(default_factory=list)
    event_discard: list[str] = field(default_factory=list)

    # Current event
    active_event: str | None = None  # event card_def_id

    # Milestones
    milestones: list[str] = field(default_factory=list)  # face-up milestone card_def_ids


@dataclass
class PendingDecision:
    """A decision that must be resolved before the game can proceed."""
    player_id: int  # who must decide
    decision_type: SubPhase
    context: dict = field(default_factory=dict)  # extra info for the decision


@dataclass
class GameState:
    """Complete game state."""
    num_players: int
    current_player: int = 0
    turn_number: int = 0
    phase: Phase = Phase.SETUP
    sub_phase: SubPhase = SubPhase.NONE

    players: list[PlayerState] = field(default_factory=list)
    markets: MarketState = field(default_factory=MarketState)

    # All card instances in the game
    talent_instances: dict[int, TalentInstance] = field(default_factory=dict)
    product_instances: dict[int, ProductInstance] = field(default_factory=dict)
    next_instance_id: int = 0

    # Decisions pending resolution
    pending_decisions: list[PendingDecision] = field(default_factory=list)

    game_over: bool = False
    market_crash_drawn: bool = False
    finish_round: bool = False  # True when Market Crash drawn but round not yet complete

    # Round modifiers from active event (reset each turn)
    round_modifiers: object = None  # RoundModifiers, typed as object to avoid circular import

    # Track which player is in engine phase (for multi-player engine processing)
    engine_player_index: int = 0

    # Track audit state
    audit_target_product: int | None = None
    audit_bids: dict[int, int] = field(default_factory=dict)  # player_id -> bid amount
    audit_current_bidder: int = 0

    # Scores (populated at game end)
    final_scores: list[int] = field(default_factory=list)

    def create_talent_instance(self, card_def_id: str, owner: int, zone: Zone) -> TalentInstance:
        inst = TalentInstance(
            instance_id=self.next_instance_id,
            card_def_id=card_def_id,
            owner=owner,
            zone=zone,
        )
        self.talent_instances[inst.instance_id] = inst
        self.next_instance_id += 1
        return inst

    def create_product_instance(self, card_def_id: str, owner: int, zone: Zone) -> ProductInstance:
        inst = ProductInstance(
            instance_id=self.next_instance_id,
            card_def_id=card_def_id,
            owner=owner,
            zone=zone,
        )
        self.product_instances[inst.instance_id] = inst
        self.next_instance_id += 1
        return inst

    def get_player(self, player_id: int) -> PlayerState:
        return self.players[player_id]

    def get_talent_on_product(self, product_instance_id: int) -> list[int]:
        """Get all talent instance_ids assigned to a product."""
        return [
            tid for tid, t in self.talent_instances.items()
            if t.assigned_product == product_instance_id
        ]

    def get_board_talent(self, player_id: int) -> list[int]:
        """Get all talent on a player's board (dev + ops zones)."""
        return [
            tid for tid, t in self.talent_instances.items()
            if t.owner == player_id and t.zone in (Zone.DEV, Zone.OPS)
        ]

    def get_all_talent_for_player(self, player_id: int) -> list[int]:
        """Get all talent instance_ids owned by a player."""
        return [
            tid for tid, t in self.talent_instances.items()
            if t.owner == player_id
        ]

    def get_player_tags(self, player_id: int) -> set[Tag]:
        """Get tags provided by a player's active maintenance products.

        Uses registry directly to avoid rebuilding product def dicts.
        """
        from .card_registry import get_registry
        registry = get_registry()
        player = self.players[player_id]
        tags: set[Tag] = set()
        for pid in player.ops_products:
            prod = self.product_instances[pid]
            if not prod.is_online:
                continue
            pdef = registry.get_product(prod.card_def_id)
            if pdef.provides is not None:
                tags.add(pdef.provides)
        return tags

    def get_player_tags_with_partners(self, player_id: int) -> set[Tag]:
        """Get tags available to a player (own + equity partners)."""
        tags = self.get_player_tags(player_id)
        player = self.players[player_id]
        for partner_id, count in player.equity_held.items():
            if count > 0:
                tags |= self.get_player_tags(partner_id)
        return tags

    def get_player_highest_tier(self, player_id: int) -> Tier:
        """Get highest active tier for a player."""
        from .card_registry import get_registry
        registry = get_registry()
        player = self.players[player_id]
        max_tier = Tier.TIER0
        for pid in player.ops_products:
            prod = self.product_instances[pid]
            if not prod.is_online:
                continue
            pdef = registry.get_product(prod.card_def_id)
            if pdef.tier > max_tier:
                max_tier = pdef.tier
        return max_tier
