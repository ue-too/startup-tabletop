"""Core enumerations and type aliases for Startup Simulator."""

from enum import IntEnum, auto


class CubeType(IntEnum):
    SOFTWARE = 0   # Blue {}
    HARDWARE = 1   # Red [Chk]
    QA = 2         # Green (for skill tokens / pending XP)


class Tier(IntEnum):
    TIER0 = 0  # Special (Market Crash)
    TIER1 = 1
    TIER2 = 2
    TIER3 = 3


class Sector(IntEnum):
    CONCEPT = 0
    CONSUMER = 1    # Pink
    FINTECH = 2     # Gold
    DEEP_TECH = 3   # Gray
    LIFE_SCI = 4    # Green
    INFRA = 5       # Blue


class Tag(IntEnum):
    APP = 0
    SOCIAL = 1
    MEDIA = 2
    COMMERCE = 3
    CRYPTO = 4
    DEFI = 5
    IOT = 6
    ROBOTICS = 7
    ENERGY = 8
    FUSION = 9
    DATA = 10
    BIO = 11
    LONGEVITY = 12
    PLATFORM = 13
    CLOUD = 14
    AI = 15
    QUANTUM = 16
    DEVICE = 17
    NETWORK = 18
    SERVICE = 19
    FINTECH_TAG = 20
    METAVERSE = 21


class TalentType(IntEnum):
    JUNIOR_SOFTWARE = 0
    JUNIOR_HARDWARE = 1
    SENIOR_BACKEND = 2
    SENIOR_HARDWARE = 3
    FIRMWARE = 4
    FULL_STACK = 5
    QA = 6
    SALES = 7
    HR = 8
    PM = 9
    SENIOR_PM = 10
    GROWTH_HACKER = 11


class Zone(IntEnum):
    BENCH = 0
    DEV = 1
    OPS = 2


class Phase(IntEnum):
    SETUP = 0
    EVENT = 1           # Phase A
    INCOME = 2          # Phase B
    ACTION = 3          # Phase C
    ENGINE = 4          # Phase D
    GAME_OVER = 5


class SubPhase(IntEnum):
    NONE = 0
    # Income sub-phases
    INCOME_BANDWIDTH = 1
    INCOME_CHOOSE_OFFLINE = 2
    INCOME_REVENUE = 3
    INCOME_SALARY = 4
    INCOME_FIRE_CHOICE = 5
    # Action sub-phases
    ACTION_MAIN = 10
    ACTION_ASSIGN_BATCH = 11    # Sequential assign within 1 AP
    ACTION_CONSENT = 12         # Waiting for opponent consent
    ACTION_COUNTER_OFFER = 13   # Waiting for poach counter-offer
    # Engine sub-phases
    ENGINE_MODE_DECLARE = 20
    ENGINE_GENERATE = 21
    ENGINE_QA = 22
    ENGINE_AUDIT_BID = 23
    ENGINE_AUDIT_RESOLVE = 24
    ENGINE_COMMIT = 25
    ENGINE_COMPLETE = 26
    ENGINE_XP_GRADUATE = 27
    ENGINE_REFILL = 28
    ENGINE_CLEANUP = 29
    ENGINE_CLEANUP_TALENT = 30
    ENGINE_CLEANUP_STRATEGY = 31
    ENGINE_CLEANUP_BACKLOG = 32


class ActionType(IntEnum):
    # 1-AP actions
    RECRUIT = 0
    ASSIGN = 1
    RECALL = 2
    REASSIGN = 3
    LAYOFF_SOURCE = 4
    IDEATION = 5
    LAUNCH = 6
    PIVOT = 7
    ACQUISITION = 8
    BRAINSTORM = 9
    INVEST = 10
    DIVEST = 11
    BUYBACK = 12
    SECONDARY_TRADE = 13
    # 0-AP actions
    GREENLIGHT = 14
    PLAY_STRATEGY = 15
    INTEGRATE = 16
    VOLUNTARY_DISCLOSURE = 17
    # Control
    PASS = 18
    END_ASSIGN_BATCH = 19
    # Micro-decisions
    ASSIGN_ONE = 20         # Single card assignment within batch
    CHOOSE_OFFLINE = 21     # Pick product to go offline
    FIRE_STAFF = 22         # Pick staff to fire for unpaid salary
    CHOOSE_MODE = 23        # Declare junior mode (SW/HW/QA)
    BID_AUDIT = 24
    PASS_AUDIT = 25
    FOLD = 26
    SETTLE = 27
    CONSENT_YES = 28
    CONSENT_NO = 29
    COUNTER_OFFER = 30
    DECLINE_COUNTER = 31
    CHOOSE_XP = 32          # Choose which pending XP to graduate
    DISCARD_TALENT = 33     # Discard from bench during cleanup
    DISCARD_STRATEGY = 34   # Discard strategy card during cleanup
    DISCARD_BACKLOG = 35    # Discard product from backlog during cleanup


# Trait identifiers
class Trait(IntEnum):
    SPAGHETTI_CODE = 0
    CLEAN_CODE = 1
    MENTOR = 2
    EGO = 3
    EFFICIENT = 4
    QA_SKILL = 5
    MERCENARY = 6
    BUG_HUNTER = 7
    RAINMAKER = 8
    GATEKEEPER = 9
    SYNERGY = 10
    AGILE_SYNERGY = 11
    VIRAL_LOOP = 12


TRAIT_NAME_MAP: dict[str, Trait] = {
    "spaghetti_code": Trait.SPAGHETTI_CODE,
    "clean_code": Trait.CLEAN_CODE,
    "mentor": Trait.MENTOR,
    "ego": Trait.EGO,
    "efficient": Trait.EFFICIENT,
    "qa_skill": Trait.QA_SKILL,
    "mercenary": Trait.MERCENARY,
    "bug_hunter": Trait.BUG_HUNTER,
    "rainmaker": Trait.RAINMAKER,
    "gatekeeper": Trait.GATEKEEPER,
    "synergy": Trait.SYNERGY,
    "agile_synergy": Trait.AGILE_SYNERGY,
    "viral_loop": Trait.VIRAL_LOOP,
}

# Maps for JSON deserialization
CUBE_TYPE_MAP: dict[str, CubeType] = {
    "SOFTWARE": CubeType.SOFTWARE,
    "HARDWARE": CubeType.HARDWARE,
    "QA": CubeType.QA,
}

SECTOR_MAP: dict[str, Sector] = {s.name: s for s in Sector}

TAG_MAP: dict[str, Tag] = {t.name: t for t in Tag}

TALENT_TYPE_MAP: dict[str, TalentType] = {t.name: t for t in TalentType}

# Which actions cost 1 AP
AP_COST: dict[ActionType, int] = {
    ActionType.RECRUIT: 1,
    ActionType.ASSIGN: 1,
    ActionType.RECALL: 1,
    ActionType.REASSIGN: 1,
    ActionType.LAYOFF_SOURCE: 1,
    ActionType.IDEATION: 1,
    ActionType.LAUNCH: 1,
    ActionType.PIVOT: 1,
    ActionType.ACQUISITION: 1,
    ActionType.BRAINSTORM: 1,
    ActionType.INVEST: 1,
    ActionType.DIVEST: 1,
    ActionType.BUYBACK: 1,
    ActionType.SECONDARY_TRADE: 1,
    # 0 AP actions
    ActionType.GREENLIGHT: 0,
    ActionType.PLAY_STRATEGY: 0,
    ActionType.INTEGRATE: 0,
    ActionType.VOLUNTARY_DISCLOSURE: 0,
    ActionType.PASS: 0,
    ActionType.END_ASSIGN_BATCH: 0,
    ActionType.ASSIGN_ONE: 0,
}

# Ideation from Open Idea Pool costs 2 AP
IDEATION_POOL_AP_COST = 2
