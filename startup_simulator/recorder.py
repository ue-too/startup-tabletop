"""Game recorder: captures action-by-action snapshots for replay."""

from __future__ import annotations

import json
import random as _random
from dataclasses import asdict, dataclass, field
from typing import Any

from .card_registry import get_registry
from .engine import GameEngine
from .state import GameState
from .types import ActionType, CubeType, Phase, SubPhase, TalentType, Tier, Zone
from .actions.base import Action
from .phases.income_phase import calculate_operational_revenue, calculate_salary_cost


@dataclass
class ProductSummary:
    name: str
    tier: int
    sector: str
    tag: str
    progress_sw: float
    progress_hw: float
    cost_sw: int
    cost_hw: int
    bugs: int
    hype: int
    scandal: int
    is_online: bool
    is_face_down: bool
    is_complete: bool
    team: list[str]
    revenue: int
    integrated: bool
    stealth_bonus: int


@dataclass
class PlayerSummary:
    player_id: int
    cash: int
    equity_own: int
    equity_held: dict[int, int]
    ap: int
    bench: list[str]
    dev_products: list[ProductSummary]
    ops_products: list[ProductSummary]
    backlog_count: int
    strategy_hand: list[str]
    total_revenue: int
    salary_cost: int


@dataclass
class MarketSummary:
    agency: list[str]
    open_jobs: list[str]
    seed_market: list[str]
    growth_market: list[str]
    idea_pool: list[str]
    seed_deck_size: int
    growth_deck_size: int
    talent_deck_size: int
    strategy_deck_size: int


@dataclass
class GameFrame:
    frame_index: int
    turn: int
    phase: str
    sub_phase: str
    current_player: int
    action: str
    result: str
    active_event: str
    players: list[PlayerSummary]
    market: MarketSummary
    scores: list[int] | None = None


def _describe_action(action: Action, state: GameState) -> str:
    """Create a human-readable description of an action."""
    registry = get_registry()
    atype = action.action_type
    pid = state.current_player

    if atype == ActionType.RECRUIT:
        src = action.source_type
        if src == "university_sw":
            return f"P{pid}: Recruit Jr Software from University ($2)"
        elif src == "university_hw":
            return f"P{pid}: Recruit Jr Hardware from University ($2)"
        elif src == "agency":
            idx = action.source_index
            if idx < len(state.markets.agency_row):
                tid = state.markets.agency_row[idx]
                t = state.talent_instances[tid]
                tdef = registry.get_talent(t.card_def_id)
                return f"P{pid}: Recruit {tdef.name} from Agency (${tdef.cost})"
            return f"P{pid}: Recruit from Agency"
        elif src == "open_market":
            return f"P{pid}: Recruit from Open Market ($1)"
        return f"P{pid}: Recruit ({src})"

    elif atype == ActionType.ASSIGN:
        return f"P{pid}: ASSIGN batch started"
    elif atype == ActionType.ASSIGN_ONE:
        tid = action.target_instance
        ppid = action.source_index
        tname = _talent_name(state, tid)
        pname = _product_name(state, ppid)
        return f"P{pid}: Assign {tname} to {pname}"
    elif atype == ActionType.END_ASSIGN_BATCH:
        return f"P{pid}: End assign batch"

    elif atype == ActionType.RECALL:
        return f"P{pid}: RECALL all from Ops to Bench"
    elif atype == ActionType.REASSIGN:
        tname = _talent_name(state, action.target_instance)
        pname = _product_name(state, action.source_index)
        return f"P{pid}: Reassign {tname} to {pname}"

    elif atype == ActionType.LAYOFF_SOURCE:
        count = len(action.target_instances)
        return f"P{pid}: Layoff/Source ({count} discarded)"

    elif atype == ActionType.IDEATION:
        src = action.source_type
        if src in ("seed_market", "growth_market"):
            idx = action.source_index
            market = (state.markets.product_market_seed if src == "seed_market"
                      else state.markets.product_market_growth)
            if idx < len(market):
                pname = _product_name(state, market[idx])
                return f"P{pid}: Draft {pname} from {'Seed' if 'seed' in src else 'Growth'} Market"
        elif src in ("seed_deck", "growth_deck"):
            return f"P{pid}: Draft from {'Seed' if 'seed' in src else 'Growth'} Deck (blind)"
        elif src == "idea_pool":
            return f"P{pid}: Draft from Idea Pool (2 AP)"
        return f"P{pid}: Ideation ({src})"

    elif atype == ActionType.GREENLIGHT:
        pname = _product_name(state, action.target_instance)
        return f"P{pid}: Greenlight {pname}"
    elif atype == ActionType.LAUNCH:
        pname = _product_name(state, action.target_instance)
        return f"P{pid}: LAUNCH {pname}"
    elif atype == ActionType.PIVOT:
        pname = _product_name(state, action.target_instance)
        return f"P{pid}: Pivot/Scrap {pname}"

    elif atype == ActionType.BRAINSTORM:
        return f"P{pid}: Brainstorm (draw strategy cards)"
    elif atype == ActionType.INVEST:
        return f"P{pid}: Invest in Player {action.target_player}"
    elif atype == ActionType.DIVEST:
        return f"P{pid}: Divest equity to Player {action.source_index}"
    elif atype == ActionType.BUYBACK:
        return f"P{pid}: Buyback equity from Player {action.target_player}"
    elif atype == ActionType.ACQUISITION:
        pname = _product_name(state, action.target_instance)
        return f"P{pid}: Acquire {pname} from Player {action.target_player}"

    elif atype == ActionType.PLAY_STRATEGY:
        idx = action.source_index
        player = state.get_player(pid)
        if 0 <= idx < len(player.strategy_hand):
            card_id = player.strategy_hand[idx]
            sdef = registry.get_strategy(card_id)
            return f"P{pid}: Play {sdef.name} (${sdef.cost})"
        return f"P{pid}: Play strategy card"

    elif atype == ActionType.INTEGRATE:
        host = _product_name(state, action.target_instance)
        client = _product_name(state, action.source_index)
        return f"P{pid}: Integrate {host} + {client}"
    elif atype == ActionType.VOLUNTARY_DISCLOSURE:
        pname = _product_name(state, action.target_instance)
        return f"P{pid}: Disclose {pname}"

    elif atype == ActionType.PASS:
        return f"P{pid}: PASS"

    elif atype == ActionType.CHOOSE_MODE:
        mode = CubeType(action.choice).name if action.choice >= 0 else "?"
        return f"P{pid}: Declare mode {mode}"
    elif atype == ActionType.CHOOSE_XP:
        xp = CubeType(action.choice).name if action.choice >= 0 else "?"
        return f"P{pid}: Graduate XP ({xp})"
    elif atype == ActionType.DISCARD_TALENT:
        tname = _talent_name(state, action.target_instance)
        return f"P{pid}: Discard {tname} (bench overflow)"
    elif atype == ActionType.DISCARD_BACKLOG:
        pname = _product_name(state, action.target_instance)
        return f"P{pid}: Discard {pname} (backlog overflow)"
    elif atype == ActionType.CHOOSE_OFFLINE:
        pname = _product_name(state, action.target_instance)
        return f"P{pid}: Take {pname} offline"
    elif atype == ActionType.FIRE_STAFF:
        tname = _talent_name(state, action.target_instance)
        return f"P{pid}: Fire {tname}"
    elif atype == ActionType.BID_AUDIT:
        return f"P{pid}: Bid ${action.amount} on audit"
    elif atype == ActionType.PASS_AUDIT:
        return f"P{pid}: Pass on audit"
    elif atype == ActionType.FOLD:
        return f"P{pid}: FOLD (audit)"
    elif atype == ActionType.SETTLE:
        return f"P{pid}: SETTLE (audit)"

    return f"P{pid}: {atype.name}"


def _talent_name(state: GameState, instance_id: int) -> str:
    registry = get_registry()
    t = state.talent_instances.get(instance_id)
    if t is None:
        return f"Talent#{instance_id}"
    tdef = registry.get_talent(t.card_def_id)
    return tdef.name


def _product_name(state: GameState, instance_id: int) -> str:
    registry = get_registry()
    p = state.product_instances.get(instance_id)
    if p is None:
        return f"Product#{instance_id}"
    pdef = registry.get_product(p.card_def_id)
    return pdef.name


def _snapshot_state(state: GameState) -> tuple[list[PlayerSummary], MarketSummary]:
    """Extract a lightweight snapshot of the current state."""
    registry = get_registry()
    players = []

    for player in state.players:
        # Bench names
        bench_names = []
        for tid in player.bench:
            t = state.talent_instances[tid]
            tdef = registry.get_talent(t.card_def_id)
            label = tdef.name
            if t.total_xp > 0:
                label += f" (XP:{t.total_xp})"
            if t.rank_badges > 0:
                label += " [Lead]"
            bench_names.append(label)

        # Dev products
        dev_prods = []
        for pid in player.dev_products:
            dev_prods.append(_product_summary(state, pid))

        # Ops products
        ops_prods = []
        for pid in player.ops_products:
            ops_prods.append(_product_summary(state, pid))

        # Strategy hand
        strat_names = []
        for card_id in player.strategy_hand:
            sdef = registry.get_strategy(card_id)
            strat_names.append(f"{sdef.name} (${sdef.cost})")

        rev = calculate_operational_revenue(state, player.player_id)
        sal = calculate_salary_cost(state, player.player_id)

        players.append(PlayerSummary(
            player_id=player.player_id,
            cash=player.cash,
            equity_own=player.equity_tokens_own,
            equity_held=dict(player.equity_held),
            ap=player.action_points,
            bench=bench_names,
            dev_products=dev_prods,
            ops_products=ops_prods,
            backlog_count=len(player.product_backlog),
            strategy_hand=strat_names,
            total_revenue=rev,
            salary_cost=sal,
        ))

    # Market
    agency_names = []
    for tid in state.markets.agency_row:
        t = state.talent_instances[tid]
        tdef = registry.get_talent(t.card_def_id)
        agency_names.append(f"{tdef.name} (${tdef.cost})")

    open_job_names = []
    for tid in state.markets.open_job_market:
        t = state.talent_instances[tid]
        tdef = registry.get_talent(t.card_def_id)
        open_job_names.append(tdef.name)

    seed_names = []
    for pid in state.markets.product_market_seed:
        p = state.product_instances[pid]
        pdef = registry.get_product(p.card_def_id)
        seed_names.append(pdef.name)

    growth_names = []
    for pid in state.markets.product_market_growth:
        p = state.product_instances[pid]
        pdef = registry.get_product(p.card_def_id)
        growth_names.append(pdef.name)

    pool_names = []
    for pid in state.markets.open_idea_pool:
        p = state.product_instances[pid]
        pdef = registry.get_product(p.card_def_id)
        pool_names.append(pdef.name)

    market = MarketSummary(
        agency=agency_names,
        open_jobs=open_job_names,
        seed_market=seed_names,
        growth_market=growth_names,
        idea_pool=pool_names,
        seed_deck_size=len(state.markets.seed_deck),
        growth_deck_size=len(state.markets.growth_deck),
        talent_deck_size=len(state.markets.talent_deck),
        strategy_deck_size=len(state.markets.strategy_deck),
    )

    return players, market


def _product_summary(state: GameState, pid: int) -> ProductSummary:
    registry = get_registry()
    prod = state.product_instances[pid]
    pdef = registry.get_product(prod.card_def_id)
    sw_cost, hw_cost = prod.get_effective_cost(pdef)

    team = state.get_talent_on_product(pid)
    team_names = []
    for tid in team:
        t = state.talent_instances[tid]
        tdef = registry.get_talent(t.card_def_id)
        label = tdef.name
        output = t.get_output(tdef)
        if output > 0:
            label += f"({output})"
        if t.onboarding:
            label += "[onb]"
        team_names.append(label)

    return ProductSummary(
        name=pdef.name,
        tier=pdef.tier,
        sector=pdef.sector.name if pdef.sector else "",
        tag=pdef.provides.name if pdef.provides else "",
        progress_sw=prod.cubes_software / max(sw_cost, 1) if sw_cost > 0 else 1.0,
        progress_hw=prod.cubes_hardware / max(hw_cost, 1) if hw_cost > 0 else 1.0,
        cost_sw=sw_cost,
        cost_hw=hw_cost,
        bugs=prod.bugs,
        hype=prod.hype,
        scandal=prod.scandal,
        is_online=prod.is_online,
        is_face_down=prod.is_face_down,
        is_complete=prod.is_feature_complete,
        team=team_names,
        revenue=pdef.revenue,
        integrated=prod.integrated_with is not None,
        stealth_bonus=prod.stealth_launch_bonus,
    )


class GameRecorder:
    """Records a game action-by-action for replay."""

    def __init__(self, num_players: int = 2, seed: int = 42) -> None:
        self.engine = GameEngine(num_players=num_players, seed=seed)
        self.frames: list[GameFrame] = []
        self.seed = seed
        self.num_players = num_players
        self._record_initial_frame()

    def _record_initial_frame(self) -> None:
        """Record the initial state before any actions."""
        state = self.engine.state
        players, market = _snapshot_state(state)
        event_name = self._event_name(state)
        self.frames.append(GameFrame(
            frame_index=0,
            turn=state.turn_number,
            phase=state.phase.name,
            sub_phase=state.sub_phase.name,
            current_player=state.current_player,
            action="Game Start",
            result=f"Seed: {self.seed}, {self.num_players} players",
            active_event=event_name,
            players=players,
            market=market,
        ))

    def step(self, action: Action) -> Any:
        """Execute an action and record the frame."""
        state = self.engine.state
        action_desc = _describe_action(action, state)
        result = self.engine.step(action)

        state = self.engine.state
        players, market = _snapshot_state(state)
        event_name = self._event_name(state)

        frame = GameFrame(
            frame_index=len(self.frames),
            turn=state.turn_number,
            phase=state.phase.name,
            sub_phase=state.sub_phase.name,
            current_player=state.current_player,
            action=action_desc,
            result=result.action_result.message if result.action_result else "",
            active_event=event_name,
            players=players,
            market=market,
            scores=list(state.final_scores) if state.game_over else None,
        )
        self.frames.append(frame)
        return result

    def get_legal_actions(self) -> list[Action]:
        return self.engine.get_legal_actions()

    def is_done(self) -> bool:
        return self.engine.is_done()

    def _event_name(self, state: GameState) -> str:
        if state.markets.active_event is None:
            return ""
        registry = get_registry()
        edef = registry.get_event(state.markets.active_event)
        return edef.name

    def to_json(self) -> str:
        """Serialize all frames to JSON."""
        data = {
            "seed": self.seed,
            "num_players": self.num_players,
            "num_frames": len(self.frames),
            "frames": [asdict(f) for f in self.frames],
        }
        return json.dumps(data, indent=2, default=str)


def record_random_game(
    num_players: int = 2, seed: int = 42, agent_seed: int = 100, max_steps: int = 5000,
) -> GameRecorder:
    """Record a full game with random agents."""
    recorder = GameRecorder(num_players=num_players, seed=seed)
    rng = _random.Random(agent_seed)
    steps = 0

    while not recorder.is_done() and steps < max_steps:
        actions = recorder.get_legal_actions()
        action = rng.choice(actions)
        recorder.step(action)
        steps += 1

    return recorder


def record_heuristic_game(
    num_players: int = 2, seed: int = 42, agent_seed: int = 100, max_steps: int = 5000,
) -> GameRecorder:
    """Record a full game with heuristic agents."""
    from agents.heuristic_agent import HeuristicAgent
    import numpy as np

    recorder = GameRecorder(num_players=num_players, seed=seed)
    agents = [HeuristicAgent(seed=agent_seed + i) for i in range(num_players)]
    steps = 0

    while not recorder.is_done() and steps < max_steps:
        pid = recorder.engine.get_current_agent()
        legal = recorder.get_legal_actions()
        mask = np.zeros(512, dtype=np.int8)
        mask[:len(legal)] = 1
        obs = {"observation": np.zeros(1), "action_mask": mask}
        action_idx = agents[pid].act(obs, legal_actions=legal)
        action = legal[min(action_idx, len(legal) - 1)]
        recorder.step(action)
        steps += 1

    return recorder
