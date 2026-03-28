"""Terminal replay viewer using rich."""

from __future__ import annotations

import sys
import tty
import termios
from typing import TextIO

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

from .recorder import GameFrame, GameRecorder, ProductSummary


# Colors per player
PLAYER_COLORS = ["bright_cyan", "bright_yellow", "bright_green", "bright_magenta"]
TIER_LABELS = {1: "T1", 2: "T2", 3: "T3"}


def _progress_bar(frac: float, width: int = 10, label: str = "") -> str:
    """Render a text progress bar."""
    filled = int(frac * width)
    empty = width - filled
    bar = "#" * filled + "-" * empty
    pct = f"{frac * 100:.0f}%"
    return f"[{bar}] {pct} {label}"


def _product_line(prod: ProductSummary) -> Text:
    """Render a single product as a rich Text."""
    text = Text()

    # Name and tier
    tier_color = {"1": "white", "2": "bright_blue", "3": "bright_red"}.get(str(prod.tier), "white")
    stealth = " [dim](stealth)[/dim]" if prod.is_face_down else ""
    name_str = f"  {TIER_LABELS.get(prod.tier, '??')} {prod.name}"

    text.append(name_str, style=tier_color)
    if prod.is_face_down:
        text.append(" (stealth)", style="dim")
    if prod.integrated:
        text.append(" +eco", style="bright_green")
    if not prod.is_online:
        text.append(" OFFLINE", style="bright_red bold")
    text.append("\n")

    # Progress (dev products)
    if prod.cost_sw > 0 or prod.cost_hw > 0:
        if prod.cost_sw > 0:
            sw_bar = _progress_bar(prod.progress_sw, 8, "{}")
            text.append(f"    SW: {sw_bar}\n", style="bright_blue")
        if prod.cost_hw > 0:
            hw_bar = _progress_bar(prod.progress_hw, 8, "[Chk]")
            text.append(f"    HW: {hw_bar}\n", style="bright_red")

    # Status tokens
    tokens = []
    if prod.bugs > 0:
        tokens.append(f"Bug:{prod.bugs}")
    if prod.hype > 0:
        tokens.append(f"Hype:{prod.hype}")
    if prod.scandal > 0:
        tokens.append(f"Scandal:{prod.scandal}")
    if prod.revenue > 0 and prod.is_online:
        tokens.append(f"Rev:${prod.revenue}")
    if tokens:
        text.append(f"    {' '.join(tokens)}\n", style="dim")

    # Team
    if prod.team:
        team_str = ", ".join(prod.team[:4])
        if len(prod.team) > 4:
            team_str += f" +{len(prod.team) - 4}"
        text.append(f"    Team: {team_str}\n", style="dim italic")

    return text


def _render_player(frame: GameFrame, player_idx: int) -> Panel:
    """Render a player panel."""
    if player_idx >= len(frame.players):
        return Panel("", title="(empty)")

    p = frame.players[player_idx]
    color = PLAYER_COLORS[player_idx % len(PLAYER_COLORS)]
    active = " *" if player_idx == frame.current_player else ""

    text = Text()

    # Header stats
    text.append(f"  Cash: ${p.cash}", style="bold")
    text.append(f"  AP: {p.ap}", style="bright_white")
    text.append(f"  Equity: {p.equity_own}/3", style="dim")
    if p.equity_held:
        held = ", ".join(f"P{k}:{v}" for k, v in p.equity_held.items())
        text.append(f"  Holds: {held}", style="dim")
    text.append("\n")
    text.append(f"  Rev: ${p.total_revenue}  Salary: ${p.salary_cost}", style="dim")
    text.append(f"  Bench: {len(p.bench)}  Backlog: {p.backlog_count}\n", style="dim")

    # Dev products
    if p.dev_products:
        text.append("  DEV:\n", style="bold bright_blue")
        for prod in p.dev_products:
            text.append_text(_product_line(prod))
    else:
        text.append("  DEV: (empty)\n", style="dim")

    # Ops products
    if p.ops_products:
        text.append("  OPS:\n", style="bold bright_green")
        for prod in p.ops_products:
            text.append_text(_product_line(prod))
    else:
        text.append("  OPS: (empty)\n", style="dim")

    # Bench summary
    if p.bench:
        bench_short = ", ".join(p.bench[:3])
        if len(p.bench) > 3:
            bench_short += f" +{len(p.bench) - 3}"
        text.append(f"  Bench: {bench_short}\n", style="dim")

    # Strategy hand
    if p.strategy_hand:
        strat_str = ", ".join(p.strategy_hand[:3])
        text.append(f"  Cards: {strat_str}\n", style="dim")

    title = f"Player {player_idx}{active}"
    return Panel(text, title=title, border_style=color, width=50)


def _render_market(frame: GameFrame) -> Panel:
    """Render the market panel."""
    m = frame.market
    text = Text()

    # Agency row
    agency_str = " | ".join(m.agency) if m.agency else "(empty)"
    text.append(f"Agency: {agency_str}\n", style="bright_white")

    # Product market
    seed_str = " | ".join(m.seed_market) if m.seed_market else "(empty)"
    growth_str = " | ".join(m.growth_market) if m.growth_market else "(empty)"
    text.append(f"Seed: {seed_str}    Growth: {growth_str}\n")

    # Deck sizes
    text.append(
        f"Decks: Seed({m.seed_deck_size}) Growth({m.growth_deck_size}) "
        f"Talent({m.talent_deck_size}) Strategy({m.strategy_deck_size})\n",
        style="dim",
    )

    # Open pools
    if m.open_jobs:
        text.append(f"Jobs: {', '.join(m.open_jobs[:3])}\n", style="dim")
    if m.idea_pool:
        text.append(f"Ideas: {', '.join(m.idea_pool[:3])}\n", style="dim")

    return Panel(text, title="Market", border_style="white")


def _render_frame(console: Console, frame: GameFrame, history: list[str], total_frames: int) -> None:
    """Render a single frame to the console."""
    console.clear()

    # Top bar
    event_str = f"Event: {frame.active_event}" if frame.active_event else "Event: (none)"
    header = Text()
    header.append(f" Turn {frame.turn}", style="bold")
    header.append(f" | {frame.phase}", style="bright_white")
    if frame.sub_phase != "NONE":
        header.append(f".{frame.sub_phase}", style="dim")
    header.append(f" | {event_str}", style="bright_yellow")
    header.append(f" | Frame {frame.frame_index}/{total_frames - 1}", style="dim")
    console.print(Panel(header, style="bold"))

    # Player panels (2 columns)
    num_players = len(frame.players)
    player_panels = [_render_player(frame, i) for i in range(num_players)]

    if num_players <= 2:
        console.print(Columns(player_panels, equal=True, expand=True))
    else:
        # 2 rows of 2
        console.print(Columns(player_panels[:2], equal=True, expand=True))
        if len(player_panels) > 2:
            console.print(Columns(player_panels[2:], equal=True, expand=True))

    # Market
    console.print(_render_market(frame))

    # Action log (last N entries)
    log_lines = history[-8:]
    log_text = Text()
    for i, line in enumerate(log_lines):
        is_current = (i == len(log_lines) - 1)
        if is_current:
            log_text.append(f"-> {line}\n", style="bold bright_white")
        else:
            log_text.append(f"   {line}\n", style="dim")
    console.print(Panel(log_text, title="Action Log", border_style="bright_blue"))

    # Scores (if game over)
    if frame.scores:
        score_text = " | ".join(f"P{i}: {s} VP" for i, s in enumerate(frame.scores))
        winner = frame.scores.index(max(frame.scores))
        console.print(Panel(
            f"GAME OVER! Scores: {score_text}\nWinner: Player {winner}",
            title="Final Scores",
            border_style="bright_green bold",
        ))

    # Controls
    console.print(
        "[dim]Controls: [->] next  [<-] prev  [f] next turn  "
        "[Home] start  [End] end  [q] quit[/dim]"
    )


def _read_key() -> str:
    """Read a single keypress (blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "C":
                    return "right"
                elif ch3 == "D":
                    return "left"
                elif ch3 == "H":
                    return "home"
                elif ch3 == "F":
                    return "end"
                elif ch3 == "1":
                    ch4 = sys.stdin.read(1)
                    if ch4 == "~":
                        return "home"
                elif ch3 == "4":
                    ch4 = sys.stdin.read(1)
                    if ch4 == "~":
                        return "end"
            return "esc"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def view_replay(recorder: GameRecorder) -> None:
    """Interactive TUI replay viewer."""
    frames = recorder.frames
    if not frames:
        print("No frames to display.")
        return

    console = Console()
    current = 0
    history: list[str] = []

    while True:
        frame = frames[current]

        # Build action history up to current frame
        history = [f.action for f in frames[1:current + 1]]

        _render_frame(console, frame, history, len(frames))

        key = _read_key()

        if key == "q" or key == "\x03":  # q or Ctrl-C
            break
        elif key == "right":
            if current < len(frames) - 1:
                current += 1
        elif key == "left":
            if current > 0:
                current -= 1
        elif key == "f":
            # Fast-forward to next turn
            current_turn = frames[current].turn
            while current < len(frames) - 1:
                current += 1
                if frames[current].turn > current_turn:
                    break
        elif key == "home":
            current = 0
        elif key == "end":
            current = len(frames) - 1

    console.clear()
    console.print("[bold]Replay ended.[/bold]")
