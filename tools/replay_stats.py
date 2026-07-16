#!/usr/bin/env python3
"""Tourney step 1 (pre-R0) — batch tournament-replay analytics.

Parses every replay JSON in ``match_replays/`` and answers, with numbers:
how do we die, who is strong, who splits, and which teams are worth cloning
(HANDOFF-tourney-diagnosis-session.md §4.1). The per-team aggregate feeds the
v3 R0 converter's team selection (BC-toward-winners) and the R1 clone list.

Semantics are grounded in the engine source (vendor/agario-public), not
inferred from the JSON:

* A replay is the engine's ``private_event_history``: ``event_game_started``
  (carries ``engine_version`` and the pid→team map), then per round —
  ``move_player`` for **alive players only**, virus/food/eaten resolution
  events, then **exactly one ``event_player_moved`` snapshot per player**
  (dead included) — and finally ``event_player_won``.
* Round index of any event = completed snapshot blocks (0-based, matching
  ``GameState.round``). Never divide move counts by player count: dead
  players issue no moves.
* mass = radius² (``BlobState.mass``); player radius = √(Σ blob mass), so a
  snapshot's ``radius²`` is the player's total mass.
* Final ranking = players sorted by final radius desc, ties → lower
  player_id (``GameState.get_rankings`` sorts a pid-ordered dict with a
  stable sort; dead players have radius 0.0). Cross-checked against
  ``event_player_won`` on every parse.
* A death = ``event_player_eaten`` with ``eaten_player_alive == false``
  (last blob gone); the engine respawns the player RESPAWN_DELAY_ROUNDS
  later, so deaths per match can exceed 1.
* Foods are counted in **pellets** (Σ ``len(food_ids)``): one
  ``event_food_eaten`` groups every pellet a blob ate that round. Each
  pellet is worth FOOD_RADIUS² mass.
* ``eater_radius`` in eaten events is **post-absorption**; the pre-eat
  radius is √(eater² − eaten²). Both ratios are reported (``eaten_radius``
  itself only exists from engine 2026.1.14 on).
* A split command is only *effective* if, at apply time (= the previous
  round's snapshot state — splits are applied before any movement or
  eating), the player had < MAX_BLOB_COUNT blobs and ≥ 1 blob with
  mass ≥ SPLIT_MIN_MASS.

Engine-version policy: 2026.1.13 / .14 / .15 are mechanics-identical for
replay purposes (.14 changed live-query censoring + added ``eaten_radius``;
.15 changed ban-event logging only — vendor diff audited 2026-07-17).
Anything else is quarantined out of the aggregates unless ``--allow-version``
says otherwise.

Usage:
    tools/replay_stats.py                      # all of match_replays/, team 53
    tools/replay_stats.py --team 53 --json data/replay_stats.json
    tools/replay_stats.py match_replays/match-37245-replay.json
    tools/replay_stats.py --min-matches 3 --timeline-step 200

Exit 0 ⇒ ≥1 replay parsed; exit 1 ⇒ nothing usable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bot.spec import (  # noqa: E402
    MAX_BLOB_COUNT,
    NUM_PLAYERS,
    SPLIT_MIN_MASS,
)

#: Versions whose game mechanics are interchangeable for replay statistics.
#: (.13→.14: censoring/logging only, physics byte-identical — see
#: notes/engine-upgrade-v2026.1.14.md; .14→.15: ban-event serialisation only.)
COMPATIBLE_ENGINE_VERSIONS = frozenset({"2026.1.13", "2026.1.14", "2026.1.15"})

KNOWN_EVENT_TYPES = frozenset(
    {
        "event_game_started",
        "move_player",
        "event_player_moved",
        "event_food_eaten",
        "event_food_spawned",
        "event_virus_spawned",
        "event_virus_consumed",
        "event_player_eaten",
        "event_player_won",
    }
)

OUR_TEAM_DEFAULT = 53


class ReplayFormatError(ValueError):
    """The file does not match the engine's replay structure."""


@dataclass
class PlayerStats:
    player_id: int
    team_id: int
    splits_cmd: int = 0
    splits_eff: int = 0
    captures: int = 0
    capture_ratios_pre: List[float] = field(default_factory=list)
    capture_ratios_post: List[float] = field(default_factory=list)
    blobs_lost: int = 0
    deaths: int = 0
    death_rounds: List[int] = field(default_factory=list)
    foods: int = 0
    food_events: int = 0
    shatters: int = 0
    shatter_pieces: int = 0
    moves: int = 0
    alive_rounds: int = 0
    max_blobs: int = 1
    final_mass: float = 0.0
    peak_mass: float = 0.0
    mass_sum: float = 0.0  # over all round snapshots → mean via /rounds
    rank: int = 0


@dataclass
class MatchStats:
    source: str
    match_id: str
    engine_version: str
    n_players: int
    rounds: int
    winner_pid: int
    players: Dict[int, PlayerStats]
    timeline: Dict[int, Dict[int, float]]  # round -> pid -> mass
    warnings: List[str] = field(default_factory=list)

    def team_of(self, pid: int) -> int:
        return self.players[pid].team_id

    def pid_of_team(self, team_id: int) -> Optional[int]:
        for pid, p in self.players.items():
            if p.team_id == team_id:
                return pid
        return None


def _mass_of_blobs(blobs: Sequence[dict]) -> float:
    return sum(b["radius"] ** 2 for b in blobs)


def parse_replay(events: Sequence[dict], source: str = "<memory>",
                 timeline_step: int = 100) -> MatchStats:
    """Single pass over one replay's event list → :class:`MatchStats`.

    Raises :class:`ReplayFormatError` on any structural violation, so a
    malformed file can never leak half-parsed numbers into the aggregates.
    """
    if not events:
        raise ReplayFormatError(f"{source}: empty event list")
    start = events[0]
    if start.get("event_type") != "event_game_started":
        raise ReplayFormatError(
            f"{source}: first event is {start.get('event_type')!r}, "
            "expected event_game_started"
        )

    n_players = len(start["players"])
    if n_players != NUM_PLAYERS:
        # Not fatal for stats, but worth surfacing loudly.
        pass
    stats: Dict[int, PlayerStats] = {
        p["player_id"]: PlayerStats(p["player_id"], p["team_id"])
        for p in start["players"]
    }
    if len(stats) != n_players:
        raise ReplayFormatError(f"{source}: duplicate player_id in header")

    # Blob state at split-apply time == previous round's snapshot (splits are
    # the first thing commit_round applies). Seed from the header.
    prev_blobs: Dict[int, List[dict]] = {
        p["player_id"]: list(p["blobs"]) for p in start["players"]
    }

    timeline: Dict[int, Dict[int, float]] = {}
    rnd = 0  # completed snapshot blocks == engine round of in-flight events
    snaps_in_block = 0
    winner_pid: Optional[int] = None
    warnings: List[str] = []
    final_radius: Dict[int, float] = {
        p["player_id"]: p["radius"] for p in start["players"]
    }
    pending_blobs: Dict[int, List[dict]] = {}

    for e in events[1:]:
        et = e.get("event_type")
        if et not in KNOWN_EVENT_TYPES:
            raise ReplayFormatError(f"{source}: unknown event type {et!r}")

        if et == "event_player_moved":
            pid = e["player_id"]
            p = stats[pid]
            m = _mass_of_blobs(e["blobs"])
            p.mass_sum += m
            p.peak_mass = max(p.peak_mass, m)
            p.final_mass = m
            p.max_blobs = max(p.max_blobs, len(e["blobs"]))
            if e["alive"]:
                p.alive_rounds += 1
            final_radius[pid] = e["radius"] if e["alive"] else 0.0
            pending_blobs[pid] = list(e["blobs"])
            snaps_in_block += 1
            if snaps_in_block == n_players:
                if len(pending_blobs) != n_players:
                    raise ReplayFormatError(
                        f"{source}: snapshot block for round {rnd} covers "
                        f"{len(pending_blobs)}/{n_players} players"
                    )
                if rnd % timeline_step == 0:
                    timeline[rnd] = {
                        q: _mass_of_blobs(pending_blobs[q]) for q in pending_blobs
                    }
                prev_blobs.update(pending_blobs)
                pending_blobs = {}
                snaps_in_block = 0
                rnd += 1
            continue

        if snaps_in_block:
            raise ReplayFormatError(
                f"{source}: {et} interleaved inside a snapshot block "
                f"(round {rnd}, after {snaps_in_block}/{n_players} snapshots)"
            )

        if et == "move_player":
            pid = e["player_id"]
            p = stats[pid]
            p.moves += 1
            if e.get("split"):
                p.splits_cmd += 1
                blobs = prev_blobs.get(pid, [])
                if len(blobs) < MAX_BLOB_COUNT and any(
                    b["radius"] ** 2 >= SPLIT_MIN_MASS for b in blobs
                ):
                    p.splits_eff += 1
        elif et == "event_food_eaten":
            p = stats[e["player_id"]]
            p.food_events += 1
            p.foods += len(e["food_ids"])
        elif et == "event_player_eaten":
            eater = stats[e["eater_player_id"]]
            eaten = stats[e["eaten_player_id"]]
            eater.captures += 1
            eaten.blobs_lost += 1
            eaten_r = e.get("eaten_radius")  # absent before engine 2026.1.14
            if eaten_r:
                post = e["eater_radius"] / eaten_r
                eater.capture_ratios_post.append(post)
                pre_sq = e["eater_radius"] ** 2 - eaten_r**2
                eater.capture_ratios_pre.append(
                    math.sqrt(max(pre_sq, 0.0)) / eaten_r
                )
            if not e["eaten_player_alive"]:
                eaten.deaths += 1
                eaten.death_rounds.append(rnd)
        elif et == "event_virus_consumed":
            p = stats[e["player_id"]]
            p.shatters += 1
            p.shatter_pieces += e["pieces_created"]
        elif et == "event_player_won":
            winner_pid = e["player_id"]
        # event_food_spawned / event_virus_spawned carry no per-player stats.

    if snaps_in_block:
        raise ReplayFormatError(
            f"{source}: trailing partial snapshot block "
            f"({snaps_in_block}/{n_players})"
        )
    if winner_pid is None:
        raise ReplayFormatError(f"{source}: no event_player_won")

    # Always include the final round in the timeline (final masses).
    if rnd and (rnd - 1) not in timeline:
        timeline[rnd - 1] = {pid: p.final_mass for pid, p in stats.items()}

    # Final ranking, engine rule: radius desc, stable over pid order.
    order = sorted(stats, key=lambda pid: (-final_radius[pid], pid))
    for i, pid in enumerate(order):
        stats[pid].rank = i + 1
    if order[0] != winner_pid:
        warnings.append(
            f"rank reconstruction disagrees with event_player_won "
            f"(computed {order[0]}, engine says {winner_pid})"
        )

    match_id = Path(source).name.replace("-replay.json", "").replace(".json", "")
    return MatchStats(
        source=source,
        match_id=match_id,
        engine_version=start.get("engine_version", "MISSING"),
        n_players=n_players,
        rounds=rnd,
        winner_pid=winner_pid,
        players=stats,
        timeline=timeline,
        warnings=warnings,
    )


def load_replay(path: Path, timeline_step: int = 100) -> MatchStats:
    with open(path) as f:
        events = json.load(f)
    if not isinstance(events, list):
        raise ReplayFormatError(f"{path}: top level is not an event list")
    return parse_replay(events, source=str(path), timeline_step=timeline_step)


# --- aggregation --------------------------------------------------------------


@dataclass
class TeamAgg:
    team_id: int
    matches: int = 0
    wins: int = 0
    rank_sum: int = 0
    splits_cmd: int = 0
    splits_eff: int = 0
    captures: int = 0
    ratios_pre: List[float] = field(default_factory=list)
    ratios_post: List[float] = field(default_factory=list)
    deaths: int = 0
    foods: int = 0
    shatters: int = 0
    moves: int = 0
    final_mass_sum: float = 0.0
    peak_mass_sum: float = 0.0

    @property
    def mean_rank(self) -> float:
        return self.rank_sum / self.matches


def aggregate_teams(matches: Sequence[MatchStats]) -> Dict[int, TeamAgg]:
    teams: Dict[int, TeamAgg] = {}
    for m in matches:
        for p in m.players.values():
            t = teams.setdefault(p.team_id, TeamAgg(p.team_id))
            t.matches += 1
            t.wins += 1 if p.rank == 1 else 0
            t.rank_sum += p.rank
            t.splits_cmd += p.splits_cmd
            t.splits_eff += p.splits_eff
            t.captures += p.captures
            t.ratios_pre.extend(p.capture_ratios_pre)
            t.ratios_post.extend(p.capture_ratios_post)
            t.deaths += p.deaths
            t.foods += p.foods
            t.shatters += p.shatters
            t.moves += p.moves
            t.final_mass_sum += p.final_mass
            t.peak_mass_sum += p.peak_mass
    return teams


def clone_candidates(
    teams: Dict[int, TeamAgg],
    exclude_team: Optional[int],
    min_matches: int,
) -> List[TeamAgg]:
    ranked = sorted(
        (
            t
            for t in teams.values()
            if t.team_id != exclude_team and t.matches >= min_matches
        ),
        key=lambda t: t.mean_rank,
    )
    return ranked


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


# --- report -------------------------------------------------------------------


def our_match_table(matches: Sequence[MatchStats], team: int) -> str:
    lines = [
        "match       rank  final_m  peak_m  splits  foods  deaths"
        "  1st_death  caps  shatters  win_team  win_final  win_splits  win_foods"
    ]
    ranks: List[int] = []
    for m in matches:
        pid = m.pid_of_team(team)
        if pid is None:
            lines.append(f"{m.match_id:<11} (team {team} not in this match)")
            continue
        p = m.players[pid]
        w = m.players[m.winner_pid]
        ranks.append(p.rank)
        first_death = p.death_rounds[0] if p.death_rounds else "-"
        lines.append(
            f"{m.match_id:<11} {p.rank:>4}  {p.final_mass:>7.1f}  {p.peak_mass:>6.1f}"
            f"  {p.splits_cmd:>6}  {p.foods:>5}  {p.deaths:>6}  {str(first_death):>9}"
            f"  {p.captures:>4}  {p.shatters:>8}  {w.team_id:>8}  {w.final_mass:>9.1f}"
            f"  {w.splits_cmd:>10}  {w.foods:>9}"
        )
    if ranks:
        lines.append(
            f"-- team {team}: mean rank {sum(ranks) / len(ranks):.2f} over "
            f"{len(ranks)} matches, {sum(1 for r in ranks if r == 1)} wins"
        )
    return "\n".join(lines)


def meta_table(teams: Dict[int, TeamAgg]) -> str:
    lines = [
        "team  matches  mean_rank  wins  splits/m  eff/m  caps/m  deaths/m"
        "  foods/m  final_m/m  peak_m/m  med_pre_ratio  bc_rows"
    ]
    for t in sorted(teams.values(), key=lambda t: t.mean_rank):
        m = t.matches
        lines.append(
            f"{t.team_id:>4}  {m:>7}  {t.mean_rank:>9.2f}  {t.wins:>4}"
            f"  {t.splits_cmd / m:>8.1f}  {t.splits_eff / m:>5.1f}"
            f"  {t.captures / m:>6.1f}  {t.deaths / m:>8.1f}"
            f"  {t.foods / m:>7.0f}  {t.final_mass_sum / m:>9.1f}"
            f"  {t.peak_mass_sum / m:>8.1f}  {_median(t.ratios_pre):>13.2f}"
            f"  {t.moves:>7}"
        )
    return "\n".join(lines)


def growth_table(matches: Sequence[MatchStats], team: int) -> str:
    """Mean mass of our team vs the eventual winner at timeline checkpoints."""
    checkpoints: Dict[int, Tuple[List[float], List[float]]] = {}
    for m in matches:
        pid = m.pid_of_team(team)
        if pid is None:
            continue
        for rnd, masses in m.timeline.items():
            us, win = checkpoints.setdefault(rnd, ([], []))
            us.append(masses[pid])
            win.append(masses[m.winner_pid])
    lines = ["round   us_mean  winner_mean   n"]
    for rnd in sorted(checkpoints):
        us, win = checkpoints[rnd]
        lines.append(
            f"{rnd:>5}  {sum(us) / len(us):>8.2f}  {sum(win) / len(win):>11.2f}"
            f"  {len(us):>3}"
        )
    return "\n".join(lines)


def to_json_dict(
    matches: Sequence[MatchStats],
    quarantined: Sequence[Tuple[str, str]],
    team: Optional[int],
) -> dict:
    return {
        "our_team": team,
        "quarantined": [{"source": s, "reason": r} for s, r in quarantined],
        "matches": [
            {
                "source": m.source,
                "match_id": m.match_id,
                "engine_version": m.engine_version,
                "rounds": m.rounds,
                "winner_pid": m.winner_pid,
                "winner_team": m.team_of(m.winner_pid),
                "warnings": m.warnings,
                "players": [
                    {
                        "player_id": p.player_id,
                        "team_id": p.team_id,
                        "rank": p.rank,
                        "final_mass": p.final_mass,
                        "peak_mass": p.peak_mass,
                        "mean_mass": p.mass_sum / m.rounds if m.rounds else 0.0,
                        "splits_cmd": p.splits_cmd,
                        "splits_eff": p.splits_eff,
                        "captures": p.captures,
                        "capture_ratios_pre": p.capture_ratios_pre,
                        "capture_ratios_post": p.capture_ratios_post,
                        "blobs_lost": p.blobs_lost,
                        "deaths": p.deaths,
                        "death_rounds": p.death_rounds,
                        "foods": p.foods,
                        "food_events": p.food_events,
                        "shatters": p.shatters,
                        "shatter_pieces": p.shatter_pieces,
                        "moves": p.moves,
                        "alive_rounds": p.alive_rounds,
                        "max_blobs": p.max_blobs,
                    }
                    for p in sorted(m.players.values(), key=lambda p: p.player_id)
                ],
                "timeline": {
                    str(rnd): {str(pid): mass for pid, mass in masses.items()}
                    for rnd, masses in sorted(m.timeline.items())
                },
            }
            for m in matches
        ],
    }


# --- CLI ------------------------------------------------------------------


def collect_paths(args_paths: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    sources = args_paths or [str(REPO / "match_replays")]
    for s in sources:
        p = Path(s)
        if p.is_dir():
            paths.extend(sorted(p.glob("match-*-replay.json")) or sorted(p.glob("*.json")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[replay_stats] no such path: {p}", file=sys.stderr)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*",
                    help="replay files or directories (default: match_replays/)")
    ap.add_argument("--team", type=int, default=OUR_TEAM_DEFAULT,
                    help=f"our team_id (default {OUR_TEAM_DEFAULT})")
    ap.add_argument("--json", type=Path, default=None,
                    help="write full per-match stats to this JSON file")
    ap.add_argument("--timeline-step", type=int, default=100,
                    help="record mass timeline every N rounds (default 100)")
    ap.add_argument("--min-matches", type=int, default=3,
                    help="clone candidates need >= this many matches (default 3)")
    ap.add_argument("--allow-version", action="append", default=[],
                    help="treat this engine version as compatible (repeatable)")
    args = ap.parse_args(argv)

    allowed = COMPATIBLE_ENGINE_VERSIONS | set(args.allow_version)
    paths = collect_paths(args.paths)
    if not paths:
        print("[replay_stats] no replay files found", file=sys.stderr)
        return 1

    matches: List[MatchStats] = []
    quarantined: List[Tuple[str, str]] = []
    for path in paths:
        try:
            m = load_replay(path, timeline_step=args.timeline_step)
        except (ReplayFormatError, json.JSONDecodeError, KeyError, OSError) as ex:
            quarantined.append((str(path), f"unparseable: {ex}"))
            continue
        if m.engine_version not in allowed:
            quarantined.append(
                (str(path),
                 f"engine {m.engine_version} not in compatible set "
                 f"(use --allow-version {m.engine_version} to override)")
            )
            continue
        matches.append(m)

    print(f"=== replay_stats: {len(matches)} parsed, "
          f"{len(quarantined)} quarantined ===")
    for src, reason in quarantined:
        print(f"  QUARANTINED {src}: {reason}")
    for m in matches:
        for w in m.warnings:
            print(f"  WARNING {m.match_id}: {w}")
    if not matches:
        return 1

    versions = sorted({m.engine_version for m in matches})
    print(f"engine versions: {', '.join(versions)}\n")

    print(f"--- team {args.team}, per match ---")
    print(our_match_table(matches, args.team))

    teams = aggregate_teams(matches)
    print("\n--- meta: per team, per-match averages "
          "(bc_rows = total move commands = BC data volume) ---")
    print(meta_table(teams))

    cands = clone_candidates(teams, args.team, args.min_matches)
    print(f"\n--- clone candidates (>= {args.min_matches} matches, "
          f"sorted by mean rank, team {args.team} excluded) ---")
    for t in cands[:8]:
        print(
            f"  team {t.team_id}: mean rank {t.mean_rank:.2f} in {t.matches} "
            f"matches, {t.wins} wins, {t.splits_eff / t.matches:.1f} effective "
            f"splits/m, {t.moves} bc_rows"
        )

    print(f"\n--- growth: team {args.team} vs eventual winners ---")
    print(growth_table(matches, args.team))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(to_json_dict(matches, quarantined, args.team), f, indent=1)
        print(f"\n[replay_stats] wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
