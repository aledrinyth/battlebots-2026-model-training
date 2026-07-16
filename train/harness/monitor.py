"""P6.4 — league-run monitoring: metric aggregation + collapse-signature alerts.

Pure functions over the run's on-disk artifacts (``metrics.jsonl``, the pool's
``league.json``, the per-actor ``results/*.jsonl``) — no torch, no engine, no
new dependencies — so every threshold is unit-testable on synthetic records and
``tools/dashboard.py`` is just formatting.

The playbook's collapse signatures (P6.4) and their detectors:

* **entropy floor** — the policy has gone (near-)deterministic: trailing-window
  mean entropy under ``entropy_floor``. A collapsed policy stops exploring and
  the league stops learning; critical.
* **win-rate ~100% vs pool** — every probed pool entry sits above
  ``domination_p``: self-play has outrun its own history, opponents teach
  nothing (the P6.5 exploiter trigger). PFSP already down-weights dominated
  entries, so this fires only when there is *nothing informative left*.
* **mass plateau** — trailing game-mass mean stopped improving vs the previous
  window. Expected near convergence — a warning to look, not an abort.

Rank convention throughout: 1 = winner, 8 = first eaten; lower is better.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Loaders (tolerant: a live run is mid-write somewhere at any moment)          #
# --------------------------------------------------------------------------- #


def load_jsonl(path: str | Path, tail_bytes: int = 0) -> List[dict]:
    """Records from a JSONL file; torn/partial lines are skipped silently.

    ``tail_bytes > 0`` reads only the file's last chunk (dashboard use on the
    growing results files); the first line of a mid-file chunk is dropped as
    likely-partial unless the chunk is the whole file.
    """
    p = Path(path)
    if not p.exists():
        return []
    size = p.stat().st_size
    with open(p, "rb") as fh:
        if tail_bytes and size > tail_bytes:
            fh.seek(size - tail_bytes)
            blob = fh.read()
            blob = blob[blob.find(b"\n") + 1:]  # drop the partial first line
        else:
            blob = fh.read()
    out: List[dict] = []
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_league_entries(league_dir: str | Path) -> Dict[str, dict]:
    """The pool registry's entries (``{}`` if the pool doesn't exist yet)."""
    reg = Path(league_dir) / "league.json"
    if not reg.exists():
        return {}
    try:
        return dict(json.loads(reg.read_text())["entries"])
    except Exception:
        return {}


def load_results(league_dir: str | Path, tail_bytes: int = 262_144) -> List[dict]:
    """Recent game-result records across every ``results/*.jsonl`` file."""
    rdir = Path(league_dir) / "results"
    if not rdir.is_dir():
        return []
    out: List[dict] = []
    for path in sorted(rdir.glob("*.jsonl")):
        out.extend(load_jsonl(path, tail_bytes=tail_bytes))
    return out


# --------------------------------------------------------------------------- #
# Result-record vocabulary (mirrors league_seats.GameResult, dict-shaped)      #
# --------------------------------------------------------------------------- #


def _rank_of(rec: dict) -> Dict[int, int]:
    return {int(pid): i + 1 for i, pid in enumerate(rec["rankings"])}


def _seats_by_kind(rec: dict) -> Tuple[List[int], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """(policy pids, [(pid, anchor name)], [(pid, snapshot_id)])."""
    policy, anchors, pool = [], [], []
    for pid_s, spec in rec["plan"].items():
        pid = int(pid_s)
        if spec == "policy":
            policy.append(pid)
        elif spec.startswith("anchor:") or spec.startswith("scripted:"):
            anchors.append((pid, spec.split(":", 1)[1]))
        elif spec.startswith("pool:"):
            pool.append((pid, spec.split(":", 1)[1]))
    return sorted(policy), sorted(anchors), sorted(pool)


def summarize_results(records: Sequence[dict]) -> Dict[str, object]:
    """Training-field aggregates over game results (the dashboard's league view).

    Returns mean policy rank/mass, per-anchor pairwise win-rate (policy seat
    outranks that anchor seat), per-snapshot pairwise win-rate, and the observed
    anchor/pool seat share.
    """
    n = 0
    rank_sum = mass_sum = seats_n = 0.0
    anchor_pairs: Dict[str, List[int]] = {}
    pool_pairs: Dict[str, List[int]] = {}
    seat_counts = {"anchor": 0, "pool": 0}
    win_games = 0
    for rec in records:
        try:
            rank_of = _rank_of(rec)
            policy, anchors, pool = _seats_by_kind(rec)
            masses = {int(k): float(v) for k, v in rec["final_masses"].items()}
        except Exception:
            continue
        if not policy:
            continue
        n += 1
        for p in policy:
            rank_sum += rank_of[p]
            mass_sum += masses.get(p, 0.0)
            seats_n += 1
        if min(rank_of[p] for p in policy) == 1:
            win_games += 1
        for pid_a, name in anchors:
            seat_counts["anchor"] += 1
            wins = anchor_pairs.setdefault(name, [])
            wins.extend(int(rank_of[p] < rank_of[pid_a]) for p in policy)
        for pid_e, sid in pool:
            seat_counts["pool"] += 1
            wins = pool_pairs.setdefault(sid, [])
            wins.extend(int(rank_of[p] < rank_of[pid_e]) for p in policy)
    out: Dict[str, object] = {
        "games": n,
        "policy_mean_rank": (rank_sum / seats_n) if seats_n else None,
        "policy_mean_mass": (mass_sum / seats_n) if seats_n else None,
        "policy_game_winrate": (win_games / n) if n else None,
        "vs_anchor": {k: sum(v) / len(v) for k, v in sorted(anchor_pairs.items()) if v},
        "vs_pool": {k: sum(v) / len(v) for k, v in sorted(pool_pairs.items()) if v},
        "seat_share": seat_counts,
    }
    return out


def policy_mass_series(records: Sequence[dict]) -> List[float]:
    """Per-game mean policy-seat final mass, in record order (plateau input)."""
    series: List[float] = []
    for rec in records:
        try:
            policy, _, _ = _seats_by_kind(rec)
            masses = {int(k): float(v) for k, v in rec["final_masses"].items()}
        except Exception:
            continue
        if policy:
            series.append(sum(masses.get(p, 0.0) for p in policy) / len(policy))
    return series


# --------------------------------------------------------------------------- #
# Alerts                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Alert:
    name: str          # "entropy_floor" | "pool_domination" | "mass_plateau"
    severity: str      # "critical" | "warning"
    message: str


@dataclass(frozen=True)
class AlertConfig:
    """Collapse-signature thresholds (defaults sized for the S2 league run)."""

    entropy_floor: float = 0.05     # nats; uniform-16-bin heading is ~2.77
    entropy_window: int = 30        # consecutive updates below the floor
    domination_p: float = 0.95      # EMA win-rate that counts as "solved"
    domination_min_games: int = 30  # only entries probed this often count
    plateau_window: int = 200       # games per comparison half
    plateau_min_gain: float = 0.0   # newer half must beat older by > this frac


def check_entropy_floor(entropy: Sequence[float], cfg: AlertConfig = AlertConfig()) -> Optional[Alert]:
    """Trailing-window mean entropy under the floor ⇒ policy collapse."""
    if len(entropy) < cfg.entropy_window:
        return None
    tail = list(entropy)[-cfg.entropy_window:]
    mean = sum(tail) / len(tail)
    if mean < cfg.entropy_floor:
        return Alert(
            "entropy_floor", "critical",
            f"mean entropy {mean:.4f} < {cfg.entropy_floor} over the last "
            f"{cfg.entropy_window} updates — policy near-deterministic "
            f"(raise entropy_coef / roll back / launch an exploiter)")
    return None


def check_pool_domination(entries: Mapping[str, Mapping], cfg: AlertConfig = AlertConfig()) -> Optional[Alert]:
    """Every well-probed pool entry beaten ≥ ``domination_p`` ⇒ pool is inert."""
    probed = {sid: e for sid, e in entries.items()
              if int(e.get("games", 0)) >= cfg.domination_min_games}
    if not probed:
        return None
    ps = {sid: float(e["p_current"]) for sid, e in probed.items()}
    if min(ps.values()) >= cfg.domination_p:
        worst = min(ps, key=ps.get)
        return Alert(
            "pool_domination", "warning",
            f"all {len(ps)} probed pool entries beaten at ≥ {cfg.domination_p:.0%} "
            f"(closest: {worst} at {ps[worst]:.0%}) — self-play has outrun its "
            f"history; snapshot more often or launch an exploiter (P6.5)")
    return None


def check_mass_plateau(masses: Sequence[float], cfg: AlertConfig = AlertConfig()) -> Optional[Alert]:
    """Newer window's mean game mass no better than the older's ⇒ plateau."""
    w = cfg.plateau_window
    if len(masses) < 2 * w:
        return None
    older = list(masses)[-2 * w:-w]
    newer = list(masses)[-w:]
    mo = sum(older) / w
    mn = sum(newer) / w
    if mo > 0 and (mn - mo) / mo <= cfg.plateau_min_gain:
        return Alert(
            "mass_plateau", "warning",
            f"mean policy game mass {mn:.2f} vs {mo:.2f} over the previous "
            f"{w} games (Δ {100 * (mn - mo) / mo:+.1f}%) — no longer improving; "
            f"check vs_anchor trends before intervening")
    return None


def check_alerts(
    metrics_records: Sequence[dict],
    league_entries: Mapping[str, Mapping],
    game_masses: Sequence[float],
    cfg: AlertConfig = AlertConfig(),
) -> List[Alert]:
    """All collapse-signature checks over a run's current artifacts."""
    entropy = [float(r["entropy"]) for r in metrics_records if "entropy" in r]
    alerts = [
        check_entropy_floor(entropy, cfg),
        check_pool_domination(league_entries, cfg),
        check_mass_plateau(game_masses, cfg),
    ]
    return [a for a in alerts if a is not None]


__all__ = [
    "Alert", "AlertConfig",
    "check_alerts", "check_entropy_floor", "check_pool_domination", "check_mass_plateau",
    "load_jsonl", "load_league_entries", "load_results",
    "summarize_results", "policy_mass_series",
]
