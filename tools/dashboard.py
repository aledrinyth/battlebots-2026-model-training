"""P6.4 — league-run dashboard: one screen of run health + collapse alerts.

Text-only (no plotting deps; the box is headless and this must work over ssh):

    python tools/dashboard.py runs/stage2_league            # one report
    python tools/dashboard.py runs/stage2_league --watch 60 # refresh loop
    python tools/dashboard.py runs/stage2_league --json     # machine-readable

Reads the run's on-disk artifacts only (``metrics.jsonl``, ``league/league.json``,
``league/results/*.jsonl``) — safe to run against a live training run from a
second shell; every loader tolerates mid-write files. Exit code 2 if any
critical alert is firing (cron-able: ``dashboard.py <run> || notify``).

The playbook P6.4 dashboard items and where they land here:
Elo-vs-anchors → per-anchor pairwise win-rates over the recent-games window
(true fixed-ladder Elo is P7.1); mass curves → trailing game-mass windows;
death rates / split usage / entropy → trailing metrics.jsonl means; alert
thresholds → ``train.harness.monitor.check_alerts``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from train.harness.monitor import (  # noqa: E402
    AlertConfig, check_alerts, load_jsonl, load_league_entries, load_results,
    policy_mass_series, summarize_results,
)


def _mean(rows: List[dict], key: str, last: int) -> Optional[float]:
    vals = [float(r[key]) for r in rows[-last:] if key in r]
    return sum(vals) / len(vals) if vals else None


def _fmt(v: Optional[float], spec: str = ".3f", none: str = "—") -> str:
    return none if v is None else format(v, spec)


def collect(run_dir: str, window: int, tail_kb: int) -> Dict[str, object]:
    """Everything the report needs, as one JSON-able dict."""
    metrics = load_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    league_dir = os.path.join(run_dir, "league")
    entries = load_league_entries(league_dir)
    results = load_results(league_dir, tail_bytes=tail_kb * 1024)
    results_summary = summarize_results(results[-window:])
    masses = policy_mass_series(results)
    alerts = check_alerts(metrics, entries, masses)

    last = metrics[-1] if metrics else {}
    n_recent = min(len(metrics), 50)
    report: Dict[str, object] = {
        "run_dir": run_dir,
        "updates": int(last.get("update", 0)),
        "wall_h": round(float(last.get("wall_s", 0.0)) / 3600.0, 2),
        "training": {
            k: _mean(metrics, k, n_recent)
            for k in ("loss", "entropy", "approx_kl", "mean_reward",
                      "eats", "deaths", "split_frac")
        },
        "league": {
            "pool_size": len(entries),
            "games_total": int(last.get("league_games_total", 0)),
            "entries": {
                sid: {"version": e.get("version"), "games": e.get("games"),
                      "p_current": round(float(e.get("p_current", 0.5)), 3)}
                for sid, e in sorted(entries.items())
            },
        },
        "recent_games": results_summary,
        "alerts": [{"name": a.name, "severity": a.severity, "message": a.message}
                   for a in alerts],
    }
    return report


def render(rep: Dict[str, object]) -> str:
    lines: List[str] = []
    add = lines.append
    add(f"══ league dashboard ══ {rep['run_dir']}  "
        f"(update {rep['updates']}, {rep['wall_h']} h)")

    t = rep["training"]
    add("── training (last ≤50 updates) ──")
    add(f"  entropy {_fmt(t['entropy'])}   kl {_fmt(t['approx_kl'], '+.4f')}   "
        f"loss {_fmt(t['loss'], '+.4f')}   mean_reward {_fmt(t['mean_reward'], '+.5f')}")
    add(f"  eats/rollout {_fmt(t['eats'], '.1f')}   deaths/rollout {_fmt(t['deaths'], '.1f')}   "
        f"split_frac {_fmt(t['split_frac'], '.4f')}")

    lg = rep["league"]
    add(f"── league pool (size {lg['pool_size']}, games folded {lg['games_total']}) ──")
    if lg["entries"]:
        add("  snapshot        ver    games  p_current")
        for sid, e in lg["entries"].items():
            add(f"  {sid:<14} {str(e['version']):>4} {str(e['games']):>8}  "
                f"{e['p_current']:.3f}")
    else:
        add("  (empty — anchors fill every opponent seat)")

    rg = rep["recent_games"]
    add(f"── recent games (window {rg['games']}) ──")
    add(f"  policy mean rank {_fmt(rg['policy_mean_rank'], '.2f')}   "
        f"mean mass {_fmt(rg['policy_mean_mass'], '.2f')}   "
        f"game win-rate {_fmt(rg['policy_game_winrate'], '.1%')}")
    if rg["vs_anchor"]:
        vs = "   ".join(f"{k} {v:.0%}" for k, v in rg["vs_anchor"].items())
        add(f"  beats-anchor (pairwise): {vs}")
    if rg["vs_pool"]:
        vs = "   ".join(f"{k} {v:.0%}" for k, v in rg["vs_pool"].items())
        add(f"  beats-pool (pairwise):   {vs}")
    share = rg.get("seat_share", {})
    tot = sum(share.values()) or 1
    add(f"  opponent seat share: anchor {share.get('anchor', 0) / tot:.0%} / "
        f"pool {share.get('pool', 0) / tot:.0%}")

    alerts = rep["alerts"]
    add("── alerts ──")
    if not alerts:
        add("  none — no collapse signature firing")
    for a in alerts:
        add(f"  [{a['severity'].upper()}] {a['name']}: {a['message']}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="P6.4 league-run dashboard")
    p.add_argument("run_dir", help="run directory, e.g. runs/stage2_league")
    p.add_argument("--window", type=int, default=500,
                   help="recent-games window for rank/anchor stats")
    p.add_argument("--tail-kb", type=int, default=1024,
                   help="bytes of each results file to read (from the end)")
    p.add_argument("--watch", type=int, default=0,
                   help="refresh every N seconds (0 = print once)")
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = p.parse_args()

    while True:
        rep = collect(args.run_dir, args.window, args.tail_kb)
        if args.json:
            print(json.dumps(rep, indent=2, sort_keys=True))
        else:
            if args.watch:
                print("\033[2J\033[H", end="")  # clear screen between refreshes
            print(render(rep), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)

    critical = any(a["severity"] == "critical" for a in rep["alerts"])
    return 2 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
