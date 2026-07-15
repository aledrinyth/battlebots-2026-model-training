"""P6.3 acceptance: the promotion gate never promotes noise.

Playbook self-tests, run against an injected game player (fast, exact) plus a
slow real-game identity check:

* incumbent-vs-itself does **not** promote (identical seeds + identical
  weights ⇒ every paired diff is exactly 0 ⇒ the CI is {0} ⇒ cap ⇒ keep);
* incumbent-vs-deliberately-crippled promotes the healthy one (and the
  mirrored comparison rejects early — the futility exit).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import promote as promote_mod  # noqa: E402
from promote import paired_verdict, promote  # noqa: E402


# --- decision math -----------------------------------------------------------

def test_verdict_all_zero_diffs_never_promotes() -> None:
    v = paired_verdict([0.0] * 400)
    assert v["decision"] == "continue"
    assert v["ci_lo"] == v["ci_hi"] == 0.0


def test_verdict_promotes_clear_improvement_and_rejects_regression() -> None:
    rng = random.Random(0)
    better = [rng.choice([-2, -1, -1, 0, 1]) for _ in range(300)]  # mean ≈ −0.6
    worse = [-d for d in better]
    assert paired_verdict(better)["decision"] == "promote"
    assert paired_verdict(worse)["decision"] == "reject"


def test_verdict_is_deterministic() -> None:
    d = [random.Random(1).uniform(-1, 1) for _ in range(100)]
    assert paired_verdict(d, seed=7) == paired_verdict(d, seed=7)


# --- the sequential gate with an injected player -------------------------------

def _fake_players(edge: float):
    """A pair-aware fake: incumbent ranks are seeded noise; the candidate's
    rank in the SAME game is shifted by ``edge`` (clipped to 1..8)."""

    def play(weights, opponents, n, device, seed0, workers):
        rng = random.Random(seed0)
        base = [(g, rng.randint(1, 8), 1.0) for g in range(n)]
        if weights == "candidate.pt":
            return [(g, max(1, min(8, round(r + edge))), m) for g, r, m in base]
        return base

    return play


def test_gate_promotes_genuinely_better_candidate() -> None:
    res = promote("candidate.pt", "incumbent.pt", block=100, cap=2000,
                  play_games_fn=_fake_players(edge=-1.0))
    assert res["promoted"]
    assert res["games_paired"] == 100  # decisive in the first block
    assert res["mean_rank_diff"] < 0


def test_gate_rejects_worse_candidate_early() -> None:
    res = promote("candidate.pt", "incumbent.pt", block=100, cap=2000,
                  play_games_fn=_fake_players(edge=+1.0))
    assert not res["promoted"]
    assert res["decision"] == "reject"
    assert res["games_paired"] == 100  # futility exit, not a 2000-game burn


def test_gate_ties_run_to_cap_and_keep_incumbent() -> None:
    res = promote("candidate.pt", "incumbent.pt", block=50, cap=200,
                  play_games_fn=_fake_players(edge=0.0))
    assert not res["promoted"]
    assert res["decision"] == "no_promotion_at_cap"
    assert res["games_paired"] == 200


def test_gate_pairs_by_game_index() -> None:
    calls = []

    def spy(weights, opponents, n, device, seed0, workers):
        calls.append((weights, n, seed0))
        return [(g, 4, 1.0) for g in range(n)]

    promote("candidate.pt", "incumbent.pt", block=10, cap=20, play_games_fn=spy)
    # Each block: both subjects, same n, same seed window; blocks don't overlap.
    assert calls[0][1:] == calls[1][1:]
    assert calls[2][1:] == calls[3][1:]
    assert calls[0][2] != calls[2][2]


# --- real games: identical weights ⇒ never promote (slow) ------------------------

@pytest.mark.slow
def test_incumbent_vs_itself_does_not_promote_real_games() -> None:
    pytest.importorskip("torch")
    # weights=None on both sides ⇒ the same seeded random-baseline policy in
    # the same seeded games: byte-identical pairs, all diffs exactly 0.
    res = promote(None, None, opponents=("dont_move",), block=2, cap=4)
    assert not res["promoted"]
    assert res["decision"] == "no_promotion_at_cap"
    assert res["mean_rank_diff"] == 0.0
