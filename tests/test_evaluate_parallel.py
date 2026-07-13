"""Parallel evaluation (--workers): partitioning + determinism.

The fast tests cover the partition logic and the seeded random baseline; the
worker-count-invariance test plays real games, so it is ``slow``-marked (run
with ``--runslow`` / ``RUN_SLOW=1``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from train.harness.evaluate import _load_policy, _strided_chunks, evaluate  # noqa: E402


def test_strided_chunks_partition_exactly():
    for n, w in [(30, 8), (5, 8), (1, 4), (100, 16), (7, 1)]:
        chunks = _strided_chunks(n, w)
        assert all(chunks), (n, w)
        flat = sorted(i for c in chunks for i in c)
        assert flat == list(range(n)), (n, w)
        assert len(chunks) <= max(1, w)


def test_random_baseline_is_seeded():
    a = _load_policy(None, torch.device("cpu"), seed=123)
    b = _load_policy(None, torch.device("cpu"), seed=123)
    pa, pb = list(a.parameters())[0], list(b.parameters())[0]
    assert torch.equal(pa, pb)
    c = _load_policy(None, torch.device("cpu"), seed=124)
    assert not torch.equal(pa, list(c.parameters())[0])


@pytest.mark.slow
def test_workers_do_not_change_results():
    seq = evaluate(None, opponents=("dont_move",), n_games=2, workers=1)
    par = evaluate(None, opponents=("dont_move",), n_games=2, workers=2)
    assert seq.mean_final_mass == par.mean_final_mass
    assert seq.rank_hist == par.rank_hist


def test_scripted_subject_excludes_weights():
    with pytest.raises(ValueError, match="not both"):
        evaluate("some_weights.pt", n_games=1, scripted_subject="greedy_food")


@pytest.mark.slow
def test_scripted_subject_plays_and_is_worker_invariant():
    """The anchor mode: a scripted bot in the subject seat, no policy at all.

    greedy_food in a dont_move field must eat its way to rank 1 in both the
    sequential and parallel paths, with identical results (same seeds/seats).
    """
    seq = evaluate(None, opponents=("dont_move",), n_games=2, workers=1,
                   scripted_subject="greedy_food")
    par = evaluate(None, opponents=("dont_move",), n_games=2, workers=2,
                   scripted_subject="greedy_food")
    assert seq.mean_final_mass == par.mean_final_mass
    assert seq.rank_hist == par.rank_hist
    assert seq.mean_rank == 1.0  # the only eater in a stationary field
    assert seq.mean_final_mass > 3.0  # it actually grew
