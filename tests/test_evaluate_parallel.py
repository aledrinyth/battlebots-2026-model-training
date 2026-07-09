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
