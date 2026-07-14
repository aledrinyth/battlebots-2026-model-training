"""P3.5 Track D — expert demo collector + BC warm-start + init_weights seam.

The collector records the real featurizer/flat_obs path with the expert's
quantized action executed; the BC trainer must be able to overfit a tiny slice
(wiring sanity, same spirit as P3.3's overfit smoke); ``RunConfig.init_weights``
must load BC weights into a fresh run and carry them into its checkpoints.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from bot.adapters.act_v1 import HEADING_BINS, bin_to_vector
from train.harness.demos import collect_demos, direction_to_bin
from train.harness.flat_obs import OBS_WIDTH


def test_direction_to_bin_roundtrips_every_bin():
    for b in range(HEADING_BINS):
        x, y = bin_to_vector(b)
        got, ok = direction_to_bin(x, y)
        assert ok and got == b, (b, got)


def test_direction_to_bin_stay_is_masked():
    got, ok = direction_to_bin(0.0, 0.0)
    assert not ok and got == 0


@pytest.fixture(scope="module")
def tiny_demos():
    # One short food-only game through the real Actor/featurizer path.
    return collect_demos(games=1, base_seed=99_991, max_rounds=160)


def test_collector_shapes_and_labels(tiny_demos):
    d = tiny_demos
    n = len(d["obs"])
    assert n == 160
    assert d["obs"].shape == (n, OBS_WIDTH) and d["obs"].dtype == np.float32
    assert d["heading"].shape == (n,) and d["heading"].min() >= 0
    assert d["heading"].max() < HEADING_BINS
    assert d["is_reset"][0] and not d["is_reset"][1:].any()  # no deaths food-only
    # greedy_food almost always sees food => labeled headings nearly everywhere
    assert d["label_ok"].mean() > 0.9
    # the observation stream is alive (not zeros) and headings vary (not constant)
    assert np.abs(d["obs"]).sum() > 0
    assert len(np.unique(d["heading"][d["label_ok"]])) > 3


def test_bc_overfits_tiny_dataset(tiny_demos):
    torch = pytest.importorskip("torch")
    from train.bc import _run_window, train_bc

    policy, metrics = train_bc(tiny_demos, epochs=30, window=8, batch_windows=4,
                               lr=2e-3, val_frac=0.1, seed=0, log=False)
    # The overfit smoke proves the recurrent BC wiring can FIT the expert (the
    # P3.3-style sanity): train-set heading accuracy far above 24-way chance
    # (~4%). Generalization (val_acc) is only meaningful on the real dataset —
    # 160 rows against 555k params just memorizes.
    d = tiny_demos
    starts = np.arange(0, len(d["obs"]) - 8 + 1, 8, dtype=np.int64)
    with torch.no_grad():
        hls, _ = _run_window(policy, d["obs"], d["is_reset"], starts, 8,
                             torch.device("cpu"))
    correct = total = 0
    for t in range(8):
        rows = starts + t
        ok = d["label_ok"][rows]
        pred = hls[t].argmax(-1).numpy()
        correct += int((pred[ok] == d["heading"][rows][ok]).sum())
        total += int(ok.sum())
    assert correct / max(total, 1) > 0.5, (correct, total, metrics)


def test_bc_checkpoint_loads_in_evaluate_and_init_weights(tiny_demos, tmp_path):
    torch = pytest.importorskip("torch")
    from train.bc import save_bc_checkpoint, train_bc
    from train.harness.evaluate import _load_policy
    from train.harness.run import RunConfig, run_from_config

    policy, metrics = train_bc(tiny_demos, epochs=1, window=16, batch_windows=16,
                               seed=0, log=False)
    # plant a recognizable value so we can trace the weights through the run
    with torch.no_grad():
        list(policy.parameters())[0].fill_(0.123)
    ckpt = str(tmp_path / "bc.pt")
    save_bc_checkpoint(policy, metrics, ckpt, data_path="tiny")

    # 1) evaluate can load it (the G1 measurement path)
    loaded = _load_policy(ckpt, torch.device("cpu"))
    assert torch.allclose(list(loaded.parameters())[0],
                          torch.full_like(list(loaded.parameters())[0], 0.123))

    # 2) a fresh run warm-starts from it; with lr=0 the weights must survive
    #    training verbatim into the run's own checkpoint
    rc = RunConfig(name="ws", out_dir=str(tmp_path), total_updates=1,
                   init_weights=ckpt,
                   train={"num_actors": 1, "policy_seats": [0],
                          "scripted_names": ["dont_move"], "rollout_len": 2,
                          "lr": 0.0, "base_seed": 7, "barrier_timeout_s": 60})
    run_from_config(rc)
    blob = torch.load(os.path.join(str(tmp_path), "ws", "checkpoints",
                                   "ckpt_v000001.pt"),
                      map_location="cpu", weights_only=False)
    first_key = next(iter(blob["state_dict"]))
    assert torch.allclose(blob["state_dict"][first_key],
                          torch.full_like(blob["state_dict"][first_key], 0.123))


def test_collector_workers_are_byte_identical():
    """--workers must not change the dataset: same games, same order, same rows.

    Uses the mixed-opponent field (the Stage-1 collection shape) so the
    parallel path is exercised on exactly what it was built to speed up.
    """
    kw = dict(expert="cautious", games=4, base_seed=77_701, max_rounds=40,
              opponents=("greedy_food", "cautious", "chaser", "coward"))
    seq = collect_demos(workers=1, **kw)
    par = collect_demos(workers=3, **kw)
    assert seq.keys() == par.keys()
    for k in seq:
        assert np.array_equal(seq[k], par[k]), f"field {k} differs across workers"
    assert len(seq["final_masses"]) == 4
