#!/usr/bin/env bash
# Track 1 (box side) — ingest full corpus, train 5 per-team meta clones in
# parallel, export them, and measure the s1_champion baseline against the
# 5-clone meta field. Run inside ~/battlebots on the box, in a tmux session,
# AFTER the two laptop->box rsyncs (code + match_replays/ + ckpt_v001470.pt).
set -euo pipefail
cd ~/battlebots
mkdir -p runs data/clones

echo "=== 0. sanity: the frozen-opponent seam + dataset path exist on the box ==="
.venv-train/bin/python -m pytest tests/test_frozen_opponent.py tests/test_replay_dataset.py -q

echo "=== 1. ingest the full corpus (idempotent; skips the 20 already done) ==="
.venv-train/bin/python -m train.replay ingest --out data/replay_shards 2>&1 | tee runs/ingest_full.log

echo "=== 2. per-team datasets (all episodes of each team) ==="
for T in 49 14 37 1 88; do
  .venv-train/bin/python -m train.replay bc-dataset --shards data/replay_shards \
    --out data/bc_team_$T.npz --teams $T --min-rank 8
done

echo "=== 3. train 5 clones in parallel (OMP=16 each -> 80 threads) ==="
for T in 49 14 37 1 88; do
  OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nohup .venv-train/bin/python -m train.bc \
    --data data/bc_team_$T.npz --out data/clone_team_$T.pt \
    --epochs 40 --window 64 --keep-best --split-pos-weight 4 \
    > runs/clone_team_$T.log 2>&1 &
done
wait
echo "clone training done; final val lines:"
for T in 49 14 37 1 88; do echo -n "team $T: "; grep -E 'restored best|val_heading' runs/clone_team_$T.log | tail -1; done

echo "=== 4. export each clone -> actor npz ==="
for T in 49 14 37 1 88; do
  .venv-train/bin/python -m train.export --ckpt data/clone_team_$T.pt --out data/clones/team_$T.npz
done

echo "=== 5. baseline: s1_champion vs the 5-clone meta field (200 games) ==="
.venv-train/bin/python -m train.harness.evaluate --weights data/ckpt_v001470.pt \
  --opponents frozen:data/clones/team_49.npz frozen:data/clones/team_14.npz \
              frozen:data/clones/team_37.npz frozen:data/clones/team_1.npz \
              frozen:data/clones/team_88.npz \
  --games 200 --workers 64 2>&1 | tee runs/baseline_champion_vs_meta.log

echo "=== DONE. rsync data/clones/ and runs/ back to the laptop. ==="
