"""P3.1 harness spike — the custom actor–learner substrate skeleton.

This package is the concrete artifact behind the P3.1 decision (Sample Factory
vs custom; see ``notes/P3.1-harness-spike.md``). It wires the in-process
:class:`~train.env.wrapper.WrapperEnv`, the shared
:class:`~bot.featurizer.Featurizer`, and the shared
:class:`~bot.adapters.act_v1.ActionAdapter` into a minimal end-to-end loop:

    WrapperEnv --obs--> Featurizer --CanonicalObs--> collate --> SkeletonPolicy
        ^                                                              |
        |                                                            logits
        +---- MovePlayer <-- ActionAdapter <-- CanonicalAction <------+

Actors are **numpy/CPU-only** (no torch import) — faithful to the shipped
design, where the match-legal surface is numpy and only the *central* learner
touches torch. The policy stub here is a throwaway wiring model; P3.3 replaces
it with the real recurrent PPO network + asymmetric critic.

Nothing here is on the match-legal path (it lives under ``train/``), so torch is
allowed. Import-linter forbids the reverse edge (``bot`` importing ``train`` or
``torch``), which keeps the boundary honest.

P3.2 promotes the in-process skeleton to a real multiprocess actor↔learner:

    vec_actor (N processes, numpy)  --shm slab-->  learner (1 process, torch/GPU)
        flat_obs.flatten                              collate.build_batched_obs
        shm.SlabSet (obs/meta/act)                    rollout.RolloutBuffer + GAE
        Δmass reward, death gaps                      ppo_update (clip/γ/λ/BPTT)

``skeleton``/``actor`` remain as the P3.1 in-process proof; ``vec_actor`` +
``learner`` + ``shm`` + ``flat_obs`` + ``rollout`` + ``config`` are the P3.2
integration. The torch/numpy split is strict: ``flat_obs``, ``shm``,
``vec_actor`` are numpy-only (actor side); ``collate``, ``rollout``, ``learner``
touch torch (learner side).
"""

from __future__ import annotations

__all__ = [
    # P3.1 in-process skeleton
    "collate",
    "policy_stub",
    "actor",
    "skeleton",
    # P3.2 vectorized multiprocess integration (single-box)
    "config",
    "flat_obs",
    "shm",
    "game_runner",
    "vec_actor",
    "rollout",
    "learner",
    # P3.2 two-box (CPU actor box ↔ GPU learner box over shard/weight files)
    "shard",
    "rollout_worker",
    "trainer",
]
