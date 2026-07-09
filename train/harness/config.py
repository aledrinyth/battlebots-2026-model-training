"""Training configuration for the vectorized actor↔learner harness (P3.2).

One frozen dataclass carries every knob the P3.2 integration needs, so a run is
reproducible from a single object (and, later, a single YAML — P3.4). The PPO
hyper-parameters here are the playbook's P3.2 defaults verbatim: ``gamma=0.999``
(1400-step episodes), ``gae_lambda=0.95``, ``clip=0.2``, ``bptt=64``, large
batch. P3.3 swaps the *network* (SkeletonPolicy → the real deep-set model +
asymmetric critic) and P3.4 adds config-hashing/checkpointing; neither changes
the fields defined here, only adds to them.

**Batch-size arithmetic (P3.2 acceptance "GPU inference batch ≥512").** The
inference batch is exactly ``num_actors × len(policy_seats)`` — one row per
(actor, policy-seat) *stream*, constant every round (dead seats become masked
zero-rows, they do not shrink the batch). So to clear the ≥512 bar with all 8
seats policy-driven, set ``num_actors ≥ 64``. :meth:`TrainConfig.batch_rows`
computes it; the learner asserts it against ``min_infer_batch`` at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TrainConfig:
    """All knobs for a P3.2 vectorized training run.

    The defaults are a small *local* configuration (a handful of actors) so the
    module is runnable and testable on a laptop; the docstring in
    ``notes/P3.2-vectorized-actors.md`` gives the B200 + 100-core rig values
    (``num_actors=96``, ``device="cuda"``, …).
    """

    # -- topology (P3.2 "N actor processes each hosting M sequential games") --
    num_actors: int = 4
    #: Two-box only: in-process games each rollout worker vectorizes locally. The
    #: shard it emits has ``games_per_worker × len(policy_seats)`` rows; the
    #: trainer can merge several shards per update to reach a large batch.
    games_per_worker: int = 8
    #: Player ids driven by the central policy. All 8 ⇒ 8× samples/game and the
    #: largest batch; a subset leaves the rest for scripted curriculum opponents.
    policy_seats: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    #: Scripted anchors filling the non-policy seats, by name (see
    #: ``train.opponents.scripted.BOTS``). Ignored for seats in ``policy_seats``.
    scripted_names: Tuple[str, ...] = ("greedy_food", "cautious", "chaser", "coward")
    base_seed: int = 20260709

    # -- PPO / rollout (playbook P3.2 config) --------------------------------
    rollout_len: int = 64          # T: steps collected per PPO update == BPTT window
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    ppo_epochs: int = 2
    #: Rows per minibatch in the PPO update. 0 ⇒ one batch of all rows (the whole
    #: (T, R) buffer). Segments (BPTT) are kept intact — we split on the row axis.
    minibatch_rows: int = 0

    # -- reward shaping (plan v1 §3.3: dense Δmass) --------------------------
    #: Δ(total mass) is divided by this to keep per-step rewards O(0.1). Food mass
    #: is 0.0225, a good eat is a few tenths — ``1.0`` keeps them near [-1, 1]
    #: after death (death loss is the full mass, clipped by GAE not here).
    mass_scale: float = 1.0
    #: Reward clip (applied to the per-step Δmass/scale). Guards the death step
    #: (which loses the whole body) from dominating the value target.
    reward_clip: float = 10.0
    #: **Optional** dense food-approach shaping (P3.5 Stage-0 bootstrap aid).
    #: ``0.0`` (default) ⇒ the reward is *exactly* the playbook's dense Δmass —
    #: nothing changes and the nearest-food scan is skipped. When ``> 0``, on any
    #: step that did **not** eat, the per-step reward gains
    #: ``coef · (d_before − d_after)`` where ``d`` is the acting player's distance
    #: to the nearest food — a dense signal for *net progress toward* food that
    #: turns the sparse "only rewarded at the instant of eating" reward into a
    #: gradient a small-batch run can follow. It is **symmetric** (approaching +,
    #: retreating −) so jittering in place nets ~0 and cannot farm it, and it is
    #: **skipped on eat steps** (``delta > 0``) so eating — which jumps the
    #: nearest-food target to the next food — is never punished (Δmass rewards the
    #: eat). Exists to bootstrap Stage-0 on constrained (laptop-CPU, tiny-batch)
    #: hardware where random exploration eats too little food to learn from; on the
    #: rig (large batch) pure Δmass suffices and this stays 0. In the config hash,
    #: so a shaped run is a distinct, traceable experiment.
    food_shaping_coef: float = 0.0

    # -- exploration (P3.5: temporally-extended / action-persistence) --------
    #: **Training-only** temporally-extended exploration (Dabney et al. 2020,
    #: arXiv:2006.01782): with this probability, a stream that is not already
    #: holding commits to its freshly-sampled action for a random number of
    #: rounds (up to ``explore_persist_max``). Per-step independent sampling
    #: jitters in place and covers little ground, so a food-only agent almost
    #: never stumbles onto food to learn from; short bursts of *persistence*
    #: sweep the arena and surface the eats that bootstrap food-seeking — the fix
    #: the Agar.io RL literature (AgarCL) reports as essential. ``0.0`` (default)
    #: ⇒ ordinary per-step sampling, unchanged. It only perturbs the *behaviour*
    #: policy during rollout collection (held actions are stored with their
    #: recomputed log-prob, so PPO's clip absorbs the mild off-policyness); the
    #: deployed bot samples/argmaxes fresh every round, so parity is unaffected.
    explore_persist_prob: float = 0.0
    #: Max hold duration (rounds) for a persistence burst; the actual duration is
    #: drawn uniformly from ``[1, explore_persist_max]`` when a burst starts.
    explore_persist_max: int = 8
    #: **Training-only** oracle-guided exploration bursts (P3.5 escalation
    #: Track C — ``notes/P3.5-escalation-alternatives.md``). With this
    #: probability a free stream starts a burst of ``U[1, explore_oracle_max]``
    #: rounds during which its action is overridden with the heading bin
    #: pointing at its current nearest food (re-aimed each round from the row's
    #: own observation; split forced 0; rows with no food in vision keep their
    #: sampled action). Random persistence buys *coverage*; this buys
    #: *guaranteed, correctly-directed eats* in every rollout, so the value and
    #: policy gradients see what food-seeking pays without having to discover it
    #: by chance. Overridden actions store their recomputed log-prob
    #: (``train.ppo.log_prob_of``) so PPO's ratio is exact; the deployed bot and
    #: the (always oracle-free, greedy) eval are untouched. ``0.0`` (default) ⇒
    #: off and byte-identical collection; hash-neutral at the default so
    #: pre-existing run identities are unchanged (see provenance).
    explore_oracle_prob: float = 0.0
    #: Max duration (rounds) of an oracle burst; drawn ``U[1, explore_oracle_max]``.
    explore_oracle_max: int = 8

    # -- run length / device -------------------------------------------------
    #: Number of PPO updates to run. Each consumes ``rollout_len`` rounds.
    num_updates: int = 4
    #: Wall-clock cap (seconds); the run stops at whichever of num_updates /
    #: duration_s comes first. 0 ⇒ ignore (bounded only by num_updates).
    duration_s: float = 0.0
    device: str = "cpu"

    # -- model selection (P3.3) ----------------------------------------------
    #: When True, ``train.model.build_policy`` returns the P3.1 SkeletonPolicy
    #: wiring stub (using ``entity_hidden``/``trunk_hidden`` below) instead of the
    #: real network — kept for the P3.2 transport tests and cheap smoke runs.
    use_skeleton: bool = False
    #: Real-model (AgarPolicy) sizes. Defaults land the actor path at ~555k params
    #: (playbook P3.3 budget 300k–800k): deep-set encoders 64 wide, trunk+GRU 256.
    model_entity_hidden: int = 64
    model_trunk_hidden: int = 256

    # -- SkeletonPolicy stub sizes (P3.1/P3.2; used iff ``use_skeleton``) -----
    entity_hidden: int = 32
    trunk_hidden: int = 64

    # -- acceptance gates ----------------------------------------------------
    #: The learner asserts ``batch_rows >= min_infer_batch`` at startup unless it
    #: is 0. Keep 0 for small local/CI runs; set 512 for the real-rig assertion.
    min_infer_batch: int = 0
    #: Barrier timeout (s). A stuck actor trips this instead of hanging forever —
    #: the run aborts cleanly with a BrokenBarrier rather than deadlocking (a
    #: P3.2 acceptance concern: "without deadlock").
    barrier_timeout_s: float = 30.0

    # -- observability -------------------------------------------------------
    log_every: int = 1             # print a stats line every N updates
    seat_slots: Tuple[int, ...] = field(default_factory=tuple)  # filled in __post_init__

    def __post_init__(self) -> None:
        # policy_seats defines the fixed row order within each actor.
        object.__setattr__(self, "seat_slots", tuple(self.policy_seats))

    # -- derived -------------------------------------------------------------

    @property
    def seats_per_actor(self) -> int:
        return len(self.policy_seats)

    @property
    def batch_rows(self) -> int:
        """Constant inference/rollout batch width ``R = num_actors × |P|``."""
        return self.num_actors * self.seats_per_actor

    def validate(self) -> None:
        if self.num_actors < 1:
            raise ValueError("num_actors must be ≥ 1")
        if not self.policy_seats:
            raise ValueError("policy_seats must be non-empty (batch would be 0)")
        if len(set(self.policy_seats)) != len(self.policy_seats):
            raise ValueError(f"policy_seats has duplicates: {self.policy_seats}")
        if self.rollout_len < 1:
            raise ValueError("rollout_len must be ≥ 1")
        if self.min_infer_batch and self.batch_rows < self.min_infer_batch:
            raise ValueError(
                f"batch_rows={self.batch_rows} < min_infer_batch={self.min_infer_batch}; "
                f"raise num_actors to ≥ {(-(-self.min_infer_batch // self.seats_per_actor))}"
            )


__all__ = ["TrainConfig"]
