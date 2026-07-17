"""Versioned adapter boundaries (P2.1).

The adaptation-pipeline discipline: the observation/action *interfaces* between
the featurizer, the policy core, and the wire are frozen in **versioned** files
(``obs_v1``, ``act_v1``). A spec change that alters the observation or action
surface bumps the version (``obs_v2`` …) rather than editing v1 in place, so old
checkpoints keep a compatible adapter and the spec-diff detector (P11.3) has a
stable thing to diff. Nothing here holds physics constants — those live only in
``bot.spec``.
"""
