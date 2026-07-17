"""The flat single-file submission (tools/package_heuristic.py).

Regression guards for the packaging bug that broke the first upload
(``ModuleNotFoundError: No module named 'bot'`` + "unapproved libraries"):

* the generated artifact imports **only** stdlib + kit (helper/lib) — never
  ``bot``, numpy, or anything off the whitelist;
* its ``heuristic_action`` is byte-behaviour-identical to the gated
  ``bot/heuristic.py`` on real game frames (the splice can't silently drift).
"""

from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import bundle_scan  # noqa: E402
from package_heuristic import build_flat  # noqa: E402

from train.env.wrapper import WrapperEnv  # noqa: E402
from train.opponents.scripted import make_bot  # noqa: E402


@pytest.fixture(scope="module")
def flat_ns(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    build_flat("potential_field_test", out)  # runs its own import-scan + self-test
    path = out / "potential_field_test.py"
    ns = runpy.run_path(str(path), run_name="_test")  # __main__ guard stays cold
    ns["_path"] = path
    return ns


def test_artifact_imports_are_kit_and_stdlib_only(flat_ns):
    tree = ast.parse(Path(flat_ns["_path"]).read_text())
    tops = bundle_scan._top_level_imports(tree)
    allowed = bundle_scan._STDLIB | bundle_scan.ALLOWED_THIRD_PARTY | {"helper", "lib"}
    assert tops <= allowed, f"unapproved imports: {sorted(tops - allowed)}"
    assert "bot" not in tops  # the exact bug: no bot package reference


def test_no_bot_or_numpy_import_statements(flat_ns):
    """No real import of bot/numpy anywhere (the exact upload failure). Prose
    like 'from bot/heuristic.py' in the docstring is fine — only actual import
    statements matter, so match precise statement prefixes."""
    bad_prefixes = ("import bot", "from bot ", "from bot.",
                    "import numpy", "from numpy")
    for line in Path(flat_ns["_path"]).read_text().splitlines():
        stripped = line.strip()
        assert not stripped.startswith(bad_prefixes), f"bad import: {line!r}"


def test_flat_controller_matches_gated_on_real_frames(flat_ns):
    """The spliced two-phase controller must reproduce bot/heuristic.py exactly,
    including the FEED->HUNT latch (per-seat state), over a full game."""
    from bot.heuristic import HeuristicController as RefController

    FlatController = flat_ns["HeuristicController"]
    env = WrapperEnv(seed=17, emit_update=False)
    obs = env.reset()
    # one flat + one ref controller per seat, fed the identical query stream
    flat_ctrls = {pid: FlatController() for pid in env.state.players}
    ref_ctrls = {pid: RefController() for pid in env.state.players}
    play = {pid: make_bot("potential_field") for pid in env.state.players}
    compared = 0
    latched = False
    for _ in range(300):
        moves = {}
        for pid, q in obs.items():
            assert flat_ctrls[pid].act(q) == ref_ctrls[pid].act(q)
            latched = latched or flat_ctrls[pid]._hunting
            compared += 1
            moves[pid] = play[pid].act(q)
        obs, done = env.step(moves)
        if done:
            break
    assert compared > 300
    assert latched  # the game actually exercised the HUNT phase, not just FEED


def test_flat_params_match_spec_constants(flat_ns):
    """The inlined `spec` namespace must equal bot/spec's engine-exact values."""
    from bot import spec

    flat_spec = flat_ns["spec"]
    assert flat_spec.EAT_SIZE_RATIO == spec.EAT_SIZE_RATIO
    assert flat_spec.SPLIT_MIN_MASS == spec.SPLIT_MIN_MASS
    assert flat_spec.ARENA_SIZE == spec.ARENA_SIZE
    assert flat_spec.SPLIT_REACH == pytest.approx(spec.SPLIT_REACH)
    assert flat_spec.VIRUS_POP_MASS_THRESHOLD == pytest.approx(
        spec.VIRUS_POP_MASS_THRESHOLD)
