"""P4.3 acceptance (scanner half): the import whitelist catches contraband.

* an injected ``import torch`` fails the scan (the playbook's acceptance
  case — the exact failure mode that would brick a submission);
* the real ``bot/`` package passes (it is the shipping surface);
* stdlib / kit / bundle-local / relative imports are all recognized;
* conditional (``try/except``) imports still count.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import bundle_scan  # noqa: E402  (path set above)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))


def test_clean_bundle_passes(tmp_path: Path) -> None:
    _write(tmp_path / "main.py", """
        import json
        import numpy as np
        from helper.game import Game
        from lib.interface.events.moves.move_player import MovePlayer
        from featurizer import Featurizer
        from . import spec
    """)
    _write(tmp_path / "featurizer.py", "import numpy\n")
    _write(tmp_path / "spec.py", "import math\n")
    report = bundle_scan.scan_bundle(tmp_path)
    assert report["ok"], report["violations"]
    assert report["files"] == 3


def test_injected_torch_fails_the_scan(tmp_path: Path) -> None:
    _write(tmp_path / "main.py", "import numpy\n")
    _write(tmp_path / "inference.py", "import torch\n")  # the acceptance case
    report = bundle_scan.scan_bundle(tmp_path)
    assert not report["ok"]
    assert any(v["import"] == "torch" for v in report["violations"])


def test_conditional_import_still_counts(tmp_path: Path) -> None:
    _write(tmp_path / "main.py", """
        try:
            import torch
        except ImportError:
            torch = None
    """)
    report = bundle_scan.scan_bundle(tmp_path)
    assert not report["ok"]
    assert report["violations"][0]["import"] == "torch"


def test_train_import_fails_the_scan(tmp_path: Path) -> None:
    # A bundle must be self-contained: reaching back into the repo's training
    # stack would NameError on the grader even without torch.
    _write(tmp_path / "main.py", "from train.model import AgarPolicy\n")
    report = bundle_scan.scan_bundle(tmp_path)
    assert not report["ok"]


def test_syntax_error_is_a_violation(tmp_path: Path) -> None:
    _write(tmp_path / "main.py", "def broken(:\n")
    report = bundle_scan.scan_bundle(tmp_path)
    assert not report["ok"]
    assert "syntax error" in report["violations"][0]["reason"]


def test_the_real_bot_package_passes() -> None:
    report = bundle_scan.scan_bundle(REPO / "bot")
    assert report["ok"], report["violations"]


def test_empty_bundle_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no .py files"):
        bundle_scan.scan_bundle(tmp_path)


# --- emitted-move validity helper ----------------------------------------------

def test_validate_emitted_move_accepts_a_legal_move() -> None:
    pytest.importorskip("engine")
    from lib.interface.events.moves.move_player import MovePlayer
    from lib.models.penguin_model import DirectionModel

    payload = MovePlayer(player_id=3, direction=DirectionModel(x=1.0, y=-0.5),
                         split=True).model_dump_json()
    bundle_scan.validate_emitted_move(payload, player_id=3)  # no raise


def test_validate_emitted_move_rejects_wrong_player_and_garbage() -> None:
    pytest.importorskip("engine")
    from lib.interface.events.moves.move_player import MovePlayer
    from lib.models.penguin_model import DirectionModel

    payload = MovePlayer(player_id=3, direction=DirectionModel(x=0.0, y=0.0),
                         split=False).model_dump_json()
    with pytest.raises(ValueError):
        bundle_scan.validate_emitted_move(payload, player_id=4)
    with pytest.raises(Exception):
        bundle_scan.validate_emitted_move("{not json", player_id=3)
    with pytest.raises(ValueError):
        bundle_scan.validate_emitted_move("x" * 5000, player_id=3)
