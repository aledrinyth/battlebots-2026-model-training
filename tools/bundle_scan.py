#!/usr/bin/env python3
"""P4.3 — bundle scanner: import whitelist + emitted-move validity helpers.

**Import whitelist.** A submission bundle may import only what the grader's
environment provides: the competition's allowed third-party set, the kit's own
packages, the Python stdlib, and the bundle's own files. Everything else —
above all ``torch`` — must fail the scan *before* packaging, not on the
grader. The scan is a static AST walk (conditional imports inside
``try/except`` count too — if it can execute, it's a dependency).

**Emitted-move validity.** The stateless half of the engine's own
``MoveValidator`` (the alive-check needs live engine state), plus the wire
limits: the payload must round-trip through the kit's ``MovePlayer`` pydantic
model (schema + finiteness — the engine parses exactly this) and fit
``MAX_CHARACTERS_READ``. Used by the P4.3 fuzz suite and the P4.4 full gate.

Usage:
    tools/bundle_scan.py <bundle_dir> [<bundle_dir> ...]
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Third-party imports the competition allows (playbook P4.3).
ALLOWED_THIRD_PARTY = frozenset(
    {"numpy", "networkx", "wasmtime", "pandas", "pydantic", "dotmap"}
)
#: Provided by the grader's kit environment itself.
KIT_PACKAGES = frozenset({"lib", "helper", "agario_kit"})

_STDLIB = frozenset(sys.stdlib_module_names)


def _top_level_imports(tree: ast.AST) -> set[str]:
    """Top-level package names imported anywhere in the file (incl. nested)."""
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                tops.add(alias.name.partition(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import — stays inside the bundle
            if node.module:
                tops.add(node.module.partition(".")[0])
    return tops


def scan_bundle(bundle_dir: Path) -> Dict:
    """Scan every ``.py`` under ``bundle_dir``; return a violations report."""
    bundle_dir = bundle_dir.resolve()
    py_files = sorted(p for p in bundle_dir.rglob("*.py"))
    if not py_files:
        raise ValueError(f"no .py files under {bundle_dir}")

    # The bundle's own modules: file stems, package dirs, and the bundle's
    # top-level name itself (a bundle laid out as a package imports through it).
    local = {p.stem for p in py_files}
    local |= {p.parent.name for p in py_files if p.parent != bundle_dir}
    local.add(bundle_dir.name)

    violations: List[Dict] = []
    total_bytes = 0
    for path in py_files:
        source = path.read_text()
        total_bytes += len(source.encode())
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            violations.append({"file": str(path.relative_to(bundle_dir)),
                               "import": None, "reason": f"syntax error: {e}"})
            continue
        for top in sorted(_top_level_imports(tree)):
            if (top in _STDLIB or top in ALLOWED_THIRD_PARTY
                    or top in KIT_PACKAGES or top in local):
                continue
            violations.append({
                "file": str(path.relative_to(bundle_dir)),
                "import": top,
                "reason": f"'{top}' is not stdlib, kit, whitelisted, or bundle-local",
            })

    return {
        "bundle": str(bundle_dir),
        "files": len(py_files),
        "total_source_bytes": total_bytes,
        "violations": violations,
        "ok": not violations,
    }


# --- emitted-move validity (the stateless half of the kit validator) ----------

def validate_emitted_move(payload: str, player_id: int) -> None:
    """Raise if ``payload`` would get this player banned at the engine's parse.

    Mirrors the pipe path: length limit → kit pydantic parse (schema +
    finiteness) → ``MoveValidator``'s stateless player-id rule. The alive
    check needs live engine state and can't be replayed offline.
    """
    from engine.config.io_config import MAX_CHARACTERS_READ
    from lib.interface.events.moves.move_player import MovePlayer

    if len(payload) > MAX_CHARACTERS_READ:
        raise ValueError(
            f"move payload {len(payload)} chars > MAX_CHARACTERS_READ "
            f"{MAX_CHARACTERS_READ}"
        )
    move = MovePlayer.model_validate_json(payload)  # ValidationError on bad schema
    if move.player_id != player_id:
        raise ValueError(
            "You set the move 'player_id' to a player_id other than your own."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", nargs="+", type=Path)
    args = ap.parse_args()

    ok = True
    for bundle in args.bundles:
        report = scan_bundle(bundle)
        status = "PASS" if report["ok"] else "FAIL"
        print(f"[scan] {status} {report['bundle']} "
              f"({report['files']} files, {report['total_source_bytes']:,} bytes)")
        for v in report["violations"]:
            print(f"        {v['file']}: {v['import']} — {v['reason']}")
        ok &= report["ok"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
