"""Shared pytest config.

`slow` marks tests that run a full 1400-round game (~50 s). They are skipped by
default so `make ci` stays fast, and run on demand with `--runslow` or
`RUN_SLOW=1` (used to validate the P1.2 full-game acceptance criterion).
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (full 1400-round games)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: full-length game, minutes to run")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow") or os.environ.get("RUN_SLOW"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow or RUN_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
