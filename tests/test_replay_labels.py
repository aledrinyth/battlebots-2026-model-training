"""R0 labels: heading-bin inverse, stay masking, rank weights (impl doc §12)."""

from __future__ import annotations

import math

import pytest

from bot.adapters import act_v1
from train.replay.labels import heading_to_bin, rank_weight


def test_heading_roundtrip_all_bins():
    for i in range(act_v1.HEADING_BINS):
        x, y = act_v1.bin_to_vector(i)
        assert heading_to_bin(x, y) == i


def test_heading_nearest_bin_and_wraparound():
    # 7° is nearest bin 0 (bin width 15°); 8° crosses to bin 1's half
    assert heading_to_bin(math.cos(math.radians(7)), math.sin(math.radians(7))) == 0
    assert heading_to_bin(math.cos(math.radians(8)), math.sin(math.radians(8))) == 1
    # just below the +x axis wraps to bin 0, not bin 24
    assert heading_to_bin(1.0, -1e-9) == 0
    # magnitude does not matter
    assert heading_to_bin(100.0, 0.0) == 0
    assert heading_to_bin(-3.0, 0.0) == act_v1.HEADING_BINS // 2


def test_stay_has_no_heading():
    assert heading_to_bin(0.0, 0.0) is None


def test_rank_weight_linear():
    assert rank_weight(1, 8) == 1.0
    assert rank_weight(8, 8) == 0.0
    assert rank_weight(2, 8) == pytest.approx(6 / 7)
    assert rank_weight(1, 2) == 1.0 and rank_weight(2, 2) == 0.0


def test_rank_weight_rejects_nonsense():
    with pytest.raises(ValueError):
        rank_weight(0, 8)
    with pytest.raises(ValueError):
        rank_weight(9, 8)
    with pytest.raises(ValueError):
        rank_weight(1, 1)
