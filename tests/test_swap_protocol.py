from __future__ import annotations

from types import SimpleNamespace

import pytest

import swap


def test_band_layers_accepts_depth_percentages():
    lens = SimpleNamespace(source_layers=list(range(20)))
    assert swap.band_layers(lens, 20, 40, 80) == list(range(8, 16))


def test_invalid_band_is_rejected():
    lens = SimpleNamespace(source_layers=list(range(4)))
    with pytest.raises(ValueError, match="invalid workspace band"):
        swap.band_layers(lens, 4, 90, 20)
