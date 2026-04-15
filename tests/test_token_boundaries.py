"""Unit tests for the token-boundary parser."""

from __future__ import annotations

import pytest

from vla_attention.tokens import build_boundaries


def test_molmoact_boundaries():
    tb = build_boundaries(
        family="molmoact",
        n_visual=729, n_depth=128, n_language=32, n_action=7,
    )
    assert tb.total_length == 1 + 729 + 128 + 32 + 7
    assert tb.span("bos").length == 1
    assert tb.span("visual").length == 729
    assert tb.span("depth").length == 128
    assert tb.span("language").length == 32
    assert tb.span("action").length == 7


def test_openvla_boundaries():
    tb = build_boundaries(
        family="openvla",
        n_visual=256, n_depth=0, n_language=32, n_action=7,
    )
    assert tb.total_length == 1 + 256 + 32 + 7
    assert not tb.has("depth")


def test_mask_selects_correct_tokens():
    tb = build_boundaries(
        family="molmoact",
        n_visual=4, n_depth=0, n_language=3, n_action=2,
    )
    mask = tb.mask(["visual"])
    assert mask.tolist() == [False, True, True, True, True, False, False, False, False, False]


def test_bad_family():
    with pytest.raises(ValueError):
        build_boundaries(family="nope", n_visual=1, n_depth=0,
                        n_language=1, n_action=1)


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        build_boundaries(family="molmoact", n_visual=-1, n_depth=0,
                        n_language=1, n_action=1)


def test_modality_of_index():
    tb = build_boundaries(
        family="molmoact",
        n_visual=4, n_depth=0, n_language=3, n_action=2,
    )
    assert tb.modality_of(0) == "bos"
    assert tb.modality_of(1) == "visual"
    assert tb.modality_of(5) == "language"
    assert tb.modality_of(8) == "action"
