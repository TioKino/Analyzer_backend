"""
Tests de sanitize_cue_submissions (community_cues_endpoint).

Hardening de input no confiable: el upload de cues comunitarios no acotaba
cantidad ni rango de position_ms. Un cliente con el secreto podía meter miles
de filas, posiciones negativas o notas gigantes en la tabla comunitaria. El
helper acota volumen y descarta lo inválido SIN rechazar cues legítimos.
"""

import pytest

from community_cues_endpoint import (
    sanitize_cue_submissions,
    MAX_CUES_PER_UPLOAD,
    MAX_CUE_NOTE_LEN,
)


class _Cue:
    def __init__(self, type="drop", position_ms=1000, end_position_ms=None, note=None):
        self.type = type
        self.position_ms = position_ms
        self.end_position_ms = end_position_ms
        self.note = note


def test_normal_cue_kept():
    out = sanitize_cue_submissions([_Cue(position_ms=1000)])
    assert out == [("drop", 1000, None, None)]


def test_position_zero_is_valid():
    # Un cue al inicio del track (0ms) es legítimo.
    out = sanitize_cue_submissions([_Cue(position_ms=0)])
    assert len(out) == 1
    assert out[0][1] == 0


def test_negative_position_dropped():
    out = sanitize_cue_submissions([_Cue(position_ms=-5), _Cue(position_ms=2000)])
    assert out == [("drop", 2000, None, None)]


def test_missing_position_dropped():
    out = sanitize_cue_submissions([_Cue(position_ms=None)])
    assert out == []


def test_invalid_region_end_before_start_nulled():
    out = sanitize_cue_submissions([_Cue(position_ms=5000, end_position_ms=3000)])
    assert out[0][2] is None  # end anulado


def test_valid_region_kept():
    out = sanitize_cue_submissions([_Cue(position_ms=1000, end_position_ms=4000)])
    assert out[0][2] == 4000


def test_long_note_truncated():
    out = sanitize_cue_submissions([_Cue(position_ms=0, note="x" * 5000)])
    assert len(out[0][3]) == MAX_CUE_NOTE_LEN


def test_count_capped():
    many = [_Cue(position_ms=i) for i in range(MAX_CUES_PER_UPLOAD + 500)]
    out = sanitize_cue_submissions(many)
    assert len(out) == MAX_CUES_PER_UPLOAD


def test_empty_list():
    assert sanitize_cue_submissions([]) == []
