"""
Tests del scoring de compatibilidad DJ (similar_tracks_endpoint).

Foco principal (bug real): un track SIN genero no debe fingir match de genero.
En Python `'' in cualquier_string` es True, asi que el substring-match de genero
marcaba genre_match=True (+10) para tracks con genero vacio contra CUALQUIER
objetivo, colandolos por delante de tracks de genero real distinto.

Tambien cubre la rueda Camelot (incluyendo wraps 1<->12) y la tolerancia BPM.
"""

import pytest

from similar_tracks_endpoint import (
    calculate_compatibility_score,
    is_key_compatible,
)


def _score(track, **targets):
    """Devuelve (score, bpm_match, key_compatible, energy_match, genre_match)."""
    return calculate_compatibility_score(track, **targets)


BASE = {"bpm": 128.0, "camelot": "8A", "energy_dj": 5, "genre": "techno"}


# ── Genero: el bug del substring vacio ───────────────────────────────

def test_empty_genre_does_not_fake_match():
    track = {**BASE, "genre": ""}
    *_, genre_match = _score(track, target_genre="techno")
    assert genre_match is False


def test_missing_genre_key_does_not_fake_match():
    track = {"bpm": 128.0, "camelot": "8A", "energy_dj": 5}  # sin 'genre'
    *_, genre_match = _score(track, target_genre="techno")
    assert genre_match is False


def test_exact_genre_matches():
    *_, genre_match = _score({**BASE, "genre": "techno"}, target_genre="techno")
    assert genre_match is True


def test_subgenre_substring_still_matches():
    *_, genre_match = _score({**BASE, "genre": "deep house"}, target_genre="house")
    assert genre_match is True


def test_different_genre_does_not_match():
    *_, genre_match = _score({**BASE, "genre": "jazz"}, target_genre="techno")
    assert genre_match is False


def test_empty_genre_scores_below_real_match():
    no_genre = _score({**BASE, "genre": ""}, target_genre="techno")[0]
    real = _score({**BASE, "genre": "techno"}, target_genre="techno")[0]
    assert no_genre < real


# ── Camelot wheel ────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("8A", "8A"),    # misma
    ("8A", "9A"),    # +1
    ("8A", "7A"),    # -1
    ("8A", "8B"),    # paralela
    ("1A", "12A"),   # wrap -1
    ("12A", "1A"),   # wrap +1
    ("12B", "1B"),   # wrap +1 en B
])
def test_compatible_keys(a, b):
    assert is_key_compatible(a, b) is True


@pytest.mark.parametrize("a,b", [
    ("8A", "10A"),   # +2 no es compatible
    ("8A", "2A"),    # lejos
    ("1A", "6B"),
])
def test_incompatible_keys(a, b):
    assert is_key_compatible(a, b) is False


def test_missing_camelot_assumed_compatible():
    assert is_key_compatible("", "8A") is True
    assert is_key_compatible("8A", None) is True


def test_camelot_symmetry():
    # La compatibilidad debe ser simetrica.
    from similar_tracks_endpoint import CAMELOT_COMPATIBLE
    for k, compat in CAMELOT_COMPATIBLE.items():
        for other in compat:
            assert k in CAMELOT_COMPATIBLE[other], f"{k}->{other} no simetrico"


# ── BPM ──────────────────────────────────────────────────────────────

def test_bpm_within_tolerance_matches():
    _, bpm_match, *_ = _score({**BASE, "bpm": 130.0}, target_bpm=128.0)  # diff 2 <= 4
    assert bpm_match is True


def test_bpm_double_tempo_matches():
    _, bpm_match, *_ = _score({**BASE, "bpm": 256.0}, target_bpm=128.0)
    assert bpm_match is True


def test_bpm_far_does_not_match():
    _, bpm_match, *_ = _score({**BASE, "bpm": 100.0}, target_bpm=128.0)
    assert bpm_match is False
