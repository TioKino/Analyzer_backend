"""
Tests de `GenreDetector._normalize_genre` — invariante de especificidad (A6).

El match parcial itera las claves de GENRE_MAP ordenadas por longitud
descendente para que una clave especifica ("industrial techno", 17 chars) gane
sobre una generica ("techno", 6 chars) cuando el input contiene ambas. Antes el
orden era el de insercion del dict y la generica ganaba arbitrariamente,
perdiendo especificidad de genero. Estos tests fijan ese contrato.

`genre_detection` no depende de librosa, asi que corren en cualquier entorno.
"""

import pytest

import genre_detection as g
from genre_detection import GenreDetector, GENRE_MAP


@pytest.fixture
def detector():
    # Sin token: discogs_client queda None, _normalize_genre no lo usa.
    return GenreDetector()


# ── Invariante de orden ──────────────────────────────────────────────

def test_keys_sorted_by_specificity_desc():
    ks = g._GENRE_KEYS_BY_SPECIFICITY
    assert all(len(ks[i]) >= len(ks[i + 1]) for i in range(len(ks) - 1))


# ── Match parcial: lo especifico gana sobre lo generico ──────────────
# Inputs que NO son claves exactas (van al loop de substring) y contienen
# tanto una clave especifica multi-palabra como una generica.

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hard Industrial Techno Banger", "Industrial Techno"),
        ("Some Deep House Vibes", "Deep House"),
        ("Melodic Tech House Set", "Tech House"),
    ],
)
def test_specific_key_wins_over_generic(detector, raw, expected):
    assert raw.lower() not in GENRE_MAP, "el input no debe ser clave exacta"
    assert detector._normalize_genre(raw) == expected


def test_generic_alone_still_matches(detector):
    assert detector._normalize_genre("random techno thing") == "Techno"


# ── Otras ramas ──────────────────────────────────────────────────────

def test_exact_match(detector):
    assert detector._normalize_genre("Techno") == "Techno"


def test_empty_defaults_electronic(detector):
    assert detector._normalize_genre("") == "Electronic"
    assert detector._normalize_genre(None) == "Electronic"


def test_no_match_is_title_cased(detector):
    assert detector._normalize_genre("zzz unknownthing") == "Zzz Unknownthing"
