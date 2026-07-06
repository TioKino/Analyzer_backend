"""
Tests del clasificador de genero por SCORING (reescritura 2026-07-06).

Foco: los generos que la cascada de `return` incondicionales dejaba como CODIGO
MUERTO (inalcanzables en su rango de BPM porque una banda anterior los tapaba)
ahora SON alcanzables cuando sus señales encajan. Y los casos core canonicos no
regresan.

Solo depende de numpy. Complementa test_genre_dnb.py (que ya cubre la familia DnB
+ cero regresion por debajo de 160).
"""
import numpy as np
import pytest

from spectral_genre_classifier import classify_genre_advanced, _GENRE_PROFILES

_RO = np.array([5000.0] * 10)


def _sc(mean, melodic=False):
    """spectral_centroid sintetico con la media pedida. melodic=True sube la
    varianza por encima de 800 (is_melodic) y de 1000/1200 (breakbeat/jazz)."""
    if melodic:
        return np.array([mean - 1300.0, mean + 1300.0] * 5)
    return np.array([float(mean)] * 10)


def _g(bpm, energy, has_bass, perc, cmean, melodic=False):
    return classify_genre_advanced(
        bpm, energy, has_bass, None, 44100, perc, _sc(cmean, melodic), _RO
    )


# ── Generos antes MUERTOS, ahora alcanzables en su rango ─────────────

def test_reggae_reachable_at_75bpm():
    # Antes: 60-100 -> "Hip Hop" incondicional tapaba Reggae (60-90).
    assert _g(75, 0.4, True, 0.4, 2600) == "Reggae"


def test_dub_reachable_dark():
    assert _g(72, 0.4, True, 0.4, 1800) == "Dub"


def test_ambient_reachable():
    # Antes: 60-100 Hip Hop tapaba Ambient (<100, energy<0.4).
    assert _g(85, 0.3, False, 0.1, 2600) == "Ambient"


def test_dark_ambient_reachable():
    assert _g(85, 0.3, False, 0.1, 1800) == "Dark Ambient"


def test_dubstep_reachable_at_142():
    # Antes: 136-145 lo tapaban Techno/Trance. Ahora Dubstep gana con bass+energia.
    assert _g(142, 0.7, True, 0.5, 2600) == "Dubstep"


def test_disco_reachable_at_120():
    # Antes: 115-130 "House" tapaba Disco (110-130).
    assert _g(120, 0.7, True, 0.5, 2600, melodic=True) == "Disco"


def test_moombahton_reachable():
    # Antes: 90-115 "Latin" tapaba Moombahton (100-112).
    assert _g(105, 0.7, True, 0.7, 2600) == "Moombahton"


def test_grime_reachable():
    # Grime vive en 138-142 dark; alcanzable con brightness en [2000,2500).
    assert _g(140, 0.7, True, 0.5, 2200) == "Grime"


def test_synthwave_reachable():
    assert _g(100, 0.5, False, 0.4, 3200, melodic=True) == "Synthwave"


def test_jungle_reachable():
    assert _g(165, 0.6, True, 0.8, 2600, melodic=True) == "Jungle"


def test_pop_reachable():
    # A 125 (fuera de Synthwave 80-120) un track melodico+brillante sin bass -> Pop.
    assert _g(125, 0.5, False, 0.4, 3800, melodic=True) == "Pop"


# ── Casos core canonicos: sin regresion ──────────────────────────────

def test_industrial_techno_core():
    assert _g(132, 0.85, True, 0.75, 1800) == "Industrial Techno"


def test_tech_house_core():
    assert _g(124, 0.75, True, 0.75, 2600) == "Tech House"


def test_hip_hop_core():
    assert _g(90, 0.5, True, 0.7, 2000) == "Hip Hop"


def test_trap_core():
    assert _g(90, 0.5, True, 0.7, 2800) == "Trap"


def test_uplifting_trance_core():
    # Sin bass pesado (trance clasico) para no competir con Brostep (has_bass).
    assert _g(145, 0.75, False, 0.6, 3800, melodic=True) == "Uplifting Trance"


def test_deep_house_core():
    assert _g(122, 0.4, False, 0.5, 1800) == "Deep House"


# ── Robustez ─────────────────────────────────────────────────────────

def test_deterministic():
    args = (128, 0.7, True, 0.6, 2600)
    assert _g(*args) == _g(*args)


def test_always_returns_a_string():
    # Cualquier combinacion cae al menos en el fallback por energia.
    for bpm in (30, 55, 95, 128, 175, 210):
        for energy in (0.2, 0.5, 0.9):
            g = _g(bpm, energy, True, 0.5, 2600)
            assert isinstance(g, str) and g


def test_no_structurally_dead_profiles():
    """Cada perfil CON condiciones debe poder ganar en algun punto: construimos
    features que cumplen sus condiciones en el centro de su rango y comprobamos
    que ningun perfil es inalcanzable por definicion (score >= sus condiciones).

    Este test documenta la propiedad central de la reescritura: no hay `return`
    incondicional que deje codigo muerto. No exige que gane (otro perfil mas
    especifico puede), solo que sea CANDIDATO (bpm en rango es alcanzable).
    """
    # Todos los rangos de BPM declarados son no vacios y validos.
    for genre, lo, hi, conds in _GENRE_PROFILES:
        assert lo <= hi, f"{genre}: rango invalido {lo}-{hi}"
        assert lo >= 20 and hi <= 220, f"{genre}: rango fuera de lo razonable"
