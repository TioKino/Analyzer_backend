"""
Tests del reorden DnB en classify_genre_advanced (review 2026-06-28).

BUG: las bandas anchas con `return` incondicional (Hard Dance 145-180, etc.)
tapaban [60,180], dejando ~17 géneros como CÓDIGO MUERTO. El más irónico: un
track de Drum & Bass a 174 BPM salía "Speedcore"/"Hardcore", nunca DnB.

FIX (iteración 1): mover el bloque DnB ANTES de Hard Dance → DnB reclama 160-180.
Estos tests fijan: (a) DnB vuelve a ser alcanzable en su rango; (b) CERO
regresión por debajo de 160 (House/Techno/Trance/etc. intactos).

NOTA: la clasificación de género espectral es heurística y de MÍNIMA prioridad
(la pisan Discogs/MusicBrainz/ID3/consenso). El tradeoff (Gabber/Speedcore 170+
ahora caen como DnB) está documentado en el código y debe validarse con tracks
reales antes de mergear.

Solo depende de numpy (no librosa) → corre en cualquier entorno.
"""

import numpy as np
import pytest

from spectral_genre_classifier import classify_genre_advanced

_RO = np.array([5000.0] * 10)
DNB_FAMILY = {
    "Drum & Bass", "Liquid Drum & Bass", "Neurofunk", "Jump Up", "Atmospheric DnB",
}


def _sc(mean, melodic=False):
    """Construye un spectral_centroid sintético con la media pedida.
    melodic=True -> alta varianza (is_melodic, std>800)."""
    if melodic:
        # alterna alrededor de la media para subir std por encima de 800
        return np.array([mean - 1300.0, mean + 1300.0] * 5)
    return np.array([float(mean)] * 10)


def _g(bpm, energy, has_bass, perc, centroid_mean, melodic=False):
    return classify_genre_advanced(
        bpm, energy, has_bass, None, 44100, perc, _sc(centroid_mean, melodic), _RO
    )


# ── DnB vuelve a ser alcanzable (antes era código muerto) ────────────

def test_liquid_dnb():
    assert _g(172, 0.6, True, 0.6, 3300, melodic=True) == "Liquid Drum & Bass"


def test_neurofunk():
    assert _g(170, 0.75, True, 0.6, 1800) == "Neurofunk"


def test_jump_up():
    assert _g(168, 0.6, True, 0.85, 2600) == "Jump Up"


def test_atmospheric_dnb():
    assert _g(165, 0.4, True, 0.5, 1800) == "Atmospheric DnB"


def test_plain_dnb():
    assert _g(174, 0.6, True, 0.6, 2600) == "Drum & Bass"


@pytest.mark.parametrize("bpm", [160, 165, 170, 174, 178, 180])
def test_dnb_range_is_no_longer_dead(bpm):
    """Cualquier track en 160-180 cae en la familia DnB (antes: Hard Dance)."""
    assert _g(bpm, 0.6, True, 0.6, 2600) in DNB_FAMILY


# ── Hard Dance sigue vivo para ~150-160 ──────────────────────────────

def test_hardstyle_still_reachable():
    g = _g(152, 0.85, True, 0.6, 2600)
    assert g in {"Rawstyle", "Euphoric Hardstyle", "Hardstyle", "Frenchcore"}


# ── CERO regresión por debajo de 160 ─────────────────────────────────

@pytest.mark.parametrize("bpm,energy,perc,cmean", [
    (80, 0.5, 0.6, 2600),    # hip hop range
    (100, 0.7, 0.7, 2600),   # latin range
    (124, 0.7, 0.7, 2600),   # house range
    (132, 0.8, 0.75, 1800),  # techno range
    (140, 0.7, 0.6, 3000),   # trance range
    (155, 0.85, 0.6, 2600),  # hard dance range
])
def test_below_160_never_classified_as_dnb(bpm, energy, perc, cmean):
    g = _g(bpm, energy, True, perc, cmean)
    assert g not in DNB_FAMILY, f"bpm={bpm} no debería ser DnB, salió {g}"
