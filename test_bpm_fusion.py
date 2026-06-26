"""
Tests del nucleo DSP: fusion de BPM por chunks (ChunkedAudioAnalyzer).

Motivacion: el path chunked es el que analiza TODO track > 4 min (la mayoria
de los tracks reales de DJ). Antes de estos tests el fuser no tenia cobertura
y arrastraba un bug de octava: un umbral inline `bpm > 170 -> bpm /= 2` que
partia el Drum & Bass (174 BPM -> 87 BPM) y divergia del normalizador del
consenso comunitario (`bpm_utils.normalize_bpm_to_canonical`, rango [60,180]).

Estos tests fijan el contrato de octava para que no vuelva a regresar.
"""

import pytest

from chunked_analyzer import ChunkedAudioAnalyzer


@pytest.fixture
def analyzer():
    """Instancia de ChunkedAudioAnalyzer.

    `fuse_bpm_results` es matematica pura (no usa estado de instancia), asi
    que si librosa no esta disponible en el entorno de test construimos la
    instancia saltando __init__ (que exige librosa). En CI con librosa
    instalado se usa el constructor normal.
    """
    try:
        return ChunkedAudioAnalyzer()
    except ImportError:
        return ChunkedAudioAnalyzer.__new__(ChunkedAudioAnalyzer)


def _bpm(analyzer, chunks):
    return analyzer.fuse_bpm_results(chunks)["bpm"]


# ── Regresion principal: DnB no se parte ─────────────────────────────

def test_dnb_174_stays_174(analyzer):
    """Drum & Bass a 174 BPM debe reportarse a 174, NO a 87 (bug de octava)."""
    chunks = [
        {"bpm": 174.0, "confidence": 0.8},
        {"bpm": 174.0, "confidence": 0.7},
        {"bpm": 173.0, "confidence": 0.75},
    ]
    assert _bpm(analyzer, chunks) == pytest.approx(173.7, abs=0.5)


def test_dnb_176_stays_in_range(analyzer):
    """176 BPM (limite alto de DnB) sigue por encima de 170, no se divide."""
    chunks = [{"bpm": 176.0, "confidence": 0.9}]
    assert _bpm(analyzer, chunks) == pytest.approx(176.0, abs=0.1)


# ── Generos comunes: no se tocan ─────────────────────────────────────

@pytest.mark.parametrize(
    "bpm",
    [124.0, 128.0, 140.0, 150.0, 160.0, 170.0, 180.0],
)
def test_in_range_bpm_unchanged(analyzer, bpm):
    """Cualquier BPM ya dentro de [60,180] se preserva."""
    chunks = [{"bpm": bpm, "confidence": 0.9}]
    assert _bpm(analyzer, chunks) == pytest.approx(bpm, abs=0.1)


# ── Normalizacion de octava hacia el rango canonico ──────────────────

def test_low_octave_doubled(analyzer):
    """Un BPM por debajo de 60 (doubletime mal leido) se dobla a [60,180]."""
    chunks = [{"bpm": 45.0, "confidence": 0.9}]  # 45 -> 90
    assert _bpm(analyzer, chunks) == pytest.approx(90.0, abs=0.1)


def test_high_octave_halved(analyzer):
    """Un BPM por encima de 180 (doubletime) se divide a [60,180]."""
    chunks = [{"bpm": 256.0, "confidence": 0.9}]  # 256 -> 128
    assert _bpm(analyzer, chunks) == pytest.approx(128.0, abs=0.1)


# ── Casos degenerados: no explotan ───────────────────────────────────

def test_empty_chunks(analyzer):
    """Sin chunks: default seguro, sin excepcion."""
    assert _bpm(analyzer, []) == 120.0


def test_zero_bpm_chunk_does_not_crash(analyzer):
    """Un chunk con bpm=0 no debe contaminar ni petar (se descarta del voto)."""
    out = analyzer.fuse_bpm_results([{"bpm": 0.0, "confidence": 0.0}])
    assert out["bpm"] > 0  # cae al default seguro, nunca 0/NaN


def test_mixed_valid_and_garbage(analyzer):
    """Con un chunk valido (174) y uno basura (0), gana el valido sin partirse."""
    chunks = [
        {"bpm": 174.0, "confidence": 0.8},
        {"bpm": 0.0, "confidence": 0.0},
    ]
    assert _bpm(analyzer, chunks) == pytest.approx(174.0, abs=0.5)
