"""
Tests del nucleo DSP: fusion de tonalidad por chunks (ChunkedAudioAnalyzer).

Motivacion (bug "Camelot fabricado"): cuando la deteccion de key fallaba
(audio plano, silencio, chunks vacios) el fuser devolvia 'C'/'8B' con
confianza ALTA, porque los chunks fallidos votaban 'C' con peso 0.5. Resultado:
un track sin tonalidad detectable se pintaba como un Camelot 8B autoritario —
justo el dato falso que arruina una mezcla armonica.

El contrato ahora: deteccion fallida -> key/camelot None + confidence 0.0
(el cliente lo muestra '--'). Solo votan los chunks con senal real.
"""

import pytest

from chunked_analyzer import ChunkedAudioAnalyzer


@pytest.fixture
def analyzer():
    try:
        return ChunkedAudioAnalyzer()
    except ImportError:
        return ChunkedAudioAnalyzer.__new__(ChunkedAudioAnalyzer)


# ── Deteccion fallida: NO se fabrica un Camelot ──────────────────────

def test_empty_chunks_returns_unknown(analyzer):
    out = analyzer.fuse_key_results([])
    assert out["key"] is None
    assert out["camelot"] is None
    assert out["confidence"] == 0.0


def test_all_failed_chunks_returns_unknown(analyzer):
    """Todos los chunks fallaron (key None, confidence 0.0): key desconocida,
    NO 'C'/'8B'."""
    chunks = [
        {"key": None, "confidence": 0.0},
        {"key": None, "confidence": 0.0},
        {"key": None, "confidence": 0.0},
    ]
    out = analyzer.fuse_key_results(chunks)
    assert out["key"] is None
    assert out["camelot"] is None
    assert out["confidence"] == 0.0


def test_zero_confidence_C_votes_do_not_fabricate(analyzer):
    """El caso insidioso: chunks que fallaron y defaultearon a 'C' con
    confidence 0.0 NO deben producir un 'C' unanime de alta confianza."""
    chunks = [
        {"key": "C", "confidence": 0.0},
        {"key": "C", "confidence": 0.0},
    ]
    out = analyzer.fuse_key_results(chunks)
    assert out["key"] is None
    assert out["camelot"] is None


# ── Deteccion real: se respeta ───────────────────────────────────────

def test_real_detection_is_kept(analyzer):
    chunks = [
        {"key": "Am", "confidence": 0.8},
        {"key": "Am", "confidence": 0.7},
    ]
    out = analyzer.fuse_key_results(chunks)
    assert out["key"] == "Am"
    assert out["camelot"] is not None
    assert out["scale"] == "minor"
    assert out["confidence"] > 0.0


def test_mixed_real_and_failed_keeps_real(analyzer):
    """Un chunk real (Am) y uno fallido (None): gana el real, sin contaminar."""
    chunks = [
        {"key": "Am", "confidence": 0.7},
        {"key": None, "confidence": 0.0},
    ]
    out = analyzer.fuse_key_results(chunks)
    assert out["key"] == "Am"
    assert out["camelot"] is not None


def test_majority_real_key_wins(analyzer):
    chunks = [
        {"key": "Am", "confidence": 0.8},
        {"key": "Am", "confidence": 0.7},
        {"key": "C", "confidence": 0.3},
    ]
    out = analyzer.fuse_key_results(chunks)
    assert out["key"] == "Am"
