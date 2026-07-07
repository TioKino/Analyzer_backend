"""
Tests del nucleo de matching acustico (acoustic_fingerprint.py).

No dependen de fpcalc ni de audio real: operan sobre arrays de int sinteticos
que simulan los tres escenarios que importan:
  1. mismo audio, otro tag/codec -> unos pocos bits volteados (Hamming ~0)
  2. mismo audio, desalineado por encoder-delay -> array desplazado
  3. audio distinto -> array aleatorio independiente (Hamming ~0.5)

La extraccion `fpcalc -raw` (compute_raw_chromaprint) es I/O y se valida en
Render, no aqui.
"""
import os
import random

import pytest

from acoustic_fingerprint import (
    MATCH_THRESHOLD,
    acoustic_key,
    compute_raw_chromaprint,
    decode_raw,
    encode_raw,
    fingerprints_match,
    hamming_distance,
)


def _rand_fp(n=400, seed=1):
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(n)]


def _flip_bits(fp, n_flips, seed=99):
    """Voltea n_flips bits repartidos por el array (simula re-encode del mismo
    audio: el array es 99%+ identico)."""
    rng = random.Random(seed)
    out = list(fp)
    total_bits = len(fp) * 32
    for _ in range(n_flips):
        pos = rng.randrange(total_bits)
        idx, bit = pos // 32, pos % 32
        out[idx] ^= (1 << bit)
    return out


# ── Distancia base ───────────────────────────────────────────────────

def test_identical_is_zero():
    fp = _rand_fp()
    assert hamming_distance(fp, fp) == 0.0


def test_same_audio_reencoded_matches():
    """Mismo audio a otro codec: ~0.1% de bits cambian -> debe matchear."""
    fp = _rand_fp(n=400)
    # 400*32 = 12800 bits; 12 flips ~= 0.09% (holgadamente < umbral)
    reenc = _flip_bits(fp, 12)
    d = hamming_distance(fp, reenc)
    assert d < MATCH_THRESHOLD, f"mismo audio re-encoded no matcheo (d={d})"
    assert fingerprints_match(fp, reenc)


def test_different_audio_does_not_match():
    """Dos audios distintos -> Hamming cercano a 0.5 (aleatorio) -> NO match."""
    a = _rand_fp(seed=1)
    b = _rand_fp(seed=2)
    d = hamming_distance(a, b)
    assert d > 0.3, f"audios distintos demasiado cerca (d={d})"
    assert not fingerprints_match(a, b)


# ── Robustez de alineacion (encoder delay) ───────────────────────────

def test_shifted_still_matches():
    """El mismo audio con padding distinto corre el array unas posiciones.
    El barrido de offset debe recuperar el match."""
    fp = _rand_fp(n=400)
    shifted = fp[2:]  # como si empezara 2 subfingerprints mas tarde
    d = hamming_distance(fp, shifted)
    assert d < MATCH_THRESHOLD, f"desalineado por 2 no matcheo (d={d})"


def test_too_short_overlap_is_max_distance():
    assert hamming_distance([1, 2, 3], [1, 2, 3]) == 1.0


def test_empty_is_max_distance():
    assert hamming_distance([], [1, 2, 3]) == 1.0
    assert hamming_distance(None, [1, 2, 3]) == 1.0


# ── Serializacion / clave exacta ─────────────────────────────────────

def test_encode_decode_roundtrip():
    fp = _rand_fp(n=123)
    assert decode_raw(encode_raw(fp)) == fp


def test_encode_empty_is_none():
    assert encode_raw([]) is None
    assert encode_raw(None) is None
    assert decode_raw(None) == []
    assert decode_raw('') == []


def test_decode_garbage_is_empty():
    assert decode_raw('not-valid-base64!!!') == []


def test_acoustic_key_deterministic_and_distinct():
    a = _rand_fp(seed=1)
    b = _rand_fp(seed=2)
    assert acoustic_key(a) == acoustic_key(list(a))
    assert acoustic_key(a) != acoustic_key(b)
    assert acoustic_key([]) is None


def test_acoustic_key_ignores_tag_but_key_differs_for_reencode():
    """Mismo array (tag distinto, mismo codec) -> misma key exacta.
    Re-encode (algun bit cambia) -> key exacta DISTINTA (por eso hace falta
    Hamming ademas de la key exacta)."""
    fp = _rand_fp()
    assert acoustic_key(fp) == acoustic_key(list(fp))       # idempotente
    assert acoustic_key(fp) != acoustic_key(_flip_bits(fp, 3))  # re-encode


def test_compute_missing_fpcalc_is_none(tmp_path, monkeypatch):
    """Sin fpcalc (binario inexistente) devuelve None sin lanzar -> el analisis
    sigue y la memoria colectiva cae al fingerprint MD5 (best-effort)."""
    monkeypatch.setenv('FPCALC_BIN', '/nonexistent/fpcalc-xyz')
    f = tmp_path / "x.mp3"
    f.write_bytes(b'not audio')
    assert compute_raw_chromaprint(str(f)) is None


def test_compute_respects_fpcalc_bin_env(monkeypatch):
    """compute_raw_chromaprint lee FPCALC_BIN (lo setea local_engine desde el
    bundle). Con un binario que no existe -> None, sin excepcion."""
    monkeypatch.setenv('FPCALC_BIN', '/definitely/not/here')
    assert compute_raw_chromaprint('/tmp/whatever.mp3') is None


@pytest.mark.parametrize("n_flips,should_match", [
    (0, True),
    (5, True),
    (12, True),
    # 4000 flips sobre 12800 bits, descontando colisiones (mismo bit volteado
    # 2x se cancela), da ~27% efectivo -> claramente por encima del umbral 0.15.
    # (Nota: por las colisiones, ~2000 flips solo dan ~14.5%, aun bajo el
    # umbral: el codigo es MAS tolerante de lo que el conteo bruto sugiere.)
    (4000, False),
])
def test_threshold_boundary(n_flips, should_match):
    fp = _rand_fp(n=400)
    other = _flip_bits(fp, n_flips)
    assert fingerprints_match(fp, other) is should_match
