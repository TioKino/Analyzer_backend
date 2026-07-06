"""
Tests del clustering acustico end-to-end contra una BD SQLite real (temporal).

Simula el flujo de /analyze: computar huella -> resolver cluster -> guardar. El
corazon de la feature es `test_reencode_shares_community_key`: dos versiones del
MISMO audio (distinto codec) resuelven a la MISMA clave de memoria colectiva,
que es lo que hoy NO pasa (se agrupa por MD5 de bytes).

Fingerprints sinteticos (arrays de int): no hace falta fpcalc ni audio real.
"""
import random

import pytest

from acoustic_fingerprint import acoustic_key, encode_raw
from database import AnalysisDB


def _rand_fp(n=400, seed=1):
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(n)]


def _flip_bits(fp, n_flips, seed=99):
    rng = random.Random(seed)
    out = list(fp)
    total = len(fp) * 32
    for _ in range(n_flips):
        pos = rng.randrange(total)
        out[pos // 32] ^= (1 << (pos % 32))
    return out


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / "test.db"))


def _analyze_and_save(db, track_id, raw, duration=300.0, fingerprint=None):
    """Replica el flujo del backend: resolver cluster + persistir."""
    acoustic_id = db.resolve_acoustic_cluster(raw, duration)
    db.save_track({
        'id': track_id,
        'filename': f'{track_id}.mp3',
        'duration': duration,
        'bpm': 128.0,
        'energy_dj': 5,
        'genre': 'Techno',
        'track_type': 'peak',
        'fingerprint': fingerprint or f'md5_{track_id}',
        'chromaprint': encode_raw(raw),
        'acoustic_id': acoustic_id,
    })
    return acoustic_id


# ── Persistencia ─────────────────────────────────────────────────────

def test_save_persists_chromaprint_and_acoustic_id(db):
    raw = _rand_fp()
    aid = _analyze_and_save(db, 't1', raw)
    row = db.get_track_by_id('t1')
    assert row is not None
    assert row['chromaprint'] == encode_raw(raw)
    assert row['acoustic_id'] == aid
    # chromaprint NO debe filtrarse al analysis_json (blob tecnico).
    assert 'chromaprint' not in (row.get('analysis_json') or '')


def test_first_track_starts_new_cluster(db):
    raw = _rand_fp()
    aid = _analyze_and_save(db, 't1', raw)
    assert aid == acoustic_key(raw)


# ── El corazon: agrupacion por sonido ────────────────────────────────

def test_reencode_shares_community_key(db):
    """Dos versiones del mismo audio (re-encode) -> MISMA clave comunitaria."""
    original = _rand_fp(n=400, seed=1)
    reencoded = _flip_bits(original, 12)  # mismo audio, otro codec

    aid1 = _analyze_and_save(db, 'flac', original, fingerprint='md5_flac')
    aid2 = _analyze_and_save(db, 'mp3', reencoded, fingerprint='md5_mp3')

    # Mismo cluster pese a tener MD5 de contenido distinto.
    assert aid1 == aid2

    # Y la clave comunitaria que devuelve el backend para CADA fingerprint
    # (el que manda el cliente) es la misma -> comparten cues/beat-grid/etc.
    assert db.canonical_community_key('md5_flac') == \
           db.canonical_community_key('md5_mp3')
    assert db.canonical_community_key('md5_mp3') == aid1


def test_different_audio_separate_clusters(db):
    a = _rand_fp(seed=1)
    b = _rand_fp(seed=2)  # audio distinto
    aid1 = _analyze_and_save(db, 'a', a, fingerprint='md5_a')
    aid2 = _analyze_and_save(db, 'b', b, fingerprint='md5_b')
    assert aid1 != aid2
    assert db.canonical_community_key('md5_a') != \
           db.canonical_community_key('md5_b')


def test_duration_filter_prevents_false_match(db):
    """Un fingerprint casi-igual pero de duracion muy distinta NO se agrupa
    (el mismo audio dura lo mismo; distinta duracion = otra cosa)."""
    raw = _rand_fp()
    _analyze_and_save(db, 'short', raw, duration=120.0)
    aid_long = _analyze_and_save(db, 'long', _flip_bits(raw, 8), duration=300.0)
    # 300 vs 120 -> fuera del +-2.5s -> no son candidatos -> cluster nuevo.
    assert aid_long == acoustic_key(_flip_bits(raw, 8))


# ── Compatibilidad hacia atras ───────────────────────────────────────

def test_unknown_fingerprint_falls_back_to_itself(db):
    """Track no visto por el backend -> la clave comunitaria es el propio
    fingerprint (la memoria colectiva vieja, guardada por MD5, sigue viva)."""
    assert db.canonical_community_key('nunca_visto') == 'nunca_visto'


def test_track_without_acoustic_id_falls_back(db):
    """Track analizado antes de la feature (sin acoustic_id) -> fallback al
    fingerprint, sin romper nada."""
    db.save_track({
        'id': 'legacy', 'filename': 'legacy.mp3', 'duration': 300.0,
        'bpm': 128.0, 'energy_dj': 5, 'genre': 'Techno', 'track_type': 'peak',
        'fingerprint': 'md5_legacy',
        # sin chromaprint ni acoustic_id (track pre-feature)
    })
    assert db.canonical_community_key('md5_legacy') == 'md5_legacy'
