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


# ── Integracion end-to-end: memoria colectiva compartida entre versiones ──

@pytest.fixture
def two_versions(db):
    """Dos usuarios con el MISMO audio en distinto codec (flac/mp3) -> mismo
    cluster acustico. Devuelve (db, fp_flac, fp_mp3)."""
    original = _rand_fp(n=400, seed=7)
    reencoded = _flip_bits(original, 12)
    _analyze_and_save(db, 'flac', original, fingerprint='md5_flac')
    _analyze_and_save(db, 'mp3', reencoded, fingerprint='md5_mp3')
    return db, 'md5_flac', 'md5_mp3'


def test_beatgrid_shared_across_versions(two_versions):
    """Un DJ corrige el beat-grid en el flac, otro en el mp3 -> al leer desde
    CUALQUIER version se ven los 2 contribuyentes (memoria compartida)."""
    db, fp_flac, fp_mp3 = two_versions
    db.submit_beat_grid_correction(fp_flac, 'devA', 0.5, 0.01, 128.0)
    db.submit_beat_grid_correction(fp_mp3, 'devB', 0.5, 0.01, 128.0)

    grid_from_mp3 = db.get_community_beat_grid(fp_mp3)
    grid_from_flac = db.get_community_beat_grid(fp_flac)
    assert grid_from_mp3['contributors'] == 2
    assert grid_from_flac['contributors'] == 2
    assert grid_from_mp3['validated'] is True  # >= 2 DJs


def test_consensus_votes_merge_across_versions(two_versions):
    """Votos de genero desde ambas versiones se SUMAN en el consenso."""
    db, fp_flac, fp_mp3 = two_versions
    db.submit_community_override(fp_flac, 'devA', 'genre', 'Techno')
    db.submit_community_override(fp_mp3, 'devB', 'genre', 'Techno')
    db.submit_community_override(fp_flac, 'devC', 'genre', 'Techno')

    consensus = db.get_community_consensus(fp_mp3, 'genre')
    assert consensus is not None, "3 votos del cluster deberian dar consenso"
    assert consensus['value'] == 'Techno'
    assert consensus['votes'] == 3


def test_rating_merges_across_versions(two_versions):
    """Ratings de ambas versiones cuentan como el mismo track."""
    db, fp_flac, fp_mp3 = two_versions
    db.rate_track(fp_flac, 'devA', 5)
    db.rate_track(fp_mp3, 'devB', 3)
    pop = db.get_track_popularity(fp_mp3)
    assert pop['total_ratings'] == 2
    assert pop['avg_rating'] == 4.0


def test_fingerprints_in_cluster_unites_versions(two_versions):
    """El helper de cues devuelve los fingerprints de todas las versiones."""
    db, fp_flac, fp_mp3 = two_versions
    cluster = set(db.fingerprints_in_cluster(fp_mp3))
    # incluye ambos fingerprints (y los track ids flac/mp3)
    assert fp_flac in cluster
    assert fp_mp3 in cluster
    assert 'flac' in cluster and 'mp3' in cluster


def test_popularity_batch_rekeys_to_original(two_versions):
    """El batch devuelve keyed por el fingerprint ORIGINAL del cliente, aunque
    internamente agrupe por cluster."""
    db, fp_flac, fp_mp3 = two_versions
    db.rate_track(fp_flac, 'devA', 5)
    out = db.get_track_popularity_batch([fp_flac, fp_mp3])
    # Ambos fingerprints originales presentes, con los datos del cluster.
    assert fp_flac in out and fp_mp3 in out
    assert out[fp_flac]['total_ratings'] == 1
    assert out[fp_mp3]['total_ratings'] == 1


# ── Metadata: la fuente mas fiable del cluster gana ──────────────────

def _save_analyzed(db, track_id, raw, fingerprint, bpm, bpm_source,
                   key=None, key_source=None, camelot=None,
                   genre='Techno', genre_source='analysis', duration=300.0):
    acoustic_id = db.resolve_acoustic_cluster(raw, duration)
    db.save_track({
        'id': track_id, 'filename': f'{track_id}.mp3', 'duration': duration,
        'bpm': bpm, 'bpm_source': bpm_source,
        'key': key, 'key_source': key_source, 'camelot': camelot,
        'energy_dj': 5, 'genre': genre, 'genre_source': genre_source,
        'track_type': 'peak', 'fingerprint': fingerprint,
        'chromaprint': encode_raw(raw), 'acoustic_id': acoustic_id,
    })
    return acoustic_id


def test_best_cluster_adopts_reliable_source(db):
    """El escenario clave: una copia de fuente dudosa (analysis, BPM 129.2) y
    otra de rekordbox (BPM 126) del MISMO audio -> best_cluster_analysis elige
    el dato de rekordbox (fuente mas fiable)."""
    original = _rand_fp(seed=42)
    reenc = _flip_bits(original, 10)
    aid = _save_analyzed(db, 'dudosa', original, 'md5_a', 129.2, 'analysis',
                         key='D#', key_source='analysis', camelot='5B',
                         genre_source='analysis')
    _save_analyzed(db, 'rkb', reenc, 'md5_b', 126.0, 'rekordbox',
                   key='D#m', key_source='rekordbox', camelot='2A',
                   genre_source='discogs')

    best = db.best_cluster_analysis(aid)
    assert best is not None
    assert best['bpm'] == 126.0, "debe adoptar el BPM de rekordbox"
    assert best['bpm_source'] == 'rekordbox'
    assert best['key'] == 'D#m' and best['camelot'] == '2A'
    assert best['key_source'] == 'rekordbox'


def test_best_cluster_never_downgrades(db):
    """Si la unica version fiable es beatport, best la mantiene aunque llegue
    despues una 'analysis' peor."""
    a = _rand_fp(seed=7)
    b = _flip_bits(a, 8)
    aid = _save_analyzed(db, 'beat', a, 'md5_beat', 126.0, 'beatport',
                         key='D#m', key_source='beatport')
    _save_analyzed(db, 'ana', b, 'md5_ana', 130.0, 'analysis',
                   key='F', key_source='analysis')
    best = db.best_cluster_analysis(aid)
    assert best['bpm'] == 126.0 and best['bpm_source'] == 'beatport'
    assert best['key'] == 'D#m'


def test_best_cluster_none_without_cluster(db):
    assert db.best_cluster_analysis(None) is None
    assert db.best_cluster_analysis('inexistente') is None


# ── find_acoustic_cluster: lookup sin crear (para /cluster-best) ──────

def test_find_acoustic_cluster_matches_existing(db):
    """find_acoustic_cluster encuentra el cluster de una version ya guardada a
    partir de la huella de OTRA copia del mismo audio."""
    original = _rand_fp(seed=11)
    aid = _analyze_and_save(db, 't', original, fingerprint='md5_t')
    # Otra copia (re-encode) del mismo audio -> debe resolver al mismo cluster.
    reenc = _flip_bits(original, 10)
    found = db.find_acoustic_cluster(reenc, duration=300.0)
    assert found == aid


def test_find_acoustic_cluster_none_when_absent(db):
    """Sin match, find devuelve None y NO crea cluster (a diferencia de
    resolve_acoustic_cluster)."""
    _analyze_and_save(db, 't', _rand_fp(seed=1), fingerprint='md5_t')
    other = _rand_fp(seed=999)  # audio distinto
    assert db.find_acoustic_cluster(other, duration=300.0) is None
    # resolve SI crea (arranca cluster nuevo) — contraste.
    assert db.resolve_acoustic_cluster(other, duration=300.0) is not None


def test_find_acoustic_cluster_empty_input(db):
    assert db.find_acoustic_cluster(None) is None
    assert db.find_acoustic_cluster([]) is None
