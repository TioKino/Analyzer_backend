"""Tests para los helpers de telemetria nuevos (Fase C admin panel):
get_fingerprint_stats y count_client_errors_by_context.

Aislados en BD temporal para no contaminar analysis.db real.
"""

import os
import tempfile

import pytest

from database import AnalysisDB


@pytest.fixture
def db():
    """BD temporal por test. tmp_path -> AnalysisDB."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        yield AnalysisDB(db_path=path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _insert_track(db: AnalysisDB, *, track_id: str, fingerprint=None):
    """Insert minimo en la tabla tracks (solo lo que get_fingerprint_stats lee)."""
    conn = db._open_conn()
    try:
        conn.execute(
            'INSERT INTO tracks (id, filename, fingerprint) VALUES (?, ?, ?)',
            (track_id, f'{track_id}.mp3', fingerprint),
        )
        conn.commit()
    finally:
        conn.close()


class TestFingerprintStats:
    """get_fingerprint_stats: contadores agregados sobre la tabla tracks."""

    def test_empty_db_returns_zeros(self, db):
        stats = db.get_fingerprint_stats()
        assert stats['total_tracks'] == 0
        assert stats['with_fingerprint'] == 0
        assert stats['without_fingerprint'] == 0
        assert stats['collision_groups'] == 0
        assert stats['collision_extra_rows'] == 0

    def test_counts_tracks_with_and_without_fingerprint(self, db):
        _insert_track(db, track_id='a', fingerprint='aaaa1111')
        _insert_track(db, track_id='b', fingerprint='bbbb2222')
        _insert_track(db, track_id='c', fingerprint=None)
        _insert_track(db, track_id='d', fingerprint='')
        stats = db.get_fingerprint_stats()
        assert stats['total_tracks'] == 4
        assert stats['with_fingerprint'] == 2
        assert stats['without_fingerprint'] == 2

    def test_unique_fingerprints_dedup(self, db):
        # 3 tracks con FP X + 1 con FP Y = 2 fingerprints unicos
        _insert_track(db, track_id='a', fingerprint='X')
        _insert_track(db, track_id='b', fingerprint='X')
        _insert_track(db, track_id='c', fingerprint='X')
        _insert_track(db, track_id='d', fingerprint='Y')
        stats = db.get_fingerprint_stats()
        assert stats['unique_fingerprints'] == 2

    def test_collision_groups_and_extras(self, db):
        # Grupo X (3 tracks) + grupo Y (2 tracks) + Z (1 track, no colisiona)
        # → groups=2, extras = (3-1)+(2-1) = 3
        for tid, fp in [('a', 'X'), ('b', 'X'), ('c', 'X'),
                        ('d', 'Y'), ('e', 'Y'),
                        ('f', 'Z')]:
            _insert_track(db, track_id=tid, fingerprint=fp)
        stats = db.get_fingerprint_stats()
        assert stats['collision_groups'] == 2
        assert stats['collision_extra_rows'] == 3

    def test_null_fingerprints_excluded_from_collisions(self, db):
        # Dos tracks con fingerprint NULL no se cuentan como colision.
        _insert_track(db, track_id='a', fingerprint=None)
        _insert_track(db, track_id='b', fingerprint=None)
        stats = db.get_fingerprint_stats()
        assert stats['collision_groups'] == 0
        assert stats['collision_extra_rows'] == 0


class TestClientErrorsByContext:
    """count_client_errors_by_context: counts por context en ventana
    reciente. Separa errores 'client:*' de 'unhandled:*' bajo la clave
    especial '_unhandled'."""

    def test_empty_db_returns_empty(self, db):
        assert db.count_client_errors_by_context(since_hours=24) == {}

    def test_counts_by_client_context(self, db):
        db.log_analysis_error(
            device_id='dev1', filename=None, fingerprint=None,
            error_class='ChromaprintError', error_msg='fpcalc not found',
            endpoint='client:chromaprint',
        )
        db.log_analysis_error(
            device_id='dev1', filename=None, fingerprint=None,
            error_class='ChromaprintError', error_msg='other',
            endpoint='client:chromaprint',
        )
        db.log_analysis_error(
            device_id='dev2', filename=None, fingerprint=None,
            error_class='SocketException', error_msg='timeout',
            endpoint='client:sync',
        )
        counts = db.count_client_errors_by_context(since_hours=24)
        assert counts.get('chromaprint') == 2
        assert counts.get('sync') == 1

    def test_unhandled_aggregated_under_special_key(self, db):
        db.log_analysis_error(
            device_id=None, filename=None, fingerprint=None,
            error_class='KeyError', error_msg='boom',
            endpoint='unhandled:/sync/push',
        )
        db.log_analysis_error(
            device_id=None, filename=None, fingerprint=None,
            error_class='ValueError', error_msg='boom2',
            endpoint='unhandled:/analyze',
        )
        counts = db.count_client_errors_by_context(since_hours=24)
        assert counts.get('_unhandled') == 2

    def test_ignores_non_client_endpoints(self, db):
        # Errores legacy de /analyze sin prefijo deben quedar fuera del
        # contador de "cliente / no manejados".
        db.log_analysis_error(
            device_id=None, filename=None, fingerprint=None,
            error_class='LibsndfileError', error_msg='',
            endpoint='/analyze',
        )
        assert db.count_client_errors_by_context(since_hours=24) == {}


def _insert_track_with_analysis(db: AnalysisDB, *, track_id: str, sources):
    """Inserta un track con analysis_json conteniendo los sources dados.
    sources es dict {bpm_source, key_source, genre_source, track_type_source}.
    """
    import json as _json
    payload = _json.dumps(sources)
    conn = db._open_conn()
    try:
        conn.execute(
            'INSERT INTO tracks (id, filename, analysis_json) VALUES (?, ?, ?)',
            (track_id, f'{track_id}.mp3', payload),
        )
        conn.commit()
    finally:
        conn.close()


class TestAnalysisSourcesBreakdown:
    """count_analysis_sources: parsea bpm/key/genre/track_type sources
    desde analysis_json y los cuenta por valor. Se basa en json_extract
    (SQLite json1), que SQLite3 estandar trae desde hace anios."""

    def test_empty_db_returns_empty_buckets(self, db):
        out = db.count_analysis_sources()
        # Debe devolver los 4 buckets siempre (aunque vacios)
        assert set(out.keys()) == {'bpm', 'key', 'genre', 'track_type'}
        for bucket in out.values():
            assert bucket == {}

    def test_counts_bpm_sources(self, db):
        _insert_track_with_analysis(db, track_id='a',
            sources={'bpm_source': 'rekordbox'})
        _insert_track_with_analysis(db, track_id='b',
            sources={'bpm_source': 'rekordbox'})
        _insert_track_with_analysis(db, track_id='c',
            sources={'bpm_source': 'local_engine'})
        out = db.count_analysis_sources()
        assert out['bpm'].get('rekordbox') == 2
        assert out['bpm'].get('local_engine') == 1

    def test_missing_source_field_uses_unknown(self, db):
        _insert_track_with_analysis(db, track_id='a',
            sources={'bpm_source': 'rekordbox'})  # falta key/genre/track_type
        out = db.count_analysis_sources()
        assert out['key'].get('unknown') == 1
        assert out['genre'].get('unknown') == 1
        assert out['track_type'].get('unknown') == 1

    def test_counts_all_four_buckets_independently(self, db):
        _insert_track_with_analysis(db, track_id='a', sources={
            'bpm_source': 'rekordbox',
            'key_source': 'rekordbox',
            'genre_source': 'discogs',
            'track_type_source': 'waveform',
        })
        out = db.count_analysis_sources()
        assert out['bpm'].get('rekordbox') == 1
        assert out['key'].get('rekordbox') == 1
        assert out['genre'].get('discogs') == 1
        assert out['track_type'].get('waveform') == 1

    def test_tracks_sin_analysis_json_son_ignorados(self, db):
        # Track sin analysis_json (caso fallback pending). NO debe contar.
        _insert_track(db, track_id='legacy', fingerprint='abc')
        out = db.count_analysis_sources()
        for bucket in out.values():
            assert bucket == {}
