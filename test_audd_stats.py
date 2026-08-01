"""Tests para get_audd_stats_by_source: desglose success/fail de AudD por via
(analyze/recognize/identify), excluyendo el marcador de sesion. Aclara que los
'fallos' son en su mayoria 'sin match' (normal), no errores.
"""

import os
import tempfile
import time

import pytest

from database import AnalysisDB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        yield AnalysisDB(db_path=path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _raw_call(db, *, source, success, days_ago=0):
    """Inserta una fila en audd_call_log con called_at controlado."""
    conn = db._open_conn()
    try:
        conn.execute(
            'INSERT INTO audd_call_log '
            '(fingerprint, called_at, success, source) VALUES (?, ?, ?, ?)',
            ('fp', time.time() - days_ago * 86400, 1 if success else 0, source),
        )
        conn.commit()
    finally:
        conn.close()


class TestAuddStatsBySource:
    def test_split_success_fail_per_source(self, db):
        # analyze: 3 llamadas, 1 match, 2 sin match.
        _raw_call(db, source='analyze', success=True)
        _raw_call(db, source='analyze', success=False)
        _raw_call(db, source='analyze', success=False)
        # recognize: 2 llamadas, ambas match.
        _raw_call(db, source='recognize', success=True)
        _raw_call(db, source='recognize', success=True)

        stats = db.get_audd_stats_by_source(days=30)
        assert stats['analyze'] == {'total': 3, 'success': 1, 'fail': 2}
        assert stats['recognize'] == {'total': 2, 'success': 2, 'fail': 0}

    def test_session_marker_excluded(self, db):
        _raw_call(db, source='analyze', success=True)
        _raw_call(db, source='recognize_session', success=False)  # marcador, no llamada
        stats = db.get_audd_stats_by_source(days=30)
        assert 'recognize_session' not in stats
        assert stats['analyze']['total'] == 1

    def test_legacy_null_source_counts_as_analyze(self, db):
        # Filas viejas sin source = 'analyze'.
        conn = db._open_conn()
        try:
            conn.execute(
                'INSERT INTO audd_call_log (fingerprint, called_at, success) '
                'VALUES (?, ?, ?)',
                ('fp', time.time(), 1),
            )
            conn.commit()
        finally:
            conn.close()
        stats = db.get_audd_stats_by_source(days=30)
        assert stats['analyze'] == {'total': 1, 'success': 1, 'fail': 0}

    def test_old_calls_excluded_from_window(self, db):
        _raw_call(db, source='analyze', success=True, days_ago=1)   # dentro
        _raw_call(db, source='analyze', success=True, days_ago=40)  # fuera de 30d
        stats = db.get_audd_stats_by_source(days=30)
        assert stats['analyze']['total'] == 1

    def test_empty_returns_empty(self, db):
        assert db.get_audd_stats_by_source(days=30) == {}
