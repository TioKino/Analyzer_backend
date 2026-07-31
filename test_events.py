"""Tests para el pipe de eventos de producto (embudo onboarding/retencion):
log_event, get_funnel_counts (distinct por device_id) y purge_old_events.

BD temporal por test (aislada de analysis.db real), mismo patron que
test_telemetry.py.
"""

import json
import os
import tempfile

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


class TestLogEvent:
    def test_log_and_count_basic(self, db):
        db.log_event(device_id='d1', event_name='app_opened', platform='ios')
        db.log_event(device_id='d2', event_name='app_opened', platform='android')
        counts = db.get_funnel_counts(days=30)
        assert counts['app_opened']['devices'] == 2
        assert counts['app_opened']['total'] == 2

    def test_counts_distinct_devices_not_repeats(self, db):
        # Mismo device dispara el evento 3 veces -> 1 device, 3 total.
        for _ in range(3):
            db.log_event(device_id='d1', event_name='import_completed')
        counts = db.get_funnel_counts(days=30)
        assert counts['import_completed']['devices'] == 1
        assert counts['import_completed']['total'] == 3

    def test_props_json_persisted(self, db):
        db.log_event(
            device_id='d1',
            event_name='import_completed',
            props=json.dumps({'count': 42}),
        )
        conn = db._open_conn()
        try:
            row = conn.execute(
                "SELECT props FROM events WHERE event_name='import_completed'"
            ).fetchone()
        finally:
            conn.close()
        assert json.loads(row['props'])['count'] == 42

    def test_null_device_id_allowed(self, db):
        # Cliente sin device_id (aun sin registrar) no rompe el insert.
        rid = db.log_event(device_id=None, event_name='app_opened')
        assert rid > 0

    def test_event_name_truncated(self, db):
        db.log_event(device_id='d1', event_name='x' * 200)
        counts = db.get_funnel_counts(days=30)
        # La clave existe y esta acotada a 80 chars.
        key = next(iter(counts))
        assert len(key) == 80


class TestFunnelWindow:
    def test_old_events_excluded_from_window(self, db):
        # Evento reciente + evento viejo (40 dias) del mismo tipo, distintos devices.
        db.log_event(device_id='d_recent', event_name='app_opened')
        conn = db._open_conn()
        try:
            conn.execute(
                "INSERT INTO events (timestamp, device_id, event_name) "
                "VALUES (datetime('now','-40 days'), 'd_old', 'app_opened')"
            )
            conn.commit()
        finally:
            conn.close()
        # Ventana 30d: solo el reciente.
        assert db.get_funnel_counts(days=30)['app_opened']['devices'] == 1
        # Ventana 60d: ambos.
        assert db.get_funnel_counts(days=60)['app_opened']['devices'] == 2


class TestPurge:
    def test_purge_removes_old_keeps_recent(self, db):
        db.log_event(device_id='d1', event_name='app_opened')  # reciente
        conn = db._open_conn()
        try:
            conn.execute(
                "INSERT INTO events (timestamp, device_id, event_name) "
                "VALUES (datetime('now','-100 days'), 'd_old', 'app_opened')"
            )
            conn.commit()
        finally:
            conn.close()
        deleted = db.purge_old_events(keep_days=90)
        assert deleted == 1
        # El reciente sobrevive.
        assert db.get_funnel_counts(days=200)['app_opened']['devices'] == 1

    def test_purge_empty_is_zero(self, db):
        assert db.purge_old_events(keep_days=90) == 0


class TestEmptyFunnel:
    def test_no_events_returns_empty(self, db):
        assert db.get_funnel_counts(days=30) == {}
