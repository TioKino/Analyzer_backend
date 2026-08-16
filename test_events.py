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

    def test_eventos_anonimos_de_la_web_se_cuentan(self, db):
        """REGRESION: la web manda sus eventos SIN device_id a proposito
        (privacy-first, esta escrito en el tracking de index.html). El embudo
        contaba `COUNT(DISTINCT device_id)`, y SQL IGNORA los NULL en un
        COUNT(DISTINCT) -> `web_visit` reportaba 0 por muchas visitas que
        hubiera, y el embudo de adquisicion era ciego sin que nada fallara.

        Se persiguio como si fuera un problema de deploy y luego de CORS; las
        dos cosas estaban bien. El fallo estaba en el contador.
        """
        for _ in range(3):
            db.log_event(device_id=None, event_name='web_visit', platform='web')
        counts = db.get_funnel_counts(days=30)
        assert counts['web_visit']['devices'] == 3, (
            'los eventos anonimos deben contar uno por fila: sin identidad, '
            '"dispositivos unicos" no esta definido'
        )
        assert counts['web_visit']['total'] == 3

    def test_anonimos_e_identificados_conviven(self, db):
        """El COALESCE no debe romper el dedupe de los eventos que SI tienen
        identidad: 1 device repetido sigue contando 1, y los anonimos suman."""
        db.log_event(device_id='d1', event_name='app_opened')
        db.log_event(device_id='d1', event_name='app_opened')
        db.log_event(device_id=None, event_name='app_opened')
        db.log_event(device_id=None, event_name='app_opened')
        counts = db.get_funnel_counts(days=30)
        assert counts['app_opened']['devices'] == 3   # d1 + los dos anonimos
        assert counts['app_opened']['total'] == 4

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


def _open(db, device, days_ago):
    """Inserta un app_opened para `device` hace `days_ago` dias."""
    conn = db._open_conn()
    try:
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name) "
            "VALUES (datetime('now', ?), ?, 'app_opened')",
            (f'-{days_ago} days', device),
        )
        conn.commit()
    finally:
        conn.close()


class TestRetention:
    def test_d1_d7_d28_basic(self, db):
        # ret_a: vuelve en D0+1, D0+7 y D0+28 -> retenido en los tres.
        _open(db, 'ret_a', 30)   # D0
        _open(db, 'ret_a', 29)   # D0+1
        _open(db, 'ret_a', 23)   # D0+7
        _open(db, 'ret_a', 2)    # D0+28
        # ret_b: solo el dia de install -> nunca vuelve.
        _open(db, 'ret_b', 30)
        # ret_c: instalo HOY -> aun no medible (excluido de todos los cohorts).
        _open(db, 'ret_c', 0)

        r = db.get_retention_cohorts()
        # c excluido por edad; cohort = {a, b} = 2 en los tres.
        assert r['d1']['cohort'] == 2
        assert r['d1']['retained'] == 1  # solo a
        assert r['d1']['rate'] == 50.0
        assert r['d7']['cohort'] == 2
        assert r['d7']['retained'] == 1
        assert r['d28']['cohort'] == 2
        assert r['d28']['retained'] == 1

    def test_recent_cohort_excluded_until_measurable(self, db):
        # Device que instalo hace 3 dias: medible para D1 (>=1d) pero NO para
        # D7/D28 (aun no han pasado 7/28 dias).
        _open(db, 'young', 3)   # D0
        _open(db, 'young', 2)   # D0+1 -> D1 retenido
        r = db.get_retention_cohorts()
        assert r['d1']['cohort'] == 1
        assert r['d1']['retained'] == 1
        # No elegible aun para D7/D28.
        assert r['d7']['cohort'] == 0
        assert r['d28']['cohort'] == 0

    def test_no_return_is_zero_rate(self, db):
        _open(db, 'solo', 10)  # instalo, nunca volvio
        r = db.get_retention_cohorts()
        assert r['d1']['cohort'] == 1
        assert r['d1']['retained'] == 0
        assert r['d1']['rate'] == 0.0

    def test_empty_events_zero_cohorts(self, db):
        r = db.get_retention_cohorts()
        assert r['d1'] == {'cohort': 0, 'retained': 0, 'rate': 0.0}
        assert r['d7']['cohort'] == 0
        assert r['d28']['cohort'] == 0

    def test_session_start_cuenta_como_dia_activo(self, db):
        # El device instala (app_opened D0) y al dia siguiente NO hace cold start
        # pero sí trae la app a primer plano -> session_start en D0+1. Debe
        # contar como retenido D1 (antes se perdia porque solo miraba app_opened).
        conn = db._open_conn()
        try:
            conn.execute(
                "INSERT INTO events (timestamp, device_id, event_name) "
                "VALUES (datetime('now','-10 days'), 'warm', 'app_opened')",
            )
            conn.execute(
                "INSERT INTO events (timestamp, device_id, event_name) "
                "VALUES (datetime('now','-9 days'), 'warm', 'session_start')",
            )
            conn.commit()
        finally:
            conn.close()
        r = db.get_retention_cohorts()
        assert r['d1']['cohort'] == 1
        assert r['d1']['retained'] == 1  # retenido gracias a session_start
