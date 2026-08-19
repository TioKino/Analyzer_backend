"""Tests del filtro por PLATAFORMA del embudo (`GET /admin/funnel?platform=`).

Motivacion (owner): la mayoria de DJs tienen la musica en PC/discos externos,
no en el movil -> el embudo de 'desktop' mide el valor real. El evento ya
guardaba `platform`, pero la vista los mezclaba. Esto cubre el desglose.

Integracion aislada: BD temporal via DATABASE_PATH -> counts deterministas
(sin depender de la analysis.db real ni de datos previos).
"""

import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient

from main import app
from routes.admin_panel import _platform_filter

_SECRET = 'test-admin-secret-1234567890'


# ── Unidad: _platform_filter (funcion pura) ────────────────────────────
class TestPlatformFilter:
    def test_none_no_filter(self):
        sql, params = _platform_filter(None)
        assert sql == "" and params == []

    def test_desktop_group_expands(self):
        sql, params = _platform_filter('desktop')
        assert 'IN (?,?,?)' in sql
        assert params == ['macos', 'windows', 'linux']

    def test_mobile_group_expands(self):
        sql, params = _platform_filter('mobile')
        assert params == ['ios', 'android']

    def test_specific_platform_single(self):
        sql, params = _platform_filter('macos')
        assert 'IN (?)' in sql
        assert params == ['macos']

    def test_case_insensitive_group(self):
        _, params = _platform_filter('DeskTop')
        assert params == ['macos', 'windows', 'linux']


# ── Integracion: /admin/funnel?platform= con BD temporal ───────────────
@pytest.fixture
def seeded_db(monkeypatch):
    """BD temporal con eventos app_opened de 3 devices en 3 plataformas."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp TEXT, "
        "device_id TEXT, event_name TEXT, platform TEXT)"
    )
    # 2 desktop (macos + windows), 1 mobile (ios). Todos app_opened, recientes.
    for dev, plat in (('d_mac', 'macos'), ('d_win', 'windows'), ('d_ios', 'ios')):
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name, platform) "
            "VALUES (datetime('now'), ?, 'app_opened', ?)",
            (dev, plat),
        )
    conn.commit()
    conn.close()
    monkeypatch.setenv('DATABASE_PATH', path)
    monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _funnel(client, platform=None):
    params = {'platform': platform} if platform else {}
    r = client.get('/admin/funnel', params=params,
                   headers={'X-Admin-Secret': _SECRET})
    assert r.status_code == 200
    return r.json()


def test_all_platforms_counts_everyone(seeded_db):
    body = _funnel(TestClient(app))
    assert body['platform'] == 'all'
    assert body['steps'][0]['devices'] == 3   # mac + win + ios


def test_desktop_excludes_mobile(seeded_db):
    body = _funnel(TestClient(app), 'desktop')
    assert body['platform'] == 'desktop'
    assert body['steps'][0]['devices'] == 2   # mac + win, NO ios


def test_mobile_only(seeded_db):
    body = _funnel(TestClient(app), 'mobile')
    assert body['steps'][0]['devices'] == 1   # solo ios


def test_specific_platform(seeded_db):
    body = _funnel(TestClient(app), 'macos')
    assert body['steps'][0]['devices'] == 1   # solo mac


# ── Pasos POR PLATAFORMA (2026-08-19) ──────────────────────────────────
#
# El embudo aplicaba los MISMOS cinco pasos a desktop y movil, con
# `import_completed` como paso clave. En movil eso mide contra un objetivo que
# no es el suyo: el DJ tiene la musica en el PC y al movil le llega por SYNC
# (el propio cliente lo dice en mobile_onboarding.dart: "CTA PRINCIPAL en movil
# = Escuchar; importar = secundario"). El sintoma que lo delataba: en movil
# `first_track_viewed` (15) superaba a `import_completed` (12) — usuarios que
# llegan a su biblioteca SIN importar. Un paso que supera al anterior no es una
# fuga, es una secuencia mal planteada.
from routes.admin_panel import _funnel_steps_for  # noqa: E402


class TestFunnelStepsPorPlataforma:
    def test_desktop_conserva_los_pasos_de_import(self):
        names = [n for n, _ in _funnel_steps_for('desktop')]
        assert 'import_started' in names
        assert 'import_completed' in names

    def test_movil_no_mide_import_como_paso(self):
        names = [n for n, _ in _funnel_steps_for('mobile')]
        assert 'import_started' not in names
        assert 'import_completed' not in names

    def test_movil_mide_vinculacion_y_sync(self):
        """Fase B: el camino real del movil es vincular + recibir biblioteca.
        Los emite el cliente desde 2.9.10; hasta entonces salen a 0."""
        names = [n for n, _ in _funnel_steps_for('mobile')]
        assert names == ['app_opened', 'onboarding_completed',
                         'device_linked', 'library_synced',
                         'first_track_viewed']

    def test_desktop_no_mide_vinculacion(self):
        # En desktop la biblioteca es local: vincular no es un paso del embudo.
        names = [n for n, _ in _funnel_steps_for('desktop')]
        assert 'device_linked' not in names
        assert 'library_synced' not in names

    def test_ios_y_android_sueltos_cuentan_como_movil(self):
        for p in ('ios', 'android', 'IOS', 'Android'):
            names = [n for n, _ in _funnel_steps_for(p)]
            assert 'import_completed' not in names, p

    def test_sin_plataforma_y_desktop_sueltos_usan_el_completo(self):
        for p in (None, 'macos', 'windows', 'linux'):
            names = [n for n, _ in _funnel_steps_for(p)]
            assert 'import_completed' in names, p

    def test_plataforma_desconocida_no_pierde_pasos(self):
        # Defensivo: un valor raro no debe silenciar pasos del embudo.
        names = [n for n, _ in _funnel_steps_for('web')]
        assert 'import_completed' in names


class TestFunnelStepsEnLaRespuesta:
    def test_movil_devuelve_sus_pasos_y_nota(self, seeded_db):
        body = _funnel(TestClient(app), 'mobile')
        events = [s['event'] for s in body['steps']]
        assert 'import_completed' not in events
        assert events == ['app_opened', 'onboarding_completed',
                          'device_linked', 'library_synced',
                          'first_track_viewed']
        assert body['steps_note']

    def test_pasos_sin_instrumentar_salen_a_cero_no_desaparecen(self, seeded_db):
        """La BD sembrada solo tiene app_opened. Los pasos que aun no emite
        ningun cliente deben salir con devices=0, NO omitirse: si se omitieran,
        el embudo parecería completo cuando en realidad falta instrumentar."""
        body = _funnel(TestClient(app), 'mobile')
        by_event = {s['event']: s['devices'] for s in body['steps']}
        assert by_event['device_linked'] == 0
        assert by_event['library_synced'] == 0
        assert by_event['app_opened'] == 1

    def test_desktop_devuelve_cinco_pasos_sin_nota(self, seeded_db):
        body = _funnel(TestClient(app), 'desktop')
        assert len(body['steps']) == 5
        assert body['steps_note'] is None

    def test_raw_sigue_trayendo_todos_los_eventos(self, seeded_db):
        # Quitar import_* de `steps` NO puede dejar de medirlos: siguen en raw.
        body = _funnel(TestClient(app), 'mobile')
        assert 'raw' in body and isinstance(body['raw'], dict)
