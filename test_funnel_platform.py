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
        # `macos-mas` y `macos-dmg` estan dentro aunque el emisor de eventos
        # todavia mande `macos` a secas: `tracks.platform` y `events.platform`
        # son dos vocabularios para la misma palabra, y el dia que se alineen
        # los usuarios del Mac App Store se caerian fuera del grupo sin un solo
        # error. Ver la nota de _PLATFORM_GROUPS.
        sql, params = _platform_filter('desktop')
        assert 'IN (?,?,?,?,?)' in sql
        assert params == ['macos', 'macos-mas', 'macos-dmg', 'windows', 'linux']

    def test_el_mac_app_store_cuenta_como_desktop(self):
        _, params = _platform_filter('desktop')
        assert 'macos-mas' in params, 'MAS es escritorio, y es la build sin motor local'
        assert 'macos-dmg' in params

    def test_mobile_group_expands(self):
        sql, params = _platform_filter('mobile')
        assert params == ['ios', 'android']

    def test_specific_platform_single(self):
        sql, params = _platform_filter('macos')
        assert 'IN (?)' in sql
        assert params == ['macos']

    def test_case_insensitive_group(self):
        _, params = _platform_filter('DeskTop')
        assert params == ['macos', 'macos-mas', 'macos-dmg', 'windows', 'linux']


# ── Integracion: /admin/funnel?platform= con BD temporal ───────────────
#
# Desde el 2026-11-25 (embudo v3, `embudo.py`) el embudo es la cohorte MADURA:
# altas de hace 8 a 37 dias, cada paso dado en los 7 dias siguientes. Por eso
# los aparatos se dan de alta hace 10 dias, no hoy: un alta de hoy aun no ha
# tenido tiempo de dar ningun paso, y medirla bajaba la conversion cuando
# llegaba mas gente (el fallo 1 de la cabecera de `embudo.py`).
import embudo  # noqa: E402


@pytest.fixture
def seeded_db(monkeypatch):
    """BD temporal: 3 aparatos de hace 10 dias en 3 plataformas, con
    app_opened, y uno de hoy que aun no cuenta."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp TEXT, "
        "device_id TEXT, event_name TEXT, platform TEXT)"
    )
    conn.execute(
        "CREATE TABLE device_first_seen (device_id TEXT PRIMARY KEY, "
        "first_day TEXT NOT NULL, first_platform TEXT, first_app_version TEXT)"
    )
    for dev, plat, dias in (('d_mac', 'macos', 10), ('d_win', 'windows', 10),
                            ('d_ios', 'ios', 10), ('d_hoy', 'ios', 0)):
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name, platform) "
            "VALUES (datetime('now', ?), ?, 'app_opened', ?)",
            (f'-{dias} days', dev, plat),
        )
        conn.execute(
            "INSERT INTO device_first_seen (device_id, first_day, first_platform) "
            "VALUES (?, date('now', ?), ?)",
            (dev, f'-{dias} days', plat),
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
    assert body['steps'][0]['devices'] == 3   # mac + win + ios, NO el de hoy


def test_desktop_excludes_mobile(seeded_db):
    body = _funnel(TestClient(app), 'desktop')
    assert body['platform'] == 'desktop'
    assert body['steps'][0]['devices'] == 2   # mac + win, NO ios


def test_mobile_only(seeded_db):
    body = _funnel(TestClient(app), 'mobile')
    assert body['steps'][0]['devices'] == 1   # solo ios (el de hoy no madura)


def test_specific_platform(seeded_db):
    body = _funnel(TestClient(app), 'macos')
    assert body['steps'][0]['devices'] == 1   # solo mac


def test_la_respuesta_dice_que_mide(seeded_db):
    body = _funnel(TestClient(app))
    assert body['basis'] == 'cohorte_madura'
    assert body['definicion'] == embudo.DEFINICION
    assert body['cohort_size'] == 3
    assert body['embudo']['grupos']['escritorio']['cohorte'] == 2
    assert body['embudo']['grupos']['movil']['cohorte'] == 1


class TestFunnelStepsEnLaRespuesta:
    def test_los_pasos_son_los_mismos_en_las_dos_plataformas(self, seeded_db):
        for p in ('mobile', 'desktop', None):
            body = _funnel(TestClient(app), p)
            assert [s['event'] for s in body['steps']] == list(embudo.PASOS), p

    def test_la_nota_es_solo_del_movil(self, seeded_db):
        assert _funnel(TestClient(app), 'mobile')['steps_note']
        assert _funnel(TestClient(app), 'desktop')['steps_note'] is None

    def test_pasos_sin_eventos_salen_a_cero_no_desaparecen(self, seeded_db):
        """La BD sembrada solo tiene app_opened. Los pasos que nadie ha dado
        salen con devices=0, NO se omiten: si se omitieran, el embudo pareceria
        completo cuando en realidad falta instrumentar."""
        body = _funnel(TestClient(app), 'mobile')
        by_event = {s['event']: s['devices'] for s in body['steps']}
        assert by_event['alta'] == 1
        assert by_event['abrio'] == 1
        assert by_event['bienvenida_vista'] == 0
        assert by_event['valor'] == 0

    def test_ningun_paso_supera_al_anterior(self, seeded_db):
        body = _funnel(TestClient(app))
        n = [s['devices'] for s in body['steps']]
        assert n == sorted(n, reverse=True)

    def test_raw_sigue_trayendo_todos_los_eventos(self, seeded_db):
        body = _funnel(TestClient(app), 'mobile')
        assert 'raw' in body and isinstance(body['raw'], dict)
