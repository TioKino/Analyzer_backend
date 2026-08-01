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
