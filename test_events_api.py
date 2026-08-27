"""Tests de integracion HTTP del pipe de eventos: POST /client-event,
GET /admin/funnel y GET /admin/retention. Garantizan la capa completa
(routing + auth + shape), no solo los helpers de BD (test_events.py).

Historicamente en este proyecto los bugs de WIRING (endpoint duplicado sin
registrar, auth ausente) han sido los mas graves — de ahi estos tests.
"""

import pytest
from fastapi.testclient import TestClient

from main import app

_SECRET = 'test-admin-secret-1234567890'


@pytest.fixture
def client():
    return TestClient(app)


class TestClientEventEndpoint:
    def test_valid_event_returns_202(self, client):
        r = client.post(
            '/client-event',
            json={'event_name': 'app_opened', 'platform': 'ios'},
            headers={'X-Device-Id': 'test_dev_api'},
        )
        assert r.status_code == 202
        assert r.json()['ok'] is True

    def test_event_with_props_202(self, client):
        r = client.post(
            '/client-event',
            json={'event_name': 'import_completed', 'props': {'count': 12}},
            headers={'X-Device-Id': 'test_dev_api'},
        )
        assert r.status_code == 202

    def test_missing_event_name_422(self, client):
        # event_name es obligatorio en el modelo -> validacion pydantic.
        r = client.post('/client-event', json={'platform': 'ios'})
        assert r.status_code == 422


class TestAdminFunnelEndpoint:
    def test_requires_admin_secret(self, client, monkeypatch):
        monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
        r = client.get('/admin/funnel')
        assert r.status_code == 401

    def test_wrong_secret_rejected(self, client, monkeypatch):
        monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
        r = client.get('/admin/funnel', headers={'X-Admin-Secret': 'nope'})
        assert r.status_code == 401

    def test_ok_shape(self, client, monkeypatch):
        monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
        r = client.get('/admin/funnel', headers={'X-Admin-Secret': _SECRET})
        assert r.status_code == 200
        body = r.json()
        assert body['window_days'] == 30
        assert isinstance(body['steps'], list)
        assert 'raw' in body
        # El embudo declara sus pasos canonicos aunque no haya datos.
        # Seis desde el 2026-08-27: `onboarding_shown` entro DELANTE de
        # `onboarding_completed`. Era el denominador que faltaba — `app_opened`
        # se emite una vez por dispositivo para siempre, asi que medir la
        # finalizacion contra el incluia a gente que nunca vio la pantalla.
        assert len(body['steps']) == 6
        assert body['steps'][0]['event'] == 'app_opened'


class TestAdminRetentionEndpoint:
    def test_requires_admin_secret(self, client, monkeypatch):
        monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
        r = client.get('/admin/retention')
        assert r.status_code == 401

    def test_ok_shape(self, client, monkeypatch):
        monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
        r = client.get('/admin/retention', headers={'X-Admin-Secret': _SECRET})
        assert r.status_code == 200
        cohorts = r.json()['cohorts']
        # Puede estar vacio (sin datos) o traer d1/d7/d28; si trae, con la forma.
        for key in cohorts:
            assert set(cohorts[key].keys()) == {'cohort', 'retained', 'rate'}
