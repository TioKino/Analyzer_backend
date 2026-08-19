"""Dedup del pre-check cuando el motor local tiene la BD vacia.

Escenario real (Mac formateado, 2026-08-19): la BD del motor local es LOCAL a
esa maquina. Tras formatear esta vacia, asi que el pre-check por huella del
cliente fallaba SIEMPRE y toda la biblioteca se volvia a subir y analizar,
aunque Render ya tuviera cada track. Ahora el motor local pregunta a Render
antes de contestar "no analizado".
"""
import pytest
from fastapi.testclient import TestClient

from main import app
import routes.analysis_artwork as lookup

client = TestClient(app)

FP = 'a' * 32


@pytest.fixture
def render_says_yes(monkeypatch):
    """Simula el motor local: Render conoce FP, la BD local no."""
    calls = []

    def fake_fetch(fp):
        calls.append(fp)
        if fp == FP:
            return {
                'id': FP, 'filename': 'x.mp3', 'artist': 'A', 'title': 'T',
                'bpm': 128.0, 'key': 'Fm', 'camelot': '4A', 'duration': 300.0,
                'bpm_source': 'rekordbox', 'key_source': 'rekordbox',
            }
        return None

    monkeypatch.setattr(lookup, 'fetch_render_cache', fake_fetch)
    return calls


def test_precheck_uses_render_when_local_db_empty(render_says_yes):
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [FP]})
    assert r.status_code == 200
    assert r.json()['analyzed'] == [FP]
    assert render_says_yes == [FP]


def test_precheck_still_reports_unknown_fingerprints(render_says_yes):
    unknown = 'b' * 32
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [unknown]})
    assert r.json()['not_analyzed'] == [unknown]
    assert r.json()['analyzed'] == []


def test_by_fingerprint_serves_render_payload(render_says_yes):
    # Sin esto el pre-check decia "ya analizado" y este endpoint devolvia 404,
    # asi que el cliente acababa subiendo el fichero igual.
    r = client.get(f'/analysis/by-fingerprint/{FP}')
    assert r.status_code == 200
    assert r.json()['bpm_source'] == 'rekordbox'


def test_render_failure_never_breaks_the_precheck(monkeypatch):
    def boom(fp):
        raise RuntimeError('Render dormido')

    monkeypatch.setattr(lookup, 'fetch_render_cache', boom)
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [FP]})
    assert r.status_code == 200
    assert r.json()['not_analyzed'] == [FP]


def test_render_not_consulted_when_not_local_engine(monkeypatch):
    # En Render el hook es None: no debe consultarse a si mismo.
    monkeypatch.setattr(lookup, 'fetch_render_cache', None)
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [FP]})
    assert r.json()['not_analyzed'] == [FP]
