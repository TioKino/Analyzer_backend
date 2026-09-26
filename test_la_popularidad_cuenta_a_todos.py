"""«N DJs lo han analizado» cuenta a todos los que tienen el tema.

Hasta el 2026-09-26 la popularidad solo sumaba en el análisis NUEVO del
servidor que analizaba (`increment_popularity`, dentro de /analyze). Se quedaban
fuera los dos casos más comunes:
  - el segundo DJ con el mismo fichero: cae en la caché y no sumaba;
  - todo lo que analiza un motor local (EXE y DMG): su popularidad se quedaba
    en su propia BD, y `/cache-analysis` no la tocaba.

    pytest test_la_popularidad_cuenta_a_todos.py -v
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from database import AnalysisDB


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / 'analysis.db'))


def _fp():
    return uuid.uuid4().hex


def test_un_acierto_de_cache_suma_un_dj_sin_sumar_un_analisis(db):
    fp = _fp()
    db.increment_popularity(fp, 'devA')            # el que lo analizó
    assert db.registrar_analista(fp, 'devB') is True
    pop = db.get_track_popularity(fp)
    assert pop['dj_count'] == 2
    assert pop['analysis_count'] == 1, 'reimportar no es analizar otra vez'


def test_volver_a_subirlo_no_suma_nada(db):
    fp = _fp()
    db.registrar_analista(fp, 'devB')
    assert db.registrar_analista(fp, 'devB') is False
    assert db.get_track_popularity(fp)['dj_count'] == 1


def test_sin_aparato_no_cuenta(db):
    fp = _fp()
    assert db.registrar_analista(fp, '') is False
    assert db.get_track_popularity(fp)['dj_count'] == 0


@pytest.fixture
def app_mod(monkeypatch):
    import main

    monkeypatch.setattr(main, '_WRITE_AUTH_SECRET', 'test-write-secret')
    return main


def _cache(app_mod, fp, analista, firmar=True):
    body = json.dumps({
        'fingerprint': fp, 'filename': 'x.mp3', 'artist': 'A', 'title': 'T',
        'duration': 300, 'bpm': 128.0, 'analista': analista,
    }).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if firmar:
        headers.update(app_mod._sign_write_payload(body))
    return TestClient(app_mod.app).post('/cache-analysis', content=body,
                                        headers=headers)


def test_lo_que_analiza_un_motor_local_cuenta_en_render(app_mod):
    fp = _fp()
    assert _cache(app_mod, fp, 'dev-motor-1').status_code == 200
    # Otro motor local con el mismo fichero: Render ya lo tiene («exists»),
    # pero este DJ también lo ha analizado.
    _cache(app_mod, fp, 'dev-motor-2')
    assert app_mod.db.get_track_popularity(fp)['dj_count'] == 2


def test_sin_firma_no_cuenta(app_mod):
    fp = _fp()
    _cache(app_mod, fp, 'dev-x', firmar=False)
    assert app_mod.db.get_track_popularity(fp)['dj_count'] == 0


def test_analizar_en_render_cuenta_tambien_los_aciertos_de_cache(app_mod, monkeypatch):
    from test_lo_importado_llega_a_todos import _resultado

    fp = _fp()

    async def _acierto(request, file, force, force_audd):
        return _resultado(fp)

    monkeypatch.setattr(app_mod, '_analizar', _acierto)
    monkeypatch.setattr(app_mod, '_mejorar_con_la_comunidad', lambda *a, **k: None)
    c = TestClient(app_mod.app)
    for aparato in ('dev-1', 'dev-2', 'dev-2'):
        c.post('/analyze', files={'file': ('x.mp3', b'abc', 'audio/mpeg')},
               headers={'X-Device-Id': aparato})
    assert app_mod.db.get_track_popularity(fp)['dj_count'] == 2
