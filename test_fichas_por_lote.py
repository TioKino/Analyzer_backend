"""Las fichas de lo ya analizado se piden por LOTE (2026-10-06).

Tras el pre-check por huella, el import pedía la ficha de cada acierto con un
`GET /analysis/by-fingerprint/{huella}`: en un Mac recién formateado con 5.000
temas que Render ya tiene, 5.000 viajes seguidos, y con motor local cada uno
pasando por él. `POST /analysis/by-fingerprint/batch` da las de hasta 100 en
una petición, la MISMA ficha que el GET, y el motor local completa con Render
lo que no tiene en otra sola petición.

    pytest test_fichas_por_lote.py -v
"""

import uuid

import pytest
from fastapi.testclient import TestClient

import main
import routes.analysis_artwork as lookup

client = TestClient(main.app)


def _fila(**extra):
    fp = uuid.uuid4().hex
    fila = {
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3', 'bpm': 128.0,
        'key': 'Am', 'camelot': '8A', 'duration': 300, 'energy_dj': 7,
        'genre': 'Techno', 'track_type': 'peak_time', 'bpm_source': 'analysis',
        'analysis_version': main.ANALYSIS_VERSION,
    }
    fila.update(extra)
    main.db.save_track(fila)
    return fp


@pytest.fixture
def en_render(monkeypatch):
    monkeypatch.setattr(lookup, 'fichas_en_render', None)


def test_EL_CASO_la_misma_ficha_que_el_get(en_render):
    a, b = _fila(), _fila(bpm=140.0)
    desconocida = uuid.uuid4().hex
    r = client.post('/analysis/by-fingerprint/batch',
                    json={'fingerprints': [a, b, desconocida]})
    assert r.status_code == 200
    fichas = r.json()['fichas']
    assert set(fichas) == {a, b}, 'lo que no hay no sale'
    for fp in (a, b):
        assert fichas[fp] == client.get(f'/analysis/by-fingerprint/{fp}').json()


def test_lo_importado_de_la_comunidad_tambien_va(en_render):
    """La ficha del lote lleva lo mejor de la memoria colectiva, como el GET."""
    fp = _fila(bpm=127.0)
    main.db.guardar_lo_importado('devA', [
        {'fingerprint': fp, 'source': 'rekordbox', 'bpm': 128.0}])
    ficha = client.post('/analysis/by-fingerprint/batch',
                        json={'fingerprints': [fp]}).json()['fichas'][fp]
    assert ficha['bpm'] == 128.0
    assert ficha['bpm_source'] == 'rekordbox'


def test_los_registros_antiguos_casan_por_id(en_render):
    viejo = uuid.uuid4().hex
    main.db.save_track({
        'id': viejo, 'fingerprint': None, 'filename': f'{viejo}.mp3',
        'bpm': 125.0, 'key': 'Fm', 'duration': 300, 'energy_dj': 6,
        'genre': 'House', 'track_type': 'groove',
        'analysis_version': main.ANALYSIS_VERSION,
    })
    fichas = client.post('/analysis/by-fingerprint/batch',
                         json={'fingerprints': [viejo]}).json()['fichas']
    assert fichas[viejo]['bpm'] == 125.0


def test_tope_de_100(en_render):
    r = client.post('/analysis/by-fingerprint/batch',
                    json={'fingerprints': [uuid.uuid4().hex for _ in range(101)]})
    assert r.status_code == 400


def test_el_motor_local_completa_con_render_en_una_peticion(monkeypatch):
    local = _fila()
    lotes = []

    def de_render(fps):
        lotes.append(list(fps))
        return {fps[0]: {'bpm': 130.0, 'key': 'Fm', 'fingerprint': fps[0]}}

    monkeypatch.setattr(lookup, 'fichas_en_render', de_render)
    otras = [uuid.uuid4().hex for _ in range(30)]
    fichas = client.post('/analysis/by-fingerprint/batch',
                         json={'fingerprints': [local] + otras}).json()['fichas']
    assert len(lotes) == 1 and lotes[0] == otras, 'lo local no se pregunta'
    assert set(fichas) == {local, otras[0]}


def test_si_render_no_contesta_el_lote_sale_igual(monkeypatch):
    local = _fila()

    def boom(fps):
        raise RuntimeError('Render dormido')

    monkeypatch.setattr(lookup, 'fichas_en_render', boom)
    r = client.post('/analysis/by-fingerprint/batch',
                    json={'fingerprints': [local, uuid.uuid4().hex]})
    assert r.status_code == 200
    assert list(r.json()['fichas']) == [local]


def test_el_motor_local_usa_la_vara_del_get(monkeypatch):
    """Lo que `_fetch_render_cache` no adoptaría (fallido, otra versión) el
    lote tampoco."""
    buena, fallida, vieja = 'a' * 32, 'b' * 32, 'c' * 32
    enviado = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'fichas': {
                buena: {'bpm': 128.0, 'key': 'Am',
                        'analysis_version': main.ANALYSIS_VERSION},
                fallida: {'bpm': 0, 'key': None, 'analysis_status': 'failed'},
                vieja: {'bpm': 128.0, 'key': 'Am', 'analysis_version': '0-vieja'},
            }}

    def falso_post(url, json=None, timeout=None):
        enviado.update(url=url, json=json)
        return _Resp()

    monkeypatch.setattr(main.requests, 'post', falso_post)
    r = main._fichas_en_render([buena, fallida, vieja])
    assert set(r) == {buena}
    assert enviado['url'].endswith('/analysis/by-fingerprint/batch')
    assert enviado['json'] == {'fingerprints': [buena, fallida, vieja]}


def test_el_lote_no_bloquea_el_worker():
    src = open('routes/analysis_artwork.py', encoding='utf-8').read()
    i = src.index('async def fichas_por_huella(')
    cuerpo = src[i:src.index('\n\n\n', i)]
    assert 'await run_in_threadpool(_de_esta_bd)' in cuerpo
    assert 'await run_in_threadpool(fichas_en_render' in cuerpo
