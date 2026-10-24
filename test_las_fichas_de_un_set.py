"""Las fichas de un set del modo set, en UN lote (desde el 2026-10-24).

El modo set guarda el tracklist sin la ficha de cada tema, y pedirla por
/search-analyzed era una llamada por tema. `POST /search-analyzed/batch`
contesta un lote con la MISMA regla (`buscar_analizado`: ISRC, nombre exacto,
mismo tema y versión, nunca una fila con BPM 0) y una ficha CORTA, en el mismo
orden en que llegan los temas.
"""

from fastapi.testclient import TestClient

from main import app, db as main_db
from routes.search import CAMPOS_DE_LA_FICHA_CORTA, MAX_FICHAS_POR_LOTE

client = TestClient(app)


def _seed(fp, artist, title, bpm=128.0, camelot='8A', isrc=None):
    fila = {
        'id': fp, 'filename': f'{fp}.mp3', 'artist': artist, 'title': title,
        'duration': 300.0, 'bpm': bpm, 'key': 'A min', 'camelot': camelot,
        'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
        'fingerprint': fp,
    }
    if isrc:
        fila['isrc'] = isrc
    main_db.save_track(fila)


def _lote(temas):
    r = client.post('/search-analyzed/batch', json={'temas': temas})
    assert r.status_code == 200, r.text
    return r.json()['fichas']


def test_un_lote_en_el_mismo_orden_y_null_donde_no_hay():
    _seed('zzlote_1', 'ZZLoteUno', 'ZZTemaUno', bpm=131, camelot='9A')
    _seed('zzlote_2', 'ZZLoteDos', 'ZZTemaDos (Extended Mix)', bpm=126)
    fichas = _lote([
        {'artist': 'ZZLoteDos', 'title': 'ZZTemaDos'},
        {'artist': 'ZZNadie', 'title': 'ZZNoExiste'},
        {'artist': 'ZZLoteUno', 'title': 'ZZTemaUno'},
    ])
    assert len(fichas) == 3
    assert fichas[0]['bpm'] == 126
    assert fichas[1] is None
    assert fichas[2]['bpm'] == 131
    assert fichas[2]['camelot'] == '9A'


def test_la_ficha_va_corta():
    _seed('zzlote_3', 'ZZLoteTres', 'ZZTemaTres')
    ficha = _lote([{'artist': 'ZZLoteTres', 'title': 'ZZTemaTres'}])[0]
    assert set(ficha) <= set(CAMPOS_DE_LA_FICHA_CORTA)
    assert 'analysis_json' not in ficha
    assert ficha['energy_dj'] == 7


def test_el_isrc_manda_como_en_la_ficha_de_uno():
    _seed('zzlote_4', 'ZZOtroNombre', 'ZZOtroTitulo', bpm=124,
          isrc='ZZISRC0000001')
    ficha = _lote([{'artist': 'Shazam Dice', 'title': 'Otra Cosa',
                    'isrc': 'ZZISRC0000001'}])[0]
    assert ficha is not None
    assert ficha['bpm'] == 124


def test_una_deteccion_con_bpm_cero_no_es_ficha():
    _seed('zzlote_5', 'ZZPendiente', 'ZZSinAnalizar', bpm=0)
    assert _lote([{'artist': 'ZZPendiente', 'title': 'ZZSinAnalizar'}]) == [None]


def test_un_tema_sin_nombre_no_tumba_el_lote():
    _seed('zzlote_6', 'ZZLoteSeis', 'ZZTemaSeis')
    fichas = _lote([{'artist': '', 'title': ''},
                    {'artist': 'ZZLoteSeis', 'title': 'ZZTemaSeis'}])
    assert fichas[0] is None
    assert fichas[1]['artist'] == 'ZZLoteSeis'


def test_demasiados_temas_413():
    temas = [{'artist': 'a', 'title': 'b'}] * (MAX_FICHAS_POR_LOTE + 1)
    r = client.post('/search-analyzed/batch', json={'temas': temas})
    assert r.status_code == 413


def test_va_fuera_del_event_loop():
    src = open('routes/search.py', encoding='utf-8').read()
    i = src.index('async def search_analyzed_batch')
    assert 'run_in_threadpool(fichas_de' in src[i:i + 800]
