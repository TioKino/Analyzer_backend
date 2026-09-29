"""¿De cuántos aparatos vienen los que entran sin huella?

El 2026-09-27 salieron en 7 días 289 fallbacks de `/analyze` y 404 análisis OK
sin huella. Con solo el conteo, «un DJ con 300 ficheros rotos» y «300 DJs con
uno cada uno» se leen igual, y piden cosas opuestas: lo primero es un lote y se
olvida, lo segundo es un fallo del producto. La única forma de saberlo fue abrir
los logs de Render.

El aparato ya estaba apuntado en `track_analyzers` (el envoltorio de `/analyze`
llama a `registrar_analista` en todas sus salidas) y nadie lo cruzaba. Y el
fallback no sellaba la plataforma. Estos tests atan las dos cosas.

    pytest test_de_cuantos_aparatos_viene_el_hueco.py -v
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from database import AnalysisDB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield AnalysisDB(db_path=path)
    try:
        os.unlink(path)
    except OSError:
        pass


def _hace(dias):
    return (datetime.now() - timedelta(days=dias)).isoformat()


def _fila(db, huella, *, fallback=False, plataforma=None, dias=1,
          aparato=None):
    fila = {
        'id': huella,
        'fingerprint': huella,
        'filename': f'{huella}.mp3',
        'duration': 0 if fallback else 300.0,
        'bpm': 0 if fallback else 128.0,
        'key': None if fallback else 'Am',
        'camelot': None if fallback else '8A',
        'energy_dj': 5,
        'genre': 'Techno',
        'track_type': 'club',
        'chromaprint': None,
        'analyzed_at': _hace(dias),
    }
    if fallback:
        fila['analysis_status'] = 'failed'
    if plataforma:
        fila['platform'] = plataforma
    db.save_track(fila)
    if aparato:
        db.registrar_analista(huella, aparato)


def test_un_dj_con_muchos_ficheros_rotos_es_UN_aparato(db):
    for i in range(5):
        _fila(db, f'roto{i}', fallback=True, plataforma='windows',
              aparato='pc-de-uno')
    d = db.acoustic_gap_breakdown()['devices_last_7d']['failed_fallback']
    assert d['devices'] == 1
    assert d['by_platform'] == {'windows': 5}
    assert d['rows_without_device'] == 0


def test_muchos_djs_son_muchos_aparatos(db):
    for i in range(3):
        _fila(db, f'roto{i}', fallback=True, plataforma='ios',
              aparato=f'tel{i}')
    d = db.acoustic_gap_breakdown()['devices_last_7d']['failed_fallback']
    assert d['devices'] == 3


def test_fallback_y_analizado_ok_van_aparte(db):
    _fila(db, 'roto', fallback=True, plataforma='windows', aparato='a')
    _fila(db, 'ok1', plataforma='macos-dmg', aparato='b')
    _fila(db, 'ok2', plataforma='macos-dmg', aparato='c')
    ap = db.acoustic_gap_breakdown()['devices_last_7d']
    assert ap['failed_fallback']['devices'] == 1
    assert ap['analyzed_ok']['devices'] == 2
    assert ap['analyzed_ok']['by_platform'] == {'macos-dmg': 2}


def test_lo_de_hace_mas_de_7_dias_sale_en_la_de_30(db):
    # Una rafaga que ya paso (la del 27-sep sale del corte de 7 dias antes de
    # la lectura siguiente) se sigue pudiendo leer un mes.
    _fila(db, 'viejo', fallback=True, aparato='a', dias=20)
    gap = db.acoustic_gap_breakdown()
    assert gap['devices_last_7d']['failed_fallback']['devices'] == 0
    assert gap['devices_last_30d']['failed_fallback']['devices'] == 1


def test_filas_sin_aparato_se_cuentan_aparte(db):
    # Un cliente que no manda X-Device-Id: la fila existe, el aparato no se
    # sabe. Contarla como «cero aparatos» mentiría.
    _fila(db, 'roto', fallback=True)
    d = db.acoustic_gap_breakdown()['devices_last_7d']['failed_fallback']
    assert d['devices'] == 0
    assert d['rows_without_device'] == 1
    assert d['by_platform'] == {'unknown': 1}


def test_recognize_no_entra(db):
    # `/recognize` no pasa por `/analyze`: no es ninguno de los dos cubos.
    db.save_track({'id': 'r', 'fingerprint': 'r', 'filename': 'r.mp3',
                   'duration': 12.0, 'bpm': 0, 'key': None, 'camelot': None,
                   'energy_dj': 5, 'genre': 'Techno', 'track_type': 'club',
                   'chromaprint': None, 'analyzed_at': _hace(1),
                   'analysis_status': 'recognize_only'})
    ap = db.acoustic_gap_breakdown()['devices_last_7d']
    assert ap['analyzed_ok']['by_platform'] == {}
    assert ap['failed_fallback']['by_platform'] == {}


def test_el_fallback_de_analyze_sella_la_plataforma():
    # Sin esto, `by_platform` del fallback sería siempre `unknown`.
    src = open(os.path.join(os.path.dirname(__file__), 'main.py'),
               encoding='utf-8').read()
    i = src.index("track_data['analysis_status'] = 'failed'")
    antes_de_guardar = src[i:src.index('db.save_track(track_data)', i)]
    assert "track_data['platform'] = client_platform(request)" in antes_de_guardar
    # Y el motor sigue SIN sellarse, que es otra decisión (ver /analyze).
    assert "track_data['engine_source']" not in antes_de_guardar
