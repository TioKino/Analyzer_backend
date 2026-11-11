"""Las zonas de cues dicen DÓNDE coinciden los DJs: los puntos que se adoptan.

Una zona junta los cues de un tipo encadenados a 5 s, con 2 s de margen: dice
«aquí pasa algo», no dónde. Para que un DJ convierta lo de la comunidad en
cues suyos con un clic (2026-11-11, `COMPETENCIA.md` punto 4) cada zona trae
sus `puntos`: los sitios donde al menos dos cuentas pusieron un cue a menos de
120 ms (la misma muestra no cae en el mismo milisegundo en un MP3 y en un
FLAC, ni en Rekordbox y en Traktor). Lo que se ata:
  - dos DJs en el mismo sitio dan un punto, en la mediana;
  - dos DJs a un compás de distancia dan una zona y NINGÚN punto;
  - una cuenta cuenta una vez (sus aparatos y sus cues repetidos);
  - un loop trae su final solo si las cuentas coinciden también en él;
  - el GET de la zona los devuelve.

    pytest test_los_puntos_de_las_zonas.py -v
"""

import json
import tempfile
import uuid

from fastapi.testclient import TestClient

from community_cues_endpoint import (
    TOLERANCIA_DE_UN_PUNTO_MS, aggregate_cues_into_zones, puntos_de_la_zona)
from database import AnalysisDB
from main import app, db

client = TestClient(app)


def _c(dev, ms, fin=None, tipo='hotcue'):
    return {'cue_type': tipo, 'device_id': dev, 'position_ms': ms,
            'end_position_ms': fin, 'note': None}


def test_dos_djs_en_el_mismo_sitio_dan_un_punto_en_la_mediana():
    zonas = aggregate_cues_into_zones([_c('a', 62000), _c('b', 62040)])
    assert len(zonas) == 1
    assert zonas[0]['puntos'] == [{'pos': 62.02, 'djs': 2}]


def test_a_un_compas_hay_zona_pero_no_punto():
    # 62,0 s y 63,9 s (un compás a 126): la zona sale (están a menos de 5 s),
    # pero ningún sitio lo marcaron dos DJs. Su centro no es de nadie.
    zonas = aggregate_cues_into_zones([_c('a', 62000), _c('b', 63900)])
    assert len(zonas) == 1 and zonas[0]['dj_count'] == 2
    assert zonas[0]['puntos'] == []


def test_la_tolerancia_es_la_de_un_mismo_golpe():
    assert TOLERANCIA_DE_UN_PUNTO_MS == 120
    dentro = puntos_de_la_zona([_c('a', 10000), _c('b', 10120)])
    fuera = puntos_de_la_zona([_c('a', 10000), _c('b', 10121)])
    assert [p['djs'] for p in dentro] == [2]
    assert fuera == []


def test_una_cuenta_cuenta_una_vez():
    # El mismo DJ con tres cues en el sitio (o con el escritorio y el móvil)
    # no hace un punto, ni mueve el de los demás.
    assert puntos_de_la_zona([_c('a', 5000), _c('a', 5010), _c('a', 5020)]) == []
    cuentas = {'mac': 'u1', 'movil': 'u1'}
    assert puntos_de_la_zona([_c('mac', 5000), _c('movil', 5000)], cuentas) == []
    # Tres cues de «a» en 5,000 y uno de «b» en 5,100: por cuenta es la
    # mediana de 5,000 y 5,100. Contando cues saldría 5,000, el sitio de «a».
    puntos = puntos_de_la_zona(
        [_c('a', 5000), _c('a', 5000), _c('a', 5000), _c('b', 5100)])
    assert puntos == [{'pos': 5.05, 'djs': 2}]


def test_varios_sitios_en_una_zona_y_el_mas_votado_primero():
    cues = [_c('a', 60000), _c('b', 60030), _c('c', 60060),
            _c('a', 64000), _c('b', 64010),
            _c('d', 62000)]  # solo
    zonas = aggregate_cues_into_zones(cues)
    assert len(zonas) == 1
    assert zonas[0]['puntos'] == [{'pos': 60.03, 'djs': 3},
                                  {'pos': 64.005, 'djs': 2}]


def test_un_loop_trae_su_final_solo_si_coinciden():
    iguales = puntos_de_la_zona([_c('a', 1000, 2875, 'loop'),
                                 _c('b', 1020, 2900, 'loop')])
    assert iguales == [{'pos': 1.01, 'djs': 2, 'fin': 2.888}]
    # 4 beats y 8 beats: la mediana (6) no sería de nadie.
    distintos = puntos_de_la_zona([_c('a', 1000, 2875, 'loop'),
                                   _c('b', 1020, 4770, 'loop')])
    assert distintos == [{'pos': 1.01, 'djs': 2}]


def test_cada_tipo_por_su_lado():
    zonas = aggregate_cues_into_zones(
        [_c('a', 30000), _c('b', 30000, 31875, 'loop')])
    assert zonas == []


def test_el_get_de_la_zona_trae_los_puntos():
    fp = uuid.uuid4().hex
    c = db.conn.cursor()
    try:
        for dev, ms in (('puntos_a_' + fp[:6], 64000), ('puntos_b_' + fp[:6], 64050)):
            c.execute(
                'INSERT INTO community_cues (fingerprint, device_id, cue_type, '
                "position_ms, created_at) VALUES (?, ?, 'hotcue', ?, 'x')",
                (fp, dev, ms))
        db.conn.commit()
        zonas = client.get(f'/community/cues/{fp}').json()['zones']
        assert len(zonas) == 1
        assert zonas[0]['puntos'] == [{'pos': 64.025, 'djs': 2, 'fin': None}]
    finally:
        c.execute('DELETE FROM community_cues WHERE fingerprint = ?', (fp,))
        db.conn.commit()


# ------------------------------------------------- lo que se adopta, medido
# El cliente manda `cues_adoptados` ({accion, cues, desde}) y el panel lo suma
# en `cues_adoptados` de /admin/telemetry: si se adopta y si se deshace.

def test_lo_adoptado_se_mide():
    adb = AnalysisDB(tempfile.mktemp(suffix='.db'))
    for dev, props in (
            ('mac-1', {'accion': 'adoptar', 'cues': 3, 'desde': 'barra'}),
            ('mac-1', {'accion': 'deshacer', 'cues': 3, 'desde': 'barra'}),
            ('pc-1', {'accion': 'adoptar', 'cues': 1, 'desde': 'menu'}),
            ('pc-1', {'accion': 'rara', 'cues': 'x'})):
        adb.log_event(device_id=dev, event_name='cues_adoptados',
                      props=json.dumps(props))
    assert adb.resumen_cues_adoptados(30) == {
        'dias': 30, 'veces': 2, 'cues': 4, 'aparatos': 2,
        'desde': {'barra': 1, 'menu': 1}, 'deshechas': 1, 'cues_deshechos': 3}


def test_lo_adoptado_sale_en_el_panel():
    import routes.admin_panel as panel
    with open(panel.__file__, encoding='utf-8') as fh:
        assert '"cues_adoptados": _cues_adoptados(),' in fh.read()
    assert panel._cues_adoptados() is not None
