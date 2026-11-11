"""Cuántas veces se pide la valoración en la tienda (`resumen_valoraciones`,
desde el 2026-11-11).

El móvil abre la ventana de valorar del sistema tras unos aciertos de Escuchar
(`pedir_valoracion.dart` en el cliente) y lo apunta en `valoracion_pedida`.
La tienda no dice si salió ni si el DJ valoró: lo que se puede medir aquí es
cuántas veces se pide, y a quién; lo demás, en App Store Connect y Play
Console con estas fechas al lado.

    .venv/bin/python -m pytest -q test_las_valoraciones_se_miden.py
"""

import json
import tempfile

from database import AnalysisDB


def _db():
    return AnalysisDB(tempfile.mktemp(suffix='.db'))


def test_cuenta_las_peticiones_por_plataforma_y_las_primeras():
    db = _db()
    for dev, plataforma, veces in (('ios-1', 'ios', 1), ('ios-1', 'ios', 2),
                                   ('and-1', 'android', 1)):
        db.log_event(device_id=dev, event_name='valoracion_pedida',
                     props=json.dumps({'aciertos': 3, 'veces': veces}),
                     platform=plataforma)
    db.log_event(device_id='ios-2', event_name='listen_result',
                 props=json.dumps({'found': True}), platform='ios')
    r = db.resumen_valoraciones(30)
    assert r == {'dias': 30, 'pedidas': 3, 'aparatos': 2,
                 'por_plataforma': {'ios': 2, 'android': 1}, 'primera_vez': 2}


def test_sin_eventos_sale_a_cero():
    r = _db().resumen_valoraciones(30)
    assert r['pedidas'] == 0 and r['aparatos'] == 0 and r['por_plataforma'] == {}


def test_el_panel_lo_da_de_verdad():
    import main  # noqa: F401  (monta el singleton que usa el panel)
    import routes.admin_panel as panel
    with open(panel.__file__, encoding='utf-8') as fh:
        assert '"valoraciones": _valoraciones(),' in fh.read()
    r = panel._valoraciones()
    assert r is not None, 'None = embudo.sh diría que el panel falló'
    assert {'pedidas', 'aparatos', 'por_plataforma'} <= set(r)


# ------------------------------------------------- la pasada de Shazam
# Desde el 2026-11-11 está en Ajustes → Biblioteca para todos los Mac (antes,
# en el Developer Mode): para saber si alguien la usa.

def test_la_pasada_de_shazam_se_mide():
    db = _db()
    db.log_event(device_id='mac-1', event_name='shazam_pasada',
                 props=json.dumps({'accion': 'empezada', 'temas': 4000}))
    db.log_event(device_id='mac-2', event_name='shazam_pasada',
                 props=json.dumps({'accion': 'empezada', 'temas': 300}))
    db.log_event(device_id='mac-1', event_name='shazam_pasada',
                 props=json.dumps({'accion': 'aplicada', 'nombres': 412}))
    db.log_event(device_id='mac-1', event_name='shazam_pasada',
                 props=json.dumps({'accion': 'rara', 'temas': 'x'}))
    r = db.resumen_shazam_pasada(30)
    assert r == {'dias': 30, 'empezadas': 2, 'aparatos': 2, 'temas': 4300,
                 'aplicadas': 1, 'nombres': 412}


def test_la_pasada_sale_en_el_panel():
    import main  # noqa: F401
    import routes.admin_panel as panel
    with open(panel.__file__, encoding='utf-8') as fh:
        assert '"shazam_pasada": _shazam_pasada(),' in fh.read()
    assert panel._shazam_pasada() is not None
