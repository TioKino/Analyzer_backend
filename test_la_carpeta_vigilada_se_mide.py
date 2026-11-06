"""La CARPETA VIGILADA del escritorio se mide (`resumen_carpeta_vigilada`,
desde el 2026-11-06).

El escritorio vigila una carpeta que elige el DJ y lo que cae en ella entra
solo en la biblioteca (`carpeta_vigilada.dart` en el cliente). Sin esto no se
sabría si alguien la usa ni si le entra algo: cuántos aparatos la tienen (por
su último `carpeta_vigilada`) y cuántos temas entrega al import
(`carpeta_vigilada_entran`).

    .venv/bin/python -m pytest -q test_la_carpeta_vigilada_se_mide.py
"""

import json
import tempfile

from database import AnalysisDB


def _db():
    return AnalysisDB(tempfile.mktemp(suffix='.db'))


def _evento(db, dev, nombre, props):
    db.log_event(device_id=dev, event_name=nombre, props=json.dumps(props))


def test_cuenta_quien_la_tiene_por_su_ultimo_evento_y_lo_que_entra():
    db = _db()
    _evento(db, 'mac-1', 'carpeta_vigilada', {'accion': 'elegida'})
    _evento(db, 'pc-1', 'carpeta_vigilada', {'accion': 'elegida'})
    _evento(db, 'pc-1', 'carpeta_vigilada', {'accion': 'dejada'})
    # Otro la deja y la vuelve a elegir: cuenta como que la tiene.
    _evento(db, 'mac-2', 'carpeta_vigilada', {'accion': 'dejada'})
    _evento(db, 'mac-2', 'carpeta_vigilada', {'accion': 'elegida'})
    _evento(db, 'mac-1', 'carpeta_vigilada_entran', {'n': 3})
    _evento(db, 'mac-1', 'carpeta_vigilada_entran', {'n': 1})
    _evento(db, 'pc-1', 'carpeta_vigilada_entran', {'n': 12})
    # Lo que no se entiende no suma.
    _evento(db, 'pc-1', 'carpeta_vigilada_entran', {'n': 'muchos'})
    _evento(db, 'pc-1', 'carpeta_vigilada', {'accion': 'otra'})
    _evento(db, 'mac-1', 'otra_cosa', {'n': 99})

    r = db.resumen_carpeta_vigilada(30)
    assert r == {
        'dias': 30,
        'con_carpeta': 2,     # mac-1 y mac-2
        'la_dejaron': 1,      # pc-1
        'entran': 16,
        'aparatos': 2,
        'pasadas': 3,
    }


def test_sin_eventos_sale_a_cero():
    r = _db().resumen_carpeta_vigilada(30)
    assert r['con_carpeta'] == 0 and r['entran'] == 0 and r['aparatos'] == 0


def test_el_panel_lo_da():
    src = open('routes/admin_panel.py', encoding='utf-8').read()
    assert '"carpeta_vigilada": _carpeta_vigilada(),' in src
    assert '_get_db().resumen_carpeta_vigilada(30)' in src
