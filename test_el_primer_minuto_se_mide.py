"""EL PRIMER MINUTO se mide (`resumen_primer_minuto`, desde el 2026-11-10).

Al terminar un import en el escritorio, el resumen dice lo que DJ Analyzer ya
sabía de esos temas —los ya analizados, la rejilla de tres DJs, el BPM o la
tonalidad de un programa de DJ, los repetidos y los de nombre sucio— y al
cerrarlo el cliente manda `primer_minuto` (`el_primer_minuto.dart`). Sin esto
no se sabría cuánto tiene la app que enseñar al que llega ni si el DJ hace
algo con ello.

    .venv/bin/python -m pytest -q test_el_primer_minuto_se_mide.py
"""

import json
import tempfile

from database import AnalysisDB


def _db():
    return AnalysisDB(tempfile.mktemp(suffix='.db'))


def _evento(db, dev, props, nombre='primer_minuto'):
    db.log_event(device_id=dev, event_name=nombre, props=json.dumps(props))


def _cierre(temas=10, conocidos=0, rejillas=0, del_programa=0, sin_nombre=0,
            repetidos=0, accion='cerrar'):
    return {
        'temas': temas, 'conocidos': conocidos, 'rejillas': rejillas,
        'del_programa': del_programa, 'sin_nombre': sin_nombre,
        'repetidos': repetidos, 'accion': accion,
    }


def test_suma_las_lineas_y_las_acciones():
    db = _db()
    _evento(db, 'mac-1', _cierre(temas=500, conocidos=312, rejillas=28,
                                 repetidos=12, sin_nombre=18,
                                 accion='identificar'))
    _evento(db, 'pc-1', _cierre(temas=40, del_programa=5, repetidos=-1,
                                accion='repetidos'))
    # Nada que contar: cuenta como resumen, no como «con algo».
    _evento(db, 'pc-1', _cierre(temas=3))
    # Lo que no se entiende no rompe nada.
    _evento(db, 'mac-2', _cierre(temas='muchos', conocidos='x',
                                 accion='rara'))
    _evento(db, 'mac-1', {'n': 99}, nombre='otra_cosa')

    r = db.resumen_primer_minuto(30)
    assert r['resumenes'] == 4
    assert r['aparatos'] == 3
    assert r['con_algo'] == 2
    assert r['temas'] == 543
    assert r['lineas']['conocidos'] == {'resumenes': 1, 'temas': 312}
    assert r['lineas']['rejillas'] == {'resumenes': 1, 'temas': 28}
    assert r['lineas']['del_programa'] == {'resumenes': 1, 'temas': 5}
    assert r['lineas']['repetidos'] == {'resumenes': 1, 'temas': 12}
    assert r['lineas']['sin_nombre'] == {'resumenes': 1, 'temas': 18}
    assert r['acciones'] == {
        'identificar': 1, 'repetidos': 1, 'cerrar': 1, 'otra': 1}


def test_repetidos_sin_saber_no_es_ninguno():
    # El cliente manda -1 cuando cerró el resumen con los repetidos aún por
    # buscar, o Render no contestó: eso no puede sumar como «0 repetidos».
    db = _db()
    _evento(db, 'mac-1', _cierre(repetidos=-1))
    _evento(db, 'mac-1', _cierre(repetidos=0))
    r = db.resumen_primer_minuto(30)
    assert r['repetidos_sin_saber'] == 1
    assert r['lineas']['repetidos'] == {'resumenes': 0, 'temas': 0}
    assert r['con_algo'] == 0


def test_sin_eventos_sale_a_cero():
    r = _db().resumen_primer_minuto(30)
    assert r['resumenes'] == 0 and r['aparatos'] == 0 and r['acciones'] == {}
    assert set(r['lineas']) == {
        'conocidos', 'rejillas', 'del_programa', 'repetidos', 'sin_nombre',
        'copias_exactas'}


def test_el_panel_lo_da():
    src = open('routes/admin_panel.py', encoding='utf-8').read()
    assert '"primer_minuto": _primer_minuto(),' in src
    assert '_get_db().resumen_primer_minuto(30)' in src


def test_las_lineas_y_las_acciones_son_las_del_cliente():
    # Espejo de `ElPrimerMinuto.paraElEvento` y `AccionDelResumen`: si el
    # cliente cambia un nombre, aquí dejaría de sumar sin un error.
    assert AnalysisDB.LINEAS_DEL_PRIMER_MINUTO == (
        'conocidos', 'rejillas', 'del_programa', 'repetidos', 'sin_nombre',
        'copias_exactas')
    assert AnalysisDB.ACCIONES_DEL_PRIMER_MINUTO == (
        'cerrar', 'identificar', 'repetidos', 'copias', 'mezclas', 'fallados')


def test_el_panel_lo_calcula_de_verdad():
    # Que el resumen salga del panel de verdad, no solo que el nombre esté
    # escrito (`test_el_panel_no_miente_con_servidor_anterior.py`).
    import main  # noqa: F401  (monta el singleton que usa el panel)
    import routes.admin_panel as panel
    r = panel._primer_minuto()
    assert r is not None, 'None = embudo.sh diría que el panel falló'
    assert {'resumenes', 'lineas', 'acciones', 'repetidos_sin_saber'} <= set(r)


def test_las_copias_exactas_cuentan_como_una_linea():
    # Ficheros que eran copias exactas de un tema que ya estaba y el import no
    # añadió (2026-11-11). Cuentan aparte, y su «Ver» cierra con `copias`.
    db = _db()
    _evento(db, 'mac-1', {'temas': 24, 'conocidos': 0, 'rejillas': 0,
                          'del_programa': 0, 'sin_nombre': 0, 'repetidos': 2,
                          'copias_exactas': 2, 'accion': 'copias'})
    r = db.resumen_primer_minuto(30)
    assert r['lineas']['copias_exactas'] == {'resumenes': 1, 'temas': 2}
    assert r['acciones'] == {'copias': 1}
    assert r['con_algo'] == 1
