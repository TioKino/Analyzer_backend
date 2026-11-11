"""La pasada del desfase de los MP3 (`resumen_desfase_mp3`, desde el
2026-11-11).

El escritorio pasa a la base de DJ Analyzer los cues y la rejilla que vinieron
de Rekordbox o Traktor (`conversion_del_desfase.dart` en el cliente) y manda
`desfase_mp3`. Lo que importa es si alguien la DESHACE: diría que la medida de
la pista de prueba no vale para su música.

    .venv/bin/python -m pytest -q test_el_desfase_de_los_mp3_se_mide.py
"""

import json
import tempfile

from database import AnalysisDB


def _db():
    return AnalysisDB(tempfile.mktemp(suffix='.db'))


def test_suma_lo_convertido_y_cuenta_quien_lo_deshace():
    db = _db()
    for dev, props in (
        ('mac-1', {'accion': 'pasada', 'temas': 1394, 'anclas': 1390, 'cues': 86}),
        ('mac-1', {'accion': 'pasada', 'temas': 6, 'anclas': 6, 'cues': 0}),
        ('pc-1', {'accion': 'pasada', 'temas': 20, 'anclas': 0, 'cues': 40}),
        ('pc-1', {'accion': 'deshacer', 'anclas': 0, 'cues': 40}),
        ('pc-1', {'accion': 'encender'}),
        ('pc-1', {'accion': 'pasada', 'temas': 'x'}),
    ):
        db.log_event(device_id=dev, event_name='desfase_mp3',
                     props=json.dumps(props), platform='macos-dmg')
    db.log_event(device_id='mac-2', event_name='cues_adoptados',
                 props=json.dumps({'accion': 'adoptar', 'cues': 3}))
    r = db.resumen_desfase_mp3(30)
    assert r == {'dias': 30, 'aparatos': 2, 'temas': 1420, 'anclas': 1396,
                 'cues': 126, 'deshechas': 1, 'aparatos_que_deshacen': 1,
                 'encendidas': 1}


def test_sin_eventos_sale_a_cero():
    r = _db().resumen_desfase_mp3(30)
    assert r['temas'] == 0 and r['deshechas'] == 0 and r['aparatos'] == 0


def test_el_panel_lo_da_de_verdad():
    import main  # noqa: F401  (monta el singleton que usa el panel)
    import routes.admin_panel as panel
    with open(panel.__file__, encoding='utf-8') as fh:
        assert '"desfase_mp3": _desfase_mp3(),' in fh.read()
    r = panel._desfase_mp3()
    assert r is not None, 'None = embudo.sh diría que el panel falló'
    assert {'temas', 'deshechas', 'aparatos_que_deshacen'} <= set(r)
