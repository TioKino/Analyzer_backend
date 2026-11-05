"""Las MEZCLAS del escritorio se miden (`resumen_mezclas`, desde el 2026-11-03).

El import de escritorio aparta lo que es una sesión entera en un fichero (una
mezcla) en vez de analizarla como un tema, y en el Mac se saca su tracklist del
fichero con Shazam (`mezclas.dart` en el cliente). Sin esto no se sabría si
alguien lo usa ni, sobre todo, cuánto se equivoca la regla: «Analizar como
tema» frente a lo apartado, por el motivo por el que se apartó.

    .venv/bin/python -m pytest -q test_las_mezclas_se_miden.py
"""

import json
import tempfile

from database import AnalysisDB


def _db():
    return AnalysisDB(tempfile.mktemp(suffix='.db'))


def _evento(db, dev, nombre, props):
    db.log_event(device_id=dev, event_name=nombre, props=json.dumps(props))


class TestElResumen:
    def test_suma_lo_que_cuentan_los_ordenadores(self):
        db = _db()
        _evento(db, 'mac-1', 'mezclas_apartadas', {
            'nuevas': 3, 'ya_estaban': 1,
            'por_duracion': 2, 'por_nombre': 0, 'por_shazam': 1})
        _evento(db, 'pc-1', 'mezclas_apartadas', {
            'nuevas': 1, 'ya_estaban': 0,
            'por_duracion': 0, 'por_nombre': 1, 'por_shazam': 0})
        _evento(db, 'mac-1', 'mezcla_como_tema',
                {'motivo': 'nombre', 'minutos': 14, 'identificada': False})
        _evento(db, 'mac-1', 'mezcla_a_mano', {'desde': 'biblioteca', 'minutos': 61})
        _evento(db, 'mac-1', 'mezcla_a_mano', {'desde': 'fichero', 'minutos': 120})
        # Una mezcla de una hora hecha de una vez: 15 temas, 300 s.
        _evento(db, 'mac-1', 'mezcla_identificada', {
            'corte': 'hecha', 'motivo': 'duracion', 'minutos': 60,
            'trozos': 240, 'con_tema': 180, 'temas': 15, 'segundos': 300,
            'seguia': False})
        # Otra de dos horas: se paró y luego siguió donde se quedó.
        _evento(db, 'mac-2', 'mezcla_identificada', {
            'corte': 'parada', 'motivo': 'aMano', 'minutos': 120,
            'trozos': 80, 'con_tema': 50, 'temas': 6, 'segundos': 100,
            'seguia': False})
        _evento(db, 'mac-2', 'mezcla_identificada', {
            'corte': 'hecha', 'motivo': 'aMano', 'minutos': 120,
            'trozos': 400, 'con_tema': 290, 'temas': 30, 'segundos': 500,
            'seguia': True})
        _evento(db, 'mac-1', 'otra_cosa', {'nuevas': 99})

        r = db.resumen_mezclas(30)
        assert r['dias'] == 30
        assert r['aparatos'] == 3
        assert r['apartadas'] == {
            'nuevas': 4, 'ya_estaban': 1, 'aparatos': 2,
            'por_motivo': {'duracion': 2, 'nombre': 1, 'shazam': 1}}
        assert r['como_tema'] == {'total': 1, 'por_motivo': {'nombre': 1}}
        assert r['a_mano'] == {'total': 2,
                               'desde': {'biblioteca': 1, 'fichero': 1}}
        i = r['identificadas']
        assert i['corridas'] == 3 and i['aparatos'] == 2
        assert i['por_corte'] == {'hecha': 2, 'parada': 1}
        assert i['trozos'] == 720 and i['con_tema'] == 520
        # Temas por hora: de las dos que terminaron (15 + 30 en 3 h).
        assert i['temas_por_hora'] == 15.0
        # Cuánto se tarda: solo la que se hizo de una vez (300 s por hora).
        # La que siguió tardó solo lo que le faltaba, y contarla diría que
        # una hora se identifica en 250 s.
        assert i['segundos_por_hora'] == 300

    def test_lo_que_el_dj_cambia_y_los_tramos_por_shazam(self):
        # 2026-11-05: las líneas del tracklist se editan, y un tramo elegido
        # se pasa por Shazam. Lo que importa: cuánto se usa, y cuántos tramos
        # reconoce Shazam (si casi ninguno, la idea no sirve).
        db = _db()
        for que in ('mover', 'mover', 'anadir', 'quitar', 'anadir_con_shazam'):
            _evento(db, 'mac-1', 'mezcla_editada', {'que': que})
        _evento(db, 'mac-2', 'mezcla_editada', {'que': 'deshacer_todo'})
        _evento(db, 'mac-1', 'mezcla_tramo',
                {'resultado': 'reconocido', 'segundos': 22, 'trozos': 2, 'votos': 2,
                 'seguro': True})
        # Uno reconocido pero sin confirmar (un trozo suelto): no es seguro.
        _evento(db, 'mac-1', 'mezcla_tramo',
                {'resultado': 'reconocido', 'segundos': 36, 'trozos': 7, 'votos': 1,
                 'seguro': False})
        _evento(db, 'mac-1', 'mezcla_tramo',
                {'resultado': 'no', 'segundos': 30, 'trozos': 4, 'votos': 0})
        _evento(db, 'mac-2', 'mezcla_tramo',
                {'resultado': 'sin_red', 'segundos': 12, 'trozos': 0, 'votos': 0})
        r = db.resumen_mezclas(30)
        assert r['editadas'] == {
            'total': 6, 'aparatos': 2,
            'por_que': {'mover': 2, 'anadir': 1, 'quitar': 1,
                        'anadir_con_shazam': 1, 'deshacer_todo': 1}}
        assert r['tramos'] == {
            'total': 4, 'aparatos': 2, 'reconocidos': 2, 'seguros': 1,
            'por_resultado': {'reconocido': 2, 'no': 1, 'sin_red': 1}}
        assert r['aparatos'] == 2

    def test_sin_eventos_dice_cero_y_no_lo_se(self):
        r = _db().resumen_mezclas(30)
        assert r['aparatos'] == 0
        assert r['apartadas']['nuevas'] == 0
        assert r['identificadas']['corridas'] == 0
        assert r['identificadas']['temas_por_hora'] is None, (
            'sin mezclas terminadas no hay temas por hora: «no lo sé», no 0')
        assert r['identificadas']['segundos_por_hora'] is None

    def test_lo_raro_no_lo_tumba(self):
        db = _db()
        db.log_event(device_id='x', event_name='mezcla_identificada', props='no es json')
        db.log_event(device_id='x', event_name='mezcla_identificada', props='[1, 2]')
        _evento(db, 'x', 'mezclas_apartadas', {'nuevas': 'tres', 'por_duracion': -4})
        r = db.resumen_mezclas(30)
        assert r['identificadas']['corridas'] == 0
        assert r['apartadas']['nuevas'] == 0
        assert r['apartadas']['por_motivo'] == {'duracion': 0}

    def test_solo_los_ultimos_dias(self):
        db = _db()
        _evento(db, 'mac-1', 'mezcla_como_tema', {'motivo': 'duracion'})
        conn = db._open_conn()
        try:
            conn.execute(
                "UPDATE events SET timestamp = datetime('now', '-40 days')")
            conn.commit()
        finally:
            conn.close()
        assert db.resumen_mezclas(30)['como_tema']['total'] == 0
        assert db.resumen_mezclas(60)['como_tema']['total'] == 1


class TestElPanel:
    def test_el_panel_lo_entrega(self):
        from routes import admin_panel

        assert isinstance(admin_panel._resumen_mezclas(), dict)

    def test_va_en_la_telemetria(self):
        src = open('routes/admin_panel.py', encoding='utf-8').read()
        assert '"mezclas": _resumen_mezclas(),' in src
