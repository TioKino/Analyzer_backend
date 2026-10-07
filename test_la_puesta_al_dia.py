"""
La PUESTA AL DÍA de lo ya analizado (`puesta_al_dia.py`, 2026-10-07).

Lo que se fija aquí:
- La regla de qué se rehace (closing, trozos) y que vive en el servidor.
- El interruptor: sin `PUESTA_AL_DIA` no se rehace nada.
- `/puesta-al-dia/analizar` recalcula y NADA más: ni guarda, ni llama a
  AudD, ni saca portada; y en Render, con tope diario por aparato.
- El panel lee lo que el cliente cuenta.
"""

import hashlib
import io
import json
import tempfile
import uuid
from datetime import date

import pytest

import puesta_al_dia as pad


CORTE = '2026-10-08'


def _fila(**k):
    base = {'bpm': 128.0, 'analyzed_at': '2026-05-01T10:00:00',
            'engine_source': 'render', 'duration': 200.0,
            'track_type': 'peak_time', 'analysis_status': None}
    base.update(k)
    return base


class TestLaRegla:
    def test_closing_viejo_se_rehace(self):
        assert pad.motivos(_fila(track_type='closing'), CORTE) == ['closing']

    def test_largo_de_render_se_rehace(self):
        assert pad.motivos(_fila(duration=400.0), CORTE) == ['trozos']

    def test_largo_sin_motor_sellado_cuenta_como_render(self):
        assert pad.motivos(_fila(duration=400.0, engine_source=None),
                           CORTE) == ['trozos']

    def test_largo_del_motor_local_no(self):
        """El motor local analiza los largos por el camino corto: su energía
        nunca salió alta."""
        assert pad.motivos(_fila(duration=400.0, engine_source='local_engine'),
                           CORTE) == []

    def test_closing_largo_de_render_da_los_dos(self):
        assert pad.motivos(_fila(track_type='closing', duration=400.0),
                           CORTE) == ['closing', 'trozos']

    def test_lo_analizado_desde_el_corte_ya_es_del_dsp_nuevo(self):
        assert pad.motivos(_fila(track_type='closing',
                                 analyzed_at='2026-10-08T00:00:01'),
                           CORTE) == []

    def test_un_analisis_fallido_no_se_rehace(self):
        assert pad.motivos(_fila(track_type='closing', bpm=0), CORTE) == []
        assert pad.motivos(_fila(track_type='closing',
                                 analysis_status='failed'), CORTE) == []

    def test_el_umbral_es_el_de_render(self):
        import main
        assert pad.UMBRAL_TROZOS == main.CHUNK_ANALYSIS_THRESHOLD


class TestElInterruptor:
    def test_apagada_sin_variable(self, monkeypatch):
        monkeypatch.delenv('PUESTA_AL_DIA', raising=False)
        assert pad.corte() is None
        monkeypatch.setenv('PUESTA_AL_DIA', 'si')
        assert pad.corte() is None, 'sin fecha no hay corte: apagada'

    def test_encendida_con_fecha(self, monkeypatch):
        monkeypatch.setenv('PUESTA_AL_DIA', CORTE)
        assert pad.corte() == CORTE

    def test_tope(self, monkeypatch):
        monkeypatch.delenv('PUESTA_AL_DIA_TOPE_RENDER', raising=False)
        assert pad.tope_render() == pad.TOPE_RENDER_POR_DEFECTO
        monkeypatch.setenv('PUESTA_AL_DIA_TOPE_RENDER', '9999')
        assert pad.tope_render() == pad.TOPE_RENDER_MAXIMO
        monkeypatch.setenv('PUESTA_AL_DIA_TOPE_RENDER', 'x')
        assert pad.tope_render() == pad.TOPE_RENDER_POR_DEFECTO

    def test_el_tope_se_reinicia_cada_dia(self):
        t = pad.TopeDiario()
        assert t.apuntar('a', 2, date(2026, 10, 8))
        assert t.apuntar('a', 2, date(2026, 10, 8))
        assert not t.apuntar('a', 2, date(2026, 10, 8))
        assert t.apuntar('b', 2, date(2026, 10, 8)), 'el tope es por aparato'
        assert t.apuntar('a', 2, date(2026, 10, 9))


def _wav(freq=440.0, secs=8.0, sr=22050):
    import numpy as np
    import soundfile as sf

    t = np.arange(int(sr * secs)) / sr
    y = (0.4 * np.sin(2 * np.pi * freq * t)).astype('float32')
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    return buf.getvalue()


@pytest.fixture()
def app_mod():
    import main

    main.GENRE_DETECTOR_ENABLED = False
    main.ARTWORK_ENABLED = False
    return main


@pytest.fixture()
def client(app_mod):
    from fastapi.testclient import TestClient

    return TestClient(app_mod.app)


def _guardar(db, fp, **k):
    datos = {
        'id': fp, 'filename': f'{fp[:6]}.mp3', 'artist': 'A', 'title': 'T',
        'duration': 400.0, 'bpm': 128.0, 'key': 'Am', 'camelot': '8A',
        'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
        'fingerprint': fp, 'engine_source': 'render',
        'analyzed_at': '2026-05-01T10:00:00',
    }
    datos.update(k)
    db.save_track(datos)


class TestCandidatos:
    def test_apagada_no_da_ninguno(self, client, app_mod, monkeypatch):
        monkeypatch.delenv('PUESTA_AL_DIA', raising=False)
        fp = uuid.uuid4().hex
        _guardar(app_mod.db, fp, track_type='closing')
        r = client.post('/puesta-al-dia/candidatos', json={'fingerprints': [fp]})
        assert r.status_code == 200
        d = r.json()
        assert d['activa'] is False and d['candidatos'] == {}

    def test_encendida_dice_cuales_y_por_que(self, client, app_mod, monkeypatch):
        monkeypatch.setenv('PUESTA_AL_DIA', CORTE)
        closing, largo, local, nuevo, nadie = (uuid.uuid4().hex for _ in range(5))
        _guardar(app_mod.db, closing, track_type='closing', duration=200.0)
        _guardar(app_mod.db, largo)
        _guardar(app_mod.db, local, engine_source='local_engine')
        _guardar(app_mod.db, nuevo, track_type='closing',
                 analyzed_at='2026-10-09T08:00:00')
        r = client.post('/puesta-al-dia/candidatos', json={
            'fingerprints': [closing, largo, local, nuevo, nadie]})
        d = r.json()
        assert d['activa'] is True and d['corte'] == CORTE
        assert d['candidatos'] == {closing: ['closing'], largo: ['trozos']}
        assert d['tope_diario_render'] == pad.tope_render()

    def test_mas_de_500_no(self, client, monkeypatch):
        monkeypatch.setenv('PUESTA_AL_DIA', CORTE)
        r = client.post('/puesta-al-dia/candidatos',
                        json={'fingerprints': ['x'] * 501})
        assert r.status_code == 400


class TestAnalizar:
    def test_apagada_en_render_no_analiza(self, client, monkeypatch):
        monkeypatch.delenv('PUESTA_AL_DIA', raising=False)
        r = client.post('/puesta-al-dia/analizar',
                        files={'file': ('a.wav', _wav(), 'audio/wav')})
        assert r.status_code == 409

    def test_recalcula_sin_guardar_ni_audd(self, client, app_mod, monkeypatch):
        monkeypatch.setenv('PUESTA_AL_DIA', CORTE)
        monkeypatch.setattr(app_mod, '_tope_de_la_puesta_al_dia',
                            pad.TopeDiario())
        monkeypatch.setattr(app_mod, 'AUDD_AUTO_ENABLED', True)
        monkeypatch.setattr(app_mod, 'AUDD_API_TOKEN', 'token')
        import audd_helper

        def _no(*a, **k):
            raise AssertionError('la puesta al día llamó a AudD')
        monkeypatch.setattr(audd_helper, 'enrich_with_audd_if_needed', _no)

        audio = _wav(freq=523.0)
        fp = hashlib.md5(audio).hexdigest()
        r = client.post('/puesta-al-dia/analizar',
                        headers={'X-Device-Id': 'mac-1'},
                        files={'file': ('basura 01.wav', audio, 'audio/wav')})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d['bpm'] >= 0 and 'energy_dj' in d and 'track_type' in d
        assert app_mod.db.get_track_by_fingerprint(fp) is None, (
            'la puesta al día guardó la fila: lo de Render se quedaría con el '
            'nombre de las etiquetas en vez del de AudD')

    def test_tope_diario_en_render(self, client, app_mod, monkeypatch):
        monkeypatch.setenv('PUESTA_AL_DIA', CORTE)
        monkeypatch.setenv('PUESTA_AL_DIA_TOPE_RENDER', '1')
        monkeypatch.setattr(app_mod, '_tope_de_la_puesta_al_dia',
                            pad.TopeDiario())
        cab = {'X-Device-Id': 'mas-1'}
        r1 = client.post('/puesta-al-dia/analizar', headers=cab,
                         files={'file': ('a.wav', _wav(330.0), 'audio/wav')})
        r2 = client.post('/puesta-al-dia/analizar', headers=cab,
                         files={'file': ('b.wav', _wav(331.0), 'audio/wav')})
        assert r1.status_code == 200, r1.text
        assert r2.status_code == 429


class TestElPanel:
    def test_suma_lo_que_cuentan_los_ordenadores(self):
        from database import AnalysisDB

        db = AnalysisDB(tempfile.mktemp(suffix='.db'))
        for dev, props in [
            ('pc-1', {'motor': 'local', 'hechos': 20, 'errores': 1,
                      'ilegibles': 2,
                      'cambios': {'tipo': 12, 'energia': 3},
                      'energia_baja': 3, 'closing_sigue': 4,
                      'closing_a': {'warmup': 6, 'peak_time': 2}}),
            ('mas-1', {'motor': 'render', 'hechos': 5,
                       'cambios': {'tipo': 4}, 'closing_sigue': 1,
                       'closing_a': {'warmup': 4}}),
        ]:
            db.log_event(device_id=dev, event_name='puesta_al_dia',
                         props=json.dumps(props))
        db.log_event(device_id='pc-1', event_name='otra_cosa', props='{}')

        r = db.resumen_puesta_al_dia(30)
        assert r['aparatos'] == 2 and r['tandas'] == 2
        assert r['hechos'] == 25 and r['errores'] == 1 and r['ilegibles'] == 2
        assert r['por_motor']['local']['hechos'] == 20
        assert r['por_motor']['render']['hechos'] == 5
        assert r['cambios'] == {'tipo': 16, 'energia': 3}
        assert r['energia'] == {'baja': 3, 'sube': 0}
        assert r['closing'] == {'siguen': 5,
                                'pasan_a': {'warmup': 10, 'peak_time': 2}}

    def test_el_panel_lo_entrega(self):
        from routes import admin_panel

        assert isinstance(admin_panel._puesta_al_dia(), dict)
