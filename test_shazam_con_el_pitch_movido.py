"""Shazam con el pitch movido (2026-10-01, `ESCUCHAR_SHAZAM_TEMPO`).

Un DJ que pincha con el pitch a +4 % deja el tema irreconocible para una
huella hecha a su velocidad. Con el interruptor encendido, si una pulsacion
con Shazam delante acaba sin acierto, el movil firma el audio a otras
velocidades y lo busca otra vez en Shazam, en el aparato y gratis. Es un
experimento: el panel tiene que poder contar que rescata.

Lo que se ata:
- Sin la variable, ni sale en los ajustes ni cambia la variante.
- Con ella, la variante lleva `+tempo` en las tres fuentes (llamadas del
  servidor, acierto de Shazam y eventos del movil).
- `movil_30d.rescatadas_por_tempo` cuenta los aciertos por velocidad.

    pytest test_shazam_con_el_pitch_movido.py -v
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

import main
from routes import admin_panel


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def sin_interruptor(monkeypatch):
    for v in ('ESCUCHAR_PRIMER_CLIP_S', 'ESCUCHAR_ENVIO', 'ESCUCHAR_MOTOR',
              'ESCUCHAR_AUDD_TRAS_S', 'ESCUCHAR_SHAZAM_TEMPO'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(main, 'check_rate_limit', lambda ip: None)
    monkeypatch.setattr(main, 'search_artwork_online', None)


def _variantes(device_id):
    conn = main.db._open_conn()
    try:
        return [f[0] for f in conn.execute(
            "SELECT variante FROM audd_call_log WHERE device_id = ? "
            "ORDER BY id", (device_id,))]
    finally:
        conn.close()


class TestInterruptor:
    def test_sin_la_variable_no_sale(self):
        assert 'shazam_tempo' not in main._ajustes_de_escuchar()

    @pytest.mark.parametrize('valor,sale', [
        ('1', True), ('si', True), ('true', True), (' ON ', True),
        ('0', False), ('no', False), ('', False)])
    def test_solo_un_si_claro(self, monkeypatch, valor, sale):
        monkeypatch.setenv('ESCUCHAR_SHAZAM_TEMPO', valor)
        assert ('shazam_tempo' in main._ajustes_de_escuchar()) is sale

    def test_el_movil_lo_pregunta(self, client, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_SHAZAM_TEMPO', '1')
        assert client.get('/escuchar/ajustes').json()['shazam_tempo'] is True


class TestVariante:
    def test_la_forma(self):
        v = main._variante_de_escuchar
        assert v(12, 'ffmpeg', 'shazam', None, True) == 'shazam+12s+ffmpeg+tempo'
        assert v(12, 'ffmpeg', 'shazam', 15, True) == \
            'shazam+12s+ffmpeg+audd15s+tempo'
        assert v(12, 'ffmpeg', 'audd', None, True) == '12s+ffmpeg'

    def test_un_acierto_de_shazam_la_apunta(self, client):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        client.post('/recognize/shazam', json={
            'artist': 'A', 'title': f'T {uuid.uuid4().hex[:6]}',
            'origen': 'escuchar', 'sesion_id': 's', 'device_id': dev,
            'shazam_tempo': '1'})
        assert _variantes(dev) == ['shazam+12s+ffmpeg+tempo']

    def test_y_audd_detras_tambien(self, client, monkeypatch):
        def _pre(entrada, salida, estrategia):
            with open(salida, 'wb') as fh:
                fh.write(b'\0' * 4096)
            return True
        monkeypatch.setattr(main, '_preprocess_audio_for_recognition', _pre)
        monkeypatch.setattr(main, '_send_to_audd',
                            lambda path, token, timeout=30: (None, True))
        dev = f'm-{uuid.uuid4().hex[:8]}'
        client.post('/recognize', files={'file': ('a.wav', b'x' * 4000)},
                    data={'device_id': dev, 'origen': 'escuchar',
                          'sesion_id': 's', 'motor': 'shazam',
                          'shazam_tempo': '1'})
        assert set(_variantes(dev)) == {'shazam+12s+ffmpeg+tempo'}

    def test_el_panel_del_movil_tambien(self):
        e = admin_panel._variante_del_evento
        assert e({'motor': 'shazam', 'shazam_tempo': True}) == \
            'shazam+12s+ffmpeg+tempo'
        assert e({'motor': 'shazam', 'audd_tras_s': 15,
                  'shazam_tempo': True}) == 'shazam+12s+ffmpeg+audd15s+tempo'
        assert e({'motor': 'audd', 'shazam_tempo': True}) == '12s+ffmpeg'


class TestPanel:
    def test_cuenta_los_rescates_por_velocidad(self):
        antes = dict(admin_panel._escuchar_segun_el_movil()
                     ['rescatadas_por_tempo'])
        dev = f'movil-{uuid.uuid4().hex[:8]}'
        for props in (
                {'outcome': 'found', 'motor': 'shazam',
                 'resuelto_por': 'shazam', 'tempo_pct': 4},
                {'outcome': 'found', 'motor': 'shazam',
                 'resuelto_por': 'shazam', 'tempo_pct': -2},
                {'outcome': 'found', 'motor': 'shazam',
                 'resuelto_por': 'shazam'},          # a su velocidad
                {'outcome': 'no_match', 'motor': 'shazam',
                 'tempo_pct': 6},                     # no cuenta: sin acierto
                {'outcome': 'found', 'motor': 'shazam',
                 'resuelto_por': 'shazam', 'tempo_pct': 'x'}):
            main.db.log_event(device_id=dev, event_name='listen_result',
                              props=json.dumps(props), platform='ios')
        d = admin_panel._escuchar_segun_el_movil()['rescatadas_por_tempo']
        assert d.get('+4', 0) - antes.get('+4', 0) == 1
        assert d.get('-2', 0) - antes.get('-2', 0) == 1
        assert d.get('+6', 0) - antes.get('+6', 0) == 0
