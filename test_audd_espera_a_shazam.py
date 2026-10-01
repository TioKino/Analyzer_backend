"""Con Shazam delante, cuanto se espera antes de pagar AudD (2026-10-01).

En la primera prueba con ShazamKit, Panic salio a AudD a los 12 s (al cerrar
el primer clip) y Shazam lo caso a los 13,5: AudD se pago para nada, y encima
dio otro tema. `ESCUCHAR_AUDD_TRAS_S` deja decidir desde Render, sin release,
cuantos segundos se le dan a Shazam antes del primer envio a AudD. AudD sigue
recibiendo los ultimos `primer_clip_s` segundos; lo que cambia es cuando.

Lo que se ata:
- Sin la variable (o por debajo del clip) todo es lo de siempre: ni la clave
  sale en los ajustes ni la variante cambia.
- Con ella, el movil la lee de /escuchar/ajustes y la variante de la
  pulsacion lleva el sufijo (`shazam+12s+ffmpeg+audd15s`) en el servidor,
  en /recognize/shazam y en el panel, para comparar las dos.

    pytest test_audd_espera_a_shazam.py -v
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

import main
from routes import admin_panel

AUDIO = b'audio-del-movil' * 300


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def sin_interruptor(monkeypatch):
    for v in ('ESCUCHAR_PRIMER_CLIP_S', 'ESCUCHAR_ENVIO', 'ESCUCHAR_MOTOR',
              'ESCUCHAR_AUDD_TRAS_S'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(main, 'check_rate_limit', lambda ip: None)
    monkeypatch.setattr(main, 'search_artwork_online', None)


@pytest.fixture
def audd(monkeypatch):
    def _pre(entrada, salida, estrategia):
        with open(salida, 'wb') as fh:
            fh.write(b'\0' * 4096)
        return True

    monkeypatch.setattr(main, '_preprocess_audio_for_recognition', _pre)
    monkeypatch.setattr(main, '_send_to_audd',
                        lambda path, token, timeout=30: (None, True))


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
        assert 'audd_tras_s' not in main._ajustes_de_escuchar()

    @pytest.mark.parametrize('valor,esperado', [
        ('15', 15), (' 18 ', 18), ('40', 20),   # tope: lo que Shazam escucha
        ('12', None), ('8', None),              # no retrasa nada
        ('x', None), ('', None)])
    def test_se_acota(self, monkeypatch, valor, esperado):
        monkeypatch.setenv('ESCUCHAR_AUDD_TRAS_S', valor)
        assert main._ajustes_de_escuchar().get('audd_tras_s') == esperado

    def test_por_encima_del_clip_que_toque(self, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_PRIMER_CLIP_S', '8')
        monkeypatch.setenv('ESCUCHAR_AUDD_TRAS_S', '10')
        assert main._ajustes_de_escuchar()['audd_tras_s'] == 10

    def test_el_movil_lo_pregunta(self, client, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_MOTOR', 'shazam')
        monkeypatch.setenv('ESCUCHAR_AUDD_TRAS_S', '15')
        j = client.get('/escuchar/ajustes').json()
        assert (j['motor'], j['audd_tras_s']) == ('shazam', 15)


class TestVariante:
    def test_la_forma(self):
        v = main._variante_de_escuchar
        assert v(12, 'ffmpeg', 'shazam', 15) == 'shazam+12s+ffmpeg+audd15s'
        assert v(12, 'ffmpeg', 'shazam', None) == 'shazam+12s+ffmpeg'
        assert v(12, 'ffmpeg', 'shazam', 12) == 'shazam+12s+ffmpeg'
        # Sin Shazam no hay a quien esperar.
        assert v(12, 'ffmpeg', 'audd', 15) == '12s+ffmpeg'

    def test_audd_detras_la_apunta(self, client, audd):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        client.post('/recognize', files={'file': ('a.wav', AUDIO)},
                    data={'device_id': dev, 'origen': 'escuchar',
                          'sesion_id': 's', 'motor': 'shazam',
                          'audd_tras_s': '15'})
        assert set(_variantes(dev)) == {'shazam+12s+ffmpeg+audd15s'}

    def test_un_acierto_de_shazam_tambien(self, client):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        client.post('/recognize/shazam', json={
            'artist': 'A', 'title': f'T {uuid.uuid4().hex[:6]}',
            'origen': 'escuchar', 'sesion_id': 's', 'device_id': dev,
            'audd_tras_s': '15'})
        assert _variantes(dev) == ['shazam+12s+ffmpeg+audd15s']

    def test_basura_en_la_peticion_es_lo_de_siempre(self, client, audd):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        client.post('/recognize', files={'file': ('a.wav', AUDIO)},
                    data={'device_id': dev, 'origen': 'escuchar',
                          'sesion_id': 's', 'motor': 'shazam',
                          'audd_tras_s': 'mucho'})
        assert set(_variantes(dev)) == {'shazam+12s+ffmpeg'}

    def test_el_panel_del_movil_la_separa(self):
        e = admin_panel._variante_del_evento
        assert e({'motor': 'shazam', 'audd_tras_s': 15}) == \
            'shazam+12s+ffmpeg+audd15s'
        assert e({'motor': 'shazam', 'audd_tras_s': 12}) == \
            'shazam+12s+ffmpeg'
        assert e({'motor': 'audd', 'audd_tras_s': 15}) == '12s+ffmpeg'
        assert e({'motor': 'shazam', 'audd_tras_s': 'x'}) == \
            'shazam+12s+ffmpeg'

    def test_y_resueltas_por_sale_tambien_con_el_sufijo(self, client):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        client.post('/recognize/shazam', json={
            'artist': 'A', 'title': f'T {uuid.uuid4().hex[:6]}',
            'origen': 'escuchar', 'sesion_id': f's{uuid.uuid4().hex[:6]}',
            'device_id': dev, 'audd_tras_s': '16'})
        d = main.db.resumen_escuchar()['por_variante'][
            'shazam+12s+ffmpeg+audd16s']
        assert d['resueltas_por']['shazam'] >= 1
        movil = {'outcome': 'found', 'motor': 'shazam',
                 'resuelto_por': 'shazam', 'audd_tras_s': 16}
        main.db.log_event(device_id=dev, event_name='listen_result',
                          props=json.dumps(movil), platform='ios')
        m = admin_panel._escuchar_segun_el_movil()['por_variante']
        assert m['shazam+12s+ffmpeg+audd16s']['resueltas_por']['shazam'] >= 1
