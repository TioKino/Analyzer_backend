"""ShazamKit delante, AudD detras (2026-09-30).

Con `ESCUCHAR_MOTOR=shazam` el movil reconoce con ShazamKit en el propio
aparato, en streaming y gratis, y solo si Shazam no encuentra el tema manda a
/recognize el MISMO audio que ya grabo. Lo que Shazam encuentra llega a
/recognize/shazam sin audio, y el servidor hace lo mismo que tras un acierto de
AudD: la ficha, la portada y el marcador de la pulsacion.

Lo que se ata:
- El interruptor: sin tocar nada el motor es AudD; `shazam` solo si se dice.
- /recognize/shazam responde con la MISMA forma que /recognize, deja el
  marcador de sesion (el cupo y el panel la cuentan) y NO apunta ninguna
  llamada a AudD (el gasto de AudD no sube por un acierto de Shazam).
- El cupo cuenta pulsaciones igual por los dos caminos.
- AudD detras lleva la variante con el prefijo, y el panel separa quien
  resolvio cada acierto, por el lado del servidor y por el del movil.

    pytest test_shazam_delante.py -v
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
    for v in ('ESCUCHAR_PRIMER_CLIP_S', 'ESCUCHAR_ENVIO', 'ESCUCHAR_MOTOR'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(main, 'check_rate_limit', lambda ip: None)
    # La deteccion nueva busca portada en internet: aqui no.
    monkeypatch.setattr(main, 'search_artwork_online', None)


@pytest.fixture
def audd(monkeypatch):
    class _Audd:
        respuestas = []
        llamadas = 0

    def _pre(entrada, salida, estrategia):
        with open(salida, 'wb') as fh:
            fh.write(b'\0' * 4096)
        return True

    def _send(path, token, timeout=30):
        _Audd.llamadas += 1
        return _Audd.respuestas.pop(0) if _Audd.respuestas else (None, True)

    monkeypatch.setattr(main, '_preprocess_audio_for_recognition', _pre)
    monkeypatch.setattr(main, '_send_to_audd', _send)
    _Audd.respuestas, _Audd.llamadas = [], 0
    return _Audd


def _acierto(**extra):
    base = {
        'artist': f'Artista {uuid.uuid4().hex[:6]}',
        'title': f'Tema {uuid.uuid4().hex[:6]}',
        'isrc': 'GBAYE0601498',
        'apple_music_url': 'https://music.apple.com/es/album/x/1?i=2',
        'artwork_url': 'https://is1-ssl.mzstatic.com/image/x/600x600bb.jpg',
        'web_url': 'https://www.shazam.com/track/123',
        'shazam_id': '123',
        'origen': 'escuchar',
        'sesion_id': f's{uuid.uuid4().hex[:8]}',
    }
    base.update(extra)
    return base


def _filas(device_id):
    conn = main.db._open_conn()
    try:
        return [dict(f) for f in conn.execute(
            "SELECT source, reason, variante, motor, success FROM "
            "audd_call_log WHERE device_id = ? ORDER BY id", (device_id,))]
    finally:
        conn.close()


class TestInterruptor:
    def test_sin_tocar_nada_el_motor_es_audd(self):
        assert main._ajustes_de_escuchar()['motor'] == 'audd'

    @pytest.mark.parametrize('valor,esperado', [
        ('shazam', 'shazam'), (' Shazam ', 'shazam'), ('audd', 'audd'),
        ('acrcloud', 'audd'), ('', 'audd')])
    def test_solo_shazam_si_se_dice(self, monkeypatch, valor, esperado):
        monkeypatch.setenv('ESCUCHAR_MOTOR', valor)
        assert main._ajustes_de_escuchar()['motor'] == esperado

    def test_el_movil_lo_pregunta(self, client, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_MOTOR', 'shazam')
        assert client.get('/escuchar/ajustes').json()['motor'] == 'shazam'

    def test_la_variante_lleva_el_prefijo(self):
        assert main._variante_de_escuchar(12, 'ffmpeg') == '12s+ffmpeg'
        assert main._variante_de_escuchar(12, 'ffmpeg', 'shazam') == \
            'shazam+12s+ffmpeg'


class TestAciertoDeShazam:
    def test_misma_forma_que_recognize_y_sin_gastar_audd(self, client, audd):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        a = _acierto(device_id=dev)
        r = client.post('/recognize/shazam', json=a)
        assert r.status_code == 200
        j = r.json()
        assert j['status'] == 'found'
        assert j['motor'] == 'shazam'
        assert (j['artist'], j['title'], j['isrc']) == (
            a['artist'], a['title'], a['isrc'])
        # Lo que el movil lee de /recognize esta todo.
        for k in ('album', 'spotify', 'deezer', 'apple_music', 'artwork_url',
                  'backend_analysis', 'busco_ficha', 'ms_servidor',
                  'ajustes', 'envio'):
            assert k in j, k
        assert j['busco_ficha'] is True
        # Las guias de ShazamKit piden el enlace a Apple Music.
        assert j['apple_music'] == {'url': a['apple_music_url']}
        assert j['artwork_url'] == a['artwork_url']
        assert audd.llamadas == 0
        filas = _filas(dev)
        assert filas == [{'source': 'recognize_session', 'reason': 'matched',
                          'variante': 'shazam+12s+ffmpeg', 'motor': 'shazam',
                          'success': 1}]

    def test_sin_artista_o_titulo_es_400(self, client):
        r = client.post('/recognize/shazam', json=_acierto(artist='  '))
        assert r.status_code == 400

    def test_una_url_que_no_es_https_no_se_devuelve(self, client):
        r = client.post('/recognize/shazam', json=_acierto(
            apple_music_url='javascript:alert(1)',
            artwork_url='http://inseguro/x.jpg'))
        j = r.json()
        assert j['apple_music'] is None
        assert j['artwork_url'] is None

    def test_trae_la_ficha_de_la_comunidad(self, client, monkeypatch):
        ficha = {'id': 'f1', 'fingerprint': None, 'bpm': 128.0,
                 'key': 'A minor', 'artwork_url': 'https://viejo/x.jpg'}
        monkeypatch.setattr(main, '_ficha_para_recognize',
                            lambda a, t, i: dict(ficha))
        j = client.post('/recognize/shazam', json=_acierto()).json()
        assert j['backend_analysis']['bpm'] == 128.0
        # La ficha lleva la MISMA portada que la respuesta (la de Shazam), no
        # la cadena guardada en la fila.
        assert j['backend_analysis']['artwork_url'] == j['artwork_url']

    def test_el_cupo_cuenta_las_pulsaciones_de_shazam(
            self, client, audd, monkeypatch):
        monkeypatch.setattr(main, 'RECOGNIZE_PRO_DAILY_CAP', 2)
        dev = f'm-{uuid.uuid4().hex[:8]}'
        for _ in range(2):
            j = client.post('/recognize/shazam',
                            json=_acierto(device_id=dev)).json()
            assert j['status'] == 'found'
        j = client.post('/recognize/shazam',
                        json=_acierto(device_id=dev)).json()
        assert j['status'] == 'cap_reached'
        # Y por AudD igual: el cupo es uno.
        r = client.post('/recognize', files={'file': ('a.m4a', AUDIO)},
                        data={'device_id': dev, 'origen': 'escuchar',
                              'sesion_id': 'otra'})
        assert r.json()['status'] == 'cap_reached'
        assert audd.llamadas == 0

    def test_una_pulsacion_ya_contada_no_se_queda_fuera(
            self, client, audd, monkeypatch):
        # AudD detras ya conto la pulsacion; Shazam la resuelve despues.
        monkeypatch.setattr(main, 'RECOGNIZE_PRO_DAILY_CAP', 1)
        dev = f'm-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(None, False), (None, False), (None, False)]
        client.post('/recognize', files={'file': ('a.m4a', AUDIO)},
                    data={'device_id': dev, 'origen': 'escuchar',
                          'sesion_id': 'misma', 'motor': 'shazam'})
        j = client.post('/recognize/shazam', json=_acierto(
            device_id=dev, sesion_id='misma')).json()
        assert j['status'] == 'found'


class TestAuddDetras:
    def test_la_variante_lleva_el_prefijo_y_el_motor_es_audd(
            self, client, audd):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(None, True)]
        client.post('/recognize', files={'file': ('a.wav', AUDIO)},
                    data={'device_id': dev, 'origen': 'escuchar',
                          'sesion_id': 's', 'motor': 'shazam'})
        marcador = [f for f in _filas(dev)
                    if f['source'] == 'recognize_session'][0]
        assert marcador['variante'] == 'shazam+12s+ffmpeg'
        assert marcador['motor'] == 'audd'
        llamada = [f for f in _filas(dev) if f['source'] == 'recognize'][0]
        assert llamada['variante'] == 'shazam+12s+ffmpeg'

    def test_sin_motor_es_lo_de_siempre(self, client, audd):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(None, True)]
        client.post('/recognize', files={'file': ('a.m4a', AUDIO)},
                    data={'device_id': dev, 'origen': 'escuchar',
                          'sesion_id': 's'})
        assert _filas(dev)[-1]['variante'] == '12s+ffmpeg'


class TestPanel:
    def test_el_servidor_separa_quien_resolvio(self, client, audd):
        antes = main.db.resumen_escuchar()['por_variante'].get(
            'shazam+12s+ffmpeg', {}).get('resueltas_por',
                                         {'shazam': 0, 'audd': 0})
        dev = f'm-{uuid.uuid4().hex[:8]}'
        # Una la resuelve Shazam; otra AudD detras.
        client.post('/recognize/shazam', json=_acierto(device_id=dev))
        audd.respuestas = [({'artist': 'A', 'title': 'T'}, True)]
        client.post('/recognize', files={'file': ('a.wav', AUDIO)},
                    data={'device_id': dev, 'origen': 'escuchar',
                          'sesion_id': f's{uuid.uuid4().hex[:6]}',
                          'motor': 'shazam'})
        despues = main.db.resumen_escuchar()['por_variante'][
            'shazam+12s+ffmpeg']['resueltas_por']
        assert despues['shazam'] - antes['shazam'] == 1
        assert despues['audd'] - antes['audd'] == 1
        # El gasto de AudD cuenta solo la llamada de verdad.
        llamadas = main.db.get_audd_stats_by_source(days=1)['recognize']
        assert llamadas['total'] >= 1

    def test_el_movil_separa_quien_resolvio_y_los_errores(self):
        antes = admin_panel._escuchar_segun_el_movil()
        v = 'shazam+12s+ffmpeg'
        rp0 = (antes['por_variante'].get(v, {}).get('resueltas_por')
               or {'shazam': 0, 'audd': 0})
        err0 = antes['shazam_errores'].get('202', 0)
        dev = f'movil-{uuid.uuid4().hex[:8]}'
        for props in (
                {'outcome': 'found', 'motor': 'shazam',
                 'resuelto_por': 'shazam'},
                {'outcome': 'found', 'motor': 'shazam',
                 'resuelto_por': 'audd'},
                {'outcome': 'no_match', 'motor': 'shazam',
                 'shazam_error': '202'}):
            main.db.log_event(device_id=dev, event_name='listen_result',
                              props=json.dumps(props), platform='ios')
        d = admin_panel._escuchar_segun_el_movil()
        rp = d['por_variante'][v]['resueltas_por']
        assert rp['shazam'] - rp0['shazam'] == 1
        assert rp['audd'] - rp0['audd'] == 1
        assert d['shazam_errores']['202'] - err0 == 1

    def test_variante_del_evento(self):
        assert admin_panel._variante_del_evento({'motor': 'shazam'}) == \
            'shazam+12s+ffmpeg'
        assert admin_panel._variante_del_evento({'motor': 'audd'}) == \
            '12s+ffmpeg'
