"""El interruptor de Escuchar (2026-09-29).

De los ~15 s que tarda un acierto, ~12 son la grabacion del primer clip, y
todo el audio pasaba por ffmpeg (loudnorm + WAV de ~1 MB) antes de ir a AudD.
Las dos palancas pueden bajar el tiempo y las dos pueden empeorar el acierto,
asi que se cambian desde Render sin release (`ESCUCHAR_PRIMER_CLIP_S`,
`ESCUCHAR_ENVIO`) y cada pulsacion apunta su variante para compararlas.

Lo que se ata:
- Sin tocar nada, todo es lo de siempre (12 s, ffmpeg).
- El envio directo manda a AudD el fichero TAL CUAL y solo cae a los
  preprocesados si AudD no saca huella. Solo en Escuchar.
- Cada pulsacion queda apuntada con su variante, y el panel las separa por
  el lado del servidor y por el del movil.

    pytest test_el_interruptor_de_escuchar.py -v
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

import main
from database import AnalysisDB

AUDIO = b'audio-del-movil' * 300


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def sin_interruptor(monkeypatch):
    monkeypatch.delenv('ESCUCHAR_PRIMER_CLIP_S', raising=False)
    monkeypatch.delenv('ESCUCHAR_ENVIO', raising=False)


@pytest.fixture
def audd(monkeypatch):
    """AudD y ffmpeg de mentira, apuntando que les llega."""
    class _Audd:
        respuestas = []
        enviados = []      # lo que recibio AudD, en bytes
        preprocesados = []  # estrategias de ffmpeg que corrieron

    def _pre(entrada, salida, estrategia):
        _Audd.preprocesados.append(estrategia)
        with open(salida, 'wb') as fh:
            fh.write(b'\0' * 4096)
        return True

    def _send(path, token, timeout=30):
        with open(path, 'rb') as fh:
            _Audd.enviados.append(fh.read())
        return _Audd.respuestas.pop(0) if _Audd.respuestas else (None, True)

    monkeypatch.setattr(main, '_preprocess_audio_for_recognition', _pre)
    monkeypatch.setattr(main, '_send_to_audd', _send)
    monkeypatch.setattr(main, 'check_rate_limit', lambda ip: None)
    _Audd.respuestas, _Audd.enviados, _Audd.preprocesados = [], [], []
    return _Audd


def _post(client, **campos):
    return client.post(
        '/recognize',
        files={'file': ('a.m4a', AUDIO, 'audio/mp4')},
        data={k: v for k, v in campos.items() if v is not None},
    )


def _marcador(device_id):
    conn = main.db._open_conn()
    try:
        return dict(conn.execute(
            "SELECT variante, ms FROM audd_call_log WHERE device_id = ? "
            "AND source = 'recognize_session' ORDER BY id DESC LIMIT 1",
            (device_id,)).fetchone())
    finally:
        conn.close()


def _td():
    return {'artist': f'A {uuid.uuid4().hex[:6]}',
            'title': f'T {uuid.uuid4().hex[:6]}'}


# ── El interruptor ──────────────────────────────────────────────────────

class TestAjustes:
    def test_sin_tocar_nada_es_lo_de_siempre(self):
        assert main._ajustes_de_escuchar() == {'primer_clip_s': 12,
                                               'envio': 'ffmpeg'}

    def test_se_lee_de_las_variables(self, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_PRIMER_CLIP_S', '8')
        monkeypatch.setenv('ESCUCHAR_ENVIO', ' Directo ')
        assert main._ajustes_de_escuchar() == {'primer_clip_s': 8,
                                               'envio': 'directo'}

    @pytest.mark.parametrize('valor,esperado', [
        ('3', 5), ('5', 5), ('30', 12), ('abc', 12), ('', 12), (' 7 ', 7)])
    def test_el_clip_se_acota(self, monkeypatch, valor, esperado):
        # Mas de 12 no es una palanca de latencia, y por debajo de 5 AudD no
        # tiene con que trabajar.
        monkeypatch.setenv('ESCUCHAR_PRIMER_CLIP_S', valor)
        assert main._ajustes_de_escuchar()['primer_clip_s'] == esperado

    def test_un_envio_desconocido_es_ffmpeg(self, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_ENVIO', 'raw')
        assert main._ajustes_de_escuchar()['envio'] == 'ffmpeg'

    def test_el_movil_lo_pregunta(self, client, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_PRIMER_CLIP_S', '8')
        r = client.get('/escuchar/ajustes')
        assert r.status_code == 200
        assert r.json() == {'primer_clip_s': 8, 'envio': 'ffmpeg'}

    def test_primer_clip_de_un_cliente_publicado_es_12(self):
        # Todas las versiones publicadas graban 12 s y no mandan el campo.
        assert main._primer_clip_de(None) == 12
        assert main._primer_clip_de('x') == 12
        assert main._primer_clip_de('0') == 12
        assert main._primer_clip_de('8') == 8


# ── /recognize ──────────────────────────────────────────────────────────

class TestEnvioDirecto:
    def test_sin_interruptor_todo_pasa_por_ffmpeg(self, client, audd):
        dev = f'm-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(None, True)]
        r = _post(client, device_id=dev, origen='escuchar', sesion_id='s1')
        assert r.json()['envio'] == 'ffmpeg'
        assert audd.preprocesados == ['normalize']
        assert audd.enviados[0] != AUDIO
        assert _marcador(dev)['variante'] == '12s+ffmpeg'

    def test_directo_manda_el_fichero_tal_cual(self, client, audd, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_ENVIO', 'directo')
        dev = f'm-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(_td(), True)]
        r = _post(client, device_id=dev, origen='escuchar', sesion_id='s2',
                  primer_clip_s='8')
        assert r.json()['status'] == 'found'
        assert r.json()['envio'] == 'directo'
        assert audd.enviados == [AUDIO], 'AudD recibe lo que grabo el movil'
        assert audd.preprocesados == [], 'sin ffmpeg'
        m = _marcador(dev)
        assert m['variante'] == '8s+directo'
        assert isinstance(m['ms'], int) and m['ms'] >= 0

    def test_directo_sin_match_no_reintenta(self, client, audd, monkeypatch):
        # AudD saco huella y no lo conoce: otro preprocesado es tirar cuota.
        monkeypatch.setenv('ESCUCHAR_ENVIO', 'directo')
        audd.respuestas = [(None, True)]
        r = _post(client, device_id='m-nm', origen='escuchar', sesion_id='s3')
        assert r.json()['reason'] == 'no_match'
        assert len(audd.enviados) == 1
        assert audd.preprocesados == []

    def test_directo_con_audio_malo_cae_a_los_preprocesados(
            self, client, audd, monkeypatch):
        monkeypatch.setenv('ESCUCHAR_ENVIO', 'directo')
        audd.respuestas = [(None, False), (None, False), (_td(), True)]
        r = _post(client, device_id='m-am', origen='escuchar', sesion_id='s4')
        assert r.json()['status'] == 'found'
        assert audd.enviados[0] == AUDIO
        # Tres llamadas como mucho, igual que antes: `raw_wav` sobra (es el
        # mismo audio que el directo).
        assert audd.preprocesados == ['normalize', 'aggressive']
        assert len(audd.enviados) == 3

    def test_la_respuesta_trae_el_interruptor(self, client, audd, monkeypatch):
        # Si el movil no pudo preguntar al arrancar (Render reiniciando: visto
        # en el iPhone del owner el 2026-09-29), se entera en la primera
        # pulsacion y la siguiente ya va con lo que toca.
        monkeypatch.setenv('ESCUCHAR_PRIMER_CLIP_S', '8')
        audd.respuestas = [(None, True), (_td(), True), (None, False)]
        for _ in range(3):
            r = _post(client, device_id='m-aj', origen='escuchar',
                      sesion_id=uuid.uuid4().hex[:8])
            assert r.json()['ajustes'] == {'primer_clip_s': 8,
                                           'envio': 'ffmpeg'}
        # El escritorio no lo necesita.
        monkeypatch.setattr(main, 'search_artwork_online', None)
        audd.respuestas = [(None, True)]
        r = _post(client, device_id='pc-aj', origen='portada')
        assert r.json()['ajustes'] is None

    def test_directo_no_toca_el_escritorio(self, client, audd, monkeypatch):
        # Portadas y Editar mandan ficheros enteros y no esperan a nadie.
        monkeypatch.setenv('ESCUCHAR_ENVIO', 'directo')
        monkeypatch.setattr(main, 'search_artwork_online', None)
        dev = f'pc-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(None, True)]
        r = _post(client, device_id=dev, origen='portada')
        assert r.json()['envio'] == 'ffmpeg'
        assert audd.preprocesados == ['normalize']
        assert _marcador(dev)['variante'] is None
        assert _marcador(dev)['ms'] is None


# ── El panel separa las variantes ───────────────────────────────────────

class TestPanel:
    def test_el_servidor_cuenta_por_variante(self, tmp_path):
        db = AnalysisDB(str(tmp_path / 'a.db'))

        def marca(sesion, reason, variante, ms, origen='escuchar'):
            db.log_audd_call(fingerprint='recognize_session',
                             success=reason == 'matched',
                             source='recognize_session', device_id='d',
                             reason=reason, origen=origen, sesion=sesion,
                             variante=variante, ms=ms)

        def llamada(variante):
            db.log_audd_call(fingerprint='recognize', success=False,
                             source='recognize', device_id='d',
                             origen='escuchar', variante=variante)

        # Una pulsacion corta: audio malo y luego acierto (cuenta el acierto).
        marca('a', 'audio_unusable', '8s+directo', 900)
        marca('a', 'matched', '8s+directo', 1100)
        marca('b', 'no_match', '8s+directo', 1000)
        # Antes del interruptor no habia variante: es la de siempre.
        marca('c', 'matched', None, None)
        marca('d', 'matched', '12s+ffmpeg', 2400)
        # El escritorio no es Escuchar.
        marca(None, 'no_match', None, None, origen='portada')
        for v in ('8s+directo', '8s+directo', '8s+directo', None):
            llamada(v)

        pv = db.resumen_escuchar(days=30)['por_variante']
        assert set(pv) == {'8s+directo', '12s+ffmpeg'}
        corto = pv['8s+directo']
        assert (corto['pulsaciones'], corto['matched'], corto['no_match'],
                corto['audio_unusable']) == (2, 1, 1, 0)
        assert corto['llamadas_audd'] == 3
        assert corto['ms_hasta_audd'] == {'n': 3, 'p50': 1000, 'p90': 1100}
        siempre = pv['12s+ffmpeg']
        assert (siempre['pulsaciones'], siempre['matched']) == (2, 2)
        assert siempre['llamadas_audd'] == 1
        assert siempre['ms_hasta_audd']['n'] == 1

    def test_el_movil_cuenta_por_variante(self):
        import routes.admin_panel as panel
        antes = panel._escuchar_segun_el_movil()['por_variante']
        dev = f'movil-{uuid.uuid4().hex[:8]}'
        for props in ({'outcome': 'found', 'ms': 9000, 'clip_s': 8,
                       'envio': 'directo'},
                      {'outcome': 'no_match', 'clip_s': 8, 'envio': 'directo'},
                      {'outcome': 'found', 'ms': 15000}):
            main.db.log_event(device_id=dev, event_name='listen_result',
                              props=json.dumps(props), platform='ios')
        pv = panel._escuchar_segun_el_movil()['por_variante']

        def n(v, k='pulsaciones'):
            return (pv.get(v) or {}).get(k, 0) - (antes.get(v) or {}).get(k, 0)
        assert n('8s+directo') == 2
        assert n('12s+ffmpeg') == 1, 'sin clip_s ni envio = lo de siempre'
        assert (pv['8s+directo']['ms_acierto']['n']
                - (antes.get('8s+directo') or {'ms_acierto': {'n': 0}})
                ['ms_acierto']['n']) == 1

    def test_no_es_este_se_cuenta_por_variante(self):
        # En el servidor una deteccion equivocada cuenta como `matched`: la
        # precision solo la da el movil (guardadas frente a «No es este»).
        import routes.admin_panel as panel
        antes = panel._escuchar_segun_el_movil()
        dev = f'movil-{uuid.uuid4().hex[:8]}'
        corta = json.dumps({'clip_s': 8, 'envio': 'ffmpeg'})
        for nombre, props in (('listen_saved', corta),
                              ('listen_wrong', corta),
                              ('listen_wrong', corta),
                              ('listen_saved', None)):  # un movil anterior
            main.db.log_event(device_id=dev, event_name=nombre, props=props,
                              platform='ios')
        r = panel._escuchar_segun_el_movil()
        assert r['equivocadas'] - antes['equivocadas'] == 2
        assert r['guardadas'] - antes['guardadas'] == 2

        def delta(v, k):
            return ((r['por_variante'].get(v) or {}).get(k, 0)
                    - (antes['por_variante'].get(v) or {}).get(k, 0))
        assert delta('8s+ffmpeg', 'equivocadas') == 2
        assert delta('8s+ffmpeg', 'guardadas') == 1
        assert delta('12s+ffmpeg', 'guardadas') == 1, 'sin variante = la de siempre'

    def test_variante_del_evento(self):
        from routes.admin_panel import _variante_del_evento
        assert _variante_del_evento({}) == '12s+ffmpeg'
        assert _variante_del_evento({'clip_s': 8}) == '8s+ffmpeg'
        assert _variante_del_evento({'clip_s': '7', 'envio': 'DIRECTO'}) \
            == '7s+directo'
        assert _variante_del_evento({'clip_s': 'x', 'envio': 'otro'}) \
            == '12s+ffmpeg'
