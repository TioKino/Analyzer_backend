"""Escuchar, medido y arreglado (auditoría 2026-09-29, fases 0 y 1).

Lo que ataba cada cosa, en el orden de `PENDING.md` → «ESCUCHAR»:

- `/recognize` sirve al boton Escuchar del movil Y al backfill de portadas y a
  «Identificar» del diálogo Editar del escritorio. Todos escribian el mismo marcador de
  sesion, asi que la tasa de `no_match` «de Escuchar» llevaba dentro ficheros
  de biblioteca. Ahora cada peticion dice su `origen`.
- El movil manda hasta cuatro peticiones por pulsacion; el cupo las contaba
  todas. Ahora cuenta pulsaciones (`sesion`).
- La ficha: `/recognize` podia devolver su propia deteccion (bpm 0 y genero,
  energia y tipo de relleno) y no pasaba por `_lo_mejor_para`. El movil la
  tiraba y volvia a preguntar a `/search-analyzed`.
- La portada: se buscaba de forma sincrona dentro del `async def` (congelando
  el worker) y el movil la tiraba.

    pytest test_escuchar_se_mide_de_verdad.py -v
"""

import inspect
import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

import main
from database import AnalysisDB


SPOTIFY_IMG = 'https://i.scdn.co/image/portada-del-audio-exacto'


def _track_data(isrc=None, artist=None, title=None):
    return {
        'artist': artist or f'Artista {uuid.uuid4().hex[:6]}',
        'title': title or f'Tema {uuid.uuid4().hex[:6]}',
        'album': 'Album', 'label': 'Label', 'release_date': '2024-01-01',
        'isrc': isrc,
        'spotify': {'album': {'images': [{'url': SPOTIFY_IMG}]}},
        'deezer': None, 'apple_music': None,
    }


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def audd(monkeypatch):
    """AudD y ffmpeg de mentira. `audd.respuestas` es la cola de lo que va a
    «contestar» AudD: (track_data, processed_ok)."""
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
    monkeypatch.setattr(main, 'check_rate_limit', lambda ip: None)
    _Audd.respuestas = []
    return _Audd


def _guardar_analizado(**campos):
    """Un tema analizado de verdad en la BD colectiva."""
    fp = uuid.uuid4().hex
    fila = {'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3',
            'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
            'duration': 400, 'key': 'Am', 'camelot': '8A'}
    fila.update(campos)
    main.db.save_track(fila)
    return fp


def _post(client, **campos):
    return client.post(
        '/recognize',
        files={'file': ('a.m4a', b'x' * 4096, 'audio/mp4')},
        data={k: v for k, v in campos.items() if v is not None},
    )


def _marcadores(device_id):
    conn = main.db._open_conn()
    try:
        return [dict(f) for f in conn.execute(
            "SELECT source, origen, sesion, reason FROM audd_call_log "
            "WHERE device_id = ? ORDER BY id", (device_id,))]
    finally:
        conn.close()


# ── Quien llama ─────────────────────────────────────────────────────────

class TestOrigen:
    def test_lo_que_dice_el_cliente_manda(self):
        for o in ('escuchar', 'portada', 'editar'):
            assert main._origen_de_recognize(o, 'dev') == o
        assert main._origen_de_recognize(' ESCUCHAR ', None) == 'escuchar'

    def test_un_cliente_viejo_se_clasifica_por_su_aparato(self, monkeypatch):
        tipos = {'movil': 'ios', 'android': 'android', 'pc': 'windows',
                 'mac': 'macos'}
        monkeypatch.setattr(main, 'tipo_de_aparato', lambda d: tipos.get(d))
        assert main._origen_de_recognize(None, 'movil') == 'escuchar'
        assert main._origen_de_recognize(None, 'android') == 'escuchar'
        assert main._origen_de_recognize(None, 'pc') == 'escritorio'
        assert main._origen_de_recognize('', 'mac') == 'escritorio'
        # Un valor que no conocemos NO se da por bueno: se clasifica igual.
        assert main._origen_de_recognize('hackeo', 'pc') == 'escritorio'

    def test_sin_aparato_conocido_no_se_inventa(self, monkeypatch):
        monkeypatch.setattr(main, 'tipo_de_aparato', lambda d: None)
        assert main._origen_de_recognize(None, 'x') is None
        assert main._origen_de_recognize(None, None) is None

    def test_la_sesion_solo_si_tiene_forma_de_id(self):
        assert main._sesion_de_recognize('abc-123_X') == 'abc-123_X'
        assert main._sesion_de_recognize(None) is None
        assert main._sesion_de_recognize('') is None
        assert main._sesion_de_recognize("x'; DROP TABLE") is None
        assert main._sesion_de_recognize('a' * 65) is None


# ── El cupo cuenta pulsaciones ──────────────────────────────────────────

class TestCupoPorPulsacion:
    def _db(self, tmp_path):
        return AnalysisDB(str(tmp_path / 'a.db'))

    def _marca(self, db, dev, sesion, reason='audio_unusable', origen='escuchar'):
        db.log_audd_call(fingerprint='recognize_session', success=False,
                         source='recognize_session', device_id=dev,
                         reason=reason, origen=origen, sesion=sesion)

    def test_cuatro_peticiones_de_una_pulsacion_son_un_uso(self, tmp_path):
        db = self._db(tmp_path)
        for _ in range(4):
            self._marca(db, 'd', 's1')
        assert db.count_recognition_sessions_today('d') == 1
        self._marca(db, 'd', 's2')
        assert db.count_recognition_sessions_today('d') == 2

    def test_sin_sesion_cuenta_cada_peticion_como_antes(self, tmp_path):
        db = self._db(tmp_path)
        for _ in range(3):
            self._marca(db, 'd', None)
        assert db.count_recognition_sessions_today('d') == 3

    def test_sesion_contada_hoy(self, tmp_path):
        db = self._db(tmp_path)
        assert not db.sesion_contada_hoy('d', 's1')
        self._marca(db, 'd', 's1')
        assert db.sesion_contada_hoy('d', 's1')
        assert not db.sesion_contada_hoy('otro', 's1')
        assert not db.sesion_contada_hoy('d', None)

    def test_el_reintento_de_una_pulsacion_no_choca_con_el_cupo(
            self, client, audd, monkeypatch):
        monkeypatch.setattr(main, 'RECOGNIZE_PRO_DAILY_CAP', 1)
        dev = f'cupo-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(None, False), (None, False), (None, False),
                           (None, False), (None, False), (None, False)]
        r1 = _post(client, device_id=dev, origen='escuchar', sesion_id='p1')
        assert r1.json()['status'] == 'not_found'
        # Mismo `sesion_id`: es la misma pulsacion, no un uso nuevo.
        r2 = _post(client, device_id=dev, origen='escuchar', sesion_id='p1')
        assert r2.json()['status'] == 'not_found'
        # Una pulsacion nueva ya no cabe.
        r3 = _post(client, device_id=dev, origen='escuchar', sesion_id='p2')
        assert r3.json()['status'] == 'cap_reached'


# ── El panel separa Escuchar ────────────────────────────────────────────

class TestResumenEscuchar:
    def test_por_pulsacion_gana_el_mejor_desenlace_y_el_escritorio_no_entra(
            self, tmp_path):
        db = AnalysisDB(str(tmp_path / 'a.db'))

        def marca(sesion, reason, origen='escuchar'):
            db.log_audd_call(fingerprint='recognize_session',
                             success=reason == 'matched',
                             source='recognize_session', device_id='d',
                             reason=reason, origen=origen, sesion=sesion)

        # Pulsacion 1: audio malo y luego acierto → UN acierto.
        marca('p1', 'audio_unusable')
        marca('p1', 'matched')
        # Pulsacion 2: no esta en AudD.
        marca('p2', 'no_match')
        # Pulsacion 3: audio malo dos veces.
        marca('p3', 'audio_unusable')
        marca('p3', 'audio_unusable')
        # El escritorio NO es Escuchar.
        for _ in range(5):
            marca(None, 'no_match', origen='portada')
        marca(None, 'matched', origen='editar')
        # Una fila vieja sin origen.
        marca(None, 'no_match', origen=None)
        # Llamadas reales a AudD.
        for o in ('escuchar', 'escuchar', 'portada'):
            db.log_audd_call(fingerprint='recognize', success=False,
                             source='recognize', device_id='d', origen=o)

        r = db.resumen_escuchar(days=30)
        assert r['pulsaciones'] == 3
        assert r['peticiones'] == 5
        assert r['desenlace'] == {'matched': 1, 'no_match': 1,
                                  'audio_unusable': 1}
        assert r['por_origen']['portada']['no_match'] == 5
        assert r['por_origen']['editar']['matched'] == 1
        assert r['por_origen']['sin_origen']['peticiones'] == 1
        assert r['llamadas_audd'] == {'escuchar': 2, 'portada': 1}

    def test_sale_en_el_panel(self):
        import routes.admin_panel as panel
        assert panel._resumen_escuchar() is not None, 'None = el panel fallo'
        movil = panel._escuchar_segun_el_movil()
        assert movil is not None, 'None = el panel fallo'
        assert {'pulsaciones', 'por_desenlace', 'ms_acierto', 'guardadas',
                'enlaces', 'pendientes'} <= set(movil)

    def test_los_eventos_del_movil_se_leen(self):
        import routes.admin_panel as panel
        antes = panel._escuchar_segun_el_movil()
        dev = f'movil-{uuid.uuid4().hex[:8]}'
        for props in ({'outcome': 'found', 'ms': 9000, 'ms_servidor': 2500},
                      {'outcome': 'found', 'ms': 15000},
                      {'outcome': 'sin_permiso'},
                      {'outcome': 'sin_red'}):
            main.db.log_event(device_id=dev, event_name='listen_result',
                              props=json.dumps(props), platform='ios')
        main.db.log_event(device_id=dev, event_name='listen_saved',
                          platform='ios')
        main.db.log_event(device_id=dev, event_name='listen_link',
                          props=json.dumps({'destino': 'beatport'}),
                          platform='ios')
        main.db.log_event(device_id=dev, event_name='listen_pendiente',
                          props=json.dumps({'outcome': 'found'}),
                          platform='ios')
        r = panel._escuchar_segun_el_movil()
        assert r['pulsaciones'] - antes['pulsaciones'] == 4
        assert r['por_desenlace']['sin_permiso'] >= 1
        assert r['por_desenlace']['sin_red'] >= 1
        assert r['ms_acierto']['n'] - antes['ms_acierto']['n'] == 2
        assert r['guardadas'] - antes['guardadas'] == 1
        assert r['enlaces'].get('beatport', 0) >= 1
        assert r['pendientes'].get('found', 0) >= 1

    def test_el_modo_set_se_lee(self):
        # El modo set (2026-09-30): cuántos sets, cuántos minutos y cuántos
        # temas por hora apunta Shazam.
        import routes.admin_panel as panel
        antes = panel._escuchar_segun_el_movil()['modo_set']
        dev = f'movil-{uuid.uuid4().hex[:8]}'
        for props in ({'minutos': 60, 'temas': 18, 'errores': 2,
                       'motivo': 'boton'},
                      {'minutos': 30, 'temas': 9, 'motivo': 'sin_audio'}):
            main.db.log_event(device_id=dev, event_name='listen_set_ended',
                              props=json.dumps(props), platform='ios')
        r = panel._escuchar_segun_el_movil()['modo_set']
        assert r['sets'] - antes['sets'] == 2
        assert r['temas'] - antes['temas'] == 27
        assert r['minutos'] - antes['minutos'] == 90
        assert r['errores'] - antes['errores'] == 2
        assert r['por_motivo'].get('sin_audio', 0) >= 1
        assert r['temas_por_hora'] is not None

    def test_una_pendiente_resuelta_por_la_firma_de_shazam(self):
        # Sin red, con Shazam delante, la captura lleva su firma y se busca en
        # Shazam antes que en AudD: el panel cuenta cuántas se ahorró AudD.
        import routes.admin_panel as panel
        antes = panel._escuchar_segun_el_movil()['pendientes_resueltas_por']
        dev = f'movil-{uuid.uuid4().hex[:8]}'
        for quien in ('shazam', 'shazam', 'audd'):
            main.db.log_event(device_id=dev, event_name='listen_pendiente',
                              props=json.dumps({'outcome': 'found',
                                                'resuelto_por': quien}),
                              platform='ios')
        main.db.log_event(device_id=dev, event_name='listen_pendiente',
                          props=json.dumps({'outcome': 'no_match',
                                            'resuelto_por': None}),
                          platform='ios')
        r = panel._escuchar_segun_el_movil()['pendientes_resueltas_por']
        assert r['shazam'] - antes['shazam'] == 2
        assert r['audd'] - antes['audd'] == 1

    def test_la_clave_esta_en_la_respuesta_del_panel(self):
        ruta = os.path.join(os.path.dirname(__file__), 'routes',
                            'admin_panel.py')
        with open(ruta, encoding='utf-8') as fh:
            src = fh.read()
        assert '"servidor_30d": _resumen_escuchar()' in src
        assert '"movil_30d": _escuchar_segun_el_movil()' in src


# ── /recognize de punta a punta ─────────────────────────────────────────

class TestRecognize:
    def test_escuchar_apunta_origen_y_sesion(self, client, audd):
        dev = f'mov-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(None, True)]
        r = _post(client, device_id=dev, origen='escuchar', sesion_id='s-1')
        assert r.json()['status'] == 'not_found'
        assert isinstance(r.json()['ms_servidor'], int)
        filas = _marcadores(dev)
        assert {(f['source'], f['origen'], f['sesion']) for f in filas} == {
            ('recognize', 'escuchar', 's-1'),
            ('recognize_session', 'escuchar', 's-1'),
        }

    def test_el_escritorio_viejo_no_cuenta_como_escuchar(
            self, client, audd, monkeypatch):
        monkeypatch.setattr(main, 'tipo_de_aparato', lambda d: 'windows')
        dev = f'pc-{uuid.uuid4().hex[:8]}'
        audd.respuestas = [(None, True)]
        _post(client, device_id=dev)
        assert {f['origen'] for f in _marcadores(dev)} == {'escritorio'}

    def test_acierto_trae_la_ficha_y_la_portada_de_audd(
            self, client, audd, monkeypatch):
        buscadas = []
        monkeypatch.setattr(main, 'search_artwork_online',
                            lambda a, t, album=None: buscadas.append((a, t)))
        td = _track_data()
        audd.respuestas = [(td, True)]
        r = _post(client, device_id='d', origen='escuchar', sesion_id='s')
        j = r.json()
        assert j['status'] == 'found'
        assert j['busco_ficha'] is True
        assert 'backend_analysis' in j and j['backend_analysis'] is None
        # La portada de Escuchar es la que AudD ya trae (la del audio exacto),
        # no la de una busqueda propia en el camino critico.
        assert j['artwork_url'] == SPOTIFY_IMG
        # La deteccion se apunta igual, despues de responder.
        detect_id = main.hashlib.md5(
            f"{td['artist'].lower()}|{td['title'].lower()}".encode()).hexdigest()
        assert main.db.get_track_by_fingerprint(detect_id) is not None
        assert buscadas == [(td['artist'], td['title'])]

    def test_la_ficha_no_es_la_deteccion_de_bpm_cero(self, client, audd):
        isrc = f'ES{uuid.uuid4().hex[:10].upper()}'
        artista, titulo = f'A {uuid.uuid4().hex[:6]}', f'T {uuid.uuid4().hex[:6]}'
        _guardar_analizado(
            artist=artista, title=titulo, bpm=128.0, genre='Techno',
            isrc=isrc, bpm_source='analysis',
            analyzed_at='2024-01-01T00:00:00')
        # La deteccion de Escuchar, MAS NUEVA y con el mismo ISRC.
        main._guardar_deteccion(artista, titulo, None, None, isrc, False)
        audd.respuestas = [(_track_data(isrc=isrc, artist=artista,
                                        title=titulo), True)]
        j = _post(client, device_id='d', origen='escuchar').json()
        ficha = j['backend_analysis']
        assert ficha is not None
        assert ficha['bpm'] == 128.0
        assert ficha['genre'] == 'Techno'

    def test_la_ficha_pasa_por_lo_mejor_del_cluster(
            self, client, audd, monkeypatch):
        artista, titulo = f'A {uuid.uuid4().hex[:6]}', f'T {uuid.uuid4().hex[:6]}'
        _guardar_analizado(
            artist=artista, title=titulo, bpm=63.9, bpm_source='analysis',
            key_source='analysis')
        monkeypatch.setattr(main, '_lo_mejor_para', lambda *a, **k: {
            'bpm': 127.8, 'bpm_source': 'rekordbox',
            'key': 'Gm', 'camelot': '6A', 'key_source': 'traktor'})
        audd.respuestas = [(_track_data(artist=artista, title=titulo), True)]
        ficha = _post(client, origen='escuchar').json()['backend_analysis']
        assert ficha['bpm'] == 127.8 and ficha['bpm_source'] == 'rekordbox'
        assert ficha['camelot'] == '6A' and ficha['key_source'] == 'traktor'

    def test_con_ficha_la_portada_es_una_que_se_puede_ver(
            self, client, audd, monkeypatch, tmp_path):
        # «Adagio for Strings» (iPhone del owner, 2026-09-29): el tema tenia
        # ficha, la fila guardaba la URL de la portada del motor local que lo
        # analizo, y Escuchar la mandaba por delante de la de AudD: hueco.
        monkeypatch.setattr(main, 'ARTWORK_CACHE_DIR', str(tmp_path))
        artista, titulo = f'A {uuid.uuid4().hex[:6]}', f'T {uuid.uuid4().hex[:6]}'
        fp = _guardar_analizado(
            artist=artista, title=titulo, bpm=140.0,
            artwork_url='http://127.0.0.1:8000/artwork/lo-que-sea')
        audd.respuestas = [(_track_data(artist=artista, title=titulo), True)]
        j = _post(client, origen='escuchar').json()
        assert j['artwork_url'] == SPOTIFY_IMG
        assert j['backend_analysis']['artwork_url'] == SPOTIFY_IMG

        # Sin portada de AudD: la de Render, SOLO si Render tiene el fichero.
        sin_portada = _track_data(artist=artista, title=titulo)
        sin_portada['spotify'] = None
        audd.respuestas = [(sin_portada, True)]
        j = _post(client, origen='escuchar').json()
        assert j['artwork_url'] is None
        assert j['backend_analysis']['artwork_url'] is None

        (tmp_path / f'{fp}.jpg').write_bytes(b'\xff\xd8jpeg')
        audd.respuestas = [(sin_portada, True)]
        j = _post(client, origen='escuchar').json()
        assert j['artwork_url'] == f'{main.BASE_URL}/artwork/{fp}'
        assert j['backend_analysis']['artwork_url'] == j['artwork_url']

    def test_la_busqueda_de_portada_nunca_va_suelta_en_el_async_def(self):
        # La busqueda propia hace red (6 s de timeout por fuente). Dentro del
        # handler async congela el worker entero: solo por `_guardar_deteccion`,
        # que se llama en un hilo o como tarea de fondo.
        src = inspect.getsource(main.recognize_audio)
        assert 'search_artwork_online(' not in src
        assert '_guardar_deteccion(' not in src

    def test_no_pide_musicbrainz_a_audd(self):
        assert 'musicbrainz' not in inspect.getsource(main._send_to_audd).split(
            "'return'")[1].split('\n')[0]


# ── /search-analyzed comparte la regla ──────────────────────────────────

class TestSearchAnalyzed:
    def test_el_isrc_no_se_queda_detras_de_una_deteccion(self, client):
        isrc = f'ES{uuid.uuid4().hex[:10].upper()}'
        artista, titulo = f'A {uuid.uuid4().hex[:6]}', f'T {uuid.uuid4().hex[:6]}'
        _guardar_analizado(
            artist='otro nombre', title='otro titulo', bpm=124.0, isrc=isrc,
            analyzed_at='2023-01-01T00:00:00')
        main._guardar_deteccion(artista, titulo, None, None, isrc, False)
        j = client.get('/search-analyzed', params={
            'artist': artista, 'title': titulo, 'isrc': isrc}).json()
        assert j['found'] is True
        assert j['track']['bpm'] == 124.0

    def test_un_titulo_que_es_solo_el_sufijo_no_casa_con_todo(self, client):
        artista = f'A {uuid.uuid4().hex[:6]}'
        _guardar_analizado(artist=artista, title='Otra Cosa', bpm=130.0)
        j = client.get('/search-analyzed', params={
            'artist': artista, 'title': 'Remix'}).json()
        assert j['found'] is False


# ── «¿Lo tengo?» por el FICHERO, no por el nombre ───────────────────────

class TestHuellasDelTema:
    def _db(self, tmp_path):
        return AnalysisDB(str(tmp_path / 'a.db'))

    def _fila(self, db, fp, **campos):
        fila = {'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3',
                'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
                'duration': 400, 'bpm': 128.0}
        fila.update(campos)
        db.save_track(fila)

    def test_el_cluster_y_el_isrc_sin_la_deteccion(self, tmp_path):
        db = self._db(tmp_path)
        self._fila(db, 'a' * 32, acoustic_id='clusterX')
        self._fila(db, 'b' * 32, acoustic_id='clusterX')   # otra codificación
        self._fila(db, 'c' * 32, isrc='ESX')               # mismo ISRC, sin huella
        self._fila(db, 'd' * 32)                           # otro tema
        # La detección de Escuchar: bpm 0 y una huella que no es de ningún
        # fichero (MD5 de «artista|titulo»).
        self._fila(db, 'e' * 32, bpm=0, isrc='ESX',
                   analysis_status='recognize_only')
        huellas = db.huellas_del_tema('a' * 32, 'ESX')
        assert set(huellas) == {'a' * 32, 'b' * 32, 'c' * 32}

    def test_sin_huella_ni_isrc_no_hay_nada(self, tmp_path):
        assert self._db(tmp_path).huellas_del_tema(None, None) == []

    def test_la_ficha_de_recognize_las_trae(self, client, audd):
        isrc = f'ES{uuid.uuid4().hex[:10].upper()}'
        artista, titulo = f'A {uuid.uuid4().hex[:6]}', f'T {uuid.uuid4().hex[:6]}'
        fp = _guardar_analizado(artist=artista, title=titulo, bpm=126.0,
                                isrc=isrc)
        otra = _guardar_analizado(artist='tags basura', title='track 01',
                                  bpm=126.0, isrc=isrc)
        audd.respuestas = [(_track_data(isrc=isrc, artist=artista,
                                        title=titulo), True)]
        ficha = _post(client, origen='escuchar').json()['backend_analysis']
        assert {fp, otra} <= set(ficha['huellas_del_tema'])
