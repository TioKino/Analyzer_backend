"""Dedup del pre-check cuando el motor local tiene la BD vacia.

Escenario real (Mac formateado, 2026-08-19): la BD del motor local es LOCAL a
esa maquina. Tras formatear esta vacia, asi que el pre-check por huella del
cliente fallaba SIEMPRE y toda la biblioteca se volvia a subir y analizar,
aunque Render ya tuviera cada track. Ahora el motor local pregunta a Render
antes de contestar "no analizado".

Y desde el 2026-10-06 le pregunta por LOTE (`_precheck_en_render`), no huella
a huella: era un GET bloqueante de hasta 5 s por cada huella que el motor no
tenia, en el event loop del motor. Con Render dormido, dos minutos por
ventana de 25 temas.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

import main
from main import app
import routes.analysis_artwork as lookup

client = TestClient(app)

FP = 'a' * 32


@pytest.fixture
def render_says_yes(monkeypatch):
    """Simula el motor local: Render conoce FP, la BD local no."""
    lotes = []

    def fake_precheck(fps, token=None):
        lotes.append(list(fps))
        return {fp for fp in fps if fp == FP}

    def fake_fetch(fp):
        if fp == FP:
            return {
                'id': FP, 'filename': 'x.mp3', 'artist': 'A', 'title': 'T',
                'bpm': 128.0, 'key': 'Fm', 'camelot': '4A', 'duration': 300.0,
                'bpm_source': 'rekordbox', 'key_source': 'rekordbox',
            }
        return None

    monkeypatch.setattr(lookup, 'precheck_en_render', fake_precheck)
    monkeypatch.setattr(lookup, 'fetch_render_cache', fake_fetch)
    return lotes


def test_precheck_uses_render_when_local_db_empty(render_says_yes):
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [FP]})
    assert r.status_code == 200
    assert r.json()['analyzed'] == [FP]
    assert render_says_yes == [[FP]]


def test_precheck_still_reports_unknown_fingerprints(render_says_yes):
    unknown = 'b' * 32
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [unknown]})
    assert r.json()['not_analyzed'] == [unknown]
    assert r.json()['analyzed'] == []


def test_lo_que_falta_va_a_render_en_UNA_peticion(render_says_yes):
    """EL CASO: 25 huellas que el motor no tiene son UNA pregunta a Render,
    no 25. Y lo que el motor ya tiene no se pregunta."""
    local = uuid.uuid4().hex
    main.db.save_track({
        'id': local, 'fingerprint': local, 'filename': f'{local}.mp3',
        'bpm': 128.0, 'key': 'Am', 'duration': 300, 'energy_dj': 7,
        'genre': 'Techno', 'track_type': 'peak_time',
        'analysis_version': main.ANALYSIS_VERSION,
    })
    otras = [uuid.uuid4().hex for _ in range(24)]
    r = client.post('/check-analyzed-by-fingerprint',
                    json={'fingerprints': [local, FP] + otras})
    body = r.json()
    assert len(render_says_yes) == 1
    assert local not in render_says_yes[0]
    assert len(render_says_yes[0]) == 25
    assert body['analyzed'] == [local, FP]
    assert body['not_analyzed'] == otras


def test_by_fingerprint_serves_render_payload(render_says_yes):
    # Sin esto el pre-check decia "ya analizado" y este endpoint devolvia 404,
    # asi que el cliente acababa subiendo el fichero igual.
    r = client.get(f'/analysis/by-fingerprint/{FP}')
    assert r.status_code == 200
    assert r.json()['bpm_source'] == 'rekordbox'


def test_render_failure_never_breaks_the_precheck(monkeypatch):
    def boom(fps, token=None):
        raise RuntimeError('Render dormido')

    monkeypatch.setattr(lookup, 'precheck_en_render', boom)
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [FP]})
    assert r.status_code == 200
    assert r.json()['not_analyzed'] == [FP]


def test_render_not_consulted_when_not_local_engine(monkeypatch):
    # En Render el hook es None: no debe consultarse a si mismo.
    monkeypatch.setattr(lookup, 'precheck_en_render', None)
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [FP]})
    assert r.json()['not_analyzed'] == [FP]


def test_el_lote_no_pregunta_huella_a_huella(monkeypatch):
    """Una consulta por huella eran 500 conexiones seguidas en el único
    worker de Render por cada lote del móvil."""
    def una_a_una(fp):
        raise AssertionError('el pre-check no consulta huella a huella')

    monkeypatch.setattr(main.db, 'get_track_by_fingerprint', una_a_una)
    monkeypatch.setattr(lookup, 'precheck_en_render', None)
    fps = [uuid.uuid4().hex for _ in range(300)]
    r = client.post('/check-analyzed-by-fingerprint', json={'fingerprints': fps})
    assert r.status_code == 200
    assert r.json()['not_analyzed'] == fps


def test_el_lote_entra_por_indice():
    vistas = []
    abrir = main.db._open_conn

    def con_traza():
        conn = abrir()
        conn.set_trace_callback(vistas.append)
        return conn

    main.db._open_conn = con_traza
    try:
        main.db.filas_por_huella([uuid.uuid4().hex for _ in range(5)],
                                 'id, fingerprint, analysis_version, bpm, key')
    finally:
        main.db._open_conn = abrir
    selects = [s for s in vistas if s.startswith('SELECT')]
    assert len(selects) == 2, 'por huella y por id, no un OR ni una por huella'
    conn = abrir()
    try:
        for sql in selects:
            plan = ' | '.join(str(f[3]) for f in
                              conn.execute('EXPLAIN QUERY PLAN ' + sql))
            assert 'SCAN tracks' not in plan, plan
    finally:
        conn.close()


def test_los_registros_antiguos_casan_por_id():
    """En los registros antiguos el id ES el MD5 y `fingerprint` es otro."""
    viejo = uuid.uuid4().hex
    main.db.save_track({
        'id': viejo, 'fingerprint': None, 'filename': f'{viejo}.mp3',
        'bpm': 125.0, 'key': 'Fm', 'duration': 300, 'energy_dj': 6,
        'genre': 'House', 'track_type': 'groove',
        'analysis_version': main.ANALYSIS_VERSION,
    })
    filas = main.db.filas_por_huella([viejo], 'bpm')
    assert filas[viejo]['bpm'] == 125.0


# ============================================================================
# LA VARA DEL MOTOR LOCAL (lo que pregunta a Render)
# ============================================================================

def _fila(**extra):
    fp = uuid.uuid4().hex
    fila = {
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3', 'bpm': 128.0,
        'key': 'Am', 'duration': 300, 'energy_dj': 7, 'genre': 'Techno',
        'track_type': 'peak_time', 'analysis_version': main.ANALYSIS_VERSION,
    }
    fila.update(extra)
    main.db.save_track(fila)
    return fp


def test_el_motor_local_no_cuenta_un_fallback_fallido(monkeypatch):
    """Una fila del fallback (`analysis_status='failed'`: bpm 0, sin
    tonalidad) no le sirve al motor local, que tiene librosa. Para un móvil sí
    cuenta como analizado, como siempre: subirlo daría lo mismo."""
    monkeypatch.setattr(lookup, 'precheck_en_render', None)
    buena = _fila()
    fallida = _fila(bpm=0, key=None)
    vieja = _fila(analysis_version='0-vieja')
    otra_version = _fila(analysis_version='9.9.9')

    def pedir(**extra):
        return client.post('/check-analyzed-by-fingerprint', json={
            'fingerprints': [buena, fallida, vieja, otra_version], **extra,
        }).json()['analyzed']

    assert pedir() == [buena, fallida]
    assert pedir(version='9.9.9', con_datos=True) == [otra_version]
    assert pedir(version=main.ANALYSIS_VERSION, con_datos=True) == [buena]


def test_el_motor_local_manda_su_vara(monkeypatch):
    enviado = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'analyzed': [FP]}

    def falso_post(url, json=None, headers=None, timeout=None):
        enviado.update(url=url, json=json, headers=headers, timeout=timeout)
        return _Resp()

    monkeypatch.setattr(main.requests, 'post', falso_post)
    assert main._precheck_en_render([FP, 'corta', ''], 'tok-x') == {FP}
    assert enviado['url'].endswith('/check-analyzed-by-fingerprint')
    assert enviado['json'] == {'fingerprints': [FP],
                               'version': main.ANALYSIS_VERSION,
                               'con_datos': True}
    assert enviado['headers'] == {'X-Device-Token': 'tok-x'}, (
        'el token del cliente va a Render: así cuenta en la popularidad')
    assert enviado['timeout'] <= 15
    main._precheck_en_render([FP])
    assert enviado['headers'] == {}


# ============================================================================
# LA POPULARIDAD CUENTA A QUIEN ENTRA POR EL PRE-CHECK
# ============================================================================

def _aparato():
    import sync_endpoints as se

    conn = se._get_conn()
    device_id, token = uuid.uuid4().hex, 'tok-' + uuid.uuid4().hex
    conn.execute('INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)',
                 ('u-' + device_id, se._now_iso()))
    conn.execute(
        'INSERT INTO user_devices (device_id, user_id, device_type, device_name, '
        "linked_at, device_token) VALUES (?, ?, 'ios', 'iPhone', ?, ?)",
        (device_id, 'u-' + device_id, se._now_iso(), token))
    conn.commit()
    return device_id, token


@pytest.fixture
def en_render(monkeypatch):
    monkeypatch.setattr(lookup, 'precheck_en_render', None)
    monkeypatch.setattr(lookup, 'registrar_analistas',
                        main.db.registrar_analistas)


def test_el_atajo_del_precheck_suma_un_dj(en_render):
    """EL CASO: el móvil importa un tema que otro DJ ya analizó. Desde el
    2026-10-02 va por el pre-check y no sube el fichero, así que /analyze —y su
    `registrar_analista`— no se llamaba: el tema seguía diciendo «1 DJ»."""
    fp = _fila()
    main.db.increment_popularity(fp, 'el-que-lo-analizo')
    _, token = _aparato()
    desconocida = uuid.uuid4().hex
    r = client.post('/check-analyzed-by-fingerprint',
                    json={'fingerprints': [fp, desconocida]},
                    headers={'X-Device-Token': token})
    assert r.json()['analyzed'] == [fp]
    pop = main.db.get_track_popularity(fp)
    assert pop['dj_count'] == 2
    assert pop['analysis_count'] == 1, 'reimportar no es analizar otra vez'
    assert main.db.get_track_popularity(desconocida)['dj_count'] == 0
    # Preguntar otra vez no suma.
    client.post('/check-analyzed-by-fingerprint',
                json={'fingerprints': [fp]}, headers={'X-Device-Token': token})
    assert main.db.get_track_popularity(fp)['dj_count'] == 2


def test_sin_un_aparato_de_verdad_no_cuenta(en_render):
    """La huella basta para preguntar: sin token cualquiera inflaría los DJs."""
    fp = _fila()
    main.db.increment_popularity(fp, 'el-que-lo-analizo')
    client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [fp]})
    client.post('/check-analyzed-by-fingerprint', json={'fingerprints': [fp]},
                headers={'X-Device-Token': 'tok-' + 'f' * 32})
    assert main.db.get_track_popularity(fp)['dj_count'] == 1


def test_el_motor_local_no_cuenta_en_su_bd(monkeypatch):
    """En el motor local no se apunta nada: su BD no es la de la comunidad.
    Lo que cuenta es lo que reenvía a Render con el token."""
    monkeypatch.setattr(lookup, 'registrar_analistas', None)
    tokens = []

    def reenvio(fps, token=None):
        tokens.append(token)
        return set()

    monkeypatch.setattr(lookup, 'precheck_en_render', reenvio)
    fp = _fila()
    _, token = _aparato()
    client.post('/check-analyzed-by-fingerprint',
                json={'fingerprints': [fp, uuid.uuid4().hex]},
                headers={'X-Device-Token': token})
    assert tokens == [token]
    assert main.db.get_track_popularity(fp)['dj_count'] == 0
