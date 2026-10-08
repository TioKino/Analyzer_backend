"""Quién es cada FICHERO, verificado por Shazam, llega a todos los que tienen
ese sonido (2026-10-07).

La pasada de Shazam del Mac del owner sabe, de cada fichero con veredicto
«seguro» (dos trozos que dan el mismo tema con los tiempos encajando), su
ISRC, su id de Shazam y su nombre. Hasta ese día eso se quedaba en el Mac:

  - Escuchar no casaba un tema de la biblioteca cuyo título no dice la versión,
    porque Render no sabía el ISRC de ese fichero;
  - `/analyze` pagaba AudD por un fichero con nombre basura aunque otra copia
    del mismo sonido ya estuviera identificada por Shazam;
  - y el nombre bueno era solo del owner.

    pytest test_la_identidad_verificada.py -v
"""

import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import AnalysisDB  # noqa: E402


def _fp():
    return uuid.uuid4().hex


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / 'analysis.db'))


def _tema(db, fp, aid=None, isrc=None, bpm=128.0):
    db.conn.execute(
        'INSERT INTO tracks (id, fingerprint, filename, acoustic_id, isrc, bpm, '
        "analysis_json) VALUES (?, ?, ?, ?, ?, ?, '{}')",
        (fp, fp, fp + '.mp3', aid, isrc, bpm))
    db.conn.commit()


def _verificada(fp, artista='Astrix & Domestic', titulo='Pure Energy',
                isrc='ILA250500123', shazam_id='1440873291'):
    return {'fingerprint': fp, 'artist': artista, 'title': titulo,
            'isrc': isrc, 'shazam_id': shazam_id}


# ── La base ───────────────────────────────────────────────────────────────────

def test_se_guarda_y_completa_el_isrc_sin_pisar_uno(db):
    con, sin = _fp(), _fp()
    _tema(db, con, isrc='GBAAA0000001')
    _tema(db, sin)
    r = db.guardar_identidades('mac', [_verificada(con), _verificada(sin)])
    assert r == {'guardadas': 2, 'isrc_completados': 1}
    isrc = {f['id']: f['isrc'] for f in db.conn.execute('SELECT id, isrc FROM tracks')}
    assert isrc[sin] == 'ILA250500123'
    assert isrc[con] == 'GBAAA0000001', 'un ISRC que ya estaba no se pisa'


def test_la_ultima_verificacion_manda_sin_perder_el_isrc(db):
    fp = _fp()
    db.guardar_identidades('mac', [_verificada(fp)])
    db.guardar_identidades('otro', [_verificada(fp, titulo='Pure Energy (Edit)',
                                                isrc=None)])
    r = db.identidad_verificada_de(fp)
    assert r['title'] == 'Pure Energy (Edit)'
    assert r['isrc'] == 'ILA250500123'
    assert r['exacta'] is True


def test_otra_copia_del_mismo_sonido_la_recibe(db):
    verificado, copia, otro, suelto = _fp(), _fp(), _fp(), _fp()
    _tema(db, verificado, aid='c1')
    _tema(db, copia, aid='c1')
    _tema(db, otro, aid='c2')
    db.guardar_identidades('mac', [_verificada(verificado)])
    r = db.identidades_verificadas([verificado, copia, otro, suelto])
    assert r[verificado]['exacta'] is True
    assert r[copia]['exacta'] is False
    assert r[copia]['title'] == 'Pure Energy'
    assert otro not in r and suelto not in r


def test_escuchar_casa_por_el_isrc_verificado(db):
    """El ISRC que da Shazam en Escuchar trae la huella del fichero aunque su
    fila tenga otro ISRC en las etiquetas."""
    fp = _fp()
    _tema(db, fp, isrc='GBAAA0000001')
    db.guardar_identidades('mac', [_verificada(fp)])
    assert fp in db.huellas_del_tema(None, 'ILA250500123')


def test_la_ficha_de_escuchar_sale_por_el_isrc(db, monkeypatch):
    """La ficha (`buscar_analizado`) entra primero por el ISRC: con el ISRC
    completado, un fichero cuyo título no dice la versión tiene ficha."""
    fp = _fp()
    _tema(db, fp)
    db.guardar_identidades('mac', [_verificada(fp)])
    assert db.get_analyzed_track_by_isrc('ILA250500123')['fingerprint'] == fp


def test_la_ficha_sale_por_el_isrc_verificado_aunque_la_fila_traiga_otro(db):
    """La fila del fichero trae el ISRC de sus etiquetas (que no se pisa) y
    Shazam verificó otro: la ficha de Escuchar sale igual por el verificado."""
    fp = _fp()
    _tema(db, fp, isrc='GBAAA0000001')
    db.guardar_identidades('mac', [_verificada(fp)])
    ficha = db.get_analyzed_track_by_isrc('ILA250500123')
    assert ficha is not None and ficha['fingerprint'] == fp
    assert db.get_analyzed_track_by_isrc('GBAAA0000001')['fingerprint'] == fp


def test_el_cluster_da_la_identidad_a_una_copia_que_aun_no_tiene_fila(db):
    verificado, viejo = _fp(), _fp()
    _tema(db, verificado, aid='c1')
    db.guardar_identidades('mac', [_verificada(verificado, titulo='Viejo')])
    _tema(db, viejo, aid='c1')
    db.guardar_identidades('otro', [_verificada(viejo)])
    r = db.identidad_verificada_del_cluster('c1')
    assert r['title'] == 'Pure Energy', 'la más reciente del cluster'
    assert r['exacta'] is False
    assert db.identidad_verificada_del_cluster('nada') is None
    assert db.identidad_verificada_del_cluster(None) is None


def test_retirar_quita_solo_lo_de_ese_aparato(db):
    mio, de_otro = _fp(), _fp()
    _tema(db, mio)
    _tema(db, de_otro)
    db.guardar_identidades('mac', [_verificada(mio)])
    db.guardar_identidades('otro', [_verificada(de_otro)])
    r = db.retirar_identidades('mac', [mio, de_otro])
    assert r == {'retiradas': 1, 'isrc_quitados': 1}
    assert db.identidad_verificada_de(mio) is None
    assert db.identidad_verificada_de(de_otro) is not None
    isrc = {f['id']: f['isrc'] for f in db.conn.execute('SELECT id, isrc FROM tracks')}
    assert isrc[mio] is None, 'el ISRC que puso la verificación se va con ella'
    assert isrc[de_otro] == 'ILA250500123'


def test_retirar_no_quita_un_isrc_que_trae_el_analisis(db):
    """Si el análisis ya traía ese ISRC (AudD o las etiquetas), no vino de la
    verificación y se queda."""
    fp = _fp()
    db.conn.execute(
        'INSERT INTO tracks (id, fingerprint, filename, isrc, bpm, analysis_json) '
        "VALUES (?, ?, 'x.mp3', 'ILA250500123', 128, ?)",
        (fp, fp, '{"isrc": "ILA250500123"}'))
    db.conn.commit()
    db.guardar_identidades('mac', [_verificada(fp)])
    assert db.retirar_identidades('mac', [fp]) == {'retiradas': 1, 'isrc_quitados': 0}
    assert db.conn.execute('SELECT isrc FROM tracks WHERE id = ?',
                           (fp,)).fetchone()['isrc'] == 'ILA250500123'


def test_el_resumen_del_panel(db):
    db.guardar_identidades('mac', [_verificada(_fp()), _verificada(_fp(), isrc=None)])
    db.apuntar_audd_no_hizo_falta('identidad')
    db.apuntar_audd_no_hizo_falta('identidad')
    db.apuntar_audd_no_hizo_falta('cluster')
    r = db.resumen_identidad_verificada(30)
    assert (r['huellas'], r['con_isrc'], r['aparatos'], r['recientes']) == (2, 1, 1, 2)
    assert r['audd_no_hizo_falta'] == {'identidad': 2, 'cluster': 1}


def test_el_panel_cuenta_lo_que_les_llega_a_los_aparatos(db):
    """Sin los eventos `identidad_comunidad`, Render sabe lo que guarda pero
    no si a alguien le sirve."""
    import json

    for dev, p in (('a', {'preguntadas': 500, 'con_identidad': 40,
                          'nombres': 3, 'isrc': 30}),
                   ('a', {'preguntadas': 20, 'con_identidad': 2,
                          'nombres': 0, 'isrc': 2}),
                   ('b', {'preguntadas': 'x'}),
                   ('c', None)):
        db.conn.execute(
            "INSERT INTO events (device_id, event_name, props) "
            "VALUES (?, 'identidad_comunidad', ?)",
            (dev, json.dumps(p) if p is not None else None))
    db.conn.execute("INSERT INTO events (device_id, event_name, props) "
                    "VALUES ('d', 'puesta_al_dia', '{\"hechos\": 9}')")
    db.conn.commit()
    r = db.resumen_identidad_verificada(30)['en_los_aparatos']
    assert r == {'aparatos': 3, 'pasadas': 4, 'preguntadas': 520,
                 'con_identidad': 42, 'nombres': 3, 'isrc': 32}


# ── La API ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def app_mod():
    import main

    return main


def _token():
    import sync_endpoints as se

    conn = se._get_conn()
    device_id, token = uuid.uuid4().hex, 'tok-' + uuid.uuid4().hex
    conn.execute('INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)',
                 ('u-' + device_id, se._now_iso()))
    conn.execute(
        'INSERT INTO user_devices (device_id, user_id, device_type, device_name, '
        "linked_at, device_token) VALUES (?, ?, 'macos-dmg', 'Mac', ?, ?)",
        (device_id, 'u-' + device_id, se._now_iso(), token))
    conn.commit()
    return token


def test_sin_aparato_registrado_401(app_mod):
    c = TestClient(app_mod.app)
    cuerpo = {'items': [{'fingerprint': _fp(), 'artist': 'A', 'title': 'B'}]}
    assert c.post('/identidad/verificada', json=cuerpo).status_code == 401


def test_se_descarta_lo_que_no_vale(app_mod):
    fp = _fp()
    r = TestClient(app_mod.app).post(
        '/identidad/verificada', headers={'X-Device-Token': _token()},
        json={'items': [
            {'fingerprint': fp, 'artist': 'Astrix & Domestic',
             'title': 'Pure Energy', 'isrc': 'il-a25-05-00123',
             'shazam_id': '1440873291'},
            {'fingerprint': 'no-es-md5', 'artist': 'A', 'title': 'B'},
            {'fingerprint': _fp(), 'artist': 'Unknown Artist', 'title': 'Track 01'},
            {'fingerprint': _fp(), 'artist': 'Alguien', 'title': 'Algo',
             'isrc': 'no-es-un-isrc'},
        ]})
    assert r.status_code == 200
    j = r.json()
    assert (j['guardadas'], j['descartadas']) == (2, 2)
    guardada = app_mod.db.identidad_verificada_de(fp)
    assert guardada['isrc'] == 'ILA250500123', 'normalizado'


def test_retirar_por_la_api(app_mod):
    fp, token = _fp(), _token()
    c = TestClient(app_mod.app)
    c.post('/identidad/verificada', headers={'X-Device-Token': token},
           json={'items': [{'fingerprint': fp, 'artist': 'Joan Reyes',
                            'title': 'Psicodelicia'}]})
    assert c.post('/identidad/verificada/retirar',
                  json={'huellas': [fp]}).status_code == 401
    r = c.post('/identidad/verificada/retirar',
               headers={'X-Device-Token': _token()}, json={'huellas': [fp]})
    assert r.json()['retiradas'] == 0, 'otro aparato no retira lo de este'
    r = c.post('/identidad/verificada/retirar',
               headers={'X-Device-Token': token}, json={'huellas': [fp, 'x']})
    assert (r.status_code, r.json()['retiradas']) == (200, 1)
    assert app_mod.db.identidad_verificada_de(fp) is None


def test_la_consulta_por_lote(app_mod):
    fp = _fp()
    TestClient(app_mod.app).post(
        '/identidad/verificada', headers={'X-Device-Token': _token()},
        json={'items': [{'fingerprint': fp, 'artist': 'Joan Reyes',
                         'title': 'Psicodelicia'}]})
    r = TestClient(app_mod.app).post(
        '/identidad/verificada/consulta', json={'huellas': [fp, _fp(), 'x']})
    assert r.status_code == 200
    assert list(r.json()['identidades']) == [fp]
    assert r.json()['identidades'][fp]['title'] == 'Psicodelicia'
    assert TestClient(app_mod.app).post(
        '/identidad/verificada/consulta',
        json={'huellas': [_fp() for _ in range(501)]}).status_code == 400


# ── /analyze no paga AudD por lo que ya se sabe ───────────────────────────────

class _Db:
    """Lo que `_identidad_y_genero` le pregunta a la BD."""

    def __init__(self, verificada):
        self.verificada, self.ahorros = verificada, []

    def identidad_verificada_de(self, fp):
        return self.verificada

    def apuntar_audd_no_hizo_falta(self, via):
        self.ahorros.append(via)


@pytest.fixture
def sin_red(monkeypatch, app_mod):
    import audd_helper

    llamadas = []
    monkeypatch.setattr(app_mod, 'AUDD_AUTO_ENABLED', True)
    monkeypatch.setattr(app_mod, 'AUDD_API_TOKEN', 'token')
    monkeypatch.setattr(app_mod, 'GENRE_DETECTOR_ENABLED', False)
    monkeypatch.setattr(app_mod, '_cluster_clean_identity', lambda *a: None)
    monkeypatch.setattr(app_mod, '_verificada_del_sonido', lambda *a: None)
    def audd(**kw):
        # Como el de verdad: solo salta con nombre basura o a la fuerza.
        if not kw['force'] and not audd_helper.is_garbage_metadata(
                kw['artist'], kw['title']):
            return None
        llamadas.append(kw)
        return {'artist': 'AudD', 'title': 'Dice', 'isrc': None}

    monkeypatch.setattr(audd_helper, 'enrich_with_audd_if_needed', audd)
    return llamadas


def _identidad(app_mod, id3, **kw):
    return app_mod._identidad_y_genero('/tmp/tmpabc.mp3', _fp(), 400.0, id3,
                                       'Track 01.mp3', **kw)


def test_EL_CASO_nombre_basura_y_ya_se_sabe_quien_es(app_mod, sin_red, monkeypatch):
    falso = _Db({'artist': 'Astrix & Domestic', 'title': 'Pure Energy',
                 'isrc': 'ILA250500123', 'exacta': False})
    monkeypatch.setattr(app_mod, 'db', falso)
    r = _identidad(app_mod, {'artist': 'Unknown Artist', 'title': 'Track 01'})
    assert (r['artist'], r['title']) == ('Astrix & Domestic', 'Pure Energy')
    assert r['audd_isrc'] == 'ILA250500123'
    assert sin_red == [], 'AudD no se llama'
    assert falso.ahorros == ['identidad']


def test_una_copia_NUEVA_con_nombre_basura_tampoco_paga(app_mod, sin_red,
                                                        monkeypatch):
    """Otros bytes del mismo tema: por su huella no hay nada (no tiene fila),
    por el cluster de su audio sí. Es el caso del ahorro: el segundo DJ."""
    falso = _Db(None)
    monkeypatch.setattr(app_mod, 'db', falso)
    monkeypatch.setattr(app_mod, '_verificada_del_sonido', lambda *a: {
        'artist': 'Astrix & Domestic', 'title': 'Pure Energy',
        'isrc': 'ILA250500123', 'exacta': False})
    r = _identidad(app_mod, {'artist': 'Unknown Artist', 'title': 'Track 01'})
    assert (r['artist'], r['title']) == ('Astrix & Domestic', 'Pure Energy')
    assert r['audd_isrc'] == 'ILA250500123'
    assert sin_red == [] and falso.ahorros == ['identidad']


def test_con_nombre_limpio_no_se_saca_el_cluster_del_audio(app_mod, sin_red,
                                                           monkeypatch):
    monkeypatch.setattr(app_mod, 'db', _Db(None))
    def no(*a):
        raise AssertionError('fpcalc sobra con un nombre limpio')
    monkeypatch.setattr(app_mod, '_verificada_del_sonido', no)
    r = _identidad(app_mod, {'artist': 'Astral Projection', 'title': 'Pure NRG'})
    assert r['artist'] == 'Astral Projection'


def test_limpiar_con_audd_tampoco_paga(app_mod, sin_red, monkeypatch):
    falso = _Db({'artist': 'Joan Reyes', 'title': 'Psicodelicia',
                 'isrc': None, 'exacta': True})
    monkeypatch.setattr(app_mod, 'db', falso)
    r = _identidad(app_mod, {'artist': 'joan reyes', 'title': 'psicodellcia'},
                   force_audd=True)
    assert (r['artist'], r['title']) == ('Joan Reyes', 'Psicodelicia')
    assert sin_red == []
    assert falso.ahorros == ['limpiar']


def test_un_nombre_limpio_no_se_toca_pero_el_isrc_si_vale(app_mod, sin_red,
                                                          monkeypatch):
    falso = _Db({'artist': 'Astrix & Domestic', 'title': 'Pure Energy',
                 'isrc': 'ILA250500123', 'exacta': True})
    monkeypatch.setattr(app_mod, 'db', falso)
    r = _identidad(app_mod, {'artist': 'Astral Projection', 'title': 'Pure NRG'})
    assert (r['artist'], r['title']) == ('Astral Projection', 'Pure NRG'), \
        'qué nombre se ve lo decide el cliente'
    assert r['audd_isrc'] == 'ILA250500123'
    assert falso.ahorros == []


def test_sin_identidad_sigue_como_siempre(app_mod, sin_red, monkeypatch):
    monkeypatch.setattr(app_mod, 'db', _Db(None))
    r = _identidad(app_mod, {'artist': 'Unknown Artist', 'title': 'Track 01'})
    assert (r['artist'], r['title']) == ('AudD', 'Dice')
    assert len(sin_red) == 1


def test_una_bd_que_falla_no_tumba_el_analisis(app_mod, sin_red, monkeypatch):
    class _Rota:
        def identidad_verificada_de(self, fp):
            raise RuntimeError('bloqueada')

    monkeypatch.setattr(app_mod, 'db', _Rota())
    r = _identidad(app_mod, {'artist': 'Unknown Artist', 'title': 'Track 01'})
    assert r['artist'] == 'AudD'


def test_el_reset_la_borra(app_mod):
    import inspect

    fuente = inspect.getsource(app_mod)
    i = fuente.index('analysis_tables = (')
    assert '"identidad_verificada"' in fuente[i:i + 600]
    assert '"audd_no_hizo_falta"' in fuente[i:i + 600]


def test_el_panel_lo_cuenta(app_mod):
    from routes import admin_panel

    assert admin_panel._identidad_verificada() is not None, \
        'None es «el panel falló», no «no hay nada»'
