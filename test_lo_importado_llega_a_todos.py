"""Lo que un DJ importa de Rekordbox/Traktor/VirtualDJ llega a TODOS los que
tienen ese sonido.

Auditado el 2026-09-26: no llegaba a nadie. Lo importado vivía en la caché del
cliente y en su sync (personal), y el servidor tenía montado el ranking que lo
adoptaría —rekordbox 110 gana a todo— sin que nada se lo mandara nunca. Un tema
reanalizado jamás salía con «Rekordbox», lo tuviera otro DJ o no.

Decisiones del owner que esto ata:
  - Con UN aparato basta («con que se detecte una vez es suficiente»). Lo que
    pide tres es el cambio hecho A MANO (test_cambio_manual_pide_tres.py).
  - Tiene que ser un aparato REGISTRADO: sin `X-Device-Token`, 401.

Y dos del diseño:
  - La rejilla (primer beat) solo vale para el MISMO fichero: dos
    codificaciones del mismo audio no empiezan en la misma muestra.
  - El género no viaja: en un XML es una etiqueta de cada uno.

    pytest test_lo_importado_llega_a_todos.py -v
"""

import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acoustic_fingerprint import encode_raw  # noqa: E402
from analysis_ranking import get_source_priority  # noqa: E402
from database import AnalysisDB  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))


def _fp():
    return uuid.uuid4().hex


# ── La base: votos, cuentas y rejilla ─────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    d = AnalysisDB(db_path=str(tmp_path / 'analysis.db'))
    d.cuentas = {}
    d.cuentas_de = lambda ids: {i: d.cuentas[i] for i in ids if i in d.cuentas}
    return d


def _voto(db, device, fp, fuente='rekordbox', bpm=None, key=None,
          camelot=None, primer=None):
    item = {'fingerprint': fp, 'source': fuente}
    if bpm is not None:
        item['bpm'] = bpm
        if primer is not None:
            item['first_beat'] = primer
    if key is not None:
        item['key'], item['camelot'] = key, camelot
    return db.guardar_lo_importado(device, [item])


def test_con_un_voto_basta(db):
    fp = _fp()
    _voto(db, 'mac', fp, bpm=128.0, key='A#m', camelot='3A', primer=0.123)
    r = db.lo_importado_de([fp], exacta=fp)
    assert r['bpm'] == 128.0 and r['bpm_source'] == 'rekordbox'
    assert r['key'] == 'A#m' and r['camelot'] == '3A'
    assert r['key_source'] == 'rekordbox'
    assert r['first_beat'] == pytest.approx(0.123)


def test_reimportar_sustituye_el_voto_del_aparato(db):
    fp = _fp()
    _voto(db, 'mac', fp, bpm=127.0)
    _voto(db, 'mac', fp, bpm=128.0)
    assert db.lo_importado_de([fp])['bpm'] == 128.0


def test_si_discrepan_gana_el_de_mas_cuentas(db):
    fp = _fp()
    _voto(db, 'a', fp, fuente='rekordbox', bpm=128.0)
    _voto(db, 'b', fp, fuente='traktor', bpm=64.0)
    _voto(db, 'c', fp, fuente='traktor', bpm=64.0)
    r = db.lo_importado_de([fp])
    assert r['bpm'] == 64.0 and r['bpm_source'] == 'traktor'


def test_a_igualdad_gana_el_programa_de_mas_rango(db):
    fp = _fp()
    _voto(db, 'a', fp, fuente='virtualdj', bpm=127.5)
    _voto(db, 'b', fp, fuente='rekordbox', bpm=128.0)
    assert db.lo_importado_de([fp])['bpm_source'] == 'rekordbox'


def test_varios_aparatos_de_una_cuenta_son_un_voto(db):
    fp = _fp()
    db.cuentas = {'mac': 'u1', 'movil': 'u1'}
    _voto(db, 'mac', fp, fuente='traktor', bpm=64.0)
    _voto(db, 'movil', fp, fuente='traktor', bpm=64.0)
    _voto(db, 'otro', fp, fuente='rekordbox', bpm=128.0)
    # 1 cuenta contra 1 cuenta: empate, gana Rekordbox por rango.
    assert db.lo_importado_de([fp])['bpm'] == 128.0


def test_la_rejilla_solo_sale_del_mismo_fichero(db):
    flac, mp3 = _fp(), _fp()
    _voto(db, 'a', flac, bpm=128.0, primer=0.5)
    # Desde el MP3 (otra versión del mismo sonido) llega el BPM, no la rejilla.
    otra = db.lo_importado_de([flac, mp3], exacta=mp3)
    assert otra['bpm'] == 128.0
    assert 'first_beat' not in otra
    assert db.lo_importado_de([flac, mp3], exacta=flac)['first_beat'] == 0.5


# ── El cluster: otra versión del mismo sonido lo hereda ───────────────────────

def _dos_versiones(db):
    """El mismo audio en FLAC y en MP3: mismo cluster, huellas distintas."""
    import random
    rnd = random.Random(11)
    raw = [rnd.getrandbits(32) for _ in range(400)]
    otro = [v ^ (1 << (i % 32)) if i % 40 == 0 else v for i, v in enumerate(raw)]
    fps = []
    for datos in (raw, otro):
        fp = _fp()
        aid = db.resolve_acoustic_cluster(datos, 300.0)
        db.save_track({
            'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3',
            'artist': 'A', 'title': 'T', 'duration': 300.0,
            'bpm': 127.9, 'key': 'C', 'camelot': '8B',
            'bpm_source': 'analysis', 'key_source': 'analysis',
            'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
            'chromaprint': encode_raw(datos), 'acoustic_id': aid,
        })
        fps.append((fp, aid))
    assert fps[0][1] == fps[1][1]
    return fps


def test_otra_version_del_mismo_sonido_lo_hereda(db):
    (flac, aid), (mp3, _) = _dos_versiones(db)
    _voto(db, 'dj-del-flac', flac, bpm=128.0, key='Am', camelot='8A',
          primer=0.25)
    best = db.best_cluster_analysis(aid)
    assert best['bpm'] == 128.0 and best['bpm_source'] == 'rekordbox'
    assert best['key'] == 'Am' and best['camelot'] == '8A'
    assert 'first_beat' not in best


def test_virtualdj_rankea_como_programa_de_dj():
    # En el servidor valía 0 (no estaba en la tabla): menos que el análisis.
    assert get_source_priority('virtualdj') == 100
    assert get_source_priority('virtualdj') > get_source_priority('consensus_3')


# ── La API, de punta a punta ──────────────────────────────────────────────────

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
        "linked_at, device_token) VALUES (?, ?, 'macos', 'Mac', ?, ?)",
        (device_id, 'u-' + device_id, se._now_iso(), token))
    conn.commit()
    return token


def _resultado(fp, fuente='analysis'):
    """Un AnalysisResult completo, como lo guarda /analyze."""
    from models import AnalysisResult

    campos = {}
    for nombre, info in AnalysisResult.model_fields.items():
        if not info.is_required():
            continue
        tipo = info.annotation
        campos[nombre] = (False if tipo is bool else 0 if tipo in (int, float)
                          else [] if 'List' in str(tipo) else '')
    campos.update(
        title='T', artist='A', duration=300.0, bpm=127.9, key='C',
        camelot='8B', bpm_source=fuente, key_source=fuente, energy_dj=7,
        genre='Techno', genre_source='analysis', track_type='peak_time',
        first_beat=0.9, fingerprint=fp,
    )
    return AnalysisResult(**campos)


def _analizado(app_mod, fuente='analysis'):
    fp = _fp()
    fila = _resultado(fp, fuente).model_dump()
    fila.update(id=fp, filename=f'{fp}.mp3')
    app_mod.db.save_track(fila)
    return fp


def test_sin_aparato_registrado_401(app_mod):
    c = TestClient(app_mod.app)
    cuerpo = {'items': [{'fingerprint': _fp(), 'source': 'rekordbox', 'bpm': 128}]}
    assert c.post('/community/imported', json=cuerpo).status_code == 401
    r = c.post('/community/imported', json=cuerpo,
               headers={'X-Device-Token': 'tok-inventado-' + 'x' * 20})
    assert r.status_code == 401


def test_se_descarta_lo_que_no_es_de_un_programa_de_dj(app_mod):
    fp = _fp()
    r = TestClient(app_mod.app).post(
        '/community/imported',
        headers={'X-Device-Token': _token()},
        json={'items': [
            {'fingerprint': fp, 'source': 'rekordbox', 'bpm': 128.0,
             'key': 'Bbm', 'first_beat': 0.2},
            {'fingerprint': fp, 'source': 'itunes', 'bpm': 128.0},
            {'fingerprint': 'no-es-md5', 'source': 'rekordbox', 'bpm': 128.0},
            {'fingerprint': _fp(), 'source': 'traktor', 'bpm': 999.0},
        ]},
    )
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'votos': 2, 'descartados': 3}
    # Bemoles de Rekordbox normalizados a la tabla de sostenidos.
    guardado = app_mod.db.lo_importado_de([fp], exacta=fp)
    assert guardado['key'] == 'A#m' and guardado['camelot'] == '3A'


def test_el_genero_no_viaja(app_mod):
    assert 'genre' not in app_mod.LoImportadoItem.model_fields


def test_la_ficha_de_otro_dj_dice_rekordbox(app_mod):
    """El pre-check del import y la ficha del cliente piden el análisis por
    huella: tiene que salir lo de Rekordbox, con su rejilla."""
    fp = _analizado(app_mod)
    TestClient(app_mod.app).post(
        '/community/imported', headers={'X-Device-Token': _token()},
        json={'items': [{'fingerprint': fp, 'source': 'rekordbox',
                         'bpm': 128.0, 'key': 'Am', 'first_beat': 0.187}]})
    r = TestClient(app_mod.app).get(f'/analysis/by-fingerprint/{fp}').json()
    assert r['bpm'] == 128.0 and r['bpm_source'] == 'rekordbox'
    assert r['key'] == 'Am' and r['key_source'] == 'rekordbox'
    assert r['first_beat'] == pytest.approx(0.187)
    assert r['grid_source'] == 'rekordbox'


def test_analyze_lo_aplica_salga_por_donde_salga(app_mod):
    """`_mejorar_con_la_comunidad` corre al salir de /analyze, también en un
    acierto de caché: el que reimporta su carpeta ve lo de Rekordbox."""
    fp = _analizado(app_mod)
    app_mod.db.guardar_lo_importado('otro-dj', [
        {'fingerprint': fp, 'source': 'traktor', 'bpm': 128.0,
         'first_beat': 0.4, 'key': 'Am', 'camelot': '8A'}])
    result = _resultado(fp)
    app_mod._mejorar_con_la_comunidad(result)
    assert (result.bpm, result.bpm_source) == (128.0, 'traktor')
    assert (result.key, result.camelot, result.key_source) == ('Am', '8A', 'traktor')
    assert (result.first_beat, result.grid_source) == (0.4, 'traktor')


def test_no_degrada_lo_que_ya_es_de_un_programa(app_mod):
    fp = _analizado(app_mod, fuente='rekordbox')
    app_mod.db.guardar_lo_importado('otro', [
        {'fingerprint': fp, 'source': 'virtualdj', 'bpm': 64.0}])
    result = _resultado(fp, fuente='rekordbox')
    app_mod._mejorar_con_la_comunidad(result)
    assert (result.bpm, result.bpm_source) == (127.9, 'rekordbox')


def test_cluster_best_contesta_por_huella_sin_chromaprint(app_mod):
    """El motor local pregunta a Render con la huella del fichero: tiene que
    llegarle lo importado, rejilla incluida, aunque no mande chromaprint."""
    fp = _fp()
    app_mod.db.guardar_lo_importado('dj', [
        {'fingerprint': fp, 'source': 'rekordbox', 'bpm': 124.0,
         'first_beat': 0.31}])
    r = TestClient(app_mod.app).post(
        '/cluster-best', json={'fingerprint': fp}).json()
    assert r['found'] is True
    assert (r['bpm'], r['bpm_source'], r['first_beat']) == (124.0, 'rekordbox', 0.31)


def test_el_reset_borra_tambien_lo_importado():
    with open(os.path.join(AQUI, 'main.py'), encoding='utf-8') as fh:
        fuente = fh.read()
    bloque = fuente[fuente.index('analysis_tables = ('):]
    assert '"imported_values"' in bloque[:bloque.index(')')]


def test_analyze_mejora_en_todas_sus_salidas():
    # /analyze tiene cinco salidas y antes solo el análisis NUEVO miraba el
    # cluster. Lo común va en el envoltorio: si alguien vuelve a meter lógica
    # en una salida suelta, este test no lo ve, pero sí que se quite de ahí.
    with open(os.path.join(AQUI, 'main.py'), encoding='utf-8') as fh:
        fuente = fh.read()
    envoltorio = fuente[fuente.index('async def analyze_track('):
                        fuente.index('async def _analizar(')]
    assert '_mejorar_con_la_comunidad' in envoltorio
