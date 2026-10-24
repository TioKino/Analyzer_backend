"""A quién LE LLEGA lo importado (2026-10-24).

Desde el 2026-09-26 el BPM, la tonalidad y la rejilla que dice el XML de
Rekordbox/Traktor/VirtualDJ de un DJ llegan a la memoria colectiva
(`imported_values`), y `/admin/telemetry` contaba cuánto se comparte. Lo que
no se medía es a cuántos les sirve: cuántos temas de OTROS DJs reciben ese
valor. Hoy `/analyze` y las fichas por huella apuntan en `imported_recibido`
a qué aparato le llega un valor de un programa de DJ cuando su cuenta no ha
votado nada de ese sonido.

    pytest test_lo_importado_le_llega_a_otros.py -v
"""

import uuid

import pytest
from fastapi.testclient import TestClient

import main
import routes.analysis_artwork as lookup
from database import AnalysisDB


def _fp():
    return uuid.uuid4().hex


@pytest.fixture
def db(tmp_path):
    d = AnalysisDB(db_path=str(tmp_path / 'analysis.db'))
    d.cuentas = {}
    d.cuentas_de = lambda ids: {i: d.cuentas[i] for i in ids if i in d.cuentas}
    return d


def _voto(db, device, fp, bpm=128.0, fuente='rekordbox'):
    db.guardar_lo_importado(device, [
        {'fingerprint': fp, 'source': fuente, 'bpm': bpm}])


def _filas(db, device=None):
    conn = db._open_conn()
    try:
        sql = 'SELECT fingerprint, device_id, field, primera, ultima FROM imported_recibido'
        args = ()
        if device:
            sql += ' WHERE device_id = ?'
            args = (device,)
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


# ── La regla ──────────────────────────────────────────────────────────────────

def test_EL_CASO_le_llega_a_otro_dj(db):
    fp = _fp()
    db.cuentas = {'a': 'U1', 'b': 'U2'}
    _voto(db, 'a', fp)
    n = db.apuntar_lo_importado_recibido(
        'b', {fp: {'campos': ['bpm'], 'huellas': [fp]}})
    assert n == 1
    assert [(f['fingerprint'], f['field']) for f in _filas(db, 'b')] == [(fp, 'bpm')]


def test_a_tu_propia_cuenta_no_le_llega_de_otro(db):
    """El Mac y el móvil del mismo DJ son una cuenta: lo que importó en el Mac
    no es «de otro» cuando vuelve al móvil."""
    fp = _fp()
    db.cuentas = {'mac': 'U1', 'movil': 'U1'}
    _voto(db, 'mac', fp)
    assert db.apuntar_lo_importado_recibido(
        'movil', {fp: {'campos': ['bpm'], 'huellas': [fp]}}) == 0
    assert db.apuntar_lo_importado_recibido(
        'mac', {fp: {'campos': ['bpm'], 'huellas': [fp]}}) == 0
    assert _filas(db) == []


def test_con_un_voto_propio_en_otra_version_tampoco(db):
    """El valor ganador no dice de quién es: si tu cuenta votó ese sonido,
    aunque fuera otra versión del cluster, no se apunta. Cota inferior."""
    fp1, fp2 = _fp(), _fp()
    db.cuentas = {'a': 'U1', 'b': 'U2'}
    _voto(db, 'a', fp1)
    _voto(db, 'b', fp2)
    assert db.apuntar_lo_importado_recibido(
        'b', {fp1: {'campos': ['bpm'], 'huellas': [fp1, fp2]}}) == 0


def test_se_cuenta_una_vez_y_se_refresca_la_ultima(db):
    fp = _fp()
    _voto(db, 'a', fp)
    entrega = {fp: {'campos': ['bpm', 'key'], 'huellas': [fp]}}
    db.apuntar_lo_importado_recibido('b', entrega)
    primera = {f['field']: f['primera'] for f in _filas(db, 'b')}
    db.apuntar_lo_importado_recibido('b', entrega)
    filas = _filas(db, 'b')
    assert len(filas) == 2, 'un tema, un aparato, dos campos'
    assert {f['field']: f['primera'] for f in filas} == primera


def test_sin_aparato_no_apunta_nada(db):
    fp = _fp()
    _voto(db, 'a', fp)
    assert db.apuntar_lo_importado_recibido(
        '', {fp: {'campos': ['bpm'], 'huellas': [fp]}}) == 0


def test_el_resumen_del_panel(db):
    fp1, fp2 = _fp(), _fp()
    db.cuentas = {'a': 'U1', 'b': 'U2', 'b2': 'U2', 'c': 'U3'}
    _voto(db, 'a', fp1)
    _voto(db, 'a', fp2)
    db.apuntar_lo_importado_recibido(
        'b', {fp1: {'campos': ['bpm', 'key'], 'huellas': [fp1]}})
    db.apuntar_lo_importado_recibido(
        'b2', {fp1: {'campos': ['bpm'], 'huellas': [fp1]}})
    db.apuntar_lo_importado_recibido(
        'c', {fp2: {'campos': ['bpm'], 'huellas': [fp2]}})
    r = db.resumen_lo_importado()['recibido']
    assert (r['temas'], r['aparatos'], r['cuentas']) == (2, 3, 2)
    assert r['por_campo'] == {'bpm': 3, 'key': 1}
    assert r['ultimos_30d'] == {'temas': 2, 'aparatos': 3}


def test_sin_nada_recibido_el_resumen_dice_cero(db):
    r = db.resumen_lo_importado()['recibido']
    assert (r['temas'], r['aparatos'], r['cuentas']) == (0, 0, 0)


# ── Las puertas: /analyze y las fichas por huella ─────────────────────────────

client = TestClient(main.app)


def _fila_analizada():
    fp = _fp()
    main.db.save_track({
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3', 'bpm': 127.0,
        'key': 'Am', 'camelot': '8A', 'duration': 300, 'energy_dj': 7,
        'genre': 'Techno', 'track_type': 'peak_time', 'bpm_source': 'analysis',
        'analysis_version': main.ANALYSIS_VERSION,
    })
    return fp


@pytest.fixture
def en_render(monkeypatch):
    monkeypatch.setattr(lookup, 'fichas_en_render', None)
    assert lookup.al_entregar_fichas is not None, 'Render apunta lo recibido'


def test_la_ficha_del_lote_lo_apunta(en_render):
    fp = _fila_analizada()
    votante, receptor = _fp(), _fp()
    main.db.guardar_lo_importado(votante, [
        {'fingerprint': fp, 'source': 'rekordbox', 'bpm': 128.0}])
    r = client.post('/analysis/by-fingerprint/batch',
                    json={'fingerprints': [fp]},
                    headers={'X-Device-Id': receptor})
    assert r.json()['fichas'][fp]['bpm_source'] == 'rekordbox'
    assert [(f['fingerprint'], f['field'])
            for f in _filas(main.db, receptor)] == [(fp, 'bpm')]


def test_la_ficha_de_una_huella_lo_apunta(en_render):
    fp = _fila_analizada()
    votante, receptor = _fp(), _fp()
    main.db.guardar_lo_importado(votante, [
        {'fingerprint': fp, 'source': 'traktor', 'bpm': 128.0}])
    client.get(f'/analysis/by-fingerprint/{fp}',
               headers={'X-Device-Id': receptor})
    assert len(_filas(main.db, receptor)) == 1


def test_sin_decir_quien_es_no_se_apunta(en_render):
    """Un cliente anterior al 2026-10-24 no manda `X-Device-Id` en las fichas:
    no se sabe a quién le llega, y no se inventa."""
    fp = _fila_analizada()
    main.db.guardar_lo_importado(_fp(), [
        {'fingerprint': fp, 'source': 'rekordbox', 'bpm': 128.0}])
    antes = len(_filas(main.db))
    client.post('/analysis/by-fingerprint/batch', json={'fingerprints': [fp]})
    assert len(_filas(main.db)) == antes


def test_lo_medido_por_el_dsp_no_es_lo_importado(en_render):
    fp = _fila_analizada()
    receptor = _fp()
    client.post('/analysis/by-fingerprint/batch', json={'fingerprints': [fp]},
                headers={'X-Device-Id': receptor})
    assert _filas(main.db, receptor) == []


def test_analyze_lo_apunta_con_lo_que_devuelve():
    """El envoltorio de /analyze pasa el resultado ya mejorado con la
    comunidad: lo que sale hacia el cliente es lo que cuenta."""
    from main import AnalysisResult

    fp = _fila_analizada()
    votante, receptor = _fp(), _fp()
    main.db.guardar_lo_importado(votante, [
        {'fingerprint': fp, 'source': 'virtualdj', 'bpm': 128.0}])
    ficha = AnalysisResult.model_construct(
        fingerprint=fp, bpm=128.0, bpm_source='virtualdj',
        key='Am', key_source='analysis')
    main._apuntar_lo_importado_recibido(receptor, {fp: ficha})
    assert [f['field'] for f in _filas(main.db, receptor)] == ['bpm']


def test_el_envoltorio_de_analyze_lo_llama():
    import inspect

    fuente = inspect.getsource(main.analyze_track)
    assert '_apuntar_lo_importado_recibido' in fuente
    # Después de mejorar con la comunidad: si no, nunca vería lo importado.
    assert (fuente.index('_mejorar_con_la_comunidad')
            < fuente.index('_apuntar_lo_importado_recibido'))


def test_el_reset_borra_tambien_lo_recibido():
    import inspect

    fuente = inspect.getsource(main)
    bloque = fuente[fuente.index('analysis_tables = ('):]
    assert '"imported_recibido"' in bloque[:bloque.index(')')]
