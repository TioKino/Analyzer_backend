"""El 60 % de los dispositivos con biblioteca tiene menos de 10 tracks, y ese
numero era UN contador para tres situaciones que piden arreglos OPUESTOS.

Estos tests atan el reparto: el orden de las comprobaciones, que las causas no
se solapen, y que el agregado nuevo no pueda tumbar el resto de la respuesta.
"""
import os
import sqlite3
import tempfile

import pytest

from routes import admin_panel


@pytest.fixture()
def analysis_db(monkeypatch):
    """Una `analysis.db` de mentira con `events` y `device_first_seen`."""
    ruta = os.path.join(tempfile.mkdtemp(), 'analysis.db')
    conn = sqlite3.connect(ruta)
    conn.execute("CREATE TABLE events (device_id TEXT, event_name TEXT,"
                 " timestamp TEXT, platform TEXT)")
    conn.execute("CREATE TABLE device_first_seen (device_id TEXT,"
                 " first_day TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setenv('ANALYSIS_DB_PATH', ruta)
    return ruta


def _sembrar(ruta, eventos=(), altas=()):
    conn = sqlite3.connect(ruta)
    conn.executemany("INSERT INTO events (device_id, event_name, timestamp,"
                     " platform) VALUES (?,?,?,?)", eventos)
    conn.executemany("INSERT INTO device_first_seen (device_id, first_day)"
                     " VALUES (?,?)", altas)
    conn.commit()
    conn.close()


def _dias(n):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc).date() - timedelta(days=n)).isoformat()


def test_solo_mira_a_los_que_estan_por_debajo_del_umbral(analysis_db):
    r = admin_panel._por_que_se_quedan_cortos({'a': 3, 'b': 9, 'c': 10,
                                               'd': 5000}, umbral=10)
    assert r['devices'] == 2          # a y b
    assert r['de_un_total_de'] == 4   # el denominador no se pierde


def test_un_alta_de_ayer_NO_cuenta_como_estancada(analysis_db):
    # Tener 3 tracks el primer dia es lo normal. Si no se saca primero, cada
    # alta nueva engorda justo el numero que intentamos bajar.
    _sembrar(analysis_db, altas=[('nuevo', _dias(1))])
    r = admin_panel._por_que_se_quedan_cortos({'nuevo': 3}, umbral=10)
    assert r['por_causa']['recien_llegado'] == 1
    assert r['por_causa']['no_volvio'] == 0


def test_el_recien_llegado_gana_aunque_tenga_un_import_a_medias(analysis_db):
    # El orden manda: de un aparato de ayer no se puede decir que se atasco.
    _sembrar(analysis_db,
             eventos=[('nuevo', 'import_started', _dias(1) + 'T10:00', 'ios')],
             altas=[('nuevo', _dias(1))])
    r = admin_panel._por_que_se_quedan_cortos({'nuevo': 2}, umbral=10)
    assert r['por_causa']['recien_llegado'] == 1
    assert r['por_causa']['import_sin_terminar'] == 0


def test_mas_imports_empezados_que_terminados_es_import_sin_terminar(analysis_db):
    _sembrar(analysis_db,
             eventos=[('x', 'import_started', _dias(3) + 'T10:00', 'windows'),
                      ('x', 'import_started', _dias(2) + 'T10:00', 'windows'),
                      ('x', 'import_completed', _dias(2) + 'T10:05', 'windows')],
             altas=[('x', _dias(40))])
    r = admin_panel._por_que_se_quedan_cortos({'x': 4}, umbral=10)
    assert r['por_causa']['import_sin_terminar'] == 1


def test_un_import_que_termino_no_cuenta_como_atascado(analysis_db):
    _sembrar(analysis_db,
             eventos=[('x', 'import_started', _dias(3) + 'T10:00', 'windows'),
                      ('x', 'import_completed', _dias(3) + 'T10:05', 'windows')],
             altas=[('x', _dias(40))])
    r = admin_panel._por_que_se_quedan_cortos({'x': 4}, umbral=10)
    assert r['por_causa']['import_sin_terminar'] == 0
    assert r['por_causa']['activo_sin_traer_mas'] == 1


def test_sin_eventos_recientes_es_no_volvio(analysis_db):
    _sembrar(analysis_db,
             eventos=[('viejo', 'app_opened', _dias(80) + 'T10:00', 'ios')],
             altas=[('viejo', _dias(90))])
    r = admin_panel._por_que_se_quedan_cortos({'viejo': 2}, umbral=10)
    assert r['por_causa']['no_volvio'] == 1


def test_vivo_pero_sin_rastro_de_import_va_a_su_propio_cajon(analysis_db):
    # `events` se purga a los 90 dias: de un veterano activo no se puede saber
    # si su import de hace seis meses termino. Decir «activo y no ha traido
    # mas» seria afirmar mas de lo que el dato sostiene.
    _sembrar(analysis_db,
             eventos=[('v', 'app_opened', _dias(2) + 'T10:00', 'macos')],
             altas=[('v', _dias(200))])
    r = admin_panel._por_que_se_quedan_cortos({'v': 6}, umbral=10)
    assert r['por_causa']['sin_rastro_de_import'] == 1
    assert r['por_causa']['activo_sin_traer_mas'] == 0


def test_las_causas_suman_EXACTAMENTE_los_devices_cortos(analysis_db):
    # Si no suman, hay un aparato contado dos veces o ninguna, y el reparto
    # deja de poder leerse como un reparto.
    _sembrar(
        analysis_db,
        eventos=[('nuevo', 'app_opened', _dias(1) + 'T10:00', 'ios'),
                 ('atasco', 'import_started', _dias(3) + 'T10:00', 'windows'),
                 ('ido', 'app_opened', _dias(70) + 'T10:00', 'android'),
                 ('vivo', 'import_started', _dias(2) + 'T10:00', 'macos'),
                 ('vivo', 'import_completed', _dias(2) + 'T10:09', 'macos'),
                 ('mudo', 'app_opened', _dias(2) + 'T10:00', 'macos')],
        altas=[('nuevo', _dias(1)), ('atasco', _dias(40)),
               ('ido', _dias(80)), ('vivo', _dias(50)), ('mudo', _dias(60))])
    corto = {'nuevo': 1, 'atasco': 2, 'ido': 3, 'vivo': 4, 'mudo': 5}
    r = admin_panel._por_que_se_quedan_cortos(corto, umbral=10)
    assert sum(r['por_causa'].values()) == r['devices'] == 5
    assert r['por_causa'] == {
        'recien_llegado': 1,
        'import_sin_terminar': 1,
        'no_volvio': 1,
        'activo_sin_traer_mas': 1,
        'sin_rastro_de_import': 1,
    }


def test_sin_biblioteca_corta_devuelve_el_esqueleto_y_no_revienta(analysis_db):
    r = admin_panel._por_que_se_quedan_cortos({'grande': 900}, umbral=10)
    assert r['devices'] == 0 and r['por_causa'] == {}


def test_sin_analysis_db_devuelve_el_esqueleto_en_vez_de_lanzar(monkeypatch):
    monkeypatch.setenv('ANALYSIS_DB_PATH', '/no/existe/de/verdad.db')
    r = admin_panel._por_que_se_quedan_cortos({'a': 2}, umbral=10)
    assert r['devices'] == 1 and r['por_causa'] == {}


def test_una_BD_vieja_sin_device_first_seen_no_tumba_el_reparto():
    ruta = os.path.join(tempfile.mkdtemp(), 'vieja.db')
    conn = sqlite3.connect(ruta)
    conn.execute("CREATE TABLE events (device_id TEXT, event_name TEXT,"
                 " timestamp TEXT, platform TEXT)")
    conn.execute("INSERT INTO events VALUES ('x','app_opened',"
                 f"'{_dias(2)}T10:00','ios')")
    conn.commit()
    conn.close()
    os.environ['ANALYSIS_DB_PATH'] = ruta
    try:
        r = admin_panel._por_que_se_quedan_cortos({'x': 3}, umbral=10)
        assert sum(r['por_causa'].values()) == 1
    finally:
        os.environ.pop('ANALYSIS_DB_PATH', None)


def test_el_diagnostico_tiene_su_PROPIO_try_except_en_el_endpoint():
    # Leccion de `#87`: un agregado nuevo que comparte el except de otro amplia
    # el radio de lo que se rompe. Aquel dia una excepcion en el reparto por
    # version devolvio el embudo ENTERO a ceros.
    import io
    src = io.open('routes/admin_panel.py', encoding='utf-8').read()
    i = src.index('stalled = _por_que_se_quedan_cortos(')
    previo = src[:i]
    # El `try:` que lo envuelve tiene que ser SUYO, no compartido con
    # `investment`.
    assert previo.rindex('try:') > previo.rindex('investment = ')


def test_investment_no_rescanea_si_le_pasan_el_conteo(monkeypatch):
    llamadas = []
    monkeypatch.setattr(admin_panel, '_tracks_por_device',
                        lambda: llamadas.append(1) or {})
    r = admin_panel._library_investment_real({'a': 5, 'b': 300})
    assert llamadas == [], 'no debe volver a escanear sync_items'
    assert r['devices_with_library'] == 2 and r['buckets']['gte_200'] == 1
