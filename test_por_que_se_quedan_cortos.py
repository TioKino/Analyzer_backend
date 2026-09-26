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


# ============================================================================
# `ultima_senal` e `idos_por_alta`: distinguir «ya paso» de «esta pasando»
# ============================================================================

def test_la_ultima_senal_cubre_a_TODOS_los_cortos_no_solo_a_los_idos(analysis_db):
    # Es el punto del histograma: quien se fue hace cinco dias NO esta en
    # `no_volvio` (tiene evento dentro de los 30 dias), esta en
    # `activo_sin_traer_mas`. Si el histograma solo mirase a los idos, los
    # abandonos recientes seguirian invisibles.
    _sembrar(
        analysis_db,
        eventos=[('reciente', 'app_opened', _dias(5) + 'T10:00', 'ios'),
                 ('reciente', 'import_started', _dias(5) + 'T10:00', 'ios'),
                 ('reciente', 'import_completed', _dias(5) + 'T10:02', 'ios'),
                 ('ido', 'app_opened', _dias(70) + 'T10:00', 'ios')],
        altas=[('reciente', _dias(60)), ('ido', _dias(80))])
    r = admin_panel._por_que_se_quedan_cortos({'reciente': 2, 'ido': 3}, umbral=10)
    assert sum(r['ultima_senal'].values()) == r['devices'] == 2
    assert r['ultima_senal']['hasta_7d'] == 1      # el «vivo»
    assert r['ultima_senal']['de_61_a_90d'] == 1   # el ido
    # Y ese «vivo» esta en el cajon que dice que no hay nada roto:
    assert r['por_causa']['activo_sin_traer_mas'] == 1


def test_de_los_GRANDES_tambien_se_sabe_si_siguen_vivos(analysis_db):
    # El hueco que quedo apuntado el 2026-09-17: la señal solo se miraba en
    # los cortos, así que no se sabía cuántos de los de biblioteca grande
    # siguen vivos — y todas las proporciones van sobre ese total.
    _sembrar(
        analysis_db,
        eventos=[('grande_vivo', 'app_opened', _dias(3) + 'T10:00', 'macos_dmg'),
                 ('grande_ido', 'app_opened', _dias(45) + 'T10:00', 'windows'),
                 ('corto', 'app_opened', _dias(3) + 'T10:00', 'ios')],
        altas=[('grande_vivo', _dias(200)), ('grande_ido', _dias(200)),
               ('corto', _dias(60))])
    r = admin_panel._por_que_se_quedan_cortos(
        {'grande_vivo': 3000, 'grande_ido': 800, 'grande_mudo': 50, 'corto': 2},
        umbral=10)
    g = r['ultima_senal_grandes']
    assert g['hasta_7d'] == 1 and g['de_31_a_60d'] == 1 and g['sin_eventos'] == 1
    assert sum(g.values()) == 3
    # Y los grandes no se cuelan en el histograma de los cortos.
    assert sum(r['ultima_senal'].values()) == r['devices'] == 1


def test_sin_eventos_cae_en_su_cajon_porque_no_se_puede_fechar(analysis_db):
    # `events` se purga a los 90 dias: sin filas no hay forma de saber si se
    # fue hace cuatro meses o si nunca reporto. Inventar una fecha seria peor
    # que decir «no lo se».
    _sembrar(analysis_db, altas=[('mudo', _dias(300))])
    r = admin_panel._por_que_se_quedan_cortos({'mudo': 1}, umbral=10)
    assert r['ultima_senal']['sin_eventos'] == 1
    assert r['por_causa']['no_volvio'] == 1


def test_los_idos_se_reparten_por_la_antiguedad_de_SU_alta(analysis_db):
    # Una cohorte vieja muriendose es un cementerio heredado; una cohorte de
    # este mes muriendose es una fuga abierta. Urgencias opuestas.
    _sembrar(
        analysis_db,
        eventos=[('nueva', 'app_opened', _dias(35) + 'T10:00', 'ios'),
                 ('media', 'app_opened', _dias(50) + 'T10:00', 'ios'),
                 ('vieja', 'app_opened', _dias(80) + 'T10:00', 'ios')],
        altas=[('nueva', _dias(20)), ('media', _dias(60)),
               ('vieja', _dias(200))])
    r = admin_panel._por_que_se_quedan_cortos(
        {'nueva': 1, 'media': 2, 'vieja': 3}, umbral=10)
    assert r['por_causa']['no_volvio'] == 3
    assert r['idos_por_alta'] == {
        'alta_ultimos_30d': 1,
        'alta_31_a_90d': 1,
        'alta_mas_de_90d': 1,
        'sin_fecha_de_alta': 0,
    }


def test_idos_por_alta_suma_exactamente_los_que_no_volvieron(analysis_db):
    _sembrar(
        analysis_db,
        eventos=[('a', 'app_opened', _dias(40) + 'T10:00', 'ios'),
                 ('b', 'app_opened', _dias(45) + 'T10:00', 'ios'),
                 ('vivo', 'app_opened', _dias(1) + 'T10:00', 'ios'),
                 ('vivo', 'import_started', _dias(1) + 'T10:00', 'ios'),
                 ('vivo', 'import_completed', _dias(1) + 'T10:03', 'ios')],
        altas=[('a', _dias(50)), ('b', _dias(120)), ('vivo', _dias(50))])
    r = admin_panel._por_que_se_quedan_cortos({'a': 1, 'b': 2, 'vivo': 3},
                                              umbral=10)
    assert sum(r['idos_por_alta'].values()) == r['por_causa']['no_volvio'] == 2


def test_un_ido_sin_fila_de_alta_no_se_pierde_ni_se_inventa(analysis_db):
    _sembrar(analysis_db,
             eventos=[('huerfano', 'app_opened', _dias(60) + 'T10:00', 'ios')])
    r = admin_panel._por_que_se_quedan_cortos({'huerfano': 2}, umbral=10)
    assert r['idos_por_alta']['sin_fecha_de_alta'] == 1
    assert sum(r['idos_por_alta'].values()) == r['por_causa']['no_volvio'] == 1


def test_el_recien_llegado_sale_del_reparto_de_causas_pero_NO_del_histograma(analysis_db):
    # El histograma describe a los 217 enteros; las causas descuentan al que
    # todavia no se puede llamar estancado. Que los dos denominadores sean
    # distintos es deliberado, y por eso los dos tienen que cuadrar con
    # `devices` por su cuenta.
    _sembrar(analysis_db,
             eventos=[('nuevo', 'app_opened', _dias(1) + 'T10:00', 'ios')],
             altas=[('nuevo', _dias(2))])
    r = admin_panel._por_que_se_quedan_cortos({'nuevo': 1}, umbral=10)
    assert r['por_causa']['recien_llegado'] == 1
    assert sum(r['por_causa'].values()) == 1
    assert r['ultima_senal']['hasta_7d'] == 1
    assert sum(r['ultima_senal'].values()) == 1
