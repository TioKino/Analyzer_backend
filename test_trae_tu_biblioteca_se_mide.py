"""«Trae tu biblioteca entera» se mide: el aviso a los que se quedan cortos.

El banner de escritorio salio en la 2.10.0 mandando `bring_library_shown`,
`_cta` y `_dismissed`, y el panel no los leia: una release entera con un aviso
del que no se podia saber si servia. Estos tests atan la cuenta por plataforma,
las puertas, el «crecieron» (el numero que dice si sirve) y que en movil el
exito se mira por vincular, que no sube los temas de ese aparato.
"""
import json
import os
import sqlite3
import tempfile

import pytest

from routes import admin_panel


@pytest.fixture()
def analysis_db(monkeypatch):
    ruta = os.path.join(tempfile.mkdtemp(), 'analysis.db')
    conn = sqlite3.connect(ruta)
    conn.execute("CREATE TABLE events (device_id TEXT, event_name TEXT,"
                 " timestamp TEXT, platform TEXT, props TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setenv('ANALYSIS_DB_PATH', ruta)
    return ruta


def _sembrar(ruta, eventos):
    conn = sqlite3.connect(ruta)
    conn.executemany(
        "INSERT INTO events (device_id, event_name, timestamp, platform, props)"
        " VALUES (?,?,'2026-09-28T10:00',?,?)",
        [(d, n, p, json.dumps(pr) if pr else None) for d, n, p, pr in eventos])
    conn.commit()
    conn.close()


def test_sin_eventos_no_hay_plataformas(analysis_db):
    r = admin_panel._trae_tu_biblioteca({'a': 3})
    assert r['por_plataforma'] == {}


def test_cuenta_aparatos_no_eventos(analysis_db):
    # Verlo en tres arranques es UN aparato que lo vio.
    _sembrar(analysis_db, [
        ('pc', 'bring_library_shown', 'macos-dmg', None),
        ('pc', 'bring_library_shown', 'macos-dmg', None),
        ('pc', 'bring_library_shown', 'macos-dmg', None),
    ])
    d = admin_panel._trae_tu_biblioteca({'pc': 3})['por_plataforma']['desktop']
    assert d['vieron'] == 1
    assert d['pulsaron'] == 0


def test_escritorio_y_movil_van_aparte(analysis_db):
    _sembrar(analysis_db, [
        ('pc', 'bring_library_shown', 'windows', None),
        ('tel', 'bring_library_shown', 'ios', None),
        ('tel2', 'bring_library_shown', 'android', None),
    ])
    p = admin_panel._trae_tu_biblioteca({})['por_plataforma']
    assert p['desktop']['vieron'] == 1
    assert p['mobile']['vieron'] == 2


def test_las_puertas_se_separan_y_la_de_antes_es_sin_dato(analysis_db):
    _sembrar(analysis_db, [
        ('a', 'bring_library_cta', 'windows', {'puerta': 'carpeta'}),
        ('b', 'bring_library_cta', 'windows', {'puerta': 'programa'}),
        # 2.10.0-2.10.1: solo habia carpeta y el evento no llevaba puerta.
        ('c', 'bring_library_cta', 'macos', None),
        ('t', 'bring_library_cta', 'ios', {'puerta': 'vincular'}),
    ])
    p = admin_panel._trae_tu_biblioteca({})['por_plataforma']
    assert p['desktop']['pulsaron'] == 3
    assert p['desktop']['por_puerta'] == {
        'carpeta': 1, 'programa': 1, 'sin_dato': 1}
    assert p['mobile']['por_puerta'] == {'vincular': 1}


def test_crecieron_es_tener_hoy_el_umbral(analysis_db):
    # El aviso solo sale por debajo de 10: tener hoy 10 o mas es haberla
    # traido. Es el numero que dice si el aviso sirve.
    _sembrar(analysis_db, [
        ('trajo', 'bring_library_shown', 'windows', None),
        ('trajo', 'bring_library_cta', 'windows', {'puerta': 'carpeta'}),
        ('sigue', 'bring_library_shown', 'windows', None),
        ('sigue', 'bring_library_cta', 'windows', {'puerta': 'carpeta'}),
        ('solo_vio', 'bring_library_shown', 'windows', None),
    ])
    d = admin_panel._trae_tu_biblioteca(
        {'trajo': 2400, 'sigue': 4, 'solo_vio': 12}, umbral=10,
    )['por_plataforma']['desktop']
    assert d['vieron'] == 3
    assert d['crecieron'] == 2            # trajo + solo_vio
    assert d['crecieron_tras_pulsar'] == 1  # trajo


def test_en_movil_el_exito_es_vincular(analysis_db):
    # Vincular no sube los temas de ESTE aparato en sync.db (los cuenta el que
    # los subio): sin esta cuenta el aviso del movil pareceria no servir nunca.
    _sembrar(analysis_db, [
        ('tel', 'bring_library_shown', 'android', None),
        ('tel', 'bring_library_cta', 'android', {'puerta': 'vincular'}),
        ('tel', 'device_linked', 'android', None),
        ('tel2', 'bring_library_cta', 'ios', {'puerta': 'vincular'}),
    ])
    m = admin_panel._trae_tu_biblioteca({'tel': 3})['por_plataforma']['mobile']
    assert m['pulsaron'] == 2
    assert m['vincularon_tras_pulsar'] == 1
    assert m['crecieron_tras_pulsar'] == 0


def test_descartaron(analysis_db):
    _sembrar(analysis_db, [
        ('pc', 'bring_library_shown', 'linux', None),
        ('pc', 'bring_library_dismissed', 'linux', None),
    ])
    d = admin_panel._trae_tu_biblioteca({})['por_plataforma']['desktop']
    assert d['descartaron'] == 1


def test_sin_analysis_db_no_revienta(monkeypatch):
    monkeypatch.setenv('ANALYSIS_DB_PATH', '/no/existe/analysis.db')
    r = admin_panel._trae_tu_biblioteca({'a': 3})
    assert r['por_plataforma'] == {}


def test_la_retencion_lo_devuelve_con_su_propio_except():
    # Si este calculo falla, la clave va a None y el resto sigue: la leccion
    # de `#87`, un agregado nuevo no puede tumbar el embudo entero.
    src = open(admin_panel.__file__, encoding='utf-8').read()
    i = src.index('trae = _trae_tu_biblioteca(_por_device)')
    tramo = src[i - 200:i + 300]
    assert 'try:' in tramo and 'except Exception' in tramo
    assert '"trae_tu_biblioteca": trae,' in src
