"""Las bibliotecas del panel: sin los temas BORRADOS y, para el paywall, por
CUENTA (2026-11-25, auditoría del embudo).

  1. Ningún recuento de `sync_items` del panel quitaba las filas borradas
     (`deleted = 1`: un borrado es un UPDATE que deja la fila, para que viaje a
     los demás aparatos). Las bibliotecas, la mediana, los ≥100/500/1000, la
     pestaña Usuarios y la cobertura las contaban como temas. Solo la
     biblioteca de un aparato en el Catálogo (`_biblioteca_de`) las quitaba,
     así que las dos pantallas daban números distintos del mismo aparato.
  2. El coste de cambio se contaba por APARATO (el último que escribió cada
     fila): un DJ con el Mac y el PC partía su biblioteca en dos aparatos de
     media biblioteca. Para el paywall la unidad es quien pagaría: la cuenta.

    pytest test_las_bibliotecas_sin_borrados_y_por_cuenta.py -v
"""

import json
import os
import re
import sqlite3
import tempfile

import pytest

import routes.admin_panel as ap


@pytest.fixture
def sync(monkeypatch):
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        'CREATE TABLE sync_items (key TEXT PRIMARY KEY, data_type TEXT NOT NULL,'
        ' item_key TEXT NOT NULL, payload TEXT NOT NULL, deleted INTEGER DEFAULT 0,'
        ' updated_at TEXT NOT NULL, last_device_id TEXT NOT NULL,'
        " device_type TEXT DEFAULT 'unknown', hash TEXT NOT NULL, user_id TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(ap, '_SYNC_DB_PATH', path)
    # Las cachés de los recuentos son por proceso: cada test empieza limpio.
    monkeypatch.setattr(ap, '_por_device_guardado',
                        {'clave': None, 'cuando': 0.0, 'valor': None})
    monkeypatch.setattr(ap, '_por_cuenta_guardado',
                        {'clave': None, 'cuando': 0.0, 'valor': None})
    yield path
    os.unlink(path)


def _fila(path, cuenta, tema, aparato, borrado=0, payload=None):
    conn = sqlite3.connect(path)
    conn.execute(
        'INSERT INTO sync_items VALUES (?,?,?,?,?,?,?,?,?,?)',
        (f'{cuenta}|analysis|{tema}', 'analysis', tema,
         json.dumps(payload if payload is not None else {'id': tema}),
         borrado, '2026-11-20', aparato, 'x', 'h', cuenta))
    conn.commit()
    conn.close()


def test_un_tema_borrado_no_es_biblioteca(sync):
    _fila(sync, 'u1', 't1', 'pc')
    _fila(sync, 'u1', 't2', 'pc', borrado=1, payload={})
    _fila(sync, 'u1', 't3', 'pc', borrado=1)
    assert ap._tracks_por_device() == {'pc': 1}
    assert ap._tracks_por_cuenta() == {'u1': {'temas': 1, 'sin_audio': 0}}


def test_un_aparato_con_todo_borrado_no_tiene_biblioteca(sync):
    _fila(sync, 'u1', 't1', 'pc', borrado=1)
    assert ap._tracks_por_device() == {}
    assert ap._tracks_por_cuenta() == {}


def test_borrar_cambia_el_recuento_aunque_no_cambie_el_numero_de_filas(sync):
    """Un borrado es un UPDATE: ni el número de filas ni la última cambian, y
    la caché se quedaba con el recuento de antes."""
    _fila(sync, 'u1', 't1', 'pc')
    _fila(sync, 'u1', 't2', 'pc')
    assert ap._tracks_por_device() == {'pc': 2}
    conn = sqlite3.connect(sync)
    conn.execute("UPDATE sync_items SET deleted = 1 WHERE item_key = 't2'")
    conn.commit()
    conn.close()
    assert ap._tracks_por_device() == {'pc': 1}


def test_por_cuenta_junta_los_aparatos_de_un_dj(sync):
    """El Mac y el PC del mismo DJ: por aparato salían dos bibliotecas de la
    mitad; por cuenta, una entera."""
    for t in ('t1', 't2', 't3'):
        _fila(sync, 'dj', t, 'mac')
    for t in ('t4', 't5'):
        _fila(sync, 'dj', t, 'pc')
    assert ap._tracks_por_device() == {'mac': 3, 'pc': 2}
    assert ap._tracks_por_cuenta() == {'dj': {'temas': 5, 'sin_audio': 0}}


def test_las_fichas_sin_audio_se_cuentan_aparte(sync):
    _fila(sync, 'u1', 't1', 'pc')
    _fila(sync, 'u1', 'imp_abc', 'pc')
    _fila(sync, 'u1', 'imp_def', 'pc')
    assert ap._tracks_por_cuenta() == {'u1': {'temas': 3, 'sin_audio': 2}}
    inv = ap._inversion_por_cuenta()
    assert inv['unidad'] == 'cuenta'
    assert inv['cuentas_con_biblioteca'] == 1
    assert inv['max_tracks'] == 3
    assert inv['con_audio']['max_tracks'] == 1


def test_el_blob_de_los_clientes_viejos_se_lee_y_se_junta(sync):
    conn = sqlite3.connect(sync)
    conn.execute(
        'INSERT INTO sync_items VALUES (?,?,?,?,?,?,?,?,?,?)',
        ('u1|analysis|all_analysis', 'analysis', 'all_analysis',
         json.dumps({'tracks': {'t1': {'id': 't1'}, 't9': {'id': 't9'}}}),
         0, '2026-11-20', 'pc', 'x', 'h', 'u1'))
    conn.commit()
    conn.close()
    _fila(sync, 'u1', 't1', 'pc')   # el mismo tema por los dos formatos
    assert ap._tracks_por_cuenta() == {'u1': {'temas': 2, 'sin_audio': 0}}


def test_sin_cuenta_cada_aparato_es_la_suya_y_lo_colectivo_no_cuenta(sync):
    _fila(sync, '', 't1', 'viejo')
    _fila(sync, '__collective__', 't2', 'pc')
    assert ap._tracks_por_cuenta() == {'aparato:viejo': {'temas': 1, 'sin_audio': 0}}


def test_ninguna_consulta_de_temas_del_panel_cuenta_los_borrados():
    """Recorre el fuente: toda consulta del panel que lee las filas de
    análisis (`payload` o `item_key`) lleva `deleted = 0`. La que no lo lleve
    vuelve a contar los temas borrados sin un solo error."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'routes', 'admin_panel.py'), encoding='utf-8').read()
    lineas = src.splitlines()
    malas = []
    for i, linea in enumerate(lineas):
        if 'FROM sync_items' not in linea or 'SELECT' not in ''.join(lineas[i - 1:i + 1]):
            continue
        trozo = ' '.join(lineas[max(0, i - 2):i + 3])
        lee_temas = re.search(r"data_type = 'analysis'|\+data_type = \?", trozo)
        lee_filas = 'payload' in trozo or 'item_key' in trozo
        if lee_temas and lee_filas and 'deleted' not in trozo:
            malas.append(f'{i + 1}: {linea.strip()}')
    assert not malas, 'consultas que cuentan temas borrados:\n' + '\n'.join(malas)
