"""Lo que el panel cuenta de UN aparato entra por su índice (2026-10-29).

Las consultas de un aparato en `sync_items` llevan `data_type = X AND
last_device_id = ?`. Con las dos columnas en el WHERE, SQLite elegía
`idx_sync_items_type`, y `data_type = 'analysis'` son casi todas las filas:
cada aparato recorría el análisis de TODOS, con su payload. `_tracks_por_device`
y `/admin/users` lo hacen aparato a aparato (unos 1.800), y la retención llegó a
tardar 4.957 s en Render. El `+data_type` deja ese índice fuera y la consulta
entra por `idx_sync_items_device`.
"""
import os
import re
import sqlite3

import sync_endpoints

_FUENTE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'routes', 'admin_panel.py')


def _bd_con_el_esquema_de_verdad():
    conn = sqlite3.connect(':memory:')
    sync_endpoints._init_tables(conn)
    filas = []
    for i in range(3000):
        filas.append((f'k{i}', 'analysis' if i % 10 else 'all_cues', f't{i}',
                      '{}', 'ahora', f'aparato{i % 300}', 'h'))
    conn.executemany(
        "INSERT INTO sync_items (key, data_type, item_key, payload, updated_at, "
        "last_device_id, hash) VALUES (?, ?, ?, ?, ?, ?, ?)", filas)
    return conn


def _plan(conn, sql, params):
    return ' | '.join(r[-1] for r in conn.execute('EXPLAIN QUERY PLAN ' + sql, params))


def test_un_aparato_entra_por_su_indice():
    conn = _bd_con_el_esquema_de_verdad()
    plan = _plan(conn, "SELECT item_key, payload FROM sync_items "
                       "WHERE +data_type = 'analysis' AND last_device_id = ?",
                 ('aparato1',))
    assert 'idx_sync_items_device' in plan, plan


def test_sin_el_mas_entra_por_el_tipo():
    """Por qué hace falta el `+`: sin él, el plan va por el índice del tipo.
    Si una versión de SQLite deja de hacerlo, este test lo dirá, y el `+`
    seguirá sin hacer daño."""
    conn = _bd_con_el_esquema_de_verdad()
    plan = _plan(conn, "SELECT item_key, payload FROM sync_items "
                       "WHERE data_type = 'analysis' AND last_device_id = ?",
                 ('aparato1',))
    assert 'idx_sync_items_type' in plan, plan


def test_ninguna_consulta_de_un_aparato_filtra_por_el_tipo_a_secas():
    with open(_FUENTE, encoding='utf-8') as f:
        src = f.read()
    # Une las cadenas partidas en varias líneas («"…" "…"») para ver cada
    # consulta entera.
    unida = re.sub(r'"\s*\n\s*"', '', src)
    malas = [m.group(0) for m in re.finditer(
        r'[^+]data_type\s*=\s*[^"]*?last_device_id\s*=\s*\?', unida)]
    assert not malas, malas
