"""La ficha de Escuchar no recorre la tabla `tracks` entera.

Log de Render del 2026-09-29 (iPhone del owner), primera pulsación de la
tarde: `ficha=35516ms` de 38 s de servidor, con AudD en 1,3 s. En caliente,
~0,75-1 s de ficha en cada acierto, más que AudD. `buscar_analizado` buscaba
con `LOWER(artist)`, que ningún índice sirve: las tres consultas recorrían la
tabla con su `analysis_json` dentro.

    pytest test_la_ficha_no_recorre_la_tabla.py -v
"""

import uuid

import main
from routes.search import FICHA_CANDIDATOS, FICHA_EXACTA, buscar_analizado


def _plan(sql, args):
    conn = main.db._open_conn()
    try:
        return ' | '.join(
            str(f[3]) for f in conn.execute('EXPLAIN QUERY PLAN ' + sql, args))
    finally:
        conn.close()


def _guardar(artist, title, bpm=128.0, analyzed_at='2025-06-01T00:00:00'):
    fp = uuid.uuid4().hex
    main.db.save_track({
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3',
        'artist': artist, 'title': title, 'bpm': bpm, 'duration': 300,
        'key': 'Am', 'camelot': '8A', 'energy_dj': 7, 'genre': 'Techno',
        'track_type': 'peak_time', 'analyzed_at': analyzed_at,
    })
    return fp


def test_el_indice_existe():
    conn = main.db._open_conn()
    try:
        filas = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'idx_tracks_ficha'"
        ).fetchall()
    finally:
        conn.close()
    assert filas, 'falta el índice de la ficha'
    assert 'NOCASE' in filas[0][0].upper()


def test_las_consultas_van_por_el_indice():
    # Con el esquema de verdad (hay índices de `analyzed_at` y de `bpm`): sin
    # forzarlo, la de los LIKE usaba uno de esos, que leen la tabla fila a
    # fila.
    for sql, a in ((FICHA_EXACTA, ('a', 'b')),
                   (FICHA_CANDIDATOS, ('%a%', '%b%'))):
        plan = _plan(sql, a)
        assert 'COVERING INDEX idx_tracks_ficha' in plan, (sql, plan)
        assert 'idx_tracks_analyzed_at' not in plan, (sql, plan)
        assert 'idx_bpm' not in plan, (sql, plan)
        # De la tabla solo se lee la fila elegida, por su rowid.
        assert 'SCAN tracks |' not in plan + ' |', (sql, plan)
        assert 'INDEXED BY idx_tracks_ficha' in sql


def test_sin_mayusculas_como_antes():
    sufijo = uuid.uuid4().hex[:6]
    artista, titulo = f'ADAM Beyer {sufijo}', f'Your MIND {sufijo}'
    fp = _guardar(artista, f'{titulo} (Extended Mix)')
    # Exacto: en minúsculas casa igual que con LOWER().
    exacto = _guardar(artista, titulo)
    assert buscar_analizado(artista.lower(), titulo.lower())['id'] == exacto
    # Artista exacto, título con «(Original Mix)»: es el original.
    otro = f'Otro Tema {sufijo}'
    fp2 = _guardar(artista, f'{otro} (Original Mix)')
    assert buscar_analizado(artista.lower(), otro.upper())['id'] == fp2
    # Los dos por aproximación (el artista dentro de otro, el tema dentro de
    # otro) y «(Extended Mix)», que también es el original.
    assert buscar_analizado(f'beyer {sufijo}', f'mind {sufijo} (extended mix)')[
        'id'] in (fp, exacto)


def test_una_deteccion_sin_bpm_sigue_sin_ser_ficha():
    sufijo = uuid.uuid4().hex[:6]
    _guardar(f'Solo Deteccion {sufijo}', f'Tema {sufijo}', bpm=0)
    assert buscar_analizado(f'solo deteccion {sufijo}', f'tema {sufijo}') is None


def test_gana_el_analisis_mas_reciente():
    sufijo = uuid.uuid4().hex[:6]
    _guardar(f'Artista {sufijo}', f'Tema {sufijo}', analyzed_at='2024-01-01')
    nuevo = _guardar(f'Artista {sufijo}', f'Tema {sufijo}',
                     analyzed_at='2026-01-01')
    assert buscar_analizado(f'artista {sufijo}', f'tema {sufijo}')['id'] == nuevo
