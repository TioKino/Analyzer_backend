"""El cache-hit por huella era un callejon sin salida para el legado.

`acoustic_gap_breakdown` dice que las filas sin chromaprint son legado y que
«las tres las cierra el backfill del cliente». Cierto para dos de las tres
vias, pero la segunda —cache-hit en el pre-check por huella— tenia un detalle
que la hacia incurable por si misma:

Cuando /analyze da cache-hit, el fichero YA se ha subido y esta en `tmp_path`.
La fila existente se re-guarda (para refrescar el `filename`) y el temporal se
borra al retornar. O sea que el audio de un track legado pasaba por delante y
se tiraba sin sacarle la huella — una y otra vez, cada vez que alguien subia
ese mismo fichero. Esa fila no se curaba nunca por esta via.

Y habia un segundo efecto, peor porque tapaba el primero: `save_track` es un
INSERT OR REPLACE que escribia `datetime.now()` en `analyzed_at` en CADA
escritura. El cache-hit no analiza nada, pero bombeaba la fecha igual, asi que
un track de 2024 sin huella figuraba como analizado HOY. `newest_without` es
justo el numero que decide si el hueco es legado (ampliar backfill) o una via
abierta (taparla) — y con las fechas bombeadas las dos causas, que piden
arreglos opuestos, eran indistinguibles.

    pytest test_cache_hit_cura_la_huella.py -v
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _src(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


# ============================================================================
# LA FECHA: analyzed_at es la del ANALISIS, no la de la ultima escritura
# ============================================================================

def _tabla(tmp_path):
    conn = sqlite3.connect(str(tmp_path / 't.db'))
    conn.row_factory = sqlite3.Row
    conn.execute(
        'CREATE TABLE tracks (id TEXT PRIMARY KEY, filename TEXT, '
        'analyzed_at TEXT, chromaprint TEXT)'
    )
    return conn


def _guardar(conn, track_data, ahora):
    """La regla que aplica `save_track` a `analyzed_at`."""
    conn.execute(
        'INSERT OR REPLACE INTO tracks (id, filename, analyzed_at, chromaprint) '
        'VALUES (?,?,?,?)',
        (
            track_data['id'],
            track_data.get('filename'),
            track_data.get('analyzed_at') or ahora,
            track_data.get('chromaprint'),
        ),
    )
    conn.commit()


def test_re_guardar_una_fila_NO_bombea_su_fecha(tmp_path):
    """El caso exacto del cache-hit: solo cambia el filename."""
    conn = _tabla(tmp_path)
    try:
        _guardar(conn, {'id': 't1', 'filename': 'viejo.mp3'}, '2024-01-15T10:00:00')

        fila = dict(conn.execute('SELECT * FROM tracks').fetchone())
        fila['filename'] = 'renombrado.mp3'
        _guardar(conn, fila, '2026-08-27T12:00:00')

        r = conn.execute('SELECT * FROM tracks').fetchone()
    finally:
        conn.close()

    assert r['filename'] == 'renombrado.mp3'
    assert r['analyzed_at'] == '2024-01-15T10:00:00', (
        'la fila sigue siendo de 2024; solo se le cambio el nombre'
    )


def test_un_analisis_NUEVO_si_se_fecha_ahora(tmp_path):
    """La otra mitad. Sin esto el arreglo dejaria todo sin fechar.

    Un analisis recien hecho no trae `analyzed_at` —no es un campo de
    AnalysisResult— asi que cae a `now` como siempre.
    """
    conn = _tabla(tmp_path)
    try:
        _guardar(conn, {'id': 't2', 'filename': 'nuevo.mp3'}, '2026-08-27T12:00:00')
        r = conn.execute('SELECT * FROM tracks WHERE id = "t2"').fetchone()
    finally:
        conn.close()

    assert r['analyzed_at'] == '2026-08-27T12:00:00'


def test_el_reparto_por_antiguedad_deja_de_mentir(tmp_path):
    """El motivo por el que esto importa, en un test.

    Con la fecha bombeada, una fila de 2024 sin huella contaba como de los
    ultimos 7 dias, y `newest_without` decia «via abierta» sobre puro legado.
    """
    conn = _tabla(tmp_path)
    try:
        _guardar(conn, {'id': 'legado', 'filename': 'a.mp3'}, '2024-01-15T10:00:00')
        fila = dict(conn.execute('SELECT * FROM tracks').fetchone())
        fila['filename'] = 'b.mp3'
        _guardar(conn, fila, '2026-08-27T12:00:00')

        newest = conn.execute(
            'SELECT MAX(analyzed_at) FROM tracks WHERE chromaprint IS NULL'
        ).fetchone()[0]
    finally:
        conn.close()

    assert newest.startswith('2024'), 'el hueco es legado, y ahora se ve'


def test_la_regla_esta_en_save_track():
    src = _src('database.py')
    i = src.index('INSERT OR REPLACE INTO tracks')
    fn = src[i:i + 3000]
    assert "track_data.get('analyzed_at') or datetime.now().isoformat()" in fn
    assert fn.count('datetime.now().isoformat()') == 1, (
        'si vuelve a haber un now() suelto, la fecha se bombea otra vez'
    )


# ============================================================================
# LA HUELLA: si el audio esta delante, se le saca
# ============================================================================

def test_el_cache_hit_saca_la_huella_que_falta():
    """El audio esta en `tmp_path` y se borra al retornar. Es la unica ocasion
    en que el fichero de un track legado vuelve a estar a mano."""
    src = _src('main.py')
    i = src.index('existing_by_fp[\'filename\'] = file.filename')
    tramo = src[i:i + 1400]
    assert "if not existing_by_fp.get('chromaprint'):" in tramo
    assert '_attach_acoustic(existing_by_fp, tmp_path)' in tramo
    # Y antes de guardar, no despues.
    assert tramo.index('_attach_acoustic') < tramo.index('db.save_track(')


def test_solo_se_calcula_si_FALTA():
    """fpcalc son ~2 s de CPU por track. Recalcular la huella de una fila que
    ya la tiene es pagarlos en cada cache-hit para escribir lo mismo."""
    src = _src('main.py')
    i = src.index("if not existing_by_fp.get('chromaprint'):")
    assert '_attach_acoustic' in src[i:i + 200]


def test_el_fallback_de_render_tampoco_nace_sin_huella():
    """Render manda su analisis pero no el chromaprint (es un blob que no
    viaja en la respuesta). El audio sigue en `tmp_path`."""
    src = _src('main.py')
    i = src.index("to_save['analysis_json'] = json.dumps(render_cached)")
    tramo = src[i:i + 800]
    assert "if not to_save.get('chromaprint'):" in tramo
    assert '_attach_acoustic(to_save, tmp_path)' in tramo


def test_sigue_siendo_best_effort():
    """Si fpcalc no esta o falla, /analyze tiene que seguir respondiendo: la
    huella es una mejora, no un requisito para devolver el analisis."""
    src = _src('main.py')
    i = src.index('def _attach_acoustic(')
    fn = src[i:i + 1500]
    assert 'except Exception as e:' in fn
    assert 'return' in fn  # el `if not raw: return` temprano
