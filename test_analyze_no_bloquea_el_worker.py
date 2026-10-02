"""El camino de /analyze no bloquea el único worker de Render, ni recorre la
tabla `tracks` entera (auditoría del flujo de análisis, 2026-10-02).

Render corre UN proceso de uvicorn (`Procfile`). Hasta ese día solo el DSP iba
al threadpool: el MD5 del fichero, `fpcalc` (`_attach_acoustic`), el preview
con ffmpeg y la consulta a Render del motor local corrían en el event loop,
segundos por análisis en los que no se atendía a nadie más — ni el sync, ni
Escuchar. Y dos consultas de cada análisis recorrían las ~122.000 filas:
`tracks.filename` y `tracks.duration` no tenían índice.

Y por NOMBRE de fichero ya no se contesta: el móvil se quedaba con el análisis
de cualquiera cuyo fichero se llamara igual (SEC-01 por otra puerta).

    pytest test_analyze_no_bloquea_el_worker.py -v
"""

import ast
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture()
def client():
    return TestClient(main.app)

# Lo que lee el fichero, lanza un subproceso o sale a la red.
_BLOQUEANTES = {
    'calculate_fingerprint', '_attach_acoustic', 'generate_preview_snippet',
    '_fetch_render_cache', '_mejorar_con_la_comunidad',
    'compute_raw_chromaprint', 'extract_id3_metadata',
}


def _llamadas_directas(fn):
    """Nombres de las funciones que `fn` llama SIN pasar por una función
    anidada (las anidadas son las que se mandan al threadpool)."""
    vistas = []

    def visitar(nodo):
        for hijo in ast.iter_child_nodes(nodo):
            if isinstance(hijo, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.Lambda)):
                continue
            if isinstance(hijo, ast.Call) and isinstance(hijo.func, ast.Name):
                vistas.append(hijo.func.id)
            visitar(hijo)

    visitar(fn)
    return vistas


def _funcion(nombre):
    arbol = ast.parse(Path('main.py').read_text(encoding='utf-8'))
    for nodo in ast.walk(arbol):
        if (isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef))
                and nodo.name == nombre):
            return nodo
    raise AssertionError(f'no está {nombre}')


def test_analizar_no_llama_en_el_event_loop_a_nada_que_bloquee():
    directas = set(_llamadas_directas(_funcion('_analizar')))
    culpables = directas & _BLOQUEANTES
    assert not culpables, (
        f'{sorted(culpables)} corre en el event loop del único worker: '
        f'va por run_in_threadpool'
    )
    # Y no porque se hayan dejado de hacer: siguen ahí, por el threadpool.
    src = Path('main.py').read_text(encoding='utf-8')
    cuerpo = src[src.index('async def _analizar('):
                 src.index('@app.post("/correction")')]
    for nombre in ('calculate_fingerprint', 'generate_preview_snippet',
                   '_fetch_render_cache', '_mejorar_con_la_comunidad'):
        assert nombre in cuerpo, nombre
    assert cuerpo.count('_attach_acoustic(') == 4, (
        'las cuatro curas/huellas de siempre: atajo por nombre, por huella, '
        'fallback a Render y análisis nuevo'
    )


def test_el_envoltorio_tampoco():
    directas = set(_llamadas_directas(_funcion('analyze_track')))
    assert not (directas & _BLOQUEANTES), directas & _BLOQUEANTES


# ============================================================================
# LOS ÍNDICES
# ============================================================================

def _plan(sql, args):
    conn = main.db._open_conn()
    try:
        return ' | '.join(
            str(f[3]) for f in conn.execute('EXPLAIN QUERY PLAN ' + sql, args))
    finally:
        conn.close()


def test_buscar_por_nombre_va_por_indice():
    plan = _plan('SELECT * FROM tracks WHERE filename = ?', ('x.mp3',))
    assert 'idx_tracks_filename' in plan, plan


def test_el_cluster_acustico_no_recorre_la_tabla(monkeypatch):
    """La consulta de verdad de `find_acoustic_cluster`, capturada al vuelo."""
    vistas = []
    abrir = main.db._open_conn

    def con_traza():
        conn = abrir()
        conn.set_trace_callback(vistas.append)
        return conn

    monkeypatch.setattr(main.db, '_open_conn', con_traza)
    main.db.find_acoustic_cluster([1, 2, 3] * 40, 300.0)
    sql = next(s for s in vistas if 'chromaprint, acoustic_id' in s)
    # La traza trae los parámetros ya puestos.
    plan = _plan(sql, ())
    assert 'idx_tracks_duration' in plan, plan
    assert 'SCAN tracks |' not in plan + ' |', plan


def test_el_cluster_sigue_casando_en_el_borde_de_la_ventana():
    """`BETWEEN` es el mismo intervalo cerrado que `ABS(...) <= 2.5`."""
    from acoustic_fingerprint import encode_raw
    huella = [((i * 2654435761) & 0xFFFFFFFF) for i in range(300)]
    cluster = f'cl-{uuid.uuid4().hex[:8]}'
    fp = uuid.uuid4().hex
    main.db.save_track({
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3', 'bpm': 128.0,
        'duration': 302.5, 'energy_dj': 7, 'genre': 'Techno',
        'track_type': 'peak_time', 'chromaprint': encode_raw(huella),
        'acoustic_id': cluster,
    })
    assert main.db.find_acoustic_cluster(huella, 300.0) == cluster
    assert main.db.find_acoustic_cluster(huella, 305.0) == cluster
    assert main.db.find_acoustic_cluster(huella, 296.0) != cluster


# ============================================================================
# POR NOMBRE NO
# ============================================================================

def test_check_analyzed_por_nombre_no_dice_que_si(client):
    """Un móvil de antes pregunta por nombre: todo «no analizado», y así sube
    el fichero y /analyze lo resuelve por su huella."""
    fp = uuid.uuid4().hex
    main.db.save_track({
        'id': fp, 'fingerprint': fp, 'filename': '01 - Intro.mp3',
        'bpm': 128.0, 'duration': 300, 'energy_dj': 7, 'genre': 'Techno',
        'track_type': 'peak_time',
    })
    r = client.post('/check-analyzed', json=['01 - Intro.mp3', 'otro.mp3'])
    assert r.status_code == 200
    body = r.json()
    assert body['analyzed'] == []
    assert body['not_analyzed'] == ['01 - Intro.mp3', 'otro.mp3']


def test_el_analisis_por_nombre_ya_no_se_da(client):
    fp = uuid.uuid4().hex
    main.db.save_track({
        'id': fp, 'fingerprint': fp, 'filename': 'track 3.mp3',
        'bpm': 128.0, 'duration': 300, 'energy_dj': 7, 'genre': 'Techno',
        'track_type': 'peak_time',
    })
    r = client.get('/analysis/track%203.mp3')
    assert r.status_code == 410
    # Por huella sí.
    assert client.get(f'/analysis/by-fingerprint/{fp}').status_code == 200


def test_check_analyzed_ya_no_toca_la_bd():
    src = Path('routes/analysis_artwork.py').read_text(encoding='utf-8')
    i = src.index('async def check_analyzed(filenames')
    cuerpo = src[i:src.index('\n\n\n', i)]
    assert 'db.' not in cuerpo
