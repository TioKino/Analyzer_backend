"""Duplicados por SONIDO: el cluster de un lote de huellas.

El caso que resuelve: el mismo tema en dos ficheros distintos —otro codec,
otro bitrate, otro tag, rippeado de otro sitio—. El MD5 del contenido no los
junta (son bytes distintos) y el nombre tampoco. El chromaprint si.

Lo que YA estaba resuelto y no hacia falta tocar: los duplicados byte a byte.
El cliente los colapsa solo en `addAnalysis` (`_dropSuperseded`), en silencio.

Lo que faltaba: el cliente no guarda `acoustic_id` ni tenia forma de pedirlo en
lote, asi que el agrupado por sonido —que el backend lleva haciendo desde
siempre— no llegaba a la biblioteca del usuario.

    pytest test_acoustic_clusters_lote.py -v
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _bd(tmp_path, filas):
    """`tracks` con lo que mira `acoustic_ids_for`."""
    conn = sqlite3.connect(str(tmp_path / 'a.db'))
    conn.row_factory = sqlite3.Row
    conn.execute(
        'CREATE TABLE tracks (id TEXT PRIMARY KEY, fingerprint TEXT, '
        'acoustic_id TEXT)'
    )
    conn.executemany('INSERT INTO tracks VALUES (?,?,?)', filas)
    conn.commit()
    return conn


def _clusters(conn, fps):
    """La misma consulta que hace `acoustic_ids_for`."""
    marcas = ','.join('?' * len(fps))
    filas = conn.execute(
        f'SELECT id, fingerprint, acoustic_id FROM tracks '
        f'WHERE acoustic_id IS NOT NULL '
        f'  AND (fingerprint IN ({marcas}) OR id IN ({marcas}))',
        fps + fps,
    ).fetchall()
    pedidas, fuera = set(fps), {}
    for r in filas:
        if not r['acoustic_id']:
            continue
        for clave in (r['fingerprint'], r['id']):
            if clave in pedidas:
                fuera[clave] = r['acoustic_id']
    return fuera


# ============================================================================
# EL AGRUPADO
# ============================================================================

def test_dos_ficheros_distintos_del_MISMO_tema_caen_juntos(tmp_path):
    """El caso entero, en un test: dos MD5 distintos —FLAC y MP3 del mismo
    tema— con el mismo cluster acustico."""
    conn = _bd(tmp_path, [
        ('t1', 'md5_flac', 'ac_shivers'),
        ('t2', 'md5_mp3',  'ac_shivers'),
        ('t3', 'md5_otro', 'ac_otro'),
    ])
    try:
        r = _clusters(conn, ['md5_flac', 'md5_mp3', 'md5_otro'])
    finally:
        conn.close()

    # El cliente agrupa por valor: 2+ huellas SUYAS en un cluster = duplicado.
    grupos = {}
    for fp, ac in r.items():
        grupos.setdefault(ac, []).append(fp)
    dupes = {ac: fps for ac, fps in grupos.items() if len(fps) > 1}

    assert list(dupes) == ['ac_shivers']
    assert sorted(dupes['ac_shivers']) == ['md5_flac', 'md5_mp3']


def test_las_huellas_SIN_cluster_no_salen_en_el_mapa(tmp_path):
    """«No tiene huella acustica todavia» y «no tiene duplicados» son cosas
    distintas. Devolver `null` para las primeras las haria indistinguibles de
    las segundas, y el usuario leeria «limpio» donde pone «no lo se»."""
    conn = _bd(tmp_path, [
        ('t1', 'md5_con', 'ac_1'),
        ('t2', 'md5_sin', None),
    ])
    try:
        r = _clusters(conn, ['md5_con', 'md5_sin'])
    finally:
        conn.close()

    assert 'md5_con' in r
    assert 'md5_sin' not in r


def test_los_registros_ANTIGUOS_se_encuentran_por_id(tmp_path):
    """En las filas viejas el id ES el MD5 (`tracks.id = tracks.fingerprint`).
    Buscar solo por la columna `fingerprint` dejaria fuera media biblioteca
    historica — y esa es justo la que lleva mas tiempo acumulando copias."""
    conn = _bd(tmp_path, [('md5_legacy', None, 'ac_9')])
    try:
        r = _clusters(conn, ['md5_legacy'])
    finally:
        conn.close()

    assert r == {'md5_legacy': 'ac_9'}


def test_la_clave_devuelta_es_la_que_PIDIO_el_cliente(tmp_path):
    """Si una fila tiene id y fingerprint distintos y el cliente pregunto por
    uno, devolverle el otro le da una clave que no reconoce y el grupo se
    queda huerfano."""
    conn = _bd(tmp_path, [('id_raro', 'md5_bueno', 'ac_5')])
    try:
        por_fp = _clusters(conn, ['md5_bueno'])
        por_id = _clusters(conn, ['id_raro'])
    finally:
        conn.close()

    assert por_fp == {'md5_bueno': 'ac_5'}
    assert por_id == {'id_raro': 'ac_5'}


def test_lote_vacio_no_revienta(tmp_path):
    conn = _bd(tmp_path, [('t1', 'md5', 'ac')])
    try:
        assert _clusters(conn, ['nada_que_ver']) == {}
    finally:
        conn.close()


# ============================================================================
# EL ENDPOINT
# ============================================================================

def _src(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


def test_el_endpoint_existe_y_tiene_tope():
    """500, igual que `/check-analyzed-by-fingerprint`. Sin tope, un cliente
    con 50.000 tracks manda una query con 100.000 parametros y SQLite corta
    por su limite de variables."""
    src = _src('routes/analysis_artwork.py')
    assert '@router.post("/acoustic-clusters")' in src
    i = src.index('async def acoustic_clusters(')
    fn = src[i:i + 1800]
    assert 'len(fps) > 500' in fn
    assert 'Máximo 500' in fn


def test_el_endpoint_separa_las_sin_cluster():
    src = _src('routes/analysis_artwork.py')
    i = src.index('async def acoustic_clusters(')
    fn = src[i:i + 1800]
    assert '"without_cluster": sin_cluster' in fn
    assert '"clusters": clusters' in fn


def test_una_sola_query_no_una_por_huella():
    """Con 500 elementos, la diferencia entre un IN y un bucle de SELECTs son
    dos ordenes de magnitud — y esto corre sobre la tabla colectiva."""
    src = _src('database.py')
    i = src.index('def acoustic_ids_for(')
    fn = src[i:i + 2200]
    assert "','.join('?' * len(fps))" in fn
    assert fn.count('c.execute(') == 1
