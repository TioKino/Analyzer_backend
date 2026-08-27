"""Un dispositivo, un voto. Antes era un voto por marca de tiempo.

SEC-12 estaba anotado como «mitigado, no cerrado»: la mitigacion por IP sube el
coste de fabricar consenso a «una botnet». Pero al ir a cerrarlo aparecio que el
agujero era mas ancho de lo que decia la nota, y no hacia falta ni inventarse
IDs:

  1. `corrections` NO tenia columna `device_id`. `save_correction` lo recibia
     desde siempre y no lo guardaba en ningun sitio.
  2. El consenso contaba `COUNT(DISTINCT track_id || corrected_at)`, o sea POR
     MARCA DE TIEMPO. El mismo aparato corrigiendo tres veces fabricaba un
     `consensus_3` — prioridad 80 en ANALYSIS_SOURCE_PRIORITY, que gana al
     motor local (50) y al tag id3 (30).
  3. El `DELETE` que parecia dedup borraba «la correccion mas reciente de este
     track+campo» SIN MIRAR DE QUIEN ERA. Con dos DJs corrigiendo el mismo
     tema, el segundo borraba al primero y el consenso no subia de 1 nunca.

El (3) es especialmente feo: la defensa y el bug eran la misma linea.

    pytest test_sec12_voto_por_dispositivo.py -v
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _bd(tmp_path):
    """`corrections` con la forma de hoy."""
    conn = sqlite3.connect(str(tmp_path / 'c.db'))
    conn.row_factory = sqlite3.Row
    conn.execute(
        'CREATE TABLE corrections (id INTEGER PRIMARY KEY AUTOINCREMENT, '
        'track_id TEXT, field TEXT, old_value TEXT, new_value TEXT, '
        'corrected_at TEXT, fingerprint TEXT, device_id TEXT)'
    )
    return conn


def _votos(conn, fp='fp1', field='bpm'):
    return conn.execute(
        "SELECT new_value,"
        "       COUNT(DISTINCT COALESCE(device_id, 'anon:' || id)) AS n"
        "  FROM corrections WHERE fingerprint = ? AND field = ?"
        " GROUP BY new_value ORDER BY n DESC", (fp, field)
    ).fetchall()


def _mete(conn, valor, device_id, cuando):
    conn.execute(
        'INSERT INTO corrections (track_id, field, new_value, corrected_at, '
        'fingerprint, device_id) VALUES (?,?,?,?,?,?)',
        ('t1', 'bpm', valor, cuando, 'fp1', device_id))
    conn.commit()


# ============================================================================
# EL CONTEO
# ============================================================================

def test_un_aparato_votando_tres_veces_es_UN_voto(tmp_path):
    """El agujero, en su forma mas barata: sin inventarse IDs, sin botnet."""
    conn = _bd(tmp_path)
    try:
        for hora in ('10:00', '11:00', '12:00'):
            _mete(conn, '128', 'dja_uno', f'2026-08-27T{hora}:00')
        filas = _votos(conn)
    finally:
        conn.close()

    assert len(filas) == 1
    assert filas[0]['n'] == 1, 'tres pulsaciones del mismo aparato = un voto'


def test_tres_aparatos_distintos_SI_son_tres(tmp_path):
    # Endurecer no puede cargarse el consenso legitimo, que es el producto.
    conn = _bd(tmp_path)
    try:
        for d in ('dja_uno', 'dja_dos', 'dja_tres'):
            _mete(conn, '128', d, '2026-08-27T10:00:00')
        filas = _votos(conn)
    finally:
        conn.close()

    assert filas[0]['n'] == 3


def test_las_filas_VIEJAS_sin_device_id_siguen_contando(tmp_path):
    """`COUNT(DISTINCT device_id)` IGNORA los NULL en SQL.

    Sin el `COALESCE(device_id, 'anon:' || id)`, todo el consenso acumulado
    antes de que la columna existiera caeria a 0 de golpe — un endurecimiento
    que borra datos legitimos no es un endurecimiento, es una regresion. Cada
    fila vieja sigue valiendo uno, que es exactamente lo que valia.

    Es la misma trampa que ya mordio en el embudo con los eventos anonimos de
    la web, donde `COUNT(DISTINCT device_id)` daba 0 por muchas visitas que
    hubiera.
    """
    conn = _bd(tmp_path)
    try:
        for _ in range(3):
            _mete(conn, '128', None, '2026-08-01T10:00:00')
        filas = _votos(conn)
    finally:
        conn.close()

    assert filas[0]['n'] == 3, 'el consenso historico se ha evaporado'


def test_mezcla_de_viejo_y_nuevo(tmp_path):
    conn = _bd(tmp_path)
    try:
        _mete(conn, '128', None, '2026-08-01T10:00:00')       # legado
        _mete(conn, '128', 'dja_uno', '2026-08-27T10:00:00')  # nuevo
        _mete(conn, '128', 'dja_uno', '2026-08-27T11:00:00')  # el mismo
        filas = _votos(conn)
    finally:
        conn.close()

    assert filas[0]['n'] == 2, 'legado + un dispositivo = 2, no 3'


# ============================================================================
# EL DEDUP QUE BORRABA VOTOS AJENOS
# ============================================================================

def test_revotar_sustituye_el_TUYO_no_el_del_vecino(tmp_path):
    """El `DELETE` viejo quitaba la correccion mas reciente del track+campo sin
    mirar de quien era. Dos DJs corrigiendo el mismo tema: el segundo borraba
    al primero y el consenso se quedaba en 1 para siempre.

    La defensa y el bug eran la misma linea.
    """
    conn = _bd(tmp_path)
    try:
        _mete(conn, '128', 'dja_uno', '2026-08-27T10:00:00')
        _mete(conn, '128', 'dja_dos', '2026-08-27T10:05:00')

        # dja_dos se corrige a si mismo: solo debe caer SU fila.
        conn.execute(
            'DELETE FROM corrections WHERE fingerprint = ? AND field = ? '
            'AND device_id = ?', ('fp1', 'bpm', 'dja_dos'))
        conn.commit()
        _mete(conn, '130', 'dja_dos', '2026-08-27T10:06:00')

        restantes = {
            (r['device_id'], r['new_value'])
            for r in conn.execute('SELECT device_id, new_value FROM corrections')
        }
    finally:
        conn.close()

    assert ('dja_uno', '128') in restantes, 'le borro el voto al otro DJ'
    assert ('dja_dos', '130') in restantes
    assert ('dja_dos', '128') not in restantes


# ============================================================================
# EL CABLEADO
# ============================================================================

def _src(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


def test_la_columna_se_crea_y_se_escribe():
    src = _src('database.py')
    assert 'ALTER TABLE corrections ADD COLUMN device_id TEXT' in src
    assert 'fingerprint, device_id)' in src


def test_el_delete_filtra_por_device_id():
    src = _src('database.py')
    i = src.index('if device_id and fingerprint:')
    bloque = src[i:i + 400]
    assert 'AND device_id = ?' in bloque
    assert 'ORDER BY corrected_at DESC LIMIT 1' not in bloque, (
        'vuelve el DELETE que borra el voto del vecino'
    )


def test_los_DOS_caminos_de_consenso_cuentan_igual():
    """`get_consensus` y `get_all_consensus` tienen su propia query. Arreglar
    una y no la otra haria que el mismo track diera consensos distintos segun
    por donde se preguntara."""
    src = _src('database.py')
    # Acotado a las dos funciones: ese mismo COUNT se usa tambien en una query
    # de dispositivos que no tiene nada que ver, y contar sobre el fichero
    # entero ataria el test a codigo ajeno.
    for fn in ('def get_consensus(', 'def get_all_consensus('):
        i = src.index(fn)
        cuerpo = src[i:i + 2500]
        assert "COUNT(DISTINCT COALESCE(device_id, 'anon:' || id))" in cuerpo, fn
    assert 'COUNT(DISTINCT track_id || corrected_at)' not in src
