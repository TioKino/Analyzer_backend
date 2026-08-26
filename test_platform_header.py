"""La plataforma del cliente se sella en el track, y sirve para repartir.

El cliente manda `X-Platform` en CADA peticion desde hace versiones —android,
ios, windows, macos, linux— y el backend **no la leia en ningun sitio**. Un
`grep -rn "X-Platform"` sobre el repo entero no daba una sola linea.

Sin ella no se puede contestar la unica pregunta que decide donde invertir en
cobertura de huella acustica: del ~64% de tracks sin `chromaprint`, ¿cuanto es
movil y cuanto es la build del Mac App Store? Son causas distintas que piden
arreglos OPUESTOS:

  - movil    -> el backfill de huella no existe ahi
  - macos-mas-> existe, pero el sandbox impide que fpcalc abra ficheros, asi
                que no puede funcionar nunca

Y `engine_source` no sirve para separarlas: los dos van a Render.

    pytest test_platform_header.py -v
"""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation import ALLOWED_PLATFORMS, client_platform  # noqa: E402


class _FakeRequest:
    """Lo minimo que `client_platform` toca: las cabeceras."""

    def __init__(self, headers):
        self.headers = headers


# ============================================================================
# LA CABECERA
# ============================================================================

@pytest.mark.parametrize('valor', sorted(ALLOWED_PLATFORMS))
def test_las_plataformas_validas_pasan(valor):
    assert client_platform(_FakeRequest({'X-Platform': valor})) == valor


def test_macos_se_declara_en_DOS_sabores():
    """La particion que motiva todo esto.

    `macos-mas` y `macos-dmg` son la misma plataforma con la diferencia que
    decide el producto: MAS no puede generar huella, el DMG si. Juntarlas bajo
    `macos` borraria justo lo que se quiere medir — el mismo error que tener
    «AudD no lo reconoce» y «lo tiramos nosotros» en un solo contador.
    """
    assert 'macos-mas' in ALLOWED_PLATFORMS
    assert 'macos-dmg' in ALLOWED_PLATFORMS
    # `macos` a secas sigue valiendo: los clientes viejos lo mandan y tirar sus
    # tracks a `unknown` seria perder informacion que si tenemos.
    assert 'macos' in ALLOWED_PLATFORMS


def test_un_valor_desconocido_es_None_y_no_una_cadena_inventada():
    """None significa «no lo se», que NO es lo mismo que un valor concreto.

    Escribir aqui un `'unknown'` convertiria un hueco en un dato — el mismo
    fallo que la ficha de Info del cliente tenia con `bpm_source`, que caia a
    `'analysis'` y afirmaba que lo habia medido el backend.
    """
    assert client_platform(_FakeRequest({'X-Platform': 'nintendo-switch'})) is None
    assert client_platform(_FakeRequest({'X-Platform': ''})) is None
    assert client_platform(_FakeRequest({})) is None


def test_la_lista_es_CERRADA():
    """La cabecera la manda el cliente y va derecha a una columna que se
    agrupa. Aceptar texto libre seria basura en el panel y cardinalidad
    ilimitada en un GROUP BY."""
    assert client_platform(_FakeRequest({'X-Platform': 'a' * 500})) is None
    assert client_platform(_FakeRequest({'X-Platform': "'; DROP TABLE"})) is None


def test_se_normaliza_mayusculas_y_espacios():
    # Un cliente que mande ' iOS ' no puede caer a NULL por eso.
    assert client_platform(_FakeRequest({'X-Platform': '  iOS '}) ) == 'ios'


# ============================================================================
# EL REPARTO
# ============================================================================

def _bd_con(tracks, tmp_path):
    """BD minima con la forma real de la tabla, para probar el GROUP BY."""
    ruta = str(tmp_path / 'a.db')
    conn = sqlite3.connect(ruta)
    conn.execute(
        'CREATE TABLE tracks (id TEXT PRIMARY KEY, chromaprint TEXT, '
        'platform TEXT)'
    )
    conn.executemany('INSERT INTO tracks VALUES (?, ?, ?)', tracks)
    conn.commit()
    return conn


def test_el_reparto_solo_cuenta_los_que_NO_tienen_huella(tmp_path):
    conn = _bd_con([
        ('1', 'AQAB', 'ios'),          # tiene huella -> fuera
        ('2', None, 'ios'),
        ('3', '', 'android'),          # cadena vacia cuenta como sin huella
        ('4', None, 'macos-mas'),
        ('5', None, None),             # pre-instrumentacion
    ], tmp_path)
    try:
        filas = conn.execute(
            "SELECT COALESCE(platform, 'unknown') AS p, COUNT(*) AS n"
            "  FROM tracks WHERE chromaprint IS NULL OR chromaprint = ''"
            " GROUP BY p"
        ).fetchall()
    finally:
        conn.close()

    reparto = {p: n for p, n in filas}
    assert reparto == {'ios': 1, 'android': 1, 'macos-mas': 1, 'unknown': 1}


def test_los_pre_instrumentacion_se_VEN_como_unknown(tmp_path):
    """No se excluyen: mientras `unknown` domine, el porcentaje NO es
    representativo, y esconderlo haria que pareciera que si lo es.

    Es literalmente lo que paso con `device_linked`: el embudo daba «2 de 141
    (1,4%)» de vinculacion movil y se leyo como un embudo roto. El evento solo
    lo emitia el cliente desde 2.9.10 y el parque corria 2.9.9 — el paso salia
    a 0 por definicion.
    """
    conn = _bd_con([('1', None, None), ('2', None, None)], tmp_path)
    try:
        filas = conn.execute(
            "SELECT COALESCE(platform, 'unknown') AS p, COUNT(*) AS n"
            "  FROM tracks WHERE chromaprint IS NULL OR chromaprint = ''"
            " GROUP BY p"
        ).fetchall()
    finally:
        conn.close()

    assert filas == [('unknown', 2)]


# ============================================================================
# EL CABLEADO
# ============================================================================

def test_analyze_sella_la_plataforma():
    """Se comprueba sobre el fuente: montar una peticion real de /analyze en un
    test exige un audio y librosa entero."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'main.py'), encoding='utf-8') as f:
        src = f.read()
    assert "track_data['platform'] = client_platform(request)" in src


def test_save_track_escribe_la_columna():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'database.py'), encoding='utf-8') as f:
        src = f.read()
    # Las dos mitades: la columna en el INSERT y el valor en la tupla. Con una
    # sola, sqlite se queja de numero de parametros… o peor, mete el valor en
    # la columna de al lado.
    assert 'analysis_version, isrc, platform)' in src
    assert "track_data.get('platform')," in src
    assert 'ALTER TABLE tracks ADD COLUMN platform TEXT' in src


# ============================================================================
# EL AGUJERO: ¿legado o via abierta?
# ============================================================================

def _bd_gap(filas, tmp_path):
    """tracks con las columnas que mira `acoustic_gap_breakdown`."""
    ruta = str(tmp_path / 'gap.db')
    conn = sqlite3.connect(ruta)
    conn.row_factory = sqlite3.Row
    conn.execute(
        'CREATE TABLE tracks (id TEXT PRIMARY KEY, chromaprint TEXT, '
        'engine_source TEXT, analyzed_at TEXT)'
    )
    conn.executemany('INSERT INTO tracks VALUES (?, ?, ?, ?)', filas)
    conn.commit()
    return conn


def _gap(conn):
    """La misma logica que `acoustic_gap_breakdown`, sobre una conexion suelta.

    Se replica en vez de instanciar la BD real porque esa abre /data y corre
    todas las migraciones; lo que se prueba aqui es el SQL.
    """
    c = conn.cursor()
    sin = "chromaprint IS NULL OR chromaprint = ''"
    c.execute(
        "SELECT"
        "  SUM(CASE WHEN analyzed_at IS NULL THEN 1 ELSE 0 END) AS sin_fecha,"
        "  SUM(CASE WHEN substr(analyzed_at,1,10) >= date('now','-7 days')"
        "           THEN 1 ELSE 0 END) AS d7,"
        "  SUM(CASE WHEN substr(analyzed_at,1,10) >= date('now','-30 days')"
        "           THEN 1 ELSE 0 END) AS d30"
        f"  FROM tracks WHERE {sin}"
    )
    r = c.fetchone()
    return int(r['d7'] or 0), int(r['d30'] or 0), int(r['sin_fecha'] or 0)


def test_la_fecha_se_compara_por_DIA_no_como_texto_entero(tmp_path):
    """La trampa que se comio un dia entero de datos.

    `analyzed_at` se guarda con `datetime.now().isoformat()`:

        2026-08-26T09:15:00.123456

    y `datetime('now')` de SQLite da:

        2026-08-26 09:15:00

    Comparadas como texto, la 'T' (0x54) es MAYOR que el espacio (0x20), asi
    que un registro de HOY ordena despues del corte de hoy y el bucket se
    desplaza un dia. Comparando solo `substr(...,1,10)` no puede pasar — y
    ademas es lo que el bucket promete: dias naturales.
    """
    from datetime import datetime, timedelta
    hoy = datetime.now().isoformat()
    hace3 = (datetime.now() - timedelta(days=3)).isoformat()
    hace20 = (datetime.now() - timedelta(days=20)).isoformat()
    hace90 = (datetime.now() - timedelta(days=90)).isoformat()

    conn = _bd_gap([
        ('1', None, 'render', hoy),
        ('2', None, 'render', hace3),
        ('3', None, 'local_engine', hace20),
        ('4', None, None, hace90),
        ('5', 'AQAB', 'render', hoy),   # con huella -> fuera de todos
        ('6', None, 'render', None),    # sin fecha
    ], tmp_path)
    try:
        d7, d30, sin_fecha = _gap(conn)
    finally:
        conn.close()

    assert d7 == 2, 'hoy y hace 3 dias'
    assert d30 == 3, 'los 7 dias van DENTRO de los 30, no aparte'
    assert sin_fecha == 1


def test_newest_without_es_el_numero_que_decide(tmp_path):
    """Si el mas reciente sin huella es de hace meses, el agujero es legado y
    lo cierra el backfill. Si es de hoy, hay una via abierta y ampliar el
    backfill seria achicar agua sin taparla."""
    conn = _bd_gap([
        ('1', 'AQAB', 'render', '2026-08-26T10:00:00'),  # con huella: no cuenta
        ('2', None, 'render', '2026-03-01T10:00:00'),
        ('3', None, 'render', '2026-05-15T10:00:00'),
    ], tmp_path)
    try:
        m = conn.execute(
            "SELECT MAX(analyzed_at) AS m FROM tracks"
            " WHERE (chromaprint IS NULL OR chromaprint = '')"
            "   AND analyzed_at IS NOT NULL"
        ).fetchone()['m']
    finally:
        conn.close()

    assert m == '2026-05-15T10:00:00', 'el track CON huella no puede colarse'


def test_la_cadena_vacia_cuenta_como_sin_huella(tmp_path):
    # `chromaprint = ''` no es lo mismo que NULL para SQLite pero si para
    # nosotros: una huella vacia no agrupa con nadie.
    conn = _bd_gap([('1', '', 'render', '2026-08-26T10:00:00')], tmp_path)
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM tracks"
            " WHERE chromaprint IS NULL OR chromaprint = ''"
        ).fetchone()['n']
    finally:
        conn.close()
    assert n == 1


def test_el_panel_expone_el_reparto():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'routes', 'admin_panel.py')
    with open(ruta, encoding='utf-8') as f:
        src = f.read()
    assert '"acoustic_gap": _acoustic_gap(),' in src
    # Best-effort como el resto: una metrica de observacion no puede tumbar el
    # panel entero (ya paso con un OOM en admin).
    assert 'def _acoustic_gap()' in src
    assert 'except Exception' in src
