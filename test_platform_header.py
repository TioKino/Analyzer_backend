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
