"""En que version esta el parque — «¿llegan mis releases?».

Hasta el 2026-09-12 esa pregunta no se podia contestar con nada del panel. Lo
unico parecido era el `download_count` de GitHub Releases, que marcaba **6
descargas** del zip de Windows de la 2.9.11 en cinco dias con ~275 escritorios
activos — pero cuenta DESCARGAS (bots y reintentos incluidos), no gente
ejecutandola, asi que de ahi no se podia concluir nada firme.

Y la pregunta no era teorica: el snapshot 37 vio 281 tracks entrando sin huella
acustica desde un motor local de Windows, y las dos causas posibles —motor
viejo (se cura al actualizar) y motor sin `fpcalc` (no se cura jamas)— se
separan sabiendo en que version esta ese parque.

Este fichero ata las dos formas de equivocarse que tiene esta metrica.

    pytest test_adopcion_por_version.py -v
"""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.admin_panel import _orden_de_version, _reparto_por_version  # noqa: E402


# ============================================================================
# 1. ORDENAR VERSIONES COMO TEXTO ESTA MAL
# ============================================================================

def test_2_9_9_no_es_mayor_que_2_9_11():
    """El fallo silencioso mas facil de esta tabla.

    Como texto, '2.9.9' > '2.9.11': compara caracter a caracter y '9' > '1'.
    O sea que la tabla de adopcion pondria arriba como «la mas reciente» una
    version que no lo es — justo en el momento en que se usa para decidir si un
    fix ha llegado a la gente.
    """
    assert '2.9.9' > '2.9.11'  # el comportamiento de texto que hay que evitar
    assert _orden_de_version('2.9.11') > _orden_de_version('2.9.9')


def test_ordena_una_lista_entera_bien():
    versiones = ['2.9.9', '2.9.11', '2.10.0', '2.9.8', '3.0.0']
    ordenadas = sorted(versiones, key=_orden_de_version, reverse=True)
    assert ordenadas == ['3.0.0', '2.10.0', '2.9.11', '2.9.9', '2.9.8']


def test_una_version_rara_no_revienta_ni_se_cuela_arriba():
    """Un `app_version` con basura no puede tumbar el endpoint ni aparecer como
    la version mas nueva del parque."""
    assert _orden_de_version('vete-a-saber') < _orden_de_version('0.0.1')
    assert _orden_de_version(None) < _orden_de_version('0.0.1')
    # `2.9.11+40` (version+build) tiene que ordenarse por encima de `2.9.11`.
    assert _orden_de_version('2.9.11+40') > _orden_de_version('2.9.11')


# ============================================================================
# 2. LA VERSION ES LA ULTIMA VISTA, Y `sin_version` NO ES UNA VERSION
# ============================================================================

@pytest.fixture
def db(tmp_path):
    con = sqlite3.connect(':memory:')
    con.execute(
        'CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, '
        'timestamp TEXT, device_id TEXT, event_name TEXT, props TEXT, '
        'platform TEXT, app_version TEXT)'
    )
    return con


def _ev(con, dev, ts, ver, plat='windows'):
    con.execute(
        'INSERT INTO events (timestamp, device_id, event_name, platform, app_version) '
        'VALUES (?,?,?,?,?)', (ts, dev, 'app_opened', plat, ver)
    )


def test_cuenta_la_ULTIMA_version_de_cada_aparato_no_la_primera(db):
    """Alguien que entro con la 2.9.9 y hoy corre la 2.9.11 cuenta como 2.9.11.

    `device_first_seen.first_app_version` contesta otra pregunta —«con que
    version entro cada uno»— que sirve para hablar de altas, no de adopcion.
    Mezclarlas daria una foto que envejece AL REVES: cuanta mas gente
    actualice, mas viejo pareceria el parque.
    """
    _ev(db, 'dev1', '2026-09-01T10:00:00', '2.9.9')
    _ev(db, 'dev1', '2026-09-12T10:00:00', '2.9.11')   # el mismo, ya actualizado
    _ev(db, 'dev2', '2026-09-11T10:00:00', '2.9.9')

    r = _reparto_por_version(db, 3650, '', [])
    assert r['por_version'] == {'2.9.11': 1, '2.9.9': 1}
    assert r['devices'] == 2


def test_sin_version_va_APARTE_y_no_se_inventa_una_etiqueta(db):
    """Un cliente que no manda `app_version` no es «la version desconocida».

    Escribir ahi un 'unknown' convertiria un hueco en un dato, que es el mismo
    fallo que `client_platform` evita devolviendo None. Y ademas se colaria en
    la tabla como si fuera una version mas del parque.
    """
    _ev(db, 'dev1', '2026-09-12T10:00:00', '2.9.11')
    _ev(db, 'dev2', '2026-09-12T10:00:00', None)

    r = _reparto_por_version(db, 3650, '', [])
    assert r['sin_version'] == 1
    assert r['por_version'] == {'2.9.11': 1}
    assert 'unknown' not in r['por_version']
    assert 'desconocida' not in r['por_version']
    # Pero SI cuenta en el total: es un aparato activo, solo que mudo.
    assert r['devices'] == 2


def test_sale_ordenado_de_mas_nueva_a_mas_vieja(db):
    for i, ver in enumerate(['2.9.9', '2.9.11', '2.9.10']):
        _ev(db, f'dev{i}', '2026-09-12T10:00:00', ver)
    r = _reparto_por_version(db, 3650, '', [])
    assert list(r['por_version'].keys()) == ['2.9.11', '2.9.10', '2.9.9']


def test_respeta_el_filtro_de_plataforma(db):
    _ev(db, 'dev1', '2026-09-12T10:00:00', '2.9.11', plat='windows')
    _ev(db, 'dev2', '2026-09-12T10:00:00', '2.9.11', plat='android')
    r = _reparto_por_version(db, 3650, ' AND LOWER(platform) IN (?)', ['windows'])
    assert r['devices'] == 1


def test_un_aparato_fuera_de_la_ventana_no_cuenta(db):
    """El parque son los ACTIVOS. Alguien que no abre la app desde hace meses
    no dice nada sobre si una release esta llegando.

    Las fechas van RELATIVAS a hoy a proposito: con timestamps fijos, este test
    pasa hoy y empieza a fallar solo cuando el calendario los saque de la
    ventana. Un test que caduca es peor que no tenerlo.
    """
    from datetime import datetime, timedelta, timezone
    ahora = datetime.now(timezone.utc)
    hace_mucho = (ahora - timedelta(days=400)).strftime('%Y-%m-%dT%H:%M:%S')
    ayer = (ahora - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
    _ev(db, 'viejo', hace_mucho, '1.0.0')
    _ev(db, 'nuevo', ayer, '2.9.11')
    r = _reparto_por_version(db, 30, '', [])
    assert r['devices'] == 1
    assert r['por_version'] == {'2.9.11': 1}
