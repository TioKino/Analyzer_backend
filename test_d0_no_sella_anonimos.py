"""Cada visita a la web metia una fila basura en la tabla que NO se purga.

`log_event` sella el D0 con:

    INSERT OR IGNORE INTO device_first_seen (device_id, ...) VALUES (?, ...)

y los eventos de la web —`web_visit`, `web_download_click`— llegan con
`device_id = NULL` a proposito (no se manda identidad desde el navegador).

En SQLite un `TEXT PRIMARY KEY` **admite NULL, y admite muchos**: dos NULL no
chocan en un indice UNIQUE, asi que el `OR IGNORE` no ignoraba nada y cada
visita anadia una fila mas.

Y da justo en la peor tabla. `device_first_seen` es la unica que se deja fuera
de la purga a proposito: guarda el D0, y purgarla reescribiria la fecha de alta
de los veteranos, que fue el bug que inflaba la retencion. O sea que esto crecia
sin techo y para siempre.

La cohorte no llego a contarlas porque su CTE filtra `device_id IS NOT NULL`.
Se salvo, pero por casualidad: el numero que decide el paywall no puede depender
de que todos los consumidores futuros se acuerden del filtro.

    pytest test_d0_no_sella_anonimos.py -v
"""

import os
import sqlite3
import tempfile

import pytest

from database import AnalysisDB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield AnalysisDB(db_path=path)
    try:
        os.unlink(path)
    except OSError:
        pass


def _filas(db, where=''):
    conn = sqlite3.connect(db.db_path)
    try:
        return conn.execute(
            f'SELECT COUNT(*) FROM device_first_seen {where}').fetchone()[0]
    finally:
        conn.close()


# ============================================================================
# EL BUG
# ============================================================================

def test_una_visita_web_NO_deja_fila_de_D0(db):
    db.log_event(device_id=None, event_name='web_visit', platform='web')
    assert _filas(db) == 0


def test_cien_visitas_web_siguen_sin_dejar_ni_una(db):
    """El `OR IGNORE` no dedupe NULLs: antes esto dejaba cien filas."""
    for _ in range(100):
        db.log_event(device_id=None, event_name='web_visit', platform='web')
    assert _filas(db) == 0


def test_pero_el_EVENTO_si_se_guarda(db):
    """Lo que no puede pasar es perder la medicion de la web por arreglar la
    tabla del D0. El evento anonimo es el dato; la fila de D0 era la basura."""
    db.log_event(device_id=None, event_name='web_visit', platform='web')
    db.log_event(device_id=None, event_name='web_download_click',
                 platform='web')
    conn = sqlite3.connect(db.db_path)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_name LIKE 'web_%'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 2


# ============================================================================
# LO QUE NO PUEDE ROMPERSE
# ============================================================================

def test_un_dispositivo_de_verdad_SIGUE_sellando_su_D0(db):
    db.log_event(device_id='dev-1', event_name='app_opened', platform='macos')
    assert _filas(db, "WHERE device_id = 'dev-1'") == 1


def test_el_D0_se_sella_UNA_vez_por_dispositivo(db):
    for _ in range(5):
        db.log_event(device_id='dev-1', event_name='app_opened',
                     platform='macos')
    assert _filas(db, "WHERE device_id = 'dev-1'") == 1


def test_mezclar_anonimos_y_reales_no_confunde_la_cuenta(db):
    for _ in range(10):
        db.log_event(device_id=None, event_name='web_visit', platform='web')
    for d in ('a', 'b', 'c'):
        db.log_event(device_id=d, event_name='app_opened', platform='ios')
    assert _filas(db) == 3


# ============================================================================
# LIMPIAR LO QUE YA HAY EN PRODUCCION
# ============================================================================

def test_el_backfill_barre_las_filas_anonimas_viejas(db):
    """El arreglo de `log_event` para la sangria; no cura lo ya escrito."""
    conn = sqlite3.connect(db.db_path)
    try:
        for _ in range(7):
            conn.execute(
                "INSERT INTO device_first_seen (device_id, first_day) "
                "VALUES (NULL, date('now'))")
        conn.execute(
            "INSERT INTO device_first_seen (device_id, first_day) "
            "VALUES ('veterano', '2026-01-01')")
        conn.commit()
    finally:
        conn.close()
    assert _filas(db) == 8

    db.backfill_first_seen()

    assert _filas(db, 'WHERE device_id IS NULL') == 0
    # Y lo que valia sigue ahi, con SU fecha: reescribirla es exactamente el
    # fallo que inflaba la retencion.
    conn = sqlite3.connect(db.db_path)
    try:
        fila = conn.execute(
            "SELECT first_day FROM device_first_seen "
            "WHERE device_id = 'veterano'").fetchone()
    finally:
        conn.close()
    assert fila is not None and fila[0] == '2026-01-01'


# ============================================================================
# EL DAÑO COLATERAL: `NOT IN` CON UN NULL DENTRO NUNCA ES CIERTO
# ============================================================================

def test_una_sola_fila_NULL_mataba_la_rama_de_respaldo_de_la_cohorte():
    """El CTE del embudo tiene una segunda rama para los veteranos:

        WHERE ... AND device_id NOT IN (SELECT device_id FROM device_first_seen)

    En SQL, `x NOT IN (<conjunto con un NULL>)` **nunca es cierto**: evalua a
    desconocido para TODA x. O sea que con UNA sola fila NULL ahi dentro —desde
    la primera visita a la web— esa rama dejaba de devolver una sola fila.
    Estaba muerta y no lo dijo nadie.

    El daño real fue pequeño porque `backfill_first_seen` ya habia sembrado a
    casi todo el parque, pero el fallo no depende de eso: depende de que no
    haya NULLs.
    """
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE dfs (device_id TEXT PRIMARY KEY, first_day TEXT)')
    conn.execute('CREATE TABLE events (device_id TEXT, event_name TEXT)')
    conn.execute("INSERT INTO events VALUES ('veterano', 'app_opened')")
    conn.execute("INSERT INTO dfs VALUES ('otro', '2026-05-01')")

    consulta = (
        "SELECT device_id FROM events "
        "WHERE event_name = 'app_opened' AND device_id IS NOT NULL "
        "  AND device_id NOT IN (SELECT device_id FROM dfs) "
        "GROUP BY device_id"
    )

    assert conn.execute(consulta).fetchall() == [('veterano',)]

    conn.execute("INSERT INTO dfs VALUES (NULL, '2026-08-27')")
    assert conn.execute(consulta).fetchall() == [], (
        'si esto deja de ser cierto, SQLite cambio de semantica — pero '
        'mientras lo sea, un NULL en esa tabla mata la rama entera'
    )
