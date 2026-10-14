"""Las versiones se parten por PLATAFORMA (lectura 56, 2026-10-13).

Dos tercios del escritorio seguian en la 2.9.11 dos semanas despues de la
2.10.1, con el appcast sirviendola. Si son del Mac App Store, actualiza la
tienda; si son del DMG o de Windows, Sparkle no les esta llegando. Son dos
arreglos distintos y el reparto por version a secas no los separaba.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

from routes.admin_panel import _reparto_por_version


def _bd():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp TEXT, "
        "device_id TEXT, event_name TEXT, platform TEXT, app_version TEXT)"
    )
    return db


def _evento(db, dev, plat, ver, hace_horas):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hace_horas)).strftime(
        "%Y-%m-%d %H:%M:%S")
    db.execute(
        "INSERT INTO events (timestamp, device_id, event_name, platform, app_version) "
        "VALUES (?, ?, 'app_opened', ?, ?)", (ts, dev, plat, ver))


def test_cada_aparato_cuenta_en_su_plataforma_con_su_ultima_version():
    db = _bd()
    _evento(db, "mac1", "macos-mas", "2.9.11", 5)
    _evento(db, "mac2", "macos-dmg", "2.9.11", 50)
    _evento(db, "mac2", "macos-dmg", "2.10.1", 1)   # actualizo: cuenta la ultima
    _evento(db, "win1", "windows", "2.9.11", 3)
    _evento(db, "win2", "windows", "2.10.1", 3)
    _evento(db, "viejo", "macos", None, 3)          # sin version: aparte

    r = _reparto_por_version(db, 30, "", [])

    assert r["devices"] == 5
    assert r["sin_version"] == 1
    assert r["por_version"] == {"2.10.1": 2, "2.9.11": 2}
    assert r["por_plataforma"] == {
        "macos-dmg": {"2.10.1": 1},
        "macos-mas": {"2.9.11": 1},
        "windows": {"2.10.1": 1, "2.9.11": 1},
    }
    # Ordenadas de mayor a menor version por numeros, no como texto.
    assert list(r["por_plataforma"]["windows"]) == ["2.10.1", "2.9.11"]


def test_un_evento_sin_plataforma_no_se_pierde():
    db = _bd()
    _evento(db, "x", None, "2.9.8", 2)
    r = _reparto_por_version(db, 30, "", [])
    assert r["por_plataforma"] == {"(sin)": {"2.9.8": 1}}
