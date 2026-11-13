"""Clics de descarga de la web por tienda frente a las altas de esa
plataforma (auditoría P1, 2026-11-13). Ver
`AnalysisDB.descargas_web_frente_a_altas`."""
import json
import os
import tempfile
from datetime import datetime, timezone

from database import AnalysisDB


def test_cada_tienda_con_sus_altas():
    db = AnalysisDB(db_path=tempfile.mktemp(suffix=".db"))
    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = db._open_conn()
    for tienda, n in (("windows", 10), ("dmg", 4), ("play", 2)):
        for _ in range(n):
            conn.execute("INSERT INTO events (timestamp, event_name, props, platform) "
                         "VALUES (?, 'web_download_click', ?, 'web')",
                         (hoy, json.dumps({"store": tienda})))
    for i, plat in enumerate(["windows"] * 3 + ["macos-dmg"] * 2 + ["android"]):
        conn.execute("INSERT INTO device_first_seen (device_id, first_day, first_platform) "
                     "VALUES (?, ?, ?)", (f"d{i}", hoy[:10], plat))
    conn.commit()
    conn.close()

    r = db.descargas_web_frente_a_altas(30)["por_tienda"]
    assert r["windows"] == {"clics": 10, "altas": 3, "altas_por_clic": 0.3}
    assert r["dmg"] == {"clics": 4, "altas": 2, "altas_por_clic": 0.5}
    assert r["play"]["altas"] == 1


def test_el_panel_lo_da():
    src = open(os.path.join(os.path.dirname(__file__), "routes", "admin_panel.py")).read()
    assert '"descargas_web": _descargas_web()' in src
