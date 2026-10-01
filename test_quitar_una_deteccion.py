"""Quitar una detección la quita de TODA la cuenta, y no vuelve (2026-10-01).

Hasta ese día quitar una detección (deslizarla, o la papelera) solo la quitaba
de la lista local: el escritorio pide el historial a Render al abrir
*DETECCIONES* y la volvía a bajar, así que no había forma de quitar una. Con
las detecciones de Escuchar guardándose solas, «No es este» tiene que poder
quitarla de aquí también.

Se marca con `borrado_at` en vez de borrar la fila: el pull da la lista de
`borrados` y cada aparato quita lo suyo que sea de antes. Volver a subirla
(DESHACER, o detectarla otra vez) la recupera.
"""
import os
import sqlite3
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.pop("RENDER", None)
    import main
    import sync_endpoints

    sync_endpoints.SYNC_AUTH_SECRET = ""  # dev mode: sin firma
    return TestClient(main.app)


def _uid(tag):
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _cuenta(client):
    """Un escritorio y un móvil vinculados."""
    pc, movil = _uid("pc"), _uid("movil")
    client.post("/sync/register", json={"device_id": pc, "device_type": "macos-dmg"})
    client.post("/sync/register", json={"device_id": movil, "device_type": "ios"})
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    r = client.post("/sync/link/join", json={"device_id": movil, "code": code,
                                             "device_type": "mobile"})
    assert r.status_code == 200, r.text
    return pc, movil


def _subir(client, device, artist, title, cuando="2026-10-01T20:00:00Z"):
    r = client.post("/sync/detected-track", json={
        "device_id": device, "artist": artist, "title": title,
        "payload": {"artist": artist, "title": title}, "detected_at": cuando})
    assert r.json()["status"] == "ok", r.text


def _pull(client, device):
    r = client.get(f"/sync/detected-tracks/{device}")
    assert r.status_code == 200
    return r.json()


def _titulos(pull):
    return {t["title"] for t in pull["tracks"]}


def test_quitada_en_el_escritorio_no_vuelve_y_el_movil_se_entera(client):
    pc, movil = _cuenta(client)
    _subir(client, movil, "Aly & Fila", "Altitude Compensation")
    _subir(client, movil, "Jamie Jones", "Panic")
    assert _titulos(_pull(client, pc)) == {"Altitude Compensation", "Panic"}

    r = client.post("/sync/detected-track/borrar", json={
        "device_id": pc, "artist": "aly & fila", "title": "ALTITUDE COMPENSATION"})
    assert r.json()["borradas"] == 1

    p = _pull(client, pc)
    assert _titulos(p) == {"Panic"}, "el pull ya no la vuelve a bajar"
    borrados = _pull(client, movil)["borrados"]
    assert [(b["artist"], b["title"]) for b in borrados] == [
        ("Aly & Fila", "Altitude Compensation")]
    assert borrados[0]["borrado_at"], "con la hora, para quitar solo lo de antes"


def test_subirla_otra_vez_la_recupera(client):
    # DESHACER tras «No es este», o volver a detectarla.
    pc, movil = _cuenta(client)
    _subir(client, movil, "Energy 52", "Café Del Mar")
    client.post("/sync/detected-track/borrar", json={
        "device_id": movil, "artist": "Energy 52", "title": "Café Del Mar"})
    assert _titulos(_pull(client, pc)) == set()
    _subir(client, movil, "Energy 52", "Café Del Mar", "2026-10-01T21:00:00Z")
    p = _pull(client, pc)
    assert _titulos(p) == {"Café Del Mar"}
    assert p["borrados"] == [], "recuperada en el único aparato que la tenía"


def test_la_papelera_las_quita_todas(client):
    pc, movil = _cuenta(client)
    for t in ("Uno", "Dos", "Tres"):
        _subir(client, movil, "Artista", t)
    r = client.post("/sync/detected-track/borrar",
                    json={"device_id": pc, "todas": True})
    assert r.json()["borradas"] == 3
    p = _pull(client, movil)
    assert p["tracks"] == []
    assert {b["title"] for b in p["borrados"]} == {"Uno", "Dos", "Tres"}


def test_no_toca_otra_cuenta(client):
    pc, movil = _cuenta(client)
    otro_pc, otro_movil = _cuenta(client)
    _subir(client, movil, "Talking Beats", "Saturn Five")
    _subir(client, otro_movil, "Talking Beats", "Saturn Five")
    client.post("/sync/detected-track/borrar", json={
        "device_id": pc, "artist": "Talking Beats", "title": "Saturn Five"})
    assert _titulos(_pull(client, otro_pc)) == {"Saturn Five"}
    assert _pull(client, otro_pc)["borrados"] == []


def test_faltan_datos(client):
    pc, _ = _cuenta(client)
    r = client.post("/sync/detected-track/borrar",
                    json={"device_id": pc, "artist": "Solo artista"})
    assert r.status_code == 400


def test_una_bd_de_antes_gana_la_columna(tmp_path):
    import sync_endpoints
    conn = sqlite3.connect(str(tmp_path / "vieja.db"))
    conn.execute("""CREATE TABLE detected_tracks_sync (
        id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL,
        artist TEXT NOT NULL, title TEXT NOT NULL, payload TEXT NOT NULL,
        detected_at TEXT NOT NULL, user_id TEXT DEFAULT '',
        UNIQUE(device_id, artist, title))""")
    sync_endpoints._migrate_detecciones_borradas(conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(detected_tracks_sync)")]
    assert "borrado_at" in cols
    sync_endpoints._migrate_detecciones_borradas(conn)  # dos veces no rompe
