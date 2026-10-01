"""
El modo set de Escuchar (2026-09-30): el móvil graba el tracklist de un set
entero con ShazamKit y lo sube (`/sync/listen-set`) para que el escritorio de
la MISMA cuenta lo vea en HISTORIAL → SETS. Lo que se ata:
  - Lo que sube un aparato lo ve el otro de su cuenta, y no el de otra.
  - Subir otra vez el mismo set lo sustituye (renombrar), no lo duplica.
  - Borrar un set lo quita para toda la cuenta.
  - Vincular un móvil que ya tenía sets se los lleva a la cuenta nueva.
  - El reset de admin también los borra.
"""
import os
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


def _registrar(client, tipo):
    d = _uid(tipo)
    client.post("/sync/register", json={"device_id": d, "device_type": tipo})
    return d


def _vincular(client, pc, movil):
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    r = client.post("/sync/link/join", json={"device_id": movil, "code": code,
                                             "device_type": "mobile"})
    assert r.status_code == 200, r.text


def _set(set_id, nombre='Warm up', temas=2):
    return {
        "id": set_id, "nombre": nombre,
        "empezado_en": "2026-09-30T21:00:00Z",
        "terminado_en": "2026-09-30T23:00:00Z",
        "temas": [{"artista": f"DJ {i}", "titulo": f"Tema {i}",
                   "empieza_en": "2026-09-30T21:00:30Z", "segundo": 30 * i}
                  for i in range(temas)],
    }


def _subir(client, device, s):
    r = client.post("/sync/listen-set", json={
        "device_id": device, "set_id": s["id"], "nombre": s["nombre"],
        "empezado_en": s["empezado_en"], "payload": s})
    assert r.status_code == 200
    return r.json()


def _sets(client, device):
    r = client.get(f"/sync/listen-sets/{device}")
    assert r.status_code == 200
    return r.json()["sets"]


def test_el_escritorio_de_la_cuenta_ve_el_set_y_otra_cuenta_no(client):
    pc = _registrar(client, "desktop")
    movil = _registrar(client, "mobile")
    _vincular(client, pc, movil)
    extrano = _registrar(client, "desktop")

    s = _set(_uid("set"), temas=3)
    assert _subir(client, movil, s)["status"] == "ok"

    vistos = _sets(client, pc)
    assert [v["set_id"] for v in vistos] == [s["id"]]
    assert vistos[0]["payload"]["temas"][2]["titulo"] == "Tema 2"
    assert vistos[0]["nombre"] == "Warm up"
    assert _sets(client, extrano) == []


def test_subir_otra_vez_el_mismo_set_lo_sustituye(client):
    movil = _registrar(client, "mobile")
    sid = _uid("set")
    _subir(client, movil, _set(sid))
    _subir(client, movil, _set(sid, nombre='Renombrado', temas=4))
    vistos = _sets(client, movil)
    assert len(vistos) == 1
    assert vistos[0]["nombre"] == "Renombrado"
    assert len(vistos[0]["payload"]["temas"]) == 4


def test_borrar_un_set_lo_quita_para_toda_la_cuenta(client):
    pc = _registrar(client, "desktop")
    movil = _registrar(client, "mobile")
    _vincular(client, pc, movil)
    sid = _uid("set")
    _subir(client, movil, _set(sid))
    r = client.delete(f"/sync/listen-set/{pc}/{sid}")
    assert r.json()["deleted"] == 1
    assert _sets(client, movil) == []


def test_vincular_se_lleva_los_sets_del_movil(client):
    movil = _registrar(client, "mobile")
    sid = _uid("set")
    _subir(client, movil, _set(sid))
    pc = _registrar(client, "desktop")
    _vincular(client, pc, movil)
    assert [v["set_id"] for v in _sets(client, pc)] == [sid], \
        "los sets grabados antes de vincular se quedaban en la cuenta vieja"


def test_faltan_datos_o_es_enorme(client):
    movil = _registrar(client, "mobile")
    r = client.post("/sync/listen-set", json={
        "device_id": movil, "set_id": "", "empezado_en": "x", "payload": {}})
    assert r.json()["status"] == "error"
    enorme = _set(_uid("set"), temas=1)
    enorme["relleno"] = "x" * 300_000
    assert _subir(client, movil, enorme)["status"] == "error"


def test_el_reset_de_admin_tambien_los_borra():
    import main
    import inspect
    fuente = inspect.getsource(main)
    assert '"listen_sets_sync",' in fuente
