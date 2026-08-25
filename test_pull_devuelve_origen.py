"""
El pull tiene que decir QUIÉN subió cada item.

`full=true` devuelve también lo que subió el propio dispositivo — y tiene que
hacerlo: un PC formateado es quien había subido casi todo, y sin eso
«Restaurar desde la nube» le devolvía cero.

Pero entonces el cliente recibe una foto vieja de sí mismo, y los bloques de
organización (`all_folders`, `all_collections`…) se aplican por sustitución
completa. El owner publicó con 0 carpetas, importó después una carpeta del HDD,
pulsó restaurar, y el pull le devolvió su propio «0 carpetas» y se la borró.

El filtro NO puede vivir aquí (sin lo propio volvemos al bug del PC formateado):
vive en el cliente, y para eso necesita `last_device_id`. Este test existe
porque ese campo es una línea en un SELECT y quitarlo no rompe nada de forma
visible — solo vuelve a borrar carpetas en silencio.
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

    sync_endpoints.SYNC_AUTH_SECRET = ""
    return TestClient(main.app)


def _uid(tag):
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _register(client, device_id, device_type="desktop"):
    r = client.post("/sync/register",
                    json={"device_id": device_id, "device_type": device_type})
    assert r.status_code == 200


def _push_carpetas(client, device, device_type="macos"):
    r = client.post("/sync/push", json={
        "device_id": device, "device_type": device_type,
        "changes": [{
            "data_type": "folder", "item_key": "all_folders",
            "payload": {"folders": [], "track_folders": {}},
            "deleted": False,
        }],
    })
    assert r.status_code == 200


def test_full_dice_quien_subio_cada_item(client):
    pc = _uid("pc")
    _register(client, pc)
    _push_carpetas(client, pc)

    r = client.get(f"/sync/pull/{pc}?full=true")
    assert r.status_code == 200
    cambios = [c for c in r.json()["changes"] if c["item_key"] == "all_folders"]

    assert cambios, "full=true tiene que devolver lo que subio este mismo aparato"
    assert cambios[0].get("last_device_id") == pc, (
        "sin last_device_id el cliente no puede reconocer su propio eco, y se "
        "pisa las carpetas nuevas con la foto vieja que el mismo subio"
    )


def test_tambien_identifica_al_otro_aparato(client):
    # El caso normal: lo que llega de OTRO device se aplica siempre, y el
    # cliente decide leyendo este campo.
    pc = _uid("pc")
    _register(client, pc)
    movil = _uid("mov")
    _register(client, movil, "mobile")
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    assert client.post("/sync/link/join", json={
        "device_id": movil, "code": code, "device_type": "mobile"}).status_code == 200

    _push_carpetas(client, movil, device_type="android")

    r = client.get(f"/sync/pull/{pc}?full=true")
    cambios = [c for c in r.json()["changes"] if c["item_key"] == "all_folders"]
    assert cambios
    assert cambios[0]["last_device_id"] == movil


def test_el_pull_normal_tambien_lo_lleva(client):
    # No es solo cosa de full=true: el campo va siempre, para que el cliente no
    # tenga que saber en que modo pidio.
    pc = _uid("pc")
    _register(client, pc)
    movil = _uid("mov")
    _register(client, movil, "mobile")
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    client.post("/sync/link/join", json={
        "device_id": movil, "code": code, "device_type": "mobile"})
    _push_carpetas(client, movil, device_type="android")

    r = client.get(f"/sync/pull/{pc}")
    cambios = [c for c in r.json()["changes"] if c["item_key"] == "all_folders"]
    assert cambios
    assert cambios[0]["last_device_id"] == movil
