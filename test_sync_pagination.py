"""
Regresión de SYNC-01 — `/sync/pull` no puede perder items al paginar.

Los dos modos de pull usan cursores distintos:

  full=true  → sin filtro de device_seen; la lista es estable entre páginas y
               `offset` es el cursor. Es la ruta que usa hoy el móvil
               (_paginatedPullMobile solo se activa con _forceFullPull), y
               está verificada correcta: 5000 tracks → 5000 entregados.

  full=false → `device_seen` YA es el cursor (cada página entregada se marca
               como vista). Honrar además el `offset` suma dos avances y salta
               `limit` items por página: medido, 5000 tracks paginados de 200
               en 200 entregaban 2600.

Ningún llamador combina hoy full=false con offset, así que el segundo caso es
blindaje, no un fallo en producción. Estos tests fijan AMBOS contratos para que
un cambio futuro en el cliente no abra el agujero. Recorren el flujo real
(register → link → push → pull) porque el bug nace de la interacción
device_seen + offset y un test unitario del slice no lo cazaría.
"""

import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # OJO: `sync_endpoints._DB_PATH` se evalúa en el IMPORT del módulo, así que
    # si otro fichero de test lo importó antes, este env var llega tarde y se
    # comparte `/data/sync.db`. Por eso los tests de abajo NO asumen una BD
    # limpia: usan device_ids e item_keys únicos por ejecución (`_uid`).
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.pop("RENDER", None)
    import main
    import sync_endpoints

    sync_endpoints.SYNC_AUTH_SECRET = ""  # dev mode: sin firma
    return TestClient(main.app)


def _uid(tag: str) -> str:
    """Identificador único por ejecución: los tests comparten BD con otros
    módulos y con ejecuciones anteriores, y `device_seen` es persistente."""
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _link(client, pc, mobile):
    client.post("/sync/register", json={"device_id": pc, "device_type": "desktop"})
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    r = client.post(
        "/sync/link/join",
        json={"device_id": mobile, "code": code, "device_type": "mobile"},
    )
    assert r.status_code == 200


def _push(client, pc, n, prefix):
    changes = [
        {
            "data_type": "analysis",
            "item_key": f"{prefix}{i:05d}",
            "payload": {"bpm": 120 + (i % 40)},
            "deleted": False,
        }
        for i in range(n)
    ]
    r = client.post(
        "/sync/push",
        json={"device_id": pc, "device_type": "desktop", "changes": changes},
    )
    assert r.json()["synced"] == n


def _drain(client, device, page_size, advance_offset):
    """Recorre todas las páginas. `advance_offset` emula al cliente viejo."""
    got, offset, pages = set(), 0, 0
    while True:
        r = client.get(f"/sync/pull/{device}?limit={page_size}&offset={offset}")
        assert r.status_code == 200
        body = r.json()
        pages += 1
        for ch in body["changes"]:
            got.add(ch["item_key"])
        if not body["changes"] or not body["has_more"]:
            break
        if advance_offset:
            offset += page_size
        assert pages < 200, "bucle infinito en la paginación"
    return got, pages


def test_ruta_real_del_movil_entrega_la_biblioteca_entera(client):
    """El camino que recorre hoy _paginatedPullMobile: full=true + offset
    incremental. Es el que importa; debe entregar los 5000 sin pérdidas."""
    pc, mobile = _uid("pc"), _uid("mov")
    _link(client, pc, mobile)
    n = 5000
    pre = _uid("tr")
    _push(client, pc, n, pre)

    got, offset, pages = set(), 0, 0
    while True:
        body = client.get(
            f"/sync/pull/{mobile}?full=true&limit=200&offset={offset}"
        ).json()
        pages += 1
        for ch in body["changes"]:
            got.add(ch["item_key"])
        if not body["changes"] or not body["has_more"]:
            break
        offset += 200
        assert pages < 200

    assert got == {f"{pre}{i:05d}" for i in range(n)}, (
        f"faltan {n - len(got)} de {n} items tras {pages} páginas"
    )


def test_paginar_sin_full_no_pierde_items(client):
    """Blindaje: con full=false, device_seen es el cursor y el offset debe
    ignorarse. Nadie combina hoy ambos, pero si un cambio futuro lo hiciera
    se perderían `limit` items por página (medido: 2600 de 5000)."""
    pc, mobile = _uid("pc"), _uid("mov")
    _link(client, pc, mobile)
    n = 1000
    pre = _uid("tb")
    _push(client, pc, n, pre)

    got, _ = _drain(client, mobile, 200, advance_offset=True)

    assert got == {f"{pre}{i:05d}" for i in range(n)}


def test_ultima_pagina_no_reenvia_desde_el_principio(client):
    """Cuando quedan menos items que `limit`, el recorte debía aplicarse
    igual: antes el `if limit < len(all_items)` lo saltaba y se devolvía
    la lista entera desde el índice 0."""
    pc, mobile = _uid("pc"), _uid("mov")
    _link(client, pc, mobile)
    pre = _uid("tc")
    _push(client, pc, 250, pre)

    first = client.get(f"/sync/pull/{mobile}?limit=200&offset=0").json()
    assert len(first["changes"]) == 200
    assert first["has_more"] is True

    second = client.get(f"/sync/pull/{mobile}?limit=200&offset=0").json()
    assert second["has_more"] is False

    keys = {c["item_key"] for c in first["changes"]} | {
        c["item_key"] for c in second["changes"]
    }
    # Filtramos por prefijo: la BD la comparten otros módulos de test y puede
    # traer items colectivos ajenos.
    assert {k for k in keys if k.startswith(pre)} == {
        f"{pre}{i:05d}" for i in range(250)
    }


def test_full_true_sigue_respetando_offset(client):
    """En modo full no hay filtro de device_seen, así que el offset ES el
    cursor y debe seguir aplicándose."""
    pc, mobile = _uid("pc"), _uid("mov")
    _link(client, pc, mobile)
    pre = _uid("td")
    _push(client, pc, 300, pre)

    p1 = client.get(f"/sync/pull/{mobile}?full=true&limit=100&offset=0").json()
    p2 = client.get(f"/sync/pull/{mobile}?full=true&limit=100&offset=100").json()
    p3 = client.get(f"/sync/pull/{mobile}?full=true&limit=100&offset=200").json()

    keys = set()
    for p in (p1, p2, p3):
        keys |= {c["item_key"] for c in p["changes"]}
    assert {k for k in keys if k.startswith(pre)} == {
        f"{pre}{i:05d}" for i in range(300)
    }, "full=true con offset debe cubrir los 300 sin solapar ni saltar"


def test_sin_limit_devuelve_todo(client):
    """Compat con clientes viejos que no paginan (limit=0)."""
    pc, mobile = _uid("pc"), _uid("mov")
    _link(client, pc, mobile)
    pre = _uid("te")
    _push(client, pc, 120, pre)

    body = client.get(f"/sync/pull/{mobile}").json()
    mine = [c for c in body["changes"] if c["item_key"].startswith(pre)]
    assert len(mine) == 120
    assert body["has_more"] is False
