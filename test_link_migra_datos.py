"""
Vincular no puede abandonar lo que el dispositivo ya había subido.

Un dispositivo que sincroniza antes de vincularse NO tiene datos huérfanos:
tiene datos de la cuenta que el servidor le creó solo. `_assign_orphan_data`
solo rescata items con `user_id` vacío, así que esos se quedaban bajo la cuenta
vieja — invisibles para siempre: no los ve el otro dispositivo, no los cuenta
`/sync/publish` y no hay forma de borrarlos.

Reportado por el owner: analizó 4 tracks en el móvil antes de vincularlo y no
hubo manera ni de verlos desde el Mac ni de quitarlos. Publicar decía
`sobran=0` porque para esa cuenta, efectivamente, no sobraba nada.

El límite del arreglo es lo que más se prueba aquí: si a la cuenta vieja le
quedan OTROS dispositivos, sus datos NO se tocan. Llevárselos sería quitárselos
a un usuario que sigue usándolos.
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


def _register(client, device_id, device_type="mobile"):
    r = client.post("/sync/register",
                    json={"device_id": device_id, "device_type": device_type})
    assert r.status_code == 200
    return r.json().get("user_id")


def _push(client, device, keys, device_type="mobile"):
    r = client.post("/sync/push", json={
        "device_id": device, "device_type": device_type,
        "changes": [
            {"data_type": "analysis", "item_key": k,
             "payload": {"bpm": 128}, "deleted": False} for k in keys
        ],
    })
    assert r.status_code == 200


def _link(client, pc, device, device_type="mobile"):
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    r = client.post("/sync/link/join", json={
        "device_id": device, "code": code, "device_type": device_type})
    assert r.status_code == 200, r.text
    return r.json()


def _pull_keys(client, device, query=""):
    r = client.get(f"/sync/pull/{device}{query}")
    assert r.status_code == 200
    return {c["item_key"] for c in r.json()["changes"]}


def test_lo_que_el_movil_subio_antes_de_vincular_llega_al_pc(client):
    """El caso del owner: 4 tracks analizados en el movil antes de vincular."""
    pc = _uid("pc")
    _register(client, pc, "desktop")
    movil = _uid("mov")
    _register(client, movil, "mobile")

    suyos = [_uid("delmovil") for _ in range(4)]
    _push(client, movil, suyos)

    _link(client, pc, movil)

    recibidos = _pull_keys(client, pc, "?full=true")
    faltan = set(suyos) - recibidos
    assert not faltan, (
        f"{len(faltan)} de 4 tracks del movil siguen invisibles para el PC "
        "despues de vincular"
    )


def test_y_publicar_ya_puede_contarlos(client):
    """La consecuencia practica: si el PC no los ve, publicar no puede
    limpiarlos y se quedan en el movil para siempre."""
    pc = _uid("pc")
    _register(client, pc, "desktop")
    movil = _uid("mov")
    _register(client, movil, "mobile")
    _push(client, movil, [_uid("t") for _ in range(3)])
    _link(client, pc, movil)

    previa = client.post("/sync/publish", json={
        "device_id": pc, "device_type": "macos",
        "track_ids": [], "apply": False}).json()
    assert previa["would_delete"] == 3, (
        "publicar seguia sin ver los tracks del movil: sobran=0 era el sintoma"
    )


def test_NO_se_tocan_si_la_cuenta_vieja_conserva_dispositivos(client):
    """El limite del arreglo, y lo mas importante que se prueba aqui.

    Si el device venia de una cuenta con OTROS aparatos, sus items son de esa
    biblioteca. Llevarselos seria robarselos a alguien que sigue usandola.
    """
    # Cuenta A: dos dispositivos.
    pcA = _uid("pcA")
    _register(client, pcA, "desktop")
    movilA = _uid("movA")
    _register(client, movilA, "mobile")
    _link(client, pcA, movilA)
    deA = [_uid("deA") for _ in range(3)]
    _push(client, pcA, deA, device_type="desktop")

    # Cuenta B se lleva el movil de A.
    pcB = _uid("pcB")
    _register(client, pcB, "desktop")
    _link(client, pcB, movilA)

    # A conserva pcA -> sus items siguen siendo suyos.
    siguenEnA = _pull_keys(client, pcA, "?full=true")
    assert set(deA) <= siguenEnA, "se le han quitado los items a la cuenta vieja"

    # Y B no los ha heredado.
    enB = _pull_keys(client, pcB, "?full=true")
    assert not (set(deA) & enB), "la cuenta nueva ha heredado datos ajenos"


def test_revincular_al_mismo_usuario_no_mueve_nada(client):
    """`already_linked`: no hay cuenta vieja que abandonar."""
    pc = _uid("pc")
    _register(client, pc, "desktop")
    movil = _uid("mov")
    _register(client, movil, "mobile")
    _link(client, pc, movil)
    keys = [_uid("t") for _ in range(2)]
    _push(client, movil, keys)

    resp = _link(client, pc, movil)
    assert resp["already_linked"] is True
    assert resp["user_id"], "la respuesta tiene que decir a que cuenta pertenece"

    assert set(keys) <= _pull_keys(client, pc, "?full=true")

# NOTA: aqui habia un test que empujaba un item colectivo para comprobar que la
# migracion no lo toca. Se quito: un `cue_memory` lleva `user_id='__collective__'`
# y por diseño es visible para TODOS los usuarios, asi que al vivir en la BD que
# comparten los ficheros de test se colaba en los pulls de test_sync_pagination,
# que comparan el conjunto EXACTO de items recibidos. Rompia tres tests ajenos.
#
# La propiedad sigue garantizada y es evidente en el SQL: la migracion filtra
# por `WHERE user_id = <cuenta vieja>`, y '__collective__' no es ninguna cuenta.
