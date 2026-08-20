"""
Regresión: `full=true` tiene que devolverle a un dispositivo lo que subió ÉL.

"Restaurar desde la nube" (`CloudSyncService.resumeSync`) existe para un caso
concreto: borraste la biblioteca en local y la quieres de vuelta. Y era justo
ese caso el que no funcionaba.

`/sync/pull` filtraba SIEMPRE por `last_device_id != <yo>` — sensato en un pull
normal, donde devolverte lo que acabas de subir es puro eco. Pero con
`full=true` ese filtro convierte la restauración en un no-op para quien más la
necesita: en este proyecto el PC es la fuente de la verdad, así que casi todos
los items del usuario llevan SU device_id. Un Mac formateado pedía full=true
sobre 5.000 tracks y recibía 0, mientras el móvil los seguía enseñando porque
los tenía de un pull viejo. Las dos nubes en verde y cada aparato con una
biblioteca distinta.

Sin el fix, el primer test devuelve 0 de 40.
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


def _uid(tag: str) -> str:
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _push(client, device, keys, device_type="desktop"):
    changes = [
        {
            "data_type": "analysis",
            "item_key": k,
            "payload": {"bpm": 128},
            "deleted": False,
        }
        for k in keys
    ]
    r = client.post(
        "/sync/push",
        json={"device_id": device, "device_type": device_type, "changes": changes},
    )
    assert r.status_code == 200
    assert r.json()["synced"] == len(keys)


def _pull_keys(client, device, query=""):
    r = client.get(f"/sync/pull/{device}{query}")
    assert r.status_code == 200
    return {c["item_key"] for c in r.json()["changes"]}


def test_full_true_devuelve_lo_que_subio_el_propio_dispositivo(client):
    """El caso del Mac formateado: subo 40, los pierdo en local, pido full=true
    y me los tienen que devolver los 40."""
    pc = _uid("pc")
    client.post("/sync/register", json={"device_id": pc, "device_type": "desktop"})

    keys = [_uid("track") for _ in range(40)]
    _push(client, pc, keys)

    # El dispositivo se formatea: en local no queda nada, pero su device_id y
    # su registro en el servidor siguen siendo los mismos.
    recuperados = _pull_keys(client, pc, "?full=true")

    faltan = set(keys) - recuperados
    assert not faltan, (
        f"full=true dejo sin devolver {len(faltan)} de {len(keys)} items que "
        "habia subido este mismo dispositivo — la restauracion no restaura"
    )


def test_pull_normal_sigue_sin_devolver_el_eco(client):
    """El filtro anti-eco no se ha perdido: sin full, lo mío no vuelve."""
    pc = _uid("pc")
    client.post("/sync/register", json={"device_id": pc, "device_type": "desktop"})

    keys = [_uid("track") for _ in range(10)]
    _push(client, pc, keys)

    devueltos = _pull_keys(client, pc)
    assert not (set(keys) & devueltos), (
        "un pull normal ha devuelto items subidos por el propio dispositivo: "
        "eso es el eco que el filtro last_device_id existe para evitar"
    )


def test_full_true_sigue_trayendo_lo_del_otro_dispositivo(client):
    """Lo que ya funcionaba tiene que seguir funcionando: con full=true el
    móvil recibe lo del PC ademas de lo suyo."""
    pc, movil = _uid("pc"), _uid("mov")
    client.post("/sync/register", json={"device_id": pc, "device_type": "desktop"})
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    r = client.post(
        "/sync/link/join",
        json={"device_id": movil, "code": code, "device_type": "mobile"},
    )
    assert r.status_code == 200

    del_pc = [_uid("pctrack") for _ in range(5)]
    del_movil = [_uid("movtrack") for _ in range(3)]
    _push(client, pc, del_pc)
    _push(client, movil, del_movil, device_type="mobile")

    recuperados = _pull_keys(client, movil, "?full=true")
    assert set(del_pc) <= recuperados, "el movil ha dejado de recibir lo del PC"
    assert set(del_movil) <= recuperados, "el movil no recupera lo suyo propio"


def test_full_true_respeta_el_filtro_de_types(client):
    """La reparación estructural del móvil (full=true + types=folder) no puede
    empezar a arrastrar los miles de items de analisis: seria un OOM."""
    pc = _uid("pc")
    client.post("/sync/register", json={"device_id": pc, "device_type": "desktop"})

    analisis = [_uid("an") for _ in range(6)]
    _push(client, pc, analisis)
    carpeta = _uid("folder")
    r = client.post(
        "/sync/push",
        json={
            "device_id": pc,
            "device_type": "desktop",
            "changes": [
                {
                    "data_type": "folder",
                    "item_key": carpeta,
                    "payload": {"name": "Techno"},
                    "deleted": False,
                }
            ],
        },
    )
    assert r.status_code == 200

    solo_carpetas = _pull_keys(client, pc, "?full=true&types=folder")
    assert carpeta in solo_carpetas
    assert not (set(analisis) & solo_carpetas), (
        "types=folder ha dejado pasar items de analisis"
    )
