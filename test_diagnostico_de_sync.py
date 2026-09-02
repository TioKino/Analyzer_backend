"""«El sync con el móvil ha dejado de funcionar» no se podía contestar.

Reportado por el owner tras la 2.9.10: cambios hechos en el Mac que no llegan
al móvil. Las causas posibles piden acciones OPUESTAS —

  - el móvil no está vinculado    -> rehacer la vinculación
  - el móvil exige token y lo perdió -> NO endurecer nada, es un cliente legítimo
  - el móvil tiene items esperando -> el fallo está en el cliente, no en el server
  - el móvil está al día           -> lo que falla es aplicar, no sincronizar

— y no había ningún sitio donde mirar cuál era. `/sync/status` cuenta items
globales, `/sync/admin/users` listaba los aparatos sin decir si su token estaba
endurecido, y `token_seen_at` —la señal que decide entre 200 y un 401
permanente— no salía por ninguna API.

Es la misma lección que ya costó un día entero: cuando el cliente y el servidor
cuentan historias distintas de la MISMA tanda, hay que poder ponerlas una al
lado de la otra. Por eso `siblings` trae lo que subió cada aparato y cuándo.

    pytest test_diagnostico_de_sync.py -v
"""

import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

_ADMIN = "test-admin-token-diag"


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.pop("RENDER", None)
    import main
    import sync_endpoints

    sync_endpoints.SYNC_AUTH_SECRET = ""
    sync_endpoints.ADMIN_TOKEN = _ADMIN
    return TestClient(main.app)


def _uid(tag):
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _admin(client, path):
    r = client.get(path, headers={"Authorization": f"Bearer {_ADMIN}"})
    assert r.status_code == 200, r.text
    return r.json()


def _register(client, device_id, device_type):
    r = client.post("/sync/register",
                    json={"device_id": device_id, "device_type": device_type})
    assert r.status_code == 200, r.text
    return r.json()


def _push(client, device, keys, device_type="macos-dmg"):
    r = client.post("/sync/push", json={
        "device_id": device, "device_type": device_type,
        "changes": [
            {"data_type": "analysis", "item_key": k,
             "payload": {"bpm": 128}, "deleted": False} for k in keys
        ],
    })
    assert r.status_code == 200, r.text


def _link(client, anfitrion, invitado, device_type="ios"):
    code = client.post("/sync/link/generate",
                       json={"device_id": anfitrion}).json()["code"]
    r = client.post("/sync/link/join", json={
        "device_id": invitado, "code": code, "device_type": device_type})
    assert r.status_code == 200, r.text


# ============================================================================
# LOS CUATRO VEREDICTOS
# ============================================================================

def test_un_aparato_que_no_existe_lo_dice(client):
    d = _admin(client, "/sync/admin/device/no-existe-jamas")
    assert d["registered"] is False
    assert d["verdict"] == "not_registered"
    # El motivo, no solo la etiqueta: es un 403, no un 401, y se arregla
    # llamando a /sync/register.
    assert "403" in d["note"]


def test_un_aparato_SOLO_en_su_cuenta_no_tiene_con_quien_sincronizar(client):
    solo = _uid("solo")
    _register(client, solo, "ios")
    d = _admin(client, f"/sync/admin/device/{solo}")
    assert d["verdict"] == "alone"
    assert d["siblings"] == []


def test_EL_CASO_el_mac_subio_y_el_movil_lo_tiene_esperando(client):
    mac, movil = _uid("mac"), _uid("movil")
    _register(client, mac, "macos-dmg")
    _register(client, movil, "ios")
    _link(client, mac, movil)
    _push(client, mac, ["t1", "t2", "t3"])

    d = _admin(client, f"/sync/admin/device/{movil}")
    assert d["verdict"] == "pending"
    assert d["pending_for_this_device"] == 3
    # Y el otro lado de la misma tanda, en la misma respuesta.
    hermano = d["siblings"][0]
    assert hermano["device_id"] == mac
    assert hermano["items_pushed"] == 3
    assert hermano["last_push"]


def test_cuando_el_movil_ya_se_lo_bajo_el_veredicto_cambia(client):
    """Si tras el pull sigue diciendo `pending`, el que miente es el contador.

    Este es el test que separa «no le llega» de «le llega y no lo aplica», que
    es justo la bifurcación que no se podía hacer desde fuera.
    """
    mac, movil = _uid("mac"), _uid("movil")
    _register(client, mac, "macos-dmg")
    _register(client, movil, "ios")
    _link(client, mac, movil)
    _push(client, mac, ["a", "b"])

    assert _admin(client, f"/sync/admin/device/{movil}")["pending_for_this_device"] == 2
    assert client.get(f"/sync/pull/{movil}").status_code == 200

    d = _admin(client, f"/sync/admin/device/{movil}")
    assert d["pending_for_this_device"] == 0
    assert d["verdict"] == "nothing_pending"


def test_un_item_MODIFICADO_vuelve_a_contar_como_pendiente(client):
    """No basta con «lo vio una vez»: si el Mac lo cambia, vuelve a deber."""
    mac, movil = _uid("mac"), _uid("movil")
    _register(client, mac, "macos-dmg")
    _register(client, movil, "ios")
    _link(client, mac, movil)
    _push(client, mac, ["x"])
    client.get(f"/sync/pull/{movil}")
    assert _admin(client, f"/sync/admin/device/{movil}")["pending_for_this_device"] == 0

    client.post("/sync/push", json={
        "device_id": mac, "device_type": "macos-dmg",
        "changes": [{"data_type": "analysis", "item_key": "x",
                     "payload": {"bpm": 174}, "deleted": False}],
    })
    assert _admin(client, f"/sync/admin/device/{movil}")["pending_for_this_device"] == 1


# ============================================================================
# EL TOKEN: LA CAUSA QUE DEJA A UN APARATO FUERA SIN AVISAR
# ============================================================================

def test_el_token_endurecido_se_ve_desde_fuera(client):
    """`token_seen_at` es la señal que convierte una petición sin token en un
    401 permanente. Estaba en la BD y no salía por ninguna API, así que la
    única causa irrecuperable era también la única invisible."""
    mac, movil = _uid("mac"), _uid("movil")
    _register(client, mac, "macos-dmg")
    token = _register(client, movil, "ios")["device_token"]
    _link(client, mac, movil)

    antes = _admin(client, f"/sync/admin/device/{movil}")
    assert antes["has_token"] is True
    assert antes["token_enforced"] is False, "aún no lo ha mandado nunca"

    # El móvil manda su token una vez: a partir de aquí se le exige. El sellado
    # vive dentro de `_verify_sync_auth`, que con `SYNC_AUTH_SECRET` vacío se
    # sale antes de mirar nada — así que este tramo necesita el HMAC de verdad.
    import hashlib
    import hmac as _h

    import sync_endpoints

    secreto = "secreto-de-test-para-firmar"
    sync_endpoints.SYNC_AUTH_SECRET = secreto
    try:
        firma = _h.new(secreto.encode(), b"", hashlib.sha256).hexdigest()
        r = client.get(f"/sync/pull/{movil}", headers={
            "X-Device-Token": token, "X-Signature": firma})
        assert r.status_code == 200, r.text
    finally:
        sync_endpoints.SYNC_AUTH_SECRET = ""

    despues = _admin(client, f"/sync/admin/device/{movil}")
    assert despues["token_enforced"] is True
    assert despues["token_seen_at"]
    assert despues["verdict"] == "token_enforced"


def test_el_diagnostico_NUNCA_devuelve_el_token(client):
    """Entregarlo aquí anularía el mecanismo entero: el token se emite UNA vez
    justamente para que tenerlo demuestre ser el aparato."""
    movil = _uid("movil")
    _register(client, movil, "ios")
    cuerpo = client.get(f"/sync/admin/device/{movil}",
                        headers={"Authorization": f"Bearer {_ADMIN}"}).text
    assert "device_token" not in cuerpo


def test_el_listado_de_usuarios_tambien_dice_si_el_token_esta_exigido(client):
    """Sin esto había que pedir el detalle aparato por aparato para saber
    cuáles estaban en riesgo de quedarse fuera."""
    d = _admin(client, "/sync/admin/users")
    algun_device = [dev for u in d["users"] for dev in u["devices"]]
    assert algun_device, "el fixture ya registró aparatos"
    for dev in algun_device:
        assert "has_token" in dev
        assert "token_enforced" in dev


# ============================================================================
# AUTH
# ============================================================================

def test_sin_token_admin_no_se_diagnostica_a_nadie(client):
    """El diagnóstico enseña device_ids y cuentas de items de TODOS."""
    assert client.get("/sync/admin/device/lo-que-sea").status_code == 401


def test_el_router_admin_de_sync_ESTA_MONTADO(client):
    """El bug de fondo: `admin_sync_router` estaba DEFINIDO y SIN MONTAR.

    Cinco endpoints escritos y documentados, cinco 404 en producción. Nadie lo
    vio porque el código está ahí y se lee perfectamente — se buscaba la
    herramienta, se encontraba, y se concluía que el problema era otro.

    Este test no mira una ruta: las mira TODAS las del router, para que añadir
    una sexta y olvidar el `include_router` no vuelva a pasar desapercibido.
    """
    import main
    import sync_endpoints

    montadas = {getattr(r, 'path', None) for r in main.app.routes}
    # `.routes` ya trae el prefijo aplicado: concatenarlo daba
    # `/sync/admin/sync/admin/...` y el test fallaba por su propia cuenta.
    declaradas = {r.path for r in sync_endpoints.admin_sync_router.routes}
    assert declaradas, 'el router no declara ninguna ruta'
    assert declaradas <= montadas, f'sin montar: {sorted(declaradas - montadas)}'
