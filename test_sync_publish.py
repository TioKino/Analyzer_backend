"""
`POST /sync/publish` — el escritorio declara su biblioteca completa.

El push manda CAMBIOS, así que la nube solo se entera de un borrado si alguien
se lo dice track a track. Cuando el usuario llega a su estado limpio por otro
camino —formateo, reinstalación, disco nuevo— no hay nada que decir: el
servidor conserva los miles de tracks viejos y se los sigue sirviendo a los
demás dispositivos. El owner lo describió así: *«tengo 5.000 en la nube, hago
limpieza, me quedo con 1.000, nunca más quiero volver a los 5.000 — ¿cómo sabe
la nube que esa nueva configuración es la buena?»*. Hoy no lo sabe. Esto es la
frase que faltaba.

Es la operación más destructiva de la app, así que lo que se fija aquí no es
solo que funcione: es que NO se dispare sola, que no la pueda pedir un móvil, y
que se pueda ver el daño antes de causarlo.
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


def _pc(client):
    d = _uid("pc")
    client.post("/sync/register", json={"device_id": d, "device_type": "desktop"})
    return d


def _push(client, device, keys, device_type="desktop"):
    r = client.post("/sync/push", json={
        "device_id": device,
        "device_type": device_type,
        "changes": [
            {"data_type": "analysis", "item_key": k,
             "payload": {"bpm": 128}, "deleted": False}
            for k in keys
        ],
    })
    assert r.status_code == 200
    return r


def _publish(client, device, ids, apply, device_type="macos"):
    return client.post("/sync/publish", json={
        "device_id": device,
        "device_type": device_type,
        "track_ids": ids,
        "apply": apply,
    })


def test_la_vista_previa_cuenta_pero_no_toca_nada(client):
    """El paso que hace defendible la operación: ver el daño antes."""
    pc = _pc(client)
    viejos = [_uid("viejo") for _ in range(7)]
    nuevos = [_uid("nuevo") for _ in range(3)]
    _push(client, pc, viejos + nuevos)

    r = _publish(client, pc, nuevos, apply=False)
    assert r.status_code == 200
    body = r.json()
    assert body["would_delete"] == 7
    assert body["declared"] == 3
    assert body["applied"] is False

    # Nada se ha tocado: la misma consulta vuelve a contar 7.
    again = _publish(client, pc, nuevos, apply=False).json()
    assert again["would_delete"] == 7


def test_al_aplicar_desaparecen_los_que_no_estan_en_la_lista(client):
    pc = _pc(client)
    viejos = [_uid("viejo") for _ in range(5)]
    supervivientes = [_uid("vivo") for _ in range(2)]
    _push(client, pc, viejos + supervivientes)

    r = _publish(client, pc, supervivientes, apply=True)
    assert r.json()["would_delete"] == 5
    assert r.json()["applied"] is True

    # Idempotente: publicar lo mismo otra vez ya no borra nada.
    assert _publish(client, pc, supervivientes, apply=False).json()["would_delete"] == 0


def test_el_borrado_llega_al_otro_dispositivo(client):
    """Publicar sin que el móvil se entere no serviría de nada."""
    pc = _pc(client)
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    movil = _uid("mov")
    assert client.post("/sync/link/join", json={
        "device_id": movil, "code": code, "device_type": "mobile"}).status_code == 200

    sobra = _uid("sobra")
    queda = _uid("queda")
    _push(client, pc, [sobra, queda])

    # El móvil ya los conoce.
    primero = client.get(f"/sync/pull/{movil}").json()["changes"]
    assert {c["item_key"] for c in primero} >= {sobra, queda}

    _publish(client, pc, [queda], apply=True)

    cambios = client.get(f"/sync/pull/{movil}").json()["changes"]
    borrados = {c["item_key"] for c in cambios if c["deleted"]}
    assert sobra in borrados, "el movil no se entera de que ese track sobra"
    assert queda not in borrados


def test_un_movil_no_puede_publicar(client):
    """La regla de siempre: el PC manda. El aparato que se reinstala y se queda
    sin espacio no le impone su estado a nadie."""
    movil = _uid("mov")
    client.post("/sync/register", json={"device_id": movil, "device_type": "mobile"})
    _push(client, movil, [_uid("t")], device_type="mobile")

    for tipo in ("android", "ios", "mobile", "unknown", ""):
        r = _publish(client, movil, [], apply=True, device_type=tipo)
        assert r.status_code == 403, f"{tipo} ha podido publicar"


def test_publicar_una_lista_vacia_no_es_un_atajo_para_vaciar(client):
    """Vaciar del todo es legítimo (formateo + publicar), pero tiene que pasar
    por la vista previa como todo lo demás: se cuenta y se ve."""
    pc = _pc(client)
    keys = [_uid("t") for _ in range(4)]
    _push(client, pc, keys)

    previa = _publish(client, pc, [], apply=False).json()
    assert previa["would_delete"] == 4
    assert previa["applied"] is False


def test_no_toca_el_blob_legacy_all_analysis(client):
    """`all_analysis` es la biblioteca entera en UNA fila (clientes < 2.9.3).
    Darla por borrada dejaría a ese cliente sin nada."""
    pc = _pc(client)
    r = client.post("/sync/push", json={
        "device_id": pc, "device_type": "desktop",
        "changes": [{"data_type": "analysis", "item_key": "all_analysis",
                     "payload": {"tracks": {"a": {"bpm": 120}}}, "deleted": False}],
    })
    assert r.status_code == 200

    assert _publish(client, pc, [], apply=False).json()["would_delete"] == 0


def test_no_toca_otros_tipos_de_dato(client):
    """Publicar habla de TRACKS. Sesiones, carpetas y colecciones tienen su
    propio camino y no pueden caerse de rebote."""
    pc = _pc(client)
    r = client.post("/sync/push", json={
        "device_id": pc, "device_type": "desktop",
        "changes": [
            {"data_type": "folder", "item_key": "all_folders",
             "payload": {"folders": []}, "deleted": False},
            {"data_type": "session", "item_key": "all_sessions",
             "payload": {"sessions": []}, "deleted": False},
        ],
    })
    assert r.status_code == 200

    assert _publish(client, pc, [], apply=False).json()["would_delete"] == 0


def test_no_toca_la_biblioteca_de_otro_usuario(client):
    """Multi-tenant: publicar es sobre lo tuyo."""
    mio = _pc(client)
    ajeno = _pc(client)  # otro user_id: registro independiente, sin vincular
    _push(client, mio, [_uid("mio")])
    suyos = [_uid("suyo") for _ in range(3)]
    _push(client, ajeno, suyos)

    _publish(client, mio, [], apply=True)

    # Los del otro siguen vivos: su propia publicación no encuentra nada raro.
    assert _publish(client, ajeno, suyos, apply=False).json()["would_delete"] == 0
