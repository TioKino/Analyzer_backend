"""`/sync/pull` recorta la página en SQL y trabaja fuera del event loop
(2026-11-13, auditoría S7).

Hasta ese día cada página leía TODAS las filas de la cuenta con su payload,
hacía `json.loads` de todas y solo entonces recortaba: el primer pull de un
móvil con 10.000 elementos, de 200 en 200, eran 50 peticiones que releían la
cuenta entera (6,9 s medidos), con el único proceso de Render parado mientras.
`test_sync_pagination.py` fija que no se pierde nada; esto, lo que cuesta.

    pytest test_el_pull_pagina_en_sql.py -v
"""
import ast
import inspect
import os
import tempfile
import textwrap
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


def _uid(t):
    return f"{t}-{uuid.uuid4().hex[:12]}"


def _pareja(client):
    pc, movil = _uid("pc"), _uid("mov")
    client.post("/sync/register", json={"device_id": pc, "device_type": "desktop"})
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    assert client.post("/sync/link/join", json={
        "device_id": movil, "code": code, "device_type": "mobile"}).status_code == 200
    return pc, movil


def _subir(client, pc, n):
    pre = _uid("t")
    r = client.post("/sync/push", json={
        "device_id": pc, "device_type": "desktop",
        "changes": [{"data_type": "analysis", "item_key": f"{pre}{i:05d}",
                     "payload": {"bpm": 120}, "deleted": False}
                    for i in range(n)]})
    assert r.json()["synced"] == n


class TestSoloSeDecodificaLaPagina:
    def test_una_pagina_de_200_no_lee_las_2000(self, client, monkeypatch):
        import sync_endpoints
        pc, movil = _pareja(client)
        _subir(client, pc, 2000)
        leidos = []
        real = sync_endpoints.json.loads

        def contar(s, *a, **k):
            leidos.append(1)
            return real(s, *a, **k)

        monkeypatch.setattr(sync_endpoints.json, "loads", contar)

        body = client.get(f"/sync/pull/{movil}?full=true&limit=200&offset=600").json()

        assert len(body["changes"]) == 200 and body["has_more"] is True
        assert len(leidos) <= 210, f"decodificó {len(leidos)} payloads para 200"

    def test_la_ultima_pagina_dice_que_no_quedan(self, client):
        pc, movil = _pareja(client)
        _subir(client, pc, 450)

        body = client.get(f"/sync/pull/{movil}?full=true&limit=200&offset=400").json()

        assert len(body["changes"]) == 50
        assert body["has_more"] is False

    def test_sin_full_lo_ya_visto_no_vuelve(self, client):
        pc, movil = _pareja(client)
        _subir(client, pc, 300)

        primero = client.get(f"/sync/pull/{movil}").json()
        segundo = client.get(f"/sync/pull/{movil}").json()

        assert primero["total"] >= 300
        assert segundo["total"] == 0


class TestFueraDelEventLoop:
    def test_la_ruta_manda_el_trabajo_a_un_hilo(self):
        import sync_endpoints
        arbol = ast.parse(textwrap.dedent(inspect.getsource(sync_endpoints.sync_pull)))
        llamadas = [n.func.id for n in ast.walk(arbol)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        assert "run_in_threadpool" in llamadas
        # Y nada de leer sync_items en el propio bucle.
        assert "FROM sync_items" not in inspect.getsource(sync_endpoints.sync_pull)

    def test_el_hilo_usa_su_propia_conexion(self):
        import sync_endpoints
        c = sync_endpoints._conexion_para_un_hilo()
        try:
            assert c is not sync_endpoints._get_conn()
        finally:
            if c is not sync_endpoints._get_conn():
                c.close()
