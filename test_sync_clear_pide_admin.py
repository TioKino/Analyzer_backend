"""`DELETE /sync/clear` pide la clave de admin SIEMPRE (2026-11-12, auditoría S1).

Hasta ese día, con `?device_id=X` borraba de verdad (sin lápida) todo lo que
ese aparato había subido, y no pedía ni la clave de admin ni el token del
aparato. La guarda del token vive en `_verify_sync_auth`, que saca el aparato
de la RUTA o del CUERPO (`_device_id_from_request`); aquí viajaba por la
QUERY. Reproducido con un aparato de token ya exigido: `pull` sin token → 401,
`clear?device_id=` sin token → 200 y sus filas borradas.

Ningún cliente ha llamado nunca a esta ruta. El borrado que pide un usuario es
`POST /sync/borrar-mis-datos`, con su token y con lápidas.

    pytest test_sync_clear_pide_admin.py -v
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


def _aparato_con_datos(client, n=5):
    """Un aparato registrado que ha subido [n] análisis. Devuelve su id."""
    device = f"pc-{uuid.uuid4().hex[:12]}"
    r = client.post("/sync/register", json={"device_id": device,
                                            "device_type": "windows"})
    assert r.status_code == 200, r.text
    token = r.json()["device_token"]
    r = client.post("/sync/push", headers={"X-Device-Token": token}, json={
        "device_id": device, "device_type": "windows",
        "changes": [{"data_type": "analysis", "item_key": f"t{i}",
                     "payload": {"bpm": 128},
                     "updated_at": "2026-11-01T00:00:00"} for i in range(n)],
    })
    assert r.status_code == 200, r.text
    return device


def _filas_de(device):
    import sync_endpoints
    return sync_endpoints._get_conn().execute(
        "SELECT COUNT(*) FROM sync_items WHERE last_device_id = ?",
        (device,)).fetchone()[0]


class TestElBorradoDeUnAparatoPideAdmin:
    def test_sin_clave_de_admin_no_borra_nada(self, client, monkeypatch):
        import sync_endpoints
        monkeypatch.setattr(sync_endpoints, "ADMIN_TOKEN", "clave-de-admin")
        device = _aparato_con_datos(client)
        assert _filas_de(device) == 5

        r = client.delete(f"/sync/clear?device_id={device}")

        assert r.status_code == 403
        assert _filas_de(device) == 5, (
            "se han borrado los datos de un aparato sin la clave de admin")

    def test_con_una_clave_mala_tampoco(self, client, monkeypatch):
        import sync_endpoints
        monkeypatch.setattr(sync_endpoints, "ADMIN_TOKEN", "clave-de-admin")
        device = _aparato_con_datos(client)

        r = client.delete(f"/sync/clear?device_id={device}",
                          headers={"X-Admin-Key": "otra-cosa"})

        assert r.status_code == 403
        assert _filas_de(device) == 5

    def test_el_token_del_aparato_no_vale_de_clave(self, client, monkeypatch):
        """Ni siquiera el dueño: su borrado es `/sync/borrar-mis-datos`."""
        import sync_endpoints
        monkeypatch.setattr(sync_endpoints, "ADMIN_TOKEN", "clave-de-admin")
        device = _aparato_con_datos(client)
        token = sync_endpoints._get_conn().execute(
            "SELECT device_token FROM user_devices WHERE device_id = ?",
            (device,)).fetchone()[0]

        r = client.delete(f"/sync/clear?device_id={device}",
                          headers={"X-Device-Token": token})

        assert r.status_code == 403
        assert _filas_de(device) == 5

    def test_con_la_clave_de_admin_si(self, client, monkeypatch):
        import sync_endpoints
        monkeypatch.setattr(sync_endpoints, "ADMIN_TOKEN", "clave-de-admin")
        device = _aparato_con_datos(client)
        otro = _aparato_con_datos(client, n=3)

        r = client.delete(f"/sync/clear?device_id={device}",
                          headers={"X-Admin-Key": "clave-de-admin"})

        assert r.status_code == 200
        assert r.json() == {"cleared": True, "scope": f"device:{device}"}
        assert _filas_de(device) == 0
        assert _filas_de(otro) == 3, "solo se borra lo del aparato pedido"

    def test_sin_admin_configurado_no_se_puede(self, client, monkeypatch):
        """Con `ADMIN_TOKEN` vacío no hay clave que valga: 403, no «pasa»."""
        import sync_endpoints
        monkeypatch.setattr(sync_endpoints, "ADMIN_TOKEN", "")
        device = _aparato_con_datos(client)

        r = client.delete(f"/sync/clear?device_id={device}",
                          headers={"X-Admin-Key": ""})

        assert r.status_code == 403
        assert _filas_de(device) == 5


class TestNingunaRutaDelSyncRecibeElAparatoPorLaQuery:
    """El `X-Device-Token` se pide en `_verify_sync_auth`, que saca el aparato
    de la ruta o del cuerpo. Una ruta que lo reciba por la QUERY se queda sin
    esa guarda sin que nada lo diga. La única que lo hace es `/sync/clear`,
    que no la necesita: pide la clave de admin."""

    SOLO_ADMIN = {"/sync/clear"}

    def test_el_aparato_va_en_la_ruta_o_en_el_cuerpo(self, client):
        import sync_endpoints

        por_query = []
        for ruta in sync_endpoints.sync_router.routes:
            dep = getattr(ruta, "dependant", None)
            if dep is None:
                continue
            nombres = {p.name for p in dep.query_params}
            if "device_id" in nombres and ruta.path not in self.SOLO_ADMIN:
                por_query.append(ruta.path)

        assert por_query == [], (
            f"{por_query} reciben `device_id` por la query: "
            "`_device_id_from_request` no lo ve y no se les pide el token "
            "del aparato. Pásalo por la ruta o por el cuerpo.")

    def test_y_la_que_lo_recibe_pide_admin(self, client):
        """La excepción de arriba solo vale mientras siga pidiendo la clave."""
        import inspect
        import sync_endpoints

        fuente = inspect.getsource(sync_endpoints.sync_clear)
        antes_de_borrar = fuente.split("DELETE FROM")[0]
        assert "if not admin_ok:" in antes_de_borrar
