"""Las notas públicas tienen tope, y se pueden moderar (2026-11-12, auditoría S2).

Una nota de la comunidad es TEXTO PÚBLICO: la ve cualquiera que tenga ese
tema. Hasta ese día `POST /community/notes` no tenía ni el guard de
procedencia de los votos ni tope ninguno, y no había forma de ver todas las
notas ni de quitar una (el panel no las nombraba). Reproducido contra el
servidor en modo producción: una nota de «Carl Cox» con un enlace de spam,
desde un aparato inventado, 200.

Lo que ata este test: el tope por IP y hora, que el guard de identidades
también vale para las notas (en `test_community_sybil.py`), y las dos rutas de
moderación del panel, con su credencial.

Sigue sin pedirse credencial para ESCRIBIR una nota: los clientes publicados
no mandan el `X-Device-Token` ahí, y exigirlo (o apagar las notas) es una
decisión del owner (`PENDING.md`).

    pytest test_las_notas_se_moderan.py -v
"""
import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_mod():
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.pop("RENDER", None)
    import main

    return main


@pytest.fixture(scope="module")
def client(app_mod):
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def _tope_limpio(app_mod):
    """El tope vive en memoria y es por IP: cada test empieza con el suyo."""
    import routes.community as rc
    from validation import RateLimiter

    rc._limite_de_notas = RateLimiter(max_requests=rc.NOTES_MAX_PER_HOUR,
                                      window_seconds=3600)
    yield


def _fp() -> str:
    return uuid.uuid4().hex


def _nota(client, fp, texto, aparato="un-aparato", ip="203.0.113.50",
          nombre="DJ", tipo="general"):
    return client.post(
        "/community/notes",
        json={"fingerprint": fp, "device_id": aparato, "note_text": texto,
              "display_name": nombre, "note_type": tipo},
        headers={"X-Forwarded-For": ip},
    )


class TestElTopePorHora:
    def test_una_persona_escribe_sus_notas_sin_notarlo(self, client):
        for i in range(5):
            r = _nota(client, _fp(), f"mezcla bien con el siguiente ({i})",
                      ip="203.0.113.51")
            assert r.status_code == 200, r.text

    def test_un_bucle_se_corta(self, client):
        import routes.community as rc

        aceptadas = 0
        for i in range(rc.NOTES_MAX_PER_HOUR + 5):
            r = _nota(client, _fp(), f"spam {i}", ip="203.0.113.52")
            if r.status_code == 200:
                aceptadas += 1
            else:
                assert r.status_code == 429
                assert r.headers.get("Retry-After") == "3600"
                break
        assert aceptadas == rc.NOTES_MAX_PER_HOUR

    def test_el_tope_es_de_cada_ip(self, client):
        import routes.community as rc

        for i in range(rc.NOTES_MAX_PER_HOUR):
            assert _nota(client, _fp(), f"n{i}", ip="203.0.113.53").status_code == 200
        assert _nota(client, _fp(), "una más", ip="203.0.113.53").status_code == 429

        assert _nota(client, _fp(), "otra persona", ip="203.0.113.54").status_code == 200


class TestLoQueSeGuarda:
    def test_un_tipo_inventado_se_queda_en_general(self, client, app_mod):
        fp = _fp()
        assert _nota(client, fp, "hola", tipo="<script>", ip="203.0.113.55").status_code == 200

        notas = app_mod.db.get_community_notes(fp)
        assert [n["note_type"] for n in notas] == ["general"]

    def test_sin_huella_o_sin_aparato_no_entra(self, client):
        assert _nota(client, "", "hola", ip="203.0.113.56").status_code == 400
        assert _nota(client, _fp(), "hola", aparato="", ip="203.0.113.56").status_code == 400
        assert _nota(client, "a" * 200, "hola", ip="203.0.113.56").status_code == 400

    def test_un_fallo_interno_no_cuenta_sus_tripas(self, client, app_mod, monkeypatch):
        def _revienta(**_k):
            raise RuntimeError("ruta/secreta/analysis.db: disk I/O error")

        monkeypatch.setattr(app_mod.db, "save_community_note", _revienta)

        r = _nota(client, _fp(), "hola", ip="203.0.113.57")

        assert r.status_code == 500
        assert "secreta" not in r.text


class TestModerarDesdeElPanel:
    def test_sin_credencial_no_se_listan_ni_se_borran(self, client, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")

        assert client.get("/admin/notas").status_code == 401
        assert client.delete("/admin/notas/1").status_code == 401

    def test_se_listan_todas_la_mas_nueva_primero(self, client, app_mod, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")
        admin = {"X-Admin-Secret": "tok-secreto"}
        antes = client.get("/admin/notas", headers=admin).json()["total"]
        fp1, fp2 = _fp(), _fp()
        assert _nota(client, fp1, "la primera", nombre="Ana", ip="203.0.113.58").status_code == 200
        assert _nota(client, fp2, "la segunda", nombre="Luis", ip="203.0.113.58").status_code == 200

        r = client.get("/admin/notas?limite=2", headers=admin)

        assert r.status_code == 200
        datos = r.json()
        assert datos["total"] == antes + 2
        assert [n["note_text"] for n in datos["notas"]] == ["la segunda", "la primera"]
        # Para moderar hace falta saber de quién es y en qué tema está.
        assert datos["notas"][0]["fingerprint"] == fp2
        assert datos["notas"][0]["device_id"] == "un-aparato"
        assert datos["notas"][0]["display_name"] == "Luis"

    def test_borrar_una_la_quita_para_todos_y_se_lleva_sus_votos(
            self, client, app_mod, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")
        admin = {"X-Admin-Secret": "tok-secreto"}
        fp = _fp()
        mala = _nota(client, fp, "COMPRA SEGUIDORES en http://spam.example",
                     aparato="spammer", nombre="Carl Cox",
                     ip="203.0.113.59").json()["note_id"]
        buena = _nota(client, fp, "el breakdown es largo", aparato="un-dj",
                      ip="203.0.113.59").json()["note_id"]
        assert client.post(f"/community/notes/{mala}/upvote?device_id=otro").status_code == 200

        r = client.delete(f"/admin/notas/{mala}", headers=admin)

        assert r.status_code == 200
        assert r.json() == {"id": mala, "borrada": True}
        quedan = client.get(f"/community/notes/{fp}").json()["notes"]
        assert [n["id"] for n in quedan] == [buena]
        conn = app_mod.db._open_conn()
        try:
            votos = conn.execute(
                "SELECT COUNT(*) FROM community_note_upvotes WHERE note_id = ?",
                (mala,)).fetchone()[0]
        finally:
            conn.close()
        assert votos == 0

    def test_borrar_una_que_no_existe_es_404(self, client, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")

        r = client.delete("/admin/notas/99999999",
                          headers={"X-Admin-Secret": "tok-secreto"})

        assert r.status_code == 404

    def test_el_listado_se_pagina(self, client, app_mod, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")
        admin = {"X-Admin-Secret": "tok-secreto"}
        for i in range(3):
            assert _nota(client, _fp(), f"pagina {i}", ip="203.0.113.60").status_code == 200

        una = client.get("/admin/notas?limite=1", headers=admin).json()["notas"]
        otra = client.get("/admin/notas?limite=1&desde=1", headers=admin).json()["notas"]

        assert len(una) == 1 and len(otra) == 1
        assert una[0]["id"] > otra[0]["id"]
