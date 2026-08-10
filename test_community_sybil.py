"""
Mitigación SEC-12 — el consenso comunitario no debe fabricarse desde una IP.

Los endpoints de /community/* reciben `device_id` como campo plano del cuerpo,
sin autenticar. Auditoría 2026-08-09: mandar tres ids inventados produce un
`consensus_3` (prioridad 80 en analysis_ranking) que gana al motor local (50),
y con diez un `consensus_10` (95). Un solo actor podía dictar el BPM, la
tonalidad, el género y el beat-grid "de la comunidad" para cualquier track.

El arreglo de fondo (derivar el device_id de la identidad de sync) toca cliente
y servidor a la vez. Estos tests cubren la mitigación que sí se despliega sola:
cortar cuando una misma IP vota el mismo track bajo demasiadas identidades.
"""

import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_mod():
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("PREVIEWS_DIR", tempfile.mkdtemp())
    os.environ.setdefault("ARTWORK_CACHE_DIR", tempfile.mkdtemp())
    os.environ.pop("RENDER", None)
    import main

    return main


@pytest.fixture(scope="module")
def client(app_mod):
    return TestClient(app_mod.app)


def _fp():
    return uuid.uuid4().hex


def _vote(client, fp, device_id, value="peak_time", ip="203.0.113.7"):
    return client.post(
        "/community/override",
        json={
            "fingerprint": fp,
            "device_id": device_id,
            "field": "track_type",
            "value": value,
        },
        headers={"X-Forwarded-For": ip},
    )


class TestGuardDeProcedencia:
    def test_una_ip_no_puede_fabricar_consenso_con_ids_inventados(self, client):
        """El ataque directo: mismo curl, ids distintos, hasta llegar a
        consensus_3. Debe cortarse antes."""
        fp = _fp()
        aceptados = 0
        for i in range(8):
            r = _vote(client, fp, f"inventado-{i}")
            if r.status_code == 200:
                aceptados += 1
            else:
                assert r.status_code == 429, f"esperaba 429, llegó {r.status_code}"
                break
        assert aceptados <= 3, (
            f"{aceptados} identidades aceptadas desde una IP: suficiente para "
            f"fabricar consensus_3"
        )

    def test_un_hogar_con_varios_dispositivos_sigue_pudiendo_votar(self, client):
        """No podemos romper el caso legítimo: el mismo DJ con desktop + móvil
        detrás del mismo router."""
        fp = _fp()
        for dev in ("desktop-de-casa", "movil-de-casa"):
            r = _vote(client, fp, dev)
            assert r.status_code == 200, f"{dev} debería poder votar"

    def test_el_limite_es_por_track_no_global(self, client):
        """Agotar el cupo en un track no puede dejar al usuario sin votar en
        otros: el contador va por (ip, fingerprint)."""
        fp_a, fp_b = _fp(), _fp()
        for i in range(5):
            _vote(client, fp_a, f"dev-a-{i}")
        r = _vote(client, fp_b, "dev-b-0")
        assert r.status_code == 200, "otro track debe empezar con el cupo limpio"

    def test_ips_distintas_no_comparten_cupo(self, client):
        """Dos DJs reales en sitios distintos votando el mismo track."""
        fp = _fp()
        for i in range(3):
            _vote(client, fp, f"casa-{i}", ip="203.0.113.10")
        r = _vote(client, fp, "otro-dj", ip="198.51.100.22")
        assert r.status_code == 200

    def test_revotar_con_el_mismo_id_no_consume_cupo(self, client):
        """Cambiar de opinión es legítimo y no debe agotar el límite."""
        fp = _fp()
        for _ in range(6):
            r = _vote(client, fp, "mi-unico-device", value="warmup")
            assert r.status_code == 200, "revotar con el mismo id nunca se corta"


class TestQueNingunaViaSeSalteElGuard:
    """El guard no vale de nada si queda una puerta sin cerrar: basta apuntar
    el mismo curl a otra ruta que escriba el mismo consenso."""

    def test_la_ruta_legacy_de_track_type_no_es_un_atajo(self, client):
        """REGRESIÓN (auditoría 2026-08-10): `/community/track-type` delegaba en
        el endpoint genérico con `http=None`, y el genérico solo llama al guard
        si recibe la Request. O sea que la mitigación se saltaba entera
        cambiando la URL. Escribe en la MISMA tabla que `/community/override`."""
        fp = _fp()
        aceptados = 0
        for i in range(8):
            r = client.post(
                "/community/track-type",
                json={"fingerprint": fp, "device_id": f"legacy-{i}",
                      "track_type": "peak_time"},
                headers={"X-Forwarded-For": "203.0.113.31"},
            )
            if r.status_code == 200:
                aceptados += 1
            else:
                assert r.status_code == 429
                break
        assert aceptados <= 3, (
            f"{aceptados} identidades por la ruta legacy: sigue siendo un bypass"
        )

    def test_las_valoraciones_tambien_estan_capadas(self, client):
        """La media de rating se muestra a todo el mundo; con ids inventados se
        empuja a donde se quiera."""
        fp = _fp()
        aceptados = 0
        for i in range(8):
            r = client.post(
                "/community/rate",
                json={"fingerprint": fp, "device_id": f"falso-{i}", "rating": 5},
                headers={"X-Forwarded-For": "203.0.113.32"},
            )
            if r.status_code == 200:
                aceptados += 1
            else:
                assert r.status_code == 429
                break
        assert aceptados <= 3, f"{aceptados} identidades valorando desde una IP"

    def test_los_cues_comunitarios_tambien(self, client):
        """Las zonas se devuelven con `dj_count` y `confidence`: es consenso
        agregado igual que un override, y se fabricaba igual de fácil."""
        fp = _fp()
        aceptados = 0
        for i in range(8):
            r = client.post(
                "/community-cues",
                json={"fingerprint": fp, "device_id": f"cue-falso-{i}",
                      "cues": [{"type": "drop", "position_ms": 60000}]},
                headers={"X-Forwarded-For": "203.0.113.33"},
            )
            if r.status_code == 200:
                aceptados += 1
            else:
                assert r.status_code == 429
                break
        assert aceptados <= 3, f"{aceptados} identidades subiendo cues desde una IP"

    def test_un_dj_real_sigue_pudiendo_subir_sus_cues(self, client):
        r = client.post(
            "/community-cues",
            json={"fingerprint": _fp(), "device_id": "mi-desktop",
                  "cues": [{"type": "mixIn", "position_ms": 15000}]},
            headers={"X-Forwarded-For": "198.51.100.77"},
        )
        assert r.status_code == 200
        assert r.json().get("status") != "error"


class TestUpvotesDeNotas:
    """`community_notes.upvotes` era un contador ciego: el endpoint no recibía
    identidad y hacía `upvotes = upvotes + 1`. Un bucle de curl colocaba
    cualquier nota la primera."""

    def _nota(self, client, texto="Sube el filtro en el break"):
        fp = _fp()
        r = client.post("/community/notes", json={
            "fingerprint": fp, "device_id": "autor", "note_text": texto,
        })
        assert r.status_code == 200, r.text
        return fp, r.json()["note_id"]

    def test_el_mismo_votante_no_suma_dos_veces(self, client):
        _, note_id = self._nota(client)
        primero = client.post(f"/community/notes/{note_id}/upvote",
                              params={"device_id": "dj-1"})
        segundo = client.post(f"/community/notes/{note_id}/upvote",
                              params={"device_id": "dj-1"})
        assert primero.json()["counted"] is True
        assert segundo.json()["counted"] is False, "repetir sumó otra vez"
        assert segundo.json()["upvotes"] == 1

    def test_repetir_no_es_un_error(self, client):
        """La UI puede reintentar (red mala, doble tap): idempotente, no 4xx."""
        _, note_id = self._nota(client)
        for _ in range(4):
            r = client.post(f"/community/notes/{note_id}/upvote",
                            params={"device_id": "dj-nervioso"})
            assert r.status_code == 200

    def test_votantes_distintos_si_suman(self, client):
        _, note_id = self._nota(client)
        for dev in ("dj-a", "dj-b", "dj-c"):
            client.post(f"/community/notes/{note_id}/upvote", params={"device_id": dev})
        r = client.post(f"/community/notes/{note_id}/upvote", params={"device_id": "dj-d"})
        assert r.json()["upvotes"] == 4

    def test_sin_device_id_cae_a_la_ip_y_tampoco_se_repite(self, client):
        """El cliente actual llama sin cuerpo. Grano peor (un hogar cuenta como
        uno) pero el bucle de curl deja de funcionar, que es lo que importaba."""
        _, note_id = self._nota(client)
        for _ in range(5):
            r = client.post(f"/community/notes/{note_id}/upvote",
                            headers={"X-Forwarded-For": "203.0.113.44"})
            assert r.status_code == 200
        assert r.json()["upvotes"] == 1


class TestRegistroDeProcedencia:
    def test_cuenta_identidades_distintas_por_ip_y_track(self, app_mod):
        db = app_mod.db
        fp = _fp()
        h = db.hash_ip("192.0.2.55")
        assert db.register_vote_source(h, fp, "d1") == 1
        assert db.register_vote_source(h, fp, "d2") == 2
        assert db.register_vote_source(h, fp, "d1") == 2, "repetir no suma"

    def test_la_ip_se_guarda_hasheada(self, app_mod):
        db = app_mod.db
        h = db.hash_ip("192.0.2.99")
        assert "192.0.2.99" not in h
        assert len(h) == 16
        assert h == db.hash_ip("192.0.2.99"), "el hash debe ser estable"

    def test_el_guard_nunca_tumba_el_voto_si_la_bd_falla(self, client, app_mod, monkeypatch):
        """La mitigación es best-effort: si el registro peta, el voto pasa.
        Preferimos un voto de más que perder los de todo el mundo."""
        def boom(*a, **k):
            raise RuntimeError("BD caída")

        monkeypatch.setattr(app_mod.db, "register_vote_source", boom)
        r = _vote(client, _fp(), "device-normal")
        assert r.status_code == 200
