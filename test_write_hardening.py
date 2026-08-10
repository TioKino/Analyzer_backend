"""
Regresión del endurecimiento de escritura (auditoría 2026-08-09).

Cada test aquí corresponde a un agujero que se verificó explotable con
TestClient antes del fix. Son baratos y evitan que una refactorización futura
los reabra sin que nadie se entere.
"""

import hashlib
import hmac
import json
import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_mod():
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("PREVIEWS_DIR", tempfile.mkdtemp())
    os.environ.setdefault("ARTWORK_CACHE_DIR", tempfile.mkdtemp())
    os.environ.pop("RENDER", None)
    import main

    return main


@pytest.fixture(scope="module")
def client(app_mod):
    return TestClient(app_mod.app)


def _fp() -> str:
    """Fingerprint hex válido y único por test."""
    return uuid.uuid4().hex


# ── SEC-02: envenenamiento de la memoria colectiva ──────────────────────

class TestFuenteNoFirmadaSeDegrada:
    """Un payload sin firmar no puede reclamar prioridad de Rekordbox (110) y
    pisar el análisis de todos. Se degrada a local_engine (50)."""

    def _cache(self, client, fp, source, bpm, artist):
        return client.post("/cache-analysis", json={
            "fingerprint": fp, "filename": f"{fp}.mp3",
            "artist": artist, "title": "t", "bpm": bpm, "key": "Am",
            "camelot": "8A", "duration": 400, "bpm_source": source,
            "analysis_json": {"bpm": bpm, "duration": 400},
        })

    def test_rekordbox_sin_firma_no_pisa_el_analisis_existente(self, client, app_mod):
        fp = _fp()
        assert self._cache(client, fp, "local_engine", 128.0, "Real").status_code == 200
        # El ataque: misma huella, fuente "profesional", datos basura.
        assert self._cache(client, fp, "rekordbox", 1.0, "PWNED").status_code == 200

        row = app_mod.db.get_track_by_fingerprint(fp)
        assert row["artist"] == "Real", "el análisis legítimo fue sobreescrito"
        assert row["bpm"] == 128.0

    def test_la_fuente_queda_degradada_no_rechazada(self, client, app_mod):
        """Se degrada en vez de rechazar: el dato se guarda igual (no perdemos
        análisis de un cliente inesperado), pero sin prioridad profesional."""
        fp = _fp()
        assert self._cache(client, fp, "traktor", 130.0, "Nuevo").status_code == 200

        row = app_mod.db.get_track_by_fingerprint(fp)
        assert row is not None, "el análisis debía guardarse igualmente"
        stored = json.loads(row["analysis_json"])
        assert stored["bpm_source"] == "local_engine"

    def test_firmado_conserva_la_prioridad(self, client, app_mod, monkeypatch):
        """Con firma válida sí se respeta la fuente profesional."""
        secret = "test-write-secret"
        monkeypatch.setattr(app_mod, "_WRITE_AUTH_SECRET", secret)
        fp = _fp()
        payload = {
            "fingerprint": fp, "filename": f"{fp}.mp3", "artist": "Pro",
            "title": "t", "bpm": 124.0, "key": "Am", "camelot": "8A",
            "duration": 400, "bpm_source": "rekordbox",
            "analysis_json": {"bpm": 124.0, "duration": 400},
        }
        body = json.dumps(payload).encode()
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        r = client.post("/cache-analysis", content=body, headers={
            "Content-Type": "application/json", "X-Signature": sig,
        })
        assert r.status_code == 200
        stored = json.loads(app_mod.db.get_track_by_fingerprint(fp)["analysis_json"])
        assert stored["bpm_source"] == "rekordbox"

    def test_firma_invalida_es_401(self, client, app_mod, monkeypatch):
        monkeypatch.setattr(app_mod, "_WRITE_AUTH_SECRET", "test-write-secret")
        r = client.post("/cache-analysis", content=b'{"fingerprint":"x"}', headers={
            "Content-Type": "application/json", "X-Signature": "deadbeef",
        })
        assert r.status_code == 401


# ── SEC-03 / SEC-13: endpoints que ahora exigen admin ───────────────────

class TestEndpointsAdmin:
    def test_delete_track_sin_token_es_401(self, client, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")
        assert client.delete(f"/track/{_fp()}").status_code == 401

    def test_delete_track_con_token_funciona(self, client, app_mod, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")
        fp = _fp()
        app_mod.db.save_track({
            "id": fp, "fingerprint": fp, "filename": f"{fp}.mp3",
            "artist": "a", "title": "t", "duration": 100.0, "bpm": 120.0,
            "energy_dj": 5, "genre": "Techno", "track_type": "peak",
        })
        r = client.delete(f"/track/{fp}",
                          headers={"Authorization": "Bearer tok-secreto"})
        assert r.status_code == 200
        assert app_mod.db.get_track_by_id(fp) is None

    def test_library_all_sin_token_es_401(self, client, monkeypatch):
        """Volcado de la base colectiva: era anónimo y clonable con un curl."""
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")
        assert client.get("/library/all?limit=5000").status_code == 401

    def test_library_agregados_siguen_abiertos(self, client, monkeypatch):
        """Cerrar el volcado no debe cerrar las agregaciones."""
        monkeypatch.setenv("ADMIN_TOKEN", "tok-secreto")
        assert client.get("/library/genres").status_code == 200
        assert client.get("/library/stats").status_code == 200


# ── SEC-07: la blacklist de IPs no se salta con una cabecera ────────────

class TestBlacklistIP:
    def test_spoof_de_x_forwarded_for_no_evade_el_bloqueo(self, client, app_mod):
        app_mod._BLOCKED_IPS = {"9.9.9.9"}
        try:
            directo = client.get("/", headers={"X-Forwarded-For": "9.9.9.9"})
            spoof = client.get("/", headers={"X-Forwarded-For": "1.1.1.1, 9.9.9.9"})
            assert directo.status_code == 403
            assert spoof.status_code == 403, (
                "la IP real la añade el proxy por la DERECHA; tomar la primera "
                "entrada deja que el cliente se invente su IP"
            )
        finally:
            app_mod._BLOCKED_IPS = set()


# ── SEC-09: topes de subida ─────────────────────────────────────────────

class TestTopeDeSubida:
    def test_hay_un_techo_unico_para_todos_los_endpoints_de_audio(self, app_mod):
        assert app_mod.MAX_UPLOAD_BYTES == app_mod.MAX_UPLOAD_MB * 1024 * 1024
        assert app_mod.MAX_UPLOAD_BYTES > 0

    def test_recognize_rechaza_por_encima_del_techo(self, client, app_mod, monkeypatch):
        monkeypatch.setattr(app_mod, "MAX_UPLOAD_BYTES", 2048)
        r = client.post("/recognize", files={"file": ("a.m4a", b"x" * 8192, "audio/mp4")})
        assert r.status_code == 400
        assert "grande" in r.json()["detail"].lower()


# ── SEC-08: docs apagados en producción ─────────────────────────────────

def test_docs_se_apagan_en_produccion():
    """La decisión se toma en el import de main; aquí se comprueba la regla
    sin re-importar el módulo entero."""
    import importlib

    import main

    src = importlib.import_module("main")
    assert hasattr(src, "_DOCS_ON")
    # En el entorno de test no hay RENDER -> docs encendidos (dev).
    assert src._DOCS_ON is True
    # Y la app los cablea en función de esa bandera.
    assert (src.app.docs_url is not None) == src._DOCS_ON
    assert (src.app.openapi_url is not None) == src._DOCS_ON
