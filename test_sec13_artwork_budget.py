"""
SEC-13 — el fallback online de /artwork lleva cupo propio.

`GET /artwork/{id}` es anónimo. Si no hay carátula cacheada sale a internet:
hasta 8 peticiones a iTunes + Deezer + Last.fm con timeouts de 5-8 s. Desde la
auditoría eso ya no congela el event loop (corre en `run_in_threadpool`), pero
seguía sin techo, y el riesgo que queda no es el bloqueo sino el **tráfico
saliente**: un bucle sobre fingerprints sin carátula puede hacer que esas APIs
limiten o baneen la IP de Render, y eso degrada el servicio para todos los
usuarios, no solo para el que abusa.

LO QUE NO SE PUEDE ROMPER: servir una carátula YA CACHEADA. La app pide
decenas o cientos de golpe al pintar la biblioteca; capar eso rompería el uso
normal. Por eso el cupo cubre solo el camino caro, y la mitad de estos tests
existen para fijar justo esa frontera.
"""

import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_mod():
    import main

    return main


@pytest.fixture(scope="module")
def client(app_mod):
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def cupo_limpio():
    """Cada test arranca con el contador a cero: si no, el orden de ejecución
    decidiría el resultado."""
    import validation

    validation.artwork_online_limiter = validation.RateLimiter(
        max_requests=validation.ARTWORK_ONLINE_MAX_PER_MIN, window_seconds=60,
    )
    yield


class TestElCaminoBaratoNoSeCapa:
    """Servir cache es leer un fichero local. Ese camino NO puede tener techo."""

    def test_cien_caratulas_cacheadas_seguidas(self, client, app_mod):
        from routes import analysis_artwork as aw

        fp = uuid.uuid4().hex
        ruta = os.path.join(aw.ARTWORK_CACHE_DIR, f"{fp}.jpg")
        os.makedirs(aw.ARTWORK_CACHE_DIR, exist_ok=True)
        with open(ruta, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0JPEGFAKE")

        try:
            for i in range(100):
                r = client.get(f"/artwork/{fp}")
                assert r.status_code == 200, (
                    f"la peticion {i} a una caratula CACHEADA fue rechazada; "
                    f"el cupo del fallback online no debe tocar este camino"
                )
        finally:
            os.unlink(ruta)


class TestElCaminoCaroSiSeCapa:
    def test_el_cupo_corta_la_salida_a_internet(self, app_mod, monkeypatch):
        """Se cuenta cuántas veces se llama de verdad a `search_artwork_online`.
        Es la métrica que importa: peticiones salientes, no códigos HTTP."""
        import validation
        from routes import analysis_artwork as aw

        llamadas = {"n": 0}

        def _fake_online(artist, title):
            llamadas["n"] += 1
            return None  # no encuentra nada -> el handler cae a 404

        monkeypatch.setattr(aw, "search_artwork_online", _fake_online)

        # Un track EN BD pero SIN fichero de carátula -> siempre cae al online.
        fp = uuid.uuid4().hex
        app_mod.db.save_track({
            'id': fp, 'filename': f'{fp}.mp3', 'artist': 'Artista',
            'title': 'Tema', 'duration': 300.0, 'bpm': 128.0,
            'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
            'fingerprint': fp,
        })

        cliente = TestClient(app_mod.app)
        intentos = validation.ARTWORK_ONLINE_MAX_PER_MIN + 15
        for _ in range(intentos):
            cliente.get(f"/artwork/{fp}", headers={"X-Forwarded-For": "203.0.113.90"})

        assert llamadas["n"] <= validation.ARTWORK_ONLINE_MAX_PER_MIN, (
            f"{llamadas['n']} salidas a internet en {intentos} peticiones: el "
            f"cupo no esta cortando y la IP de Render sigue expuesta a que "
            f"iTunes/Deezer/Last.fm la limiten"
        )
        assert llamadas["n"] > 0, "el cupo no puede bloquear TODAS desde la primera"

    def test_al_agotarse_devuelve_404_y_no_429(self, app_mod, monkeypatch):
        """404 es la respuesta normal de 'no hay caratula' y el cliente ya la
        pinta como placeholder. Un 429 seria un estado nuevo que la UI no
        maneja y saldrian imagenes rotas."""
        from routes import analysis_artwork as aw

        monkeypatch.setattr(aw, "search_artwork_online", lambda a, t: None)

        fp = uuid.uuid4().hex
        app_mod.db.save_track({
            'id': fp, 'filename': f'{fp}.mp3', 'artist': 'Artista',
            'title': 'Tema', 'duration': 300.0, 'bpm': 128.0,
            'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
            'fingerprint': fp,
        })

        cliente = TestClient(app_mod.app)
        ultimo = None
        for _ in range(40):
            ultimo = cliente.get(f"/artwork/{fp}",
                                 headers={"X-Forwarded-For": "203.0.113.91"})
        assert ultimo.status_code == 404, (
            f"tras agotar el cupo se devolvio {ultimo.status_code}; se esperaba "
            f"404 para que la UI siga pintando placeholder"
        )

    def test_ips_distintas_no_comparten_cupo(self, app_mod, monkeypatch):
        """Dos usuarios reales en sitios distintos no pueden estorbarse."""
        from routes import analysis_artwork as aw

        llamadas = {"n": 0}

        def _fake_online(artist, title):
            llamadas["n"] += 1
            return None

        monkeypatch.setattr(aw, "search_artwork_online", _fake_online)

        fp = uuid.uuid4().hex
        app_mod.db.save_track({
            'id': fp, 'filename': f'{fp}.mp3', 'artist': 'Artista',
            'title': 'Tema', 'duration': 300.0, 'bpm': 128.0,
            'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
            'fingerprint': fp,
        })

        cliente = TestClient(app_mod.app)
        # Un abusador agota lo suyo...
        for _ in range(40):
            cliente.get(f"/artwork/{fp}", headers={"X-Forwarded-For": "203.0.113.92"})
        antes = llamadas["n"]

        # ...y otra IP sigue pudiendo buscar.
        cliente.get(f"/artwork/{fp}", headers={"X-Forwarded-For": "198.51.100.40"})
        assert llamadas["n"] == antes + 1, (
            "una IP agotando su cupo dejo sin busqueda online a otra distinta"
        )
