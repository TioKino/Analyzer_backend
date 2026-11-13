"""Las rutas públicas que nadie usa no vuelven (2026-11-12, auditoría S6).

Ese día había veinte rutas que contestaban 200 a cualquiera, sin credencial, y
que ningún cliente llamaba: `/search/*`, `/library/artists|genres|stats`,
`GET /track/{id}`, `/similar-tracks`, `/user-cues/*`, `/community-cues/…/raw`,
`/community/track-type` y `/community/popularity/batch`. Se comprobó en el
historial entero de `lib/` (desde el primer commit), en la web, en los scripts
y en el motor local. Lo que hacían: dar las huellas, los artistas y los nombres
de fichero de la memoria colectiva a quien las pidiera, con SQLite dentro del
event loop. Con esas huellas se escribía en lo que aún acepta escrituras sin
acreditar (notas públicas, previews).

Este test fija la superficie pública: una ruta nueva hace fallar
`test_no_hay_rutas_publicas_de_mas`, y entonces toca decidir a conciencia
—quién la llama, qué credencial pide, si corre fuera del event loop— y
apuntarla en `RUTAS_PUBLICAS`. `/admin/*` y `/sync/admin/*` quedan fuera:
piden la credencial de admin y tienen sus propios tests.

    pytest test_las_rutas_sin_uso_no_vuelven.py -v
"""
import os
import tempfile

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


# Todo lo que el servidor contesta sin la credencial de admin en la ruta.
RUTAS_PUBLICAS = {
    "GET /",
    "POST /acoustic-clusters",
    "POST /acoustic/pending",
    "POST /analysis/by-fingerprint/batch",
    "GET /analysis/by-fingerprint/{fingerprint}",
    "GET /analysis/{filename:path}",
    "POST /analyze",
    "GET /announcement",
    "POST /artwork/upload/{fingerprint}",
    "GET /artwork/{track_id}",
    "POST /artworks/check",
    "POST /audd/motor-local",
    "POST /audd/motor-local/identificar",  # el motor sin token de AudD (S3)
    "POST /backfill-audio",
    "POST /backfill-fingerprint",
    "POST /cache-analysis",
    "POST /check-analyzed",
    "POST /check-analyzed-by-fingerprint",
    "POST /client-error",
    "POST /client-event",
    "POST /cluster-best",
    "POST /community-cues",
    "GET /community-cues/{fingerprint}",
    "POST /community/beat-grid",
    "POST /community/beat-grid/batch",
    "GET /community/beat-grid/{fingerprint}",
    "POST /community/cues",
    "POST /community/cues/batch",
    "GET /community/cues/{fingerprint}",
    "GET /community/de-este-aparato",
    "POST /community/imported",
    "POST /community/library-updates",
    "POST /community/my-ratings/batch",
    "POST /community/notes",
    "GET /community/notes/{fingerprint}",
    "POST /community/notes/{note_id}/upvote",
    "POST /community/notes/{note_id}/denunciar",
    "POST /community/override",
    "DELETE /community/override/{field}/{fingerprint}",
    "GET /community/override/{field}/{fingerprint}",
    "GET /community/popularity/{fingerprint}",
    "POST /community/rate",
    "POST /correction",
    "GET /escuchar/ajustes",
    "GET /health",
    "POST /identidad/verificada",
    "POST /identidad/verificada/consulta",
    "POST /identidad/verificada/retirar",
    "POST /identify",
    "GET /library/all",
    "POST /preview/upload/{track_id}",
    "GET /preview/{track_id}",
    "POST /previews/check",
    "POST /puesta-al-dia/analizar",
    "POST /puesta-al-dia/candidatos",
    "POST /recognize",
    "POST /recognize/shazam",
    "GET /search-analyzed",
    "POST /search-analyzed/batch",
    "GET /shazam/token",
    "POST /sync/borrar-mis-datos",
    "DELETE /sync/clear",
    "POST /sync/detected-track",
    "POST /sync/detected-track/borrar",
    "DELETE /sync/detected-tracks/{device_id}",
    "GET /sync/detected-tracks/{device_id}",
    "POST /sync/link/generate",
    "POST /sync/link/join",
    "POST /sync/link/unlink",
    "POST /sync/listen-set",
    "DELETE /sync/listen-set/{device_id}/{set_id}",
    "GET /sync/listen-sets/{device_id}",
    "GET /sync/pending/{device_id}",
    "POST /sync/publish",
    "GET /sync/pull/{device_id}",
    "POST /sync/push",
    "POST /sync/register",
    "GET /sync/status",
    "DELETE /track/{track_id}",
}

# Lo que se quitó, con el método con el que contestaba.
QUITADAS = [
    ("GET", "/search/artist/Deadmau5"),
    ("GET", "/search/genre/Techno"),
    ("GET", "/search/bpm?min_bpm=120&max_bpm=130"),
    ("GET", "/search/energy?min_energy=5&max_energy=8"),
    ("GET", "/search/key/Am"),
    ("GET", "/search/compatible/8A"),
    ("GET", "/search/track-type/peak"),
    ("POST", "/search/advanced"),
    ("GET", "/library/artists"),
    ("GET", "/library/genres"),
    ("GET", "/library/stats"),
    ("POST", "/similar-tracks"),
    ("GET", "/similar-tracks/" + "ab" * 16),
    ("GET", "/user-cues/un-aparato"),
    ("GET", "/user-cues/un-aparato/" + "ab" * 16),
    ("GET", "/community-cues/" + "ab" * 16 + "/raw"),
    ("POST", "/community/track-type"),
    ("GET", "/community/track-type/" + "ab" * 16),
    ("POST", "/community/popularity/batch"),
]

_DE_ADMIN = ("/admin/", "/sync/admin/")
_DE_LA_DOCUMENTACION = ("/docs", "/redoc", "/openapi.json")


def _montadas(app):
    from rutas_montadas import rutas_montadas

    return {
        f"{metodo} {path}"
        for metodo, path in rutas_montadas(app)
        if not path.startswith(_DE_ADMIN)
        and not path.startswith(_DE_LA_DOCUMENTACION)
    }


def test_no_hay_rutas_publicas_de_mas(app_mod):
    de_mas = _montadas(app_mod.app) - RUTAS_PUBLICAS
    assert not de_mas, (
        f"Rutas públicas nuevas: {sorted(de_mas)}. Antes de apuntarlas en "
        "RUTAS_PUBLICAS: ¿quién las llama?, ¿qué credencial piden?, ¿hacen su "
        "trabajo fuera del event loop? Una ruta sin llamador es superficie.")


def test_la_lista_no_guarda_rutas_que_ya_no_existen(app_mod):
    """Si se quita una ruta, sale de la lista: una lista con sobras deja
    volver a montarla sin que este test diga nada."""
    sobran = RUTAS_PUBLICAS - _montadas(app_mod.app)
    assert not sobran, f"En RUTAS_PUBLICAS y sin montar: {sorted(sobran)}"


@pytest.mark.parametrize("metodo,ruta", QUITADAS)
def test_las_quitadas_no_contestan(client, metodo, ruta):
    r = client.request(metodo, ruta, json={} if metodo == "POST" else None)
    # 404 si no queda nada en esa ruta; 405 si queda otra con otro método.
    assert r.status_code in (404, 405), f"{metodo} {ruta} -> {r.status_code}"


def test_get_track_ya_no_da_la_fila_a_cualquiera(client, app_mod):
    """`GET /track/{id}` devolvía la fila entera (nombre de fichero incluido)
    de cualquier huella. `DELETE /track/{id}` sigue, y es solo de admin."""
    fp = "cd" * 16
    app_mod.db.save_track({
        "id": fp, "fingerprint": fp, "filename": "privado - de alguien.mp3",
        "artist": "a", "title": "t", "duration": 100.0, "bpm": 120.0,
        "energy_dj": 5, "genre": "Techno", "track_type": "peak",
    })
    r = client.get(f"/track/{fp}")
    assert r.status_code in (404, 405)
    assert "privado" not in r.text


def test_lo_que_el_cliente_usa_para_buscar_sigue(client):
    """Lo quitado no se lleva por delante la ficha de Escuchar."""
    r = client.get("/search-analyzed", params={"artist": "Nadie", "title": "Nada"})
    assert r.status_code in (200, 404)
    r = client.post("/search-analyzed/batch", json={"temas": []})
    assert r.status_code == 200
