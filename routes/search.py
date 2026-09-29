"""
Search, library and single-track endpoints for DJ Analyzer Pro API.

PASO 2 del troceo de main.py (review 2026-06-29). Estos 15 endpoints
(/search/*, /search-analyzed, /library/*, /track/{id}) eran inline en
main.py; aqui se mueven VERBATIM (mismo comportamiento, sin cambio de
logica) a un router montado con `include_router`. Las dependencias que
antes eran globales de main.py se inyectan via `init(database, camelot_compatible)`:

  - `db`: la instancia AnalysisDB (analysis.db).
  - `CAMELOT_COMPATIBLE`: el dict de keys compatibles (vive en
    similar_tracks_endpoint.py, opcional — si ese modulo no carga,
    main.py inyecta {}).

A diferencia de los 5 modulos muertos que se borraron en el paso 1
(#28), ESTE router SI se monta: main.py hace `init(...)` +
`app.include_router(search_router)` y BORRA los endpoints inline. No hay
duplicacion stale (la causa del doble incidente de /admin/reset-database).
"""

import json
import logging
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from validation import (
    ValidationError,
    sanitize_string,
    validate_bpm_range,
    validate_camelot,
    validate_energy_range,
    validate_genre,
    validate_key,
    validate_limit,
    validate_track_type,
)

logger = logging.getLogger(__name__)


def _require_admin(request: Request) -> None:
    """Auth Bearer contra ADMIN_TOKEN para los endpoints destructivos de aquí.

    Mismo esquema que `main._verify_admin_bearer` y `sync_endpoints._verify_admin`,
    reimplementado en local para no crear un import circular
    (main importa este módulo). En dev sin ADMIN_TOKEN se deja pasar; en
    Render/Railway sin token se falla rápido con 500, igual que el resto.
    """
    import hmac as _hmac
    import os as _os

    token = _os.environ.get('ADMIN_TOKEN', '')
    if not token:
        if _os.getenv('RENDER') or _os.getenv('RAILWAY_ENVIRONMENT'):
            raise HTTPException(500, "ADMIN_TOKEN required in production")
        return
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer ') or not _hmac.compare_digest(auth[7:], token):
        raise HTTPException(401, "Admin token requerido")


# ── Dependencias inyectadas desde main.py ────────────────────
db = None
CAMELOT_COMPATIBLE = {}


def init(database, camelot_compatible=None):
    """Inyecta la instancia de BD y el mapa Camelot desde main.py.

    Debe llamarse ANTES de `app.include_router(search_router)`.
    """
    global db, CAMELOT_COMPATIBLE
    db = database
    if camelot_compatible is not None:
        CAMELOT_COMPATIBLE = camelot_compatible


# ── Modelo de request ────────────────────────────────────────

class SearchRequest(BaseModel):
    artist: Optional[str] = None
    genre: Optional[str] = None
    min_bpm: Optional[float] = None
    max_bpm: Optional[float] = None
    min_energy: Optional[int] = None
    max_energy: Optional[int] = None
    key: Optional[str] = None
    track_type: Optional[str] = None
    limit: int = 100


# ── Router ───────────────────────────────────────────────────

search_router = APIRouter(tags=["search"])


# ==================== ENDPOINTS DE BUSQUEDA ====================

@search_router.get("/search/artist/{artist}")
async def search_by_artist(artist: str, limit: int = Query(50, ge=1, le=200)):
    """Buscar tracks por artista"""
    artist = sanitize_string(artist, max_length=200, allow_empty=False, field_name="artist")
    limit = validate_limit(limit, max_limit=200)
    results = db.search_by_artist(artist, limit)
    return {"query": artist, "count": len(results), "tracks": results}

@search_router.get("/search/genre/{genre}")
async def search_by_genre(genre: str, limit: int = Query(100, ge=1, le=500)):
    #  Sanitizar g(c)nero
    genre = validate_genre(genre)
    limit = validate_limit(limit, max_limit=500)

    return {"tracks": db.search_by_genre(genre, limit)}

@search_router.get("/search/bpm")
async def search_by_bpm(
    request: Request,
    min_bpm: Optional[float] = None,
    max_bpm: Optional[float] = None,
    limit: int = Query(100, ge=1, le=500)
):
    #  Validar rangos
    min_bpm, max_bpm = validate_bpm_range(min_bpm, max_bpm)
    limit = validate_limit(limit, max_limit=500)

    return {"tracks": db.search_by_bpm_range(min_bpm, max_bpm, limit)}

@search_router.get("/search/energy")
async def search_by_energy(
    request: Request,
    min_energy: Optional[int] = None,
    max_energy: Optional[int] = None,
    limit: int = Query(100, ge=1, le=500)
):
    #  Validar rangos
    min_energy, max_energy = validate_energy_range(min_energy, max_energy)
    limit = validate_limit(limit, max_limit=500)

    return {"tracks": db.search_by_energy(min_energy, max_energy, limit)}

@search_router.get("/search/key/{key}")
async def search_by_key(key: str, limit: int = Query(100, ge=1, le=500)):
    #  Validar tonalidad
    try:
        key = validate_key(key)
    except ValidationError:
        # Si no es vlido como key, intentar como est
        key = sanitize_string(key, max_length=10)

    limit = validate_limit(limit, max_limit=500)

    return {"tracks": db.search_by_key(key, limit)}

@search_router.get("/search/compatible/{camelot}")
async def search_compatible_keys(camelot: str, limit: int = Query(50, ge=1, le=200)):
    #  Validar Camelot
    camelot = validate_camelot(camelot)
    limit = validate_limit(limit, max_limit=200)

    # Obtener keys compatibles
    compatible = CAMELOT_COMPATIBLE.get(camelot, [camelot])

    return {
        "camelot": camelot,
        "compatible_keys": compatible,
        "tracks": db.search_compatible_keys(camelot, limit)
    }
# Sufijos de mezcla que no cambian el tema a efectos de «¿alguien lo analizo?».
_SUFIJOS_DE_MEZCLA = re.compile(
    r'\s*\(?(Original Mix|Extended Mix|Radio Edit|Remix|Club Mix|Dub Mix)\)?',
    re.IGNORECASE,
)


def _aplanar(track_dict: Optional[dict]) -> Optional[dict]:
    """El `analysis_json` al nivel de arriba y fuera el crudo: la ficha lista
    para el cliente, sin que tenga que re-parsear un blob."""
    if not track_dict:
        return None
    crudo = track_dict.pop('analysis_json', None)
    if crudo:
        try:
            detalle = json.loads(crudo) if isinstance(crudo, str) else crudo
            if isinstance(detalle, dict):
                track_dict.update(detalle)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("analysis_json corrupto en track %s: %s",
                           track_dict.get('id'), e)
    return track_dict


def buscar_analizado(artist: str, title: str,
                     isrc: Optional[str] = None) -> Optional[dict]:
    """La ficha de un tema que ALGUIEN ya analizo, o None.

    Una sola regla para `/search-analyzed` y `/recognize`. Hasta el 2026-09-29
    cada uno buscaba a su manera: `/recognize` probaba ISRC y titulo exacto, no
    miraba si la fila tenia analisis de verdad y podia devolver su propia
    deteccion (bpm 0 y genero, energia y tipo de relleno); el movil lo tiraba
    y volvia a preguntar aqui, una peticion mas por cada acierto de Escuchar.

    Orden: ISRC (identidad exacta de la grabacion) → artista y titulo exactos →
    artista exacto y titulo sin sufijo de mezcla → los dos por aproximacion.
    Solo filas con bpm > 0: una deteccion de Escuchar no es un analisis.
    """
    if isrc:
        t = db.get_analyzed_track_by_isrc(isrc)
        if t:
            return _aplanar(t)

    artista = (artist or '').lower().strip()
    titulo = (title or '').lower().strip()
    titulo_sin_mezcla = _SUFIJOS_DE_MEZCLA.sub('', title or '').lower().strip()
    # Sin artista o con un titulo que era solo el sufijo («Remix»), el LIKE
    # '%%' casaria con cualquier tema del artista.
    if not artista or not titulo_sin_mezcla:
        return None

    cursor = db.conn.cursor()
    consultas = (
        ("LOWER(artist) = ? AND LOWER(title) = ?", (artista, titulo)),
        ("LOWER(artist) = ? AND LOWER(title) LIKE ?",
         (artista, f"%{titulo_sin_mezcla}%")),
        ("LOWER(artist) LIKE ? AND LOWER(title) LIKE ?",
         (f"%{artista}%", f"%{titulo_sin_mezcla}%")),
    )
    for donde, args in consultas:
        cursor.execute(
            f"SELECT * FROM tracks WHERE {donde} "
            "AND bpm IS NOT NULL AND bpm > 0 "
            "ORDER BY analyzed_at DESC LIMIT 1",
            args,
        )
        row = cursor.fetchone()
        if row:
            return _aplanar(db._row_to_dict(row))
    return None


@search_router.get("/search-analyzed")
async def search_analyzed_track(
    artist: str = Query(..., description="Nombre del artista"),
    title: str = Query(..., description="Titulo del track"),
    isrc: Optional[str] = Query(None, description="ISRC (identidad exacta, opcional)")
):
    """
    Busca si un track ya fue analizado por algun usuario.
    Devuelve TODA la informacion del analisis si existe (ver `buscar_analizado`).

    Returns:
        - found: bool - Si se encontro el track
        - track: dict - Toda la informacion del analisis (si existe)
        - in_collective: bool - Si esta en la memoria colectiva
    """
    artist_clean = sanitize_string(artist, max_length=200, allow_empty=False, field_name="artist")
    title_clean = sanitize_string(title, max_length=200, allow_empty=False, field_name="title")

    try:
        ficha = buscar_analizado(artist_clean, title_clean, isrc)
        if ficha:
            return {"found": True, "in_collective": True, "track": ficha}
        return {"found": False, "in_collective": False, "track": None}
    except Exception as e:
        logger.error(f"Error en search-analyzed: {e}")
        return {
            "found": False,
            "in_collective": False,
            "track": None,
            "error": str(e)
        }

@search_router.get("/search/track-type/{track_type}")
async def search_by_track_type(track_type: str, limit: int = Query(100, ge=1, le=500)):
    #  Validar tipo de track
    track_type = validate_track_type(track_type)
    limit = validate_limit(limit, max_limit=500)

    return {"tracks": db.search_by_track_type(track_type, limit)}

@search_router.post("/search/advanced")
async def search_advanced(search_request: SearchRequest):
    #  Validar y sanitizar todos los campos
    filters = {}

    if search_request.artist:
        filters['artist'] = sanitize_string(search_request.artist, max_length=100)

    if search_request.genre:
        filters['genre'] = validate_genre(search_request.genre)

    if search_request.min_bpm is not None or search_request.max_bpm is not None:
        filters['min_bpm'], filters['max_bpm'] = validate_bpm_range(
            search_request.min_bpm,
            search_request.max_bpm
        )

    if search_request.min_energy is not None or search_request.max_energy is not None:
        filters['min_energy'], filters['max_energy'] = validate_energy_range(
            search_request.min_energy,
            search_request.max_energy
        )

    if search_request.key:
        try:
            filters['key'] = validate_key(search_request.key)
        except ValidationError:
            filters['key'] = sanitize_string(search_request.key, max_length=10)

    if search_request.track_type:
        filters['track_type'] = validate_track_type(search_request.track_type)

    filters['limit'] = validate_limit(search_request.limit, max_limit=500)

    return {"tracks": db.search_advanced(**filters)}

# ==================== ENDPOINTS DE BIBLIOTECA ====================

@search_router.get("/library/all")
async def get_all_tracks(request: Request, limit: int = Query(1000, ge=1, le=5000)):
    """Volcado completo de la base colectiva. SOLO admin.

    Devolvía hasta 5000 tracks con sus fingerprints a cualquiera que llamara.
    La memoria colectiva es el activo diferencial del producto y se podía
    clonar con un curl; además los fingerprints obtenidos aquí servían para
    cebar el fallback online de /artwork (auditoría 2026-08-09, SEC-13).
    Ningún cliente lo consume — el panel admin usa /admin/all-tracks — así que
    cerrarlo no rompe nada. Las agregaciones (/library/artists, /genres,
    /stats) siguen abiertas: no exponen la biblioteca fila a fila.
    """
    _require_admin(request)
    limit = validate_limit(limit, max_limit=5000)

    return {"tracks": db.get_all_tracks(limit)}

@search_router.get("/library/artists")
async def get_unique_artists():
    """Obtener lista de artistas nicos"""
    artists = db.get_unique_artists()
    return {"count": len(artists), "artists": artists}

@search_router.get("/library/genres")
async def get_unique_genres():
    """Obtener lista de g(c)neros nicos"""
    genres = db.get_unique_genres()
    return {"count": len(genres), "genres": genres}

@search_router.get("/library/stats")
async def get_library_stats():
    """Obtener estadsticas de la biblioteca"""
    return db.get_stats()

@search_router.get("/track/{track_id}")
async def get_track(track_id: str):
    """Obtener informacin de un track especfico"""
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track no encontrado")
    return track

@search_router.delete("/track/{track_id}")
async def delete_track(track_id: str, request: Request):
    """Eliminar un track de la base de datos. SOLO admin.

    `tracks` es la memoria colectiva COMPARTIDA por todos los usuarios, no la
    biblioteca de quien llama. Sin auth, cualquiera podía borrar el análisis de
    cualquier track para todo el mundo (verificado en la auditoría 2026-08-09:
    DELETE anónimo -> 200 y la fila desaparecía). Ningún cliente consume este
    endpoint; se mantiene como herramienta de mantenimiento del owner.
    """
    _require_admin(request)
    deleted = db.delete_track(track_id)
    if not deleted:
        raise HTTPException(404, "Track no encontrado")
    return {"status": "ok", "message": "Track eliminado"}
