"""
Library route handlers for DJ Analyzer Pro API.
"""
import logging

from fastapi import APIRouter, Query, HTTPException

from validation import validate_limit

logger = logging.getLogger(__name__)

# Module-level database reference, set by init()
db = None


def init(database):
    """Initialize this module with the shared database instance."""
    global db
    db = database


library_router = APIRouter(tags=["library"])


@library_router.get("/library/all")
async def get_all_tracks(limit: int = Query(1000, ge=1, le=5000)):
    #  Validar límite
    limit = validate_limit(limit, max_limit=5000)

    return {"tracks": db.get_all_tracks(limit)}


@library_router.get("/library/artists")
async def get_unique_artists():
    """Obtener lista de artistas únicos"""
    artists = db.get_unique_artists()
    return {"count": len(artists), "artists": artists}


@library_router.get("/library/genres")
async def get_unique_genres():
    """Obtener lista de géneros únicos"""
    genres = db.get_unique_genres()
    return {"count": len(genres), "genres": genres}


@library_router.get("/library/stats")
async def get_library_stats():
    """Obtener estadísticas de la biblioteca"""
    return db.get_stats()


@library_router.get("/track/{track_id}")
async def get_track(track_id: str):
    """Obtener información de un track específico"""
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track no encontrado")
    return track


@library_router.delete("/track/{track_id}")
async def delete_track(track_id: str):
    """Eliminar un track de la base de datos"""
    deleted = db.delete_track(track_id)
    if not deleted:
        raise HTTPException(404, "Track no encontrado")
    return {"status": "ok", "message": "Track eliminado"}
