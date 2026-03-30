"""
Admin route handlers for DJ Analyzer Pro API.
"""
import os
import sqlite3
import logging

from fastapi import APIRouter, Query, HTTPException, Request, Depends

from config import ADMIN_TOKEN

logger = logging.getLogger(__name__)

# Module-level references, set by init()
db = None
ARTWORK_CACHE_DIR = None
ARTWORK_ENABLED = False
GENRE_DETECTOR_ENABLED = False
SIMILAR_TRACKS_ENABLED = False


def init(database, artwork_cache_dir=None, artwork_enabled=False,
         genre_detector_enabled=False, similar_tracks_enabled=False):
    """Initialize this module with the shared database instance and config."""
    global db, ARTWORK_CACHE_DIR, ARTWORK_ENABLED, GENRE_DETECTOR_ENABLED, SIMILAR_TRACKS_ENABLED
    db = database
    if artwork_cache_dir is not None:
        ARTWORK_CACHE_DIR = artwork_cache_dir
    ARTWORK_ENABLED = artwork_enabled
    GENRE_DETECTOR_ENABLED = genre_detector_enabled
    SIMILAR_TRACKS_ENABLED = similar_tracks_enabled


async def _verify_admin_token(request: Request):
    """Verifica token admin en header Authorization: Bearer <token>."""
    if not ADMIN_TOKEN:
        # Solo permitir sin token en entorno local
        if os.getenv('RENDER') or os.getenv('RAILWAY_ENVIRONMENT'):
            raise HTTPException(status_code=500, detail="ADMIN_TOKEN required in production")
        return  # Dev mode local: sin token
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != ADMIN_TOKEN:
        raise HTTPException(401, "Admin token requerido")


admin_router = APIRouter(tags=["admin"])


@admin_router.get("/")
async def root():
    return {
        "name": "DJ Analyzer Pro API",
        "version": "2.3.0",
        "status": "running",
        "modules": {
            "artwork": ARTWORK_ENABLED,
            "genre_detector": GENRE_DETECTOR_ENABLED,
            "similar_tracks": SIMILAR_TRACKS_ENABLED,
            "preview_snippets": True,
        },
        "features": [
            "BPM detection (ID3 + analysis)",
            "Key & Camelot (ID3 + analysis)",
            "Energy analysis (1-10 scale)",
            "Structure analysis (intro/drop/breakdown/outro)",
            "Cue points detection",
            "Beat grid detection",
            "Artwork extraction (ID3)",
            "Genre detection (ID3/AcousticBrainz/spectral)",
            "Collective memory",
            "Similar tracks search",
            "Advanced search filters",
            "Preview snippets (6s streaming)",
        ]
    }


@admin_router.get("/health")
async def health():
    return {"status": "healthy", "version": "2.3.0"}


@admin_router.delete("/admin/reset-database", dependencies=[Depends(_verify_admin_token)])
async def reset_database(confirm: str = Query(..., description="Escribe 'CONFIRMAR' para borrar")):
    """
    PELIGROSO: Borra TODA la base de datos.
    Requiere confirmar escribiendo 'CONFIRMAR' como parámetro.
    """
    if confirm != "CONFIRMAR":
        raise HTTPException(400, "Debes escribir 'CONFIRMAR' para borrar la base de datos")

    try:
        import shutil

        # Borrar artwork cache
        if ARTWORK_CACHE_DIR and os.path.exists(ARTWORK_CACHE_DIR):
            shutil.rmtree(ARTWORK_CACHE_DIR)
            os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)

        # Borrar y recrear BD
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM tracks")
        c.execute("DELETE FROM corrections")
        c.execute("DELETE FROM dj_notes")
        conn.commit()
        conn.close()

        # Borrar sync DB también
        sync_db_path = os.environ.get("SYNC_DB_PATH", "/data/sync.db")
        sync_cleared = False
        if os.path.exists(sync_db_path):
            sync_conn = sqlite3.connect(sync_db_path)
            sync_conn.execute("DELETE FROM sync_items")
            sync_conn.commit()
            sync_conn.close()
            sync_cleared = True

        return {
            "status": "ok",
            "message": "Base de datos reseteada completamente",
            "artwork_cache": "limpiado",
            "tracks": "eliminados",
            "corrections": "eliminadas",
            "sync": "limpiado" if sync_cleared else "no encontrado",
        }
    except (sqlite3.DatabaseError, OSError, PermissionError) as e:
        raise HTTPException(500, f"Error reseteando: {str(e)}")


@admin_router.delete("/admin/clear-artwork-cache", dependencies=[Depends(_verify_admin_token)])
async def clear_artwork_cache():
    """Limpia solo el caché de artwork"""
    import shutil
    try:
        if ARTWORK_CACHE_DIR and os.path.exists(ARTWORK_CACHE_DIR):
            shutil.rmtree(ARTWORK_CACHE_DIR)
            os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)
        return {"status": "ok", "message": "Caché de artwork limpiado"}
    except (OSError, PermissionError) as e:
        raise HTTPException(500, f"Error: {str(e)}")
