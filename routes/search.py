"""
Search route handlers for DJ Analyzer Pro API.
"""
import re
import json
import sqlite3
import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Request
from pydantic import BaseModel

from validation import (
    validate_bpm_range,
    validate_energy_range,
    validate_limit,
    sanitize_string,
    validate_key,
    validate_camelot,
    validate_track_type,
    validate_genre,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Module-level database reference, set by init()
db = None

# Will be set by init() from main module
CAMELOT_COMPATIBLE = {}


def init(database, camelot_compatible=None):
    """Initialize this module with the shared database instance."""
    global db, CAMELOT_COMPATIBLE
    db = database
    if camelot_compatible is not None:
        CAMELOT_COMPATIBLE = camelot_compatible


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


search_router = APIRouter(tags=["search"])


@search_router.get("/search/artist/{artist}")
async def search_by_artist(artist: str, limit: int = Query(50, ge=1, le=200)):
    """Buscar tracks por artista"""
    results = db.search_by_artist(artist, limit)
    return {"query": artist, "count": len(results), "tracks": results}


@search_router.get("/search/genre/{genre}")
async def search_by_genre(genre: str, limit: int = Query(100, ge=1, le=500)):
    #  Sanitizar género
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
        # Si no es válido como key, intentar como está
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

    # db.search_compatible_keys expects a single camelot string and computes
    # compatible keys internally. Pass the original camelot string, not the list.
    return {
        "camelot": camelot,
        "compatible_keys": compatible,
        "tracks": db.search_compatible_keys(camelot, limit)
    }

@search_router.get("/search-analyzed")
async def search_analyzed_track(
    artist: str = Query(..., description="Nombre del artista"),
    title: str = Query(..., description="Título del track")
):
    """
    Busca si un track ya fue analizado por algún usuario.
    Devuelve TODA la información del análisis si existe.

    Returns:
        - found: bool - Si se encontró el track
        - track: dict - Toda la información del análisis (si existe)
        - in_collective: bool - Si está en la memoria colectiva
    """
    # Validar y sanitizar entrada
    artist_clean = sanitize_string(artist, max_length=200, allow_empty=False, field_name="artist")
    title_clean = sanitize_string(title, max_length=200, allow_empty=False, field_name="title")

    # Normalizar para búsqueda
    artist_normalized = artist_clean.lower().strip()
    title_normalized = re.sub(
        r'\s*\(?(Original Mix|Extended Mix|Radio Edit|Remix|Club Mix|Dub Mix)\)?',
        '',
        title_clean,
        flags=re.IGNORECASE
    ).lower().strip()

    try:
        conn = db.conn
        cursor = conn.cursor()

        # Búsqueda exacta primero
        cursor.execute("""
            SELECT * FROM tracks
            WHERE LOWER(artist) = ? AND LOWER(title) LIKE ?
            AND bpm IS NOT NULL AND bpm > 0
            ORDER BY analyzed_at DESC
            LIMIT 1
        """, (artist_normalized, f"%{title_normalized}%"))

        row = cursor.fetchone()

        if not row:
            # Búsqueda más flexible
            cursor.execute("""
                SELECT * FROM tracks
                WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
                AND bpm IS NOT NULL AND bpm > 0
                ORDER BY analyzed_at DESC
                LIMIT 1
            """, (f"%{artist_normalized}%", f"%{title_normalized}%"))
            row = cursor.fetchone()

        if row:
            # Convertir a dict usando el método existente
            track_dict = db._row_to_dict(row)

            # Si hay analysis_json, parsear para obtener todos los campos
            if track_dict and track_dict.get('analysis_json'):
                try:
                    full_analysis = json.loads(track_dict['analysis_json'])
                    # Combinar con los campos básicos
                    track_dict.update(full_analysis)
                except:
                    pass

            # Eliminar el JSON crudo del response
            if track_dict and 'analysis_json' in track_dict:
                del track_dict['analysis_json']

            return {
                "found": True,
                "in_collective": True,
                "track": track_dict
            }

        return {
            "found": False,
            "in_collective": False,
            "track": None
        }

    except (sqlite3.DatabaseError, ValueError, TypeError, KeyError) as e:
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
