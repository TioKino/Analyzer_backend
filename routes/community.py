"""
Community route handlers for DJ Analyzer Pro API.
"""
import sqlite3
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Module-level database reference, set by init()
db = None


def init(database):
    """Initialize this module with the shared database instance."""
    global db
    db = database


class BeatGridCorrectionRequest(BaseModel):
    fingerprint: str
    device_id: str
    bpm_adjust: float = 0.0
    beat_offset: float = 0.0
    original_bpm: float = 0.0


class CommunityNoteRequest(BaseModel):
    fingerprint: str
    device_id: str
    note_text: str
    display_name: str = "DJ"
    note_type: str = "general"  # general, technique, mixing, warning


class TrackRatingRequest(BaseModel):
    fingerprint: str
    device_id: str
    rating: int  # 1-5


community_router = APIRouter(tags=["community"])


@community_router.post("/community/beat-grid")
async def submit_beat_grid_correction(request: BeatGridCorrectionRequest):
    """Recibe correccion de beat grid de un DJ"""
    try:
        db.submit_beat_grid_correction(
            fingerprint=request.fingerprint,
            device_id=request.device_id,
            bpm_adjust=request.bpm_adjust,
            beat_offset=request.beat_offset,
            original_bpm=request.original_bpm,
        )
        logger.info(f"[Community] Beat grid correction: fp={request.fingerprint[:8]}... "
              f"BPM+{request.bpm_adjust:.2f} OFF+{request.beat_offset*1000:.1f}ms")
        return {"status": "ok"}
    except (sqlite3.DatabaseError, ValueError, TypeError) as e:
        logger.error(f"[Community] Error saving beat grid: {e}")
        raise HTTPException(500, f"Error: {str(e)}")


@community_router.get("/community/beat-grid/{fingerprint}")
async def get_community_beat_grid(fingerprint: str):
    """Obtiene la correccion promedio de la comunidad"""
    try:
        result = db.get_community_beat_grid(fingerprint)
        return result
    except (sqlite3.DatabaseError, ValueError, TypeError) as e:
        logger.error(f"[Community] Error fetching beat grid: {e}")
        return {"bpm_adjust": 0.0, "beat_offset": 0.0, "contributors": 0, "validated": False}


# ==================== COMMUNITY NOTES (SoundCloud-style) ====================

@community_router.post("/community/notes")
async def post_community_note(req: CommunityNoteRequest):
    """Un DJ deja una nota pública en un track (visible para todos)."""
    if not req.note_text.strip():
        raise HTTPException(400, "note_text vacío")
    if len(req.note_text) > 500:
        raise HTTPException(400, "Nota demasiado larga (max 500 chars)")
    try:
        note_id = db.save_community_note(
            fingerprint=req.fingerprint,
            device_id=req.device_id,
            note_text=req.note_text.strip(),
            display_name=req.display_name.strip()[:30] or "DJ",
            note_type=req.note_type,
        )
        logger.info(f"[Community] Nota guardada: fp={req.fingerprint[:8]}... by {req.display_name}")
        return {"status": "ok", "note_id": note_id}
    except Exception as e:
        logger.error(f"[Community] Error guardando nota: {e}")
        raise HTTPException(500, str(e))


@community_router.get("/community/notes/{fingerprint}")
async def get_community_notes(fingerprint: str):
    """Devuelve todas las notas de la comunidad para un track."""
    try:
        notes = db.get_community_notes(fingerprint)
        return {"fingerprint": fingerprint, "notes": notes, "total": len(notes)}
    except Exception as e:
        logger.error(f"[Community] Error leyendo notas: {e}")
        return {"fingerprint": fingerprint, "notes": [], "total": 0}


@community_router.post("/community/notes/{note_id}/upvote")
async def upvote_note(note_id: int):
    """Sube un voto a una nota comunitaria."""
    try:
        db.upvote_community_note(note_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ==================== TRACK POPULARITY & RATINGS ====================

@community_router.post("/community/rate")
async def rate_track(req: TrackRatingRequest):
    """Un DJ puntúa un track (1-5 estrellas). Una valoración por DJ por track."""
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(400, "Rating debe ser 1-5")
    try:
        result = db.rate_track(req.fingerprint, req.device_id, req.rating)
        logger.info(f"[Community] Rating: fp={req.fingerprint[:8]}... = {req.rating}★ (avg {result['avg_rating']})")
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"[Community] Error rating: {e}")
        raise HTTPException(500, str(e))


@community_router.get("/community/popularity/{fingerprint}")
async def get_popularity(fingerprint: str, device_id: str = ""):
    """Devuelve popularidad + rating medio + tu rating para un track."""
    try:
        pop = db.get_track_popularity(fingerprint)
        my_rating = db.get_my_rating(fingerprint, device_id) if device_id else 0
        return {**pop, "my_rating": my_rating}
    except Exception as e:
        return {"analysis_count": 0, "dj_count": 0, "avg_rating": 0, "total_ratings": 0, "my_rating": 0}
