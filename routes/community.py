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
