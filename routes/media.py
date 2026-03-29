"""
Media endpoints — artwork images, stored analysis, and batch checks.
"""
import os
import json
import logging
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

media_router = APIRouter(tags=["media"])

# Module-level references set by init()
_db = None
_artwork_cache_dir = ""


def init(database, artwork_cache_dir: str):
    """Initialize with shared database and artwork cache directory."""
    global _db, _artwork_cache_dir
    _db = database
    _artwork_cache_dir = artwork_cache_dir


@media_router.get("/analysis/{filename:path}")
async def get_analysis(filename: str):
    """Get stored analysis by filename."""
    filename = unquote(filename)

    existing = _db.get_track_by_filename(filename)
    if existing:
        try:
            analysis_json = existing[11]
            if analysis_json:
                return json.loads(analysis_json)
        except (json.JSONDecodeError, TypeError, IndexError):
            pass

        return {
            "id": existing[0],
            "filename": existing[1],
            "artist": existing[2],
            "title": existing[3],
            "duration": existing[4],
            "bpm": existing[5],
            "key": existing[6],
            "camelot": existing[7],
            "energy_dj": existing[8],
            "genre": existing[9],
            "track_type": existing[10],
            "fingerprint": existing[13] if len(existing) > 13 else None,
        }

    raise HTTPException(404, f"Analysis not found for: {filename}")


@media_router.get("/artwork/{track_id}")
async def get_artwork(track_id: str):
    """Return track artwork image."""
    for ext in ['jpg', 'png', 'jpeg']:
        cache_path = os.path.join(_artwork_cache_dir, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            media_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else "image/png"
            return FileResponse(cache_path, media_type=media_type)

    raise HTTPException(404, "Artwork not found")


@media_router.post("/check-analyzed")
async def check_analyzed(filenames: list[str]):
    """Check which tracks are already analyzed."""
    analyzed = []
    not_analyzed = []

    for filename in filenames:
        existing = _db.get_track_by_filename(filename)
        if existing:
            analyzed.append(filename)
        else:
            not_analyzed.append(filename)

    return {
        "analyzed": analyzed,
        "not_analyzed": not_analyzed,
        "total": len(filenames),
        "analyzed_count": len(analyzed),
        "not_analyzed_count": len(not_analyzed),
    }
