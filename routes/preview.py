"""
Preview endpoints — serve, generate, and check 6s MP3 preview snippets.
"""
import os
import re
import json
import logging
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

preview_router = APIRouter(tags=["preview"])

# Module-level references set by init()
_db = None
_previews_dir = ""
_generate_snippet = None  # Reference to generate_preview_snippet function


def init(database, previews_dir: str, generate_snippet_fn):
    """Initialize with shared database, previews directory, and snippet generator."""
    global _db, _previews_dir, _generate_snippet
    _db = database
    _previews_dir = previews_dir
    _generate_snippet = generate_snippet_fn


@preview_router.get("/preview/{track_id}")
async def get_preview(track_id: str):
    """
    Serve 6s preview MP3 snippet.
    1. If cached in PREVIEWS_DIR → serve directly
    2. If not cached but track in DB → generate on-the-fly
    """
    if not track_id or len(track_id) > 64:
        raise HTTPException(400, "track_id inválido")

    safe_id = re.sub(r'[^a-fA-F0-9]', '', track_id)
    if safe_id != track_id:
        raise HTTPException(400, "track_id contiene caracteres inválidos")

    preview_path = os.path.join(_previews_dir, f"{safe_id}.mp3")

    # 1. Serve from cache
    if os.path.exists(preview_path):
        return FileResponse(
            preview_path,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=31536000",
                "Content-Disposition": f"inline; filename={safe_id}_preview.mp3",
            }
        )

    # 2. Generate on-the-fly
    track = _db.get_track_by_id(safe_id) or _db.get_track_by_fingerprint(safe_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    filename = track.get('filename', '')
    audio_path = None
    analysis_json = track.get('analysis_json')
    original_path = None
    drop_ts = 30.0
    duration = track.get('duration', 0) or 180.0

    if analysis_json:
        try:
            aj = json.loads(analysis_json) if isinstance(analysis_json, str) else analysis_json
            original_path = (
                aj.get('original_file_path')
                or aj.get('track', {}).get('filePath')
                or aj.get('track', {}).get('file_path')
            )
            drop_ts = aj.get('drop_timestamp', 0) or 30.0
            duration = aj.get('duration', duration)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    candidates = [
        original_path,
        f"/tmp/{filename}" if filename else None,
        filename if filename and os.path.isabs(filename) else None,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            audio_path = path
            break

    if not audio_path:
        raise HTTPException(status_code=404, detail="Preview not available — audio file not found on server")

    result = _generate_snippet(
        file_path=audio_path,
        fingerprint=safe_id,
        drop_timestamp=drop_ts,
        duration=duration,
    )

    if result and os.path.exists(result):
        return FileResponse(
            result,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=31536000",
                "Content-Disposition": f"inline; filename={safe_id}_preview.mp3",
                "X-Preview-Generated": "on-the-fly",
            }
        )

    raise HTTPException(status_code=500, detail="Failed to generate preview")


class PreviewGenerateRequest(BaseModel):
    track_id: str
    file_path: str
    drop_timestamp: float = 30.0
    duration: float = 180.0


@preview_router.post("/preview/generate")
async def generate_preview_on_demand(req: PreviewGenerateRequest):
    """Generate preview snippet on-demand from a local file path."""
    safe_id = re.sub(r'[^a-fA-F0-9]', '', req.track_id)
    if not safe_id:
        raise HTTPException(400, "track_id inválido")

    preview_path = os.path.join(_previews_dir, f"{safe_id}.mp3")
    if os.path.exists(preview_path):
        return {"status": "exists", "url": f"/preview/{safe_id}"}

    if not os.path.exists(req.file_path):
        raise HTTPException(404, f"File not found: {req.file_path}")

    result = _generate_snippet(
        file_path=req.file_path,
        fingerprint=safe_id,
        drop_timestamp=req.drop_timestamp,
        duration=req.duration,
    )

    if result and os.path.exists(result):
        track = _db.get_track_by_id(safe_id) or _db.get_track_by_fingerprint(safe_id)
        if track and track.get('analysis_json'):
            try:
                aj = json.loads(track['analysis_json']) if isinstance(track['analysis_json'], str) else track['analysis_json']
                aj['original_file_path'] = req.file_path
                aj['id'] = safe_id
                aj['filename'] = track.get('filename', '')
                aj['fingerprint'] = safe_id
                _db.save_track(aj)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        return {"status": "generated", "url": f"/preview/{safe_id}"}

    raise HTTPException(500, "Failed to generate preview")


class PreviewCheckRequest(BaseModel):
    track_ids: List[str]


@preview_router.post("/previews/check")
async def check_previews(request: PreviewCheckRequest):
    """Check which track_ids have preview snippets available. Max 500."""
    track_ids = request.track_ids
    if len(track_ids) > 500:
        raise HTTPException(400, "Máximo 500 track_ids por petición")

    available = []
    for tid in track_ids:
        safe_id = re.sub(r'[^a-fA-F0-9]', '', tid)
        if safe_id and os.path.exists(os.path.join(_previews_dir, f"{safe_id}.mp3")):
            available.append(tid)

    return {
        "available": available,
        "total_checked": len(track_ids),
        "total_available": len(available),
    }
