"""
Media endpoints — artwork images, stored analysis, and batch checks.
"""
import os
import re
import json
import hashlib
import logging
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, UploadFile, File as FastAPIFile, Request
from fastapi.responses import FileResponse, Response

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
async def get_artwork(track_id: str, request: Request):
    """Return track artwork image with caching headers."""
    for ext in ['jpg', 'png', 'jpeg']:
        cache_path = os.path.join(_artwork_cache_dir, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            # ETag basado en tamaño + mtime del archivo
            stat = os.stat(cache_path)
            etag = hashlib.md5(f"{stat.st_size}-{stat.st_mtime}".encode()).hexdigest()

            # Si el cliente ya tiene esta versión, devolver 304
            if_none_match = request.headers.get("if-none-match")
            if if_none_match and if_none_match.strip('"') == etag:
                return Response(status_code=304)

            media_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else "image/png"
            return FileResponse(
                cache_path,
                media_type=media_type,
                headers={
                    "Cache-Control": "public, max-age=2592000",  # 30 días
                    "ETag": f'"{etag}"',
                },
            )

    raise HTTPException(404, "Artwork not found")


@media_router.post("/artwork/upload/{track_id}")
async def upload_artwork(track_id: str, file: UploadFile = FastAPIFile(...)):
    """Upload artwork image from desktop to Render for mobile access."""
    safe_id = re.sub(r'[^a-fA-F0-9]', '', track_id)
    if not safe_id or len(safe_id) > 64:
        raise HTTPException(400, "track_id inválido")

    content = await file.read()
    if len(content) < 100:
        raise HTTPException(400, "Archivo demasiado pequeño")
    if len(content) > 2_000_000:  # 2MB max
        raise HTTPException(400, "Archivo demasiado grande")

    # Detectar extensión
    ext = 'jpg'
    if content[:8].startswith(b'\x89PNG'):
        ext = 'png'

    os.makedirs(_artwork_cache_dir, exist_ok=True)
    artwork_path = os.path.join(_artwork_cache_dir, f"{safe_id}.{ext}")

    with open(artwork_path, 'wb') as f:
        f.write(content)

    logger.info(f"[Artwork] Uploaded: {safe_id} ({len(content)} bytes)")
    return {"status": "ok", "track_id": safe_id, "size": len(content)}


@media_router.post("/check-analyzed")
async def check_analyzed(filenames: list[str]):
    """Check which tracks are already analyzed."""
    if len(filenames) > 500:
        filenames = filenames[:500]

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
