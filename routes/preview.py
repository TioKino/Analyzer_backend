"""
Preview snippet + batch-availability endpoints for DJ Analyzer Pro API.

PASO 4 del troceo de main.py (review 2026-06-29). Bloque movido VERBATIM
desde main.py (mismo comportamiento, sin cambio de logica):

  - GET  /preview/{track_id}        sirve el snippet MP3 (6s)
  - POST /previews/check            batch: que track_ids tienen preview
  - POST /artworks/check            batch: que track_ids tienen artwork cacheado
  - POST /preview/upload/{track_id} guarda snippet subido por engine/cliente

Dependencias antes globales de main.py se inyectan con
init(previews_dir, artwork_cache_dir) ANTES de include_router. Se inyectan
(en vez de re-importar de config) para usar EXACTAMENTE los mismos paths
que main.py resolvio — ARTWORK_CACHE_DIR puede venir de artwork_and_cuepoints
o del fallback de config, y no queremos divergir.

Como search/community (pasos 2-3): este router SI se monta y los endpoints
inline se BORRAN -> sin duplicacion stale.
"""

import logging
import os
import re
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Dependencias inyectadas desde main.py ────────────────────
PREVIEWS_DIR = None
ARTWORK_CACHE_DIR = None


def init(previews_dir, artwork_cache_dir):
    """Inyecta los paths de cache (previews + artwork) desde main.py.
    Llamar ANTES de app.include_router(preview_router)."""
    global PREVIEWS_DIR, ARTWORK_CACHE_DIR
    PREVIEWS_DIR = previews_dir
    ARTWORK_CACHE_DIR = artwork_cache_dir


# ── Router ───────────────────────────────────────────────────
preview_router = APIRouter(tags=["preview"])


@preview_router.get("/preview/{track_id}")
async def get_preview(track_id: str):
    """
    Sirve el snippet de preview de un track (MP3 6s mono 64kbps).
    
    El track_id es el fingerprint (MD5 hash del archivo).
    Cache agresivo: el snippet no cambia una vez generado.
    """
    # Validar formato de track_id (MD5 = 32 chars hexadecimales)
    if not track_id or len(track_id) > 64:
        raise HTTPException(400, "track_id inválido")
    
    # Sanitizar para evitar path traversal
    safe_id = re.sub(r'[^a-fA-F0-9]', '', track_id)
    if safe_id != track_id:
        raise HTTPException(400, "track_id contiene caracteres inválidos")
    
    preview_path = os.path.join(PREVIEWS_DIR, f"{safe_id}.mp3")
    
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview not available")
    
    return FileResponse(
        preview_path,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=31536000",  # Cache 1 año
            "Content-Disposition": f"inline; filename={safe_id}_preview.mp3",
        }
    )


class PreviewCheckRequest(BaseModel):
    track_ids: List[str]

@preview_router.post("/previews/check")
async def check_previews(request: PreviewCheckRequest):
    """
    Devuelve qué track_ids tienen preview snippet disponible.
    
    Útil para que el cliente sepa de antemano cuáles puede preescuchar
    sin tener que intentar reproducir y fallar.
    
    Máximo 500 IDs por petición.
    """
    track_ids = request.track_ids
    
    if len(track_ids) > 500:
        raise HTTPException(400, "Máximo 500 track_ids por petición")
    
    available = []
    for tid in track_ids:
        safe_id = re.sub(r'[^a-fA-F0-9]', '', tid)
        if safe_id and os.path.exists(os.path.join(PREVIEWS_DIR, f"{safe_id}.mp3")):
            available.append(tid)
    
    return {
        "available": available,
        "total_checked": len(track_ids),
        "total_available": len(available),
    }


@preview_router.post("/artworks/check")
async def check_artworks(request: PreviewCheckRequest):
    """
    Devuelve qué track_ids ya tienen artwork cacheado en Render.

    Espejo de /previews/check para que el cliente desktop compruebe en UNA
    petición batch (en lugar de un HEAD por archivo) cuáles portadas faltan
    por subir. Máximo 500 IDs por petición.
    """
    track_ids = request.track_ids

    if len(track_ids) > 500:
        raise HTTPException(400, "Máximo 500 track_ids por petición")

    available = []
    for tid in track_ids:
        safe_id = re.sub(r'[^a-fA-F0-9]', '', tid)
        if not safe_id:
            continue
        for ext in ('jpg', 'png', 'jpeg', 'webp', 'gif'):
            if os.path.exists(os.path.join(ARTWORK_CACHE_DIR, f"{safe_id}.{ext}")):
                available.append(tid)
                break

    return {
        "available": available,
        "total_checked": len(track_ids),
        "total_available": len(available),
    }


@preview_router.post("/preview/upload/{track_id}")
async def upload_preview(track_id: str, file: UploadFile = File(...)):
    """
    Recibe un snippet MP3 desde un engine local o desde el cliente Flutter
    y lo guarda en PREVIEWS_DIR para que el resto de dispositivos puedan
    reproducirlo via GET /preview/{track_id}.

    Autenticación: ninguna (por ahora) — el endpoint acepta cualquier MP3
    dentro de los límites de tamaño. El track_id se sanitiza para evitar
    path traversal. Si luego se considera necesario, se puede proteger con
    HMAC o rate-limit.

    Límites:
      - mínimo 100B (filtro de uploads vacíos/corruptos)
      - máximo 500KB (un snippet de 6s a 64kbps son ~48KB; 500KB da
        margen de sobra para compresiones laxas).
    """
    safe_id = re.sub(r'[^a-fA-F0-9]', '', track_id)
    if not safe_id or len(safe_id) > 64:
        raise HTTPException(400, "track_id inválido")

    content = await file.read()
    if len(content) < 100:
        raise HTTPException(400, "Archivo demasiado pequeño")
    if len(content) > 500_000:
        raise HTTPException(400, "Archivo demasiado grande")

    preview_path = os.path.join(PREVIEWS_DIR, f"{safe_id}.mp3")
    os.makedirs(PREVIEWS_DIR, exist_ok=True)

    with open(preview_path, 'wb') as f:
        f.write(content)

    return {"status": "ok", "track_id": safe_id, "size": len(content)}
