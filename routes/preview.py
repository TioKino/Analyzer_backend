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

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sync_endpoints import dispositivo_del_token
from validation import hay_sitio_en_disco

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


# Un snippet de 6 s a 64 kbps son ~48 KB; 500 KB da margen de sobra.
MAX_PREVIEW_BYTES = 500_000
_MIN_PREVIEW_BYTES = 100


def parece_mp3(contenido: bytes) -> bool:
    """¿Empieza como un MP3? La etiqueta ID3v2 («ID3», que es como los escribe
    ffmpeg) o una cabecera de trama MPEG (once bits a 1). No decodifica nada:
    basta para que «un texto cualquiera» no entre como preview."""
    if contenido[:3] == b'ID3':
        return True
    return (len(contenido) >= 2 and contenido[0] == 0xFF
            and (contenido[1] & 0xE0) == 0xE0)


async def _leer_con_tope(file: UploadFile, tope: int) -> bytes:
    """El cuerpo de la subida, por bloques, cortando en cuanto pasa de [tope].

    `await file.read()` a secas trae el fichero ENTERO a memoria antes de
    poder mirar cuánto pesa: con una subida anónima eso es un OOM a la carta.
    """
    trozos = []
    total = 0
    while True:
        trozo = await file.read(64 * 1024)
        if not trozo:
            break
        total += len(trozo)
        if total > tope:
            raise HTTPException(400, "Archivo demasiado grande")
        trozos.append(trozo)
    return b''.join(trozos)


@preview_router.post("/preview/upload/{track_id}")
async def upload_preview(track_id: str, request: Request,
                         file: UploadFile = File(...),
                         solo_si_falta: int = 0):
    """
    Recibe un snippet MP3 desde un engine local o desde el cliente Flutter
    y lo guarda en PREVIEWS_DIR para que el resto de dispositivos puedan
    reproducirlo via GET /preview/{track_id}.

    Las mismas reglas que la portada (`/artwork/upload`), desde el 2026-11-12
    (auditoría, S4; era SEC-05 en agosto y la portada se arregló el 26-sep):

      - **No se pisa sin credencial.** El preview de una huella lo oyen todos
        los que tienen ese fichero, y se sirve con un año de caché. Sin el
        `X-Device-Token` de un aparato registrado la subida vale igual, pero
        como `solo_si_falta`: si ya hay uno, se contesta `exists` y no se
        toca. Los clientes de siempre no notan nada: suben solo lo que
        `/previews/check` dice que falta. Hasta ese día un curl sin nada lo
        cambiaba, y con cualquier cosa dentro.
      - **Tiene que parecer un MP3** (`parece_mp3`).
      - **Se lee por bloques**, con el tope de 500 KB.
      - **No se guarda uno NUEVO sin sitio en el disco** (`hay_sitio_en_disco`,
        507): el id es cualquier cadena hexadecimal, así que sin eso se podía
        llenar `/data`, que es donde viven las dos bases de datos. El cliente
        trata un 5xx como «luego» y no lo da por fallido.

    El track_id se sanitiza para evitar path traversal.
    """
    safe_id = re.sub(r'[^a-fA-F0-9]', '', track_id)
    if not safe_id or len(safe_id) > 64:
        raise HTTPException(400, "track_id inválido")

    quien = None
    if not solo_si_falta:
        quien = await run_in_threadpool(
            dispositivo_del_token, request.headers.get("X-Device-Token", ""))
        if not quien:
            solo_si_falta = 1

    preview_path = os.path.join(PREVIEWS_DIR, f"{safe_id}.mp3")
    habia = os.path.exists(preview_path)
    if solo_si_falta and habia:
        return {"status": "exists", "track_id": safe_id}

    content = await _leer_con_tope(file, MAX_PREVIEW_BYTES)
    if len(content) < _MIN_PREVIEW_BYTES:
        raise HTTPException(400, "Archivo demasiado pequeño")
    if not parece_mp3(content):
        raise HTTPException(400, "formato no soportado (sólo MP3)")

    os.makedirs(PREVIEWS_DIR, exist_ok=True)
    if not habia and not hay_sitio_en_disco(PREVIEWS_DIR):
        logger.warning(f"[Preview] sin sitio en el disco: {safe_id[:8]} no se guarda")
        raise HTTPException(507, "Sin espacio para previews nuevos")

    # Atómico: quien lo esté sirviendo ve el de antes o el nuevo, nunca medio.
    tmp_path = f"{preview_path}.tmp"
    with open(tmp_path, 'wb') as f:
        f.write(content)
    os.replace(tmp_path, preview_path)
    if habia:
        logger.info(f"[Preview] {safe_id} pisado por {quien}")

    return {"status": "ok", "track_id": safe_id, "size": len(content)}
