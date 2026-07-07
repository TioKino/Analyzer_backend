"""
Cache-lookup + artwork endpoints for DJ Analyzer Pro API.

PASO 5 del troceo de main.py (review 2026-06-29). Bloque movido VERBATIM
desde main.py (mismo comportamiento, sin cambio de logica):

  - POST /check-analyzed                       que filenames ya estan analizados
  - POST /check-analyzed-by-fingerprint        idem por fingerprint (dedup multi-device)
  - GET  /analysis/by-fingerprint/{fingerprint}  hidrata cache local sin re-subir
  - GET  /analysis/{filename:path}             analisis cacheado por filename
  - HEAD /artwork/{track_id}                   pre-check de existencia (sin fallback online)
  - GET  /artwork/{track_id}                   sirve artwork (cache -> fallback online)
  - POST /artwork/upload/{fingerprint}         sube artwork desde el motor local

Dependencias antes globales de main.py se inyectan con init(...). Se inyectan
(no se re-importan) para usar EXACTAMENTE lo que main resolvio:
  - db: instancia AnalysisDB.
  - is_analysis_current: helper de frescura (main.py:_is_analysis_current),
    usado tambien por /analyze -> se queda en main, aqui se inyecta.
  - artwork_cache_dir: ARTWORK_CACHE_DIR resuelto por main (artwork module o
    fallback de config).
  - search_artwork_online / save_artwork_to_cache: funcs de artwork_and_cuepoints
    (None si ARTWORK deshabilitado, igual que main). save_* solo se invoca tras
    `if ... search_artwork_online:`, asi que None nunca se llama.

_artwork_media_type y CheckAnalyzedByFingerprintRequest se mueven con el bloque
(grep confirmo que solo se usaban aqui).

Como pasos 2-4: el router SI se monta (init + include_router) y los endpoints
inline se BORRAN -> sin duplicacion stale.
"""

import logging
import os
import re
from typing import List

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Dependencias inyectadas desde main.py ────────────────────
db = None
_is_analysis_current = None
ARTWORK_CACHE_DIR = None
search_artwork_online = None
save_artwork_to_cache = None


def init(database, is_analysis_current, artwork_cache_dir,
         search_online=None, save_to_cache=None):
    """Inyecta deps desde main.py. Llamar ANTES de include_router(router)."""
    global db, _is_analysis_current, ARTWORK_CACHE_DIR
    global search_artwork_online, save_artwork_to_cache
    db = database
    _is_analysis_current = is_analysis_current
    ARTWORK_CACHE_DIR = artwork_cache_dir
    search_artwork_online = search_online
    save_artwork_to_cache = save_to_cache


# ── Router ───────────────────────────────────────────────────
router = APIRouter(tags=["analysis-artwork"])


def _merge_cluster_best_into(result, acoustic_id):
    """Sobre un dict de analisis, adopta la metadata MAS FIABLE del cluster
    acustico (otra version del mismo audio con fuente superior). Solo sube de
    fiabilidad (compara analysis_ranking); best-effort, nunca lanza."""
    if not acoustic_id or not isinstance(result, dict):
        return
    try:
        from analysis_ranking import get_source_priority
        best = db.best_cluster_analysis(acoustic_id)
        if not best:
            return
        if ('bpm' in best and get_source_priority(best.get('bpm_source'))
                > get_source_priority(result.get('bpm_source'))):
            result['bpm'] = best['bpm']
            result['bpm_source'] = best['bpm_source']
        if ('key' in best and get_source_priority(best.get('key_source'))
                > get_source_priority(result.get('key_source'))):
            result['key'] = best['key']
            result['key_source'] = best['key_source']
            if best.get('camelot'):
                result['camelot'] = best['camelot']
        if ('genre' in best and get_source_priority(best.get('genre_source'))
                > get_source_priority(result.get('genre_source'))):
            result['genre'] = best['genre']
            result['genre_source'] = best['genre_source']
    except Exception as e:  # noqa: BLE001 - best-effort
        logger.warning(f"[Cluster] merge best (lookup) fallo: {e}")


@router.post("/check-analyzed")
async def check_analyzed(filenames: list[str]):
    """Verificar cules tracks ya estn analizados"""
    analyzed = []
    not_analyzed = []

    for filename in filenames:
        existing = db.get_track_by_filename(filename)
        if existing:
            analyzed.append(filename)
        else:
            not_analyzed.append(filename)

    return {
        "analyzed": analyzed,
        "not_analyzed": not_analyzed,
        "total": len(filenames),
        "analyzed_count": len(analyzed),
        "not_analyzed_count": len(not_analyzed)
    }


class CheckAnalyzedByFingerprintRequest(BaseModel):
    fingerprints: List[str]


@router.post("/check-analyzed-by-fingerprint")
async def check_analyzed_by_fingerprint(request: CheckAnalyzedByFingerprintRequest):
    """
    Dedup multi-dispositivo: dado un lote de fingerprints (MD5 del contenido
    del archivo) devuelve cuáles ya están analizados en Render. Esto
    permite que el cliente (especialmente móvil) evite subir y re-analizar
    tracks que ya fueron procesados desde otro dispositivo aunque el nombre
    del fichero sea distinto.

    Máximo 500 IDs por petición.
    """
    fps = request.fingerprints or []
    if len(fps) > 500:
        raise HTTPException(400, "Máximo 500 fingerprints por petición")

    analyzed: list[str] = []
    not_analyzed: list[str] = []
    for fp in fps:
        if not fp:
            continue
        # `get_track_by_fingerprint` ya cubre el caso `id == fingerprint`
        # para registros antiguos donde el id legacy es el propio MD5.
        existing = db.get_track_by_fingerprint(fp)
        if existing and _is_analysis_current(existing):
            analyzed.append(fp)
        else:
            not_analyzed.append(fp)

    return {
        "analyzed": analyzed,
        "not_analyzed": not_analyzed,
        "total": len(fps),
        "analyzed_count": len(analyzed),
        "not_analyzed_count": len(not_analyzed),
    }


@router.get("/analysis/by-fingerprint/{fingerprint}")
async def get_analysis_by_fingerprint(fingerprint: str):
    """Devuelve el análisis cacheado de un track por su fingerprint
    (MD5 del contenido). El cliente puede usar este endpoint tras
    `/check-analyzed-by-fingerprint` para hidratar su cache local sin
    subir el archivo otra vez."""
    safe_fp = re.sub(r'[^a-fA-F0-9]', '', fingerprint or '')
    if not safe_fp:
        raise HTTPException(400, "fingerprint inválido")
    existing = db.get_track_by_fingerprint(safe_fp)
    if not existing:
        raise HTTPException(404, "fingerprint no encontrado")
    raw = existing.get('analysis_json')
    result = None
    if raw:
        try:
            import json
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            result = None
    if result is None:
        # Fallback: construir desde columnas. Incluir *_source para que el
        # cliente sepa si vale la pena sobreescribir su analisis local (ranking
        # "mejor gana" en sync comunitario, item 8).
        result = {
            "id": existing.get('id'),
            "filename": existing.get('filename'),
            "artist": existing.get('artist'),
            "title": existing.get('title'),
            "duration": existing.get('duration') or 0,
            "bpm": existing.get('bpm') or 0,
            "key": existing.get('key'),
            "camelot": existing.get('camelot'),
            "energy_dj": existing.get('energy_dj') or 5,
            "genre": existing.get('genre'),
            "track_type": existing.get('track_type'),
            "fingerprint": existing.get('fingerprint'),
            "bpm_source": existing.get('bpm_source'),
            "key_source": existing.get('key_source'),
            "genre_source": existing.get('genre_source'),
            "engine_source": existing.get('engine_source'),
            "analysis_version": existing.get('analysis_version') or '1',
        }
    # Corregir con la MEJOR metadata del cluster acustico (RETROACTIVO): si otra
    # version del mismo audio (otro usuario) aporto una fuente superior despues
    # de que este track se analizara, el cliente la recibe al re-consultar.
    _merge_cluster_best_into(result, existing.get('acoustic_id'))
    return result


# ==================== ENDPOINTS DE ARTWORK ====================

@router.get("/analysis/{filename:path}")
async def get_analysis(filename: str):
    """Obtener anlisis guardado de un track por filename"""
    # Decodificar filename si viene con URL encoding
    from urllib.parse import unquote
    import json
    filename = unquote(filename)
    
    existing = db.get_track_by_filename(filename)
    if existing:
        # existing es una tupla, convertir a diccionario
        # Columnas: id, filename, artist, title, duration, bpm, key, camelot, 
        #           energy_dj, genre, track_type, analysis_json, analyzed_at, fingerprint
        try:
            # Si hay analysis_json guardado, usarlo directamente
            analysis_json = existing[11]  # ndice de analysis_json
            if analysis_json:
                return json.loads(analysis_json)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("analysis_json cacheado corrupto, fallback a respuesta basica: %s", e)
        
        # Fallback: construir respuesta bsica
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
    
    raise HTTPException(404, f"Anlisis no encontrado para: {filename}")

@router.head("/artwork/{track_id}")
async def head_artwork(track_id: str):
    """HEAD para /artwork/{track_id} - el cliente desktop pre-comprueba
    existencia antes de subir su propio artwork (evita re-upload). Solo
    mira el cache local del disco; NO dispara el fallback online del GET
    (search_artwork_online tiene side effects: red + escritura a cache).
    Devuelve 200 con Content-Type/Content-Length, o 404 sin body.
    """
    for ext in ('jpg', 'png', 'jpeg', 'webp', 'gif'):
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            return Response(
                status_code=200,
                headers={
                    "Content-Type": _artwork_media_type(ext),
                    "Content-Length": str(os.path.getsize(cache_path)),
                },
            )
    raise HTTPException(404, "Artwork no encontrado")


def _artwork_media_type(ext: str) -> str:
    """Mimetype para servir un artwork cacheado según su extensión."""
    return {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'webp': 'image/webp',
        'gif': 'image/gif',
    }.get(ext, 'image/jpeg')


@router.get("/artwork/{track_id}")
async def get_artwork(track_id: str):
    """Devuelve el artwork de un track como imagen.

    Cascade:
      1. Cache local (`{ARTWORK_CACHE_DIR}/{track_id}.{ext}`).
      2. Si la BD tiene el track pero falta el archivo (típicamente
         tracks analizados con motor local cuyo PUSH a Render falló o
         no se hizo), buscamos artwork online (iTunes/Deezer) usando
         artist+title de la BD y lo cacheamos para futuras peticiones.
      3. 404 si nada de lo anterior funciona.
    """
    for ext in ['jpg', 'png', 'jpeg', 'webp', 'gif']:
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            return FileResponse(cache_path, media_type=_artwork_media_type(ext))

    # Fallback: buscar online por artist+title si tenemos el track en BD.
    try:
        existing = db.get_track_by_fingerprint(track_id) or db.get_track_by_id(track_id)
        if existing:
            artist = existing.get('artist')
            title = existing.get('title')
            if artist and title and search_artwork_online:
                logger.info(f"[Artwork] Cache MISS para {track_id[:8]}, buscando online...")
                online = search_artwork_online(artist, title)
                if online and online.get('data'):
                    saved = save_artwork_to_cache(
                        track_id, online['data'], online['mime_type'])
                    saved_path = os.path.join(ARTWORK_CACHE_DIR, saved)
                    media_type = online['mime_type']
                    return FileResponse(saved_path, media_type=media_type)
    except Exception as e:
        logger.warning(f"[Artwork] Fallback online error: {e}")

    raise HTTPException(404, "Artwork no encontrado")


@router.post("/artwork/upload/{fingerprint}")
async def upload_artwork(fingerprint: str, file: UploadFile = File(...)):
    """Recibe artwork desde el local engine para que Render lo sirva
    también a otros devices vía `/artwork/{fingerprint}`. Sin esto,
    cuando el local engine analiza un track el artwork se queda en
    disco PC y los móviles ven placeholder.

    Sanitiza el fingerprint (solo hex 32 chars). Acepta JPEG/PNG.
    Idempotente: re-subir el mismo fp sobreescribe.
    """
    safe_fp = re.sub(r'[^a-fA-F0-9]', '', fingerprint or '')
    if not safe_fp or len(safe_fp) > 64:
        raise HTTPException(400, "fingerprint inválido")

    content = await file.read()
    if not content or len(content) < 100:
        raise HTTPException(400, "archivo vacío o demasiado pequeño")
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(400, "artwork demasiado grande (max 5MB)")

    # Detectar tipo por bytes mágicos. Defaults a jpg si no clarifica.
    # WEBP/GIF se aceptan porque algunas carátulas embebidas vienen en esos
    # formatos; el motor local las escribe como `.jpg` aunque el contenido
    # sea webp/gif, así que sin esto el upload daba 400 en bucle.
    if content[:3] == b'\xff\xd8\xff':
        ext = 'jpg'
    elif content[:8] == b'\x89PNG\r\n\x1a\n':
        ext = 'png'
    elif content[:4] == b'RIFF' and content[8:12] == b'WEBP':
        ext = 'webp'
    elif content[:6] in (b'GIF87a', b'GIF89a'):
        ext = 'gif'
    else:
        # No reconocido — rechazar para no llenar el disco con basura.
        raise HTTPException(400, "formato no soportado (sólo JPEG/PNG/WEBP/GIF)")

    # Eliminar versiones previas con otra extensión para evitar dos
    # archivos del mismo fingerprint en el cache.
    for prev_ext in ('jpg', 'jpeg', 'png', 'webp', 'gif'):
        prev = os.path.join(ARTWORK_CACHE_DIR, f"{safe_fp}.{prev_ext}")
        if os.path.exists(prev):
            try:
                os.unlink(prev)
            except OSError:
                pass

    cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{safe_fp}.{ext}")
    with open(cache_path, 'wb') as f:
        f.write(content)

    return {"status": "ok", "fingerprint": safe_fp, "size": len(content), "ext": ext}
