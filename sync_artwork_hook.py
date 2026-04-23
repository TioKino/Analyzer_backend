"""Hook post-sync_push para enriquecer portadas de tracks analizados en local.

El endpoint /analyze del flujo móvil ya busca portadas (ID3 → iTunes → Deezer
→ Last.fm) y las cachea bajo {fingerprint}.jpg. Pero el análisis local (PC/Mac)
no pasa por /analyze — sube directo a /sync/push, que históricamente era un
key/value tonto. Resultado: tracks analizados en PC se sincronizaban al móvil
con preview pero sin portada.

Este módulo se invoca como BackgroundTask tras sync_push. Busca la portada
online y la cachea bajo TODAS las ids conocidas del track (fingerprint,
dict_key del payload, inner id) para sortear el 'NOT IN LIBRARY mismatch'
donde Flutter usa track.id (MD5 de filename|size) y el backend almacena bajo
fingerprint (MD5 de contenido). De este modo /artwork/{id} del backend
responde con la imagen sin importar qué id use el cliente.

No muta el payload sincronizado: la resolución ocurre vía el fallback HTTP
/artwork/{trackId} que Flutter (SmartArtwork / DesktopArtwork) ya invoca
cuando artworkUrl es null. El hash de sync no cambia → no se dispara re-pull
masivo a dispositivos.
"""
import logging
import os
import shutil
from typing import Optional

from config import ARTWORK_CACHE_DIR

logger = logging.getLogger(__name__)

try:
    from artwork_and_cuepoints import search_artwork_online, save_artwork_to_cache
    ARTWORK_ENABLED = True
except ImportError:
    logger.warning("artwork_and_cuepoints no disponible — enrichment de sync deshabilitado")
    search_artwork_online = None
    save_artwork_to_cache = None
    ARTWORK_ENABLED = False


def _artwork_cached(track_id: str) -> bool:
    if not track_id or not ARTWORK_ENABLED:
        return False
    for ext in ("jpg", "jpeg", "png"):
        if os.path.exists(os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")):
            return True
    return False


def _propagate_artwork(source_key: str, target_keys: list) -> None:
    """Copia la portada ya cacheada de source_key a target_keys que aún no la tienen."""
    for ext in ("jpg", "jpeg", "png"):
        src = os.path.join(ARTWORK_CACHE_DIR, f"{source_key}.{ext}")
        if not os.path.exists(src):
            continue
        for tk in target_keys:
            if not tk or tk == source_key:
                continue
            dst = os.path.join(ARTWORK_CACHE_DIR, f"{tk}.{ext}")
            if os.path.exists(dst):
                continue
            try:
                shutil.copyfile(src, dst)
            except OSError as e:
                logger.error(f"No se pudo propagar artwork a {tk[:12]}...: {e}")
        return


def _iter_analysis_tracks(payload) -> list:
    """[(dict_key, track_dict), ...] soportando {"tracks": {...}} y {"track": {...}}."""
    if not isinstance(payload, dict):
        return []
    items = []
    tracks = payload.get("tracks")
    if isinstance(tracks, dict):
        for k, v in tracks.items():
            if isinstance(v, dict):
                items.append((str(k), v))
        return items
    single = payload.get("track")
    if isinstance(single, dict):
        items.append((payload.get("id") or "", single))
    return items


def _extract_artwork_keys(dict_key: str, track: dict):
    """Deriva (cache_keys, artist, title). Dedupe de keys preservando orden."""
    keys = []
    for candidate in (track.get("fingerprint"), dict_key, track.get("id"), track.get("trackId")):
        c = (candidate or "").strip() if isinstance(candidate, str) else ""
        if c and c not in keys:
            keys.append(c)
    artist = (track.get("artist") or "").strip() or None
    title = (track.get("title") or "").strip() or None
    return keys, artist, title


def _enrich_one_track(dict_key: str, track: dict) -> str:
    """'cached' | 'fetched' | 'skipped' | 'failed'."""
    if not ARTWORK_ENABLED:
        return "skipped"
    keys, artist, title = _extract_artwork_keys(dict_key, track)
    if not keys:
        return "skipped"
    existing = next((k for k in keys if _artwork_cached(k)), None)
    if existing:
        others = [k for k in keys if k != existing]
        if others:
            _propagate_artwork(existing, others)
        return "cached"
    if not artist or not title:
        return "skipped"
    try:
        result = search_artwork_online(artist, title, track.get("album"))
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Fallo buscando artwork para '{artist} - {title}': {e}")
        return "failed"
    if not result or not result.get("data"):
        return "failed"
    mime = result.get("mime_type", "image/jpeg")
    primary = keys[0]
    try:
        save_artwork_to_cache(primary, result["data"], mime)
    except OSError as e:
        logger.error(f"No se pudo guardar artwork en {primary[:12]}...: {e}")
        return "failed"
    _propagate_artwork(primary, keys[1:])
    logger.info(f"Artwork enriquecido ({result.get('source')}): {artist} - {title} → {len(keys)} keys")
    return "fetched"


def enrich_analysis_payloads(payloads: list) -> dict:
    """Recorre payloads de analysis y enriquece portadas. Pensado para BackgroundTask."""
    stats = {"scanned": 0, "fetched": 0, "cached": 0, "failed": 0, "skipped": 0}
    for payload in payloads:
        for dict_key, track in _iter_analysis_tracks(payload):
            stats["scanned"] += 1
            status = _enrich_one_track(dict_key, track)
            stats[status] = stats.get(status, 0) + 1
    logger.info(f"Artwork enrichment stats: {stats}")
    return stats


def backfill_from_db(conn, limit: int = 500, user_id: Optional[str] = None) -> dict:
    """Lee sync_items de tipo 'analysis' y los procesa. Útil para tracks previos al hook."""
    import json as _json

    where = "WHERE data_type = 'analysis' AND deleted = 0"
    params = []
    if user_id:
        where += " AND user_id = ?"
        params.append(user_id)
    rows = conn.execute(
        f"SELECT payload FROM sync_items {where} LIMIT ?",
        params + [limit],
    ).fetchall()
    payloads = []
    for row in rows:
        try:
            payloads.append(_json.loads(row[0]))
        except (ValueError, TypeError):
            continue
    if not payloads:
        return {"items_processed": 0, "scanned": 0, "fetched": 0, "cached": 0, "failed": 0, "skipped": 0}
    stats = enrich_analysis_payloads(payloads)
    return {"items_processed": len(payloads), **stats}
