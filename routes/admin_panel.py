"""
Admin Panel endpoints for DJ Analyzer Pro.

Provides read-only endpoints for the Flutter AdminScreen to view
users, tracks, previews, sessions, and global statistics.

Auth: X-Admin-Secret header checked against ADMIN_SECRET env var.
"""

import csv
import hmac
import io
import json
import logging
import os
import sqlite3
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Importar paths desde config (unica fuente de verdad). Antes leiamos
# ENV var con default relativo, lo que en Render hacia que buscaramos
# en directorios que NO eran los persistentes (/data/...). Resultado:
# total_previews=0 aunque /data/previews tuviera 800 mp3s.
from config import (  # noqa: E402  (import dinamico tras logger)
    PREVIEWS_DIR as _PREVIEWS_DIR,
    ARTWORK_CACHE_DIR as _ARTWORK_CACHE_DIR,
)

_SYNC_DB_PATH = os.environ.get("SYNC_DB_PATH", "/data/sync.db")


# ── Auth dependency ─────────────────────────────────────────

def _get_admin_secret() -> str:
    # Preferencia: ADMIN_TOKEN (env var unica para todos los admin endpoints).
    # Fallback a ADMIN_SECRET para compat legacy — los deploys mas viejos la
    # tenian. Si los dos estan, gana ADMIN_TOKEN.
    return os.environ.get("ADMIN_TOKEN") or os.environ.get("ADMIN_SECRET") or ""


async def _verify_admin_secret(request: Request):
    """Verify X-Admin-Secret header against ADMIN_SECRET env var."""
    secret = _get_admin_secret()
    if not secret:
        if os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT"):
            raise HTTPException(500, "ADMIN_SECRET required in production")
        return  # Dev mode: no auth
    header = request.headers.get("X-Admin-Secret", "")
    # Constant-time comparison para evitar timing attack (finding B-H1 que
    # se habia pasado por alto en este router; sync_endpoints y routes/admin
    # si lo tenian aplicado desde el AUDIT 2026-04-20).
    if not hmac.compare_digest(header, secret):
        raise HTTPException(401, "Invalid or missing X-Admin-Secret")


# ── Router ──────────────────────────────────────────────────

admin_panel_router = APIRouter(
    prefix="/admin",
    tags=["admin-panel"],
    dependencies=[],  # auth applied per-endpoint via _verify_admin_secret
)


def _get_sync_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_SYNC_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ── Helpers ─────────────────────────────────────────────────

def _parse_analysis_payload(payload_str: str) -> dict:
    """Parse an analysis payload, returning the tracks dict."""
    try:
        payload = json.loads(payload_str)
        if isinstance(payload, dict):
            return payload.get("tracks", payload)
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _compute_telemetry_from_sync(conn) -> dict:
    """Itera sync_items WHERE data_type='analysis' UNA sola vez y devuelve
    fingerprint_stats + sources_breakdown + total_users + platforms.

    Justificacion: los tracks "reales" de los usuarios viven en sync.db
    (payload de sync_items), NO en analysis.db. La BD de analysis solo
    se llena cuando un cliente analiza directamente contra /analyze de
    Render — pero la mayoria usa motor local y solo sincroniza. Por eso
    las metricas tienen que mirar aqui para reflejar la realidad.

    Tolerante a payloads camelCase (cliente Flutter via cloud_sync) y
    snake_case (re-uploads desde local_engine).
    """
    fp_total = 0
    fp_with = 0
    fp_seen: dict[str, int] = {}  # fingerprint -> count para detectar colisiones
    sources = {
        'bpm': {},
        'key': {},
        'genre': {},
        'track_type': {},
    }
    # Artwork breakdown: cuantos tracks tienen artwork conseguido via el
    # flow de analisis (ID3 embedded, iTunes, Deezer, LastFM, Discogs).
    # NO mide el cache server-side de /data/artwork_cache (eso es otra
    # cosa). Esta metrica refleja la realidad del USUARIO: cuantos de sus
    # tracks lograron artwork con cualquier fuente.
    artwork_total = 0
    artwork_embedded_count = 0
    artwork_url_only_count = 0  # tiene URL pero no embedded (online)
    artwork_sources: dict[str, int] = {}

    def _get(track: dict, *keys: str):
        """Devuelve el primer valor truthy entre las keys dadas."""
        for k in keys:
            v = track.get(k)
            if v not in (None, '', 0):
                return v
        return None

    def _bump(bucket: dict, key, value):
        if value is None or value == '':
            value = 'unknown'
        bucket[str(value)] = bucket.get(str(value), 0) + 1

    # Devices: contar sobre TODOS los data_types (consistente con /admin/stats),
    # no solo 'analysis' — un device puede sincronizar sesiones/favoritos sin
    # haber analizado nada. Antes contaba solo 'analysis', que daba 44 frente a
    # los 50 de /admin/stats.
    distinct_devices: set[str] = set()
    device_types: dict[str, str] = {}  # device_id -> device_type
    for drow in conn.execute(
        "SELECT last_device_id, device_type FROM sync_items GROUP BY last_device_id"
    ).fetchall():
        ddid = drow["last_device_id"]
        if ddid:
            distinct_devices.add(ddid)
            if drow["device_type"]:
                device_types[ddid] = drow["device_type"]

    # Metricas de tracks/sources/fingerprints/artwork: solo de payloads 'analysis'.
    rows = conn.execute(
        "SELECT payload FROM sync_items WHERE data_type = 'analysis'"
    ).fetchall()
    for row in rows:
        tracks = _parse_analysis_payload(row["payload"])
        for t in tracks.values():
            if not isinstance(t, dict):
                continue
            fp_total += 1

            fp = _get(t, 'fingerprint')
            if fp:
                fp_with += 1
                fp_seen[str(fp)] = fp_seen.get(str(fp), 0) + 1

            _bump(sources['bpm'], None, _get(t, 'bpm_source', 'bpmSource'))
            _bump(sources['key'], None, _get(t, 'key_source', 'keySource'))
            _bump(sources['genre'], None, _get(t, 'genre_source', 'genreSource'))
            _bump(sources['track_type'], None,
                  _get(t, 'track_type_source', 'trackTypeSource'))

            # Artwork: cuenta tracks con cualquier fuente de artwork.
            # Tolerante a shapes camelCase (cliente) y snake_case (backend).
            art_url = _get(t, 'artwork_url', 'artworkUrl')
            art_embed = bool(
                t.get('artwork_embedded') or t.get('hasArtworkEmbedded')
            )
            art_source = _get(t, 'artwork_source', 'artworkSource')
            if art_url or art_embed:
                artwork_total += 1
                if art_embed:
                    artwork_embedded_count += 1
                elif art_url:
                    artwork_url_only_count += 1
                # Determinar fuente: si declara art_source la usamos, sino
                # inferimos 'id3' para embedded y 'unknown' para URL suelta.
                src = art_source or ('id3' if art_embed else 'unknown')
                artwork_sources[str(src)] = artwork_sources.get(str(src), 0) + 1

    # Colisiones: cuantos fingerprints aparecen >1 vez y cuantas filas extra
    # aportan en total.
    collision_groups = sum(1 for n in fp_seen.values() if n > 1)
    collision_extras = sum((n - 1) for n in fp_seen.values() if n > 1)

    # Platforms sobre TODOS los devices distintos (incluye los que no declaran
    # device_type, como 'unknown') -> la suma cuadra con total_users.
    platforms: dict[str, int] = {}
    for did in distinct_devices:
        key = (device_types.get(did) or 'unknown').lower()
        platforms[key] = platforms.get(key, 0) + 1

    return {
        'fingerprints': {
            'total_tracks': fp_total,
            'with_fingerprint': fp_with,
            'without_fingerprint': fp_total - fp_with,
            'unique_fingerprints': len(fp_seen),
            'collision_groups': collision_groups,
            'collision_extra_rows': collision_extras,
        },
        'sources': sources,
        'total_users': len(distinct_devices),
        'platforms': platforms,
        'artwork_coverage': {
            'total_tracks': fp_total,
            'with_artwork': artwork_total,
            'with_artwork_embedded': artwork_embedded_count,
            'with_artwork_url_only': artwork_url_only_count,
            'rate': round(artwork_total / fp_total, 3) if fp_total else 0.0,
            'sources': artwork_sources,
        },
    }


def _parse_session_payload(payload_str: str) -> list:
    """Parse a session payload, returning the sessions list."""
    try:
        payload = json.loads(payload_str)
        if isinstance(payload, dict):
            return payload.get("sessions", [])
        if isinstance(payload, list):
            return payload
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _preview_exists(fingerprint: str) -> bool:
    """Check if a preview MP3 exists for the given fingerprint."""
    if not fingerprint:
        return False
    path = os.path.join(_PREVIEWS_DIR, f"{fingerprint}.mp3")
    return os.path.isfile(path)


def _count_unique_tracks(analysis_rows, check_previews: bool = True):
    """Cuenta tracks UNICOS a partir de filas sync_items (data_type='analysis').

    Maneja los DOS formatos que coexisten en produccion:

      1. Incremental (v2.9.3+): cada fila es UN track, item_key=<trackId>,
         payload = el track suelto (un dict de campos: id, bpm, key, ...).
         NO viene envuelto en {"tracks": {...}}. La identidad del track es
         directamente el item_key. OJO: pasar este payload por
         _parse_analysis_payload + .items() iteraria los CAMPOS del track
         como si cada uno fuera un track => inflaba el contador a cientos de
         miles. Por eso se trata por separado mirando el item_key.

      2. Blob legacy (clientes < v2.9.3): item_key='all_analysis',
         payload = {"tracks": {trackId: {...}}}. Aqui si se itera el dict.

    Identidad canonica = track.id (chromaprint MD5, idempotente al
    filename/tags/re-codec e identico cross-device para el mismo archivo),
    con fallback a fingerprint y al item_key / key del dict. Esto deduplica
    el mismo track sincronizado desde varios devices (PC+Mac+movil) y entre
    el blob legacy y las filas incrementales.

    Devuelve (seen_ids:set, preview_fps:set, no_id_tracks:int).
    """
    seen_ids: set = set()
    preview_fps: set = set()
    no_id_tracks = 0

    def _maybe_preview(fp):
        if check_previews and fp and fp not in preview_fps and _preview_exists(fp):
            preview_fps.add(fp)

    for arow in analysis_rows:
        try:
            ikey = arow["item_key"] or ""
        except (KeyError, IndexError, TypeError):
            ikey = ""
        raw = arow["payload"]

        if ikey and ikey != "all_analysis":
            # Incremental: una fila = un track. item_key ES el trackId.
            fp = None
            try:
                pl = json.loads(raw)
                if isinstance(pl, dict):
                    fp = pl.get("fingerprint") or pl.get("id")
            except (json.JSONDecodeError, TypeError):
                pass
            seen_ids.add(str(fp or ikey))
            _maybe_preview(fp)
            continue

        # Blob legacy all_analysis: iterar el dict de tracks.
        tracks = _parse_analysis_payload(raw)
        for tkey, t in tracks.items():
            ident = None
            fp = None
            if isinstance(t, dict):
                ident = t.get("id") or t.get("fingerprint")
                fp = t.get("fingerprint", "")
            ident = ident or tkey
            if ident:
                seen_ids.add(str(ident))
            else:
                no_id_tracks += 1
            _maybe_preview(fp)

    return seen_ids, preview_fps, no_id_tracks


# ── GET /admin/users ────────────────────────────────────────

@admin_panel_router.get("/users")
async def list_users(request: Request):
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        # Get all distinct devices from sync_items
        rows = conn.execute("""
            SELECT last_device_id, device_type,
                   MAX(updated_at) as last_sync_at,
                   MIN(updated_at) as first_seen_at
            FROM sync_items
            GROUP BY last_device_id
        """).fetchall()

        users = []
        for row in rows:
            device_id = row["last_device_id"]

            # Count tracks from analysis payloads. Usa fetchall() + helper:
            # con sync incremental v2.9.3 un device tiene MUCHAS filas analysis
            # (una por track, item_key=<trackId>), no una sola. El antiguo
            # fetchone()+len(tracks) devolvia basura para esos usuarios (contaba
            # los CAMPOS de un unico track). _count_unique_tracks dedup-ea por id
            # y maneja blob legacy + incremental por igual.
            analysis_rows = conn.execute(
                "SELECT item_key, payload FROM sync_items WHERE data_type = 'analysis' AND last_device_id = ?",
                (device_id,),
            ).fetchall()
            u_ids, u_prev, u_noid = _count_unique_tracks(analysis_rows)
            track_count = len(u_ids) + u_noid
            preview_count = len(u_prev)

            # Count sessions
            session_row = conn.execute(
                "SELECT payload FROM sync_items WHERE data_type = 'session' AND last_device_id = ?",
                (device_id,),
            ).fetchone()
            session_count = 0
            if session_row:
                sessions = _parse_session_payload(session_row["payload"])
                session_count = len(sessions)

            users.append({
                "device_id": device_id,
                "device_type": row["device_type"] or "unknown",
                "track_count": track_count,
                "preview_count": preview_count,
                "session_count": session_count,
                "last_sync_at": row["last_sync_at"],
                "first_seen_at": row["first_seen_at"],
            })

        return {"users": users, "total": len(users)}
    finally:
        conn.close()


# ── GET /admin/users/{device_id}/tracks ─────────────────────

@admin_panel_router.get("/users/{device_id}/tracks")
async def user_tracks(device_id: str, request: Request):
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        # Get analysis data — could be uploaded by this device or by any device
        # We search all analysis items and look for tracks belonging to this device
        row = conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'analysis' AND last_device_id = ?",
            (device_id,),
        ).fetchone()

        if not row:
            return {"tracks": [], "total": 0}

        tracks_map = _parse_analysis_payload(row["payload"])
        result = []
        base_url = os.environ.get("BASE_URL", "").rstrip("/")

        for track_id, t in tracks_map.items():
            if not isinstance(t, dict):
                continue
            fingerprint = t.get("fingerprint", "")
            has_preview = _preview_exists(fingerprint)

            # Build artwork URL
            artwork_url = t.get("artworkUrl", "")
            if not artwork_url and fingerprint and base_url:
                artwork_url = f"{base_url}/artwork/{fingerprint}"

            # Extract nested fields
            track_info = t.get("track", t)
            tempo_info = t.get("tempo", t)
            key_info = t.get("key", t)
            energy_info = t.get("energy", t)

            result.append({
                "track_id": track_id,
                "file_name": track_info.get("fileName", t.get("fileName", "")),
                "artist": track_info.get("artist", t.get("artist", "")),
                "title": track_info.get("title", t.get("title", "")),
                "bpm": tempo_info.get("bpm", t.get("bpm")),
                "key": key_info.get("key", t.get("key", "")),
                "camelot": key_info.get("camelot", t.get("camelot", "")),
                "genre": t.get("genre", ""),
                "energy": energy_info.get("energy_dj", t.get("energy_dj", t.get("energy"))),
                "track_type": t.get("trackType", t.get("track_type", "")),
                "analyzed_at": t.get("analyzedAt", t.get("analyzed_at", "")),
                "has_preview": has_preview,
                "artwork_url": artwork_url,
            })

        return {"tracks": result, "total": len(result)}
    finally:
        conn.close()


# ── GET /admin/users/{device_id}/previews ───────────────────

@admin_panel_router.get("/users/{device_id}/previews")
async def user_previews(device_id: str, request: Request):
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        row = conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'analysis' AND last_device_id = ?",
            (device_id,),
        ).fetchone()

        if not row:
            return {"previews": [], "total": 0}

        tracks_map = _parse_analysis_payload(row["payload"])
        base_url = os.environ.get("BASE_URL", "").rstrip("/")
        result = []

        for track_id, t in tracks_map.items():
            if not isinstance(t, dict):
                continue
            fingerprint = t.get("fingerprint", "")
            if not _preview_exists(fingerprint):
                continue

            track_info = t.get("track", t)
            preview_url = f"{base_url}/preview/{fingerprint}" if base_url else ""

            result.append({
                "track_id": track_id,
                "file_name": track_info.get("fileName", t.get("fileName", "")),
                "artist": track_info.get("artist", t.get("artist", "")),
                "title": track_info.get("title", t.get("title", "")),
                "preview_url": preview_url,
            })

        return {"previews": result, "total": len(result)}
    finally:
        conn.close()


# ── GET /admin/users/{device_id}/sessions ───────────────────

@admin_panel_router.get("/users/{device_id}/sessions")
async def user_sessions(device_id: str, request: Request):
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        row = conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'session' AND last_device_id = ?",
            (device_id,),
        ).fetchone()

        if not row:
            return {"sessions": [], "total": 0}

        sessions = _parse_session_payload(row["payload"])
        return {"sessions": sessions, "total": len(sessions)}
    finally:
        conn.close()


# ── GET /admin/stats ────────────────────────────────────────

# ── GET /admin/all-tracks ───────────────────────────────────
# Admin endpoint: devuelve TODOS los análisis de TODOS los usuarios

@admin_panel_router.get("/all-tracks")
async def all_tracks(request: Request):
    """Return all analyzed tracks from all users (for admin user)."""
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        analysis_rows = conn.execute(
            "SELECT last_device_id, device_type, payload FROM sync_items WHERE data_type = 'analysis'"
        ).fetchall()

        base_url = os.environ.get("BASE_URL", "").rstrip("/")
        result = []
        seen_fingerprints = set()  # Deduplicar por fingerprint

        for arow in analysis_rows:
            device_id = arow["last_device_id"]
            device_type = arow["device_type"] or "unknown"
            tracks = _parse_analysis_payload(arow["payload"])

            for track_id, t in tracks.items():
                if not isinstance(t, dict):
                    continue
                fingerprint = t.get("fingerprint", "")

                # Deduplicar: si ya vimos este fingerprint, skip
                if fingerprint and fingerprint in seen_fingerprints:
                    continue
                if fingerprint:
                    seen_fingerprints.add(fingerprint)

                has_preview = _preview_exists(fingerprint)
                artwork_url = t.get("artworkUrl", "")
                if not artwork_url and fingerprint and base_url:
                    artwork_url = f"{base_url}/artwork/{fingerprint}"

                track_info = t.get("track", t)
                tempo_info = t.get("tempo", t)
                key_info = t.get("key", t)
                energy_info = t.get("energy", t)

                result.append({
                    "track_id": track_id,
                    "fingerprint": fingerprint,
                    "file_name": track_info.get("fileName", t.get("fileName", "")),
                    "artist": track_info.get("artist", t.get("artist", "")),
                    "title": track_info.get("title", t.get("title", "")),
                    "bpm": tempo_info.get("bpm", t.get("bpm")),
                    "key": key_info.get("key", t.get("key", "")),
                    "camelot": key_info.get("camelot", t.get("camelot", "")),
                    "genre": t.get("genre", ""),
                    "energy": energy_info.get("energy_dj", t.get("energy_dj", t.get("energy"))),
                    "track_type": t.get("trackType", t.get("track_type", "")),
                    "analyzed_at": t.get("analyzedAt", t.get("analyzed_at", "")),
                    "has_preview": has_preview,
                    "artwork_url": artwork_url,
                    "preview_url": f"{base_url}/preview/{fingerprint}" if has_preview and base_url else "",
                    "owner_device": device_id,
                    "owner_device_type": device_type,
                })

        return {"tracks": result, "total": len(result)}
    finally:
        conn.close()


# ── GET /admin/all-previews ────────────────────────────────

@admin_panel_router.get("/all-previews")
async def all_previews(request: Request):
    """Return all available previews from all users (for admin user)."""
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        analysis_rows = conn.execute(
            "SELECT last_device_id, payload FROM sync_items WHERE data_type = 'analysis'"
        ).fetchall()

        base_url = os.environ.get("BASE_URL", "").rstrip("/")
        result = []
        seen_fingerprints = set()

        for arow in analysis_rows:
            tracks = _parse_analysis_payload(arow["payload"])

            for track_id, t in tracks.items():
                if not isinstance(t, dict):
                    continue
                fingerprint = t.get("fingerprint", "")
                if not fingerprint or fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)

                if not _preview_exists(fingerprint):
                    continue

                track_info = t.get("track", t)
                result.append({
                    "track_id": track_id,
                    "fingerprint": fingerprint,
                    "artist": track_info.get("artist", t.get("artist", "")),
                    "title": track_info.get("title", t.get("title", "")),
                    "preview_url": f"{base_url}/preview/{fingerprint}" if base_url else "",
                })

        return {"previews": result, "total": len(result)}
    finally:
        conn.close()


# ── GET /admin/stats ────────────────────────────────────────

@admin_panel_router.get("/stats")
async def global_stats(request: Request):
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        # Distinct devices
        device_rows = conn.execute("""
            SELECT last_device_id, device_type
            FROM sync_items
            GROUP BY last_device_id
        """).fetchall()

        total_users = len(device_rows)
        desktop_users = sum(1 for r in device_rows if (r["device_type"] or "").lower() in ("desktop", "macos", "windows", "linux"))
        mobile_users = sum(1 for r in device_rows if (r["device_type"] or "").lower() in ("mobile", "ios", "android"))

        # Count tracks and previews across all devices, DEDUPLICADO por track.id.
        # Antes se hacia total_tracks += len(tracks) por cada fila de sync_items,
        # lo que contaba la MISMA cancion una vez por cada device que la sincroniza
        # (PC + Mac + movil = x3). Peor aun: con la sync incremental v2.9.3 cada
        # fila es UN track suelto (item_key=<trackId>, payload sin wrapper
        # "tracks"), y al pasarla por _parse_analysis_payload().items() se
        # iteraban sus CAMPOS como si fueran tracks => el contador se disparaba a
        # cientos de miles. _count_unique_tracks maneja ambos formatos y dedup-ea
        # por track.id (chromaprint MD5, identico cross-device). Ver su docstring.
        analysis_rows = conn.execute(
            "SELECT item_key, payload FROM sync_items WHERE data_type = 'analysis'"
        ).fetchall()
        seen_ids, preview_fps, no_id_tracks = _count_unique_tracks(analysis_rows)
        total_tracks = len(seen_ids) + no_id_tracks
        total_previews = len(preview_fps)

        # Count sessions
        total_sessions = 0
        session_rows = conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'session'"
        ).fetchall()
        for srow in session_rows:
            sessions = _parse_session_payload(srow["payload"])
            total_sessions += len(sessions)

        return {
            "total_users": total_users,
            "total_tracks": total_tracks,
            "total_previews": total_previews,
            "total_sessions": total_sessions,
            "desktop_users": desktop_users,
            "mobile_users": mobile_users,
            # Conteos reales desde analysis_errors (analysis.db). Antes
            # estaban hardcodeados a 0 (la tabla no existia); ya existe, asi
            # que el panel mostraba siempre 0 errores aunque los hubiera.
            "errors_unresolved": _get_db().count_errors(resolved=False),
            "errors_resolved": _get_db().count_errors(resolved=True),
        }
    finally:
        conn.close()


# ── GET /admin/users/{device_id}/summary ────────────────────
# Counts agregados por usuario (privacy-first: solo numeros, sin
# nombres de archivo, artistas ni titulos). Reemplaza el granular
# /admin/users/{id}/{tracks,previews,sessions} que el cliente Flutter
# privacy-first ya no consume.

@admin_panel_router.get("/users/{device_id}/summary")
async def user_summary(device_id: str, request: Request):
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        def _count_payload(data_type: str, key: str = "") -> int:
            row = conn.execute(
                "SELECT payload FROM sync_items WHERE data_type = ? AND last_device_id = ?",
                (data_type, device_id),
            ).fetchone()
            if not row:
                return 0
            try:
                data = json.loads(row["payload"])
            except (json.JSONDecodeError, TypeError):
                return 0
            if isinstance(data, dict):
                if key:
                    inner = data.get(key, data)
                else:
                    inner = data
                if isinstance(inner, dict):
                    return len(inner)
                if isinstance(inner, list):
                    return len(inner)
            if isinstance(data, list):
                return len(data)
            return 0

        tracks = _count_payload("analysis", "tracks")
        sessions = _count_payload("session")
        folders = _count_payload("folder")
        collections = _count_payload("collection")
        cues = _count_payload("cue")
        favorites = _count_payload("favorite")
        overrides = _count_payload("override")

        # Previews: contar fingerprints del usuario que tienen .mp3 cacheado
        previews = 0
        arow = conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'analysis' AND last_device_id = ?",
            (device_id,),
        ).fetchone()
        if arow:
            tracks_map = _parse_analysis_payload(arow["payload"])
            for t in tracks_map.values():
                fp = t.get("fingerprint", "") if isinstance(t, dict) else ""
                if _preview_exists(fp):
                    previews += 1

        return {
            "device_id": device_id,
            "counts": {
                "tracks": tracks,
                "previews": previews,
                "sessions": sessions,
                "folders": folders,
                "collections": collections,
                "cues": cues,
                "favorites": favorites,
                "overrides": overrides,
                # Conteo real de errores sin resolver de este device.
                "errors_unresolved": _get_db().count_errors(
                    resolved=False, device_id=device_id),
            },
        }
    finally:
        conn.close()


# ── Errors endpoints ───────────────────────────────────────
# Consultan la tabla analysis_errors via los helpers de AnalysisDB.
# Lazy-import del modulo main para evitar circular imports a load time.

def _get_db():
    """Lazy import del singleton AnalysisDB definido en main.py."""
    import main
    return main.db


@admin_panel_router.get("/users/{device_id}/errors")
async def user_errors(device_id: str, request: Request, resolved: Optional[int] = None):
    await _verify_admin_secret(request)
    resolved_bool = None if resolved is None else bool(resolved)
    errors = _get_db().get_analysis_errors(
        device_id=device_id, resolved=resolved_bool, limit=200,
    )
    return {"device_id": device_id, "errors": errors, "total": len(errors)}


@admin_panel_router.get("/errors")
async def global_errors(request: Request, resolved: Optional[int] = None, limit: int = 200):
    await _verify_admin_secret(request)
    resolved_bool = None if resolved is None else bool(resolved)
    errors = _get_db().get_analysis_errors(resolved=resolved_bool, limit=limit)
    return {"errors": errors, "total": len(errors)}


def _norm_since(s: Optional[str]) -> Optional[str]:
    """Normaliza una fecha de query al formato de la columna `timestamp`
    ('YYYY-MM-DD HH:MM:SS', UTC). Acepta 'YYYY-MM-DD' (compara igual)."""
    return s.strip() if s else None


def _norm_until(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    # Fecha sin hora: hacer el rango inclusivo hasta el final del dia.
    return (s + ' 23:59:59') if len(s) == 10 else s


@admin_panel_router.get("/errors-grouped")
async def errors_grouped(
    request: Request,
    resolved: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
):
    await _verify_admin_secret(request)
    resolved_bool = None if resolved is None else bool(resolved)
    groups = _get_db().get_errors_grouped(
        resolved=resolved_bool, since=_norm_since(since), until=_norm_until(until),
    )
    return {"groups": groups, "total": len(groups), "since": since, "until": until}


@admin_panel_router.get("/errors-grouped.csv")
async def errors_grouped_csv(
    request: Request,
    resolved: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
):
    """Igual que /errors-grouped pero como CSV descargable, para bajar el
    historial de un dia/rango y analizarlo offline. Ej:
      curl -H "X-Admin-Secret: $ADMIN_TOKEN" \\
        "https://<render>/admin/errors-grouped.csv?since=2026-05-26&until=2026-05-27" -o errores.csv
    """
    await _verify_admin_secret(request)
    resolved_bool = None if resolved is None else bool(resolved)
    groups = _get_db().get_errors_grouped(
        resolved=resolved_bool, since=_norm_since(since), until=_norm_until(until),
    )
    cols = [
        'error_class', 'origin', 'platform', 'app_version', 'human_message',
        'count', 'devices_affected', 'unresolved_count', 'first_seen',
        'last_seen', 'context', 'clean_msg', 'sample_filename',
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction='ignore')
    writer.writeheader()
    for g in groups:
        writer.writerow({k: ('' if g.get(k) is None else g.get(k)) for k in cols})
    fname = f"errores_{since or 'all'}_{until or 'now'}.csv".replace(' ', '_').replace(':', '')
    return Response(
        content=buf.getvalue(),
        media_type='text/csv; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename="{fname}"'},
    )


@admin_panel_router.post("/errors/prune")
async def prune_errors(request: Request, days: int = 180, only_resolved: int = 1):
    """Purga errores viejos (por defecto solo resueltos de mas de `days` dias)
    para que la tabla no crezca sin limite. Manual a proposito: el backend NO
    auto-borra telemetria por su cuenta."""
    await _verify_admin_secret(request)
    n = _get_db().prune_old_errors(days=days, only_resolved=bool(only_resolved))
    return {"deleted": n, "days": days, "only_resolved": bool(only_resolved)}


@admin_panel_router.post("/engine-source/backfill")
async def backfill_engine_source(
    request: Request, dry_run: int = 1, value: str = "local_engine"
):
    """Sella engine_source de los tracks historicos sin etiquetar (NULL ->
    'local_engine', que es de donde vienen casi todos via /cache-analysis).
    `dry_run=1` (default) SOLO inspecciona: devuelve cuantos NULL hay y su
    reparto por bpm_source. `dry_run=0` aplica el UPDATE. Ej:
      curl -H "X-Admin-Secret: $ADMIN_TOKEN" -X POST \\
        ".../admin/engine-source/backfill?dry_run=1"   # ver antes
      curl -H "X-Admin-Secret: $ADMIN_TOKEN" -X POST \\
        ".../admin/engine-source/backfill?dry_run=0"   # aplicar
    """
    await _verify_admin_secret(request)
    return _get_db().backfill_engine_source(value=value, dry_run=bool(dry_run))


# Orden importante: /errors/group/resolve va ANTES que /errors/{error_id}/resolve.
# FastAPI usa first-match routing, asi que con el orden inverso un POST a
# /errors/group/resolve matcheaba la ruta {error_id} con error_id="group",
# fallaba int_parsing y devolvia 422 sin nunca llegar a este handler.
# Era el origen de los 422 que aparecen en el log del panel admin.
@admin_panel_router.post("/errors/group/resolve")
async def resolve_error_group_endpoint(request: Request):
    await _verify_admin_secret(request)
    # await request.json() lanza JSONDecodeError -> FastAPI lo convierte a
    # 422 sin contexto util. Cliente real envia application/json valido,
    # pero el panel a veces hace POST vacio (botones doble click, sondas).
    # Devolvemos 400 con mensaje explicito en lugar del 422 generico.
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(400, "Body debe ser JSON valido")
    if not isinstance(body, dict):
        raise HTTPException(400, "Body debe ser un objeto JSON")
    error_class = body.get("error_class", "")
    msg_short = body.get("msg_short", "")  # puede ser '' para errores sin mensaje
    if not error_class:
        raise HTTPException(400, "error_class requerido")
    # msg_short puede ser vacio (errores sin mensaje como ClientDisconnect).
    # En ese caso resolve_error_group resuelve por error_class sin filtro de msg.
    n = _get_db().resolve_error_group(error_class, msg_short)
    return {"resolved": n}


@admin_panel_router.post("/errors/{error_id}/resolve")
async def resolve_error(error_id: int, request: Request):
    await _verify_admin_secret(request)
    new_state = _get_db().toggle_error_resolved(error_id)
    return {"id": error_id, "resolved": new_state}


# ── GET /admin/telemetry ────────────────────────────────────
# Snapshot agregado privacy-first para tu panel. Cuenta llamadas a
# AudD (success/fail) desde audd_call_log y le suma cobertura de
# previews y artwork sobre el total de tracks. NO devuelve filenames,
# artistas ni titulos.

@admin_panel_router.get("/telemetry")
async def telemetry(request: Request):
    await _verify_admin_secret(request)
    # 1) AudD stats desde la BD de analisis. La tabla vive en analysis.db
    # (no sync.db), por eso abrimos otra conexion.
    analysis_db_path = os.environ.get(
        "DATABASE_PATH",
        os.environ.get("ANALYSIS_DB_PATH", "analysis.db"),
    )
    audd_total = 0
    audd_success = 0
    audd_last_7d = 0
    audd_last_30d = 0
    if os.path.exists(analysis_db_path):
        adb = sqlite3.connect(f"file:{analysis_db_path}?mode=ro", uri=True)
        try:
            r = adb.execute(
                "SELECT COUNT(*), COALESCE(SUM(success),0) FROM audd_call_log"
            ).fetchone()
            if r:
                audd_total, audd_success = int(r[0] or 0), int(r[1] or 0)
            # called_at es REAL (UNIX timestamp) — comparamos con timestamps,
            # no con datetime('now',...) que devuelve string. Verificado
            # 2026-05-13: la query antigua daba siempre 0 porque comparaba
            # numero vs string lexicograficamente.
            import time as _time
            now_ts = _time.time()
            r7 = adb.execute(
                "SELECT COUNT(*) FROM audd_call_log WHERE called_at >= ?",
                (now_ts - 7 * 86400,),
            ).fetchone()
            audd_last_7d = int(r7[0]) if r7 else 0
            r30 = adb.execute(
                "SELECT COUNT(*) FROM audd_call_log WHERE called_at >= ?",
                (now_ts - 30 * 86400,),
            ).fetchone()
            audd_last_30d = int(r30[0]) if r30 else 0
        except sqlite3.OperationalError:
            # Tabla no existe en BDs antiguas — skip silencioso.
            pass
        finally:
            adb.close()
    audd_success_rate = (audd_success / audd_total) if audd_total else 0.0

    # 2) Cobertura preview + artwork sobre el total de tracks sync.
    conn = _get_sync_conn()
    try:
        total_tracks = 0
        with_preview = 0
        with_artwork = 0
        # _ARTWORK_CACHE_DIR viene de config (persistente /data/ en Render).
        # Antes leiamos env var directa con default 'artwork_cache' relativo,
        # asi que en Render contaba siempre 0 aunque hubiera artwork en disco.
        artwork_dir = _ARTWORK_CACHE_DIR
        artwork_dir_exists = os.path.isdir(artwork_dir)
        arows = conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'analysis'"
        ).fetchall()
        for arow in arows:
            tracks_map = _parse_analysis_payload(arow["payload"])
            for t in tracks_map.values():
                if not isinstance(t, dict):
                    continue
                total_tracks += 1
                fp = t.get("fingerprint", "")
                if _preview_exists(fp):
                    with_preview += 1
                # Artwork file convencion: <fp>.jpg / <fp>.png
                if fp and artwork_dir_exists:
                    for ext in (".jpg", ".jpeg", ".png", ".webp"):
                        if os.path.exists(os.path.join(artwork_dir, f"{fp}{ext}")):
                            with_artwork += 1
                            break
    finally:
        conn.close()
    preview_rate = (with_preview / total_tracks) if total_tracks else 0.0
    artwork_rate = (with_artwork / total_tracks) if total_tracks else 0.0

    # 3) Errores agregados de analysis_errors via AnalysisDB helpers.
    db = _get_db()
    unresolved_errs = db.get_analysis_errors(resolved=False, limit=999999)
    resolved_errs = db.get_analysis_errors(resolved=True, limit=999999)
    # Top-N grupos de errores no resueltos para que el panel los muestre
    # sin tener que hacer otra request.
    top_groups = db.get_errors_grouped(resolved=False)[:5]

    # 4) Engine source breakdown. 'unknown' = tracks sin engine_source sellado
    # (entraron via /cache-analysis antes del fix). Se exponen aparte en vez de
    # ocultarse, que era lo que hacia parecer "todo render".
    engine_counts = db.count_engine_sources()
    engine_render = engine_counts.get('render', 0)
    engine_local = engine_counts.get('local_engine', 0)
    engine_unknown = engine_counts.get('unknown', 0)
    engine_tagged = engine_render + engine_local
    engine_total = engine_tagged + engine_unknown

    # 5/7/8) Fingerprints + sources + usuarios/plataformas: TODO sale de
    # sync.db, no de analysis.db. La razon: los usuarios analizan en
    # local_engine y suben el resultado via /sync; analysis.db solo se
    # llena cuando alguien analiza directamente contra Render, que es
    # un caso minoritario. _compute_telemetry_from_sync hace UNA pasada
    # sobre sync_items y deriva todas las metricas a la vez.
    sync_telemetry = {}
    try:
        sconn2 = _get_sync_conn()
        try:
            sync_telemetry = _compute_telemetry_from_sync(sconn2)
        finally:
            sconn2.close()
    except sqlite3.OperationalError:
        sync_telemetry = {
            'fingerprints': {
                'total_tracks': 0, 'with_fingerprint': 0,
                'without_fingerprint': 0, 'unique_fingerprints': 0,
                'collision_groups': 0, 'collision_extra_rows': 0,
            },
            'sources': {'bpm': {}, 'key': {}, 'genre': {}, 'track_type': {}},
            'total_users': 0,
            'platforms': {},
        }
    fp_stats = sync_telemetry['fingerprints']
    sources_breakdown = sync_telemetry['sources']
    total_users = sync_telemetry['total_users']
    platforms = sync_telemetry['platforms']
    artwork_coverage_real = sync_telemetry['artwork_coverage']
    # total_devices = usuarios distintos vistos en sync (mismo dato que
    # total_users porque por ahora cada device es un user). En el futuro,
    # cuando se introduzca user_id agrupando varios devices, separar.
    total_devices = sum(platforms.values()) if platforms else total_users

    # 6) Errores del cliente / no manejados en las ultimas 24h.
    client_errors_24h = db.count_client_errors_by_context(since_hours=24)

    # Flag visible: hay token configurado en este entorno? Permite al panel
    # decir "AudD no configurado" en vez de un 0% misterioso si Render no
    # tiene la env var.
    audd_token_present = bool(os.environ.get("AUDD_API_TOKEN", "").strip())
    audd_auto_enabled = (os.environ.get("AUDD_AUTO_ENABLED", "true").lower() == "true")

    return {
        "audd": {
            "configured": audd_token_present,
            "auto_enabled": audd_auto_enabled,
            "total_calls": audd_total,
            "success_calls": audd_success,
            "fail_calls": audd_total - audd_success,
            "success_rate": round(audd_success_rate, 3),
            "calls_last_7d": audd_last_7d,
            "calls_last_30d": audd_last_30d,
        },
        "coverage": {
            "total_tracks": total_tracks,
            "with_preview": with_preview,
            "with_artwork": with_artwork,
            "preview_rate": round(preview_rate, 3),
            "artwork_rate": round(artwork_rate, 3),
        },
        "errors": {
            "unresolved": len(unresolved_errs),
            "resolved": len(resolved_errs),
            "top_groups": [
                {
                    "error_class": g["error_class"],
                    "msg_short": g["msg_short"],
                    "count": g["count"],
                    "devices_affected": g["devices_affected"],
                    "last_seen": g["last_seen"],
                }
                for g in top_groups
            ],
        },
        "engine_source": {
            "render": engine_render,
            "local_engine": engine_local,
            "unknown": engine_unknown,
            "render_pct": round(engine_render / engine_tagged, 3) if engine_tagged else None,
            "tagged_total": engine_tagged,
            "total": engine_total,
            "note": (
                "Los 'unknown' son tracks sin engine_source sellado (entraron "
                "via /cache-analysis antes del fix). Los analisis nuevos ya se "
                "reparten en render/local_engine."
                if engine_unknown else None
            ),
        },
        # "fingerprints" informa la decision de invertir en Hamming distance
        # (item 9 PENDING). collision_extra_rows > 0 significa que el dedup
        # actual exact-match esta dejando entrar tracks duplicados — si
        # esa cifra crece, vale la pena el threshold de Hamming.
        "fingerprints": {
            "total_tracks": fp_stats.get('total_tracks', 0),
            "with_fingerprint": fp_stats.get('with_fingerprint', 0),
            "without_fingerprint": fp_stats.get('without_fingerprint', 0),
            "unique_fingerprints": fp_stats.get('unique_fingerprints', 0),
            "collision_groups": fp_stats.get('collision_groups', 0),
            "collision_extra_rows": fp_stats.get('collision_extra_rows', 0),
            "coverage_pct": (
                round(fp_stats.get('with_fingerprint', 0) / fp_stats.get('total_tracks', 1), 3)
                if fp_stats.get('total_tracks', 0) > 0 else None
            ),
        },
        # "client_errors_24h": contadores por context para que el panel pinte
        # un health-check rapido. _unhandled = errores no manejados (middleware
        # global). El resto son contexts que reporto Flutter via /client-error.
        "client_errors_24h": client_errors_24h,
        # "sources": breakdown de bpm/key/genre/track_type sources en la BD.
        # Permite calcular % "fiable" en el cliente sin enviar datos pesados.
        "sources": sources_breakdown,
        # "users": numero total + dispositivos por plataforma. Privacidad:
        # solo conteos agregados, jamas device_ids.
        "users": {
            "total_users": total_users,
            "total_devices": total_devices,
            "platforms": platforms,
        },
        # "artwork_real": cobertura de artwork conseguido por el flow de
        # analisis (ID3 embedded + iTunes/Deezer/LastFM/Discogs en /analyze)
        # leyendo los payloads de sync. NO es el cache server-side del HDD
        # persistente (eso esta en coverage.with_artwork y siempre sera
        # bajo porque solo se sube cuando el motor local lo decide). Esta
        # metrica refleja "del total de tracks de los usuarios, cuantos
        # tienen ALGUNA fuente de artwork".
        "artwork_real": artwork_coverage_real,
    }


# ── GET /admin/activity ────────────────────────────────────
# Feed cronologico de actividad reciente (analisis + AudD + errores) para ver
# "que esta pasando ahora" desde el panel SIN abrir los logs de Render. Solo
# LECTURA sobre tablas que ya existen (tracks.analyzed_at, audd_call_log,
# analysis_errors) -> cero coste en el hot-path, sin tablas nuevas.
#
# Privacy: el owner es el unico con el ADMIN_TOKEN. A diferencia del resto del
# panel (que oculta artist/title), este feed SI muestra artist/title del
# analisis: es justo la info operativa que el owner ya ve en los logs de Render
# y que este endpoint reemplaza. Los errores siguen anonimizados (el filename
# va hasheado en analysis_errors y aqui ni se devuelve).

@admin_panel_router.get("/activity")
async def activity(request: Request):
    await _verify_admin_secret(request)
    import time as _time
    from datetime import datetime

    try:
        limit = max(1, min(200, int(request.query_params.get("limit", "60"))))
    except (TypeError, ValueError):
        limit = 60

    analysis_db_path = os.environ.get(
        "DATABASE_PATH",
        os.environ.get("ANALYSIS_DB_PATH", "analysis.db"),
    )
    now_ts = _time.time()
    _EPOCH = datetime(1970, 1, 1)

    def _iso_to_epoch(s):
        """'YYYY-MM-DDTHH:MM:SS[.ffffff]' o 'YYYY-MM-DD HH:MM:SS' tratado como
        UTC naive -> epoch float. None si no parsea. (Render corre en UTC, asi
        que datetime.now() alli ya es UTC: el delta con now_ts es correcto.)"""
        if not s:
            return None
        try:
            txt = str(s).strip().replace("T", " ")[:19]
            return (datetime.strptime(txt, "%Y-%m-%d %H:%M:%S") - _EPOCH).total_seconds()
        except (ValueError, TypeError):
            return None

    events = []
    pulse = {
        "analyses_10m": 0, "analyses_1h": 0,
        "audd_10m": 0, "audd_1h": 0,
        "errors_10m": 0, "errors_1h": 0,
        "last_analysis_epoch": None,
    }

    if os.path.exists(analysis_db_path):
        adb = sqlite3.connect(f"file:{analysis_db_path}?mode=ro", uri=True)
        adb.row_factory = sqlite3.Row
        try:
            cut10_iso = datetime.utcfromtimestamp(now_ts - 600).isoformat()
            cut60_iso = datetime.utcfromtimestamp(now_ts - 3600).isoformat()
            cut10_sql = datetime.utcfromtimestamp(now_ts - 600).strftime("%Y-%m-%d %H:%M:%S")
            cut60_sql = datetime.utcfromtimestamp(now_ts - 3600).strftime("%Y-%m-%d %H:%M:%S")

            # --- Analisis recientes (tracks.analyzed_at, ISO TEXT) ---
            try:
                for r in adb.execute(
                    "SELECT analyzed_at, artist, title, bpm, genre, engine_source "
                    "FROM tracks WHERE analyzed_at IS NOT NULL "
                    "ORDER BY analyzed_at DESC LIMIT ?", (limit,)
                ).fetchall():
                    ep = _iso_to_epoch(r["analyzed_at"])
                    if ep is None:
                        continue
                    bpm = float(r["bpm"]) if r["bpm"] is not None else 0.0
                    events.append({
                        "kind": "analysis",
                        "ts_epoch": ep,
                        "ok": bpm > 0,  # bpm=0 = fallback "pendiente/fallido"
                        "title": (r["title"] or r["artist"] or "—"),
                        "artist": r["artist"] or "",
                        "bpm": round(bpm, 1) if bpm > 0 else None,
                        "genre": r["genre"] or "",
                        "engine": r["engine_source"] or "",
                    })
                pulse["analyses_10m"] = adb.execute(
                    "SELECT COUNT(*) FROM tracks WHERE analyzed_at >= ?", (cut10_iso,)
                ).fetchone()[0]
                pulse["analyses_1h"] = adb.execute(
                    "SELECT COUNT(*) FROM tracks WHERE analyzed_at >= ?", (cut60_iso,)
                ).fetchone()[0]
                pulse["last_analysis_epoch"] = _iso_to_epoch(
                    adb.execute("SELECT MAX(analyzed_at) FROM tracks").fetchone()[0]
                )
            except sqlite3.OperationalError:
                pass

            # --- AudD recientes (audd_call_log.called_at, epoch REAL) ---
            try:
                for r in adb.execute(
                    "SELECT called_at, artist, title, success FROM audd_call_log "
                    "ORDER BY called_at DESC LIMIT ?", (limit,)
                ).fetchall():
                    if r["called_at"] is None:
                        continue
                    events.append({
                        "kind": "audd",
                        "ts_epoch": float(r["called_at"]),
                        "ok": bool(r["success"]),
                        "title": (r["title"] or r["artist"] or "?"),
                        "artist": r["artist"] or "",
                    })
                pulse["audd_10m"] = adb.execute(
                    "SELECT COUNT(*) FROM audd_call_log WHERE called_at >= ?", (now_ts - 600,)
                ).fetchone()[0]
                pulse["audd_1h"] = adb.execute(
                    "SELECT COUNT(*) FROM audd_call_log WHERE called_at >= ?", (now_ts - 3600,)
                ).fetchone()[0]
            except sqlite3.OperationalError:
                pass

            # --- Errores recientes (analysis_errors.timestamp 'YYYY-MM-DD HH:MM:SS') ---
            try:
                for r in adb.execute(
                    "SELECT timestamp, error_class, msg_short, endpoint "
                    "FROM analysis_errors ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall():
                    ep = _iso_to_epoch(r["timestamp"])
                    if ep is None:
                        continue
                    events.append({
                        "kind": "error",
                        "ts_epoch": ep,
                        "ok": False,
                        "title": r["error_class"] or "Error",
                        "msg": r["msg_short"] or "",
                        "endpoint": r["endpoint"] or "",
                    })
                pulse["errors_10m"] = adb.execute(
                    "SELECT COUNT(*) FROM analysis_errors WHERE timestamp >= ?", (cut10_sql,)
                ).fetchone()[0]
                pulse["errors_1h"] = adb.execute(
                    "SELECT COUNT(*) FROM analysis_errors WHERE timestamp >= ?", (cut60_sql,)
                ).fetchone()[0]
            except sqlite3.OperationalError:
                pass
        finally:
            adb.close()

    # Merge + orden cronologico descendente, recorte al limite global.
    events.sort(key=lambda e: e["ts_epoch"], reverse=True)
    events = events[:limit]

    return {
        "server_now_epoch": now_ts,
        "pulse": pulse,
        "events": events,
    }


# ── GET /admin/disk-usage ──────────────────────────────────
# Breakdown del disco persistente /data/ en Render: tamano de cada BD,
# count y tamano de previews/, count y tamano de artwork_cache/, total
# usado vs libre. Util para diagnosticar uso de HDD persistente sin SSH.

def _du_path(path: str) -> dict:
    """Devuelve {'bytes': total_size, 'files': file_count} para un path
    (archivo o directorio). Si no existe, ceros."""
    if not os.path.exists(path):
        return {'bytes': 0, 'files': 0, 'exists': False}
    if os.path.isfile(path):
        return {'bytes': os.path.getsize(path), 'files': 1, 'exists': True}
    total_bytes = 0
    total_files = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total_bytes += os.path.getsize(fp)
                    total_files += 1
                except OSError:
                    pass
    except OSError:
        pass
    return {'bytes': total_bytes, 'files': total_files, 'exists': True}


def _fmt_mb(b: int) -> float:
    return round(b / (1024 * 1024), 1)


@admin_panel_router.get("/disk-usage")
async def disk_usage(request: Request):
    """Reporta uso del disco persistente /data/ con breakdown por
    subdirectorio. Privacy-first: solo tamanos y conteos, jamas nombres
    de archivos o contenido. Util para diagnosticar:
    - Si artwork_cache esta vacio (problema de upload) o lleno.
    - Si los previews crecen segun esperado.
    - Cuanto pesan las dos SQLite (analysis.db, sync.db).
    - Espacio libre restante del disco persistente.
    """
    await _verify_admin_secret(request)

    # Paths principales. Usar los mismos que la app: importados de config.
    analysis_db_path = os.environ.get(
        'DATABASE_PATH',
        os.environ.get('ANALYSIS_DB_PATH', '/data/analysis.db'),
    )
    sync_db_path = _SYNC_DB_PATH

    breakdown = {
        'analysis_db': _du_path(analysis_db_path),
        'sync_db': _du_path(sync_db_path),
        'previews': _du_path(_PREVIEWS_DIR),
        'artwork_cache': _du_path(_ARTWORK_CACHE_DIR),
    }

    # Listar contenido top-level de /data/ para detectar archivos huerfanos
    # (logs, .tmp, backups inesperados). Solo nombres y tamanos.
    data_root_contents = []
    data_root = '/data'
    if os.path.isdir(data_root):
        try:
            for entry in sorted(os.listdir(data_root)):
                full = os.path.join(data_root, entry)
                if os.path.isdir(full):
                    info = _du_path(full)
                    data_root_contents.append({
                        'name': entry + '/',
                        'kind': 'dir',
                        'mb': _fmt_mb(info['bytes']),
                        'files': info['files'],
                    })
                else:
                    try:
                        sz = os.path.getsize(full)
                    except OSError:
                        sz = 0
                    data_root_contents.append({
                        'name': entry,
                        'kind': 'file',
                        'mb': _fmt_mb(sz),
                        'files': 1,
                    })
        except OSError:
            pass

    # Disco completo del FS donde vive /data/
    free_mb = 0
    total_mb = 0
    used_mb = 0
    try:
        import shutil as _sh
        du = _sh.disk_usage('/data' if os.path.isdir('/data') else '/')
        total_mb = _fmt_mb(du.total)
        used_mb = _fmt_mb(du.used)
        free_mb = _fmt_mb(du.free)
    except OSError:
        pass

    return {
        'totals': {
            'disk_total_mb': total_mb,
            'disk_used_mb': used_mb,
            'disk_free_mb': free_mb,
            'used_pct': round(used_mb / total_mb * 100, 1) if total_mb else None,
        },
        'breakdown': {
            'analysis_db_mb': _fmt_mb(breakdown['analysis_db']['bytes']),
            'sync_db_mb': _fmt_mb(breakdown['sync_db']['bytes']),
            'previews': {
                'mb': _fmt_mb(breakdown['previews']['bytes']),
                'files': breakdown['previews']['files'],
                'avg_kb': round(
                    breakdown['previews']['bytes'] / breakdown['previews']['files'] / 1024, 1
                ) if breakdown['previews']['files'] else None,
                'path': _PREVIEWS_DIR,
            },
            'artwork_cache': {
                'mb': _fmt_mb(breakdown['artwork_cache']['bytes']),
                'files': breakdown['artwork_cache']['files'],
                'avg_kb': round(
                    breakdown['artwork_cache']['bytes'] / breakdown['artwork_cache']['files'] / 1024, 1
                ) if breakdown['artwork_cache']['files'] else None,
                'path': _ARTWORK_CACHE_DIR,
            },
        },
        # data_root_contents: lista top-level de /data/ para detectar
        # cualquier cosa fuera de los 4 esperados (analysis.db, sync.db,
        # previews/, artwork_cache/). Si aparecen .log, .tmp, backups o
        # algo raro -> alerta.
        'data_root_contents': data_root_contents,
    }
