"""
Admin Panel endpoints for DJ Analyzer Pro.

Provides read-only endpoints for the Flutter AdminScreen to view
users, tracks, previews, sessions, and global statistics.

Auth: X-Admin-Secret header checked against ADMIN_SECRET env var.
"""

import hmac
import json
import logging
import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

_SYNC_DB_PATH = os.environ.get("SYNC_DB_PATH", "/data/sync.db")
_PREVIEWS_DIR = os.environ.get("PREVIEWS_DIR", "previews_cache")


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

            # Count tracks from analysis payloads
            analysis_row = conn.execute(
                "SELECT payload FROM sync_items WHERE data_type = 'analysis' AND last_device_id = ?",
                (device_id,),
            ).fetchone()
            track_count = 0
            preview_count = 0
            if analysis_row:
                tracks = _parse_analysis_payload(analysis_row["payload"])
                track_count = len(tracks)
                for t in tracks.values():
                    fp = t.get("fingerprint", "") if isinstance(t, dict) else ""
                    if _preview_exists(fp):
                        preview_count += 1

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

        # Count tracks and previews across all devices
        total_tracks = 0
        total_previews = 0
        analysis_rows = conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'analysis'"
        ).fetchall()
        for arow in analysis_rows:
            tracks = _parse_analysis_payload(arow["payload"])
            total_tracks += len(tracks)
            for t in tracks.values():
                fp = t.get("fingerprint", "") if isinstance(t, dict) else ""
                if _preview_exists(fp):
                    total_previews += 1

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
        }
    finally:
        conn.close()
