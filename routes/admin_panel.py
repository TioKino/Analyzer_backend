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
            # Errores y resueltos: stubeados hasta que exista la tabla
            # analysis_errors (sprint propio). El cliente Flutter espera
            # estas claves; devolver 0 evita un crash de Pydantic.
            "errors_unresolved": 0,
            "errors_resolved": 0,
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
                # Stub hasta tener analysis_errors table real.
                "errors_unresolved": 0,
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


@admin_panel_router.get("/errors-grouped")
async def errors_grouped(request: Request, resolved: Optional[int] = None):
    await _verify_admin_secret(request)
    resolved_bool = None if resolved is None else bool(resolved)
    groups = _get_db().get_errors_grouped(resolved=resolved_bool)
    return {"groups": groups, "total": len(groups)}


@admin_panel_router.post("/errors/{error_id}/resolve")
async def resolve_error(error_id: int, request: Request):
    await _verify_admin_secret(request)
    new_state = _get_db().toggle_error_resolved(error_id)
    return {"id": error_id, "resolved": new_state}


@admin_panel_router.post("/errors/group/resolve")
async def resolve_error_group_endpoint(request: Request):
    await _verify_admin_secret(request)
    body = await request.json()
    error_class = body.get("error_class", "")
    msg_short = body.get("msg_short", "")
    if not error_class or not msg_short:
        raise HTTPException(400, "error_class y msg_short requeridos")
    n = _get_db().resolve_error_group(error_class, msg_short)
    return {"resolved": n}


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
            r7 = adb.execute(
                "SELECT COUNT(*) FROM audd_call_log "
                "WHERE called_at >= datetime('now','-7 days')"
            ).fetchone()
            audd_last_7d = int(r7[0]) if r7 else 0
            r30 = adb.execute(
                "SELECT COUNT(*) FROM audd_call_log "
                "WHERE called_at >= datetime('now','-30 days')"
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
        artwork_dir = os.environ.get("ARTWORK_CACHE_DIR", "artwork_cache")
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
                if fp and os.path.isdir(artwork_dir):
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

    # 4) Engine source breakdown.
    engine_counts = db.count_engine_sources()
    engine_render = engine_counts.get('render', 0)
    engine_local = engine_counts.get('local_engine', 0)
    engine_total = engine_render + engine_local

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
            "render_pct": round(engine_render / engine_total, 3) if engine_total else None,
            "tracked_total": engine_total,
            "note": (
                None if engine_total else
                "Sin tracks con engine_source seteado (pre-instrumentacion)."
            ),
        },
    }
