"""
Admin Panel endpoints for DJ Analyzer Pro.

Read-only endpoints para el panel de desarrollador del Flutter app.
Politica de privacidad por diseño:

- NO exponemos contenido del usuario (filenames, artist/title editados,
  nombres de sesion, cue labels libres, ediciones manuales). Esos datos
  viven en sync.db pero nunca salen por la API admin.
- Exponemos counts agregados, estadisticas globales y errores de
  analisis ya anonimizados (filename hashed por log_analysis_error).
- Si un dia hace falta auditar datos concretos por soporte, debe ser
  un endpoint con doble confirmacion + audit log, no abierto.

Auth: X-Admin-Secret/ADMIN_TOKEN via header. constant-time comparison.
"""

import hmac
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

_SYNC_DB_PATH = os.environ.get("SYNC_DB_PATH", "/data/sync.db")
_PREVIEWS_DIR = os.environ.get("PREVIEWS_DIR", "previews_cache")


# ── Auth dependency ─────────────────────────────────────────

def _get_admin_secret() -> str:
    # Preferencia: ADMIN_TOKEN. Fallback a ADMIN_SECRET para compat.
    return os.environ.get("ADMIN_TOKEN") or os.environ.get("ADMIN_SECRET") or ""


async def _verify_admin_secret(request: Request):
    secret = _get_admin_secret()
    if not secret:
        if os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT"):
            raise HTTPException(500, "ADMIN_SECRET required in production")
        return  # Dev mode local
    header = request.headers.get("X-Admin-Secret", "")
    if not hmac.compare_digest(header, secret):
        raise HTTPException(401, "Invalid or missing X-Admin-Secret")


# ── Router ──────────────────────────────────────────────────

admin_panel_router = APIRouter(
    prefix="/admin",
    tags=["admin-panel"],
    dependencies=[],
)


def _get_sync_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_SYNC_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ── Helpers (counts, no contenido) ──────────────────────────

def _count_in_payload(payload_str: str, data_type: str) -> int:
    """Cuenta items dentro de un payload de sync_items, sin extraer
    nombres ni metadata sensible. Solo numero.
    """
    try:
        payload = json.loads(payload_str)
    except (json.JSONDecodeError, TypeError):
        return 0
    if data_type == "analysis":
        if isinstance(payload, dict):
            inner = payload.get("tracks", payload)
            return len(inner) if isinstance(inner, dict) else 0
        return 0
    if data_type == "session":
        if isinstance(payload, dict):
            return len(payload.get("sessions", []))
        if isinstance(payload, list):
            return len(payload)
        return 0
    if data_type == "cue":
        # cuepoints viven anidados {trackId: [...]}
        try:
            inner = payload.get("cues", payload) if isinstance(payload, dict) else payload
            if isinstance(inner, dict):
                return sum(len(v) if isinstance(v, list) else 0 for v in inner.values())
            if isinstance(inner, list):
                return len(inner)
        except Exception:  # noqa: BLE001
            return 0
        return 0
    # Resto (folder, collection, favorite, override): conteo plano.
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("items", "folders", "collections", "favorites", "overrides"):
            if key in payload and isinstance(payload[key], (list, dict)):
                inner = payload[key]
                return len(inner) if isinstance(inner, (list, dict)) else 0
        return len(payload)
    return 0


def _previews_count_for_payload(payload_str: str) -> int:
    """Cuenta cuantos tracks de un payload analysis tienen preview.
    No expone fingerprints, solo numero.
    """
    try:
        payload = json.loads(payload_str)
    except (json.JSONDecodeError, TypeError):
        return 0
    tracks = payload.get("tracks", payload) if isinstance(payload, dict) else {}
    if not isinstance(tracks, dict):
        return 0
    n = 0
    for t in tracks.values():
        if not isinstance(t, dict):
            continue
        fp = t.get("fingerprint", "")
        if fp and os.path.isfile(os.path.join(_PREVIEWS_DIR, f"{fp}.mp3")):
            n += 1
    return n


# ── GET /admin/users ────────────────────────────────────────

@admin_panel_router.get("/users")
async def list_users(request: Request):
    """Lista de devices (usuarios) con counts agregados.

    NO expone contenido. Solo: device_id, device_type, counts por tipo.
    """
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
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
            track_count = 0
            preview_count = 0
            session_count = 0

            for item in conn.execute(
                "SELECT data_type, payload FROM sync_items WHERE last_device_id = ?",
                (device_id,),
            ).fetchall():
                dt = item["data_type"]
                if dt == "analysis":
                    track_count = _count_in_payload(item["payload"], "analysis")
                    preview_count = _previews_count_for_payload(item["payload"])
                elif dt == "session":
                    session_count = _count_in_payload(item["payload"], "session")

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


# ── GET /admin/users/{device_id}/summary ────────────────────

@admin_panel_router.get("/users/{device_id}/summary")
async def user_summary(device_id: str, request: Request):
    """Counts agregados para un device. NO expone contenido."""
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        rows = conn.execute(
            "SELECT data_type, payload FROM sync_items WHERE last_device_id = ?",
            (device_id,),
        ).fetchall()

        counts: dict = {
            "tracks": 0, "previews": 0, "sessions": 0, "folders": 0,
            "collections": 0, "cues": 0, "favorites": 0, "overrides": 0,
        }

        for r in rows:
            dt = r["data_type"]
            if dt == "analysis":
                counts["tracks"] = _count_in_payload(r["payload"], "analysis")
                counts["previews"] = _previews_count_for_payload(r["payload"])
            elif dt == "session":
                counts["sessions"] = _count_in_payload(r["payload"], "session")
            elif dt == "folder":
                counts["folders"] = _count_in_payload(r["payload"], "folder")
            elif dt == "collection":
                counts["collections"] = _count_in_payload(r["payload"], "collection")
            elif dt == "cue":
                counts["cues"] = _count_in_payload(r["payload"], "cue")
            elif dt == "favorite":
                counts["favorites"] = _count_in_payload(r["payload"], "favorite")
            elif dt == "override":
                counts["overrides"] = _count_in_payload(r["payload"], "override")

        err_row = conn.execute(
            "SELECT COUNT(*) as c FROM analysis_errors "
            "WHERE device_id = ? AND resolved = 0",
            (device_id,),
        ).fetchone()
        counts["errors_unresolved"] = err_row["c"] if err_row else 0

        return {"device_id": device_id, "counts": counts}
    finally:
        conn.close()


# ── GET /admin/stats ────────────────────────────────────────

@admin_panel_router.get("/stats")
async def global_stats(request: Request):
    """Estadisticas globales agregadas. Sin contenido."""
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        device_rows = conn.execute("""
            SELECT last_device_id, device_type
            FROM sync_items
            GROUP BY last_device_id
        """).fetchall()

        total_users = len(device_rows)
        desktop_users = sum(
            1 for r in device_rows
            if (r["device_type"] or "").lower() in ("desktop", "macos", "windows", "linux")
        )
        mobile_users = sum(
            1 for r in device_rows
            if (r["device_type"] or "").lower() in ("mobile", "ios", "android")
        )

        total_tracks = 0
        total_previews = 0
        for arow in conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'analysis'"
        ).fetchall():
            total_tracks += _count_in_payload(arow["payload"], "analysis")
            total_previews += _previews_count_for_payload(arow["payload"])

        total_sessions = 0
        for srow in conn.execute(
            "SELECT payload FROM sync_items WHERE data_type = 'session'"
        ).fetchall():
            total_sessions += _count_in_payload(srow["payload"], "session")

        # Errores: counts globales para dashboard
        err_unresolved = conn.execute(
            "SELECT COUNT(*) as c FROM analysis_errors WHERE resolved = 0"
        ).fetchone()
        err_resolved = conn.execute(
            "SELECT COUNT(*) as c FROM analysis_errors WHERE resolved = 1"
        ).fetchone()

        return {
            "total_users": total_users,
            "total_tracks": total_tracks,
            "total_previews": total_previews,
            "total_sessions": total_sessions,
            "desktop_users": desktop_users,
            "mobile_users": mobile_users,
            "errors_unresolved": err_unresolved["c"] if err_unresolved else 0,
            "errors_resolved": err_resolved["c"] if err_resolved else 0,
        }
    finally:
        conn.close()


# ── ANALYSIS ERRORS ─────────────────────────────────────────
# Filename ya viene anonimizado desde log_analysis_error.

def _opportunistic_cleanup():
    """Llama al cleanup oportunisticamente (1 de cada 50 requests)
    para que la tabla no crezca sin limite. Coste despreciable.
    """
    import random
    if random.random() < 0.02:
        try:
            from sync_endpoints import cleanup_old_errors
            cleanup_old_errors()
        except Exception:  # noqa: BLE001
            pass


@admin_panel_router.get("/errors")
async def list_errors(
    request: Request,
    resolved: Optional[int] = None,
    limit: int = 200,
    since: Optional[str] = None,
):
    """Listado plano de errores ordenado por timestamp desc."""
    await _verify_admin_secret(request)
    _opportunistic_cleanup()
    if limit < 1 or limit > 1000:
        limit = 200

    conn = _get_sync_conn()
    try:
        query = "SELECT * FROM analysis_errors WHERE 1=1"
        params: list = []
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(int(bool(resolved)))
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        errors = [dict(r) for r in rows]

        totals = conn.execute(
            "SELECT resolved, COUNT(*) as c FROM analysis_errors GROUP BY resolved"
        ).fetchall()
        by_state = {int(r["resolved"]): r["c"] for r in totals}

        return {
            "errors": errors,
            "total": len(errors),
            "unresolved_total": by_state.get(0, 0),
            "resolved_total": by_state.get(1, 0),
        }
    finally:
        conn.close()


@admin_panel_router.get("/errors-grouped")
async def list_errors_grouped(
    request: Request,
    resolved: Optional[int] = None,
    limit: int = 100,
):
    """Errores agrupados por (error_class + primera linea de error_msg).

    Cuando un bug afecta a 200 usuarios, el listado plano se vuelve ruido.
    Aqui devolvemos N filas, una por bug distinto, con count y ultimo
    timestamp. Es lo que el dev quiere ver primero.
    """
    await _verify_admin_secret(request)
    _opportunistic_cleanup()
    if limit < 1 or limit > 500:
        limit = 100

    conn = _get_sync_conn()
    try:
        # Agrupa por error_class + 80 primeros chars del msg (estable
        # frente a paths o ids que cambien dentro del msg).
        where = ""
        params: list = []
        if resolved is not None:
            where = "WHERE resolved = ?"
            params.append(int(bool(resolved)))

        rows = conn.execute(
            f"""
            SELECT
                error_class,
                substr(error_msg, 1, 80) as msg_short,
                COUNT(*) as count,
                COUNT(DISTINCT device_id) as devices_affected,
                MAX(timestamp) as last_seen,
                MIN(timestamp) as first_seen,
                MAX(id) as latest_id,
                SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) as unresolved_count
            FROM analysis_errors
            {where}
            GROUP BY error_class, msg_short
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()

        groups = []
        for r in rows:
            # Ejemplo del traceback del registro mas reciente (anonimo
            # ya por log_analysis_error).
            sample = conn.execute(
                "SELECT traceback, error_msg, filename FROM analysis_errors WHERE id = ?",
                (r["latest_id"],),
            ).fetchone()
            groups.append({
                "error_class": r["error_class"],
                "msg_short": r["msg_short"],
                "count": r["count"],
                "devices_affected": r["devices_affected"],
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "latest_id": r["latest_id"],
                "unresolved_count": r["unresolved_count"],
                "sample_msg": sample["error_msg"] if sample else "",
                "sample_traceback": sample["traceback"] if sample else "",
                "sample_filename": sample["filename"] if sample else "",
            })

        return {"groups": groups, "total": len(groups)}
    finally:
        conn.close()


@admin_panel_router.get("/users/{device_id}/errors")
async def user_errors(device_id: str, request: Request,
                      resolved: Optional[int] = None,
                      limit: int = 200):
    """Errores de un device concreto. Filename ya anonimizado en BD."""
    await _verify_admin_secret(request)
    if limit < 1 or limit > 1000:
        limit = 200

    conn = _get_sync_conn()
    try:
        query = "SELECT * FROM analysis_errors WHERE device_id = ?"
        params: list = [device_id]
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(int(bool(resolved)))
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        errors = [dict(r) for r in rows]
        return {"device_id": device_id, "errors": errors, "total": len(errors)}
    finally:
        conn.close()


@admin_panel_router.post("/errors/{error_id}/resolve")
async def resolve_error(error_id: int, request: Request):
    """Toggle resolved flag de un error individual."""
    await _verify_admin_secret(request)
    conn = _get_sync_conn()
    try:
        row = conn.execute(
            "SELECT resolved FROM analysis_errors WHERE id = ?",
            (error_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, f"Error {error_id} no encontrado")
        new_state = 0 if row["resolved"] else 1
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE analysis_errors SET resolved = ?, resolved_at = ? WHERE id = ?",
            (new_state, now_iso if new_state else None, error_id),
        )
        conn.commit()
        return {"id": error_id, "resolved": bool(new_state),
                "resolved_at": now_iso if new_state else None}
    finally:
        conn.close()


@admin_panel_router.post("/errors/group/resolve")
async def resolve_error_group(request: Request):
    """Resuelve todas las filas de un grupo (mismo error_class + msg_short).

    Body JSON: {"error_class": "...", "msg_short": "..."}
    Util cuando arreglas un bug y quieres cerrar las 200 ocurrencias
    de un golpe.
    """
    await _verify_admin_secret(request)
    body = await request.json()
    error_class = body.get("error_class") or ""
    msg_short = body.get("msg_short") or ""
    if not error_class or not msg_short:
        raise HTTPException(400, "error_class y msg_short requeridos")

    conn = _get_sync_conn()
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """UPDATE analysis_errors
               SET resolved = 1, resolved_at = ?
               WHERE error_class = ?
                 AND substr(error_msg, 1, 80) = ?
                 AND resolved = 0""",
            (now_iso, error_class, msg_short),
        )
        conn.commit()
        return {"resolved": cur.rowcount, "resolved_at": now_iso}
    finally:
        conn.close()
