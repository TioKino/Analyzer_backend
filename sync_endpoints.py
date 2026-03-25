# ============================================================================
# SYNC ENDPOINTS v5.0 — DJ Analyzer Pro
# ============================================================================
# CAMBIOS vs v4.0:
#   1. PERSISTENCIA SQLite — ya no se pierden datos al reiniciar Render
#   2. FIX _compute_detail — dispositivo nuevo usa categorías "_initial"
#      en vez de "_added" para que el diálogo no mienta
#   3. Reutiliza la ruta /data/ que ya usa el resto del backend
#
# SINGLE SOURCE OF TRUTH — Last-write-wins
# Claves: "data_type|item_key" (sin device_id)
# Cada item guarda quién lo subió último (last_device_id) y cuándo.
# ============================================================================

import hmac as _hmac
import logging

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Optional
from datetime import datetime, timezone
import json, hashlib, sqlite3, os

from config import SYNC_AUTH_SECRET

logger = logging.getLogger(__name__)

sync_router = APIRouter(
    prefix="/sync",
    tags=["sync"],
    dependencies=[Depends(_verify_sync_auth)],
)


# ── HMAC-SHA256 Auth ─────────────────────────────────────────
# Si SYNC_AUTH_SECRET está configurado, todos los endpoints de sync
# requieren header X-Signature con HMAC-SHA256(secret, body).
# Si NO está configurado, auth se desactiva (modo desarrollo).

async def _verify_sync_auth(request: Request):
    """Dependency de FastAPI que valida HMAC si el secret está configurado."""
    if not SYNC_AUTH_SECRET:
        return  # Dev mode: sin auth

    body = await request.body()
    signature = request.headers.get("X-Signature", "")

    if not signature:
        raise HTTPException(status_code=401, detail="Missing X-Signature header")

    expected = _hmac.new(
        SYNC_AUTH_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    if not _hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")

# ── SQLite persistente ───────────────────────────────────────

_DB_PATH = os.environ.get("SYNC_DB_PATH", "/data/sync.db")
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _init_tables(_conn)
    return _conn


def _init_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sync_items (
            key         TEXT PRIMARY KEY,
            data_type   TEXT NOT NULL,
            item_key    TEXT NOT NULL,
            payload     TEXT NOT NULL,
            deleted     INTEGER DEFAULT 0,
            updated_at  TEXT NOT NULL,
            last_device_id TEXT NOT NULL,
            device_type TEXT DEFAULT 'unknown',
            hash        TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sync_items_device
            ON sync_items(last_device_id);
        CREATE INDEX IF NOT EXISTS idx_sync_items_type
            ON sync_items(data_type);

        CREATE TABLE IF NOT EXISTS device_seen (
            device_id TEXT NOT NULL,
            item_key  TEXT NOT NULL,
            hash      TEXT NOT NULL,
            payload   TEXT,
            PRIMARY KEY (device_id, item_key)
        );
    """)
    conn.commit()


# ── Helpers ──────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_hash(payload) -> str:
    normalized = _normalize_for_hash(payload)
    raw = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _normalize_for_hash(obj):
    if isinstance(obj, dict):
        skip_keys = {"filePath", "artworkUrl"}
        return {k: _normalize_for_hash(v) for k, v in obj.items() if k not in skip_keys}
    if isinstance(obj, list):
        return [_normalize_for_hash(i) for i in obj]
    return obj


class SyncChange(BaseModel):
    data_type: str
    item_key: str
    payload: Any
    deleted: bool = False
    updated_at: Optional[str] = None


class PushRequest(BaseModel):
    device_id: str
    device_type: str = "unknown"
    changes: list[SyncChange]


# ── PUSH ─────────────────────────────────────────────────────

@sync_router.post("/push")
async def sync_push(req: PushRequest):
    """Push sobreescribe la verdad del backend. Last-write-wins."""
    conn = _get_conn()
    synced = 0
    skipped = 0
    now = _now_iso()

    for change in req.changes:
        key = f"{change.data_type}|{change.item_key}"
        change_time = change.updated_at or now
        new_hash = _payload_hash(change.payload)
        payload_json = json.dumps(change.payload, default=str)

        # Si ya existe con el mismo hash, skip
        row = conn.execute(
            "SELECT hash FROM sync_items WHERE key = ?", (key,)
        ).fetchone()
        if row and row[0] == new_hash:
            skipped += 1
            # Igual actualizar device_seen con payload
            conn.execute(
                """INSERT INTO device_seen (device_id, item_key, hash, payload)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(device_id, item_key) DO UPDATE SET hash = ?, payload = ?""",
                (req.device_id, key, new_hash, payload_json, new_hash, payload_json),
            )
            continue

        # Upsert del item
        conn.execute(
            """INSERT INTO sync_items
                   (key, data_type, item_key, payload, deleted,
                    updated_at, last_device_id, device_type, hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   payload = excluded.payload,
                   deleted = excluded.deleted,
                   updated_at = excluded.updated_at,
                   last_device_id = excluded.last_device_id,
                   device_type = excluded.device_type,
                   hash = excluded.hash""",
            (key, change.data_type, change.item_key, payload_json,
             1 if change.deleted else 0, change_time,
             req.device_id, req.device_type, new_hash),
        )
        synced += 1

        # Registrar que este dispositivo conoce este hash + payload
        conn.execute(
            """INSERT INTO device_seen (device_id, item_key, hash, payload)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(device_id, item_key) DO UPDATE SET hash = ?, payload = ?""",
            (req.device_id, key, new_hash, payload_json, new_hash, payload_json),
        )

    conn.commit()
    return {"synced": synced, "skipped": skipped, "conflicts": [], "timestamp": now}


# ── PULL ─────────────────────────────────────────────────────

@sync_router.get("/pull/{device_id}")
async def sync_pull(
    device_id: str,
    since: Optional[str] = None,
    only_from: Optional[str] = None,
):
    """Descarga items cuyo hash sea DIFERENTE al que este dispositivo conoce."""
    conn = _get_conn()

    # Obtener todos los items que NO subió este dispositivo
    rows = conn.execute(
        """SELECT si.key, si.data_type, si.item_key, si.payload, si.deleted,
                  si.updated_at, si.device_type, si.hash
           FROM sync_items si
           WHERE si.last_device_id != ?""",
        (device_id,),
    ).fetchall()

    changes = []
    update_seen = []

    for row in rows:
        key, data_type, item_key, payload_json, deleted, updated_at, device_type, item_hash = row

        # Verificar si el dispositivo ya conoce este hash
        seen = conn.execute(
            "SELECT hash FROM device_seen WHERE device_id = ? AND item_key = ?",
            (device_id, key),
        ).fetchone()
        if seen and seen[0] == item_hash:
            continue

        changes.append({
            "data_type":   data_type,
            "item_key":    item_key,
            "payload":     json.loads(payload_json),
            "deleted":     bool(deleted),
            "updated_at":  updated_at,
            "device_type": device_type,
        })
        update_seen.append((device_id, key, item_hash))

    # Marcar como vistos con payload
    for dev_id, key, h in update_seen:
        # Buscar el payload del item para guardarlo en device_seen
        row = conn.execute("SELECT payload FROM sync_items WHERE key = ?", (key,)).fetchone()
        payload_to_save = row[0] if row else None
        conn.execute(
            """INSERT INTO device_seen (device_id, item_key, hash, payload)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(device_id, item_key) DO UPDATE SET hash = ?, payload = ?""",
            (dev_id, key, h, payload_to_save, h, payload_to_save),
        )
    conn.commit()

    return {
        "changes":   changes,
        "total":     len(changes),
        "alerts":    [],
        "timestamp": _now_iso(),
    }


# ── PENDING ──────────────────────────────────────────────────

@sync_router.get("/pending/{device_id}")
async def sync_pending(device_id: str, since: Optional[str] = None):
    """Desglose detallado: añadidos, eliminados, modificados por tipo.
    
    Usa device_seen para saber qué versión conoce este dispositivo,
    y compara con el estado actual del backend para calcular el diff real.
    """
    conn = _get_conn()

    # Migración: añadir columna payload si no existe
    try:
        conn.execute("ALTER TABLE device_seen ADD COLUMN payload TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Ya existe

    # Items remotos (no subidos por este dispositivo)
    remote_rows = conn.execute(
        """SELECT si.key, si.data_type, si.item_key, si.payload, si.hash
           FROM sync_items si
           WHERE si.last_device_id != ?""",
        (device_id,),
    ).fetchall()

    detail: dict[str, int] = {}
    total = 0

    for row in remote_rows:
        key, data_type, item_key, payload_json, item_hash = row

        # Buscar qué versión conoce este dispositivo (de device_seen)
        seen = conn.execute(
            "SELECT hash, payload FROM device_seen WHERE device_id = ? AND item_key = ?",
            (device_id, key),
        ).fetchone()

        # Si ya conoce este hash exacto → no hay cambio
        if seen and seen[0] == item_hash:
            continue

        try:
            remote_payload = json.loads(payload_json)
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

        # Si tiene payload guardado en device_seen → diff real
        # Si no tiene nada → initial (primera vez que ve este tipo)
        my_payload = None
        is_initial = True
        if seen and seen[1]:
            try:
                my_payload = json.loads(seen[1])
                is_initial = False
            except (json.JSONDecodeError, TypeError, KeyError):
                pass  # TODO: handle specifically

        changes = _compute_detail(
            data_type, my_payload, remote_payload,
            is_initial=is_initial,
        )
        for change_key, count in changes.items():
            if count > 0:
                detail[change_key] = detail.get(change_key, 0) + count
                total += count

    return {
        "total":       total,
        "summary":     detail,
        "has_pending": total > 0,
    }


# ── COMPUTE DETAIL (con fix v5) ─────────────────────────────

def _compute_detail(
    data_type: str,
    local_payload,
    remote_payload,
    is_initial: bool = False,
) -> dict[str, int]:
    """Calcula añadidos/eliminados/modificados entre local y remoto.

    Si is_initial=True, el dispositivo nunca subió este tipo,
    así que no podemos hacer diff real. Usamos categoría "_initial"
    para que el diálogo Flutter muestre "Sincronización inicial"
    en vez de mentir con "X añadidos".
    """
    result: dict[str, int] = {}

    if remote_payload is None:
        return result
    if not isinstance(remote_payload, dict):
        result[f"{data_type}_modified"] = 1
        return result

    # ── Dispositivo sin datos de este tipo → initial ──
    if is_initial:
        count = _count_items_in(data_type, remote_payload)
        if count > 0:
            result[f"{data_type}_initial"] = count
        return result

    if not isinstance(local_payload, dict):
        result[f"{data_type}_modified"] = 1
        return result

    # ── Diff real por tipo ──
    if data_type == "analysis":
        local_keys = set((local_payload.get("tracks") or {}).keys())
        remote_keys = set((remote_payload.get("tracks") or {}).keys())
        added = len(remote_keys - local_keys)
        removed = len(local_keys - remote_keys)
        common = local_keys & remote_keys
        modified = 0
        for k in common:
            lh = _payload_hash((local_payload.get("tracks") or {})[k])
            rh = _payload_hash((remote_payload.get("tracks") or {})[k])
            if lh != rh:
                modified += 1
        if added: result["analysis_added"] = added
        if removed: result["analysis_removed"] = removed
        if modified: result["analysis_modified"] = modified

    elif data_type == "session":
        local_s = local_payload.get("sessions") or []
        remote_s = remote_payload.get("sessions") or []
        local_names = {s.get("name", s.get("id", "")) for s in local_s if isinstance(s, dict)}
        remote_names = {s.get("name", s.get("id", "")) for s in remote_s if isinstance(s, dict)}
        added = len(remote_names - local_names)
        removed = len(local_names - remote_names)
        modified = 0
        for name in (local_names & remote_names):
            l_item = next((s for s in local_s if isinstance(s, dict) and s.get("name", s.get("id")) == name), None)
            r_item = next((s for s in remote_s if isinstance(s, dict) and s.get("name", s.get("id")) == name), None)
            if l_item and r_item and _payload_hash(l_item) != _payload_hash(r_item):
                modified += 1
        if added: result["session_added"] = added
        if removed: result["session_removed"] = removed
        if modified: result["session_modified"] = modified

    elif data_type == "favorite":
        local_ids = set(local_payload.get("ids") or [])
        remote_ids = set(remote_payload.get("ids") or [])
        added = len(remote_ids - local_ids)
        removed = len(local_ids - remote_ids)
        if added: result["favorite_added"] = added
        if removed: result["favorite_removed"] = removed

    elif data_type == "folder":
        local_f = local_payload.get("folders") or []
        remote_f = remote_payload.get("folders") or []
        local_ids = {f.get("id", "") for f in local_f if isinstance(f, dict)}
        remote_ids = {f.get("id", "") for f in remote_f if isinstance(f, dict)}
        added = len(remote_ids - local_ids)
        removed = len(local_ids - remote_ids)
        if added: result["folder_added"] = added
        if removed: result["folder_removed"] = removed

    elif data_type == "collection":
        local_c = local_payload.get("collections") or []
        remote_c = remote_payload.get("collections") or []
        local_ids = {c.get("id", c.get("name", "")) for c in local_c if isinstance(c, dict)}
        remote_ids = {c.get("id", c.get("name", "")) for c in remote_c if isinstance(c, dict)}
        added = len(remote_ids - local_ids)
        removed = len(local_ids - remote_ids)
        if added: result["collection_added"] = added
        if removed: result["collection_removed"] = removed

    elif data_type == "override":
        local_keys = set((local_payload.get("overrides") or {}).keys())
        remote_keys = set((remote_payload.get("overrides") or {}).keys())
        added = len(remote_keys - local_keys)
        removed = len(local_keys - remote_keys)
        modified = 0
        for k in (local_keys & remote_keys):
            lh = _payload_hash((local_payload.get("overrides") or {})[k])
            rh = _payload_hash((remote_payload.get("overrides") or {})[k])
            if lh != rh:
                modified += 1
        if added: result["override_added"] = added
        if removed: result["override_removed"] = removed
        if modified: result["override_modified"] = modified

    else:
        result[f"{data_type}_modified"] = 1

    return result


def _count_items_in(data_type: str, payload) -> int:
    if not isinstance(payload, dict):
        return 1
    mapping = {
        "analysis": "tracks", "session": "sessions", "favorite": "ids",
        "folder": "folders", "collection": "collections", "override": "overrides",
    }
    key = mapping.get(data_type)
    if key and key in payload:
        v = payload[key]
        return len(v) if isinstance(v, (dict, list)) else 1
    return 1


# ── STATUS ───────────────────────────────────────────────────

@sync_router.get("/status")
async def sync_status():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT data_type, last_device_id FROM sync_items"
    ).fetchall()
    by_type: dict[str, int] = {}
    devices: set[str] = set()
    for dt, dev in rows:
        by_type[dt] = by_type.get(dt, 0) + 1
        devices.add(dev)
    device_seen_counts = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT device_id, COUNT(*) FROM device_seen GROUP BY device_id"
        ).fetchall()
    }
    return {
        "total_items":    len(rows),
        "by_type":        by_type,
        "devices":        list(devices),
        "device_seen":    device_seen_counts,
    }


# ── CLEAR ─────────────────────────────────────────────────────

@sync_router.delete("/clear")
async def sync_clear():
    conn = _get_conn()
    conn.execute("DELETE FROM sync_items")
    conn.execute("DELETE FROM device_seen")
    conn.commit()
    return {"cleared": True}


# ── DETECTED TRACKS (Shazam sync) ────────────────────────────

class DetectedTrackSync(BaseModel):
    device_id: str
    artist: str
    title: str
    payload: Any
    detected_at: str


def _init_detected_table(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS detected_tracks_sync (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            artist TEXT NOT NULL,
            title TEXT NOT NULL,
            payload TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            UNIQUE(device_id, artist, title)
        );
        CREATE INDEX IF NOT EXISTS idx_dts_device
            ON detected_tracks_sync(device_id);
        CREATE INDEX IF NOT EXISTS idx_dts_date
            ON detected_tracks_sync(detected_at);
    """)
    conn.commit()


@sync_router.post("/detected-track")
async def sync_push_detected_track(track: DetectedTrackSync):
    """Sube un track detectado vía Shazam para sincronizarlo entre dispositivos"""
    if not track.device_id or not track.artist or not track.title:
        return {"status": "error", "message": "device_id, artist y title requeridos"}

    conn = _get_conn()
    _init_detected_table(conn)

    try:
        payload_str = json.dumps(track.payload, ensure_ascii=False)

        conn.execute("""
            INSERT INTO detected_tracks_sync
                (device_id, artist, title, payload, detected_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(device_id, artist, title)
            DO UPDATE SET
                payload = excluded.payload,
                detected_at = excluded.detected_at
        """, (
            track.device_id,
            track.artist,
            track.title,
            payload_str,
            track.detected_at,
        ))
        conn.commit()

        logger.info(f"Detected track saved: {track.artist} - {track.title}")
        return {"status": "ok"}

    except (sqlite3.Error, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Detected track error: {e}")
        return {"status": "error", "message": str(e)}


@sync_router.get("/detected-tracks/{device_id}")
async def sync_pull_detected_tracks(
    device_id: str,
    since: Optional[str] = None,
    limit: int = 200,
):
    """
    Descarga tracks detectados de TODOS los dispositivos.
    Útil para ver en el PC lo que escaneaste con el móvil.
    """
    conn = _get_conn()
    _init_detected_table(conn)

    try:
        if since:
            rows = conn.execute("""
                SELECT artist, title, payload, detected_at, device_id
                FROM detected_tracks_sync
                WHERE detected_at > ?
                ORDER BY detected_at DESC LIMIT ?
            """, (since, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT artist, title, payload, detected_at, device_id
                FROM detected_tracks_sync
                ORDER BY detected_at DESC LIMIT ?
            """, (limit,)).fetchall()

        tracks = []
        for row in rows:
            try:
                payload = json.loads(row[2])
            except (json.JSONDecodeError, TypeError, KeyError):
                payload = {}

            tracks.append({
                "artist": row[0],
                "title": row[1],
                "payload": payload,
                "detected_at": row[3],
                "from_device": row[4][:8] + "...",
            })

        return {
            "tracks": tracks,
            "total": len(tracks),
            "server_time": _now_iso(),
        }

    except (sqlite3.Error, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Pull detected tracks error: {e}")
        return {"tracks": [], "total": 0, "error": str(e)}

