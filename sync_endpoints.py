# ============================================================================
# SYNC ENDPOINTS v6.0 — DJ Analyzer Pro (Multi-Tenant)
# ============================================================================
# CAMBIOS vs v5.0:
#   1. MULTI-TENANT — Aislamiento total entre usuarios
#   2. USER ACCOUNTS — Registro anónimo + vinculación de dispositivos por código
#   3. MEMORIA COLECTIVA — data_types compartidos entre todos los usuarios
#   4. ADMIN ENDPOINTS — Vista completa de todos los usuarios y datos
#   5. MIGRACIÓN AUTOMÁTICA — Datos existentes se asignan al primer usuario
#
# ARQUITECTURA:
#   - Cada usuario tiene un user_id (UUID)
#   - Cada device_id se vincula a un user_id
#   - Pull/Push/Pending solo ven datos del MISMO usuario
#   - Tipos "collective" se comparten entre todos
#   - Admin con ADMIN_TOKEN ve todo
# ============================================================================

import hmac as _hmac
import logging
import uuid
import random
import string

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Optional
from datetime import datetime, timezone, timedelta
import json, hashlib, sqlite3, os

from config import SYNC_AUTH_SECRET, ADMIN_TOKEN

logger = logging.getLogger(__name__)


# ── HMAC-SHA256 Auth ─────────────────────────────────────────
# Si SYNC_AUTH_SECRET está configurado, todos los endpoints de sync
# requieren header X-Signature con HMAC-SHA256(secret, body).
# Si NO está configurado, auth se desactiva (modo desarrollo).

async def _verify_sync_auth(request: Request):
    """Dependency de FastAPI que valida HMAC si el secret está configurado."""
    if not SYNC_AUTH_SECRET:
        # Dev mode: solo permitir sin auth si es entorno local
        if os.getenv('RENDER') or os.getenv('RAILWAY_ENVIRONMENT'):
            raise HTTPException(status_code=500, detail="SYNC_AUTH_SECRET required in production")
        return  # Dev mode local: sin auth

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


sync_router = APIRouter(
    prefix="/sync",
    tags=["sync"],
    dependencies=[Depends(_verify_sync_auth)],
)

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

        -- v6.0: Multi-tenant tables
        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            label       TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS user_devices (
            device_id   TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            device_type TEXT DEFAULT 'unknown',
            device_name TEXT DEFAULT '',
            linked_at   TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_user_devices_user
            ON user_devices(user_id);

        CREATE TABLE IF NOT EXISTS link_codes (
            code        TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
    """)

    # Migración: añadir user_id a sync_items si no existe
    _migrate_add_user_id(conn)

    conn.commit()


def _migrate_add_user_id(conn: sqlite3.Connection):
    """Añade columna user_id a sync_items y detected_tracks_sync si no existe."""
    # sync_items
    cols = [row[1] for row in conn.execute("PRAGMA table_info(sync_items)").fetchall()]
    if "user_id" not in cols:
        conn.execute("ALTER TABLE sync_items ADD COLUMN user_id TEXT DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sync_items_user ON sync_items(user_id)")
        logger.info("Migrated sync_items: added user_id column")

    # detected_tracks_sync (puede no existir aún)
    try:
        cols_dt = [row[1] for row in conn.execute("PRAGMA table_info(detected_tracks_sync)").fetchall()]
        if cols_dt and "user_id" not in cols_dt:
            conn.execute("ALTER TABLE detected_tracks_sync ADD COLUMN user_id TEXT DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dts_user ON detected_tracks_sync(user_id)")
            logger.info("Migrated detected_tracks_sync: added user_id column")
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet


# ── Data types compartidos (Memoria Colectiva) ──────────────

COLLECTIVE_DATA_TYPES = frozenset({
    "cue_memory",
    "collective_notes",
    "manual_edits",
})


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


# ── User / Device helpers ───────────────────────────────────

def _get_user_id_for_device(conn: sqlite3.Connection, device_id: str) -> Optional[str]:
    """Busca el user_id asociado a un device_id. Retorna None si no está registrado."""
    row = conn.execute(
        "SELECT user_id FROM user_devices WHERE device_id = ?", (device_id,)
    ).fetchone()
    return row[0] if row else None


def _get_all_device_ids_for_user(conn: sqlite3.Connection, user_id: str) -> list[str]:
    """Retorna todos los device_id vinculados a un user_id."""
    rows = conn.execute(
        "SELECT device_id FROM user_devices WHERE user_id = ?", (user_id,)
    ).fetchall()
    return [r[0] for r in rows]


def _require_user_id(conn: sqlite3.Connection, device_id: str) -> str:
    """Obtiene user_id o lanza 403 si el dispositivo no está registrado."""
    user_id = _get_user_id_for_device(conn, device_id)
    if not user_id:
        raise HTTPException(
            status_code=403,
            detail=f"Device '{device_id}' not registered. Call POST /sync/register first."
        )
    return user_id


def _generate_link_code() -> str:
    """Genera un código alfanumérico de 6 caracteres (sin ambigüedades)."""
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sin 0/O/1/I
    return "".join(random.choices(chars, k=6))


def _is_collective(data_type: str) -> bool:
    """Retorna True si este data_type es memoria colectiva (compartido)."""
    return data_type in COLLECTIVE_DATA_TYPES


# ── REGISTER & LINK ────────────────────────────────────────

class RegisterRequest(BaseModel):
    device_id: str
    device_type: str = "unknown"
    device_name: str = ""


@sync_router.post("/register")
async def sync_register(req: RegisterRequest):
    """
    Registra un dispositivo. Si ya existe, retorna su user_id.
    Si es nuevo, crea un usuario nuevo y lo vincula.
    """
    conn = _get_conn()

    existing = _get_user_id_for_device(conn, req.device_id)
    if existing:
        devices = _get_all_device_ids_for_user(conn, existing)
        return {
            "user_id": existing,
            "device_id": req.device_id,
            "already_registered": True,
            "linked_devices": len(devices),
        }

    # Crear usuario nuevo
    user_id = str(uuid.uuid4())
    now = _now_iso()

    conn.execute(
        "INSERT INTO users (user_id, created_at) VALUES (?, ?)",
        (user_id, now),
    )
    conn.execute(
        "INSERT INTO user_devices (device_id, user_id, device_type, device_name, linked_at) VALUES (?, ?, ?, ?, ?)",
        (req.device_id, user_id, req.device_type, req.device_name, now),
    )

    # Migración: asignar datos existentes sin user_id a este usuario
    _assign_orphan_data(conn, req.device_id, user_id)

    conn.commit()
    logger.info(f"New user registered: {user_id} (device: {req.device_id})")

    return {
        "user_id": user_id,
        "device_id": req.device_id,
        "already_registered": False,
        "linked_devices": 1,
    }


def _assign_orphan_data(conn: sqlite3.Connection, device_id: str, user_id: str):
    """Asigna datos huérfanos (sin user_id) que pertenecen a este device_id."""
    conn.execute(
        "UPDATE sync_items SET user_id = ? WHERE last_device_id = ? AND (user_id = '' OR user_id IS NULL)",
        (user_id, device_id),
    )
    try:
        conn.execute(
            "UPDATE detected_tracks_sync SET user_id = ? WHERE device_id = ? AND (user_id = '' OR user_id IS NULL)",
            (user_id, device_id),
        )
    except sqlite3.OperationalError:
        pass  # Table may not exist yet


class LinkGenerateRequest(BaseModel):
    device_id: str


@sync_router.post("/link/generate")
async def sync_link_generate(req: LinkGenerateRequest):
    """
    Genera un código de vinculación de 6 caracteres (válido 10 minutos).
    El dispositivo que genera el código ya debe estar registrado.
    """
    conn = _get_conn()
    user_id = _require_user_id(conn, req.device_id)

    # Limpiar códigos expirados
    conn.execute("DELETE FROM link_codes WHERE expires_at < ?", (_now_iso(),))

    code = _generate_link_code()
    now = _now_iso()
    expires = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    conn.execute(
        "INSERT INTO link_codes (code, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (code, user_id, now, expires),
    )
    conn.commit()

    return {
        "code": code,
        "expires_at": expires,
        "user_id": user_id,
    }


class LinkJoinRequest(BaseModel):
    device_id: str
    code: str
    device_type: str = "unknown"
    device_name: str = ""


@sync_router.post("/link/join")
async def sync_link_join(req: LinkJoinRequest):
    """
    Vincula un dispositivo a un usuario existente usando el código de 6 caracteres.
    Si el dispositivo ya está registrado con otro usuario, se re-vincula.
    """
    conn = _get_conn()

    # Buscar código válido
    row = conn.execute(
        "SELECT user_id, expires_at FROM link_codes WHERE code = ?",
        (req.code.upper(),),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Invalid or expired link code")

    target_user_id, expires_at = row
    if expires_at < _now_iso():
        conn.execute("DELETE FROM link_codes WHERE code = ?", (req.code.upper(),))
        conn.commit()
        raise HTTPException(status_code=410, detail="Link code expired")

    now = _now_iso()

    # Si el dispositivo ya estaba registrado, re-vincular
    existing_user = _get_user_id_for_device(conn, req.device_id)
    if existing_user == target_user_id:
        # Ya vinculado al mismo usuario
        conn.execute("DELETE FROM link_codes WHERE code = ?", (req.code.upper(),))
        conn.commit()
        devices = _get_all_device_ids_for_user(conn, target_user_id)
        return {
            "user_id": target_user_id,
            "device_id": req.device_id,
            "already_linked": True,
            "linked_devices": len(devices),
        }

    if existing_user:
        # Re-vincular de un usuario a otro
        conn.execute("DELETE FROM user_devices WHERE device_id = ?", (req.device_id,))

    conn.execute(
        "INSERT INTO user_devices (device_id, user_id, device_type, device_name, linked_at) VALUES (?, ?, ?, ?, ?)",
        (req.device_id, target_user_id, req.device_type, req.device_name, now),
    )

    # Migrar datos huérfanos
    _assign_orphan_data(conn, req.device_id, target_user_id)

    # Consumir código
    conn.execute("DELETE FROM link_codes WHERE code = ?", (req.code.upper(),))
    conn.commit()

    devices = _get_all_device_ids_for_user(conn, target_user_id)
    logger.info(f"Device {req.device_id} linked to user {target_user_id} via code {req.code}")

    return {
        "user_id": target_user_id,
        "device_id": req.device_id,
        "already_linked": False,
        "linked_devices": len(devices),
    }


# ── Models ──────────────────────────────────────────────────

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
    """Push sobreescribe la verdad del backend. Last-write-wins.

    Multi-tenant: cada item se etiqueta con el user_id del dispositivo.
    Tipos colectivos (cue_memory, collective_notes, manual_edits) se
    almacenan con user_id='__collective__' y son visibles para todos.
    """
    conn = _get_conn()
    user_id = _require_user_id(conn, req.device_id)
    synced = 0
    skipped = 0
    now = _now_iso()

    for change in req.changes:
        # Colectivos: clave global. Privados: clave scoped por usuario.
        if _is_collective(change.data_type):
            key = f"{change.data_type}|{change.item_key}"
            item_user_id = "__collective__"
        else:
            key = f"{user_id}|{change.data_type}|{change.item_key}"
            item_user_id = user_id

        change_time = change.updated_at or now
        new_hash = _payload_hash(change.payload)
        payload_json = json.dumps(change.payload, default=str)

        # Si ya existe con el mismo hash, skip
        row = conn.execute(
            "SELECT hash FROM sync_items WHERE key = ?", (key,)
        ).fetchone()
        if row and row[0] == new_hash:
            skipped += 1
            conn.execute(
                """INSERT INTO device_seen (device_id, item_key, hash, payload)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(device_id, item_key) DO UPDATE SET hash = ?, payload = ?""",
                (req.device_id, key, new_hash, payload_json, new_hash, payload_json),
            )
            continue

        # Upsert del item con user_id
        conn.execute(
            """INSERT INTO sync_items
                   (key, data_type, item_key, payload, deleted,
                    updated_at, last_device_id, device_type, hash, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   payload = excluded.payload,
                   deleted = excluded.deleted,
                   updated_at = excluded.updated_at,
                   last_device_id = excluded.last_device_id,
                   device_type = excluded.device_type,
                   hash = excluded.hash,
                   user_id = excluded.user_id""",
            (key, change.data_type, change.item_key, payload_json,
             1 if change.deleted else 0, change_time,
             req.device_id, req.device_type, new_hash, item_user_id),
        )
        synced += 1

        conn.execute(
            """INSERT INTO device_seen (device_id, item_key, hash, payload)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(device_id, item_key) DO UPDATE SET hash = ?, payload = ?""",
            (req.device_id, key, new_hash, payload_json, new_hash, payload_json),
        )

    conn.commit()
    return {"synced": synced, "skipped": skipped, "conflicts": [], "timestamp": now, "user_id": user_id}


# ── PULL ─────────────────────────────────────────────────────

@sync_router.get("/pull/{device_id}")
async def sync_pull(
    device_id: str,
    since: Optional[str] = None,
):
    """Descarga items cuyo hash sea DIFERENTE al que este dispositivo conoce.

    Multi-tenant: solo devuelve items del MISMO usuario + colectivos.
    Filtra por user_id (no por device_id como antes).
    """
    conn = _get_conn()
    user_id = _require_user_id(conn, device_id)

    # Items del mismo usuario + colectivos, que NO subió este device
    rows = conn.execute(
        """SELECT si.key, si.data_type, si.item_key, si.payload, si.deleted,
                  si.updated_at, si.device_type, si.hash
           FROM sync_items si
           WHERE si.last_device_id != ?
             AND (si.user_id = ? OR si.user_id = '__collective__')""",
        (device_id, user_id),
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
        "user_id":   user_id,
    }


# ── PENDING ──────────────────────────────────────────────────

@sync_router.get("/pending/{device_id}")
async def sync_pending(device_id: str, since: Optional[str] = None):
    """Desglose detallado: añadidos, eliminados, modificados por tipo.

    Multi-tenant: solo cuenta items del MISMO usuario + colectivos.
    """
    conn = _get_conn()
    user_id = _require_user_id(conn, device_id)

    # Migración: añadir columna payload si no existe
    try:
        conn.execute("ALTER TABLE device_seen ADD COLUMN payload TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Ya existe

    # Items remotos del mismo usuario + colectivos
    remote_rows = conn.execute(
        """SELECT si.key, si.data_type, si.item_key, si.payload, si.hash
           FROM sync_items si
           WHERE si.last_device_id != ?
             AND (si.user_id = ? OR si.user_id = '__collective__')""",
        (device_id, user_id),
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
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_registered_devices = conn.execute("SELECT COUNT(*) FROM user_devices").fetchone()[0]
    return {
        "total_items":    len(rows),
        "by_type":        by_type,
        "devices":        list(devices),
        "device_seen":    device_seen_counts,
        "total_users":    total_users,
        "total_registered_devices": total_registered_devices,
        "version":        "6.0",
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
    """Sube un track detectado vía Shazam, etiquetado con user_id."""
    if not track.device_id or not track.artist or not track.title:
        return {"status": "error", "message": "device_id, artist y title requeridos"}

    conn = _get_conn()
    _init_detected_table(conn)
    user_id = _require_user_id(conn, track.device_id)

    try:
        payload_str = json.dumps(track.payload, ensure_ascii=False)

        conn.execute("""
            INSERT INTO detected_tracks_sync
                (device_id, artist, title, payload, detected_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id, artist, title)
            DO UPDATE SET
                payload = excluded.payload,
                detected_at = excluded.detected_at,
                user_id = excluded.user_id
        """, (
            track.device_id,
            track.artist,
            track.title,
            payload_str,
            track.detected_at,
            user_id,
        ))
        conn.commit()

        logger.info(f"Detected track saved: {track.artist} - {track.title} (user: {user_id[:8]})")
        return {"status": "ok", "user_id": user_id}

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
    Descarga tracks detectados SOLO del mismo usuario.
    Útil para ver en el PC lo que escaneaste con el móvil.
    Solo muestra tracks de dispositivos vinculados al mismo user_id.
    """
    conn = _get_conn()
    _init_detected_table(conn)
    user_id = _require_user_id(conn, device_id)

    try:
        if since:
            rows = conn.execute("""
                SELECT artist, title, payload, detected_at, device_id
                FROM detected_tracks_sync
                WHERE user_id = ? AND detected_at > ?
                ORDER BY detected_at DESC LIMIT ?
            """, (user_id, since, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT artist, title, payload, detected_at, device_id
                FROM detected_tracks_sync
                WHERE user_id = ?
                ORDER BY detected_at DESC LIMIT ?
            """, (user_id, limit)).fetchall()

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
            "user_id": user_id,
        }

    except (sqlite3.Error, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Pull detected tracks error: {e}")
        return {"tracks": [], "total": 0, "error": str(e)}


# ════════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS — Vista de red para el administrador
# ════════════════════════════════════════════════════════════════
# Protegidos por ADMIN_TOKEN (header Authorization: Bearer <token>)
# Proporcionan acceso a TODOS los datos de TODOS los usuarios.

async def _verify_admin(request: Request):
    """Verifica ADMIN_TOKEN para endpoints admin de sync."""
    if not ADMIN_TOKEN:
        if os.getenv('RENDER') or os.getenv('RAILWAY_ENVIRONMENT'):
            raise HTTPException(status_code=500, detail="ADMIN_TOKEN required in production")
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != ADMIN_TOKEN:
        raise HTTPException(401, "Admin token required")


admin_sync_router = APIRouter(
    prefix="/sync/admin",
    tags=["sync-admin"],
    dependencies=[Depends(_verify_sync_auth), Depends(_verify_admin)],
)


@admin_sync_router.get("/users")
async def admin_list_users():
    """Lista todos los usuarios registrados con sus dispositivos y estadísticas."""
    conn = _get_conn()

    users = conn.execute(
        "SELECT user_id, created_at, label FROM users ORDER BY created_at DESC"
    ).fetchall()

    result = []
    for user_id, created_at, label in users:
        devices = conn.execute(
            "SELECT device_id, device_type, device_name, linked_at FROM user_devices WHERE user_id = ?",
            (user_id,),
        ).fetchall()

        item_count = conn.execute(
            "SELECT COUNT(*) FROM sync_items WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

        by_type = {}
        for row in conn.execute(
            "SELECT data_type, COUNT(*) FROM sync_items WHERE user_id = ? GROUP BY data_type",
            (user_id,),
        ).fetchall():
            by_type[row[0]] = row[1]

        result.append({
            "user_id": user_id,
            "created_at": created_at,
            "label": label,
            "devices": [
                {
                    "device_id": d[0],
                    "device_type": d[1],
                    "device_name": d[2],
                    "linked_at": d[3],
                }
                for d in devices
            ],
            "total_items": item_count,
            "items_by_type": by_type,
        })

    # Datos colectivos
    collective_count = conn.execute(
        "SELECT COUNT(*) FROM sync_items WHERE user_id = '__collective__'"
    ).fetchone()[0]

    return {
        "total_users": len(result),
        "users": result,
        "collective_items": collective_count,
    }


@admin_sync_router.get("/users/{user_id}")
async def admin_get_user_data(user_id: str, data_type: Optional[str] = None):
    """Obtiene TODOS los datos de un usuario específico. Vista detallada."""
    conn = _get_conn()

    # Verificar que el usuario existe
    user = conn.execute("SELECT created_at, label FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not user:
        raise HTTPException(404, "User not found")

    devices = conn.execute(
        "SELECT device_id, device_type, device_name, linked_at FROM user_devices WHERE user_id = ?",
        (user_id,),
    ).fetchall()

    # Items del usuario
    if data_type:
        items = conn.execute(
            "SELECT key, data_type, item_key, payload, deleted, updated_at, last_device_id, device_type FROM sync_items WHERE user_id = ? AND data_type = ?",
            (user_id, data_type),
        ).fetchall()
    else:
        items = conn.execute(
            "SELECT key, data_type, item_key, payload, deleted, updated_at, last_device_id, device_type FROM sync_items WHERE user_id = ?",
            (user_id,),
        ).fetchall()

    items_list = []
    for row in items:
        try:
            payload = json.loads(row[3])
        except (json.JSONDecodeError, TypeError):
            payload = row[3]
        items_list.append({
            "key": row[0],
            "data_type": row[1],
            "item_key": row[2],
            "payload": payload,
            "deleted": bool(row[4]),
            "updated_at": row[5],
            "last_device_id": row[6],
            "device_type": row[7],
        })

    # Detected tracks del usuario
    detected = []
    try:
        dt_rows = conn.execute(
            "SELECT artist, title, detected_at, device_id FROM detected_tracks_sync WHERE user_id = ? ORDER BY detected_at DESC LIMIT 100",
            (user_id,),
        ).fetchall()
        for r in dt_rows:
            detected.append({"artist": r[0], "title": r[1], "detected_at": r[2], "device_id": r[3][:8] + "..."})
    except sqlite3.OperationalError:
        pass

    return {
        "user_id": user_id,
        "created_at": user[0],
        "label": user[1],
        "devices": [
            {"device_id": d[0], "device_type": d[1], "device_name": d[2], "linked_at": d[3]}
            for d in devices
        ],
        "items": items_list,
        "total_items": len(items_list),
        "detected_tracks": detected,
    }


@admin_sync_router.get("/network")
async def admin_network_overview():
    """Vista de red completa: todos los usuarios, dispositivos, items, colectivos.
    Diseñada para el panel de administrador.
    """
    conn = _get_conn()

    # Usuarios
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_devices = conn.execute("SELECT COUNT(*) FROM user_devices").fetchone()[0]
    total_items = conn.execute("SELECT COUNT(*) FROM sync_items").fetchone()[0]
    collective_items = conn.execute(
        "SELECT COUNT(*) FROM sync_items WHERE user_id = '__collective__'"
    ).fetchone()[0]

    # Items por tipo global
    by_type = {}
    for row in conn.execute("SELECT data_type, COUNT(*) FROM sync_items GROUP BY data_type").fetchall():
        by_type[row[0]] = row[1]

    # Top usuarios por cantidad de items
    top_users = conn.execute("""
        SELECT u.user_id, u.label, COUNT(si.key) as item_count,
               (SELECT COUNT(*) FROM user_devices ud WHERE ud.user_id = u.user_id) as device_count
        FROM users u
        LEFT JOIN sync_items si ON si.user_id = u.user_id
        GROUP BY u.user_id
        ORDER BY item_count DESC
        LIMIT 50
    """).fetchall()

    # Detected tracks global
    total_detected = 0
    try:
        total_detected = conn.execute("SELECT COUNT(*) FROM detected_tracks_sync").fetchone()[0]
    except sqlite3.OperationalError:
        pass

    # Link codes activos
    active_codes = conn.execute(
        "SELECT COUNT(*) FROM link_codes WHERE expires_at > ?", (_now_iso(),)
    ).fetchone()[0]

    return {
        "total_users": total_users,
        "total_devices": total_devices,
        "total_items": total_items,
        "collective_items": collective_items,
        "total_detected_tracks": total_detected,
        "active_link_codes": active_codes,
        "items_by_type": by_type,
        "top_users": [
            {
                "user_id": r[0],
                "label": r[1] or "",
                "items": r[2],
                "devices": r[3],
            }
            for r in top_users
        ],
    }


@admin_sync_router.get("/all-items")
async def admin_all_items(
    data_type: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
):
    """Lista paginada de TODOS los items. Filtrable por data_type y/o user_id."""
    conn = _get_conn()

    where_parts = []
    params: list = []

    if data_type:
        where_parts.append("data_type = ?")
        params.append(data_type)
    if user_id:
        where_parts.append("user_id = ?")
        params.append(user_id)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    total = conn.execute(
        f"SELECT COUNT(*) FROM sync_items {where_sql}", params
    ).fetchone()[0]

    rows = conn.execute(
        f"""SELECT key, data_type, item_key, payload, deleted, updated_at,
                   last_device_id, device_type, user_id, hash
            FROM sync_items {where_sql}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?""",
        params + [limit, offset],
    ).fetchall()

    items = []
    for row in rows:
        try:
            payload = json.loads(row[3])
        except (json.JSONDecodeError, TypeError):
            payload = row[3]
        items.append({
            "key": row[0],
            "data_type": row[1],
            "item_key": row[2],
            "payload": payload,
            "deleted": bool(row[4]),
            "updated_at": row[5],
            "last_device_id": row[6],
            "device_type": row[7],
            "user_id": row[8],
            "hash": row[9],
        })

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }

