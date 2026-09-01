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
import secrets
import uuid
import random
import string

from fastapi import APIRouter, Request, HTTPException, Depends
from starlette.requests import ClientDisconnect
from pydantic import BaseModel
from typing import Any, List, Optional
from datetime import datetime, timezone, timedelta
import json, hashlib, sqlite3, os
from validation import is_desktop_platform

from config import SYNC_AUTH_SECRET, ADMIN_TOKEN

logger = logging.getLogger(__name__)


# ── LIMITES ─────────────────────────────────────────────────
# Soft cap de dispositivos por usuario. Evita abuso (alguien registrando
# miles de device_ids contra el mismo user_id) y mantiene la UI manejable.
# 20 cubre uso normal con margen (2-3 desktops + movil + tablet + trabajo
# + sobra). Si se alcanza, /sync/link/join devuelve 409 con code
# "MAX_DEVICES_REACHED" y el cliente muestra mensaje claro al usuario.
MAX_DEVICES_PER_USER = 20


# ── HMAC-SHA256 Auth ─────────────────────────────────────────
# Si SYNC_AUTH_SECRET está configurado, todos los endpoints de sync
# requieren header X-Signature con HMAC-SHA256(secret, body).
# Si NO está configurado, auth se desactiva (modo desarrollo).

# Ventana de frescura (segundos) para el esquema anti-replay con X-Timestamp.
# Generosa por defecto para tolerar reloj desajustado del dispositivo; bajar
# via env cuando se confirme que no genera 401s en campo.
SYNC_REPLAY_WINDOW_SEC = int(os.getenv('SYNC_REPLAY_WINDOW_SEC', '3600'))


async def _verify_sync_auth(request: Request):
    """Dependency de FastAPI que valida HMAC si el secret está configurado."""
    if not SYNC_AUTH_SECRET:
        # Dev mode: solo permitir sin auth si es entorno local
        if os.getenv('RENDER') or os.getenv('RAILWAY_ENVIRONMENT'):
            raise HTTPException(status_code=500, detail="SYNC_AUTH_SECRET required in production")
        return  # Dev mode local: sin auth

    try:
        body = await request.body()
    except ClientDisconnect:
        raise HTTPException(status_code=499, detail="Client disconnected")
    signature = request.headers.get("X-Signature", "")

    if not signature:
        raise HTTPException(status_code=401, detail="Missing X-Signature header")

    # SYNC_AUTH_SECRET admite una LISTA separada por comas para ROTAR el secret
    # SIN downtime: durante la transicion se pone "nuevo,viejo" y aceptamos la
    # firma si coincide con CUALQUIERA. Asi los clientes viejos (firman con el
    # viejo) siguen sincronizando hasta que actualizan; luego quitas el viejo de
    # la env var. Un solo valor (sin coma) se comporta igual que antes.
    # OJO con el nombre: llamar `secrets` a esta lista SOMBREABA el modulo
    # `secrets` de la stdlib dentro de la funcion. Nadie lo usaba aqui todavia,
    # pero el modulo si se usa en este fichero (_issue_device_token), asi que
    # era una mina esperando a que alguien anadiera una linea.
    auth_secrets = [s.strip() for s in SYNC_AUTH_SECRET.split(',') if s.strip()]

    # ANTI-REPLAY (rollout por fases, backward-compatible):
    # - Cliente NUEVO manda `X-Timestamp` (unix ms) y firma "<ts>.<body>". Aqui
    #   verificamos freshness (rechaza peticiones viejas = replay) + esa firma.
    # - Cliente VIEJO (sin `X-Timestamp`) firma solo "<body>": se acepta como antes
    #   (SIN anti-replay) para NO romper sync durante la transicion.
    # Cuando la mayoria este actualizada, endurecer = exigir X-Timestamp (fase 2).
    # Ventana generosa (`SYNC_REPLAY_WINDOW_SEC`, default 3600s) para tolerar reloj
    # desajustado del dispositivo; bajarla cuando se confirme que no da 401s.
    ts = request.headers.get("X-Timestamp", "")
    valid = False
    # Slot del secreto que valida: 0 = el primero de la lista (el vigente),
    # 1+ = los que se mantienen por rotacion. Se registra para poder retirar
    # un secreto viejo SOLO cuando nadie lo use ya (hoy eso se decidia a ojo).
    matched_slot = -1
    if ts:
        try:
            ts_ms = int(ts)
        except (ValueError, TypeError):
            ts_ms = 0
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if ts_ms <= 0 or abs(now_ms - ts_ms) > SYNC_REPLAY_WINDOW_SEC * 1000:
            raise HTTPException(status_code=401, detail="Stale or invalid timestamp")
        signed = ts.encode() + b"." + body
        for slot, secret in enumerate(auth_secrets):
            expected = _hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
            if _hmac.compare_digest(signature, expected):
                valid = True
                matched_slot = slot
                break
    else:
        for slot, secret in enumerate(auth_secrets):
            expected = _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            if _hmac.compare_digest(signature, expected):
                valid = True
                matched_slot = slot
                break

    if not valid:
        raise HTTPException(status_code=401, detail="Invalid signature")

    # ── Auth POR DISPOSITIVO (fase 1 de 3) ──────────────────────────────
    # El HMAC de arriba solo demuestra "tengo el secreto de la app", y ese
    # secreto lo tiene cualquiera que abra el binario con `strings`. NO
    # demuestra ser el dueño del device_id, que es lo unico que separa la
    # biblioteca de un usuario de la de otro.
    #
    # FASE 1 (esto): si el cliente manda X-Device-Token, se EXIGE que sea el
    # suyo. Si no lo manda, se acepta igual que hasta ahora. Los dispositivos
    # ya en campo siguen funcionando sin tocar nada, y los clientes nuevos
    # empiezan a acreditarse.
    # FASE 3 (mas adelante, NO aqui): exigirlo siempre. Solo se puede hacer
    # cuando sync_auth_stats demuestre que practicamente todo el parque manda
    # token — endurecer antes deja gente fuera de SUS PROPIOS datos.
    #
    # Mientras tanto el agujero sigue abierto para quien omita la cabecera.
    # Esta fase NO lo cierra: despliega el mecanismo y mide, que es el unico
    # camino seguro hasta poder cerrarlo.
    token = request.headers.get("X-Device-Token", "")
    device_id = _device_id_from_request(request, body)
    if device_id:
        conn = _get_conn()
        row = conn.execute(
            "SELECT device_token, token_seen_at FROM user_devices "
            "WHERE device_id = ?",
            (device_id,),
        ).fetchone()
        stored = row[0] if row and row[0] else None
        enforced = bool(row and row[1])

        if token:
            # Si el dispositivo aun no tiene token emitido NO se rechaza: es un
            # cliente ya actualizado hablando de un device_id que todavia no lo
            # ha reclamado. Rechazarlo romperia sync justo en la migracion.
            if stored and not _hmac.compare_digest(token, stored):
                raise HTTPException(
                    status_code=401, detail="Invalid device token")
            # Token correcto: a partir de AHORA se le exige. Se sella una sola
            # vez (WHERE token_seen_at IS NULL) para no escribir en cada
            # peticion de sync.
            if stored and not enforced:
                try:
                    conn.execute(
                        "UPDATE user_devices SET token_seen_at = ? "
                        "WHERE device_id = ? AND token_seen_at IS NULL",
                        (_now_iso(), device_id),
                    )
                    conn.commit()
                except sqlite3.Error:
                    pass  # sellar es best-effort, nunca tumba un sync
        elif enforced:
            # FASE 2.5 — este dispositivo YA demostro saber mandar su token y
            # ahora llega sin el. O es un downgrade del cliente (raro) o es
            # alguien suplantando el device_id con el secreto global, que es
            # justo el agujero que esto cierra.
            #
            # Se cierra POR DISPOSITIVO, no con un dia D: cada usuario queda
            # protegido en cuanto actualiza, sin esperar a que se actualice
            # nadie mas. Esperar al 100 % del parque —el criterio de la fase 3
            # original— es esperar a un numero que no llega nunca, y mientras
            # tanto el agujero sigue abierto para todos.
            raise HTTPException(
                status_code=401, detail="Device token required")

    _record_sync_auth(matched_slot, bool(token))


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
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False, timeout=30.0)
        # journal_mode=WAL una vez (esta conexion singleton es el init de sync.db).
        _conn.execute("PRAGMA journal_mode=WAL")
        # busy_timeout FALTABA: sin el, cualquier escritura de /sync/* fallaba al
        # instante con "database is locked" bajo contienda (fix outage 2026-07-14).
        _conn.execute("PRAGMA busy_timeout=30000")
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

        -- Adopcion de la auth por dispositivo + que secreto valida cada
        -- peticion. Agregado por dia, sin device_id ni nada identificable.
        -- Existe para poder tomar DOS decisiones con datos en vez de a ojo:
        --   1) retirar el secreto viejo de la lista rotada de SYNC_AUTH_SECRET
        --      sin dejar tirado a ningun cliente que aun firme con el;
        --   2) saber cuando TODOS los clientes mandan ya X-Device-Token, que es
        --      el requisito para EXIGIRLO (fase 3) sin bloquear a nadie.
        CREATE TABLE IF NOT EXISTS sync_auth_stats (
            day         TEXT NOT NULL,
            secret_slot INTEGER NOT NULL,
            has_token   INTEGER NOT NULL,
            n           INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (day, secret_slot, has_token)
        );
    """)

    # Migration: token POR DISPOSITIVO.
    #
    # Por que hace falta: SYNC_AUTH_SECRET viaja compilado en el binario de
    # cada cliente (--dart-define), asi que cualquiera puede extraerlo con
    # `strings` del .app/.exe/.apk. No es un secreto "comprometido": es que un
    # secreto compartido dentro de una app distribuida NO PUEDE serlo. Y hoy
    # ese HMAC es la UNICA puerta: _require_user_id solo comprueba que el
    # device_id este registrado, no que quien llama SEA ese dispositivo. Con el
    # secreto (extraible) y un device_id conocido se puede leer y sobrescribir
    # la biblioteca de cualquier usuario.
    #
    # El token es aleatorio, se emite UNA sola vez por dispositivo y no viaja
    # en ningun binario, asi que un device_id filtrado deja de bastar.
    try:
        conn.execute("ALTER TABLE user_devices ADD COLUMN device_token TEXT")
    except sqlite3.OperationalError:
        pass  # Already exists

    # FASE 2.5 — marca de que este dispositivo YA DEMOSTRO saber mandar su
    # token. Se sella la primera vez que llega una peticion con el token
    # correcto, y a partir de ahi se le EXIGE (ver _verify_sync_auth).
    #
    # Hace falta una columna aparte y no vale `device_token IS NOT NULL`: el
    # plan original decia "si ya tiene token emitido, exigelo", pero eso es
    # FALSO como premisa. `_issue_device_token` se llama en TODO /sync/register
    # sin mirar la version del cliente, asi que un cliente <= 2.9.8 recibe el
    # token en la respuesta, lo IGNORA (su codigo no conoce el campo) y deja el
    # device_token guardado en el servidor. Exigirlo por "emitido" habria
    # echado de su propia biblioteca a todo ese parque — el 71 % de las
    # peticiones de sync todavia llegan sin token.
    try:
        conn.execute("ALTER TABLE user_devices ADD COLUMN token_seen_at TEXT")
    except sqlite3.OperationalError:
        pass  # Already exists

    # Migration: add payload column to device_seen if missing
    try:
        conn.execute("ALTER TABLE device_seen ADD COLUMN payload TEXT")
    except sqlite3.OperationalError:
        pass  # Already exists

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


def _get_all_devices_for_user(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    """Retorna info completa de todos los dispositivos de un user_id."""
    rows = conn.execute(
        "SELECT device_id, device_type, device_name, linked_at FROM user_devices WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    return [
        {"device_id": r[0], "device_type": r[1], "device_name": r[2], "linked_at": r[3]}
        for r in rows
    ]


def _issue_device_token(conn: sqlite3.Connection, device_id: str) -> Optional[str]:
    """Emite el token de un dispositivo si aun no tiene, y lo devuelve.

    Devuelve None si YA tenia uno. Esto es deliberado y es la parte importante:
    el token se entrega UNA sola vez. Si /sync/register lo devolviera siempre,
    cualquiera con el secreto global (que es cualquiera: viaja en el binario)
    podria pedir el token de un device_id conocido y quedarse con la cuenta —
    justo lo que este mecanismo existe para impedir.

    Consecuencia asumida durante la migracion: los dispositivos ya registrados
    no tienen token, asi que el PRIMERO que llame a register se lo lleva. En la
    practica sera su cliente legitimo al actualizar, y mientras el token no sea
    obligatorio (fase 3) reclamarlo no da mas acceso del que el secreto global
    ya da hoy. Cuando un dispositivo pierde sus prefs pierde tambien su
    device_id (los dos viven en SharedPreferences), asi que vuelve como
    dispositivo nuevo y recibe token nuevo: no hay estado irrecuperable.
    """
    row = conn.execute(
        "SELECT device_token FROM user_devices WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    if row is None:
        return None
    if row[0]:
        return None  # ya emitido: no se revela nunca mas
    token = secrets.token_urlsafe(32)
    conn.execute(
        "UPDATE user_devices SET device_token = ? WHERE device_id = ?",
        (token, device_id),
    )
    conn.commit()
    return token


def _record_sync_auth(secret_slot: int, has_token: bool) -> None:
    """Cuenta (por dia) que secreto valido y si venia token. Best-effort: nunca
    puede tumbar una peticion de sync."""
    try:
        conn = _get_conn()
        day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn.execute(
            "INSERT INTO sync_auth_stats (day, secret_slot, has_token, n) "
            "VALUES (?, ?, ?, 1) "
            "ON CONFLICT(day, secret_slot, has_token) DO UPDATE SET n = n + 1",
            (day, int(secret_slot), 1 if has_token else 0),
        )
        conn.commit()
    except Exception:  # noqa: BLE001
        pass


def _device_id_from_request(request: Request, body: bytes) -> Optional[str]:
    """device_id de la peticion, venga por path (/pull/{device_id}) o por body.

    Se resuelve aqui, en la dependency, y no en cada endpoint: asi la
    verificacion del token vive en UN sitio y no hay que tocar las ocho firmas
    que hoy llaman a _require_user_id (ninguna recibe el Request).
    """
    did = request.path_params.get('device_id')
    if did:
        return str(did)
    if not body:
        return None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(data, dict):
        v = data.get('device_id')
        if v:
            return str(v)
    return None


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
    """Genera un código alfanumérico de 6 caracteres (sin ambigüedades).

    Usa `secrets`, NO `random`: un código de vinculación da acceso COMPLETO a
    la cuenta (pull de toda la biblioteca y push a todos sus dispositivos), así
    que es material criptográfico. `random.choices` usa Mersenne Twister, cuyo
    estado interno se puede reconstruir observando suficientes salidas — y
    cualquiera puede pedir códigos propios para obtenerlas.
    """
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sin 0/O/1/I
    return "".join(secrets.choice(chars) for _ in range(6))


# Fuerza bruta contra /sync/link/join: el espacio es 32^6 (~1.07e9) pero los
# códigos viven 10 minutos y puede haber varios activos a la vez, así que sin
# límite de intentos el ataque es viable con paciencia. Ventana deslizante en
# memoria por IP; el proceso es único (un worker) así que basta con un dict.
_JOIN_MAX_ATTEMPTS = int(os.getenv('LINK_JOIN_MAX_ATTEMPTS', '20'))
_JOIN_WINDOW_SEC = int(os.getenv('LINK_JOIN_WINDOW_SEC', '600'))
_join_attempts: dict = {}


def _check_join_attempts(ip: str) -> None:
    """Cuenta intentos FALLIDOS de join por IP y corta al superar el umbral."""
    import time as _time

    now = _time.time()
    recent = [t for t in _join_attempts.get(ip, []) if now - t < _JOIN_WINDOW_SEC]
    # Purga de claves vacías: sin esto el dict crece con cada IP vista.
    if recent:
        _join_attempts[ip] = recent
    else:
        _join_attempts.pop(ip, None)
    if len(recent) >= _JOIN_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Demasiados intentos de vinculación. Espera unos minutos.",
            headers={"Retry-After": str(_JOIN_WINDOW_SEC)},
        )


def _record_join_failure(ip: str) -> None:
    import time as _time

    _join_attempts.setdefault(ip, []).append(_time.time())


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
        devices = _get_all_devices_for_user(conn, existing)
        # Migracion de los dispositivos que se registraron antes de que
        # existieran los tokens: se les emite aqui, la primera vez que su
        # cliente actualizado vuelva a llamar. Si ya tenian, _issue_device_token
        # devuelve None y NO se revela — ver su docstring.
        issued = _issue_device_token(conn, req.device_id)
        resp = {
            "user_id": existing,
            "device_id": req.device_id,
            "already_registered": True,
            "linked_devices": devices,
            "max_devices": MAX_DEVICES_PER_USER,
        }
        if issued:
            resp["device_token"] = issued
        return resp

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

    devices = _get_all_devices_for_user(conn, user_id)
    resp = {
        "user_id": user_id,
        "device_id": req.device_id,
        "already_registered": False,
        "linked_devices": devices,
        "max_devices": MAX_DEVICES_PER_USER,
    }
    # Dispositivo nuevo: token recien emitido. Es la UNICA vez que se entrega
    # (ver _issue_device_token); el cliente tiene que persistirlo.
    issued = _issue_device_token(conn, req.device_id)
    if issued:
        resp["device_token"] = issued
    return resp


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


def _migrate_abandoned_account(conn: sqlite3.Connection, old_user: str, new_user: str):
    """Si el device era el ULTIMO de su cuenta vieja, se lleva sus datos.

    `_assign_orphan_data` solo rescata items SIN `user_id`. Pero un dispositivo
    que sincroniza antes de vincularse no tiene datos huerfanos: tiene datos de
    la cuenta que el servidor le creo solo. Al vincularlo, esos items se
    quedaban bajo la cuenta vieja, invisibles para siempre — no los ve el otro
    dispositivo, no los cuenta `/sync/publish`, y no hay forma de borrarlos.
    Reportado por el owner: analizo 4 tracks en el movil antes de vincular y no
    hubo manera ni de verlos desde el Mac ni de quitarlos.

    **Solo si la cuenta vieja se queda SIN dispositivos.** Si le quedan otros,
    esos items son parte de SU biblioteca y llevarselos seria quitarselos a un
    usuario que sigue usandolos. Cuando no queda ninguno, en cambio, los datos
    son inalcanzables para todo el mundo: migrarlos no se los quita a nadie.

    Se llama DESPUES del INSERT del device en la cuenta nueva, para que la rama
    de MAX_DEVICES (que revierte y lanza 409) no deje datos movidos a medias.
    Los items colectivos llevan `user_id='__collective__'`, asi que el WHERE por
    la cuenta vieja no los toca.
    """
    quedan = conn.execute(
        "SELECT COUNT(*) FROM user_devices WHERE user_id = ?", (old_user,)
    ).fetchone()[0]
    if quedan > 0:
        return

    movidos = conn.execute(
        "UPDATE sync_items SET user_id = ? WHERE user_id = ?",
        (new_user, old_user),
    ).rowcount
    try:
        conn.execute(
            "UPDATE detected_tracks_sync SET user_id = ? WHERE user_id = ?",
            (new_user, old_user),
        )
    except sqlite3.OperationalError:
        pass  # la tabla puede no existir en BDs viejas
    try:
        conn.execute("DELETE FROM users WHERE user_id = ?", (old_user,))
    except sqlite3.OperationalError:
        pass
    logger.info(
        f"Cuenta {old_user} sin dispositivos: {movidos} items migrados a {new_user}"
    )


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
async def sync_link_join(req: LinkJoinRequest, request: Request):
    """
    Vincula un dispositivo a un usuario existente usando el código de 6 caracteres.
    Si el dispositivo ya está registrado con otro usuario, se re-vincula.
    """
    from validation import get_client_ip

    ip = get_client_ip(request)
    _check_join_attempts(ip)

    conn = _get_conn()

    # Buscar código válido
    row = conn.execute(
        "SELECT user_id, expires_at FROM link_codes WHERE code = ?",
        (req.code.upper(),),
    ).fetchone()

    if not row:
        _record_join_failure(ip)
        raise HTTPException(status_code=404, detail="Invalid or expired link code")

    target_user_id, expires_at = row
    if expires_at < _now_iso():
        conn.execute("DELETE FROM link_codes WHERE code = ?", (req.code.upper(),))
        conn.commit()
        _record_join_failure(ip)
        raise HTTPException(status_code=410, detail="Link code expired")

    now = _now_iso()

    # Si el dispositivo ya estaba registrado, re-vincular
    existing_user = _get_user_id_for_device(conn, req.device_id)
    if existing_user == target_user_id:
        # Ya vinculado al mismo usuario
        conn.commit()
        devices = _get_all_devices_for_user(conn, target_user_id)
        return {
            "user_id": target_user_id,
            "device_id": req.device_id,
            "already_linked": True,
            "linked_devices": devices,
            "max_devices": MAX_DEVICES_PER_USER,
        }

    # El token del dispositivo se preserva a traves del DELETE/INSERT de abajo.
    # Vincular es una operacion de CUENTA (a que usuario pertenece el device),
    # mientras que el token acredita la IDENTIDAD del device: cambiar de usuario
    # no lo convierte en otro aparato. Sin esto, unir dos dispositivos con el
    # codigo le borraria el token al que se une, y en fase 3 (token obligatorio)
    # eso seria dejarlo fuera de sus propios datos.
    # FASE 2.5: se preserva TAMBIEN `token_seen_at`. Si solo se arrastrara el
    # token, un dispositivo YA PROTEGIDO que se vincule a otra cuenta volveria a
    # aceptar peticiones sin token — o sea, vincular reabriria el agujero justo
    # en el aparato que ya estaba cerrado, y en silencio.
    _tok_row = conn.execute(
        "SELECT device_token, token_seen_at FROM user_devices "
        "WHERE device_id = ?",
        (req.device_id,),
    ).fetchone()
    preserved_token = _tok_row[0] if _tok_row else None
    preserved_seen = _tok_row[1] if _tok_row else None

    if existing_user:
        # Re-vincular de un usuario a otro
        conn.execute("DELETE FROM user_devices WHERE device_id = ?", (req.device_id,))

    # Soft cap: max dispositivos por usuario.
    # Si el user ya tiene MAX_DEVICES_PER_USER, bloquear el join.
    # Si el device venia re-vinculandose desde otro user, ya hicimos DELETE
    # arriba asi que el count refleja el estado sin el nuevo device.
    current_count = conn.execute(
        "SELECT COUNT(*) FROM user_devices WHERE user_id = ?",
        (target_user_id,),
    ).fetchone()[0]
    if current_count >= MAX_DEVICES_PER_USER:
        # Si hicimos DELETE antes, restaurar para no corromper estado.
        if existing_user:
            conn.execute(
                "INSERT INTO user_devices (device_id, user_id, device_type, device_name, "
                "linked_at, device_token, token_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (req.device_id, existing_user, req.device_type, req.device_name, now,
                 preserved_token, preserved_seen),
            )
            conn.commit()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MAX_DEVICES_REACHED",
                "message": f"Max {MAX_DEVICES_PER_USER} devices per user. Unlink one to add another.",
                "current": current_count,
                "max": MAX_DEVICES_PER_USER,
            },
        )

    conn.execute(
        "INSERT INTO user_devices (device_id, user_id, device_type, device_name, "
        "linked_at, device_token, token_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (req.device_id, target_user_id, req.device_type, req.device_name, now,
         preserved_token, preserved_seen),
    )

    # Migrar datos huérfanos
    _assign_orphan_data(conn, req.device_id, target_user_id)
    if existing_user and existing_user != target_user_id:
        _migrate_abandoned_account(conn, existing_user, target_user_id)

    # No consumir el código: se mantiene válido hasta que expire (10 min)
    # para permitir vincular múltiples dispositivos con el mismo código.
    conn.commit()

    devices = _get_all_devices_for_user(conn, target_user_id)
    logger.info(f"Device {req.device_id} linked to user {target_user_id} via code {req.code}")

    return {
        "user_id": target_user_id,
        "device_id": req.device_id,
        "already_linked": False,
        "linked_devices": devices,
        "max_devices": MAX_DEVICES_PER_USER,
    }


class UnlinkRequest(BaseModel):
    device_id: str
    target_device_id: str


@sync_router.post("/link/unlink")
async def sync_link_unlink(req: UnlinkRequest):
    """
    Desvincula un dispositivo del usuario actual.
    El dispositivo desvinculado queda como un nuevo usuario independiente.
    No se puede desvincular a uno mismo.
    """
    conn = _get_conn()
    user_id = _require_user_id(conn, req.device_id)

    if req.device_id == req.target_device_id:
        raise HTTPException(status_code=400, detail="Cannot unlink yourself")

    # Verificar que el target pertenece al mismo usuario
    target_user = _get_user_id_for_device(conn, req.target_device_id)
    if target_user != user_id:
        raise HTTPException(status_code=404, detail="Target device not found in your account")

    # Crear nuevo usuario para el dispositivo desvinculado
    new_user_id = str(uuid.uuid4())
    now = _now_iso()

    conn.execute(
        "INSERT INTO users (user_id, created_at) VALUES (?, ?)",
        (new_user_id, now),
    )

    # Mover el dispositivo al nuevo usuario
    conn.execute(
        "UPDATE user_devices SET user_id = ? WHERE device_id = ?",
        (new_user_id, req.target_device_id),
    )

    conn.commit()

    # Retornar lista actualizada de dispositivos del usuario
    remaining = conn.execute(
        "SELECT device_id, device_type, device_name, linked_at FROM user_devices WHERE user_id = ?",
        (user_id,),
    ).fetchall()

    devices_list = [
        {"device_id": r[0], "device_type": r[1], "device_name": r[2], "linked_at": r[3]}
        for r in remaining
    ]

    logger.info(f"Device {req.target_device_id} unlinked from user {user_id}")

    return {
        "user_id": user_id,
        "unlinked_device_id": req.target_device_id,
        "linked_devices": devices_list,
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
    full: bool = False,
    types: Optional[str] = None,
    limit: int = 0,   # 0 = sin paginación (compat con clientes viejos)
    offset: int = 0,  # solo se usa cuando limit > 0
):
    """Descarga items cuyo hash sea DIFERENTE al que este dispositivo conoce.

    Multi-tenant: solo devuelve items del MISMO usuario + colectivos.
    Filtra por user_id (no por device_id como antes).

    full=true → IGNORA device_seen Y la exclusión por `last_device_id`, y
    reenvía TODO lo del usuario, incluido lo que subió este mismo dispositivo
    (sin eso, un PC que se formatea recibe 0: es él quien había subido casi
    todo). Necesario
    cuando un cliente se quedó a medias aplicando un pull anterior (p.ej. la
    app se congeló/cerró tras recibir la respuesta pero antes de persistir):
    el servidor ya había marcado esos items como "vistos", así que un pull
    normal los saltaría para siempre. El cliente pide full=true para reparar.

    types=folder,collection,... → limita el pull a esos data_types. El cliente
    móvil lo usa con full=true para reparar SOLO la estructura (carpetas/
    colecciones), sin re-bajar los miles de items per-track de análisis que ya
    tiene en caché (re-bajarlos sería un OOM inútil en móvil).

    limit/offset → paginación para installs frescos con bibliotecas grandes.
    El cliente móvil descarga 200 items a la vez y aplica cada página antes
    de pedir la siguiente. Así el pico de RAM es ~200 items en vez de ~5000.
    has_more=true en la respuesta indica que quedan páginas por descargar.
    """
    conn = _get_conn()
    user_id = _require_user_id(conn, device_id)

    type_list = (
        [t.strip() for t in types.split(",") if t.strip()] if types else None
    )

    # Orden por rowid garantiza paginación estable (rowid no cambia para
    # rows existentes en SQLite; nuevas inserciones van al final).
    # El filtro `last_device_id != ?` evita el eco: en un pull normal no tiene
    # sentido devolverle a un dispositivo lo que acaba de subir el.
    #
    # Con full=true hay que QUITARLO, y esa es justo la razon de ser de este
    # endpoint. `full=true` lo pide un dispositivo que perdio sus datos en
    # local y quiere recuperarlos de la nube. Si ademas excluimos lo que subio
    # el mismo, al PC —que es la fuente de la verdad y por tanto quien subio
    # casi todo— se le devuelve CERO: el unico caso para el que existe
    # "Restaurar desde la nube" era el unico que no funcionaba. Reportado con
    # un Mac formateado que pidio full=true sobre 5.000 tracks y recibio 0,
    # mientras el movil seguia enseñandolos porque los tenia de un pull viejo.
    #
    # Sin riesgo de eco en el PUSH: al aplicar un pull el cliente NO marca dirty
    # (escribe a disco y recarga por replaceFromDisk/loadFromDisk), asi que
    # recibir lo propio no genera un push de vuelta.
    #
    # Si hay riesgo de eco al APLICARLO, y costo una carpeta: devolver lo que
    # subio este mismo dispositivo significa devolverle una FOTO VIEJA DE SI
    # MISMO. Los bloques de organizacion (all_folders, all_collections…) se
    # aplican por sustitucion completa, asi que un aparato que subio "0
    # carpetas", creo una despues y luego pidio full=true, se pisaba su propia
    # carpeta con el vacio de antes. Por eso ahora se devuelve `last_device_id`:
    # el cliente necesita poder reconocer su propio eco y no aplicarlo encima de
    # datos que ya tiene. El filtro NO se puede hacer aqui — sin lo propio, el
    # PC formateado vuelve a recibir cero, que es el bug de arriba.
    sql = (
        "SELECT si.key, si.data_type, si.item_key, si.payload, si.deleted, "
        "       si.updated_at, si.device_type, si.hash, si.last_device_id "
        "FROM sync_items si "
        "WHERE (si.user_id = ? OR si.user_id = '__collective__')"
    )
    params: list = [user_id]
    if not full:
        sql += " AND si.last_device_id != ?"
        params.append(device_id)
    if type_list:
        placeholders = ",".join("?" for _ in type_list)
        sql += f" AND si.data_type IN ({placeholders})"
        params.extend(type_list)
    sql += " ORDER BY si.rowid"
    rows = conn.execute(sql, params).fetchall()

    # Pre-cargar lo que este dispositivo ya conoce en UNA query (en vez de un
    # SELECT por item dentro del loop). Lookup O(1) en memoria.
    seen_map: dict = {}
    if not full:
        for ik, h in conn.execute(
            "SELECT item_key, hash FROM device_seen WHERE device_id = ?",
            (device_id,),
        ):
            seen_map[ik] = h

    # Construir lista completa de cambios (con o sin filtro device_seen)
    all_items: list = []  # list of (change_dict, (key, hash, payload_json))

    for row in rows:
        (key, data_type, item_key, payload_json, deleted, updated_at,
         device_type, item_hash, last_device_id) = row

        # Verificar si el dispositivo ya conoce este hash (salvo full=true)
        if not full and seen_map.get(key) == item_hash:
            continue

        all_items.append((
            {
                "data_type":   data_type,
                "item_key":    item_key,
                "payload":     json.loads(payload_json),
                "deleted":     bool(deleted),
                "updated_at":  updated_at,
                "device_type": device_type,
                # Quien lo subio. El cliente lo usa para no aplicarse su propio
                # eco encima de datos mas nuevos (ver la nota del SELECT).
                "last_device_id": last_device_id,
            },
            (key, item_hash, payload_json),
        ))

    # Paginación. Los dos modos usan cursores DISTINTOS y mezclarlos pierde
    # items en silencio (SYNC-01 de la auditoría 2026-08-09):
    #
    #   full=true  -> NO se filtra por device_seen, así que `all_items` es
    #                 estable entre páginas y `offset` ES el cursor correcto.
    #                 Es la ruta que usa hoy _paginatedPullMobile (solo se
    #                 pagina cuando _forceFullPull), y funciona bien.
    #
    #   full=false -> `device_seen` YA actúa de cursor: cada página entregada
    #                 se marca como vista, así que la siguiente petición
    #                 arranca donde acabó la anterior. Honrar ADEMÁS el
    #                 `offset` suma dos avances y salta `limit` items por
    #                 página. Medido: 5000 tracks paginados de 200 en 200
    #                 entregaban 2600. Por eso aquí el offset se IGNORA.
    #
    # Hoy ningún llamador combina full=false con offset, así que esto es
    # blindaje: deja el endpoint correcto sea cual sea el modo, para que un
    # cambio futuro en el cliente (o un script) no reabra el agujero.
    #
    # `has_more` sale además del `if` interior: antes vivía dentro de
    # `limit < len(all_items)`, así que cuando quedaban menos items que el
    # `limit` el recorte NO se aplicaba y se devolvía la lista entera desde
    # el principio, ignorando el offset.
    effective_offset = offset if full else 0
    has_more = False
    if limit > 0:
        has_more = len(all_items) > effective_offset + limit
        all_items = all_items[effective_offset:effective_offset + limit]

    changes = [item[0] for item in all_items]
    update_seen = [item[1] for item in all_items]

    # Marcar como vistos reutilizando el payload ya leído arriba (antes se
    # re-consultaba sync_items una vez por item).
    for key, h, payload_to_save in update_seen:
        conn.execute(
            """INSERT INTO device_seen (device_id, item_key, hash, payload)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(device_id, item_key) DO UPDATE SET hash = ?, payload = ?""",
            (device_id, key, h, payload_to_save, h, payload_to_save),
        )
    conn.commit()

    return {
        "changes":   changes,
        "total":     len(changes),
        "has_more":  has_more,
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

    # Items remotos del mismo usuario + colectivos
    remote_rows = conn.execute(
        """SELECT si.key, si.data_type, si.item_key, si.payload, si.hash
           FROM sync_items si
           WHERE si.last_device_id != ?
             AND (si.user_id = ? OR si.user_id = '__collective__')""",
        (device_id, user_id),
    ).fetchall()

    # Pre-cargar device_seen de este dispositivo en UNA query (en vez de un
    # SELECT por item dentro del loop). Lookup O(1) en memoria.
    seen_map: dict = {}
    for ik, h, p in conn.execute(
        "SELECT item_key, hash, payload FROM device_seen WHERE device_id = ?",
        (device_id,),
    ):
        seen_map[ik] = (h, p)

    detail: dict[str, int] = {}
    total = 0

    for row in remote_rows:
        key, data_type, item_key, payload_json, item_hash = row

        # Qué versión conoce este dispositivo (de device_seen)
        seen = seen_map.get(key)

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
        # Items per-track (2.9.3): el payload es UN análisis (clave 'track'),
        # no el blob {tracks:{...}}. Contar como una unidad modificada. Sin
        # esto, la lógica de blob de abajo daría 0 y el contador mentiría.
        if "tracks" not in remote_payload:
            if _payload_hash(local_payload) != _payload_hash(remote_payload):
                result["analysis_modified"] = 1
            return result
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


# ── PUBLISH ───────────────────────────────────────────────────


class PublishLibraryRequest(BaseModel):
    device_id: str
    device_type: str = "unknown"
    track_ids: List[str]
    apply: bool = False


@sync_router.post("/publish")
async def sync_publish(req: PublishLibraryRequest):
    """El escritorio declara su biblioteca COMPLETA; lo que no esté, se borra.

    Por qué hace falta un endpoint aparte del push: el push manda CAMBIOS
    ("añade esto", "borra aquello"), así que la nube solo se entera de un
    borrado si alguien se lo dice track a track. Cuando el usuario llega a su
    estado limpio por otro camino —formateo, reinstalación, disco nuevo— no hay
    nada que decir: el servidor conserva los miles de tracks viejos y los sigue
    sirviendo a los demás dispositivos. Este endpoint es la frase que faltaba:
    "mi biblioteca es exactamente esta".

    Dos fases a propósito. Con `apply=false` solo CUENTA lo que se perdería y
    no toca nada, para que el cliente pueda enseñar "esto marcará N tracks como
    borrados" ANTES de que el usuario decida. Es la operación más destructiva
    de la app: no puede ejecutarse sin que se vea antes el daño.

    Solo escritorio. Misma regla que el resto de borrados propagados: el móvil
    es el aparato que se reinstala y se queda sin espacio, y no puede imponerle
    su estado al resto.

    NO toca `all_analysis` (el blob de clientes < v2.9.3): ese item es la
    biblioteca entera en una sola fila, y darlo por borrado dejaría a un cliente
    legacy sin nada. Se ignora en la comparación.
    """
    # `is_desktop_platform`, NO una lista a mano. La lista literal que habia
    # aqui —("windows","macos","linux")— empezo a devolver 403 a TODOS los Mac
    # en cuanto `Analyzer#109` afino `platformHeader` a `macos-dmg`/`macos-mas`:
    # el cliente mandaba el valor nuevo y el guard solo conocia el viejo.
    # Publicar, la feature estrella de la 2.9.10, no funcionaba en ningun Mac.
    if not is_desktop_platform(req.device_type):
        raise HTTPException(
            status_code=403,
            detail=(
                "Publish requires a desktop device "
                f"(got {req.device_type!r})"
            ),
        )

    conn = _get_conn()
    user_id = _require_user_id(conn, req.device_id)

    declarados = set(req.track_ids)

    # Candidatos: los análisis vivos de ESTE usuario. Los colectivos no entran
    # (no son suyos) y `all_analysis` tampoco (ver docstring).
    sobrantes = []
    for item_key, in conn.execute(
        "SELECT item_key FROM sync_items "
        " WHERE user_id = ? AND data_type = 'analysis' AND deleted = 0",
        (user_id,),
    ):
        if item_key == "all_analysis":
            continue
        if item_key not in declarados:
            sobrantes.append(item_key)

    if not req.apply:
        return {
            "would_delete": len(sobrantes),
            "declared": len(declarados),
            "sample": sobrantes[:20],
            "applied": False,
        }

    now = _now_iso()
    payload_json = json.dumps({})
    borrados = 0
    for item_key in sobrantes:
        key = f"{user_id}|analysis|{item_key}"
        # Se marca borrado y se sella con el device que publica, para que el
        # filtro anti-eco del pull se lo entregue a TODOS los demás.
        conn.execute(
            "UPDATE sync_items "
            "   SET deleted = 1, payload = ?, updated_at = ?, "
            "       last_device_id = ?, device_type = ?, hash = ? "
            " WHERE key = ?",
            (payload_json, now, req.device_id, req.device_type,
             _payload_hash({}), key),
        )
        # Sin esto, el dispositivo que publica se guarda su propio borrado como
        # "ya visto" con el hash viejo y podría reenviárselo a sí mismo.
        conn.execute("DELETE FROM device_seen WHERE item_key = ?", (key,))
        borrados += 1

    conn.commit()
    return {
        "would_delete": borrados,
        "declared": len(declarados),
        "applied": True,
        "timestamp": now,
    }


# ── CLEAR ─────────────────────────────────────────────────────

@sync_router.delete("/clear")
async def sync_clear(request: Request, device_id: Optional[str] = None):
    # Full clear requires valid ADMIN_TOKEN
    admin_key = request.headers.get("X-Admin-Key", "")

    # Constant-time comparison to prevent timing attacks on the admin key.
    admin_ok = bool(admin_key and ADMIN_TOKEN and _hmac.compare_digest(admin_key, ADMIN_TOKEN))
    if not device_id and not admin_ok:
        raise HTTPException(
            status_code=403,
            detail="Full clear requires valid X-Admin-Key header. Per-device clear requires device_id param."
        )

    conn = _get_conn()

    if device_id:
        # Only clear data for a specific device (still protected by sync HMAC auth)
        conn.execute("DELETE FROM sync_items WHERE last_device_id = ?", (device_id,))
        conn.execute("DELETE FROM device_seen WHERE device_id = ?", (device_id,))
        conn.commit()
        return {"cleared": True, "scope": f"device:{device_id}"}

    # Full clear (admin only — verified above)
    conn.execute("DELETE FROM sync_items")
    conn.execute("DELETE FROM device_seen")
    conn.commit()
    return {"cleared": True, "scope": "all"}


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


@sync_router.delete("/detected-tracks/{device_id}")
async def sync_clear_detected_tracks(device_id: str):
    """
    Borra TODOS los detected tracks del usuario asociado a este device_id.
    Multi-tenant: afecta a todos los devices vinculados al mismo user_id.

    Lo invoca el cliente desde "BORRADO TOTAL" en Settings: sin esto, al
    re-abrir el History tab el pullFromCloud() repuebla el historial desde
    el backend porque los registros sobreviven al wipe local.

    Devuelve cuántas filas se borraron para feedback del cliente.
    """
    conn = _get_conn()
    user_id = _require_user_id(conn, device_id)

    try:
        cur = conn.execute(
            "DELETE FROM detected_tracks_sync WHERE user_id = ?",
            (user_id,),
        )
        deleted = cur.rowcount
        conn.commit()
        logger.info(
            f"Cleared {deleted} detected tracks for user_id={user_id[:8]}..."
        )
        return {
            "status": "ok",
            "deleted": deleted,
            "user_id": user_id,
            "server_time": _now_iso(),
        }
    except sqlite3.Error as e:
        logger.error(f"Clear detected tracks error: {e}")
        return {"status": "error", "deleted": 0, "message": str(e)}


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
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Admin token required")
    # Constant-time comparison to prevent timing attacks.
    if not _hmac.compare_digest(auth[7:], ADMIN_TOKEN):
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
        # `token_seen_at` viaja porque es LA senal que decide si un dispositivo
        # va a comerse un 401 «Device token required» — y no habia forma de
        # verla desde fuera. Dos consultas mas no: son columnas de la fila que
        # ya se estaba leyendo.
        devices = conn.execute(
            "SELECT device_id, device_type, device_name, linked_at, "
            "       device_token IS NOT NULL, token_seen_at "
            "FROM user_devices WHERE user_id = ?",
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
                    "has_token": bool(d[4]),
                    "token_seen_at": d[5],
                    "token_enforced": bool(d[5]),
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


@admin_sync_router.get("/device/{device_id}")
async def admin_device_diagnosis(device_id: str):
    """Por que ESTE aparato no esta sincronizando. Una llamada, una respuesta.

    «El sync con el movil ha dejado de funcionar» no se podia contestar desde
    ningun sitio. Las causas posibles piden acciones OPUESTAS y hasta ahora no
    habia forma de distinguirlas sin entrar a la BD a mano:

      | `verdict`            | Que pasa                                    |
      |----------------------|---------------------------------------------|
      | `not_registered`     | el device_id no existe -> todo da 403       |
      | `token_enforced`     | mando token alguna vez; si ahora llega sin  |
      |                      | el, 401 «Device token required» PARA SIEMPRE|
      | `alone`              | esta solo en su cuenta: no hay nada que     |
      |                      | sincronizar, se perdio la vinculacion       |
      | `nothing_pending`    | esta al dia — el problema no es el servidor |
      | `pending`            | tiene N items esperando y no viene a por    |
      |                      | ellos: mirar el cliente                     |

    Y el dato que de verdad zanja la discusion: `siblings`, con lo que subio
    CADA aparato de la cuenta y cuando. Si el Mac tiene `last_push` de hace un
    minuto y el movil `pending: 0`, el que no habla es el movil aunque el
    usuario jure lo contrario. Es la misma leccion que ya costo un dia entero:
    los dos lados de la MISMA tanda, uno al lado del otro.

    Coste: consultas puntuales por device_id y un COUNT, nada de recorrer la
    biblioteca. Nunca devuelve el `device_token`.
    """
    conn = _get_conn()

    row = conn.execute(
        "SELECT user_id, device_type, device_name, linked_at, "
        "       device_token IS NOT NULL, token_seen_at "
        "FROM user_devices WHERE device_id = ?",
        (device_id,),
    ).fetchone()

    if row is None:
        return {
            "device_id": device_id,
            "registered": False,
            "verdict": "not_registered",
            "note": (
                "El device_id no existe en user_devices: /sync/push y /sync/pull "
                "devuelven 403. El cliente tiene que llamar a /sync/register."
            ),
        }

    user_id, device_type, device_name, linked_at, has_token, token_seen_at = row

    # Lo que ESTE aparato ha subido.
    pushed, last_push = conn.execute(
        "SELECT COUNT(*), MAX(updated_at) FROM sync_items WHERE last_device_id = ?",
        (device_id,),
    ).fetchone()

    # Lo que le espera: items de su usuario (o colectivos) que no ha visto, o
    # que ha visto con OTRO hash. El mismo criterio que /sync/pull, contado.
    pending = conn.execute(
        """SELECT COUNT(*)
           FROM sync_items si
           LEFT JOIN device_seen ds
                  ON ds.device_id = ? AND ds.item_key = si.key
           WHERE si.last_device_id != ?
             AND (si.user_id = ? OR si.user_id = '__collective__')
             AND (ds.hash IS NULL OR ds.hash != si.hash)""",
        (device_id, device_id, user_id),
    ).fetchone()[0]

    siblings = []
    for d in conn.execute(
        "SELECT device_id, device_type, device_name, linked_at, "
        "       device_token IS NOT NULL, token_seen_at "
        "FROM user_devices WHERE user_id = ? AND device_id != ?",
        (user_id, device_id),
    ).fetchall():
        s_pushed, s_last = conn.execute(
            "SELECT COUNT(*), MAX(updated_at) FROM sync_items "
            "WHERE last_device_id = ?",
            (d[0],),
        ).fetchone()
        siblings.append({
            "device_id": d[0],
            "device_type": d[1],
            "device_name": d[2],
            "linked_at": d[3],
            "has_token": bool(d[4]),
            "token_enforced": bool(d[5]),
            "items_pushed": s_pushed,
            "last_push": s_last,
        })

    if token_seen_at:
        verdict = "token_enforced"
    elif not siblings:
        verdict = "alone"
    elif pending > 0:
        verdict = "pending"
    else:
        verdict = "nothing_pending"

    return {
        "device_id": device_id,
        "registered": True,
        "user_id": user_id,
        "device_type": device_type,
        "device_name": device_name,
        "linked_at": linked_at,
        "has_token": bool(has_token),
        "token_seen_at": token_seen_at,
        "token_enforced": bool(token_seen_at),
        "items_pushed": pushed,
        "last_push": last_push,
        "pending_for_this_device": pending,
        "siblings": siblings,
        "verdict": verdict,
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

