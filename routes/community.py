"""
Community endpoints for DJ Analyzer Pro API: beat-grid corrections, generic
field overrides (track_type/key/camelot/genre/subgenre/bpm/energy/year),
legacy track-type compat, notes, ratings and popularity.

PASO 3 del troceo de main.py (review 2026-06-29). Bloque movido VERBATIM
desde main.py (mismo comportamiento, sin cambio de logica). Todos los
modelos, helpers (_validate_community_field, _community_override_response)
y constantes (COMMUNITY_*) viven aqui porque SOLO los usaban estos
endpoints — verificado por grep antes del move.

La unica dependencia global de main.py es la instancia de BD, inyectada via
init(database) ANTES de app.include_router(community_router). Igual que
routes/search.py (paso 2): este router SI se monta y los endpoints inline
se BORRAN -> sin duplicacion stale.
"""

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Dependencia inyectada desde main.py ──────────────────────
db = None


def init(database):
    """Inyecta la instancia AnalysisDB desde main.py. Llamar ANTES de
    app.include_router(community_router)."""
    global db
    db = database


# ── Router ───────────────────────────────────────────────────
community_router = APIRouter(tags=["community"])


# ── Mitigación SEC-12: consenso Sybil-abierto ────────────────
#
# Los endpoints de comunidad reciben `device_id` como campo PLANO del cuerpo,
# sin autenticar, así que "un dispositivo, un voto" es autodeclarado: mandar
# tres ids inventados produce un `consensus_3` (prioridad 80) que gana al motor
# local (50), y con diez un `consensus_10` (95). Un solo actor podía fabricar el
# consenso de BPM, tonalidad, género, track type, beat grid y ratings de
# cualquier track, y eso se sirve a todos como "validado por la comunidad".
#
# EL ARREGLO DE FONDO — CORREGIDO 2026-08-10, la versión anterior de este
# comentario decía una cosa que NO se sostiene al mirar el código:
#
#   ❌ "derivar el device_id de la identidad de sync, que ya está autenticada
#      por HMAC". NO SIRVE. `SYNC_AUTH_SECRET` es un secreto COMPARTIDO,
#      embebido por dart-define en TODOS los builds (y admite lista para
#      rotarlo, `sync_endpoints.py:80`). Su HMAC demuestra "esto es un cliente
#      legítimo", no "esto es el dispositivo X". Derivar identidad de ahí es
#      imposible: la firma es idéntica para todos los usuarios del mundo.
#
#   ✅ Lo que SÍ existe es un registro real de dispositivos: `user_devices`
#      (device_id → user_id), que se puebla en POST /sync/register, exige
#      código de vinculación para unir un device a un user y está topado en
#      MAX_DEVICES_PER_USER = 20. El arreglo es (a) exigir que el device_id que
#      llega a /community/* esté registrado ahí, y (b) contar el consenso por
#      user_id, NO por device_id.
#
#      (b) es lo que de verdad cierra el agujero, y además arregla un sesgo que
#      hoy tenemos: el DJ con desktop + móvil + tablet cuenta como TRES votos
#      cuando es una sola opinión. Fabricar consensus_3 pasaría a exigir tres
#      CUENTAS distintas, con su vinculación cada una.
#
#      Ojo al desplegarlo: (a) deja fuera a quien use la app sin haber vinculado
#      nunca. Va con el mismo rollout por fases que el HMAC de escritura —
#      primero contar en log cuántos votos llegan de devices no registrados, y
#      solo entonces rechazar. No lo actives de golpe. (Ver PENDING.md.)
#
# ESTO es la mitigación que se puede desplegar sola: contar cuántas identidades
# DISTINTAS usa una misma IP para votar el MISMO track. Un usuario real vota un
# track concreto desde uno o dos dispositivos; fabricar consenso exige tres o
# más. Al superar el umbral se rechaza con 429.
#
# No es infalible —quien use proxies rota de IP— pero sube el coste del ataque
# de "un curl en bucle" a "una botnet", que para el tamaño del proyecto es la
# diferencia entre pasar y no pasar. Se registra en log cada corte para poder
# ver si hay falsos positivos antes de endurecer.
MAX_VOTE_IDENTITIES_PER_IP = int(os.getenv('MAX_VOTE_IDENTITIES_PER_IP', '3'))


def _device_is_registered(device_id: str):
    """¿Está este device_id vinculado a un usuario en `user_devices`?

    SOLO OBSERVACIÓN (fase 1 de SEC-12). Devuelve True/False, o None si no se
    pudo comprobar — y `None` es un valor real, no un fallo silencioso: se
    contabiliza aparte para no confundir "no registrado" con "no lo sabemos".

    Vive en `sync.db`, que es OTRA base que la de comunidad (`analysis.db`), de
    ahí el import perezoso a sync_endpoints en vez de tocar `db`.
    """
    if not device_id:
        return None
    try:
        import sync_endpoints

        return sync_endpoints._get_user_id_for_device(
            sync_endpoints._get_conn(), device_id
        ) is not None
    except Exception as e:  # noqa: BLE001 - observar nunca puede tumbar un voto
        logger.debug(f"[Community] no se pudo comprobar registro de device: {e}")
        return None


def _guard_vote_source(request, fingerprint: str, device_id: str) -> None:
    """Rechaza con 429 si esta IP ya votó este track con demasiadas identidades.

    De paso anota si el device estaba registrado (fase 1 de SEC-12, ver
    `_device_is_registered`): es el dato que falta para decidir si se puede
    exigir registro sin dejar fuera a media base de usuarios.
    """
    if not fingerprint or not device_id:
        return
    try:
        from validation import get_client_ip

        ip_hash = db.hash_ip(get_client_ip(request))
        registrado = _device_is_registered(device_id)
        distintas = db.register_vote_source(
            ip_hash, fingerprint, device_id, registered=registrado,
        )
        if registrado is False:
            logger.info(
                f"[Community][SEC-12 obs] voto de device NO registrado "
                f"({device_id[:8]}…) sobre fp={fingerprint[:12]}. Exigir registro "
                f"hoy lo rechazaría."
            )
    except Exception as e:  # noqa: BLE001 - la mitigación nunca tumba el voto
        logger.warning(f"[Community] guard de procedencia falló (se deja pasar): {e}")
        return

    if distintas > MAX_VOTE_IDENTITIES_PER_IP:
        logger.warning(
            f"[Community] posible Sybil: ip_hash={ip_hash} ha usado {distintas} "
            f"identidades sobre fp={fingerprint[:12]} (max {MAX_VOTE_IDENTITIES_PER_IP})"
        )
        raise HTTPException(
            status_code=429,
            detail="Demasiados dispositivos distintos votando este track desde "
                   "la misma conexión.",
        )


# ==================== COMMUNITY BEAT GRID ====================

class BeatGridCorrectionRequest(BaseModel):
    fingerprint: str
    device_id: str
    bpm_adjust: float = 0.0
    beat_offset: float = 0.0
    original_bpm: float = 0.0

@community_router.post("/community/beat-grid")
async def submit_beat_grid_correction(request: BeatGridCorrectionRequest, http: Request):
    """Recibe correccion de beat grid de un DJ"""
    _guard_vote_source(http, request.fingerprint, request.device_id)
    try:
        db.submit_beat_grid_correction(
            fingerprint=request.fingerprint,
            device_id=request.device_id,
            bpm_adjust=request.bpm_adjust,
            beat_offset=request.beat_offset,
            original_bpm=request.original_bpm,
        )
        print(f"[Community] Beat grid correction: fp={request.fingerprint[:8]}... "
              f"BPM+{request.bpm_adjust:.2f} OFF+{request.beat_offset*1000:.1f}ms")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[Community] Error saving beat grid: {e}")
        # El texto de la excepción (rutas, SQL, internals) no debe viajar a un
        # cliente anónimo: ya queda en el log de arriba con el detalle completo.
        raise HTTPException(500, "Error interno")

@community_router.get("/community/beat-grid/{fingerprint}")
async def get_community_beat_grid(fingerprint: str):
    """Obtiene la correccion promedio de la comunidad"""
    try:
        result = db.get_community_beat_grid(fingerprint)
        return result
    except Exception as e:
        logger.error(f"[Community] Error fetching beat grid: {e}")
        return {"bpm_adjust": 0.0, "beat_offset": 0.0, "contributors": 0, "validated": False}

# ==================== COMMUNITY OVERRIDES (Fase 4 - generico) ====================
# Sistema unificado de votos comunitarios para CUALQUIER campo categorico:
# track_type, key, camelot, genre, subgenre. Mismas reglas de consensus
# (>=3 votos al winner, supera al 2do por >=2). Whitelist por campo aplicada
# en el endpoint POST.

# Whitelist por campo. Valores validos por field. None = string libre con
# normalizacion (genre/subgenre — el cliente envia normalizado).
COMMUNITY_TRACK_TYPES = {
    'warmup', 'peak_time', 'closing', 'opener', 'builder', 'anthem', 'cooldown',
}
COMMUNITY_KEYS = {
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
    'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm',
}
COMMUNITY_CAMELOT = {f'{n}{l}' for n in range(1, 13) for l in 'AB'}

# Validacion + normalizacion por field. Devuelve (normalized_value, error)
# donde error es None si OK, o un mensaje 400 si invalido.
def _validate_community_field(field: str, value: str):
    if not value:
        return None, "value requerido"
    normalized = value.strip()
    if field == 'track_type':
        normalized = normalized.lower()
        if normalized == 'peak':
            normalized = 'peak_time'
        if normalized not in COMMUNITY_TRACK_TYPES:
            return None, (
                f"track_type invalido: {value}. "
                f"Permitidos: {', '.join(sorted(COMMUNITY_TRACK_TYPES))}"
            )
        return normalized, None
    if field == 'key':
        # Normalizacion key: preservar mayuscula raiz + 'm' minuscula para minor.
        if normalized.endswith('m') or normalized.endswith('M'):
            base = normalized[:-1].upper().replace('B', '#').replace('b', '#') if False else normalized[:-1]
            base = base[0].upper() + base[1:] if len(base) > 1 else base.upper()
            normalized = base + 'm'
        else:
            normalized = normalized[0].upper() + (normalized[1:] if len(normalized) > 1 else '')
        if normalized not in COMMUNITY_KEYS:
            return None, f"key invalida: {value}. Esperado p.ej. 'C', 'C#', 'Dm', 'D#m'"
        return normalized, None
    if field == 'camelot':
        normalized = normalized.upper()
        if normalized not in COMMUNITY_CAMELOT:
            return None, f"camelot invalida: {value}. Esperado p.ej. '1A', '12B'"
        return normalized, None
    if field in ('genre', 'subgenre'):
        # Strings libres con normalizacion suave: capitalize palabras.
        # Limite longitud para evitar abuso.
        if len(normalized) > 100:
            return None, f"{field} demasiado largo (max 100 caracteres)"
        # Capitalize each word (Title Case).
        normalized = ' '.join(w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper()
                              for w in normalized.split())
        return normalized, None
    if field == 'bpm':
        try:
            bpm_val = float(normalized)
        except (TypeError, ValueError):
            return None, "bpm debe ser numerico"
        if bpm_val <= 0 or bpm_val > 999:
            return None, "bpm fuera de rango (0.1-999)"
        from bpm_utils import normalize_bpm_to_canonical
        try:
            canonical = normalize_bpm_to_canonical(bpm_val)
        except ValueError as e:
            return None, str(e)
        return str(canonical), None
    if field == 'energy':
        try:
            e_val = int(float(normalized))
        except (TypeError, ValueError):
            return None, "energy debe ser entero"
        if e_val < 1 or e_val > 10:
            return None, "energy debe estar entre 1 y 10"
        return str(e_val), None
    if field == 'year':
        try:
            y_val = int(normalized)
        except (TypeError, ValueError):
            return None, "year debe ser entero"
        import datetime as _dt
        current_year = _dt.datetime.now().year
        if y_val < 1900 or y_val > current_year + 1:
            return None, f"year fuera de rango (1900-{current_year + 1})"
        return str(y_val), None
    return None, f"field no soportado: {field}"


class CommunityOverrideRequest(BaseModel):
    fingerprint: str
    device_id: str
    field: str
    value: str


COMMUNITY_NUMERIC_FIELDS = {'bpm', 'energy'}
COMMUNITY_CATEGORICAL_FIELDS = {'track_type', 'key', 'camelot', 'genre', 'subgenre', 'year'}
COMMUNITY_VALID_FIELDS = COMMUNITY_NUMERIC_FIELDS | COMMUNITY_CATEGORICAL_FIELDS


def _community_override_response(fingerprint: str, field: str) -> dict:
    """Helper compartido: distribucion + consensus de un (fp, field).

    Numericos (bpm, energy) usan mediana via get_community_consensus_numeric.
    Categoricos (track_type/key/camelot/genre/subgenre/year) usan moda via
    get_community_consensus. Shape de respuesta es identica para no romper
    al frontend Fase 4.
    """
    if field in COMMUNITY_NUMERIC_FIELDS:
        result = db.get_community_consensus_numeric(fingerprint, field)
        return {
            "fingerprint": fingerprint,
            "field": field,
            "consensus": result['consensus'],
            "consensus_votes": result['consensus_votes'],
            "votes": result['votes_distribution'],
            "total_voters": result['total_voters'],
        }
    consensus = db.get_community_consensus(fingerprint, field)
    votes = db.get_community_votes(fingerprint, field)
    return {
        "fingerprint": fingerprint,
        "field": field,
        "consensus": consensus['value'] if consensus else None,
        "consensus_votes": consensus['votes'] if consensus else 0,
        "votes": votes,
        "total_voters": sum(votes.values()),
    }


@community_router.post("/community/override")
async def submit_community_override(request: CommunityOverrideRequest, http: Request = None):
    """Recibe un voto de un DJ sobre cualquier campo categorico de un track.

    Campos soportados: track_type, key, camelot, genre, subgenre.
    Un device puede votar 1 campo 1 vez por track; segundo POST sobreescribe.
    Cuando >=3 votos al winner Y supera al 2do por >=2, la respuesta de
    /analyze para ese fingerprint devuelve {field}_source='community'.
    """
    try:
        if not request.fingerprint or not request.device_id or not request.field:
            raise HTTPException(400, "fingerprint, device_id y field requeridos")
        if http is not None:
            _guard_vote_source(http, request.fingerprint, request.device_id)
        normalized, error = _validate_community_field(request.field, request.value)
        if error:
            raise HTTPException(400, error)
        db.submit_community_override(
            fingerprint=request.fingerprint,
            device_id=request.device_id,
            field=request.field,
            value=normalized,
        )
        logger.info(
            f"[Community] {request.field}: fp={request.fingerprint[:8]}... "
            f"device={request.device_id[:8]}... -> {normalized}"
        )
        return {"status": "ok", **_community_override_response(request.fingerprint, request.field)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Community] Error saving override: {e}")
        # El texto de la excepción (rutas, SQL, internals) no debe viajar a un
        # cliente anónimo: ya queda en el log de arriba con el detalle completo.
        raise HTTPException(500, "Error interno")


@community_router.get("/community/override/{field}/{fingerprint}")
async def get_community_override(field: str, fingerprint: str):
    """Devuelve consensus + distribucion de votos para (field, fingerprint)."""
    if field not in COMMUNITY_VALID_FIELDS:
        raise HTTPException(400, f"field no soportado: {field}")
    try:
        return _community_override_response(fingerprint, field)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Community] Error fetching {field} consensus: {e}")
        return {
            "fingerprint": fingerprint, "field": field,
            "consensus": None, "consensus_votes": 0,
            "votes": {}, "total_voters": 0,
        }


@community_router.delete("/community/override/{field}/{fingerprint}")
async def delete_community_override(field: str, fingerprint: str, device_id: str):
    """Retira el voto del device sobre (field, fingerprint). Fase 5.5.

    Idempotente: si el voto no existia, devuelve 200 con deleted=False. Asi
    el cliente puede llamar "retirar voto" sin chequear previamente.

    Args:
        field: campo del cual retirar el voto. Whitelisted contra
            COMMUNITY_VALID_FIELDS para evitar abuse.
        fingerprint: hash del track.
        device_id: identificador del device (query param: ?device_id=X).
    """
    if field not in COMMUNITY_VALID_FIELDS:
        raise HTTPException(400, f"field no soportado: {field}")
    if not device_id:
        raise HTTPException(400, "device_id requerido")
    try:
        deleted = db.delete_community_override(fingerprint, device_id, field)
        logger.info(
            f"[Community] DELETE {field}: fp={fingerprint[:8]}... "
            f"device={device_id[:8]}... deleted={deleted}"
        )
        return {
            "status": "ok",
            "deleted": deleted,
            **_community_override_response(fingerprint, field),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Community] Error deleting {field} vote: {e}")
        # El texto de la excepción (rutas, SQL, internals) no debe viajar a un
        # cliente anónimo: ya queda en el log de arriba con el detalle completo.
        raise HTTPException(500, "Error interno")


# ==================== COMMUNITY TRACK TYPE (Fase 2 backwards-compat) ====================
# Mantenidos para no romper clientes Fase 2 que no se actualizaron al
# endpoint generico. Internamente delegan al generico via DB.

class TrackTypeOverrideRequest(BaseModel):
    fingerprint: str
    device_id: str
    track_type: str

@community_router.post("/community/track-type")
async def submit_track_type_override(request: TrackTypeOverrideRequest, http: Request = None):
    """Legacy Fase 2: voto de track_type. Delega al endpoint generico.

    OJO: `http` se propaga al generico A PROPOSITO. Sin ello este endpoint era
    un BYPASS COMPLETO del guard SEC-12: el generico solo llama a
    `_guard_vote_source` si recibe la Request, y aqui se delegaba con
    `http=None`, asi que bastaba apuntar el curl a /community/track-type en vez
    de a /community/override para fabricar consenso de track_type sin limite.
    """
    proxy = CommunityOverrideRequest(
        fingerprint=request.fingerprint,
        device_id=request.device_id,
        field='track_type',
        value=request.track_type,
    )
    result = await submit_community_override(proxy, http)
    # Shape Fase 2: 'consensus' string (no objeto con field).
    return {
        "status": result.get("status", "ok"),
        "votes": result.get("votes", {}),
        "consensus": result.get("consensus"),
        "consensus_votes": result.get("consensus_votes", 0),
    }

@community_router.get("/community/track-type/{fingerprint}")
async def get_community_track_type(fingerprint: str):
    """Legacy Fase 2: consensus de track_type."""
    r = _community_override_response(fingerprint, 'track_type')
    # Shape Fase 2.
    return {
        "fingerprint": fingerprint,
        "consensus": r["consensus"],
        "consensus_votes": r["consensus_votes"],
        "votes": r["votes"],
        "total_voters": r["total_voters"],
    }


# ==================== COMMUNITY NOTES / RATINGS / POPULARITY ====================
# Portados inline desde routes/community.py: ese community_router NUNCA se monta
# en main.py (solo sync_router + admin_panel_router via include_router), asi que
# /community/notes, /community/rate y /community/popularity devolvian 404 (ruta
# inexistente). Beat-grid y override ya estaban inline mas arriba; estos faltaban.

class CommunityNoteRequest(BaseModel):
    fingerprint: str
    device_id: str
    note_text: str
    display_name: str = "DJ"
    note_type: str = "general"  # general, technique, mixing, warning


class TrackRatingRequest(BaseModel):
    fingerprint: str
    device_id: str
    rating: int  # 1-5


@community_router.post("/community/notes")
async def post_community_note(req: CommunityNoteRequest):
    """Un DJ deja una nota publica en un track (visible para todos)."""
    if not req.note_text.strip():
        raise HTTPException(400, "note_text vacio")
    if len(req.note_text) > 500:
        raise HTTPException(400, "Nota demasiado larga (max 500 chars)")
    try:
        note_id = db.save_community_note(
            fingerprint=req.fingerprint,
            device_id=req.device_id,
            note_text=req.note_text.strip(),
            display_name=req.display_name.strip()[:30] or "DJ",
            note_type=req.note_type,
        )
        logger.info(f"[Community] Nota guardada: fp={req.fingerprint[:8]}... by {req.display_name}")
        return {"status": "ok", "note_id": note_id}
    except Exception as e:
        logger.error(f"[Community] Error guardando nota: {e}")
        raise HTTPException(500, str(e))


@community_router.get("/community/notes/{fingerprint}")
async def get_community_notes(fingerprint: str):
    """Devuelve todas las notas de la comunidad para un track."""
    try:
        notes = db.get_community_notes(fingerprint)
        return {"fingerprint": fingerprint, "notes": notes, "total": len(notes)}
    except Exception as e:
        logger.error(f"[Community] Error leyendo notas: {e}")
        return {"fingerprint": fingerprint, "notes": [], "total": 0}


@community_router.post("/community/notes/{note_id}/upvote")
async def upvote_note(note_id: int, device_id: str = "", http: Request = None):
    """Sube un voto a una nota comunitaria. Uno por votante.

    `device_id` es opcional para no romper al cliente actual, que llama sin
    cuerpo. Cuando no viene, la identidad es el hash de la IP: peor grano (un
    hogar entero cuenta como uno) pero acota el bucle de curl, que era lo que
    dejaba el contador a merced de cualquiera.
    """
    voter = (device_id or "").strip()
    if not voter and http is not None:
        try:
            from validation import get_client_ip

            voter = db.hash_ip(get_client_ip(http))
        except Exception:  # noqa: BLE001 - sin identidad, el INSERT cae en 'anon'
            voter = ""
    try:
        result = db.upvote_community_note(note_id, voter)
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"[Community] Error en upvote de nota {note_id}: {e}")
        raise HTTPException(500, "Error interno")


@community_router.post("/community/rate")
async def rate_track_endpoint(req: TrackRatingRequest, http: Request = None):
    """Un DJ puntua un track (1-5 estrellas). Una valoracion por DJ por track.
    rating=0 QUITA la valoracion del DJ (toggle off desde la UI).

    El guard SEC-12 aplica aqui igual que en los overrides: la media de rating
    es un dato agregado que se muestra a todo el mundo, y con ids inventados
    desde una sola IP se empuja donde se quiera.
    """
    if req.rating < 0 or req.rating > 5:
        raise HTTPException(400, "Rating debe ser 0-5 (0 = quitar)")
    if http is not None:
        _guard_vote_source(http, req.fingerprint, req.device_id)
    try:
        result = db.rate_track(req.fingerprint, req.device_id, req.rating)
        logger.info(f"[Community] Rating: fp={req.fingerprint[:8]}... = {req.rating} (avg {result.get('avg_rating')})")
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"[Community] Error rating: {e}")
        raise HTTPException(500, str(e))


@community_router.get("/community/popularity/{fingerprint}")
async def get_popularity(fingerprint: str, device_id: str = ""):
    """Devuelve popularidad + rating medio + tu rating para un track."""
    try:
        pop = db.get_track_popularity(fingerprint)
        my_rating = db.get_my_rating(fingerprint, device_id) if device_id else 0
        return {**pop, "my_rating": my_rating}
    except Exception:
        return {"analysis_count": 0, "dj_count": 0, "avg_rating": 0, "total_ratings": 0, "my_rating": 0}


class PopularityBatchRequest(BaseModel):
    fingerprints: List[str]


@community_router.post("/community/popularity/batch")
async def get_popularity_batch(req: PopularityBatchRequest):
    """Popularidad de varios tracks en UNA llamada. Lo usa la columna de
    popularidad de la libreria desktop (evita N peticiones por-track).
    Devuelve {fingerprint: {analysis_count, dj_count, avg_rating, total_ratings}}.
    Los fingerprints sin datos no aparecen (el cliente asume 0)."""
    try:
        return db.get_track_popularity_batch(req.fingerprints)
    except Exception as e:
        logger.error(f"[Community] Error popularity batch: {e}")
        return {}


class MyRatingsBatchRequest(BaseModel):
    fingerprints: List[str]
    device_id: str


@community_router.post("/community/my-ratings/batch")
async def get_my_ratings_batch(req: MyRatingsBatchRequest):
    """Rating PROPIO (de este device_id) de varios tracks en UNA llamada. Lo usa
    la columna de rating personal de la libreria desktop. Devuelve {fingerprint:
    rating}; los no valorados por este device no aparecen (cliente asume 0)."""
    try:
        return db.get_my_ratings_batch(req.fingerprints, req.device_id)
    except Exception as e:
        logger.error(f"[Community] Error my-ratings batch: {e}")
        return {}


class LibraryUpdatesRequest(BaseModel):
    fingerprints: List[str]
    since: Optional[str] = None       # ISO-8601; None = sin filtro temporal
    device_id: Optional[str] = None   # el propio: se EXCLUYE del conteo


@community_router.post("/community/library-updates")
async def get_library_updates(req: LibraryUpdatesRequest):
    """MEMORIA COLECTIVA — que gano MI biblioteca desde `since`.

    Gancho de retencion (FASE 4 del RETENTION_PLAN): el cliente manda los
    fingerprints de su biblioteca y recibe cuantos de SUS tracks recibieron
    aportaciones de OTROS DJs (cues, beat-grid, correcciones, notas, ratings)
    desde la ultima vez. Es el motivo para volver a abrir la app y el
    diferenciador frente a Rekordbox.

    El propio `device_id` se excluye: contar las aportaciones del usuario como
    "mejoras de la comunidad" seria mentira y quemaria el mensaje.

    Devuelve {tracks_updated, by_type, examples}. Best-effort: {} si falla."""
    try:
        return db.get_community_updates_for_library(
            req.fingerprints,
            since=req.since,
            exclude_device_id=req.device_id,
        )
    except Exception as e:
        logger.error(f"[Community] Error library-updates: {e}")
        return {}
