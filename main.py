"""
DJ Analyzer Pro API v2.3.0
==========================
Backend principal - Importa funcionalidad de mdulos separados

Estructura:
- main.py (este archivo) - FastAPI app, endpoints principales
- database.py - Gestin de SQLite
- genre_detection.py - Deteccin de g(c)nero con mltiples fuentes
- artwork_and_cuepoints.py - Extraccin artwork + cue points
- similar_tracks_endpoint.py - Bsqueda de tracks similares
- essentia_analyzer.py - Anlisis con Essentia (opcional)
- api_config.py / config.py - Configuracin
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sync_endpoints import sync_router
from routes.admin_panel import admin_panel_router
from routes.search import search_router, init as init_search
from routes.community import community_router, init as init_community
from routes.preview import preview_router, init as init_preview
from routes.analysis_artwork import router as lookup_router, init as init_lookup
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
import librosa
import numpy as np
import sys
import tempfile
import os
import re
import json
import hashlib
import hmac
import requests
import shutil
import sqlite3
import time
import asyncio
from typing import Any, Dict, List, Optional

# Logger global del modulo, definido temprano para que las llamadas
# logger.info/warning/error en el codigo de inicializacion no fallen
# con NameError. Antes habia prints aqui; B-L1 los migro a logger.
import logging
logger = logging.getLogger('dj_analyzer')
logger.setLevel(logging.INFO)
from pydantic import BaseModel
from audio_helpers import silence_native_stderr
from spectral_genre_classifier import classify_genre_advanced
from config import (
    AUDD_API_TOKEN,
    AUDD_AUTO_ENABLED,
    AUDD_DAILY_CAP,
    AUDD_COOLDOWN_DAYS,
    AUDD_MIN_DURATION,
    AUDD_MAX_DURATION,
    DISCOGS_TOKEN,
    print_config,
    BASE_URL,
    CORS_ORIGINS,
    DEBUG,
    PREVIEWS_DIR,
    ADMIN_TOKEN,
    RATE_LIMIT_ENABLED,
    DATABASE_PATH,
    ARTWORK_CACHE_DIR as _CONFIG_ARTWORK_CACHE_DIR,
)

#  Importar mdulo de validacin
from validation import (
    validate_audio_file,
    validate_bpm_range,
    validate_energy_range,
    validate_limit,
    sanitize_string,
    validate_key,
    validate_camelot,
    validate_track_type,
    validate_genre,
    validate_track_id,
    check_rate_limit,
    get_client_ip,
    ValidationError,
)

if __name__ == "__main__":
    print_config()  # Muestra configuracin al arrancar

try:
    from chunked_analyzer import ChunkedAudioAnalyzer, get_chunked_analyzer
    CHUNKED_ANALYZER_ENABLED = True
    logger.info(" ChunkedAudioAnalyzer disponible para tracks largos")
except ImportError as e:
    CHUNKED_ANALYZER_ENABLED = False
    logger.info(f" ChunkedAudioAnalyzer no disponible: {e}")

# ==================== MODO LOCAL ====================
# Cuando corre via local_engine.py, tiene toda la CPU del usuario.
# No necesita chunked, puede cargar tracks enteros en RAM,
# y usar análisis más intensivos.
IS_LOCAL_ENGINE = os.environ.get('LOCAL_ENGINE', 'false').lower() == 'true'

# Versión del motor de análisis. Incrementar cuando se mejore el algoritmo
# (BPM, key, energy, etc.) para invalidar la cache y forzar re-análisis.
# Convención: NULL en BD == "1" (registros pre-versionado no se reanalizan).
ANALYSIS_VERSION = "1"


def _is_analysis_current(track: dict) -> bool:
    """True si el análisis guardado coincide con ANALYSIS_VERSION actual."""
    return (track.get('analysis_version') or '1') == ANALYSIS_VERSION


if IS_LOCAL_ENGINE:
    CHUNKED_ANALYZER_ENABLED = False  # No necesario: RAM del usuario suficiente
    logger.info("=" * 50)
    logger.info("  MODO LOCAL: Análisis completo activado")
    logger.info("  - Chunked deshabilitado (RAM suficiente)")
    logger.info("  - Track completo en memoria (sr=44100)")
    logger.info("  - Key: Krumhansl-Kessler + multi-pasada")
    logger.info("  - BPM: Smart correction + double/half")
    logger.info("  - Energy: Power curve optimizada")
    logger.info("=" * 50)

# Umbral de duracin para usar anlisis por chunks (en segundos)
# Tracks > 4 minutos usarn el analizador por chunks
# SOLO aplica en Render (IS_LOCAL_ENGINE=false)
CHUNK_ANALYSIS_THRESHOLD = 240  # 4 minutos


# ==================== AUTO-UPLOAD A RENDER (modo local) ====================

RENDER_BACKEND_URL = os.environ.get('RENDER_BACKEND_URL', 'https://dj-analyzer-api.onrender.com')

# ==================== RANKING DE FUENTES (item 8: "mejor gana") ====================
# Logica de prioridad de bpm_source extraida a analysis_ranking.py para
# que sea testeable sin importar la app entera (librosa pesado).
from analysis_ranking import (
    ANALYSIS_SOURCE_PRIORITY,
    get_source_priority,
    should_overwrite_analysis,
)


# ==================== CONCURRENCIA DE ANALISIS ====================
# /analyze corre en UN worker uvicorn (ver Procfile) y analyze_audio (librosa)
# es CPU-bound y SINCRONO. Llamarlo directo en el event loop lo BLOQUEA durante
# segundos -> cualquier peticion ligera concurrente (/admin/stats, /sync/*,
# /artwork) se queda en cola y da timeout en el cliente cuando hay usuarios
# analizando. Solucion: ejecutar el analisis en un threadpool
# (run_in_threadpool) para liberar el event loop, ACOTADO por un semaforo para
# no reventar la RAM de Render con varios arrays de audio a la vez.
#
# =1 mantiene el MISMO perfil CPU/RAM que el modelo bloqueante actual (un solo
# analisis a la vez) pero deja de bloquear el loop. Subir SOLO si hay RAM de
# sobra (cada analisis carga el track entero en memoria a sr=44100).
_ANALYSIS_CONCURRENCY = max(1, int(os.environ.get('ANALYSIS_CONCURRENCY', '1')))
_analysis_semaphore: Optional[asyncio.Semaphore] = None


def _get_analysis_semaphore() -> asyncio.Semaphore:
    """Crea el semaforo de forma perezosa DENTRO del event loop en marcha.

    Crearlo a nivel de modulo (import time) lo bindea a un loop que en Python
    3.9 puede no ser el de uvicorn (DeprecationWarning / cross-loop). Crearlo
    en la primera llamada a /analyze garantiza el loop correcto en 3.9 y 3.10+.
    """
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(_ANALYSIS_CONCURRENCY)
    return _analysis_semaphore


def _upload_to_render_cache(track_data: dict):
    """
    Sube resultado de análisis local a Render como cache comunitario.
    Fire & forget: no bloquea si falla.
    """
    import threading
    def _do_upload():
        try:
            payload = {
                'fingerprint': track_data.get('fingerprint', track_data.get('id', '')),
                'filename': track_data.get('filename', ''),
                'artist': track_data.get('artist', ''),
                'title': track_data.get('title', ''),
                'album': track_data.get('album', ''),
                'label': track_data.get('label', ''),
                'duration': track_data.get('duration', 0),
                'bpm': track_data.get('bpm', 0),
                'key': track_data.get('key'),
                'camelot': track_data.get('camelot'),
                'energy_dj': track_data.get('energy_dj', 5),
                'genre': track_data.get('genre', ''),
                'track_type': track_data.get('track_type', ''),
                'bpm_source': track_data.get('bpm_source', 'local_engine'),
                'key_source': track_data.get('key_source', 'local_engine'),
                'genre_source': track_data.get('genre_source', 'local_engine'),
                # Propagar el origen real para que Render no lo marque NULL.
                'engine_source': track_data.get('engine_source') or 'local_engine',
                # Sellar la version: sin esto Render guardaba analysis_version
                # NULL -> _is_analysis_current trataba la fila como stale -> el
                # fallback por fingerprint re-analizaba aunque el dato fuera bueno.
                'analysis_version': track_data.get('analysis_version'),
                # Detalle COMPLETO del analisis. Antes mandabamos
                # track_data.get('analysis_json', {}) que era SIEMPRE {} porque
                # AnalysisResult no tiene campo 'analysis_json' -> se perdian
                # bpm_confidence/key_confidence/structure_sections/cue_points/
                # beat-grid/has_*, y la fila en Render quedaba "fina" (disparando
                # "[Cache] corrupto" + re-analisis al re-subir el track por filename).
                'analysis_json': {k: v for k, v in track_data.items() if k != 'analysis_json'},
            }
            resp = requests.post(
                f"{RENDER_BACKEND_URL}/cache-analysis",
                json=payload,
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"  [Cache→Render] ✓ Subido: {track_data.get('artist', '?')} - {track_data.get('title', '?')}")
            else:
                logger.error(f"  [Cache→Render] Error {resp.status_code}: {resp.text[:100]}")
        except requests.Timeout:
            logger.info(f"  [Cache→Render] Timeout (Render dormido?)")
        except Exception as e:
            logger.error(f"  [Cache→Render] Error: {e}")

    threading.Thread(target=_do_upload, daemon=True).start()


def _fetch_render_cache(fingerprint: str) -> Optional[dict]:
    """
    Consulta Render para el análisis cacheado de un fingerprint. Solo se
    usa cuando el motor local NO tiene el track en su BD local, como
    fallback antes de invocar librosa.

    El endpoint `/analysis/by-fingerprint/{fp}` en Render solo hace SELECT
    en la BD y devuelve JSON: cero CPU/memoria de análisis, es una lectura
    barata. Acelera escenarios como Mac nuevo / reanalisis post-wipe del
    HDD donde Render ya tiene los análisis del PC.

    Devuelve el dict del análisis si Render lo tiene y es válido
    (bpm > 0, key no vacío, no marcado como `analysis_status='failed'`).
    None si Render no lo tiene, está dormido (timeout), o devuelve datos
    basura que no merecen reusarse.
    """
    if not fingerprint or len(fingerprint) < 16:
        return None
    try:
        resp = requests.get(
            f"{RENDER_BACKEND_URL}/analysis/by-fingerprint/{fingerprint}",
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.Timeout, requests.RequestException, ValueError):
        return None

    # Validar que NO es un fallback "failed" (bpm=0, key=null,
    # analysis_status='failed'). Si lo es, mejor analizar local
    # de cero — el motor local ya tiene librosa OK aquí.
    try:
        bpm_val = float(data.get('bpm') or 0)
    except (TypeError, ValueError):
        bpm_val = 0
    key_val = (data.get('key') or '').strip() if data.get('key') else ''
    if bpm_val <= 0 or not key_val:
        return None
    if data.get('analysis_status') == 'failed':
        return None
    # Si Render tiene una versión antigua del análisis, mejor re-analizar local
    if (data.get('analysis_version') or '1') != ANALYSIS_VERSION:
        logger.info(
            f"[Render fallback] version obsoleta "
            f"({data.get('analysis_version') or 'NULL'} != {ANALYSIS_VERSION}), "
            f"re-analizando local"
        )
        return None
    return data


# ==================== IMPORTS LOCALES ====================
from database import AnalysisDB


# Importar funciones de artwork y cue points
try:
    from artwork_and_cuepoints import (
        extract_artwork_from_file,
        extract_id3_metadata,
        detect_cue_points,
        detect_beat_grid,
        save_artwork_to_cache,
        search_artwork_online,
        ARTWORK_CACHE_DIR
    )
    ARTWORK_ENABLED = True
except ImportError:
    logger.info("artwork_and_cuepoints.py no encontrado - funciones deshabilitadas")
    ARTWORK_ENABLED = False
    ARTWORK_CACHE_DIR = _CONFIG_ARTWORK_CACHE_DIR
    search_artwork_online = None

# Importar clasificador de géneros
try:
    from genre_detection import GenreDetector
    from api_config import DISCOGS_TOKEN
    genre_detector = GenreDetector(discogs_token=DISCOGS_TOKEN)
    GENRE_DETECTOR_ENABLED = True
    logger.info(f"GenreDetector inicializado (Discogs: {'S' if DISCOGS_TOKEN else 'No'})")
except ImportError as e:
    logger.info(f"genre_detection.py no encontrado: {e}")
    GENRE_DETECTOR_ENABLED = False
    genre_detector = None

# Importar similar tracks
try:
    from similar_tracks_endpoint import (
        SimilarTracksRequest,
        SimilarTrackResult,
        calculate_compatibility_score,
        is_key_compatible,
        CAMELOT_COMPATIBLE
    )
    SIMILAR_TRACKS_ENABLED = True
except ImportError:
    logger.info("similar_tracks_endpoint.py no encontrado")
    SIMILAR_TRACKS_ENABLED = False


# ==================== FLOAT SANITIZER ====================
import math

def sanitize_float(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return v

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        return sanitize_float(obj)
    return obj

class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        content = sanitize_for_json(content)
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def try_bpm_double_half(y, sr, original_bpm: float, bpm_confidence: float, onset_env=None) -> float:
    """
    Si la confianza del BPM es baja, probar con doble y mitad.
    
    Logica: si librosa dice 131 con confianza 0.4, probar 262 y 65.5.
    Si alguno de esos tiene sentido musical (60-200 BPM range) Y tiene
    mejor alineacion con los beats, usarlo.
    
    Args:
        onset_env: Si ya se calculó onset_strength, pasarlo para no duplicar CPU.
    """
    if bpm_confidence >= 0.7:
        return original_bpm  # Alta confianza, no tocar
    
    candidates = [original_bpm]
    
    # Probar doble
    double = original_bpm * 2
    if 60 <= double <= 200:
        candidates.append(double)
    
    # Probar mitad
    half = original_bpm / 2
    if 60 <= half <= 200:
        candidates.append(half)
    
    if len(candidates) == 1:
        return original_bpm
    
    # Evaluar cual se alinea mejor con onset strength
    try:
        if onset_env is None:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        best_bpm = original_bpm
        best_score = 0
        
        for candidate in candidates:
            # Crear pulso teorico para este BPM
            beat_interval = 60.0 / candidate
            sr_onset = sr / 512  # hop_length default
            
            # Autocorrelacion con el BPM candidato
            period = int(round(sr_onset * beat_interval))
            if period > 0 and period < len(onset_env) // 2:
                corr = np.correlate(onset_env[:len(onset_env)//2], 
                                     onset_env[period:period + len(onset_env)//2])
                score = float(np.max(corr)) if len(corr) > 0 else 0
                if score > best_score:
                    best_score = score
                    best_bpm = candidate
        
        if best_bpm != original_bpm:
            logger.info(f"   BPM auto-corregido: {original_bpm:.1f} -> {best_bpm:.1f} (confianza baja: {bpm_confidence:.2f})")
        
        return best_bpm
    except Exception:
        return original_bpm

# ==================== APP ====================

app = FastAPI(title="DJ Analyzer Pro API", version="2.9.5", default_response_class=SafeJSONResponse)
_startup_time = time.time()
app.include_router(sync_router)
app.include_router(admin_panel_router)

# CORS: en DEBUG permitimos los origins tipicos de dev local (Flutter web,
# desktop con file://, localhost). "*" + allow_credentials es invalido por
# RFC 6750 — los browsers modernos lo rechazan silencio. Ver AUDIT 2026-04-20 B-M5.
_DEBUG_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if not DEBUG else _DEBUG_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-Signature", "X-Device-Id", "X-Original-Path", "X-Admin-Secret"],
)

#  Manejador de errores de validacin
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "detail": exc.detail,
            "field": getattr(exc, 'field', None)
        }
    )


# ==================== TELEMETRIA: ERRORES NO MANEJADOS ====================
#
# Antes solo /analyze y /identify capturaban errores en analysis_errors
# (try/except manual). Cualquier endpoint que peteara sin try/except se
# perdia silenciosamente en logs. Este middleware global registra TODO
# 500 no manejado en la misma tabla, marcado con endpoint=request.url.path
# para que el panel admin pueda filtrar por ruta y diagnosticar puntos
# calientes. Las HTTPException intencionales (4xx) NO se loguean — solo
# fallos genuinos. Privacy-first: no se guarda body ni query params.
@app.middleware("http")
async def telemetry_unhandled_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        # 4xx intencionales: dejar pasar sin logear como error
        raise
    except ClientDisconnect:
        # El cliente corto la conexion a mitad de peticion (p.ej. cerro la app
        # durante /sync). No es un fallo del servidor: no lo registramos, que
        # ensuciaba el panel como un falso 500.
        raise
    except Exception as exc:
        # "No response returned" lo lanza BaseHTTPMiddleware cuando el cliente
        # se desconecta a mitad de la peticion (p.ej. cierra la app durante un
        # /analyze largo). No es un fallo del servidor: lo dejamos pasar sin
        # registrarlo (ensuciaba el panel como un falso 500).
        if isinstance(exc, RuntimeError) and "No response returned" in str(exc):
            raise
        try:
            import traceback as _tb
            db.log_analysis_error(
                device_id=request.headers.get('X-Device-Id'),
                filename=None,
                fingerprint=None,
                error_class=type(exc).__name__,
                error_msg=str(exc),
                traceback_str=_tb.format_exc(),
                endpoint=f"unhandled:{request.url.path}",
            )
        except Exception as log_err:
            logger.error(f"[Telemetry] No se pudo loguear error no manejado: {log_err}")
        # Re-elevar para que FastAPI devuelva 500 al cliente
        raise


# ==================== BLOQUEO DE IPs ABUSIVAS ====================
#
# Blacklist configurable via env var BLOCKED_IPS (lista separada por
# comas, ej. "143.59.170.73,1.2.3.4"). Bots/scrapers de datacenter que
# martillean la API acaparan logs, queman CPU de Render y cuota de AudD.
# Bloquearlos aqui (antes de cualquier procesamiento) corta el gasto sin
# necesidad de redeploy: basta editar la env var en el dashboard de
# Render y reiniciar.
#
# La IP real del cliente viene en X-Forwarded-For (Render esta detras de
# un proxy; request.client.host daria la IP interna del proxy, inutil
# para filtrar). El primer valor del header es el cliente original:
# "client, proxy1, proxy2".
_BLOCKED_IPS = {
    ip.strip()
    for ip in os.environ.get('BLOCKED_IPS', '').split(',')
    if ip.strip()
}
if _BLOCKED_IPS:
    logger.info(f"[Security] {len(_BLOCKED_IPS)} IP(s) en blacklist")


def _client_ip(request: Request) -> str:
    xff = request.headers.get('x-forwarded-for')
    if xff:
        return xff.split(',')[0].strip()
    return request.client.host if request.client else ''


@app.middleware("http")
async def block_abusive_ips(request: Request, call_next):
    if _BLOCKED_IPS:
        ip = _client_ip(request)
        if ip in _BLOCKED_IPS:
            # 403 minimo, sin body util — no damos pistas al bot. No se
            # loguea como error (es trafico esperado-rechazado, no un bug).
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    try:
        return await call_next(request)
    except ClientDisconnect:
        return Response(status_code=499)
    except RuntimeError as exc:
        if "No response returned" in str(exc):
            return Response(status_code=499)
        raise


# ==================== TELEMETRIA: ERRORES DEL CLIENTE ====================
#
# Endpoint para que Flutter reporte sus propios errores (network, parsing,
# Chromaprint, sync). Hasta hoy todo se quedaba en debugPrint local: si en
# produccion peta el sync o falla fpcalc, no nos enteramos. Privacy-first:
# nada de filename real (se hashea), nada de paths sin sanitizar (el cliente
# debe truncar/sanear antes de mandar).
class ClientErrorPayload(BaseModel):
    error_class: str
    error_msg: Optional[str] = None
    stack: Optional[str] = None
    context: str  # "chromaprint", "sync", "analysis_api", "uncaught", ...
    platform: Optional[str] = None  # "windows" | "macos" | "android" | "ios" | "linux"
    app_version: Optional[str] = None
    fingerprint: Optional[str] = None
    filename: Optional[str] = None  # se hashea server-side, NUNCA se persiste literal


@app.post("/client-error")
async def report_client_error(payload: ClientErrorPayload, request: Request):
    """Recibe un error del cliente y lo persiste en analysis_errors.

    El cliente envia error_class + msg + stack truncados y un context que
    identifica el modulo (ej: "chromaprint", "sync"). El endpoint en BD se
    formatea como "client:{context}" para que el panel pueda filtrar.

    Fire-and-forget desde el cliente: la respuesta es siempre 202 si la
    forma es valida, aun si el INSERT falla (no queremos que un fallo de
    telemetria desencadene otro error en el cliente).
    """
    error_class = (payload.error_class or 'Unknown')[:120]
    # Tamanos por encima del limite del schema: el cliente deberia truncar,
    # pero por defensa server-side reaplicamos.
    error_msg = (payload.error_msg or '')[:500]
    stack = (payload.stack or '')[:2000] if payload.stack else None
    context = (payload.context or 'unknown')[:60]
    platform = (payload.platform or '')[:20] or None
    app_version = (payload.app_version or '')[:20] or None

    # Tag platform+version dentro de error_msg para que el panel pueda
    # segmentar sin necesidad de columnas nuevas en el schema. Formato
    # estable, parseable: "[ios 2.8.0] <msg original>".
    tag_parts = []
    if platform:
        tag_parts.append(platform)
    if app_version:
        tag_parts.append(app_version)
    tagged_msg = f"[{' '.join(tag_parts)}] {error_msg}" if tag_parts else error_msg

    try:
        db.log_analysis_error(
            device_id=request.headers.get('X-Device-Id'),
            filename=payload.filename,
            fingerprint=payload.fingerprint,
            error_class=error_class,
            error_msg=tagged_msg,
            traceback_str=stack,
            endpoint=f"client:{context}",
        )
    except Exception as e:
        logger.warning(f"[ClientError] log fallo: {e}")
        # No reelevamos: el cliente no debe sufrir por nuestra telemetria
        return JSONResponse(status_code=202, content={"ok": False, "logged": False})

    return JSONResponse(status_code=202, content={"ok": True, "logged": True})


# Inicializar BD con path de config (no hardcoded)
db = AnalysisDB(db_path=DATABASE_PATH)

# Registrar endpoints de community cues (/community-cues, /user-cues).
# El modulo community_cues_endpoint.py expone register_community_endpoints
# pero NUNCA se cableaba aqui -> el cliente recibia 404 al subir/leer cues
# de comunidad. La tabla `community_cues` ya se crea en database.py:init_db().
from community_cues_endpoint import register_community_endpoints
register_community_endpoints(app, db)

# Endpoints de busqueda/biblioteca/track (movidos de inline a routes/search.py,
# paso 2 del troceo de main.py). init() inyecta la BD + el mapa Camelot ANTES
# del include_router. CAMELOT_COMPATIBLE solo existe si similar_tracks cargo;
# si no, inyectamos {} (los endpoints siguen vivos, solo /search/compatible
# devuelve la propia key como unica compatible — mismo fallback que el .get()).
init_search(db, CAMELOT_COMPATIBLE if SIMILAR_TRACKS_ENABLED else {})
app.include_router(search_router)

# Endpoints de comunidad (beat-grid, overrides, notes, ratings, popularity),
# movidos de inline a routes/community.py (paso 3 del troceo de main.py).
init_community(db)
app.include_router(community_router)

# Endpoints de preview/artwork batch, movidos de inline a routes/preview.py
# (paso 4). Se inyectan los paths EXACTOS que main resolvio (PREVIEWS_DIR de
# config, ARTWORK_CACHE_DIR de artwork_and_cuepoints o fallback de config).
init_preview(PREVIEWS_DIR, ARTWORK_CACHE_DIR)
app.include_router(preview_router)

# Endpoints de lookup de cache (check-analyzed, analysis/*) + artwork serving,
# movidos de inline a routes/analysis_artwork.py (paso 5). _is_analysis_current
# se inyecta (lo comparte con /analyze). search/save_artwork van con guard
# ARTWORK_ENABLED -> None si el modulo de artwork no cargo (mismo fallback que
# main; save_* solo se invoca tras chequear search_artwork_online truthy).
init_lookup(
    db,
    _is_analysis_current,
    ARTWORK_CACHE_DIR,
    search_online=search_artwork_online,
    save_to_cache=(save_artwork_to_cache if ARTWORK_ENABLED else None),
)
app.include_router(lookup_router)

# Crear directorio de cach(c) para artwork
os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)

# ==================== BUSQUEDA EN BD COLECTIVA ====================

def search_collective_db(artist: str, title: str) -> Optional[Dict]:
    """
    Busca BPM y Key en nuestra BD colectiva.
    Si otro usuario ya analiz este track con (c)xito, usamos esos datos.
    """
    try:
        # Normalizar para bsqueda
        clean_artist = artist.lower().strip()
        clean_title = re.sub(r'\s*\(?(Original Mix|Extended Mix|Radio Edit|Remix)\)?', '', title, flags=re.IGNORECASE).lower().strip()
        
        # Buscar en BD por artista y ttulo similar
        conn = db.conn
        cursor = conn.cursor()
        
        # Bsqueda exacta primero
        # Only query columns that exist in the tracks table schema
        cursor.execute("""
            SELECT bpm, key, camelot, duration, genre, analysis_json
            FROM tracks
            WHERE LOWER(artist) = ? AND LOWER(title) LIKE ?
            AND bpm IS NOT NULL AND bpm > 0
            ORDER BY analyzed_at DESC
            LIMIT 1
        """, (clean_artist, f"%{clean_title}%"))

        row = cursor.fetchone()

        if not row:
            # Bsqueda ms flexible
            cursor.execute("""
                SELECT bpm, key, camelot, duration, genre, analysis_json
                FROM tracks
                WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
                AND bpm IS NOT NULL AND bpm > 0
                ORDER BY analyzed_at DESC
                LIMIT 1
            """, (f"%{clean_artist}%", f"%{clean_title}%"))
            row = cursor.fetchone()

        if row:
            # Extract label, bpm_source, key_source from analysis_json if available
            label = None
            bpm_source = None
            key_source = None
            if row[5]:
                try:
                    aj = json.loads(row[5])
                    label = aj.get('label')
                    bpm_source = aj.get('bpm_source')
                    key_source = aj.get('key_source')
                except (json.JSONDecodeError, TypeError):
                    pass
            result = {
                'bpm': row[0],
                'key': row[1],
                'camelot': row[2],
                'duration': row[3],
                'genre': row[4],
                'label': label,
                'bpm_source': bpm_source,
                'key_source': key_source,
                'source': 'collective_db'
            }
            return result
        
        return None
        
    except Exception as e:
        logger.error(f" Error buscando en BD colectiva: {e}")
        return None


# ==================== MODELOS ====================

class AnalysisResult(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    label: Optional[str] = None
    year: Optional[str] = None
    isrc: Optional[str] = None
    duration: float
    bpm: float
    # Los 18 campos extendidos abajo llevan default. Motivo: tracks viejos
    # cuyo analysis_json en BD se guardo ANTES de que existieran estos campos
    # reventaban en AnalysisResult(**analysis_data) (cache-hit por fingerprint,
    # main.py:~2302) -> spam "[Cache] Error parseando analysis_json: 18
    # validation errors". El analisis real siempre los setea explicito, asi que
    # el default solo aplica al parsear JSON legacy. Defaults alineados con el
    # fallback de reconstruccion desde columnas.
    bpm_confidence: float = 0.0
    bpm_source: str = "analysis"
    key: Optional[str] = None
    camelot: Optional[str] = None
    key_confidence: float = 0.0
    key_source: str = "analysis"
    energy_raw: float = 0.0
    energy_normalized: float = 0.0
    energy_dj: int = 5
    groove_score: float = 0.0
    swing_factor: float = 0.0
    has_intro: bool = False
    has_buildup: bool = False
    has_drop: bool = False
    has_breakdown: bool = False
    has_outro: bool = False
    structure_sections: List[Dict] = []
    track_type: str = "peak"
    track_type_source: str = "waveform"
    # Fase 1 Track Type v2: confianza algoritmica (0..1) + top-3 alternativas.
    # Permite a la UI mostrar honestidad sobre tracks ambiguos. Cuando
    # `track_type_source == 'beatport'`, confidence=1.0 (señal externa).
    # Cuando viene de la heuristica spectral, confidence depende del margin
    # entre el winner y el segundo. Ver classify_track_type.
    track_type_confidence: float = 0.5
    track_type_alternatives: List[Dict[str, Any]] = []
    genre: str = "unknown"
    subgenre: Optional[str] = None
    genre_source: str = "spectral_analysis"
    has_vocals: bool = False
    has_heavy_bass: bool = False
    has_pads: bool = False
    percussion_density: float = 0.0
    mix_energy_start: float = 0.0
    mix_energy_end: float = 0.0
    drop_timestamp: float = 0.0
    #  Cue Points
    cue_points: List[Dict] = []
    #  Beat Grid
    first_beat: float = 0.0
    beat_interval: float = 0.5
    #  Artwork
    artwork_embedded: bool = False
    artwork_url: Optional[str] = None
    # Preview snippet URL (6 segundos)
    preview_url: Optional[str] = None
    # Fingerprint del archivo
    fingerprint: Optional[str] = None
    # Señal honesta para la UI (solo en /analyze?force_audd=true, p.ej.
    # "Limpiar con AudD"): True cuando el usuario forzó AudD pero el cap diario
    # GLOBAL ya estaba agotado, así que AudD NO corrió para este track. La UI
    # muestra "límite diario alcanzado, inténtalo mañana" en vez de un genérico
    # "no identificado". Default False => parsear analysis_json legacy sin el
    # campo no rompe (mismo criterio que los 18 campos extendidos de arriba).
    audd_daily_cap_reached: bool = False

class CorrectionRequest(BaseModel):
    track_id: str
    field: str
    old_value: str
    new_value: str
    fingerprint: Optional[str] = None
    device_id: Optional[str] = None

# SearchRequest se movio a routes/search.py (lo usa search_advanced, ya no inline).

# ==================== MAPEOS ====================

KEY_TO_CAMELOT = {
    'C': '8B', 'C#': '3B', 'D': '10B', 'D#': '5B',
    'E': '12B', 'F': '7B', 'F#': '2B', 'G': '9B',
    'G#': '4B', 'A': '11B', 'A#': '6B', 'B': '1B',
    'Cm': '5A', 'C#m': '12A', 'Dm': '7A', 'D#m': '2A',
    'Em': '9A', 'Fm': '4A', 'F#m': '11A', 'Gm': '6A',
    'G#m': '1A', 'Am': '8A', 'A#m': '3A', 'Bm': '10A'
}

def get_camelot(key: str) -> str:
    """Convierte key musical a notacion Camelot"""
    return KEY_TO_CAMELOT.get(key, '?')


CAMELOT_TO_KEY = {v: k for k, v in KEY_TO_CAMELOT.items()}


def camelot_to_key(camelot: str) -> str:
    """Convierte notacion Camelot (1A-12B) a nota cruda (C, Cm, F#, etc.).

    Raises ValueError si el input es invalido (incluye None / no-str).
    """
    if not isinstance(camelot, str):
        raise ValueError(f"Invalid Camelot notation: {camelot!r}")
    norm = camelot.strip().upper()
    if norm not in CAMELOT_TO_KEY:
        raise ValueError(f"Invalid Camelot notation: {camelot}")
    return CAMELOT_TO_KEY[norm]


# ==================== HELPERS ====================

def calculate_fingerprint(file_path):
    """Calcular fingerprint simple del archivo"""
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def parse_filename(filename: str) -> dict:
    name = re.sub(r'\.(mp3|wav|flac|m4a)$', '', filename, flags=re.IGNORECASE)
    name = re.sub(r'^\d+[\s\-_.]+', '', name)
    name = re.sub(r'[\[\(][A-Z0-9]+[\]\)]', '', name)
    name = re.sub(r'(?i)\(original mix\)|\[original mix\]', '', name)
    name = re.sub(r'(?i)\(extended\)|\[extended\]', '', name)
    
    artist = None
    title = None
    
    if ' - ' in name:
        parts = name.split(' - ', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    elif '-' in name and len(name.split('-')) == 2:
        parts = name.split('-', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    else:
        title = name.strip()
    
    if artist:
        artist = ' '.join(artist.split())
    if title:
        title = ' '.join(title.split())
    
    return {'artist': artist, 'title': title if title else name.strip()}

def detect_structure(y, sr, duration):
    segment_length = int(sr * 8)
    n_segments = len(y) // segment_length
    
    if n_segments == 0:
        return {
            'has_intro': False, 'has_buildup': False, 'has_drop': False,
            'has_breakdown': False, 'has_outro': False, 'sections': []
        }
    
    energies = []
    for i in range(n_segments):
        segment = y[i*segment_length:(i+1)*segment_length]
        energies.append(np.mean(librosa.feature.rms(y=segment)))
    
    energies = np.array(energies)
    avg_energy = np.mean(energies)
    
    has_intro = energies[0] < avg_energy * 0.6 if len(energies) > 0 else False
    has_outro = energies[-1] < avg_energy * 0.6 if len(energies) > 0 else False
    has_drop = np.max(energies) > avg_energy * 1.4 if len(energies) > 0 else False
    has_breakdown = np.min(energies[1:-1]) < avg_energy * 0.5 if len(energies) > 2 else False
    has_buildup = has_drop
    
    sections = []
    for i, e in enumerate(energies):
        section_type = "body"
        if i == 0 and has_intro:
            section_type = "intro"
        elif i == len(energies)-1 and has_outro:
            section_type = "outro"
        elif e > avg_energy * 1.4:
            section_type = "drop"
        elif e < avg_energy * 0.5:
            section_type = "breakdown"
        
        sections.append({
            "type": section_type,
            "start": i * 8.0,
            "end": (i + 1) * 8.0,
            "energy": float(e)
        })
    
    return {
        'has_intro': has_intro, 'has_buildup': has_buildup, 'has_drop': has_drop,
        'has_breakdown': has_breakdown, 'has_outro': has_outro, 'sections': sections
    }

def find_drop_timestamp(y, sr, segments: dict) -> float:
    if not segments['sections']:
        return 30.0
    
    for section in segments['sections']:
        if section['type'] == 'drop':
            return section['start'] + 4.0
    
    max_energy = max(s['energy'] for s in segments['sections'])
    for section in segments['sections']:
        if section['energy'] == max_energy:
            return section['start'] + 4.0
    
    duration = segments['sections'][-1]['end'] if segments['sections'] else 180
    return duration / 3

def classify_track_type(energy: float, segments: dict, duration: float) -> dict:
    # Fase 1 Track Type v2: pasamos de cascada de returns simples a scoring
    # con margin top-1 vs top-2 -> confidence (0..1). Permite a la UI mostrar
    # honestidad sobre tracks ambiguos en lugar de mentir con un tipo forzado.
    # Plan completo en Analyzer/PENDING_NEXT_SESSION_TRACKTYPE_V2.md.
    #
    # Mismas señales que la cascada original (has_intro/has_drop/has_outro +
    # energy + duration), pero acumulamos en lugar de decidir inmediato.
    # Ej. Oxia - Domino (energy=0.7, has_outro=True, duration=433): closing
    # acumula 1.0 + 0.3 = 1.3, peak_time solo 0.2 (energy>0.6 soft signal),
    # warmup 0. Margin grande -> confidence ~0.85.
    scores = {'warmup': 0.0, 'peak_time': 0.0, 'closing': 0.0}

    if energy < 0.5 and segments['has_intro']:
        scores['warmup'] += 1.0
    if energy < 0.4 and segments['has_intro']:
        scores['warmup'] += 0.5
    if energy > 0.7 and segments['has_drop']:
        scores['peak_time'] += 1.0
    if energy > 0.8 and segments['has_drop']:
        scores['peak_time'] += 0.5
    if segments['has_outro'] and duration > 300:
        scores['closing'] += 1.0
    if segments['has_outro'] and duration > 420:
        scores['closing'] += 0.3
    # Soft signals para desempates: cualquier track con energia alta
    # tira hacia peak_time, cualquiera con energia baja hacia warmup.
    if energy > 0.6:
        scores['peak_time'] += 0.2
    elif energy < 0.5:
        scores['warmup'] += 0.2

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    winner_type, winner_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    if winner_score == 0.0:
        # Track sin señales claras: caer al fallback de la cascada original
        # (energy>0.6 -> peak_time, sino warmup) y reportar confidence 0
        # para que la UI muestre el badge como "incierto".
        winner_type = 'peak_time' if energy > 0.6 else 'warmup'
        confidence = 0.0
    else:
        margin = winner_score - second_score
        confidence = min(1.0, margin / max(winner_score, 0.5))

    return {
        'type': winner_type,
        'confidence': round(confidence, 2),
        'alternatives': [
            {'type': t, 'score': round(s, 2)} for t, s in sorted_scores
        ],
        'reason': (
            f"energy={energy:.2f} duration={duration:.0f} "
            f"intro={segments['has_intro']} drop={segments['has_drop']} "
            f"outro={segments['has_outro']}"
        ),
        'source': 'waveform',
    }

def detect_vocals_improved(y, sr, spectral_centroid):
    try:
        centroid_mean = float(np.mean(spectral_centroid))
        has_high_centroid = centroid_mean > 3500
        
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = float(np.mean(flatness))
        is_tonal = flatness_mean < 0.15
        
        centroid_std = float(np.std(spectral_centroid))
        has_variation = centroid_std > 500
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        zcr_in_voice_range = 0.05 < zcr_mean < 0.15
        
        criteria_met = sum([has_high_centroid, is_tonal, has_variation, zcr_in_voice_range])
        return criteria_met >= 3
        
    except Exception as e:
        logger.error(f"Error detectando vocals: {e}")
        return False

def get_acousticbrainz_genre(fingerprint=None, artist=None, title=None):
    """AcousticBrainz cerró en 2022. Stub que retorna None para no romper llamadas."""
    return None


# ==================== PREVIEW SNIPPET ====================

import subprocess
import threading
import logging as _logging
from pathlib import Path as PathLib

# Logger dedicado para el módulo de previews. Usa handlers heredados de
# local_engine.py (fichero engine.log junto al binario) cuando corremos
# como engine local detached; en Render van a stdout.
_preview_logger = _logging.getLogger('preview_push')
_preview_logger.setLevel(_logging.INFO)

logger = _logging.getLogger('dj_analyzer')
logger.setLevel(_logging.INFO)

# Asegurar directorio de previews
PathLib(PREVIEWS_DIR).mkdir(parents=True, exist_ok=True)


def _push_preview_to_render(fingerprint: str, local_path: str) -> None:
    """
    Sube el snippet MP3 al Render remoto en cuanto lo genera el engine local.
    Patrón "push at source": evita que el cliente tenga que escanear carpetas
    locales y saltar al sync para que otros dispositivos puedan reproducir.

    Activo solo cuando:
      - LOCAL_ENGINE=true (somos el engine de usuario, no el Render)
      - RENDER_SYNC_URL y SYNC_AUTH_SECRET están configurados (Flutter los
        pasa al lanzar el proceso vía LocalEngineService.start()).

    Silencioso si falla: el siguiente sync manual del cliente lo recuperará
    mediante CloudSyncService._uploadLocalPreviews() (red de seguridad).

    Los logs van vía `_preview_logger` → cuando corremos bajo
    local_engine.py, el handler global escribe en `engine.log` junto al
    binario (Flutter no captura el stdout del engine porque lo lanza
    detached, así que `print` se perdería).
    """
    short_fp = fingerprint[:8] if fingerprint else '--------'
    try:
        if not os.getenv('LOCAL_ENGINE'):
            _preview_logger.debug('[Preview] Push skip: LOCAL_ENGINE no set')
            return
        render_url = (os.getenv('RENDER_SYNC_URL') or '').rstrip('/')
        if not render_url:
            _preview_logger.warning('[Preview] Push skip: RENDER_SYNC_URL vacío')
            return
        if not os.path.exists(local_path):
            _preview_logger.warning(f'[Preview] Push skip: archivo inexistente {local_path}')
            return
        # El endpoint /preview/upload no requiere HMAC (preview_router no
        # lleva dependency de _verify_sync_auth). Se autentica por existencia
        # del fingerprint en la petición — bastante para nuestro uso.
        with open(local_path, 'rb') as fp:
            files = {'file': (f'{fingerprint}.mp3', fp, 'audio/mpeg')}
            resp = requests.post(
                f'{render_url}/preview/upload/{fingerprint}',
                files=files,
                timeout=15,
            )
        if resp.status_code == 200:
            _preview_logger.info(f'[Preview] Push -> Render OK: {short_fp}')
        else:
            body = (resp.text or '')[:200]
            _preview_logger.warning(
                f'[Preview] Push -> Render {resp.status_code}: {short_fp} | {body}'
            )
    except Exception as push_err:
        # Fire-and-forget: no queremos reventar el analyze por esto.
        _preview_logger.error(f'[Preview] Push -> Render error ({short_fp}): {push_err}')


def _push_preview_async(fingerprint: str, local_path: str) -> None:
    """Lanza `_push_preview_to_render` en un thread daemon (no bloquea)."""
    threading.Thread(
        target=_push_preview_to_render,
        args=(fingerprint, local_path),
        daemon=True,
    ).start()


def _push_artwork_to_render(fingerprint: str, local_path: str) -> None:
    """Sube el artwork extraído al Render remoto cuando el engine local
    acaba de cachearlo. Sin esto, los devices que sincronicen desde
    Render verían 404 al pedir `/artwork/{fingerprint}` (las portadas
    se quedaban en el disco local del PC).

    Mismo patrón que `_push_preview_to_render`. Silencioso si falla,
    pero loguea el motivo para diagnosticar (engine.log).
    """
    short_fp = fingerprint[:8] if fingerprint else '--------'
    try:
        if not os.getenv('LOCAL_ENGINE'):
            logger.debug('[Artwork] Push skip: LOCAL_ENGINE no set')
            return
        render_url = (os.getenv('RENDER_SYNC_URL') or '').rstrip('/')
        if not render_url:
            logger.warning('[Artwork] Push skip: RENDER_SYNC_URL vacío')
            return
        if not os.path.exists(local_path):
            logger.warning(f'[Artwork] Push skip: archivo inexistente {local_path}')
            return
        ext = os.path.splitext(local_path)[1].lstrip('.') or 'jpg'
        mime = 'image/jpeg' if ext.lower() in ('jpg', 'jpeg') else 'image/png'
        with open(local_path, 'rb') as fp:
            files = {'file': (f'{fingerprint}.{ext}', fp, mime)}
            resp = requests.post(
                f'{render_url}/artwork/upload/{fingerprint}',
                files=files,
                timeout=15,
            )
        if resp.status_code == 200:
            logger.info(f'[Artwork] Push -> Render OK: {short_fp}')
        else:
            body = (resp.text or '')[:200]
            logger.warning(
                f'[Artwork] Push -> Render {resp.status_code}: {short_fp} | {body}'
            )
    except Exception as push_err:
        logger.error(f'[Artwork] Push -> Render error ({short_fp}): {push_err}')


def _push_artwork_async(fingerprint: str, local_path: str) -> None:
    """Lanza `_push_artwork_to_render` en un thread daemon."""
    threading.Thread(
        target=_push_artwork_to_render,
        args=(fingerprint, local_path),
        daemon=True,
    ).start()

def generate_preview_snippet(
    file_path: str,
    fingerprint: str,
    drop_timestamp: float,
    duration: float,
) -> Optional[str]:
    """
    Genera snippet MP3 de 6s desde el punto más interesante del track.
    
    Formato: MP3 mono, 64kbps, 22050Hz ≈ 48KB por track.
    Incluye fade in (0.3s) y fade out (0.5s) via ffmpeg.
    
    Args:
        file_path: Ruta al archivo de audio temporal
        fingerprint: Hash MD5 del archivo (usado como ID)
        drop_timestamp: Timestamp del drop/punto energético (en segundos)
        duration: Duración total del track (en segundos)
    
    Returns:
        Ruta al snippet generado, o None si falla
    """
    output_path = os.path.join(PREVIEWS_DIR, f"{fingerprint}.mp3")
    
    # Si ya existe, no regenerar
    if os.path.exists(output_path):
        logger.info(f"  [Preview] Ya existe snippet para {fingerprint[:8]}...")
        return output_path
    
    # Calcular punto de inicio: 2s antes del drop para capturar buildup
    start = max(0, drop_timestamp - 2.0)
    # Asegurar que no nos pasamos del final
    if start + 6 > duration:
        start = max(0, duration - 6)
    # Si el track dura menos de 6s, empezar desde 0
    if duration < 6:
        start = 0
    
    try:
        # FFmpeg con ruta absoluta cuando FFMPEG_BIN esta seteado (lo pone
        # local_engine.py). Evita WinError 448 en Windows 11 24H2+ cuando
        # hay reparse points en el PATH. Fallback a 'ffmpeg' para produccion
        # Render donde esta en PATH limpio.
        ffmpeg_bin = os.environ.get('FFMPEG_BIN', 'ffmpeg')
        cmd = [
            ffmpeg_bin, '-y',
            '-ss', str(round(start, 2)),
            '-i', file_path,
            '-t', '6',
            '-ac', '1',           # Mono
            '-ab', '64k',         # 64kbps
            '-ar', '22050',       # Sample rate bajo (suficiente para preview)
            '-af', 'afade=t=in:st=0:d=0.3,afade=t=out:st=5.5:d=0.5',  # Fade in/out
            output_path
        ]
        
        run_kwargs = dict(capture_output=True, timeout=15, check=True)
        if sys.platform == 'win32':
            run_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        proc_result = subprocess.run(cmd, **run_kwargs)
        
        # Verificar que el archivo se generó y tiene tamaño razonable
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            if size > 1000:  # Mínimo 1KB
                print(f"  [Preview] Snippet generado: {fingerprint[:8]}... "
                      f"({size//1024}KB, start={start:.1f}s)")
                return output_path
            else:
                logger.info(f"  [Preview] Snippet demasiado pequeño ({size}B), eliminando")
                os.unlink(output_path)
                return None
        
        return None
        
    except subprocess.TimeoutExpired:
        logger.info(f"  [Preview] Timeout generando snippet para {fingerprint[:8]}...")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None
    except subprocess.CalledProcessError as e:
        # stderr[:200] solo recoge el banner "ffmpeg version 5.1.8..." y
        # disparaba alertas falsas. Filtramos lineas que parecen error real
        # o caemos al tail. Mismo patron que preview_generator.py y
        # _preprocess_audio_for_recognition.
        stderr_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else ''
        err_lines = [
            ln for ln in stderr_msg.splitlines()
            if ln and (
                ln.lower().startswith('error')
                or 'invalid' in ln.lower()
                or 'no such file' in ln.lower()
                or 'permission denied' in ln.lower()
                or 'failed' in ln.lower()
            )
        ]
        real_err = ' | '.join(err_lines[-3:])[:400] if err_lines else stderr_msg[-400:].strip() or 'unknown'
        logger.error(f"  [Preview] ffmpeg exit {e.returncode}: {real_err}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None
    except Exception as e:
        logger.error(f"  [Preview] Error generando snippet: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None


# ==================== ANALISIS PRINCIPAL ====================

def robust_audio_load(file_path: str, sr: int = 44100, mono: bool = True):
    """Carga audio resiliente. En Render el libsndfile no decodifica algunos
    mp3 y audioread no encuentra backend (-> NoBackendError / LibsndfileError
    'File does not exist or is not a regular file'). ffmpeg SI esta disponible
    (health lo confirma), asi que si librosa.load falla transcodificamos a WAV
    con ffmpeg y lo leemos con soundfile. Si ffmpeg tampoco puede, el archivo
    es de verdad ilegible y propagamos la excepcion (el middleware la registra
    una vez, no en bucle, porque el track ya no se reintenta sin cambios).
    """
    try:
        with silence_native_stderr():
            return librosa.load(file_path, sr=sr, mono=mono)
    except Exception as e:
        logger.warning(
            f"[decode] librosa.load fallo ({type(e).__name__}: {e}); "
            f"reintentando via ffmpeg->wav"
        )
        import soundfile as sf
        wav_path = file_path + '.dec.wav'
        try:
            subprocess.run(
                ['ffmpeg', '-v', 'error', '-y', '-i', file_path,
                 '-ac', '1' if mono else '2', '-ar', str(sr), wav_path],
                capture_output=True, timeout=180, check=True,
            )
            y, file_sr = sf.read(wav_path, dtype='float32')
            if mono and getattr(y, 'ndim', 1) > 1:
                y = y.mean(axis=1)
            return y, file_sr
        finally:
            if os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass


def analyze_audio(file_path: str, fingerprint: str = None, force_audd: bool = False,
                  original_filename: Optional[str] = None) -> AnalysisResult:
    import warnings
    warnings.filterwarnings('ignore')

    # Defensa basica contra cleanup race (ver bug del 14/05 con
    # /tmp/tmpyvolczw6.mp3 desaparecido al llegar a librosa.load).
    # Mejor mensaje + falla rapido sin pasear por audioread/soundfile.
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Archivo a analizar no existe en disco: {file_path}. "
            "Posible cleanup race del tmp_path."
        )

    #  Obtener duracin SIN cargar audio completo
    try:
        with silence_native_stderr():
            duration = librosa.get_duration(path=file_path)
    except Exception:
        # soundfile no soporta mp3 nativo en algunos entornos; ffprobe fallback
        try:
            proc = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'csv=p=0', file_path],
                capture_output=True, text=True, timeout=30
            )
            duration = float(proc.stdout.strip() or '0')
        except Exception:
            duration = 0.0

    #  Si el track es largo (>4 min), usar anlisis por chunks
    if CHUNKED_ANALYZER_ENABLED and duration > CHUNK_ANALYSIS_THRESHOLD:
        logger.info(f" Track largo ({duration/60:.1f} min) - Usando anlisis por chunks")
        return analyze_audio_chunked(file_path, fingerprint, duration, force_audd=force_audd,
                                     original_filename=original_filename)

    # Track corto: anlisis tradicional (carga todo en RAM)
    logger.info(f" Track corto ({duration/60:.1f} min) - Usando anlisis tradicional")
    y, sr = robust_audio_load(file_path, sr=44100, mono=True)

    
    # ==================== ID3 METADATA ====================
    id3_data = {}
    if ARTWORK_ENABLED:
        id3_data = extract_id3_metadata(file_path)
    
    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # librosa >=0.10 puede devolver tempo como np.ndarray (ej. array([120.5]))
    # en lugar de un escalar. Normalizar antes de convertir a float.
    if hasattr(tempo, '__iter__'):
        tempo = tempo[0] if len(tempo) > 0 else 0.0
    bpm = float(tempo)
    bpm_source = "analysis"
    
    # Usar BPM de ID3 si existe y es razonable
    if id3_data.get('bpm') and 60 < id3_data['bpm'] < 200:
        bpm = id3_data['bpm']
        bpm_source = "id3"
    
    beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
    bpm_confidence = 1.0 - min(np.std(beat_intervals) * 2, 0.5) if len(beat_intervals) > 0 else 0.5
    
    # Calcular onset_env una sola vez (se reusa en BPM correction + percussion_density)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # MEJORA 3: Auto-correccion half/double tempo si confianza baja
    if bpm_source == "analysis":
        bpm = try_bpm_double_half(y, sr, bpm, bpm_confidence, onset_env=onset_env)
    
    if len(beat_intervals) > 1:
        groove_score = min(np.std(beat_intervals) * 10, 1.0)
        swing_factor = float(np.mean(beat_intervals[::2]) / np.mean(beat_intervals[1::2]) 
                           if len(beat_intervals) > 2 else 0.5)
    else:
        groove_score = 0.0
        swing_factor = 0.5
    
    # Key - Krumhansl-Kessler profiles (academicamente validados)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, n_octaves=7)
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Perfiles Krumhansl-Kessler REALES (no binarios)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile = major_profile / np.sum(major_profile)
    minor_profile = minor_profile / np.sum(minor_profile)
    
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-10)
    
    # Escanear TODAS las keys (no solo key_idx de argmax)
    best_key = None
    best_corr = -1
    best_scale = None
    
    for i, key_name in enumerate(keys):
        major_rot = np.roll(major_profile, i)
        minor_rot = np.roll(minor_profile, i)
        
        major_corr = np.corrcoef(chroma_mean, major_rot)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rot)[0, 1]
        
        if not np.isnan(major_corr) and major_corr > best_corr:
            best_corr = major_corr
            best_key = key_name
            best_scale = 'major'
        if not np.isnan(minor_corr) and minor_corr > best_corr:
            best_corr = minor_corr
            best_key = key_name
            best_scale = 'minor'
    
    is_minor = best_scale == 'minor'
    # Defensa contra audio degenerado: si todos los major_corr y minor_corr
    # fueron NaN (chroma plano, silencio, archivo corrupto), best_key
    # se queda en None. Antes la concatenacion `None + 'm'` petaba con
    # TypeError NoneType+str (visto en panel admin 2026-05-20).
    # Fallback: marcar como 'C' con confianza minima — el track aun se
    # analiza, key_confidence=0 indica al cliente "no fiable".
    if best_key is None:
        logger.warning(
            "[Key] best_key=None tras escaneo Krumhansl (audio plano / "
            "todos NaN). Fallback a 'C' con confidence=0."
        )
        best_key = 'C'
        is_minor = False
        best_corr = 0.0
    key = best_key + ('m' if is_minor else '')
    key_source = "analysis"
    camelot = KEY_TO_CAMELOT.get(key, '?')
    key_confidence = float(best_corr) if best_corr > 0 else 0.5
    
    # Confianza major/minor: si la diferencia es minima, marcar como baja
    # para que Beatport/ID3 tengan prioridad
    major_best = -1
    minor_best = -1
    for i in range(12):
        mc = np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
        nc = np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]
        if not np.isnan(mc) and mc > major_best:
            major_best = mc
        if not np.isnan(nc) and nc > minor_best:
            minor_best = nc
    mode_margin = abs(major_best - minor_best)
    if mode_margin < 0.05:
        key_confidence = min(key_confidence, 0.55)  # Baja confianza en modo
        logger.info(f"   Key: {key} ({camelot}) [modo ambiguo: margen={mode_margin:.3f}]")
    
    # Usar Key de ID3 solo si existe y es vlido
    id3_key = id3_data.get('key')
    if id3_key and id3_key.strip() and id3_key.strip() not in ['?', '', 'Unknown', 'None']:
        id3_key = id3_key.strip()
        key_source = "id3"
        # Intentar mapear a Camelot
        id3_camelot = KEY_TO_CAMELOT.get(id3_key, None)
        if id3_camelot:
            key = id3_key
            camelot = id3_camelot
        # Si es formato Camelot directo (ej: "8A", "11B")
        elif len(id3_key) >= 2 and id3_key[-1].upper() in ['A', 'B'] and id3_key[:-1].isdigit():
            camelot = id3_key.upper()
            key = id3_key
        else:
            # ID3 key no reconocida, mantener anlisis
            key_source = "analysis"
    
    # Energy - Escala DJ 1-10 con curva power para mejor distribucion
    rms = librosa.feature.rms(y=y)[0]
    energy_raw = float(np.mean(rms))
    
    # RMS tipico en musica electronica: 0.02 (ambient) a 0.42+ (hardstyle)
    # MEJORA 2: Curva power 0.55 que expande el rango medio
    # El frontend puede ADEMAS aplicar percentiles sobre la biblioteca local
    # para distribucion aun mas precisa (energy_raw se guarda para esto)
    
    if energy_raw <= 0.02:
        energy_dj = 1
    elif energy_raw >= 0.42:
        energy_dj = 10
    elif not math.isfinite(energy_raw):
        # NaN/Inf: el RMS puede salir NaN con audios muy cortos, silencio
        # total o frames problematicos. Sin esta guard, int(NaN) explota
        # con ValueError 'cannot convert float NaN to integer' (era el
        # error #1 mas frecuente en el panel admin, 112 ocurrencias).
        energy_dj = 5
        logger.warning(f"   Energia: energy_raw={energy_raw} NaN/Inf, fallback a 5")
    else:
        normalized = (energy_raw - 0.02) / (0.42 - 0.02)
        powered = normalized ** 0.55  # expande rango bajo-medio
        energy_dj = int(round(1 + powered * 9))
        energy_dj = max(1, min(10, energy_dj))
    
    energy_normalized = energy_dj / 10.0
    logger.info(f"   Energia: raw={energy_raw:.4f} -> DJ level {energy_dj}")
    
    chunk_size = int(sr * 30)
    mix_energy_start = float(np.mean(rms[:min(chunk_size//512, len(rms))]))
    mix_energy_end = float(np.mean(rms[max(0, len(rms)-chunk_size//512):]))
    
    # Structure
    segments = detect_structure(y, sr, duration)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    has_vocals = detect_vocals_improved(y, sr, spectral_centroid)
    
    low_freq_energy = np.mean(np.abs(y[:int(sr*10)]))
    has_heavy_bass = low_freq_energy > energy_raw * 0.8
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    has_pads = float(np.std(rolloff)) < 1000
    
    # onset_env ya calculado arriba (antes de BPM correction) - reusar
    percussion_density = min(float(np.mean(onset_env)) / 10, 1.0)
    
    # Classification (heuristic, Fase 1 v2)
    track_type = 'peak_time'  # default seguro
    track_type_confidence = 0.5  # neutral si la clasificacion falla
    track_type_alternatives: List[Dict[str, Any]] = []
    try:
        classification = classify_track_type(energy_normalized, segments, duration)
        track_type = classification['type']
        track_type_confidence = classification['confidence']
        track_type_alternatives = classification['alternatives']
    except Exception as e:
        logger.error(f"  [TrackType] Error clasificando: {e}")
        classification = None

    # Spectral + ensemble (Fase 3 v2): metrics FFT + scoring 7 tipos
    # (vs 3 del heuristic). El spectral pesa β=1.5 vs α=1.0 del heuristic.
    # Refina tambien has_heavy_bass con bassRatio normalizado per-band.
    try:
        from spectral_classifier import (
            compute_spectral_metrics,
            classify_track_type_spectral,
            detect_heavy_bass as _spectral_detect_heavy_bass,
            ensemble_classify,
        )
        spectral_metrics = compute_spectral_metrics(y, sr)
        spectral_classification = classify_track_type_spectral(
            spectral_metrics, bpm, duration
        )
        ensemble = ensemble_classify(classification, spectral_classification)
        track_type = ensemble['type']
        track_type_confidence = ensemble['confidence']
        track_type_alternatives = ensemble['alternatives']
        # Heavy bass refinado (per-band ratio > heuristic crudo de low_freq_energy)
        has_heavy_bass = _spectral_detect_heavy_bass(spectral_metrics)
        logger.info(
            f"  [Spectral+Ensemble] {ensemble['type']} "
            f"conf={ensemble['confidence']:.2f} | {ensemble['reason']}"
        )
    except Exception as e:
        logger.warning(f"  [Spectral] Failed, usando solo heuristic: {e}")
    genre = classify_genre_advanced(
        bpm, energy_normalized, has_heavy_bass,
        y, sr, percussion_density,
        spectral_centroid, rolloff
    )
    genre_source = "spectral_analysis"
    label = id3_data.get('label')
    year = id3_data.get('year')
    
    # Guardar g(c)nero ID3 como fallback (suele ser gen(c)rico: "House", "Techno")
    id3_genre = id3_data.get('genre')
    
    # ==================== PRIORIDAD DE GENEROS ====================
    # Discogs > MusicBrainz > ID3 > Anlisis espectral
    # Discogs/MusicBrainz dan g(c)neros especficos (ej: "Minimal Techno" vs "Techno")
    
    artist_name = id3_data.get('artist')
    title_name = id3_data.get('title')
    
    if GENRE_DETECTOR_ENABLED and genre_detector and artist_name and title_name:
        logger.info(f" Buscando g(c)nero: {artist_name} - {title_name}")
        # 1. Intentar Discogs primero (mejor para electrnica)
        try:
            discogs_result = genre_detector.get_discogs_genre(artist_name, title_name)
            if discogs_result and discogs_result.get('genre'):
                genre = discogs_result.get('genre')
                genre_source = "discogs"
                # Tambi(c)n obtener label y year si no los tenemos
                if not label and discogs_result.get('label'):
                    label = discogs_result['label']
                if not year and discogs_result.get('year'):
                    year = str(discogs_result['year'])
                logger.info(f" Discogs: {genre} | {label} ({year})")
            else:
                logger.info(f" Discogs: No encontrado")
        except Exception as e:
            # 404/JSON-empty/timeout son respuestas esperadas de un servicio
            # externo, no bugs nuestros. Tenemos fallback a MusicBrainz/ID3.
            # Warning evita disparar alertas de error en el panel admin.
            logger.warning(f" Discogs no disponible ({type(e).__name__}): {e}")

        # 2. Si no hay Discogs, intentar MusicBrainz
        if genre_source not in ["discogs"]:
            try:
                mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                if mb_result and mb_result.get('genre'):
                    genre = mb_result.get('genre')
                    genre_source = "musicbrainz"
                    logger.info(f"   MusicBrainz: {genre}")
            except Exception as e:
                logger.warning(f" MusicBrainz no disponible ({type(e).__name__}): {e}")
    
    # 3. Si no hay Discogs ni MusicBrainz, usar ID3 (gen(c)rico pero mejor que nada)
    if genre_source == "spectral_analysis" and id3_genre:
        genre = id3_genre
        genre_source = "id3"
        logger.info(f" ID3 (fallback): {genre}")

    # ==================== RESOLVER ARTIST / TITLE ====================
    # Beatport solia ir aqui pero se elimino (WAF de Cloudflare bloquea el
    # scraping desde datacenter e IP residencial sin browser real; los 84
    # tracks de produccion tienen 0 con bpm_source='beatport'). El boton de
    # Flutter "Buscar en Beatport" sigue funcionando porque solo abre URL.

    if not artist_name:
        artist_name = id3_data.get('artist')
    if not title_name:
        title_name = id3_data.get('title')

    # Si no hay metadata ID3, intentar con filename parseado.
    # IMPORTANTE: usar el filename REAL (original_filename) y NO el basename
    # del file_path, que en /analyze es el tmp_path (/tmp/tmpXXXXXX.mp3). Si
    # parseabamos el basename del temp, el "title" salia "tmpXXXXXX" (basura)
    # y como no quedaba vacio, el fallback del endpoint con file.filename
    # nunca disparaba -> la limpieza proponia nombres tmpXXXX. Bug 2026-06.
    if not artist_name or not title_name:
        parsed = parse_filename(original_filename or os.path.basename(file_path))
        if not artist_name:
            artist_name = parsed.get('artist')
        if not title_name:
            title_name = parsed.get('title')

    # ==================== AUDD AUTO-TRIGGER ====================
    # Si tras ID3 + filename seguimos sin artist/title utilizable, AudD como
    # ultimo recurso (con presupuesto y cooldown). Discogs/iTunes/MusicBrainz
    # requieren artist+title para arrancar, asi que recuperar la identidad
    # aqui desbloquea el resto.
    audd_artwork = None  # portada exacta del match AudD (apple_music/deezer/spotify)
    if AUDD_AUTO_ENABLED and AUDD_API_TOKEN:
        try:
            from audd_helper import enrich_with_audd_if_needed, download_artwork_from_audd
            audd_track = enrich_with_audd_if_needed(
                file_path=file_path,
                fingerprint=fingerprint,
                duration=duration,
                artist=artist_name,
                title=title_name,
                api_token=AUDD_API_TOKEN,
                db=db,
                min_duration=AUDD_MIN_DURATION,
                max_duration=AUDD_MAX_DURATION,
                daily_cap=AUDD_DAILY_CAP,
                cooldown_days=AUDD_COOLDOWN_DAYS,
                force=force_audd,
            )
            if audd_track:
                if audd_track.get('artist'):
                    artist_name = audd_track['artist']
                if audd_track.get('title'):
                    title_name = audd_track['title']
                if not label and audd_track.get('label'):
                    label = audd_track['label']
                if not year and audd_track.get('release_date'):
                    year = audd_track['release_date'][:4]
                # Portada exacta del match AudD (Apple Music/Deezer/Spotify):
                # AudD ya identifico el track preciso, asi que su caratula es la
                # oficial del release — mejor que re-buscar por texto. Gratis
                # (la respuesta ya venia con esos campos). Se usa como candidata
                # preferente en el bloque de ARTWORK de abajo.
                audd_artwork = download_artwork_from_audd(audd_track)
                # Re-correr Discogs/MusicBrainz si la cascada anterior no aporto
                # genero (sigue siendo el default analitico).
                if (genre_source in ('spectral_analysis', 'chunked_analysis')
                        and GENRE_DETECTOR_ENABLED and genre_detector):
                    try:
                        discogs_result = genre_detector.get_discogs_genre(artist_name, title_name)
                        if discogs_result and discogs_result.get('genre'):
                            genre = discogs_result['genre']
                            genre_source = 'discogs'
                            if not label and discogs_result.get('label'):
                                label = discogs_result['label']
                            if not year and discogs_result.get('year'):
                                year = str(discogs_result['year'])
                    except Exception as e:
                        # Mismo criterio que la cascada principal: fallo de
                        # servicio externo es warning, no error.
                        logger.warning(f"  [AudD-auto] re-run Discogs ({type(e).__name__}): {e}")
                    if genre_source in ('spectral_analysis', 'chunked_analysis'):
                        try:
                            mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                            if mb_result and mb_result.get('genre'):
                                genre = mb_result['genre']
                                genre_source = 'musicbrainz'
                        except Exception as e:
                            logger.warning(f"  [AudD-auto] re-run MusicBrainz ({type(e).__name__}): {e}")
        except Exception as e:
            logger.warning(f"  [AudD-auto] error ({type(e).__name__}): {e}")

    drop_time = find_drop_timestamp(y, sr, segments)
    
    # ==================== CUE POINTS ====================
    cue_points = []
    first_beat = 0.0
    beat_interval = 0.5
    
    if ARTWORK_ENABLED:
        cue_points = detect_cue_points(y, sr, duration, segments)
        beat_grid = detect_beat_grid(y, sr, bpm)
        first_beat = beat_grid.get('first_beat', 0.0)
        beat_interval = beat_grid.get('beat_interval', 0.5)
    
    # ==================== ARTWORK ====================
    artwork_embedded = False
    artwork_url = None
    artwork_source = None

    # ID3 grande (>=100KB) se acepta directo. Por debajo, comparamos con
    # online y nos quedamos con el de mayor tamaño — mejor proxy de
    # calidad sin parsear dimensiones de imagen. Esto cubre el caso real:
    # ID3 default cutre (20-50KB, 300x300 borroso) cuando iTunes/Deezer
    # tienen la version oficial 1200x1200 (>200KB).
    ID3_TRUSTED_THRESHOLD = 100_000  # 100 KB

    if ARTWORK_ENABLED and fingerprint:
        artwork_info = extract_artwork_from_file(file_path)
        id3_size = artwork_info.get('size', 0) if artwork_info else 0

        # Decidir si pedir online: solo si ID3 no es claramente bueno
        online_artwork = None
        online_size = 0
        # Candidata preferente: la portada exacta que AudD ya nos dio (match
        # exacto del track). Evita la busqueda por texto (menos fiable, mas
        # latencia) cuando AudD identifico el track.
        if audd_artwork:
            online_artwork = audd_artwork
            online_size = audd_artwork.get('size', 0)
        # Bug fix 2026-06-05: no sobreescribir artist_name/title_name con id3_data
        # crudo aqui — ya estan enriquecidos por AudD (linea ~1553). Usar directo.
        # El path chunked (analyze_audio_chunked) ya lo hacia bien desde siempre.
        elif id3_size < ID3_TRUSTED_THRESHOLD and artist_name and title_name:
            album_name = id3_data.get('album')
            try:
                from artwork_and_cuepoints import search_artwork_online
                online_artwork = search_artwork_online(
                    artist_name, title_name, album_name
                )
                online_size = online_artwork.get('size', 0) if online_artwork else 0
            except Exception as e:
                logger.warning(f"   Artwork online fallo: {e}")
                try:
                    import traceback as _tb
                    db.log_analysis_error(
                        device_id=None, filename=None, fingerprint=fingerprint,
                        error_class=type(e).__name__, error_msg=str(e),
                        traceback_str=_tb.format_exc(), endpoint='artwork',
                    )
                except Exception:
                    pass

        # Elegir: ID3 si es grande (>=100KB) o si supera al online en tamaño.
        # Online si es valido (>=10KB) y mayor que ID3. Sino, nada.
        use_id3 = id3_size >= ID3_TRUSTED_THRESHOLD or (
            id3_size > 10000 and id3_size >= online_size
        )
        use_online = (
            online_artwork is not None
            and online_size >= 10000
            and online_size > id3_size
        )

        if use_id3 and artwork_info:
            artwork_embedded = True
            artwork_source = "id3"
            saved_filename = save_artwork_to_cache(
                fingerprint, artwork_info['data'], artwork_info['mime_type']
            )
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            _push_artwork_async(
                fingerprint, os.path.join(ARTWORK_CACHE_DIR, saved_filename)
            )
            logger.info(
                f"   Artwork ID3: {id3_size} bytes "
                f"(online disponible: {online_size} bytes)"
            )
        elif use_online and online_artwork:
            artwork_embedded = False
            artwork_source = online_artwork.get('source', 'online')
            saved_filename = save_artwork_to_cache(
                fingerprint, online_artwork['data'], online_artwork['mime_type']
            )
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            _push_artwork_async(
                fingerprint, os.path.join(ARTWORK_CACHE_DIR, saved_filename)
            )
            logger.info(
                f"   Artwork {artwork_source}: {online_size} bytes "
                f"(ID3 era: {id3_size} bytes)"
            )
        else:
            logger.info(
                f"   Sin artwork: ID3={id3_size}b, online={online_size}b"
            )

    # ==================== TRACK TYPE: defaults + guards ====================
    track_type_source = 'waveform'
    # Guard: asegurar que track_type / confidence / alternatives existen
    # (por si classify_track_type fallo arriba).
    try:
        track_type
    except NameError:
        track_type = 'peak_time'
    try:
        track_type_confidence
    except NameError:
        track_type_confidence = 0.5
    try:
        track_type_alternatives
    except NameError:
        track_type_alternatives = []

    # ==================== TRACK TYPE: COMMUNITY CONSENSUS (Fase 2) ====================
    # Si NO hay Beatport hint, probar consensus comunitario. Si motor local,
    # pregunta a Render via HTTP (timeout 2s). Si Render no responde o no hay
    # consensus, fallback al algoritmico que ya estaba.
    if track_type_source != 'beatport' and fingerprint:
        try:
            community = (_fetch_community_track_type(fingerprint)
                         if IS_LOCAL_ENGINE
                         else db.get_track_type_consensus(fingerprint))
            if community:
                logger.info(
                    f"  [Community] Track type consensus override: "
                    f"{track_type} -> {community['type']} ({community['votes']} votos)"
                )
                track_type = community['type']
                track_type_source = 'community'
                track_type_confidence = 1.0
                track_type_alternatives = [
                    {'type': community['type'], 'score': float(community['votes'])}
                ]
        except Exception as e:
            logger.debug(f"  [Community] Consensus check fallo: {e}")

    # ==================== GENRE / KEY: COMMUNITY CONSENSUS (Fase 4) ====================
    # Solo aplica si la fuente no es ya autoritativa (beatport / id3 fuerte).
    # Cascada general: beatport > id3 > community > algoritmico/spectral.
    # Si community consensus existe y la fuente actual no es beatport, override.
    if fingerprint:
        # Genre
        if genre_source != 'beatport':
            try:
                cm_genre = (_fetch_community_override(fingerprint, 'genre')
                            if IS_LOCAL_ENGINE
                            else db.get_community_consensus(fingerprint, 'genre'))
                if cm_genre:
                    logger.info(
                        f"  [Community] Genre consensus override: "
                        f"{genre} -> {cm_genre['value']} ({cm_genre['votes']} votos)"
                    )
                    genre = cm_genre['value']
                    genre_source = 'community'
            except Exception as e:
                logger.debug(f"  [Community] Genre consensus fallo: {e}")
        # Key + camelot (derivado de key via KEY_TO_CAMELOT)
        if key_source != 'beatport' and key_source != 'id3':
            try:
                cm_key = (_fetch_community_override(fingerprint, 'key')
                          if IS_LOCAL_ENGINE
                          else db.get_community_consensus(fingerprint, 'key'))
                if cm_key:
                    new_key = cm_key['value']
                    new_camelot = KEY_TO_CAMELOT.get(new_key)
                    if new_camelot:  # Solo override si la key es mapeable
                        logger.info(
                            f"  [Community] Key consensus override: "
                            f"{key} -> {new_key} ({new_camelot}) ({cm_key['votes']} votos)"
                        )
                        key = new_key
                        camelot = new_camelot
                        key_source = 'community'
                        key_confidence = 1.0
            except Exception as e:
                logger.debug(f"  [Community] Key consensus fallo: {e}")

    # Fix 2026-05-13: usar title_name/artist_name (enriquecidos por AudD
    # auto-trigger si se disparo) en vez de id3_data.get(...) crudo. Antes
    # AudD identificaba el track, actualizaba artist_name/title_name local,
    # Discogs corria con ellos (genre correcto) pero la respuesta JSON
    # devolvia los basura originales ("Unknown"/"Track 01") porque se leia
    # de id3_data en este return. Verificado en test ficticio con Oxia -
    # Domino renombrado a track01.mp3 (audd success=1 pero artist="Unknown"
    # en la respuesta).
    return AnalysisResult(
        title=title_name or id3_data.get('title'),
        artist=artist_name or id3_data.get('artist'),
        album=id3_data.get('album'),
        label=label,
        year=year,
        isrc=id3_data.get('isrc'),
        duration=duration,
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        bpm_source=bpm_source,
        key=key,
        camelot=camelot,
        key_confidence=key_confidence,
        key_source=key_source,
        energy_raw=energy_raw,
        energy_normalized=energy_normalized,
        energy_dj=energy_dj,
        groove_score=groove_score,
        swing_factor=swing_factor,
        has_intro=segments['has_intro'],
        has_buildup=segments['has_buildup'],
        has_drop=segments['has_drop'],
        has_breakdown=segments['has_breakdown'],
        has_outro=segments['has_outro'],
        structure_sections=segments['sections'],
        track_type=track_type,
        track_type_source=track_type_source,
        track_type_confidence=track_type_confidence,
        track_type_alternatives=track_type_alternatives,
        genre=genre,
        genre_source=genre_source,
        has_vocals=has_vocals,
        has_heavy_bass=has_heavy_bass,
        has_pads=has_pads,
        percussion_density=percussion_density,
        mix_energy_start=mix_energy_start,
        mix_energy_end=mix_energy_end,
        drop_timestamp=drop_time,
        cue_points=cue_points,
        first_beat=first_beat,
        beat_interval=beat_interval,
        artwork_embedded=artwork_embedded,
        artwork_url=artwork_url,
    )

def analyze_audio_chunked(file_path: str, fingerprint: str, duration: float, force_audd: bool = False,
                          original_filename: Optional[str] = None) -> AnalysisResult:
    """
    Analiza tracks largos por chunks para reducir uso de RAM.
    Usado automticamente para tracks > 4 minutos.
    """
    import gc
    
    # Crear analizador por chunks
    analyzer = get_chunked_analyzer(chunk_duration=60)
    
    # Ejecutar anlisis chunked
    result = analyzer.full_analysis(file_path)
    
    # Limpiar memoria
    del analyzer
    gc.collect()
    
    # ==================== ID3 METADATA ====================
    id3_data = {}
    if ARTWORK_ENABLED:
        id3_data = extract_id3_metadata(file_path)
    
    # Sobrescribir con ID3 si existe y es vlido
    bpm = result['bpm']
    bpm_source = result['bpm_source']
    if id3_data.get('bpm') and 60 < id3_data['bpm'] < 200:
        bpm = id3_data['bpm']
        bpm_source = "id3"
    
    key = result['key']
    camelot = result['camelot']
    key_source = result['key_source']
    id3_key = id3_data.get('key')
    if id3_key and id3_key.strip() and id3_key.strip() not in ['?', '', 'Unknown', 'None']:
        id3_key = id3_key.strip()
        id3_camelot = KEY_TO_CAMELOT.get(id3_key, None)
        if id3_camelot:
            key = id3_key
            camelot = id3_camelot
            key_source = "id3"
        elif len(id3_key) >= 2 and id3_key[-1].upper() in ['A', 'B'] and id3_key[:-1].isdigit():
            camelot = id3_key.upper()
            key = id3_key
            key_source = "id3"
    
    # ==================== GNERO ====================
    genre = "Electronic"
    genre_source = "chunked_analysis"
    label = id3_data.get('label')
    year = id3_data.get('year')
    id3_genre = id3_data.get('genre')
    
    artist_name = id3_data.get('artist')
    title_name = id3_data.get('title')
    
    # Intentar obtener g(c)nero de Discogs/MusicBrainz
    if GENRE_DETECTOR_ENABLED and genre_detector and artist_name and title_name:
        logger.info(f"   Buscando g(c)nero: {artist_name} - {title_name}")
        try:
            discogs_result = genre_detector.get_discogs_genre(artist_name, title_name)
            if discogs_result and discogs_result.get('genre'):
                genre = discogs_result.get('genre')
                genre_source = "discogs"
                if not label and discogs_result.get('label'):
                    label = discogs_result['label']
                if not year and discogs_result.get('year'):
                    year = str(discogs_result['year'])
                logger.info(f"   Discogs: {genre}")
        except Exception as e:
            logger.error(f"   Error Discogs: {e}")
        
        if genre_source not in ["discogs"]:
            try:
                mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                if mb_result and mb_result.get('genre'):
                    genre = mb_result.get('genre')
                    genre_source = "musicbrainz"
                    logger.info(f"   MusicBrainz: {genre}")
            except Exception as e:
                logger.error(f"   Error MusicBrainz: {e}")
    
    if genre_source == "chunked_analysis" and id3_genre:
        genre = id3_genre
        genre_source = "id3"

    # ==================== RESOLVER ARTIST / TITLE ====================
    # Beatport solia ir aqui pero se elimino (WAF de Cloudflare bloquea el
    # scraping desde datacenter e IP residencial sin browser real; los 84
    # tracks de produccion tienen 0 con bpm_source='beatport'). El boton de
    # Flutter "Buscar en Beatport" sigue funcionando porque solo abre URL.

    if not artist_name:
        artist_name = id3_data.get('artist')
    if not title_name:
        title_name = id3_data.get('title')

    # Si no hay metadata ID3, intentar con filename parseado.
    # IMPORTANTE: usar el filename REAL (original_filename) y NO el basename
    # del file_path, que en /analyze es el tmp_path (/tmp/tmpXXXXXX.mp3). Si
    # parseabamos el basename del temp, el "title" salia "tmpXXXXXX" (basura)
    # y como no quedaba vacio, el fallback del endpoint con file.filename
    # nunca disparaba -> la limpieza proponia nombres tmpXXXX. Bug 2026-06.
    if not artist_name or not title_name:
        parsed = parse_filename(original_filename or os.path.basename(file_path))
        if not artist_name:
            artist_name = parsed.get('artist')
        if not title_name:
            title_name = parsed.get('title')

    # ==================== AUDD AUTO-TRIGGER ====================
    # Mismo trigger que en analyze_audio: si tras ID3+filename seguimos sin
    # artist/title utilizable, AudD como ultimo recurso.
    audd_artwork = None  # portada exacta del match AudD (apple_music/deezer/spotify)
    if AUDD_AUTO_ENABLED and AUDD_API_TOKEN:
        try:
            from audd_helper import enrich_with_audd_if_needed, download_artwork_from_audd
            audd_track = enrich_with_audd_if_needed(
                file_path=file_path,
                fingerprint=fingerprint,
                duration=duration,
                artist=artist_name,
                title=title_name,
                api_token=AUDD_API_TOKEN,
                db=db,
                min_duration=AUDD_MIN_DURATION,
                max_duration=AUDD_MAX_DURATION,
                daily_cap=AUDD_DAILY_CAP,
                cooldown_days=AUDD_COOLDOWN_DAYS,
                force=force_audd,
            )
            if audd_track:
                if audd_track.get('artist'):
                    artist_name = audd_track['artist']
                if audd_track.get('title'):
                    title_name = audd_track['title']
                if not label and audd_track.get('label'):
                    label = audd_track['label']
                if not year and audd_track.get('release_date'):
                    year = audd_track['release_date'][:4]
                # Portada exacta del match AudD (ver path no-chunked arriba).
                audd_artwork = download_artwork_from_audd(audd_track)
                if (genre_source in ('spectral_analysis', 'chunked_analysis')
                        and GENRE_DETECTOR_ENABLED and genre_detector):
                    try:
                        discogs_result = genre_detector.get_discogs_genre(artist_name, title_name)
                        if discogs_result and discogs_result.get('genre'):
                            genre = discogs_result['genre']
                            genre_source = 'discogs'
                            if not label and discogs_result.get('label'):
                                label = discogs_result['label']
                            if not year and discogs_result.get('year'):
                                year = str(discogs_result['year'])
                    except Exception as e:
                        # Mismo criterio que la cascada principal: fallo de
                        # servicio externo es warning, no error.
                        logger.warning(f"  [AudD-auto] re-run Discogs ({type(e).__name__}): {e}")
                    if genre_source in ('spectral_analysis', 'chunked_analysis'):
                        try:
                            mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                            if mb_result and mb_result.get('genre'):
                                genre = mb_result['genre']
                                genre_source = 'musicbrainz'
                        except Exception as e:
                            logger.warning(f"  [AudD-auto] re-run MusicBrainz ({type(e).__name__}): {e}")
        except Exception as e:
            logger.warning(f"  [AudD-auto] error ({type(e).__name__}): {e}")

    # ==================== ARTWORK ====================
    # Misma logica que en el flow no-chunked (linea ~1487): ID3 grande
    # (>=100KB) se acepta directo; por debajo, comparamos con online y
    # nos quedamos con el de mayor tamaño.
    artwork_embedded = False
    artwork_url = None
    ID3_TRUSTED_THRESHOLD = 100_000

    if ARTWORK_ENABLED and fingerprint:
        artwork_info = extract_artwork_from_file(file_path)
        id3_size = artwork_info.get('size', 0) if artwork_info else 0

        online_artwork = None
        online_size = 0
        # Candidata preferente: portada exacta del match AudD (ver path
        # no-chunked). Solo si no la hay caemos a la busqueda por texto.
        if audd_artwork:
            online_artwork = audd_artwork
            online_size = audd_artwork.get('size', 0)
        elif id3_size < ID3_TRUSTED_THRESHOLD and artist_name and title_name:
            try:
                online_artwork = search_artwork_online(
                    artist_name, title_name, id3_data.get('album')
                )
                online_size = online_artwork.get('size', 0) if online_artwork else 0
            except Exception as e:
                logger.warning(f"   Artwork online fallo: {e}")

        use_id3 = id3_size >= ID3_TRUSTED_THRESHOLD or (
            id3_size > 10000 and id3_size >= online_size
        )
        use_online = (
            online_artwork is not None
            and online_size >= 10000
            and online_size > id3_size
        )

        if use_id3 and artwork_info:
            artwork_embedded = True
            saved_filename = save_artwork_to_cache(
                fingerprint, artwork_info['data'], artwork_info['mime_type']
            )
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            _push_artwork_async(
                fingerprint, os.path.join(ARTWORK_CACHE_DIR, saved_filename)
            )
            logger.info(
                f"   Artwork ID3: {id3_size}b (online: {online_size}b)"
            )
        elif use_online and online_artwork:
            saved_filename = save_artwork_to_cache(
                fingerprint, online_artwork['data'], online_artwork['mime_type']
            )
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            _push_artwork_async(
                fingerprint, os.path.join(ARTWORK_CACHE_DIR, saved_filename)
            )
            logger.info(
                f"   Artwork online: {online_size}b (ID3: {id3_size}b)"
            )
    
    # ==================== TRACK TYPE ====================
    track_type = result['track_type']
    track_type_source = 'waveform'
    # Chunked analyzer (chunked_analyzer.py) ya devuelve confidence + alternatives
    # con el mismo shape Fase 1; si por algún motivo no estan, default a neutro.
    track_type_confidence = result.get('track_type_confidence', 0.5)
    track_type_alternatives = result.get('track_type_alternatives', [])

    # ==================== TRACK TYPE: COMMUNITY CONSENSUS (Fase 2) ====================
    # Ver flujo no-chunked para detalle. Mismo patron aqui.
    if track_type_source != 'beatport' and fingerprint:
        try:
            community = (_fetch_community_track_type(fingerprint)
                         if IS_LOCAL_ENGINE
                         else db.get_track_type_consensus(fingerprint))
            if community:
                logger.info(
                    f"  [Community] Track type consensus override: "
                    f"{track_type} -> {community['type']} ({community['votes']} votos)"
                )
                track_type = community['type']
                track_type_source = 'community'
                track_type_confidence = 1.0
                track_type_alternatives = [
                    {'type': community['type'], 'score': float(community['votes'])}
                ]
        except Exception as e:
            logger.debug(f"  [Community] Consensus check fallo: {e}")

    # ==================== GENRE / KEY: COMMUNITY CONSENSUS (Fase 4, chunked) ====================
    if fingerprint:
        if genre_source != 'beatport':
            try:
                cm_genre = (_fetch_community_override(fingerprint, 'genre')
                            if IS_LOCAL_ENGINE
                            else db.get_community_consensus(fingerprint, 'genre'))
                if cm_genre:
                    logger.info(
                        f"  [Community] Genre consensus override (chunked): "
                        f"{genre} -> {cm_genre['value']} ({cm_genre['votes']} votos)"
                    )
                    genre = cm_genre['value']
                    genre_source = 'community'
            except Exception as e:
                logger.debug(f"  [Community] Genre consensus fallo: {e}")
        if key_source != 'beatport' and key_source != 'id3':
            try:
                cm_key = (_fetch_community_override(fingerprint, 'key')
                          if IS_LOCAL_ENGINE
                          else db.get_community_consensus(fingerprint, 'key'))
                if cm_key:
                    new_key = cm_key['value']
                    new_camelot = KEY_TO_CAMELOT.get(new_key)
                    if new_camelot:
                        logger.info(
                            f"  [Community] Key consensus override (chunked): "
                            f"{key} -> {new_key} ({new_camelot})"
                        )
                        key = new_key
                        camelot = new_camelot
                        key_source = 'community'
            except Exception as e:
                logger.debug(f"  [Community] Key consensus fallo: {e}")

    # ==================== RESULTADO ====================
    # Fix 2026-05-13: mismo bug que en analyze_audio — usar title_name/
    # artist_name (enriquecidos por AudD) en vez de id3_data crudo.
    return AnalysisResult(
        title=title_name or id3_data.get('title'),
        artist=artist_name or id3_data.get('artist'),
        album=id3_data.get('album'),
        label=label,
        year=year,
        isrc=id3_data.get('isrc'),
        duration=duration,
        bpm=bpm,
        bpm_confidence=result['bpm_confidence'],
        bpm_source=bpm_source,
        key=key,
        camelot=camelot,
        key_confidence=result['key_confidence'],
        key_source=key_source,
        energy_raw=result['energy_raw'],
        energy_normalized=result['energy_normalized'],
        energy_dj=result['energy_dj'],
        groove_score=result['groove_score'],
        swing_factor=result['swing_factor'],
        has_intro=result['has_intro'],
        has_buildup=result['has_buildup'],
        has_drop=result['has_drop'],
        has_breakdown=result['has_breakdown'],
        has_outro=result['has_outro'],
        structure_sections=result['structure_sections'],
        track_type=track_type,
        track_type_source=track_type_source,
        track_type_confidence=track_type_confidence,
        track_type_alternatives=track_type_alternatives,
        genre=genre,
        genre_source=genre_source,
        has_vocals=result['has_vocals'],
        has_heavy_bass=result['has_heavy_bass'],
        has_pads=result['has_pads'],
        percussion_density=result['percussion_density'],
        mix_energy_start=result['mix_energy_start'],
        mix_energy_end=result['mix_energy_end'],
        drop_timestamp=result['drop_timestamp'],
        cue_points=result['cue_points'],
        first_beat=result['first_beat'],
        beat_interval=result['beat_interval'],
        artwork_embedded=artwork_embedded,
        artwork_url=artwork_url,
    )    

def _is_cached_analysis_valid(analysis_json: dict) -> bool:
    """
    Sanity check del payload `analysis_json` recuperado de la BD antes
    de devolverlo como `AnalysisResult` desde el cache-hit.

    Visto en produccion 2026-05-20: tracks con `analysis_json='{}'` o
    con un payload anidado raro (`{"id": "...", "analysis_json": "{}"}`)
    devolvian un dict sin los campos requeridos y `AnalysisResult(**d)`
    petaba con `ValidationError` de 19 campos missing → 500 al user.

    Solo exigimos los campos OBLIGATORIOS (sin default) del AnalysisResult de
    este modulo: `bpm` y `duration`. El resto (bpm_confidence, key_confidence,
    estructura, cues...) llevan default, asi que AnalysisResult(**analysis_json)
    reconstruye sin petar aunque falten. Exigir los *_confidence aqui (como
    hacia la version previa) rechazaba de mas las filas legitimas que llegaron
    via /cache-analysis del motor local — traen bpm/key/duration pero
    historicamente no aplanaban los confidence — disparando "[Cache] corrupto"
    + un re-analisis inutil en Render. Se valida no-None (no solo presencia)
    para cubrir el caso de un bpm/duration explicito a None.
    """
    if not isinstance(analysis_json, dict) or not analysis_json:
        return False
    required = ('bpm', 'duration')
    return all(analysis_json.get(k) is not None for k in required)


# ==================== ENDPOINTS PRINCIPALES ====================

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_track(
    request: Request,
    file: UploadFile = File(...),
    force: bool = Query(False, description="Forzar reanalisis ignorando cache"),
    force_audd: bool = Query(
        False,
        description="Forzar AudD aunque metadata sea utilizable y saltar cooldown 7d. "
                    "Implica force=true porque el cache existente ya tiene metadata "
                    "que el usuario quiere reemplazar. Daily cap se respeta.",
    ),
):
    # force_audd implica force=true: el usuario pidio explicitamente AudD y el
    # registro cacheado debe sobreescribirse con el resultado nuevo.
    if force_audd:
        force = True
    # Rate limiting — /analyze es CPU-bound (librosa) y acepta hasta 100MB,
    # por lo que es un vector de DoS trivial sin limite. Ver AUDIT 2026-04-20 B-H2.
    check_rate_limit(get_client_ip(request))

    # Obtener path original del cliente (para generacion de previews).
    # Validar para mitigar path traversal: solo aceptamos rutas absolutas
    # normalizadas; si el cliente envia algo raro lo descartamos silenciosamente.
    # Ver AUDIT 2026-04-20 B-H3.
    raw_original_path = request.headers.get("X-Original-Path", "")
    original_path = ""
    if raw_original_path:
        try:
            normalized = os.path.abspath(raw_original_path)
            # Solo usar la ruta si apunta a un fichero existente y legible.
            # La unica razon para aceptarla es que el engine local la use para
            # generar previews desde el mismo PC donde corren backend + app.
            if os.path.isfile(normalized) and os.access(normalized, os.R_OK):
                original_path = normalized
        except (OSError, ValueError):
            original_path = ""

    #  Validacin mejorada de archivo
    if not file.filename:
        raise HTTPException(400, "No se proporcion archivo")
    
    # AIFF (.aiff/.aif) es estandar en el mundo DJ Mac (Serato/Rekordbox lo
    # exportan a saco) y libsndfile lo decodifica nativo; opus/wma los abre
    # ffmpeg (fallback de robust_audio_load). Su ausencia causaba 400s en
    # reales (ej. device con biblioteca AIFF rechazada track a track).
    if not file.filename.lower().endswith(
        ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg',
         '.aiff', '.aif', '.opus', '.wma')
    ):
        raise HTTPException(
            400,
            "Formato no soportado. Permitidos: mp3, wav, flac, m4a, aac, "
            "ogg, aiff, aif, opus, wma",
        )

    # Upload streaming a disco en bloques de 1 MB. ANTES hacíamos
    # `content = await file.read()` que chupaba hasta 100 MB en RAM de golpe
    # — con 512 MB de Render + librosa + chunked_analyzer nos llevaba a OOM
    # en tracks largos. Ahora el pico en memoria durante el upload es ~1 MB.
    # El análisis posterior ya usa chunked_analyzer para tracks >4 min.
    max_size = 100 * 1024 * 1024  # 100 MB
    min_size = 1000
    total_bytes = 0
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1],
    ) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_size:
                tmp.close()
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise HTTPException(400, "Archivo demasiado grande. Máximo: 100 MB")
            tmp.write(chunk)
    if total_bytes < min_size:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(400, "Archivo demasiado pequeño o corrupto")

    # Si force=true, eliminar registro antiguo para reanalisis completo
    if force:
        db.delete_track_by_filename(file.filename)
    else:
        # Verificar si ya existe en BD por filename
        existing = db.get_track_by_filename(file.filename)
        if existing:
            analysis_json = json.loads(existing[11]) if len(existing) > 11 and existing[11] else {}

            # Defensa contra cache corrupto: si el row de BD tiene un
            # analysis_json vacio o le faltan campos minimos
            # (visto en produccion 2026-05-20: tracks con
            # analysis_json='{"id": "...", "analysis_json": "{}"}' que al
            # parsear daban un dict sin bpm/key/duration y AnalysisResult
            # petaba con ValidationError de 19 campos missing). En este
            # caso ignoramos el cache-hit y caemos al flujo de re-analisis
            # normal — sin tirar 500 al user.
            if not _is_cached_analysis_valid(analysis_json):
                logger.warning(
                    f"[Cache] Track '{file.filename}' tiene analysis_json corrupto "
                    f"(keys={list(analysis_json.keys())[:5]}). Ignorando cache y re-analizando."
                )
            else:
                # Si no tiene preview, intentar generarlo ahora
                fp = analysis_json.get('fingerprint') or (existing[13] if len(existing) > 13 else None)
                if fp:
                    preview_file = os.path.join(PREVIEWS_DIR, f"{fp}.mp3")
                    if not os.path.exists(preview_file) and original_path and os.path.exists(original_path):
                        logger.debug(f"[Preview] Cache hit pero sin snippet, generando para {fp[:8]}...")
                        try:
                            regen_path = generate_preview_snippet(
                                file_path=original_path,
                                fingerprint=fp,
                                drop_timestamp=analysis_json.get('drop_timestamp', 30.0),
                                duration=analysis_json.get('duration', 180.0),
                            )
                            if regen_path:
                                _push_preview_async(fp, regen_path)
                        except (FileNotFoundError, IOError, OSError) as e:
                            logger.error(f"[Preview] Error generando desde cache: {e}")

                    # Asegurar que el original_path se guarda para futuras peticiones
                    if original_path and not analysis_json.get('original_file_path'):
                        analysis_json['original_file_path'] = original_path
                        existing_dict = db._row_to_dict(existing) or {}
                        existing_dict['analysis_json'] = json.dumps(analysis_json)
                        # Re-save no es trivial, pero al menos guardamos el path
                # Limpiar el tmp_path creado durante el upload streaming: este
                # cache-hit no necesita el archivo subido. (No reventamos si ya
                # no existe — race improbable con otro handler.)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return AnalysisResult(**analysis_json)

    # tmp_path ya fue creado arriba por el upload streaming. Seguimos.
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        # Calcular fingerprint del archivo
        fingerprint = calculate_fingerprint(tmp_path)
        
        # NUEVO: Buscar por fingerprint si no se encontro por filename
        # Esto recupera datos de AudD guardados previamente
        if not force:
            existing_by_fp = db.get_track_by_fingerprint(fingerprint)
            if existing_by_fp and _is_analysis_current(existing_by_fp):
                # Log dedup multi-dispositivo: si un track sube desde móvil
                # con nombre distinto al que tenía en PC, este log permite
                # verificar que NO se reanaliza.
                print(
                    f"[Dedup] Match por fingerprint {fingerprint[:8]}... — "
                    f"{existing_by_fp.get('artist')} - {existing_by_fp.get('title')} "
                    f"(filename original en BD: {existing_by_fp.get('filename')!r}, "
                    f"recibido: {file.filename!r}); reutilizando análisis sin reprocesar."
                )
                
                # Actualizar el filename en la BD para futuras busquedas
                existing_by_fp['filename'] = file.filename
                db.save_track(existing_by_fp)
                
                # Intentar construir respuesta desde analysis_json
                if existing_by_fp.get('analysis_json'):
                    try:
                        analysis_data = json.loads(existing_by_fp['analysis_json'])
                        # Limpiar archivo temporal antes de retornar
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        return AnalysisResult(**analysis_data)
                    except Exception as e:
                        logger.error(f"[Cache] Error parseando analysis_json: {e}")
                
                # Si no hay analysis_json valido, construir desde los campos
                try:
                    result = AnalysisResult(
                        title=existing_by_fp.get('title'),
                        artist=existing_by_fp.get('artist'),
                        album=existing_by_fp.get('album'),
                        label=existing_by_fp.get('label'),
                        year=existing_by_fp.get('year'),
                        duration=existing_by_fp.get('duration') or 0,
                        bpm=existing_by_fp.get('bpm') or 0,
                        bpm_confidence=0.8,
                        bpm_source=existing_by_fp.get('bpm_source') or 'cached',
                        key=existing_by_fp.get('key'),
                        camelot=existing_by_fp.get('camelot'),
                        key_confidence=0.8,
                        key_source=existing_by_fp.get('key_source') or 'cached',
                        energy_raw=0,
                        energy_normalized=0,
                        energy_dj=existing_by_fp.get('energy_dj') or 5,
                        genre=existing_by_fp.get('genre') or 'Unknown',
                        genre_source=existing_by_fp.get('genre_source') or 'cached',
                        track_type=existing_by_fp.get('track_type') or 'peak',
                        artwork_url=existing_by_fp.get('artwork_url'),
                        artwork_embedded=existing_by_fp.get('artwork_embedded') or False,
                        has_intro=False,
                        has_buildup=False,
                        has_drop=False,
                        has_breakdown=False,
                        has_outro=False,
                        structure_sections=[],
                        has_vocals=False,
                        has_heavy_bass=False,
                        has_pads=False,
                        groove_score=0,
                        swing_factor=0,
                        percussion_density=0,
                        mix_energy_start=0,
                        mix_energy_end=0,
                        drop_timestamp=0,
                        cue_points=existing_by_fp.get('cue_points') or [],
                        first_beat=existing_by_fp.get('first_beat') or 0,
                        beat_interval=existing_by_fp.get('beat_interval') or 0,
                    )
                    # Limpiar archivo temporal antes de retornar
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    result.fingerprint = fingerprint
                    return result
                except Exception as e:
                    logger.error(f"[Cache] Error construyendo resultado desde cache: {e}")
                    # Continuar con analisis normal si falla
            elif IS_LOCAL_ENGINE:
                # Fallback Render: el motor local NO tiene el fingerprint
                # en su BD local. Antes de meter librosa a trabajar (segundos
                # de CPU por track), preguntamos a Render si ya existe. El
                # endpoint /analysis/by-fingerprint solo hace SELECT en BD,
                # cero CPU. Acelera reanalisis post-wipe / Mac nuevo / HDD
                # nuevo donde Render ya tiene los analisis de otros equipos
                # del mismo usuario.
                render_cached = _fetch_render_cache(fingerprint)
                if render_cached:
                    logger.info(
                        f"[Render fallback] Hit {fingerprint[:8]}... — "
                        f"{render_cached.get('artist', '?')} - "
                        f"{render_cached.get('title', '?')} "
                        f"(source={render_cached.get('bpm_source', '?')})"
                    )
                    # Guardar local con el filename actual (para futuras
                    # busquedas por filename) y devolver sin invocar librosa.
                    # No sobreescribir un analisis local si el de Render es
                    # de fuente peor (ranking "mejor gana", item 8).
                    try:
                        if not should_overwrite_analysis(existing_by_fp, render_cached):
                            logger.info(
                                f"[Render fallback] local existente es mejor, "
                                f"no se machaca con Render"
                            )
                        else:
                            to_save = dict(render_cached)
                            to_save['filename'] = file.filename
                            to_save['fingerprint'] = fingerprint
                            to_save['id'] = fingerprint
                            if 'analysis_json' not in to_save:
                                to_save['analysis_json'] = json.dumps(render_cached)
                            db.save_track(to_save)
                    except Exception as e:
                        logger.warning(f"[Render fallback] save_track fallo: {e}")
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    try:
                        return AnalysisResult(**render_cached)
                    except Exception as e:
                        logger.warning(
                            f"[Render fallback] No se pudo construir "
                            f"AnalysisResult, cayendo a librosa: {e}"
                        )

        consensus = db.get_all_consensus(fingerprint)

        # Defensa contra "tmp_path desaparecio entre creacion y analisis".
        # Falla rapido y con mensaje claro si pasa, en vez de irse al
        # exception handler de abajo con un LibsndfileError opaco.
        if not os.path.exists(tmp_path):
            logger.error(f"tmp_path {tmp_path} desaparecio antes de analyze_audio")
            raise HTTPException(500, "Archivo temporal desaparecio antes del analisis")

        # Offload del analisis CPU-bound a un threadpool para NO bloquear el
        # event loop del unico worker (acotado por _analysis_semaphore para no
        # disparar la RAM). Asi /admin/stats y demas peticiones ligeras siguen
        # respondiendo mientras se analiza. Ver "CONCURRENCIA DE ANALISIS".
        async with _get_analysis_semaphore():
            result = await run_in_threadpool(
                analyze_audio,
                tmp_path,
                fingerprint,
                force_audd=force_audd,
                original_filename=file.filename,
            )

        # Señal honesta de cap para la UI de "Limpiar con AudD" (force_audd):
        # si tras forzar AudD el track SIGUE sin metadata utilizable Y el cap
        # diario GLOBAL ya está agotado, AudD no se ejecutó por presupuesto
        # (force ya salta cooldown y garbage-check, así que el cap es lo único
        # que pudo bloquearlo). La UI lo usa para un mensaje honesto ("límite
        # diario alcanzado, inténtalo mañana") en lugar de "no identificado".
        #
        # MONETIZACIÓN (paywall futuro): este auto-trigger de AudD durante
        # /analyze pasará a ser Pro-only. La feature Escuchar (/identify, tipo
        # Shazam) seguirá siendo GRATIS para todos. Para "AudD ilimitado en
        # Pro" hay que mover el cap de GLOBAL a per-device_id (hoy
        # count_audd_calls_today() es global y lo comparten todos). Ver
        # should_trigger_audd() en audd_helper.py y CLAUDE.md > "AudD: tiers".
        if force_audd and AUDD_AUTO_ENABLED and AUDD_API_TOKEN:
            try:
                from audd_helper import is_garbage_metadata
                if (is_garbage_metadata(result.artist, result.title)
                        and db is not None
                        and db.count_audd_calls_today() >= AUDD_DAILY_CAP):
                    result.audd_daily_cap_reached = True
            except Exception:
                pass

        # Parsear nombre si falta metadata
        if not result.artist or not result.title:
            parsed = parse_filename(file.filename)
            if not result.artist and parsed['artist']:
                result.artist = parsed['artist']
            if not result.title:
                result.title = parsed['title']

        # Intentar AcousticBrainz (SOLO si no tenemos Discogs/MusicBrainz)
        ab_genre = None
        if result.genre_source not in ["discogs", "musicbrainz"]:
            ab_genre = get_acousticbrainz_genre(
                fingerprint=fingerprint,
                artist=result.artist,
                title=result.title
            )

        # ================================================================
        # PRIORIDAD DE GENERO:
        #   Consenso >=3 > Discogs > MusicBrainz > ID3 > AcousticBrainz > Analisis
        # ================================================================
        genre_consensus = consensus.get('genre')
        if genre_consensus and genre_consensus[1] >= 3:
            result.genre = genre_consensus[0]
            result.genre_source = f"consensus_{genre_consensus[1]}"
        elif result.genre_source in ["discogs", "musicbrainz"]:
            pass
        elif genre_consensus and genre_consensus[1] >= 2:
            result.genre = genre_consensus[0]
            result.genre_source = f"suggestion_{genre_consensus[1]}"
        elif result.genre_source == "id3":
            if ab_genre and ab_genre.lower() not in ["electronic", "dance"]:
                result.subgenre = result.genre
                result.genre = ab_genre
                result.genre_source = "acousticbrainz"
        elif ab_genre:
            result.genre = ab_genre
            result.genre_source = "acousticbrainz"

        # ================================================================
        # CONSENSO PROGRESIVO PARA BPM, KEY, ENERGY
        # ================================================================
        # 1 usuario  = se guarda, no se aplica
        # 2 usuarios = sugerencia (no sobreescribe Rekordbox/Beatport)
        # 3+ usuarios = se aplica automaticamente
        # 5+ usuarios = sobreescribe todo incluido Rekordbox/Beatport
        # ================================================================

        bpm_consensus = consensus.get('bpm')
        if bpm_consensus:
            votes = bpm_consensus[1]
            try:
                consensus_bpm = float(bpm_consensus[0])
                if votes >= 5:
                    result.bpm = consensus_bpm
                    result.bpm_source = f"consensus_{votes}"
                elif votes >= 3 and result.bpm_source not in ["rekordbox", "traktor", "beatport"]:
                    result.bpm = consensus_bpm
                    result.bpm_source = f"consensus_{votes}"
            except (ValueError, TypeError):
                pass

        key_consensus = consensus.get('key')
        if key_consensus:
            votes = key_consensus[1]
            if votes >= 5:
                result.key = key_consensus[0]
                result.key_source = f"consensus_{votes}"
            elif votes >= 3 and result.key_source not in ["rekordbox", "traktor"]:
                result.key = key_consensus[0]
                result.key_source = f"consensus_{votes}"

        energy_consensus = consensus.get('energy')
        if energy_consensus:
            votes = energy_consensus[1]
            try:
                consensus_energy = float(energy_consensus[0])
                if votes >= 3:
                    result.energy_dj = int(consensus_energy)
                    result.energy_source = f"consensus_{votes}"
            except (ValueError, TypeError):
                pass
        
        # Guardar en BD. engine_source distingue render vs local_engine
        # para que el panel admin pueda comparar latencias y cobertura por
        # motor. Se marca aqui porque /analyze es la unica via de creacion
        # de tracks "analizados" (vs los del fallback de mas abajo, que son
        # tracks pendientes y no representan analisis reales).
        track_data = result.model_dump()
        track_data['id'] = fingerprint
        track_data['filename'] = file.filename
        track_data['fingerprint'] = fingerprint
        track_data['engine_source'] = 'local_engine' if IS_LOCAL_ENGINE else 'render'
        track_data['analysis_version'] = ANALYSIS_VERSION
        db.save_track(track_data)

        # Incrementar contador de popularidad
        try:
            db.increment_popularity(fingerprint)
        except Exception:
            pass

        # Generar preview snippet (no bloquea si falla).
        # Logueamos fallos en analysis_errors con endpoint='preview' para
        # que el panel admin pueda contar la tasa de fallo del generador.
        try:
            preview_path = generate_preview_snippet(
                file_path=tmp_path,
                fingerprint=fingerprint,
                drop_timestamp=result.drop_timestamp,
                duration=result.duration,
            )
            if preview_path:
                result.preview_url = f"{BASE_URL}/preview/{fingerprint}"
                # Si somos engine local, empuja el snippet al Render remoto
                # para que otros dispositivos puedan reproducirlo al instante.
                _push_preview_async(fingerprint, preview_path)
        except Exception as preview_err:
            logger.error(f"  [Preview] Error (no crítico): {preview_err}")
            try:
                import traceback as _tb
                db.log_analysis_error(
                    device_id=None,
                    filename=file.filename,
                    fingerprint=fingerprint,
                    error_class=type(preview_err).__name__,
                    error_msg=str(preview_err),
                    traceback_str=_tb.format_exc(),
                    endpoint='preview',
                )
            except Exception:
                pass

        # Auto-upload a Render como cache comunitario (solo en modo local)
        if IS_LOCAL_ENGINE:
            _upload_to_render_cache(track_data)

        result.fingerprint = fingerprint
        return result
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()

        # Distinguir "audio genuinamente ilegible" (problema de DATOS del
        # cliente, no bug del backend) de un error real accionable. Cuando
        # librosa Y el fallback ffmpeg fallan, robust_audio_load propaga el
        # CalledProcessError de ffmpeg: el fichero subido esta corrupto/vacio
        # o con un codec que ffmpeg no abre. No hay nada que arreglar en el
        # servidor — antes esto ensuciaba el panel admin (ej. 18x desde 1
        # device con ficheros rotos), tratandolo como un fallo del backend.
        # Mismo criterio que los timeouts de AudD: WARNING y NO se registra en
        # la telemetria de errores. El fallback de abajo igualmente devuelve un
        # resultado degradado (pending) + 200, asi que el track aparece con la
        # metadata del filename.
        is_corrupt_audio = isinstance(e, subprocess.CalledProcessError)

        if is_corrupt_audio:
            logger.warning(
                f"[decode] audio ilegible — cliente subio fichero corrupto: "
                f"{file.filename} ({type(e).__name__})"
            )
        else:
            # exc_info=True asegura que handlers que descartan \n en el message
            # (uvicorn default formatter) sigan recibiendo el traceback via el
            # campo exc_info del LogRecord. Sin esto, en Render solo veiamos
            # "ERROR en anlisis de audio:" sin contexto. Ademas pasamos el
            # tipo y mensaje en el header del log para verlo de un vistazo.
            logger.error(
                f"ERROR en anlisis de audio: {type(e).__name__}: {e}",
                exc_info=True,
            )

            # Telemetria privacy-first: registra el error en analysis_errors
            # para que el panel admin lo muestre. Filename se hashea dentro
            # de log_analysis_error. device_id viene del header X-Device-Id
            # cuando el cliente lo manda (cliente desktop lo hace en sync;
            # /analyze raw no siempre lo incluye).
            try:
                db.log_analysis_error(
                    device_id=request.headers.get("X-Device-Id"),
                    filename=file.filename,
                    fingerprint=None,
                    error_class=type(e).__name__,
                    error_msg=str(e),
                    traceback_str=error_detail,
                    endpoint='/analyze',
                )
            except Exception:
                # No bloquear el flow si la tabla aun no esta migrada.
                pass

        # ==================== FALLBACK: Track corrupto ====================
        # Intentar crear resultado bsico con ID3 y/o filename
        logger.info(f" Intentando fallback para: {file.filename}")

        try:
            # Intentar fingerprint del contenido primero
            try:
                fingerprint = calculate_fingerprint(tmp_path)
            except Exception:
                # Si falla (archivo muy corrupto), usar md5 del nombre
                fingerprint = hashlib.md5(file.filename.encode()).hexdigest()
            
            # Intentar leer metadatos ID3 aunque el audio est(c) corrupto
            id3_data = {}
            if ARTWORK_ENABLED:
                try:
                    id3_data = extract_id3_metadata(tmp_path)
                except Exception as e:
                    logger.warning("ID3 extract fallo en %s: %s", file.filename, e)
            
            # Parsear filename
            parsed = parse_filename(file.filename)
            
            artist = id3_data.get('artist') or parsed.get('artist') or 'Artista Desconocido'
            title = id3_data.get('title') or parsed.get('title') or file.filename
            
            # Crear resultado mnimo marcado como "pendiente"
            result = AnalysisResult(
                title=title,
                artist=artist,
                album=id3_data.get('album'),
                duration=0,
                bpm=0,
                bpm_confidence=0,
                bpm_source="pending",  # Marcado como pendiente
                key=None,
                camelot=None,
                key_confidence=0,
                key_source="pending",
                energy_raw=0,
                energy_normalized=0,
                energy_dj=5,  # Valor medio por defecto
                genre=id3_data.get('genre') or 'Unknown',
                genre_source="pending",
                track_type="unknown",
                has_intro=False,
                has_buildup=False,
                has_drop=False,
                has_breakdown=False,
                has_outro=False,
                structure_sections=[],
                has_vocals=False,
                has_heavy_bass=False,
                has_pads=False,
                groove_score=0,
                swing_factor=0,
                percussion_density=0,
                mix_energy_start=0,
                mix_energy_end=0,
                drop_timestamp=0,
                cue_points=[],
                first_beat=0,
                beat_interval=0,
                artwork_embedded=False,
                artwork_url=None,
            )
            
            # Guardar en BD (marcado como pendiente de anlisis real)
            track_data = result.model_dump()
            track_data['id'] = fingerprint
            track_data['filename'] = file.filename
            track_data['fingerprint'] = fingerprint
            track_data['analysis_status'] = 'failed'  # Marcador especial
            db.save_track(track_data)
            
            logger.info(f"Fallback creado: {artist} - {title} (anlisis pendiente)")
            
            result.fingerprint = fingerprint
            return result
            
        except Exception as fallback_error:
            logger.error(f"Fallback tambi(c)n fall: {fallback_error}")
            raise HTTPException(500, f"Error analizando: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/correction")
async def save_correction(request: CorrectionRequest):
    #  Validar campos
    track_id = validate_track_id(request.track_id)
    field = sanitize_string(request.field, max_length=50, allow_empty=False, field_name="field")
    old_value = sanitize_string(request.old_value, max_length=200, field_name="old_value")
    new_value = sanitize_string(request.new_value, max_length=200, allow_empty=False, field_name="new_value")
    
    # Validar que el campo sea uno permitido
    allowed_fields = {'genre', 'bpm', 'key', 'camelot', 'energy', 'artist', 'title', 'label', 'track_type'}
    if field not in allowed_fields:
        raise HTTPException(400, f"Campo no permitido: {field}. Permitidos: {', '.join(allowed_fields)}")
    
    db.save_correction(track_id, field, old_value, new_value, request.fingerprint, request.device_id)

    votes = 0
    status = "saved"
    if request.fingerprint:
        _, votes = db.get_consensus(request.fingerprint, field)
        if votes >= 5:
            status = "applied_override"
        elif votes >= 3:
            status = "applied"
        elif votes >= 2:
            status = "suggestion"
        else:
            status = "saved"

    return {
        "status": "ok",
        "consensus_status": status,
        "votes": votes,
        "message": f"Correccion guardada ({votes} votos, {status})",
    }

# ==================== IDENTIFICAR TRACK CON AUDD ====================

@app.post("/identify")
async def identify_track(request: Request, file: UploadFile = File(...)):
    """
    Identifica un track usando AudD y hace RE-ANALISIS COMPLETO.

    Flujo:
    1. AudD identifica artista/ttulo
    2. Busca g(c)nero en Discogs con el nuevo nombre
    3. Intenta re-analizar audio (BPM, Key, Energy)
    4. Busca artwork online
    5. Actualiza todo en BD
    """
    # Rate limiting — endpoint caro (AudD + Discogs + re-analisis).
    check_rate_limit(get_client_ip(request))

    try:
        from api_config import AUDD_API_TOKEN
    except ImportError:
        AUDD_API_TOKEN = None

    if not AUDD_API_TOKEN:
        raise HTTPException(500, "AudD API token no configurado")
    
    tmp_path = None
    fragment_path = None
    
    try:
        # Upload streaming a disco en bloques de 1 MB (mismo patrón que /analyze)
        # para no cargar el archivo entero en RAM de Render.
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)

        logger.info(f"Identificando track: {file.filename}")

        # Calcular fingerprint del CONTENIDO del archivo (igual que en /analyze)
        fingerprint = calculate_fingerprint(tmp_path)
        logger.info(f"  Fingerprint (contenido): {fingerprint[:12]}...")
        
        # ==================== PASO 1: IDENTIFICAR CON AUDD ====================
        audio_to_send = tmp_path
        
        try:
            with silence_native_stderr():
                y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=20, offset=30)
            import soundfile as sf
            fragment_path = tmp_path + "_fragment.wav"
            sf.write(fragment_path, y, sr)
            audio_to_send = fragment_path
            logger.info(f"Fragmento extrado: 20 seg desde 0:30")
        except Exception as e:
            logger.info(f"No se pudo extraer fragmento, usando archivo completo: {e}")
        
        with open(audio_to_send, 'rb') as audio_file:
            audd_response = requests.post(
                'https://api.audd.io/',
                data={
                    'api_token': AUDD_API_TOKEN,
                    'return': 'spotify,deezer,apple_music,musicbrainz',
                },
                files={'file': audio_file},
                timeout=30
            )
        
        if audd_response.status_code != 200:
            raise HTTPException(500, f"Error AudD: {audd_response.status_code}")
        
        result = audd_response.json()
        
        if result.get('status') != 'success':
            error_msg = result.get('error', {}).get('error_message', 'Unknown')
            return {"status": "error", "message": error_msg}
        
        track_data = result.get('result')
        
        if not track_data:
            return {"status": "not_found", "message": "No se pudo identificar"}
        
        # Extraer datos de AudD
        artist = track_data.get('artist', '')
        title = track_data.get('title', '')
        album = track_data.get('album')
        label = track_data.get('label')
        release_date = track_data.get('release_date')
        year = release_date[:4] if release_date and len(release_date) >= 4 else None
        
        logger.info(f"  AudD identifico: {artist} - {title}")
        
        # ==================== PASO 2: BUSCAR GENERO EN DISCOGS ====================
        genre = "Electronic"
        genre_source = "default"
        
        if GENRE_DETECTOR_ENABLED and genre_detector and artist and title:
            logger.info(f"Buscando g(c)nero: {artist} - {title}")
            try:
                discogs_result = genre_detector.get_discogs_genre(artist, title)
                if discogs_result and discogs_result.get('genre'):
                    genre = discogs_result['genre']
                    genre_source = "discogs"
                    if not label and discogs_result.get('label'):
                        label = discogs_result['label']
                    if not year and discogs_result.get('year'):
                        year = str(discogs_result['year'])
                    logger.info(f"Discogs: {genre} | {label} ({year})")
            except Exception as e:
                logger.error(f"Error Discogs: {e}")
            
            if genre_source != "discogs":
                try:
                    mb_result = genre_detector.get_musicbrainz_info(artist, title)
                    if mb_result and mb_result.get('genre'):
                        genre = mb_result['genre']
                        genre_source = "musicbrainz"
                        logger.info(f"MusicBrainz: {genre}")
                except Exception as e:
                    logger.error(f"Error MusicBrainz: {e}")
        
        # ==================== PASO 3: RE-ANALIZAR AUDIO ====================
        bpm = None
        bpm_confidence = 0.0
        key = None
        camelot = None
        energy_dj = 5
        duration = 0.0
        bpm_source = 'pending'
        key_source = 'pending'
        logger.info(f"Re-analizando audio...")
        try:
            # Local: sr=44100 para maxima precision. Render: sr=22050 para ahorrar RAM.
            analysis_sr = 44100 if IS_LOCAL_ENGINE else 22050
            with silence_native_stderr():
                y_full, sr_full = librosa.load(tmp_path, sr=analysis_sr, mono=True)
            duration = librosa.get_duration(y=y_full, sr=sr_full)
            
            # BPM
            tempo, beat_frames = librosa.beat.beat_track(y=y_full, sr=sr_full)
            if hasattr(tempo, '__iter__'):
                tempo = float(tempo[0])
            bpm = round(tempo, 1)
            bpm_source = 'analysis'
            
            # BPM confidence
            beat_intervals = np.diff(librosa.frames_to_time(beat_frames, sr=sr_full))
            bpm_confidence = 1.0 - min(np.std(beat_intervals) * 2, 0.5) if len(beat_intervals) > 0 else 0.5
            
            # MEJORA 3: Auto-correccion half/double
            onset_env_re = librosa.onset.onset_strength(y=y_full, sr=sr_full)
            bpm = try_bpm_double_half(y_full, sr_full, bpm, bpm_confidence, onset_env=onset_env_re)
            logger.info(f"BPM: {bpm} (confianza: {bpm_confidence:.2f})")
            
            # MEJORA 1: Key con Krumhansl-Kessler (mismo codigo que analisis principal)
            chroma = librosa.feature.chroma_cqt(y=y_full, sr=sr_full, n_chroma=12, n_octaves=7)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            major_profile_re = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile_re = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            major_profile_re = major_profile_re / np.sum(major_profile_re)
            minor_profile_re = minor_profile_re / np.sum(minor_profile_re)
            
            chroma_mean_re = np.mean(chroma, axis=1)
            chroma_mean_re = chroma_mean_re / (np.sum(chroma_mean_re) + 1e-10)
            
            best_key_re = 'C'
            best_corr_re = -1
            best_scale_re = 'minor'
            
            for i, kn in enumerate(key_names):
                mc = np.corrcoef(chroma_mean_re, np.roll(major_profile_re, i))[0, 1]
                nc = np.corrcoef(chroma_mean_re, np.roll(minor_profile_re, i))[0, 1]
                if not np.isnan(mc) and mc > best_corr_re:
                    best_corr_re, best_key_re, best_scale_re = mc, kn, 'major'
                if not np.isnan(nc) and nc > best_corr_re:
                    best_corr_re, best_key_re, best_scale_re = nc, kn, 'minor'
            
            key = best_key_re + ('m' if best_scale_re == 'minor' else '')
            camelot = get_camelot(key)
            key_source = 'analysis'
            logger.info(f"Key: {key} ({camelot})")
            
            # MEJORA 2: Energy con power curve
            rms = librosa.feature.rms(y=y_full)[0]
            avg_rms = float(np.mean(rms))
            if avg_rms <= 0.02:
                energy_dj = 1
            elif avg_rms >= 0.42:
                energy_dj = 10
            elif not math.isfinite(avg_rms):
                # Mismo guard NaN/Inf que en analyze_audio (linea 1266).
                # Defensivo aqui aunque el except externo ya lo come — asi
                # no perdemos el track al fallback de BD colectiva.
                energy_dj = 5
            else:
                normalized = (avg_rms - 0.02) / (0.42 - 0.02)
                powered = normalized ** 0.55
                energy_dj = int(round(1 + powered * 9))
                energy_dj = max(1, min(10, energy_dj))
            logger.info(f"Energy: {energy_dj} (raw: {avg_rms:.4f})")
            
        except Exception as e:
            logger.error(f"Re-anlisis fall: {e}")
            # FALLBACK 1: Buscar en BD colectiva
            if artist and title:
                logger.info(f"Buscando en BD colectiva...")
                collective_data = search_collective_db(artist, title)
                if collective_data:
                    if collective_data.get('bpm'):
                        bpm = collective_data['bpm']
                        bpm_confidence = 0.9
                        bpm_source = 'collective'
                        logger.info(f"BD Colectiva BPM: {bpm}")
                    if collective_data.get('key'):
                        key = collective_data['key']
                        camelot = collective_data.get('camelot') or get_camelot(key)
                        key_source = 'collective'
                        logger.info(f"BD Colectiva Key: {key} ({camelot})")
                    if collective_data.get('duration') and collective_data['duration'] > 0:
                        duration = collective_data['duration']
                else:
                    logger.info(f"No encontrado en BD colectiva")

        # ==================== PASO 4: BUSCAR ARTWORK ====================
        artwork_url = None
        artwork_source = None
        
        if artist and title and search_artwork_online:
            logger.info(f"   Buscando artwork...")
            artwork_info = search_artwork_online(artist, title)
            if artwork_info:
                saved_filename = save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
                artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                artwork_source = artwork_info.get('source', 'online')
                _push_artwork_async(fingerprint, os.path.join(ARTWORK_CACHE_DIR, saved_filename))
                logger.info(f"Artwork: {artwork_source} ({artwork_info['size']} bytes)")
            else:
                logger.info(f"No se encontr artwork")
        elif not search_artwork_online:
            logger.info(f"search_artwork_online no disponible")
        
        # ==================== PASO 5: ACTUALIZAR BD ====================
        track_db_data = {
            'id': fingerprint,
            'filename': file.filename,
            'fingerprint': fingerprint,
            'title': title,
            'artist': artist,
            'album': album,
            'bpm': bpm,
            'bpm_confidence': bpm_confidence,
            'bpm_source': bpm_source,
            'key': key,
            'camelot': camelot,
            'key_confidence': 0.8 if key else 0,
            'key_source': key_source,
            'energy_raw': 0,
            'energy_normalized': 0,
            'energy_dj': energy_dj,
            'duration': duration,
            'genre': genre,
            'genre_source': genre_source,
            'label': label,
            'year': year,
            'isrc': None,
            'artwork_url': artwork_url,
            'artwork_embedded': False,
            'track_type': 'peak_time',
            'track_type_source': 'waveform',
            # Campos necesarios para que AnalysisResult funcione al recuperar
            'has_intro': False,
            'has_buildup': False,
            'has_drop': False,
            'has_breakdown': False,
            'has_outro': False,
            'structure_sections': [],
            'has_vocals': False,
            'has_heavy_bass': False,
            'has_pads': False,
            'groove_score': 0,
            'swing_factor': 0,
            'percussion_density': 0,
            'mix_energy_start': 0,
            'mix_energy_end': 0,
            'drop_timestamp': 0,
            'cue_points': [],
            'first_beat': 0,
            'beat_interval': 0,
        }
        
        db.save_track(track_db_data)
        logger.info(f"  Guardado en BD con fingerprint: {fingerprint[:12]}...")
        
        # ==================== RESPUESTA ====================
        return {
            "status": "found",
            "artist": artist,
            "title": title,
            "album": album,
            "label": label,
            "year": year,
            "genre": genre,
            "genre_source": genre_source,
            "bpm": bpm,
            "bpm_source": bpm_source,
            "key": key,
            "key_source": key_source,
            "camelot": camelot,
            "energy_dj": energy_dj,
            "duration": duration,
            "artwork_url": artwork_url,
            "artwork_source": artwork_source,
            "filename": file.filename,
            "reanalyzed": True,
        }
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Error identificando: {tb}")
        # Persistir en analysis_errors: /identify relanza como HTTPException(500),
        # que el middleware global ignora a proposito (solo loguea 500 NO
        # manejados). Sin esto, un fallo real de /identify solo salia en los
        # logs de Render y era invisible para el panel admin.
        try:
            db.log_analysis_error(
                device_id=request.headers.get('X-Device-Id'),
                filename=getattr(file, 'filename', None),
                fingerprint=None,
                error_class=type(e).__name__,
                error_msg=str(e),
                traceback_str=tb,
                endpoint='/identify',
            )
        except Exception:
            pass
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if fragment_path and os.path.exists(fragment_path):
            os.unlink(fragment_path)

# ==================== RECONOCIMIENTO DE AUDIO (SHAZAM-LIKE) ====================

def _preprocess_audio_for_recognition(input_path: str, output_path: str, strategy: str = "normalize") -> bool:
    """
    Preprocesa audio para mejorar la detección por AudD.
    Convierte a WAV 44.1kHz mono y aplica filtros según la estrategia.

    Estrategias:
    - "normalize": Normalización de volumen + high-pass filter (elimina ruido grave ambiente)
    - "aggressive": Normalización agresiva + filtro de ruido más fuerte + compresión
    - "raw_wav": Solo conversión a WAV sin procesamiento (fallback)
    """
    import subprocess
    try:
        # Ver comentario en generate_preview_snippet sobre FFMPEG_BIN.
        ffmpeg_bin = os.environ.get('FFMPEG_BIN', 'ffmpeg')
        if strategy == "normalize":
            # Normalizar volumen + filtro high-pass a 80Hz (elimina ruido de ambiente/aire acondicionado)
            # + limitar frecuencias altas innecesarias para fingerprinting
            cmd = [
                ffmpeg_bin, '-y', '-i', input_path,
                '-af', 'highpass=f=80,lowpass=f=16000,loudnorm=I=-16:TP=-1.5:LRA=11,silenceremove=start_periods=1:start_silence=0.5:start_threshold=-40dB:stop_periods=-1:stop_silence=0.3:stop_threshold=-40dB',
                '-ar', '44100', '-ac', '1',
                '-acodec', 'pcm_s16le',
                output_path
            ]
        elif strategy == "aggressive":
            # Filtrado más agresivo: banda 200-8000Hz (rango vocal/melódico principal),
            # compresión dinámica fuerte para igualar volúmenes, normalización
            cmd = [
                ffmpeg_bin, '-y', '-i', input_path,
                '-af', 'highpass=f=200,lowpass=f=8000,acompressor=threshold=-20dB:ratio=6:attack=5:release=50,loudnorm=I=-14:TP=-1:LRA=7,silenceremove=start_periods=1:start_silence=0.3:start_threshold=-35dB:stop_periods=-1:stop_silence=0.2:stop_threshold=-35dB',
                '-ar', '44100', '-ac', '1',
                '-acodec', 'pcm_s16le',
                output_path
            ]
        else:  # raw_wav
            # Solo convertir a WAV sin procesamiento
            cmd = [
                ffmpeg_bin, '-y', '-i', input_path,
                '-ar', '44100', '-ac', '1',
                '-acodec', 'pcm_s16le',
                output_path
            ]

        run_kwargs = {
            'capture_output': True, 'timeout': 15,
        }
        if os.name == 'nt':
            run_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(cmd, **run_kwargs)

        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='replace')
            # stderr[:200] solo recoge el banner de version de ffmpeg. Nos
            # quedamos con las lineas que parecen errores reales o el tail.
            err_lines = [ln for ln in stderr.splitlines() if ln and (
                ln.lower().startswith('error') or 'invalid' in ln.lower()
                or 'no such file' in ln.lower() or 'failed' in ln.lower()
            )]
            real_err = ' | '.join(err_lines[-3:])[:400] if err_lines else stderr[-400:].strip()
            logger.error(f"  [Recognize] ffmpeg {strategy} exit {result.returncode}: {real_err or 'unknown'}")
            return False

        # Verificar que el archivo resultante tiene contenido útil
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return True

        logger.info(f"  [Recognize] ffmpeg {strategy}: archivo resultante muy pequeño o vacío")
        return False

    except subprocess.TimeoutExpired:
        logger.info(f"  [Recognize] ffmpeg {strategy} timeout")
        return False
    except Exception as e:
        logger.error(f"  [Recognize] ffmpeg {strategy} error: {e}")
        return False


def _send_to_audd(audio_path: str, api_token: str, timeout: int = 30) -> Optional[dict]:
    """
    Envía audio a AudD y devuelve track_data si se identifica, None si no.
    """
    with open(audio_path, 'rb') as audio_file:
        audd_response = requests.post(
            'https://api.audd.io/',
            data={
                'api_token': api_token,
                'return': 'spotify,deezer,apple_music,musicbrainz',
            },
            files={'file': audio_file},
            timeout=timeout
        )

    if audd_response.status_code != 200:
        logger.error(f"  [AudD] HTTP error: {audd_response.status_code}")
        return None

    result = audd_response.json()

    if result.get('status') != 'success':
        error_msg = result.get('error', {}).get('error_message', 'Unknown error')
        # AudD devuelve mensajes largos (~500 chars con links y FAQ).
        # Truncamos para no inundar logs. Bajamos a warning porque el
        # caso comun es "audio invalido / silencioso / muy corto" que no
        # es un bug nuestro y tenemos fallback con preprocesamiento.
        short_msg = error_msg.split('.')[0][:140] if error_msg else 'unknown'
        logger.warning(f"  [AudD] API rechazo audio: {short_msg}")
        return None

    track_data = result.get('result')
    if track_data:
        logger.info(f"  [AudD] ✓ Identificado: {track_data.get('artist')} - {track_data.get('title')}")

    return track_data


@app.post("/recognize")
async def recognize_audio(request: Request, file: UploadFile = File(...)):
    """
    Reconoce una canción a partir de audio grabado usando AudD API.
    Preprocesa el audio (normalización, filtrado de ruido) y reintenta
    con diferentes estrategias si el primer intento falla.
    """
    # Rate limiting — endpoint caro (preprocesado + AudD retries).
    check_rate_limit(get_client_ip(request))

    try:
        from api_config import AUDD_API_TOKEN
    except ImportError:
        AUDD_API_TOKEN = None

    if not AUDD_API_TOKEN:
        raise HTTPException(500, "AudD API token no configurado en api_config.py")

    tmp_path = None
    processed_paths = []
    try:
        # Upload streaming a disco (1 MB por bloque) para no saturar RAM.
        total_bytes = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                tmp.write(chunk)

        logger.info(f"[Recognize] Audio recibido: {file.filename} ({total_bytes} bytes)")

        if total_bytes < 1000:
            return {"status": "not_found", "message": "Audio demasiado corto o vacío"}

        # ── Estrategia multi-intento ──
        # Intento 1: Audio normalizado + filtro de ruido ambiente
        # Intento 2: Filtrado agresivo (banda vocal/melódica) + compresión
        # Intento 3: WAV sin procesar (por si los filtros eliminaron info útil)
        strategies = ["normalize", "aggressive", "raw_wav"]
        track_data = None

        for i, strategy in enumerate(strategies):
            processed_path = tmp_path.replace('.m4a', f'_processed_{strategy}.wav')
            processed_paths.append(processed_path)

            logger.info(f"[Recognize] Intento {i+1}/3 - estrategia: {strategy}")

            if _preprocess_audio_for_recognition(tmp_path, processed_path, strategy):
                processed_size = os.path.getsize(processed_path)
                logger.info(f"  Procesado: {processed_size} bytes")

                track_data = _send_to_audd(processed_path, AUDD_API_TOKEN, timeout=30)
                if track_data:
                    logger.info(f"[Recognize] ✓ Éxito en intento {i+1} ({strategy})")
                    break
                else:
                    logger.info(f"  Intento {i+1} ({strategy}): no identificado")
            else:
                logger.info(f"  Intento {i+1} ({strategy}): preprocesamiento falló")
                # Si normalize falla, intentar enviar el original directamente
                if strategy == "normalize":
                    logger.info(f"  Intentando enviar archivo original sin procesar...")
                    track_data = _send_to_audd(tmp_path, AUDD_API_TOKEN, timeout=30)
                    if track_data:
                        logger.info(f"[Recognize] ✓ Éxito con archivo original")
                        break

        if not track_data:
            logger.info("[Recognize] ✗ No identificado tras todos los intentos")
            return {"status": "not_found", "message": "No se pudo identificar la canción"}

        # ── Extraer datos del resultado ──
        artist = track_data.get('artist', 'Unknown Artist')
        title = track_data.get('title', 'Unknown Title')
        album = track_data.get('album')
        release_date = track_data.get('release_date')
        label = track_data.get('label')
        isrc = track_data.get('isrc')

        spotify_data = track_data.get('spotify')
        deezer_data = track_data.get('deezer')
        apple_music_data = track_data.get('apple_music')

        logger.info(f"[Recognize] Resultado: {artist} - {title}")

        # ── Buscar análisis existente en BD ──
        backend_analysis = None
        existing_tracks = db.search_by_artist(artist, limit=50)
        for track in existing_tracks:
            if track.get('title', '').lower() == title.lower():
                backend_analysis = track
                logger.info(f"  Encontrado en biblioteca: {track.get('id')}")
                break

        response = {
            "status": "found",
            "artist": artist,
            "title": title,
            "album": album,
            "release_date": release_date,
            "label": label,
            "isrc": isrc,
            "spotify": spotify_data,
            "deezer": deezer_data,
            "apple_music": apple_music_data,
        }

        if backend_analysis:
            response["backend_analysis"] = backend_analysis

        # Guardar reconocimiento en BD colectiva para enriquecer futuras consultas
        if not backend_analysis and artist and title:
            try:
                artwork_url = None
                if search_artwork_online:
                    artwork_info = search_artwork_online(artist, title)
                    if artwork_info:
                        detect_id = hashlib.md5(f"{artist.lower().strip()}|{title.lower().strip()}".encode()).hexdigest()
                        save_artwork_to_cache(detect_id, artwork_info['data'], artwork_info['mime_type'])
                        artwork_url = f"{BASE_URL}/artwork/{detect_id}"
                        response["artwork_url"] = artwork_url
                        logger.info(f"Artwork guardado para deteccion: {detect_id[:12]}")

                detect_id = hashlib.md5(f"{artist.lower().strip()}|{title.lower().strip()}".encode()).hexdigest()
                detect_data = {
                    'id': detect_id,
                    'filename': f"{artist} - {title}",
                    'fingerprint': detect_id,
                    'title': title,
                    'artist': artist,
                    'album': album,
                    'label': label,
                    'duration': 0,
                    'bpm': 0,
                    'key': None,
                    'camelot': None,
                    'energy_dj': 5,
                    'genre': 'Electronic',
                    'track_type': 'peak_time',
                    'bpm_source': 'pending',
                    'key_source': 'pending',
                }
                existing = db.get_track_by_fingerprint(detect_id)
                if not existing:
                    db.save_track(detect_data)
                    logger.info(f"Deteccion guardada en BD colectiva: {detect_id[:12]}")
            except Exception as e:
                logger.error(f"Error guardando deteccion en BD: {e}")

        return response

    except requests.Timeout:
        logger.info("[Recognize] AudD timeout")
        raise HTTPException(504, "Timeout conectando con AudD")
    except Exception as e:
        import traceback
        logger.error(f"[Recognize] Error: {traceback.format_exc()}")
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        # Limpiar todos los archivos temporales
        for path in [tmp_path] + processed_paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

# ==================== CACHE COMUNITARIO (LOCAL ENGINE → RENDER) ====================

@app.post("/cache-analysis")
async def cache_analysis(request: Request):
    """
    Recibe resultado de análisis desde motor local y lo guarda como cache comunitario.
    Esto permite que tracks analizados localmente estén disponibles para todos los usuarios
    en las búsquedas por artist/title (e.g. desde /recognize o /search-analyzed).

    El motor local envía esto automáticamente tras cada análisis exitoso.
    No sobreescribe datos existentes con mejor calidad.
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "JSON inválido")

    fingerprint = data.get('fingerprint')
    if not fingerprint:
        raise HTTPException(400, "fingerprint requerido")

    # Ranking "mejor gana": comparamos bpm_source nuevo vs existente y
    # solo sobreescribimos si la nueva fuente tiene mayor prioridad (o
    # empate con existente vacio). Ver ANALYSIS_SOURCE_PRIORITY arriba.
    existing = db.get_track_by_fingerprint(fingerprint)
    if existing and not should_overwrite_analysis(existing, data):
        existing_source = '?'
        try:
            ej_raw = existing.get('analysis_json')
            if ej_raw:
                ej = json.loads(ej_raw) if isinstance(ej_raw, str) else ej_raw
                existing_source = ej.get('bpm_source', '?')
        except (json.JSONDecodeError, TypeError):
            pass
        logger.info(
            f"[Cache] {fingerprint[:12]} skip (existente={existing_source} "
            f"prio={get_source_priority(existing_source)} "
            f">= nuevo={data.get('bpm_source', '?')} "
            f"prio={get_source_priority(data.get('bpm_source'))})"
        )
        return {"status": "exists", "fingerprint": fingerprint}

    # Detalle completo que el motor local manda anidado en `analysis_json`.
    # Lo aplanamos al top-level para que la fila guardada sea tan RICA como una
    # de /analyze directo. Antes se guardaba el nested como string (en la
    # practica "{}" por el bug del productor) y la columna quedaba "fina" — sin
    # bpm_confidence/key_confidence/structure_sections/cue_points/beat-grid — lo
    # que disparaba "[Cache] corrupto" + re-analisis cuando el track se re-subia
    # por filename desde otro dispositivo.
    nested = data.get('analysis_json')
    if isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except (json.JSONDecodeError, TypeError):
            nested = {}
    if not isinstance(nested, dict):
        nested = {}

    # Construir datos para guardar: base = detalle completo anidado; encima los
    # campos resumen explicitos del payload (autoritativos: pueden venir ya
    # limpiados por AudD / corregidos por la comunidad). save_track serializa el
    # track_data ENTERO como la columna analysis_json, asi que NO re-anidamos un
    # 'analysis_json' string -> la columna queda flat, completa y parseable por
    # AnalysisResult(**analysis_json) con todo el detalle.
    track_data = {
        **{k: v for k, v in nested.items() if k != 'analysis_json'},
        'id': fingerprint,
        'fingerprint': fingerprint,
        'filename': data.get('filename', ''),
        'artist': data.get('artist', ''),
        'title': data.get('title', ''),
        'album': data.get('album', ''),
        'label': data.get('label', ''),
        'duration': data.get('duration', 0),
        'bpm': data.get('bpm', 0),
        'key': data.get('key'),
        'camelot': data.get('camelot'),
        'energy_dj': data.get('energy_dj', 5),
        'genre': data.get('genre', ''),
        'track_type': data.get('track_type', ''),
        'bpm_source': data.get('bpm_source', 'local_engine'),
        'key_source': data.get('key_source', 'local_engine'),
        'genre_source': data.get('genre_source', 'local_engine'),
        # /cache-analysis es POR DEFINICION la via por la que un analisis hecho
        # FUERA de Render (motor local) se cachea en el servidor. Sellamos el
        # origen — antes quedaba NULL y el panel lo excluia, haciendo que el
        # reparto engine pareciera "todo render". Respetamos un valor explicito
        # si el cliente lo manda.
        'engine_source': data.get('engine_source') or 'local_engine',
        # Sellar la version (explicita del payload o la del detalle anidado).
        # Sin esto quedaba NULL -> _is_analysis_current la trataba como stale.
        'analysis_version': data.get('analysis_version') or nested.get('analysis_version'),
    }

    db.save_track(track_data)
    logger.info(f"[Cache] Análisis local cacheado: {data.get('artist', '?')} - {data.get('title', '?')} ({fingerprint[:12]})")

    return {"status": "cached", "fingerprint": fingerprint}


# ==================== CACHE-LOOKUP / ARTWORK ====================
# Movidos a routes/analysis_artwork.py (paso 5 del troceo de main.py). Se montan
# via init_lookup(...) + app.include_router(lookup_router) arriba. _is_analysis_current
# (lo usa /analyze) y search/save_artwork (las usan /analyze, /identify, etc) SE QUEDAN.

# ==================== ENDPOINTS DE BUSQUEDA ====================

# ==================== ENDPOINTS DE BUSQUEDA / BIBLIOTECA / TRACK ====================
# Movidos a routes/search.py (paso 2 del troceo de main.py). Se montan via
# init_search(db, CAMELOT_COMPATIBLE) + app.include_router(search_router) arriba.

# ==================== ENDPOINTS SIMILAR TRACKS ====================

if SIMILAR_TRACKS_ENABLED:
    @app.post("/similar-tracks")
    async def get_similar_tracks(request: SimilarTracksRequest):
        """Buscar tracks similares por compatibilidad DJ"""
        all_tracks = db.get_all_tracks(5000)
        
        if not all_tracks:
            return []
        
        results = []
        
        for track in all_tracks:
            score, bpm_match, key_compatible_flag, energy_match, genre_match = calculate_compatibility_score(
                track,
                target_bpm=request.bpm,
                target_camelot=request.camelot,
                target_energy=request.energy_dj,
                target_genre=request.genre,
                bpm_tolerance=request.bpm_tolerance,
                energy_tolerance=request.energy_tolerance
            )
            
            if request.track_type and track.get('track_type') != request.track_type:
                score -= 10
            
            if request.has_vocals is not None and track.get('has_vocals') != request.has_vocals:
                score -= 5
            
            if request.has_drop is not None and track.get('has_drop') != request.has_drop:
                score -= 5
            
            if score >= 40:
                results.append(SimilarTrackResult(
                    id=track.get('id', ''),
                    filename=track.get('filename', ''),
                    title=track.get('title'),
                    artist=track.get('artist'),
                    bpm=track.get('bpm', 0),
                    key=track.get('key'),
                    camelot=track.get('camelot'),
                    energy_dj=track.get('energy_dj', 5),
                    genre=track.get('genre', 'unknown'),
                    track_type=track.get('track_type', 'peak'),
                    compatibility_score=score,
                    bpm_match=bpm_match,
                    key_compatible=key_compatible_flag,
                    energy_match=energy_match,
                    genre_match=genre_match,
                ))
        
        results.sort(key=lambda x: x.compatibility_score, reverse=True)
        return results[:request.limit]

    @app.get("/similar-tracks/{track_id}")
    async def get_similar_to_track(track_id: str, limit: int = 20):
        """Buscar tracks similares a uno especfico"""
        reference = db.get_track_by_id(track_id)
        
        if not reference:
            raise HTTPException(404, "Track no encontrado")
        
        request = SimilarTracksRequest(
            bpm=reference.get('bpm'),
            camelot=reference.get('camelot'),
            energy_dj=reference.get('energy_dj'),
            genre=reference.get('genre'),
            track_type=reference.get('track_type'),
            limit=limit + 1
        )
        
        results = await get_similar_tracks(request)
        results = [r for r in results if r.id != track_id]
        
        return results[:limit]

# ==================== INFO ====================

# ==================== PREVIEW / ARTWORK BATCH ====================
# Movidos a routes/preview.py (paso 4 del troceo de main.py). Se montan via
# init_preview(PREVIEWS_DIR, ARTWORK_CACHE_DIR) + include_router(preview_router) arriba.


# ==================== INFO ====================

@app.get("/")
async def root():
    return {
        "name": "DJ Analyzer Pro API",
        "version": "2.9.5",
        "status": "running",
        "modules": {
            "artwork": ARTWORK_ENABLED,
            "genre_detector": GENRE_DETECTOR_ENABLED,
            "similar_tracks": SIMILAR_TRACKS_ENABLED,
            "preview_snippets": True,
        },
        "features": [
            "BPM detection (ID3 + analysis)",
            "Key & Camelot (ID3 + analysis)",
            "Energy analysis (1-10 scale)",
            "Structure analysis (intro/drop/breakdown/outro)",
            "Cue points detection",
            "Beat grid detection",
            "Artwork extraction (ID3)",
            "Genre detection (ID3/AcousticBrainz/spectral)",
            "Collective memory",
            "Similar tracks search",
            "Advanced search filters",
            "Preview snippets (6s streaming)",
        ]
    }

@app.get("/announcement")
async def announcement():
    """
    Anuncio in-app para todos los clientes (canal de comunicacion del
    owner). La app lo consulta al arrancar y muestra un banner dismissable
    si hay uno nuevo (compara el `id` con el ultimo descartado en prefs).

    Controlado por env vars en Render (sin tabla, sin redeploy de codigo):
      ANNOUNCEMENT_MESSAGE  texto a mostrar (vacio/ausente = sin anuncio)
      ANNOUNCEMENT_ID       id estable; cambialo para forzar re-mostrar a
                            quien ya lo descarto. Si no se setea, usamos
                            un hash del mensaje (cambiar el texto = nuevo
                            id automatico).
      ANNOUNCEMENT_URL      link opcional "saber mas"
      ANNOUNCEMENT_LEVEL    "info" (default) | "warning"

    Publico, sin auth: es un mensaje para todos. Privacy-first: no recibe
    ni guarda nada del cliente.
    """
    msg = (os.environ.get('ANNOUNCEMENT_MESSAGE') or '').strip()
    if not msg:
        return {"active": False}
    ann_id = (os.environ.get('ANNOUNCEMENT_ID') or '').strip()
    if not ann_id:
        import hashlib
        ann_id = hashlib.md5(msg.encode('utf-8', errors='replace')).hexdigest()[:12]
    return {
        "active": True,
        "id": ann_id,
        "message": msg[:500],
        "url": (os.environ.get('ANNOUNCEMENT_URL') or '').strip() or None,
        "level": (os.environ.get('ANNOUNCEMENT_LEVEL') or 'info').strip(),
    }


@app.get("/health")
async def health():
    # Database check
    db_status = "ok"
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        db_status = f"error: {e}"

    # FFmpeg check: solo verificar que es accesible como archivo.
    # NO usamos subprocess.run porque en Windows 11 24H2+ lanza WinError 448
    # si algun dir del PATH tiene un reparse point (OneDrive/junctions/symlinks).
    # Basta con saber que el binario existe para el health check.
    ffmpeg_bin = os.environ.get('FFMPEG_BIN')
    if ffmpeg_bin and os.path.isfile(ffmpeg_bin):
        ffmpeg_status = "available"
    elif shutil.which('ffmpeg'):
        ffmpeg_status = "available"
    else:
        ffmpeg_status = "not_found"

    # Disk space
    try:
        disk_usage = shutil.disk_usage("/")
        disk_space_mb = round(disk_usage.free / (1024 * 1024), 1)
    except OSError:
        disk_space_mb = -1

    uptime_seconds = round(time.time() - _startup_time, 1)

    return {
        "status": "ok",
        "version": "2.9.5",
        "uptime_seconds": uptime_seconds,
        "checks": {
            "database": db_status,
            "ffmpeg": ffmpeg_status,
            "disk_space_mb": disk_space_mb,
        },
    }

# ==================== ADMIN / RESET ====================

# HTTPBearer security scheme declarado para que Swagger UI (`/docs`) muestre
# el boton "Authorize" y mande el header `Authorization: Bearer <token>`
# automaticamente. `auto_error=False` deja que `_verify_admin_bearer` devuelva
# nuestro 401 con mensaje consistente en lugar del 403 default de FastAPI.
bearer_scheme = HTTPBearer(auto_error=False, description="ADMIN_TOKEN")


def _verify_admin_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    """Auth Bearer para endpoints admin destructivos. Mismo esquema que
    routes/admin.py y sync_endpoints.py. En dev local sin ADMIN_TOKEN
    configurado se permite sin auth; en Render/Railway sin token se rechaza
    con 500 para fallar fast."""
    if not ADMIN_TOKEN:
        if os.getenv('RENDER') or os.getenv('RAILWAY_ENVIRONMENT'):
            raise HTTPException(500, "ADMIN_TOKEN required in production")
        return  # Dev local: sin token
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Admin token requerido")
    if not hmac.compare_digest(credentials.credentials, ADMIN_TOKEN):
        raise HTTPException(401, "Admin token requerido")


@app.delete("/admin/reset-database", dependencies=[Depends(_verify_admin_bearer)])
async def reset_database(
    confirm: str = Query(..., description="Escribe 'CONFIRMAR' para borrar"),
):
    """
    PELIGROSO: Borra TODA la base de datos (wipe brutal).

    Requiere `Authorization: Bearer $ADMIN_TOKEN` y `?confirm=CONFIRMAR`.

    Borra:
        analysis.db: tracks, corrections, dj_notes, community_cues,
            community_notes, track_ratings, track_popularity,
            beat_grid_corrections, audd_call_log
        sync.db: sync_items, device_seen, users, user_devices,
            link_codes, detected_tracks_sync
        Filesystem: ARTWORK_CACHE_DIR, PREVIEWS_DIR (.mp3 cacheados)

    Tras este reset los devices vinculados pierden su user_id y deben
    volver a /sync/register desde la app. La memoria colectiva (cues,
    notes, ratings, popularity, beat-grid) tambien se borra.
    """
    if confirm != "CONFIRMAR":
        raise HTTPException(400, "Debes escribir 'CONFIRMAR' para borrar la base de datos")

    try:
        # ── Borrar artwork cache (filesystem) ──
        artwork_cleared = False
        if ARTWORK_CACHE_DIR and os.path.exists(ARTWORK_CACHE_DIR):
            shutil.rmtree(ARTWORK_CACHE_DIR)
            os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)
            artwork_cleared = True

        # ── Borrar previews cache (filesystem, .mp3 cacheados) ──
        previews_cleared = False
        previews_count = 0
        if PREVIEWS_DIR and os.path.exists(PREVIEWS_DIR):
            try:
                previews_count = sum(1 for _ in os.scandir(PREVIEWS_DIR))
            except OSError:
                previews_count = 0
            shutil.rmtree(PREVIEWS_DIR)
            os.makedirs(PREVIEWS_DIR, exist_ok=True)
            previews_cleared = True

        # ── Wipe analysis DB ──
        analysis_tables = (
            "tracks", "corrections", "dj_notes",
            "community_cues", "community_notes",
            "track_ratings", "track_popularity",
            "beat_grid_corrections", "audd_call_log",
        )
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        cleared_analysis = []
        for table in analysis_tables:
            try:
                c.execute(f"DELETE FROM {table}")
                cleared_analysis.append(table)
            except sqlite3.OperationalError:
                pass  # Tabla no existe en BDs antiguas — skip silencioso
        conn.commit()
        conn.close()

        # ── Wipe sync DB ──
        sync_db_path = os.environ.get("SYNC_DB_PATH", "/data/sync.db")
        sync_tables = (
            "sync_items", "device_seen", "users", "user_devices",
            "link_codes", "detected_tracks_sync",
        )
        cleared_sync = []
        sync_cleared = False
        if os.path.exists(sync_db_path):
            sync_conn = sqlite3.connect(sync_db_path)
            for table in sync_tables:
                try:
                    sync_conn.execute(f"DELETE FROM {table}")
                    cleared_sync.append(table)
                except sqlite3.OperationalError:
                    pass
            sync_conn.commit()
            sync_conn.close()
            sync_cleared = True

        return {
            "status": "ok",
            "message": "Base de datos reseteada completamente (wipe brutal)",
            "artwork_cache": "limpiado" if artwork_cleared else "no encontrado",
            "previews_cache": {
                "cleared": previews_cleared,
                "files_deleted": previews_count,
            },
            "analysis_tables_cleared": cleared_analysis,
            "sync_tables_cleared": cleared_sync if sync_cleared else "no encontrado",
        }
    except (sqlite3.DatabaseError, OSError, PermissionError) as e:
        raise HTTPException(500, f"Error reseteando: {str(e)}")


@app.delete("/admin/clear-artwork-cache", dependencies=[Depends(_verify_admin_bearer)])
async def clear_artwork_cache():
    """Limpia solo el caché de artwork. Auth Bearer requerida."""
    try:
        if os.path.exists(ARTWORK_CACHE_DIR):
            shutil.rmtree(ARTWORK_CACHE_DIR)
            os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)
        return {"status": "ok", "message": "Caché de artwork limpiado"}
    except (OSError, PermissionError) as e:
        raise HTTPException(500, f"Error: {str(e)}")

# ==================== COMMUNITY (beat-grid / overrides / notes / ratings) ====================
# Movidos a routes/community.py (paso 3 del troceo de main.py). Se montan via
# init_community(db) + app.include_router(community_router) arriba.

def _fetch_community_override(fingerprint: str, field: str) -> Optional[Dict]:
    """Si somos motor local, pregunta a Render por consensus de (fp, field).

    Si no somos motor local (estamos en Render), devuelve None — el caller
    consultara db.get_community_consensus directo. Si Render no responde,
    fail-open (devuelve None).
    """
    if not IS_LOCAL_ENGINE or not fingerprint:
        return None
    render_url = (os.getenv('RENDER_SYNC_URL') or '').rstrip('/')
    if not render_url:
        return None
    try:
        r = requests.get(
            f"{render_url}/community/override/{field}/{fingerprint}",
            timeout=2.0,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get('consensus'):
                return {
                    'value': data['consensus'],
                    'votes': data.get('consensus_votes', 0),
                }
    except (requests.RequestException, ValueError):
        pass
    return None


def _fetch_community_track_type(fingerprint: str) -> Optional[Dict]:
    """Wrapper legacy: delega al generico con field='track_type'.

    Shape Fase 2 ('type' en lugar de 'value') para no romper callers.
    """
    result = _fetch_community_override(fingerprint, 'track_type')
    if not result:
        return None
    return {'type': result['value'], 'votes': result['votes']}
