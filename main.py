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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
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
from typing import Any, Dict, List, Optional

# Logger global del modulo, definido temprano para que las llamadas
# logger.info/warning/error en el codigo de inicializacion no fallen
# con NameError. Antes habia prints aqui; B-L1 los migro a logger.
import logging
logger = logging.getLogger('dj_analyzer')
logger.setLevel(logging.INFO)
from pydantic import BaseModel
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
                'analysis_json': track_data.get('analysis_json', {}),
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

def validate_beatport_bpm(local_bpm: float, beatport_bpm: float, tolerance: float = 0.12) -> bool:
    """Valida si el BPM de Beatport corresponde al track local."""
    if local_bpm <= 0 or beatport_bpm <= 0:
        return True
    ratio = beatport_bpm / local_bpm
    if abs(ratio - 1.0) <= tolerance:
        return True
    if abs(ratio - 2.0) <= tolerance and beatport_bpm >= 80:
        return True
    if abs(ratio - 0.5) <= tolerance and beatport_bpm >= 80:
        return True
    return False


def smart_bpm_correction(local_bpm: float, beatport_bpm: float) -> float:
    """
    Correccion de BPM: Beatport SIEMPRE tiene prioridad.

    Beatport obtiene el BPM directamente del sello discografico,
    por lo que es mas fiable que librosa. Librosa puede detectar
    half/double tempo o valores incorrectos en tracks complejos.

    Returns: BPM de Beatport siempre (nunca None).
    """
    if beatport_bpm and beatport_bpm > 0:
        if local_bpm > 0:
            ratio = beatport_bpm / local_bpm
            if abs(ratio - 1.0) <= 0.12:
                logger.info(f"  [Beatport] BPM match directo: {beatport_bpm}")
            elif abs(ratio - 2.0) <= 0.15 or abs(ratio - 0.5) <= 0.15:
                logger.info(f"  [Beatport] BPM half/double corregido: local {local_bpm:.1f} -> Beatport {beatport_bpm}")
            else:
                logger.info(f"  [Beatport] BPM override: local {local_bpm:.1f} -> Beatport {beatport_bpm} (Beatport tiene prioridad)")
        return beatport_bpm
    return local_bpm


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

app = FastAPI(title="DJ Analyzer Pro API", version="2.3.0", default_response_class=SafeJSONResponse)
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

# Inicializar BD con path de config (no hardcoded)
db = AnalysisDB(db_path=DATABASE_PATH)

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


# ==================== BEATPORT SEARCH ====================

def search_beatport(artist: str, title: str) -> Optional[Dict]:
    """
    Busca BPM y Key de un track en Beatport via scraping HTML.
    Beatport es Next.js y embebe datos en __NEXT_DATA__.
    Campos: track_name, bpm, key_name, genre[].genre_name, artists[].artist_name, length (ms)
    """
    try:
        import urllib.parse
        
        # Limpiar titulo
        clean_title = re.sub(r'\s*\(?(Original Mix|Extended Mix|Radio Edit)\)?', '', title, flags=re.IGNORECASE).strip()
        clean_title = re.sub(r'^[A-D]\d\s+', '', clean_title).strip()
        
        # Limpiar artista
        clean_artist = artist.strip().rstrip('.')
        
        query = f"{clean_artist} {clean_title}"
        encoded_query = urllib.parse.quote(query)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        search_url = f"https://www.beatport.com/search?q={encoded_query}"
        
        response = requests.get(search_url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.info(f"  [Beatport] HTTP {response.status_code}")
            return None
        
        # Extraer __NEXT_DATA__
        next_data_match = re.search(
            r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
            response.text, re.DOTALL
        )
        if not next_data_match:
            logger.info(f"  [Beatport] No __NEXT_DATA__ encontrado")
            return None
        
        data = json.loads(next_data_match.group(1))
        
        # Navegar a la ruta de tracks
        # props.pageProps.dehydratedState.queries[0].state.data.tracks.data
        try:
            tracks = data["props"]["pageProps"]["dehydratedState"]["queries"][0]["state"]["data"]["tracks"]["data"]
        except (KeyError, IndexError, TypeError):
            logger.info(f"  [Beatport] Estructura JSON no esperada")
            return None
        
        if not tracks:
            return None
        
        # Buscar match
        artist_lower = clean_artist.lower()
        title_lower = clean_title.lower()
        
        for track in tracks:
            if not isinstance(track, dict):
                continue
            
            track_name = track.get('track_name', '').lower()
            
            # Match de titulo
            title_match = (title_lower in track_name) or (track_name in title_lower)
            if not title_match:
                title_words = set(title_lower.split())
                track_words = set(track_name.split())
                if title_words and track_words:
                    overlap = len(title_words & track_words) / max(len(title_words), 1)
                    title_match = overlap >= 0.6
            
            if not title_match:
                continue
            
            # Match de artista
            track_artists = track.get('artists', [])
            artist_match = False
            for a in track_artists:
                a_name = a.get('artist_name', '').lower() if isinstance(a, dict) else str(a).lower()
                if artist_lower in a_name or a_name in artist_lower:
                    artist_match = True
                    break
            
            if not artist_match and track_artists:
                continue
            
            # Extraer resultado
            result = {}
            
            # BPM
            if track.get('bpm'):
                try:
                    result['bpm'] = float(track['bpm'])
                except (ValueError, TypeError):
                    pass
            
            # Key - viene como key_name: "C Major", "D Minor", etc.
            key_name = track.get('key_name', '')
            if key_name:
                result['key'] = convert_beatport_key(key_name)
            
            # Duration - viene en milisegundos en campo "length"
            if track.get('length'):
                try:
                    result['duration'] = float(track['length']) / 1000.0
                except (ValueError, TypeError):
                    pass
            
            # Genre - viene como lista: [{"genre_id": 6, "genre_name": "Techno (Peak Time / Driving)"}]
            genres = track.get('genre', [])
            if genres and isinstance(genres, list):
                genre_names = [g.get('genre_name', '') for g in genres if isinstance(g, dict)]
                if genre_names:
                    raw_genre = genre_names[0]
                    cleaned = clean_beatport_genre(raw_genre)
                    result['genre'] = cleaned['genre']
                    result['genre_raw'] = raw_genre
                    result['is_junk_genre'] = cleaned['is_junk']
                    if cleaned['track_type_hint']:
                        result['track_type_hint'] = cleaned['track_type_hint']
            
            if result.get('bpm') or result.get('key'):
                return result
        
        return None
        
    except requests.exceptions.Timeout:
        logger.info(f"  [Beatport] Timeout")
        return None
    except Exception as e:
        logger.error(f"  [Beatport] Error: {e}")
        return None


def find_tracks_in_json(obj, results):
    """Busca recursivamente tracks en estructura JSON"""
    if isinstance(obj, dict):
        # Si tiene bpm y name, probablemente es un track
        if 'bpm' in obj and ('name' in obj or 'title' in obj or 'track_name' in obj):
            results.append(obj)
        # Buscar en valores
        for v in obj.values():
            find_tracks_in_json(v, results)
    elif isinstance(obj, list):
        for item in obj:
            find_tracks_in_json(item, results)
    return results


# ==================== BEATPORT GENRE INTELLIGENCE ====================

# Generos Beatport que NO son generos reales (son categorias comerciales)
BEATPORT_JUNK_GENRES = {
    'Mainstage', 'DJ Tools', 'Beats', 'Dance / Pop',
}

# Mapeo de calificadores Beatport entre parentesis -> track_type
BEATPORT_QUALIFIER_TO_TYPE = {
    # Peak / High energy
    'Peak Time': 'peak_time',
    'Driving': 'peak_time',
    'Raw': 'peak_time',
    'Hard': 'peak_time',
    # Melodic / Builder
    'Melodic': 'builder',
    'Progressive': 'builder',
    'Uplifting': 'builder',
    # Deep / Opener
    'Deep': 'opener',
    'Hypnotic': 'opener',
    'Minimal': 'opener',
    'Deep Tech': 'opener',
    # Chill / Warmup
    'Downtempo': 'warmup',
    'Ambient': 'warmup',
    'Organic': 'warmup',
    'Chill': 'warmup',
    'Electronica': 'warmup',
    # Anthem / Big room
    'Big Room': 'anthem',
    'Electro House': 'anthem',
    'Future Rave': 'anthem',
}

def clean_beatport_genre(raw_genre: str) -> dict:
    """
    Procesa un genero de Beatport y extrae:
    - genre: el genero limpio (sin calificadores entre parentesis)
    - track_type_hint: sugerencia de track_type basada en calificadores
    - is_junk: True si el genero es una categoria comercial sin valor
    
    Ejemplo:
      "Techno (Peak Time / Driving)" -> {
          'genre': 'Techno',
          'track_type_hint': 'peak_time',
          'is_junk': False
      }
      "Mainstage" -> {
          'genre': 'Mainstage',
          'track_type_hint': None,
          'is_junk': True
      }
    """
    if not raw_genre:
        return {'genre': raw_genre, 'track_type_hint': None, 'is_junk': True}
    
    result = {
        'genre': raw_genre,
        'track_type_hint': None,
        'is_junk': raw_genre in BEATPORT_JUNK_GENRES,
    }
    
    # Extraer calificadores entre parentesis: "Techno (Peak Time / Driving)"
    paren_match = re.match(r'^(.+?)\s*\((.+)\)$', raw_genre)
    if paren_match:
        base_genre = paren_match.group(1).strip()
        qualifiers_str = paren_match.group(2).strip()
        
        # El genero limpio es la parte antes del parentesis
        result['genre'] = base_genre
        
        # Buscar track_type en los calificadores
        qualifiers = [q.strip() for q in qualifiers_str.replace('/', ',').split(',')]
        for q in qualifiers:
            q_clean = q.strip()
            if q_clean in BEATPORT_QUALIFIER_TO_TYPE:
                result['track_type_hint'] = BEATPORT_QUALIFIER_TO_TYPE[q_clean]
                break
    
    # Generos compuestos con "/" pero sin parentesis: "Minimal / Deep Tech"
    elif '/' in raw_genre and '(' not in raw_genre:
        parts = [p.strip() for p in raw_genre.split('/')]
        # Usar el primer componente como genero principal
        result['genre'] = raw_genre  # mantener completo, es descriptivo
        # Buscar hints en las partes
        for p in parts:
            if p in BEATPORT_QUALIFIER_TO_TYPE:
                result['track_type_hint'] = BEATPORT_QUALIFIER_TO_TYPE[p]
                break
    
    return result


def convert_beatport_key(beatport_key: str) -> str:
    """
    Convierte key de formato Beatport a formato estandar.
    Beatport: "D Minor", "G Major", "F# Minor", "G Flat Minor", "E Flat Major"
    Estandar: "Dm", "G", "F#m", "F#m", "D#"
    """
    if not beatport_key:
        return None
    
    key = beatport_key.strip()
    
    # Convertir "Flat" a "b" primero
    key = key.replace(' Flat', 'b').replace(' flat', 'b')
    # Convertir "Sharp" a "#"
    key = key.replace(' Sharp', '#').replace(' sharp', '#')
    
    # Convertir "D Minor" -> "Dm", "G Major" -> "G"
    key = key.replace(' Minor', 'm').replace(' minor', 'm')
    key = key.replace(' Major', '').replace(' major', '')
    key = key.replace(' min', 'm').replace(' maj', '')
    
    # Convertir bemoles a sostenidos (KEY_TO_CAMELOT solo tiene sostenidos)
    # Enharmonicos: Cb=B, Db=C#, Eb=D#, Fb=E, Gb=F#, Ab=G#, Bb=A#
    FLAT_TO_SHARP = {
        'Cb': 'B',  'Db': 'C#', 'Eb': 'D#', 'Fb': 'E',
        'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    }
    
    # Extraer la nota base y el sufijo (m o nada)
    is_minor = key.endswith('m')
    base = key[:-1] if is_minor else key
    
    if base in FLAT_TO_SHARP:
        base = FLAT_TO_SHARP[base]
    
    return base + ('m' if is_minor else '')

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
    bpm_confidence: float
    bpm_source: str = "analysis"
    key: Optional[str] = None
    camelot: Optional[str] = None
    key_confidence: float
    key_source: str = "analysis"
    energy_raw: float
    energy_normalized: float
    energy_dj: int
    groove_score: float
    swing_factor: float
    has_intro: bool
    has_buildup: bool
    has_drop: bool
    has_breakdown: bool
    has_outro: bool
    structure_sections: List[Dict]
    track_type: str
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
    has_vocals: bool
    has_heavy_bass: bool
    has_pads: bool
    percussion_density: float
    mix_energy_start: float
    mix_energy_end: float
    drop_timestamp: float
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

class CorrectionRequest(BaseModel):
    track_id: str
    field: str
    old_value: str
    new_value: str
    fingerprint: Optional[str] = None
    device_id: Optional[str] = None

class SearchRequest(BaseModel):
    artist: Optional[str] = None
    genre: Optional[str] = None
    min_bpm: Optional[float] = None
    max_bpm: Optional[float] = None
    min_energy: Optional[int] = None
    max_energy: Optional[int] = None
    key: Optional[str] = None
    track_type: Optional[str] = None
    limit: int = 100

# ==================== MAPEOS ====================

KEY_TO_CAMELOT = {
    'C': '8B', 'C#': '3B', 'D': '10B', 'D#': '5B',
    'E': '12B', 'F': '7B', 'F#': '2B', 'G': '9B',
    'G#': '4B', 'A': '11B', 'A#': '6B', 'B': '1B',
    'Cm': '5A', 'C#m': '12A', 'Dm': '7A', 'D#m': '2A',
    'Em': '9A', 'Fm': '4A', 'F#m': '11A', 'Gm': '6A',
    'G#m': '1A', 'Am': '8A', 'A#m': '3A', 'Bm': '10A'
}

# Mapa inverso para normalizar votos comunitarios cuando un DJ envia
# notacion Camelot ('8B') pero la BD agrupa por nota raw ('C').
CAMELOT_TO_KEY = {v: k for k, v in KEY_TO_CAMELOT.items()}


def get_camelot(key: str) -> str:
    """Convierte key musical a notacion Camelot"""
    return KEY_TO_CAMELOT.get(key, '?')


def camelot_to_key(camelot: str) -> str:
    """
    Convierte notacion Camelot (1A-12B) a nota cruda (C, Cm, F#, etc.).

    Acepta input con espacios al inicio/fin y letra A/B en mayuscula o
    minuscula. Raises ValueError si el input es invalido.
    """
    if camelot is None:
        raise ValueError("Camelot no puede ser None")
    norm = str(camelot).strip().upper()
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
        stderr_msg = e.stderr.decode('utf-8', errors='replace')[:200] if e.stderr else 'unknown'
        logger.error(f"  [Preview] ffmpeg error: {stderr_msg}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None
    except Exception as e:
        logger.error(f"  [Preview] Error generando snippet: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None


# ==================== ANALISIS PRINCIPAL ====================

def analyze_audio(file_path: str, fingerprint: str = None) -> AnalysisResult:
    import warnings
    warnings.filterwarnings('ignore')
    
    #  Obtener duracin SIN cargar audio completo
    duration = librosa.get_duration(path=file_path)
    
    #  Si el track es largo (>4 min), usar anlisis por chunks
    if CHUNKED_ANALYZER_ENABLED and duration > CHUNK_ANALYSIS_THRESHOLD:
        logger.info(f" Track largo ({duration/60:.1f} min) - Usando anlisis por chunks")
        return analyze_audio_chunked(file_path, fingerprint, duration)
    
    # Track corto: anlisis tradicional (carga todo en RAM)
    logger.info(f" Track corto ({duration/60:.1f} min) - Usando anlisis tradicional")
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    
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
        # Ambient/quiet
        energy_normalized = energy_raw / 0.02 * 0.1  # 0-0.1
    elif energy_raw <= 0.08:
        # Chill/downtempo
        energy_normalized = 0.1 + ((energy_raw - 0.02) / 0.06) * 0.2  # 0.1-0.3
    elif energy_raw <= 0.20:
        # Medium energy
        energy_normalized = 0.3 + ((energy_raw - 0.08) / 0.12) * 0.4  # 0.3-0.7
    elif energy_raw <= 0.30:
        # High energy
        energy_normalized = 0.7 + ((energy_raw - 0.20) / 0.10) * 0.2  # 0.7-0.9
    else:
        # Very high (hardstyle/peak)
        energy_normalized = 0.9 + min((energy_raw - 0.30) / 0.20, 1.0) * 0.1  # 0.9-1.0
    
    energy_normalized = max(0.0, min(1.0, energy_normalized))
    
    # Mapear a escala 1-10
    energy_dj = max(1, min(10, int(round(energy_normalized * 10))))
    
    # Structure
    segments = detect_structure(y, sr, duration)
    drop_timestamp = find_drop_timestamp(y, sr, segments)
    
    # Track type
    tt_result = classify_track_type(energy_normalized, segments, duration)
    track_type = tt_result['type']
    track_type_confidence = tt_result['confidence']
    track_type_alternatives = tt_result['alternatives']
    track_type_source = 'waveform'
    
    # Genre & subgenre (spectral + Discogs hybrid)
    genre = 'unknown'
    subgenre = None
    genre_source = 'spectral_analysis'
    
    try:
        # Spectral primero (rapido, local)
        spectral_result = classify_genre_advanced(y, sr, bpm)
        genre = spectral_result.get('genre', 'unknown')
        subgenre = spectral_result.get('subgenre')
        
        # Si tenemos artist+title, intentar Discogs (autoridad mas alta)
        parsed = parse_filename(os.path.basename(file_path))
        artist_guess = id3_data.get('artist') or parsed.get('artist')
        title_guess = id3_data.get('title') or parsed.get('title')
        
        if GENRE_DETECTOR_ENABLED and artist_guess and title_guess and genre_detector:
            try:
                discogs_result = genre_detector.detect_from_discogs(artist_guess, title_guess)
                if discogs_result and discogs_result.get('genre'):
                    genre = discogs_result['genre']
                    if discogs_result.get('subgenre'):
                        subgenre = discogs_result['subgenre']
                    genre_source = 'discogs'
            except Exception as e:
                logger.warning(f"Discogs lookup falló: {e}")
        
        # ID3 puede contener un genre tag tambien
        if id3_data.get('genre') and not genre_source.startswith('discogs'):
            id3_genre = id3_data['genre'].strip()
            # Solo usar si parece un genero real (no junk como "(123)")
            if id3_genre and len(id3_genre) > 1 and not id3_genre.startswith('('):
                genre = id3_genre
                genre_source = 'id3'
    except Exception as e:
        logger.error(f"Error detectando genero: {e}")
    
    # Vocals (mejorado con 4 criterios)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    has_vocals = detect_vocals_improved(y, sr, spectral_centroid)
    
    # Heavy bass (low freq energy)
    try:
        S = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr)
        low_freq_mask = freqs < 150
        low_energy = float(np.mean(S[low_freq_mask]))
        total_energy = float(np.mean(S))
        has_heavy_bass = (low_energy / (total_energy + 1e-9)) > 0.15
    except Exception:
        has_heavy_bass = False
    
    # Pads (high spectral spread + sustained energy)
    try:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = float(np.mean(spectral_bandwidth))
        has_pads = bandwidth_mean > 2000
    except Exception:
        has_pads = False
    
    # Percussion density (onset rate)
    try:
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        percussion_density = len(onset_frames) / max(duration, 1.0)
        percussion_density = min(percussion_density / 10.0, 1.0)  # normalize 0-1
    except Exception:
        percussion_density = 0.0
    
    # Mix energy at start/end (para transiciones DJ)
    try:
        n_samples_chunk = int(sr * 30)  # 30s chunks
        start_energy = float(np.mean(librosa.feature.rms(y=y[:n_samples_chunk])))
        end_energy = float(np.mean(librosa.feature.rms(y=y[-n_samples_chunk:])))
        # Normalizar igual que energy_dj
        mix_energy_start = min(start_energy / 0.30, 1.0)
        mix_energy_end = min(end_energy / 0.30, 1.0)
    except Exception:
        mix_energy_start = 0.5
        mix_energy_end = 0.5
    
    # Cue points + beat grid
    cue_points = []
    first_beat = 0.0
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    if ARTWORK_ENABLED:
        try:
            cue_points = detect_cue_points(y, sr, segments, drop_timestamp)
            beat_data = detect_beat_grid(y, sr)
            first_beat = beat_data.get('first_beat', 0.0)
            beat_interval = beat_data.get('beat_interval', beat_interval)
        except Exception as e:
            logger.warning(f"Error detectando cue points / beat grid: {e}")
    
    # Artwork embedded?
    artwork_embedded = bool(id3_data.get('artwork_data'))
    
    # Fingerprint
    if not fingerprint:
        fingerprint = calculate_fingerprint(file_path)
    
    # Metadata extra
    title = id3_data.get('title') or parse_filename(os.path.basename(file_path)).get('title')
    artist = id3_data.get('artist') or parse_filename(os.path.basename(file_path)).get('artist')
    album = id3_data.get('album')
    label = id3_data.get('label')
    year = id3_data.get('year')
    isrc = id3_data.get('isrc')
    
    # Beatport lookup (BPM+Key autorizados por sello)
    if artist and title:
        try:
            beatport_data = search_beatport(artist, title)
            if beatport_data:
                # BPM
                if beatport_data.get('bpm'):
                    corrected_bpm = smart_bpm_correction(bpm, beatport_data['bpm'])
                    if corrected_bpm:
                        bpm = corrected_bpm
                        bpm_source = "beatport"
                # Key
                if beatport_data.get('key') and key_confidence < 0.85:
                    bp_key = beatport_data['key']
                    bp_camelot = KEY_TO_CAMELOT.get(bp_key)
                    if bp_camelot:
                        key = bp_key
                        camelot = bp_camelot
                        key_source = "beatport"
                # Genre override de Beatport (mas autoridad que spectral)
                if beatport_data.get('genre') and not beatport_data.get('is_junk_genre'):
                    genre = beatport_data['genre']
                    genre_source = "beatport"
                # Track type hint
                if beatport_data.get('track_type_hint'):
                    track_type = beatport_data['track_type_hint']
                    track_type_source = 'beatport'
                    track_type_confidence = 1.0
        except Exception as e:
            logger.warning(f"Error Beatport lookup: {e}")
    
    # Community overrides (Fase 4 v2 - DEBE estar despues de toda fuente externa)
    if fingerprint:
        try:
            # Track type
            tt_override = _fetch_community_track_type(fingerprint)
            if tt_override:
                track_type = tt_override['type']
                track_type_source = 'community'
                track_type_confidence = 1.0

            # Genre / Key (Fase 4 generico)
            cm_genre = (_fetch_community_override(fingerprint, 'genre')
                        or _fetch_community_override(fingerprint, 'subgenre'))
            if cm_genre:
                genre = cm_genre['value']
                genre_source = 'community'

            # Camelot/Key: priorizamos key sobre camelot porque key es mas
            # natural para mostrar (Cm vs 5A).
            cm_key = (_fetch_community_override(fingerprint, 'key')
                      or _fetch_community_override(fingerprint, 'camelot'))
            if cm_key:
                cm_value = cm_key['value']
                if cm_value in KEY_TO_CAMELOT:
                    key = cm_value
                    camelot = KEY_TO_CAMELOT[cm_value]
                    key_source = 'community'
                elif cm_value in {f'{n}{l}' for n in range(1, 13) for l in 'AB'}:
                    camelot = cm_value
                    # Invertir: buscar la key correspondiente al camelot.
                    for k, c in KEY_TO_CAMELOT.items():
                        if c == cm_value:
                            key = k
                            break
                    key_source = 'community'
        except Exception as e:
            logger.warning(f"Community overrides lookup error: {e}")
    
    # Preview snippet (6s desde el drop)
    preview_url = None
    if fingerprint:
        try:
            preview_path = generate_preview_snippet(file_path, fingerprint, drop_timestamp, duration)
            if preview_path:
                preview_url = f"{BASE_URL.rstrip('/')}/preview/{fingerprint}"
                # Push fire-and-forget si somos local engine
                _push_preview_async(fingerprint, preview_path)
        except Exception as e:
            logger.warning(f"Error generando preview: {e}")
    
    # Artwork URL (cache hit o lookup online)
    artwork_url = None
    if fingerprint:
        for ext in ['jpg', 'png', 'jpeg']:
            cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{fingerprint}.{ext}")
            if os.path.exists(cache_path):
                artwork_url = f"{BASE_URL.rstrip('/')}/artwork/{fingerprint}"
                break
        # Si no cache local y tenemos ID3 artwork, guardar
        if not artwork_url and id3_data.get('artwork_data') and ARTWORK_ENABLED:
            try:
                saved = save_artwork_to_cache(
                    fingerprint, id3_data['artwork_data'],
                    id3_data.get('artwork_mime', 'image/jpeg'))
                if saved:
                    artwork_url = f"{BASE_URL.rstrip('/')}/artwork/{fingerprint}"
                    _push_artwork_async(fingerprint, os.path.join(ARTWORK_CACHE_DIR, saved))
            except Exception as e:
                logger.warning(f"Error guardando artwork ID3: {e}")
    
    # Construir result
    result_dict = {
        'title': title,
        'artist': artist,
        'album': album,
        'label': label,
        'year': year,
        'isrc': isrc,
        'duration': duration,
        'bpm': round(bpm, 1),
        'bpm_confidence': round(bpm_confidence, 2),
        'bpm_source': bpm_source,
        'key': key,
        'camelot': camelot,
        'key_confidence': round(key_confidence, 2),
        'key_source': key_source,
        'energy_raw': round(energy_raw, 4),
        'energy_normalized': round(energy_normalized, 2),
        'energy_dj': energy_dj,
        'groove_score': round(groove_score, 2),
        'swing_factor': round(swing_factor, 2),
        'has_intro': segments['has_intro'],
        'has_buildup': segments['has_buildup'],
        'has_drop': segments['has_drop'],
        'has_breakdown': segments['has_breakdown'],
        'has_outro': segments['has_outro'],
        'structure_sections': segments['sections'],
        'track_type': track_type,
        'track_type_source': track_type_source,
        'track_type_confidence': track_type_confidence,
        'track_type_alternatives': track_type_alternatives,
        'genre': genre,
        'subgenre': subgenre,
        'genre_source': genre_source,
        'has_vocals': has_vocals,
        'has_heavy_bass': has_heavy_bass,
        'has_pads': has_pads,
        'percussion_density': round(percussion_density, 2),
        'mix_energy_start': round(mix_energy_start, 2),
        'mix_energy_end': round(mix_energy_end, 2),
        'drop_timestamp': round(drop_timestamp, 1),
        'cue_points': cue_points,
        'first_beat': first_beat,
        'beat_interval': beat_interval,
        'artwork_embedded': artwork_embedded,
        'artwork_url': artwork_url,
        'preview_url': preview_url,
        'fingerprint': fingerprint,
    }
    
    return AnalysisResult(**result_dict)


def analyze_audio_chunked(file_path: str, fingerprint: str, duration: float) -> AnalysisResult:
    """Wrapper para tracks largos: usa ChunkedAudioAnalyzer para no cargar todo en RAM."""
    logger.info(f"  [Chunked] Analizando {duration/60:.1f}min con ChunkedAudioAnalyzer")
    analyzer = get_chunked_analyzer()
    
    # ID3
    id3_data = {}
    if ARTWORK_ENABLED:
        id3_data = extract_id3_metadata(file_path)
    
    # BPM + Key + Energy + Structure desde el chunked analyzer
    bpm_data = analyzer.analyze_bpm(file_path)
    bpm = bpm_data['bpm']
    bpm_confidence = bpm_data['confidence']
    bpm_source = 'analysis'
    
    if id3_data.get('bpm') and 60 < id3_data['bpm'] < 200:
        bpm = id3_data['bpm']
        bpm_source = 'id3'
    
    key_data = analyzer.analyze_key(file_path)
    key = key_data['key']
    key_confidence = key_data['confidence']
    key_source = 'analysis'
    camelot = KEY_TO_CAMELOT.get(key, '?')
    
    id3_key = id3_data.get('key')
    if id3_key and id3_key.strip() not in ['?', '', 'Unknown', 'None']:
        id3_key = id3_key.strip()
        key_source = 'id3'
        if id3_key in KEY_TO_CAMELOT:
            key = id3_key
            camelot = KEY_TO_CAMELOT[id3_key]
        elif len(id3_key) >= 2 and id3_key[-1].upper() in 'AB' and id3_key[:-1].isdigit():
            camelot = id3_key.upper()
            key = id3_key
        else:
            key_source = 'analysis'
    
    energy_data = analyzer.analyze_energy(file_path)
    energy_raw = energy_data['energy_raw']
    energy_normalized = energy_data['energy_normalized']
    energy_dj = energy_data['energy_dj']
    
    structure_data = analyzer.analyze_structure(file_path, duration)
    segments = {
        'has_intro': structure_data['has_intro'],
        'has_buildup': structure_data['has_buildup'],
        'has_drop': structure_data['has_drop'],
        'has_breakdown': structure_data['has_breakdown'],
        'has_outro': structure_data['has_outro'],
        'sections': structure_data['sections'],
    }
    drop_timestamp = structure_data['drop_timestamp']
    
    # Track type
    tt_result = classify_track_type(energy_normalized, segments, duration)
    track_type = tt_result['type']
    track_type_confidence = tt_result['confidence']
    track_type_alternatives = tt_result['alternatives']
    track_type_source = 'waveform'
    
    # Genre/subgenre (uses spectral chunked analyzer)
    genre = energy_data.get('genre', 'unknown')
    subgenre = energy_data.get('subgenre')
    genre_source = 'spectral_analysis'
    
    parsed = parse_filename(os.path.basename(file_path))
    artist_guess = id3_data.get('artist') or parsed.get('artist')
    title_guess = id3_data.get('title') or parsed.get('title')
    
    if GENRE_DETECTOR_ENABLED and artist_guess and title_guess and genre_detector:
        try:
            discogs_result = genre_detector.detect_from_discogs(artist_guess, title_guess)
            if discogs_result and discogs_result.get('genre'):
                genre = discogs_result['genre']
                if discogs_result.get('subgenre'):
                    subgenre = discogs_result['subgenre']
                genre_source = 'discogs'
        except Exception:
            pass
    
    if id3_data.get('genre') and not genre_source.startswith('discogs'):
        id3_genre = id3_data['genre'].strip()
        if id3_genre and len(id3_genre) > 1 and not id3_genre.startswith('('):
            genre = id3_genre
            genre_source = 'id3'
    
    # Vocals / bass / pads / percussion - from chunked
    has_vocals = energy_data.get('has_vocals', False)
    has_heavy_bass = energy_data.get('has_heavy_bass', False)
    has_pads = energy_data.get('has_pads', False)
    percussion_density = energy_data.get('percussion_density', 0.0)
    
    mix_energy_start = energy_data.get('mix_energy_start', 0.5)
    mix_energy_end = energy_data.get('mix_energy_end', 0.5)
    
    # Cue points + beat grid (sampled chunks)
    cue_points = []
    first_beat = 0.0
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    groove_score = 0.5
    swing_factor = 0.5
    
    # Metadata extra
    title = id3_data.get('title') or parsed.get('title')
    artist = id3_data.get('artist') or parsed.get('artist')
    album = id3_data.get('album')
    label = id3_data.get('label')
    year = id3_data.get('year')
    isrc = id3_data.get('isrc')
    
    # Beatport lookup
    if artist and title:
        try:
            beatport_data = search_beatport(artist, title)
            if beatport_data:
                if beatport_data.get('bpm'):
                    corrected_bpm = smart_bpm_correction(bpm, beatport_data['bpm'])
                    if corrected_bpm:
                        bpm = corrected_bpm
                        bpm_source = 'beatport'
                if beatport_data.get('key') and key_confidence < 0.85:
                    bp_key = beatport_data['key']
                    bp_camelot = KEY_TO_CAMELOT.get(bp_key)
                    if bp_camelot:
                        key = bp_key
                        camelot = bp_camelot
                        key_source = 'beatport'
                if beatport_data.get('genre') and not beatport_data.get('is_junk_genre'):
                    genre = beatport_data['genre']
                    genre_source = 'beatport'
                if beatport_data.get('track_type_hint'):
                    track_type = beatport_data['track_type_hint']
                    track_type_source = 'beatport'
                    track_type_confidence = 1.0
        except Exception:
            pass
    
    if fingerprint:
        try:
            tt_override = _fetch_community_track_type(fingerprint)
            if tt_override:
                track_type = tt_override['type']
                track_type_source = 'community'
                track_type_confidence = 1.0

            cm_genre = (_fetch_community_override(fingerprint, 'genre')
                        or _fetch_community_override(fingerprint, 'subgenre'))
            if cm_genre:
                genre = cm_genre['value']
                genre_source = 'community'

            cm_key = (_fetch_community_override(fingerprint, 'key')
                      or _fetch_community_override(fingerprint, 'camelot'))
            if cm_key:
                cm_value = cm_key['value']
                if cm_value in KEY_TO_CAMELOT:
                    key = cm_value
                    camelot = KEY_TO_CAMELOT[cm_value]
                    key_source = 'community'
                elif cm_value in {f'{n}{l}' for n in range(1, 13) for l in 'AB'}:
                    camelot = cm_value
                    for k, c in KEY_TO_CAMELOT.items():
                        if c == cm_value:
                            key = k
                            break
                    key_source = 'community'
        except Exception:
            pass
    
    artwork_embedded = bool(id3_data.get('artwork_data'))
    
    if not fingerprint:
        fingerprint = calculate_fingerprint(file_path)
    
    # Preview snippet
    preview_url = None
    try:
        preview_path = generate_preview_snippet(file_path, fingerprint, drop_timestamp, duration)
        if preview_path:
            preview_url = f"{BASE_URL.rstrip('/')}/preview/{fingerprint}"
            _push_preview_async(fingerprint, preview_path)
    except Exception as e:
        logger.warning(f"Error preview chunked: {e}")
    
    # Artwork URL
    artwork_url = None
    for ext in ['jpg', 'png', 'jpeg']:
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{fingerprint}.{ext}")
        if os.path.exists(cache_path):
            artwork_url = f"{BASE_URL.rstrip('/')}/artwork/{fingerprint}"
            break
    if not artwork_url and id3_data.get('artwork_data') and ARTWORK_ENABLED:
        try:
            saved = save_artwork_to_cache(
                fingerprint, id3_data['artwork_data'],
                id3_data.get('artwork_mime', 'image/jpeg'))
            if saved:
                artwork_url = f"{BASE_URL.rstrip('/')}/artwork/{fingerprint}"
                _push_artwork_async(fingerprint, os.path.join(ARTWORK_CACHE_DIR, saved))
        except Exception:
            pass
    
    result_dict = {
        'title': title,
        'artist': artist,
        'album': album,
        'label': label,
        'year': year,
        'isrc': isrc,
        'duration': duration,
        'bpm': round(bpm, 1),
        'bpm_confidence': round(bpm_confidence, 2),
        'bpm_source': bpm_source,
        'key': key,
        'camelot': camelot,
        'key_confidence': round(key_confidence, 2),
        'key_source': key_source,
        'energy_raw': round(energy_raw, 4),
        'energy_normalized': round(energy_normalized, 2),
        'energy_dj': energy_dj,
        'groove_score': round(groove_score, 2),
        'swing_factor': round(swing_factor, 2),
        'has_intro': segments['has_intro'],
        'has_buildup': segments['has_buildup'],
        'has_drop': segments['has_drop'],
        'has_breakdown': segments['has_breakdown'],
        'has_outro': segments['has_outro'],
        'structure_sections': segments['sections'],
        'track_type': track_type,
        'track_type_source': track_type_source,
        'track_type_confidence': track_type_confidence,
        'track_type_alternatives': track_type_alternatives,
        'genre': genre,
        'subgenre': subgenre,
        'genre_source': genre_source,
        'has_vocals': has_vocals,
        'has_heavy_bass': has_heavy_bass,
        'has_pads': has_pads,
        'percussion_density': round(percussion_density, 2),
        'mix_energy_start': round(mix_energy_start, 2),
        'mix_energy_end': round(mix_energy_end, 2),
        'drop_timestamp': round(drop_timestamp, 1),
        'cue_points': cue_points,
        'first_beat': first_beat,
        'beat_interval': beat_interval,
        'artwork_embedded': artwork_embedded,
        'artwork_url': artwork_url,
        'preview_url': preview_url,
        'fingerprint': fingerprint,
    }
    
    return AnalysisResult(**result_dict)


# ==================== AudD AUTO-TRIGGER HELPER ====================

def _audd_eligibility(fingerprint: str, duration: float) -> Optional[str]:
    """Devuelve None si elegible, o un mensaje de motivo si no.

    Reglas (todas configurables vía env, ver config.py):
      - AUDD_AUTO_ENABLED debe ser True.
      - AUDD_API_TOKEN debe estar configurado.
      - duration entre AUDD_MIN_DURATION y AUDD_MAX_DURATION.
      - Llamadas hoy < AUDD_DAILY_CAP.
      - Última llamada para este fingerprint > AUDD_COOLDOWN_DAYS atrás.
    """
    if not AUDD_AUTO_ENABLED:
        return "auto disabled"
    if not AUDD_API_TOKEN:
        return "no API token"
    if duration < AUDD_MIN_DURATION:
        return f"duration {duration:.1f}s < {AUDD_MIN_DURATION}s"
    if duration > AUDD_MAX_DURATION:
        return f"duration {duration:.1f}s > {AUDD_MAX_DURATION}s"
    calls_today = db.count_audd_calls_today()
    if calls_today >= AUDD_DAILY_CAP:
        return f"daily cap reached ({calls_today}/{AUDD_DAILY_CAP})"
    last = db.get_last_audd_call(fingerprint)
    if last:
        elapsed_days = (time.time() - last) / 86400
        if elapsed_days < AUDD_COOLDOWN_DAYS:
            return f"cooldown {elapsed_days:.1f}d < {AUDD_COOLDOWN_DAYS}d"
    return None


def _maybe_call_audd_async(file_path: str, fingerprint: str, duration: float,
                            existing_artist: Optional[str], existing_title: Optional[str]) -> None:
    """Llama AudD en background si el track no tiene metadatos confiables.

    No bloquea la respuesta al cliente. Si AudD encuentra match, los
    metadatos enriquecidos se guardan al re-analizarse o vía endpoint
    explícito. Por ahora simplemente loguea el resultado.
    """
    # Si ya hay artist+title razonables, no llamar AudD.
    if existing_artist and existing_title and len(existing_artist) > 2 and len(existing_title) > 2:
        return
    
    reason = _audd_eligibility(fingerprint, duration)
    if reason:
        logger.info(f"  [AudD] Skip: {reason}")
        return
    
    def _do_call():
        try:
            from audd_helper import recognize_track
            logger.info(f"  [AudD] Llamando para {fingerprint[:8]}...")
            r = recognize_track(file_path)
            ok = bool(r and r.get('artist') and r.get('title'))
            db.log_audd_call(fingerprint, ok, r.get('artist'), r.get('title'))
            if ok:
                logger.info(f"  [AudD]  Match: {r['artist']} - {r['title']}")
            else:
                logger.info(f"  [AudD] Sin match")
        except Exception as e:
            logger.error(f"  [AudD] Error: {e}")
            try:
                db.log_audd_call(fingerprint, False)
            except Exception:
                pass
    
    import threading
    threading.Thread(target=_do_call, daemon=True).start()


# ==================== ENDPOINTS ====================

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_track(file: UploadFile = File(...)):
    """Analizar un archivo de audio - SOPORTA: MP3, WAV, FLAC, M4A"""
    
    #  Validar archivo (tipo, tamao, nombre)
    file = validate_audio_file(file)
    
    # CRITICAL: Crear archivo temporal en disco antes de cualquier I/O.
    # Antes guardabamos en memoria; con tracks > 100MB el dyno OOM-killed
    # silenciosamente y devolvia 502 sin trace en logs.
    suffix = os.path.splitext(file.filename)[1] or '.tmp'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        
        fingerprint = calculate_fingerprint(tmp_path)
        
        # Cache hit?
        existing = db.get_track_by_filename(file.filename) or db.get_track_by_fingerprint(fingerprint)
        if existing:
            existing_json = existing.get('analysis_json')
            if existing_json:
                try:
                    cached = json.loads(existing_json) if isinstance(existing_json, str) else existing_json
                    if cached.get('bpm', 0) > 0 and cached.get('bpm_source') != 'pending':
                        logger.info(f"[Cache] Hit para {fingerprint[:12]}: {existing.get('artist')} - {existing.get('title')}")
                        # Aplicar community overrides actualizados
                        # (puede haber consensus nuevo desde el cache)
                        try:
                            tt_override = _fetch_community_track_type(fingerprint)
                            if tt_override:
                                cached['track_type'] = tt_override['type']
                                cached['track_type_source'] = 'community'
                                cached['track_type_confidence'] = 1.0
                            cm_genre = (_fetch_community_override(fingerprint, 'genre')
                                        or _fetch_community_override(fingerprint, 'subgenre'))
                            if cm_genre:
                                cached['genre'] = cm_genre['value']
                                cached['genre_source'] = 'community'
                            cm_key = (_fetch_community_override(fingerprint, 'key')
                                      or _fetch_community_override(fingerprint, 'camelot'))
                            if cm_key:
                                cm_value = cm_key['value']
                                if cm_value in KEY_TO_CAMELOT:
                                    cached['key'] = cm_value
                                    cached['camelot'] = KEY_TO_CAMELOT[cm_value]
                                    cached['key_source'] = 'community'
                                elif cm_value in {f'{n}{l}' for n in range(1, 13) for l in 'AB'}:
                                    cached['camelot'] = cm_value
                                    for k, c in KEY_TO_CAMELOT.items():
                                        if c == cm_value:
                                            cached['key'] = k
                                            break
                                    cached['key_source'] = 'community'
                        except Exception as e:
                            logger.warning(f"Cache + community override merge error: {e}")
                        db.increment_popularity(fingerprint)
                        # Asegurarse que se devuelve como AnalysisResult
                        return AnalysisResult(**cached)
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning(f"Cache invalido para {file.filename}: {e}")
        
        # Analizar
        logger.info(f"\n[Analyze] {file.filename} ({len(content)/1024/1024:.1f}MB)")
        result = analyze_audio(tmp_path, fingerprint)
        
        # Guardar en BD
        track_id = fingerprint
        track_data = result.dict()
        track_data['id'] = track_id
        track_data['filename'] = file.filename
        track_data['analyzed_at'] = time.time()
        db.save_track(track_data)
        db.increment_popularity(fingerprint)
        
        # Auto-upload a Render si somos local engine
        if IS_LOCAL_ENGINE:
            _upload_to_render_cache(track_data)
        
        # AudD background si falta metadata
        _maybe_call_audd_async(
            tmp_path, fingerprint, result.duration,
            result.artist, result.title,
        )
        
        return result
        
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/cache-analysis")
async def cache_analysis(data: dict):
    """Endpoint para que el motor local suba un anlisis ya completo.
    
    No re-analiza; solo guarda lo que recibe. Idempotente: si ya existe
    con anlisis completo (BPM > 0), no sobreescribe.
    """
    
    fingerprint = data.get('fingerprint')
    if not fingerprint:
        raise HTTPException(400, "fingerprint requerido")
    
    # No sobreescribir si ya existe con análisis completo. El método
    # correcto es `get_track_by_fingerprint` — `db.get_track` nunca
    # existió y rompía el endpoint con 500 cuando el motor local
    # intentaba subir el resultado a Render.
    existing = db.get_track_by_fingerprint(fingerprint)
    if existing:
        existing_json = existing.get('analysis_json')
        if existing_json:
            try:
                ej = json.loads(existing_json) if isinstance(existing_json, str) else existing_json
                # Si el existente tiene BPM real (no 0, no pending), no sobreescribir
                if ej.get('bpm', 0) > 0 and ej.get('bpm_source') != 'pending':
                    logger.info(f"[Cache] {fingerprint[:12]} ya existe con análisis completo, skip")
                    return {"status": "exists", "fingerprint": fingerprint}
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Construir datos para guardar
    track_data = {
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
        'analysis_json': json.dumps(data.get('analysis_json', {})) if isinstance(data.get('analysis_json'), dict) else data.get('analysis_json', '{}'),
    }
    
    db.save_track(track_data)
    logger.info(f"[Cache] Análisis local cacheado: {data.get('artist', '?')} - {data.get('title', '?')} ({fingerprint[:12]})")
    
    return {"status": "cached", "fingerprint": fingerprint}


@app.post("/check-analyzed")
async def check_analyzed(filenames: list[str]):
    """Verificar cules tracks ya estn analizados"""
    analyzed = []
    not_analyzed = []
    
    for filename in filenames:
        existing = db.get_track_by_filename(filename)
        if existing:
            analyzed.append(filename)
        else:
            not_analyzed.append(filename)
    
    return {
        "analyzed": analyzed,
        "not_analyzed": not_analyzed,
        "total": len(filenames),
        "analyzed_count": len(analyzed),
        "not_analyzed_count": len(not_analyzed)
    }


class CheckAnalyzedByFingerprintRequest(BaseModel):
    fingerprints: List[str]


@app.post("/check-analyzed-by-fingerprint")
async def check_analyzed_by_fingerprint(request: CheckAnalyzedByFingerprintRequest):
    """
    Dedup multi-dispositivo: dado un lote de fingerprints (MD5 del contenido
    del archivo) devuelve cuáles ya están analizados en Render. Esto
    permite que el cliente (especialmente móvil) evite subir y re-analizar
    tracks que ya fueron procesados desde otro dispositivo aunque el nombre
    del fichero sea distinto.

    Máximo 500 IDs por petición.
    """
    fps = request.fingerprints or []
    if len(fps) > 500:
        raise HTTPException(400, "Máximo 500 fingerprints por petición")

    analyzed: list[str] = []
    not_analyzed: list[str] = []
    for fp in fps:
        if not fp:
            continue
        # `get_track_by_fingerprint` ya cubre el caso `id == fingerprint`
        # para registros antiguos donde el id legacy es el propio MD5.
        existing = db.get_track_by_fingerprint(fp)
        if existing:
            analyzed.append(fp)
        else:
            not_analyzed.append(fp)

    return {
        "analyzed": analyzed,
        "not_analyzed": not_analyzed,
        "total": len(fps),
        "analyzed_count": len(analyzed),
        "not_analyzed_count": len(not_analyzed),
    }


@app.get("/analysis/by-fingerprint/{fingerprint}")
async def get_analysis_by_fingerprint(fingerprint: str):
    """Devuelve el análisis cacheado de un track por su fingerprint
    (MD5 del contenido). El cliente puede usar este endpoint tras
    `/check-analyzed-by-fingerprint` para hidratar su cache local sin
    subir el archivo otra vez."""
    safe_fp = re.sub(r'[^a-fA-F0-9]', '', fingerprint or '')
    if not safe_fp:
        raise HTTPException(400, "fingerprint inválido")
    existing = db.get_track_by_fingerprint(safe_fp)
    if not existing:
        raise HTTPException(404, "fingerprint no encontrado")
    raw = existing.get('analysis_json')
    if raw:
        try:
            import json
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: construir desde columnas
    return {
        "id": existing.get('id'),
        "filename": existing.get('filename'),
        "artist": existing.get('artist'),
        "title": existing.get('title'),
        "duration": existing.get('duration') or 0,
        "bpm": existing.get('bpm') or 0,
        "key": existing.get('key'),
        "camelot": existing.get('camelot'),
        "energy_dj": existing.get('energy_dj') or 5,
        "genre": existing.get('genre'),
        "track_type": existing.get('track_type'),
        "fingerprint": existing.get('fingerprint'),
    }


# ==================== ENDPOINTS DE ARTWORK ====================

@app.get("/analysis/{filename:path}")
async def get_analysis(filename: str):
    """Obtener anlisis guardado de un track por filename"""
    # Decodificar filename si viene con URL encoding
    from urllib.parse import unquote
    import json
    filename = unquote(filename)
    
    existing = db.get_track_by_filename(filename)
    if existing:
        # existing es una tupla, convertir a diccionario
        # Columnas: id, filename, artist, title, duration, bpm, key, camelot, 
        #           energy_dj, genre, track_type, analysis_json, analyzed_at, fingerprint
        try:
            # Si hay analysis_json guardado, usarlo directamente
            analysis_json = existing[11]  # ndice de analysis_json
            if analysis_json:
                return json.loads(analysis_json)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("analysis_json cacheado corrupto, fallback a respuesta basica: %s", e)
        
        # Fallback: construir respuesta bsica
        return {
            "id": existing[0],
            "filename": existing[1],
            "artist": existing[2],
            "title": existing[3],
            "duration": existing[4],
            "bpm": existing[5],
            "key": existing[6],
            "camelot": existing[7],
            "energy_dj": existing[8],
            "genre": existing[9],
            "track_type": existing[10],
            "fingerprint": existing[13] if len(existing) > 13 else None,
        }
    
    raise HTTPException(404, f"Anlisis no encontrado para: {filename}")

@app.head("/artwork/{track_id}")
async def head_artwork(track_id: str):
    """HEAD para /artwork/{track_id} - el cliente desktop pre-comprueba
    existencia antes de subir su propio artwork (evita re-upload). Solo
    mira el cache local del disco; NO dispara el fallback online del GET
    (search_artwork_online tiene side effects: red + escritura a cache).
    Devuelve 200 con Content-Type/Content-Length, o 404 sin body.
    """
    for ext in ('jpg', 'png', 'jpeg'):
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            media_type = "image/jpeg" if ext in ('jpg', 'jpeg') else "image/png"
            return Response(
                status_code=200,
                headers={
                    "Content-Type": media_type,
                    "Content-Length": str(os.path.getsize(cache_path)),
                },
            )
    raise HTTPException(404, "Artwork no encontrado")


@app.get("/artwork/{track_id}")
async def get_artwork(track_id: str):
    """Devuelve el artwork de un track como imagen.

    Cascade:
      1. Cache local (`{ARTWORK_CACHE_DIR}/{track_id}.{ext}`).
      2. Si la BD tiene el track pero falta el archivo (típicamente
         tracks analizados con motor local cuyo PUSH a Render falló o
         no se hizo), buscamos artwork online (iTunes/Deezer) usando
         artist+title de la BD y lo cacheamos para futuras peticiones.
      3. 404 si nada de lo anterior funciona.
    """
    for ext in ['jpg', 'png', 'jpeg']:
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            media_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else "image/png"
            return FileResponse(cache_path, media_type=media_type)

    # Fallback: buscar online por artist+title si tenemos el track en BD.
    try:
        existing = db.get_track_by_fingerprint(track_id) or db.get_track_by_id(track_id)
        if existing:
            artist = existing.get('artist')
            title = existing.get('title')
            if artist and title and search_artwork_online:
                logger.info(f"[Artwork] Cache MISS para {track_id[:8]}, buscando online...")
                online = search_artwork_online(artist, title)
                if online and online.get('data'):
                    saved = save_artwork_to_cache(
                        track_id, online['data'], online['mime_type'])
                    saved_path = os.path.join(ARTWORK_CACHE_DIR, saved)
                    media_type = online['mime_type']
                    return FileResponse(saved_path, media_type=media_type)
    except Exception as e:
        logger.warning(f"[Artwork] Fallback online error: {e}")

    raise HTTPException(404, "Artwork no encontrado")


@app.post("/artwork/upload/{fingerprint}")
async def upload_artwork(fingerprint: str, file: UploadFile = File(...)):
    """Recibe artwork desde el local engine para que Render lo sirva
    también a otros devices vía `/artwork/{fingerprint}`. Sin esto,
    cuando el local engine analiza un track el artwork se queda en
    disco PC y los móviles ven placeholder.

    Sanitiza el fingerprint (solo hex 32 chars). Acepta JPEG/PNG.
    Idempotente: re-subir el mismo fp sobreescribe.
    """
    safe_fp = re.sub(r'[^a-fA-F0-9]', '', fingerprint or '')
    if not safe_fp or len(safe_fp) > 64:
        raise HTTPException(400, "fingerprint inválido")

    content = await file.read()
    if not content or len(content) < 100:
        raise HTTPException(400, "archivo vacío o demasiado pequeño")
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(400, "artwork demasiado grande (max 5MB)")

    # Detectar tipo por bytes mágicos. Defaults a jpg si no clarifica.
    if content[:3] == b'\xff\xd8\xff':
        ext = 'jpg'
    elif content[:8] == b'\x89PNG\r\n\x1a\n':
        ext = 'png'
    else:
        # No reconocido — rechazar para no llenar el disco con basura.
        raise HTTPException(400, "formato no soportado (sólo JPEG/PNG)")

    # Eliminar versiones previas con otra extensión para evitar dos
    # archivos del mismo fingerprint en el cache.
    for prev_ext in ('jpg', 'jpeg', 'png'):
        prev = os.path.join(ARTWORK_CACHE_DIR, f"{safe_fp}.{prev_ext}")
        if os.path.exists(prev):
            try:
                os.unlink(prev)
            except OSError:
                pass

    cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{safe_fp}.{ext}")
    with open(cache_path, 'wb') as f:
        f.write(content)

    return {"status": "ok", "fingerprint": safe_fp, "size": len(content), "ext": ext}

# ==================== ENDPOINTS DE BUSQUEDA ====================

@app.get("/search/artist/{artist}")
async def search_by_artist(artist: str, limit: int = Query(50, ge=1, le=200)):
    """Buscar tracks por artista"""
    artist = sanitize_string(artist, max_length=200, allow_empty=False, field_name="artist")
    limit = validate_limit(limit, max_limit=200)
    results = db.search_by_artist(artist, limit)
    return {"query": artist, "count": len(results), "tracks": results}

@app.get("/search/genre/{genre}")
async def search_by_genre(genre: str, limit: int = Query(100, ge=1, le=500)):
    #  Sanitizar g(c)nero
    genre = validate_genre(genre)
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_genre(genre, limit)}

@app.get("/search/bpm")
async def search_by_bpm(
    request: Request,
    min_bpm: Optional[float] = None,
    max_bpm: Optional[float] = None,
    limit: int = Query(100, ge=1, le=500)
):
    #  Validar rangos
    min_bpm, max_bpm = validate_bpm_range(min_bpm, max_bpm)
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_bpm_range(min_bpm, max_bpm, limit)}

@app.get("/search/energy")
async def search_by_energy(
    request: Request,
    min_energy: Optional[int] = None,
    max_energy: Optional[int] = None,
    limit: int = Query(100, ge=1, le=500)
):
    #  Validar rangos
    min_energy, max_energy = validate_energy_range(min_energy, max_energy)
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_energy(min_energy, max_energy, limit)}

@app.get("/search/key/{key}")
async def search_by_key(key: str, limit: int = Query(100, ge=1, le=500)):
    #  Validar tonalidad
    try:
        key = validate_key(key)
    except ValidationError:
        # Si no es vlido como key, intentar como est
        key = sanitize_string(key, max_length=10)
    
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_key(key, limit)}

@app.get("/search/compatible/{camelot}")
async def search_compatible_keys(camelot: str, limit: int = Query(50, ge=1, le=200)):
    #  Validar Camelot
    camelot = validate_camelot(camelot)
    limit = validate_limit(limit, max_limit=200)
    
    # Obtener keys compatibles
    compatible = CAMELOT_COMPATIBLE.get(camelot, [camelot])
    
    return {
        "camelot": camelot,
        "compatible_keys": compatible,
        "tracks": db.search_compatible_keys(camelot, limit)
    }
@app.get("/search-analyzed")
async def search_analyzed_track(
    artist: str = Query(..., description="Nombre del artista"),
    title: str = Query(..., description="Ttulo del track")
):
    """
    Busca si un track ya fue analizado por algn usuario.
    Devuelve TODA la informacin del anlisis si existe.
    
    Returns:
        - found: bool - Si se encontr el track
        - track: dict - Toda la informacin del anlisis (si existe)
        - in_collective: bool - Si est en la memoria colectiva
    """
    import re
    
    # Validar y sanitizar entrada
    artist_clean = sanitize_string(artist, max_length=200, allow_empty=False, field_name="artist")
    title_clean = sanitize_string(title, max_length=200, allow_empty=False, field_name="title")
    
    # Normalizar para bsqueda
    artist_normalized = artist_clean.lower().strip()
    title_normalized = re.sub(
        r'\s*\(?(Original Mix|Extended Mix|Radio Edit|Remix|Club Mix|Dub Mix)\)?', 
        '', 
        title_clean, 
        flags=re.IGNORECASE
    ).lower().strip()
    
    try:
        conn = db.conn
        cursor = conn.cursor()
        
        # Bsqueda exacta primero
        cursor.execute("""
            SELECT * FROM tracks 
            WHERE LOWER(artist) = ? AND LOWER(title) LIKE ?
            AND bpm IS NOT NULL AND bpm > 0
            ORDER BY analyzed_at DESC
            LIMIT 1
        """, (artist_normalized, f"%{title_normalized}%"))
        
        row = cursor.fetchone()
        
        if not row:
            # Bsqueda ms flexible
            cursor.execute("""
                SELECT * FROM tracks 
                WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
                AND bpm IS NOT NULL AND bpm > 0
                ORDER BY analyzed_at DESC
                LIMIT 1
            """, (f"%{artist_normalized}%", f"%{title_normalized}%"))
            row = cursor.fetchone()
        
        if row:
            # Convertir a dict usando el m(c)todo existente
            track_dict = db._row_to_dict(row)
            
            # Si hay analysis_json, parsear para obtener todos los campos
            if track_dict and track_dict.get('analysis_json'):
                try:
                    full_analysis = json.loads(track_dict['analysis_json'])
                    # Combinar con los campos bsicos
                    track_dict.update(full_analysis)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("analysis_json corrupto en track %s: %s",
                                   track_dict.get('id'), e)
            
            # Eliminar el JSON crudo del response
            if track_dict and 'analysis_json' in track_dict:
                del track_dict['analysis_json']
            
            return {
                "found": True,
                "in_collective": True,
                "track": track_dict
            }
        
        return {
            "found": False,
            "in_collective": False,
            "track": None
        }
        
    except Exception as e:
        logger.error(f"Error en search-analyzed: {e}")
        return {
            "found": False,
            "in_collective": False,
            "track": None,
            "error": str(e)
        }

@app.get("/search/track-type/{track_type}")
async def search_by_track_type(track_type: str, limit: int = Query(100, ge=1, le=500)):
    #  Validar tipo de track
    track_type = validate_track_type(track_type)
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_track_type(track_type, limit)}

@app.post("/search/advanced")
async def search_advanced(search_request: SearchRequest):
    #  Validar y sanitizar todos los campos
    filters = {}
    
    if search_request.artist:
        filters['artist'] = sanitize_string(search_request.artist, max_length=100)
    
    if search_request.genre:
        filters['genre'] = validate_genre(search_request.genre)
    
    if search_request.min_bpm is not None or search_request.max_bpm is not None:
        filters['min_bpm'], filters['max_bpm'] = validate_bpm_range(
            search_request.min_bpm, 
            search_request.max_bpm
        )
    
    if search_request.min_energy is not None or search_request.max_energy is not None:
        filters['min_energy'], filters['max_energy'] = validate_energy_range(
            search_request.min_energy,
            search_request.max_energy
        )
    
    if search_request.key:
        try:
            filters['key'] = validate_key(search_request.key)
        except ValidationError:
            filters['key'] = sanitize_string(search_request.key, max_length=10)
    
    if search_request.track_type:
        filters['track_type'] = validate_track_type(search_request.track_type)
    
    filters['limit'] = validate_limit(search_request.limit, max_limit=500)
    
    return {"tracks": db.search_advanced(**filters)}

# ==================== ENDPOINTS DE BIBLIOTECA ====================

@app.get("/library/all")
async def get_all_tracks(limit: int = Query(1000, ge=1, le=5000)):
    #  Validar lmite
    limit = validate_limit(limit, max_limit=5000)
    
    return {"tracks": db.get_all_tracks(limit)}

@app.get("/library/artists")
async def get_unique_artists():
    """Obtener lista de artistas nicos"""
    artists = db.get_unique_artists()
    return {"count": len(artists), "artists": artists}

@app.get("/library/genres")
async def get_unique_genres():
    """Obtener lista de g(c)neros nicos"""
    genres = db.get_unique_genres()
    return {"count": len(genres), "genres": genres}

@app.get("/library/stats")
async def get_library_stats():
    """Obtener estadsticas de la biblioteca"""
    return db.get_stats()

@app.get("/track/{track_id}")
async def get_track(track_id: str):
    """Obtener informacin de un track especfico"""
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track no encontrado")
    return track

@app.delete("/track/{track_id}")
async def delete_track(track_id: str):
    """Eliminar un track de la base de datos"""
    deleted = db.delete_track(track_id)
    if not deleted:
        raise HTTPException(404, "Track no encontrado")
    return {"status": "ok", "message": "Track eliminado"}

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

@app.get("/preview/{track_id}")
async def get_preview(track_id: str):
    """
    Sirve el snippet de preview de un track (MP3 6s mono 64kbps).
    
    El track_id es el fingerprint (MD5 hash del archivo).
    Cache agresivo: el snippet no cambia una vez generado.
    """
    # Validar formato de track_id (MD5 = 32 chars hexadecimales)
    if not track_id or len(track_id) > 64:
        raise HTTPException(400, "track_id inválido")
    
    # Sanitizar para evitar path traversal
    safe_id = re.sub(r'[^a-fA-F0-9]', '', track_id)
    if safe_id != track_id:
        raise HTTPException(400, "track_id contiene caracteres inválidos")
    
    preview_path = os.path.join(PREVIEWS_DIR, f"{safe_id}.mp3")
    
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview not available")
    
    return FileResponse(
        preview_path,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=31536000",  # Cache 1 año
            "Content-Disposition": f"inline; filename={safe_id}_preview.mp3",
        }
    )


class PreviewCheckRequest(BaseModel):
    track_ids: List[str]

@app.post("/previews/check")
async def check_previews(request: PreviewCheckRequest):
    """
    Devuelve qué track_ids tienen preview snippet disponible.
    
    Útil para que el cliente sepa de antemano cuáles puede preescuchar
    sin tener que intentar reproducir y fallar.
    
    Máximo 500 IDs por petición.
    """
    track_ids = request.track_ids
    
    if len(track_ids) > 500:
        raise HTTPException(400, "Máximo 500 track_ids por petición")
    
    available = []
    for tid in track_ids:
        safe_id = re.sub(r'[^a-fA-F0-9]', '', tid)
        if safe_id and os.path.exists(os.path.join(PREVIEWS_DIR, f"{safe_id}.mp3")):
            available.append(tid)
    
    return {
        "available": available,
        "total_checked": len(track_ids),
        "total_available": len(available),
    }


@app.post("/preview/upload/{track_id}")
async def upload_preview(track_id: str, file: UploadFile = File(...)):
    """
    Recibe un snippet MP3 desde un engine local o desde el cliente Flutter
    y lo guarda en PREVIEWS_DIR para que el resto de dispositivos puedan
    reproducirlo via GET /preview/{track_id}.

    Autenticación: ninguna (por ahora) — el endpoint acepta cualquier MP3
    dentro de los límites de tamaño. El track_id se sanitiza para evitar
    path traversal. Si luego se considera necesario, se puede proteger con
    HMAC o rate-limit.

    Límites:
      - mínimo 100B (filtro de uploads vacíos/corruptos)
      - máximo 500KB (un snippet de 6s a 64kbps son ~48KB; 500KB da
        margen de sobra para compresiones laxas).
    """
    safe_id = re.sub(r'[^a-fA-F0-9]', '', track_id)
    if not safe_id or len(safe_id) > 64:
        raise HTTPException(400, "track_id inválido")

    content = await file.read()
    if len(content) < 100:
        raise HTTPException(400, "Archivo demasiado pequeño")
    if len(content) > 500_000:
        raise HTTPException(400, "Archivo demasiado grande")

    preview_path = os.path.join(PREVIEWS_DIR, f"{safe_id}.mp3")
    os.makedirs(PREVIEWS_DIR, exist_ok=True)

    with open(preview_path, 'wb') as f:
        f.write(content)

    return {"status": "ok", "track_id": safe_id, "size": len(content)}


# ==================== INFO ====================

@app.get("/")
async def root():
    return {
        "name": "DJ Analyzer Pro API",
        "version": "2.3.0",
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
        "version": "2.6.0",
        "uptime_seconds": uptime_seconds,
        "checks": {
            "database": db_status,
            "ffmpeg": ffmpeg_status,
            "disk_space_mb": disk_space_mb,
        },
    }

# ==================== ADMIN / RESET ====================

def _verify_admin_token(authorization: Optional[str] = None) -> bool:
    """Verifica el ADMIN_TOKEN para endpoints de admin.
    
    No requerimos auth si:
      - DEBUG=true (entorno de dev)
      - ADMIN_TOKEN no esta configurado en env (default a abierto)
    
    Si esta configurado, requiere header "Authorization: Bearer <token>".
    """
    if DEBUG:
        return True
    if not ADMIN_TOKEN:
        return True
    if not authorization or not authorization.startswith('Bearer '):
        return False
    return authorization[7:] == ADMIN_TOKEN


@app.post("/admin/clear-cache")
async def admin_clear_cache(request: Request):
    """Limpia el cache de tracks (BD entera). PELIGROSO en produccion."""
    auth = request.headers.get('authorization')
    if not _verify_admin_token(auth):
        raise HTTPException(401, "ADMIN_TOKEN requerido")
    
    deleted = db.delete_all_tracks() if hasattr(db, 'delete_all_tracks') else 0
    return {"status": "ok", "deleted": deleted}


@app.delete("/admin/track-by-filename")
async def admin_delete_by_filename(request: Request, filename: str = Query(...)):
    """Elimina un track por filename. Util cuando un analisis salio mal y queremos re-analizar."""
    auth = request.headers.get('authorization')
    if not _verify_admin_token(auth):
        raise HTTPException(401, "ADMIN_TOKEN requerido")
    
    deleted = db.delete_track_by_filename(filename)
    return {"status": "ok", "deleted": deleted, "filename": filename}


# ==================== CORRECTIONS (memoria colectiva) ====================

@app.post("/correction")
async def submit_correction(correction: CorrectionRequest):
    """
    Submit una correccion manual del DJ. Se guarda y cuenta para el
    consensus comunitario (>=3 votos coincidentes + winner supera al 2do por >=2).
    """
    # Validar track_id
    track_id = validate_track_id(correction.track_id)
    
    # Validar field
    valid_fields = ['bpm', 'key', 'genre', 'energy_dj', 'track_type', 'camelot', 'artist', 'title']
    if correction.field not in valid_fields:
        raise HTTPException(400, f"field debe ser uno de: {', '.join(valid_fields)}")
    
    # Sanitizar values
    old_value = sanitize_string(correction.old_value or '', max_length=500)
    new_value = sanitize_string(correction.new_value, max_length=500)
    
    db.save_correction(
        track_id=track_id,
        field=correction.field,
        old_value=old_value,
        new_value=new_value,
        fingerprint=correction.fingerprint,
        device_id=correction.device_id,
    )
    
    return {"status": "ok"}


# ==================== COMMUNITY NOTES ====================

class CommunityNoteRequest(BaseModel):
    fingerprint: str
    device_id: str
    note_text: str
    display_name: Optional[str] = 'DJ'
    note_type: Optional[str] = 'general'


@app.post("/community/note")
async def submit_community_note(request: CommunityNoteRequest):
    if not request.fingerprint or not request.device_id or not request.note_text:
        raise HTTPException(400, "fingerprint, device_id y note_text requeridos")
    note_text = sanitize_string(request.note_text, max_length=2000, allow_empty=False, field_name="note_text")
    display_name = sanitize_string(request.display_name or 'DJ', max_length=50)
    note_type = sanitize_string(request.note_type or 'general', max_length=30)
    note_id = db.save_community_note(
        fingerprint=request.fingerprint,
        device_id=request.device_id,
        note_text=note_text,
        display_name=display_name,
        note_type=note_type,
    )
    return {"status": "ok", "id": note_id}


@app.get("/community/notes/{fingerprint}")
async def get_community_notes(fingerprint: str):
    return {"notes": db.get_community_notes(fingerprint)}


@app.post("/community/note/{note_id}/upvote")
async def upvote_note(note_id: int):
    db.upvote_community_note(note_id)
    return {"status": "ok"}


# ==================== TRACK RATINGS ====================

class TrackRatingRequest(BaseModel):
    fingerprint: str
    device_id: str
    rating: int  # 1-5


@app.post("/community/rate")
async def rate_track(request: TrackRatingRequest):
    if not request.fingerprint or not request.device_id:
        raise HTTPException(400, "fingerprint y device_id requeridos")
    if request.rating < 1 or request.rating > 5:
        raise HTTPException(400, "rating debe estar entre 1 y 5")
    summary = db.rate_track(request.fingerprint, request.device_id, request.rating)
    return {"status": "ok", **summary}


@app.get("/community/popularity/{fingerprint}")
async def get_popularity(fingerprint: str, device_id: Optional[str] = None):
    pop = db.get_track_popularity(fingerprint)
    my_rating = db.get_my_rating(fingerprint, device_id) if device_id else 0
    return {"popularity": pop, "my_rating": my_rating}


# ==================== COMMUNITY BEAT GRID ====================

class BeatGridCorrectionRequest(BaseModel):
    fingerprint: str
    device_id: str
    bpm_adjust: float
    beat_offset: float
    original_bpm: float


@app.post("/community/beat-grid")
async def submit_beat_grid(request: BeatGridCorrectionRequest):
    """Submit correccion de beat grid de un DJ"""
    try:
        if not request.fingerprint or not request.device_id:
            raise HTTPException(400, "fingerprint y device_id requeridos")
        db.submit_beat_grid_correction(
            fingerprint=request.fingerprint,
            device_id=request.device_id,
            bpm_adjust=request.bpm_adjust,
            beat_offset=request.beat_offset,
            original_bpm=request.original_bpm,
        )
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Community] Error saving beat grid: {e}")
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/community/beat-grid/{fingerprint}")
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

# Sets de clasificacion por tipo de campo (Fase 5):
# - CATEGORICAL_FIELDS usan modo (winner-by-majority con tiebreak).
# - NUMERIC_FIELDS usan mediana sobre community_overrides.
COMMUNITY_NUMERIC_FIELDS = {'bpm', 'energy'}
COMMUNITY_CATEGORICAL_FIELDS = {
    'track_type', 'key', 'camelot', 'genre', 'subgenre', 'year',
}

# Validacion + normalizacion por field. Devuelve (normalized_value, error)
# donde error es None si OK, o un mensaje 400 si invalido.
def _validate_community_field(field: str, value: str):
    if value is None or (isinstance(value, str) and not value):
        return None, "value requerido"
    normalized = str(value).strip()
    if not normalized:
        return None, "value requerido"
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
        # BPM numerico: validar + colapsar al rango canonico [60, 180] via
        # bpm_utils para que halftime/doubletime contribuyan al mismo bucket.
        try:
            bpm_val = float(normalized)
        except (TypeError, ValueError):
            return None, f"BPM debe ser numerico: {value}"
        if bpm_val <= 0 or bpm_val > 999:
            return None, "BPM fuera de rango valido (0.1-999)"
        from bpm_utils import normalize_bpm_to_canonical
        try:
            canonical = normalize_bpm_to_canonical(bpm_val)
        except ValueError as e:
            return None, str(e)
        # Guardamos como string para mantener schema actual (value TEXT).
        # Format consistente: 1 decimal (ej "128.0").
        return f"{canonical:.1f}", None
    if field == 'energy':
        try:
            e_val = int(float(normalized))
        except (TypeError, ValueError):
            return None, f"Energy debe ser entero: {value}"
        if e_val < 1 or e_val > 10:
            return None, "Energy debe estar entre 1 y 10"
        return str(e_val), None
    if field == 'year':
        try:
            y_val = int(float(normalized))
        except (TypeError, ValueError):
            return None, f"Year debe ser entero: {value}"
        # Usar time.localtime() para evitar agregar import datetime nuevo.
        current_year = time.localtime().tm_year
        if y_val < 1900 or y_val > current_year + 1:
            return None, f"Year fuera de rango (1900-{current_year + 1})"
        return str(y_val), None
    return None, f"field no soportado: {field}"


class CommunityOverrideRequest(BaseModel):
    fingerprint: str
    device_id: str
    field: str
    value: str


def _community_override_response(fingerprint: str, field: str) -> dict:
    """Helper compartido: distribucion + consensus de un (fp, field).

    Fase 5: branching segun tipo de campo.
    - NUMERIC_FIELDS (bpm, energy): consensus = mediana sobre community_overrides
      cuando N >= 3 votos. Implementado en `get_community_consensus_numeric`.
    - CATEGORICAL_FIELDS (track_type, key, camelot, genre, subgenre, year):
      consensus = moda con tiebreak (Fase 4, sin cambios).

    Shape de respuesta uniforme para que el frontend de Fase 4 no rompa:
    siempre devolvemos las mismas keys ('consensus', 'consensus_votes',
    'votes', 'total_voters'). En modo numerico `votes` es alias de la
    distribucion (igual contenido que en modo categorico).
    """
    if field in COMMUNITY_NUMERIC_FIELDS:
        numeric = db.get_community_consensus_numeric(fingerprint, field)
        distribution = numeric['votes_distribution']
        return {
            "fingerprint": fingerprint,
            "field": field,
            "consensus": numeric['consensus'],
            "consensus_votes": numeric['consensus_votes'],
            "votes": distribution,
            "total_voters": numeric['total_voters'],
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


@app.post("/community/override")
async def submit_community_override(request: CommunityOverrideRequest):
    """Recibe un voto de un DJ sobre cualquier campo categorico de un track.

    Campos soportados: track_type, key, camelot, genre, subgenre.
    Un device puede votar 1 campo 1 vez por track; segundo POST sobreescribe.
    Cuando >=3 votos al winner Y supera al 2do por >=2, la respuesta de
    /analyze para ese fingerprint devuelve {field}_source='community'.
    """
    try:
        if not request.fingerprint or not request.device_id or not request.field:
            raise HTTPException(400, "fingerprint, device_id y field requeridos")
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
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/community/override/{field}/{fingerprint}")
async def get_community_override(field: str, fingerprint: str):
    """Devuelve consensus + distribucion de votos para (field, fingerprint).

    Fase 5: rechaza con 400 fields no soportados. Branching numeric vs
    categorical se hace en `_community_override_response`.
    """
    if field not in COMMUNITY_NUMERIC_FIELDS and field not in COMMUNITY_CATEGORICAL_FIELDS:
        raise HTTPException(400, f"Field '{field}' no soportado")
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


# ==================== COMMUNITY TRACK TYPE (Fase 2 backwards-compat) ====================
# Mantenidos para no romper clientes Fase 2 que no se actualizaron al
# endpoint generico. Internamente delegan al generico via DB.

class TrackTypeOverrideRequest(BaseModel):
    fingerprint: str
    device_id: str
    track_type: str

@app.post("/community/track-type")
async def submit_track_type_override(request: TrackTypeOverrideRequest):
    """Legacy Fase 2: voto de track_type. Delega al endpoint generico."""
    proxy = CommunityOverrideRequest(
        fingerprint=request.fingerprint,
        device_id=request.device_id,
        field='track_type',
        value=request.track_type,
    )
    result = await submit_community_override(proxy)
    # Shape Fase 2: 'consensus' string (no objeto con field).
    return {
        "status": result.get("status", "ok"),
        "votes": result.get("votes", {}),
        "consensus": result.get("consensus"),
        "consensus_votes": result.get("consensus_votes", 0),
    }

@app.get("/community/track-type/{fingerprint}")
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
