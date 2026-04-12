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

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from sync_endpoints import sync_router
from routes.admin_panel import admin_panel_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
import librosa
import numpy as np
import tempfile
import os
import re
import json
import hashlib
import requests
import sqlite3
from typing import Dict, List, Optional
from pydantic import BaseModel
from pydantic import BaseModel
from spectral_genre_classifier import classify_genre_advanced
from config import (
    AUDD_API_TOKEN, 
    DISCOGS_TOKEN, 
    print_config, 
    BASE_URL,
    CORS_ORIGINS,
    DEBUG,
    PREVIEWS_DIR,
    ADMIN_TOKEN,
    RATE_LIMIT_ENABLED,
    DATABASE_PATH,
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
    print(" ChunkedAudioAnalyzer disponible para tracks largos")
except ImportError as e:
    CHUNKED_ANALYZER_ENABLED = False
    print(f" ChunkedAudioAnalyzer no disponible: {e}")

# ==================== MODO LOCAL ====================
# Cuando corre via local_engine.py, tiene toda la CPU del usuario.
# No necesita chunked, puede cargar tracks enteros en RAM,
# y usar análisis más intensivos.
IS_LOCAL_ENGINE = os.environ.get('LOCAL_ENGINE', 'false').lower() == 'true'

if IS_LOCAL_ENGINE:
    CHUNKED_ANALYZER_ENABLED = False  # No necesario: RAM del usuario suficiente
    print("=" * 50)
    print("  MODO LOCAL: Análisis completo activado")
    print("  - Chunked deshabilitado (RAM suficiente)")
    print("  - Track completo en memoria (sr=44100)")
    print("  - Key: Krumhansl-Kessler + multi-pasada")
    print("  - BPM: Smart correction + double/half")
    print("  - Energy: Power curve optimizada")
    print("=" * 50)

# Umbral de duracin para usar anlisis por chunks (en segundos)
# Tracks > 4 minutos usarn el analizador por chunks
# SOLO aplica en Render (IS_LOCAL_ENGINE=false)
CHUNK_ANALYSIS_THRESHOLD = 240  # 4 minutos


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
    print("artwork_and_cuepoints.py no encontrado - funciones deshabilitadas")
    ARTWORK_ENABLED = False
    ARTWORK_CACHE_DIR = "/data/artwork_cache"
    search_artwork_online = None

# Importar clasificador de géneros
try:
    from genre_detection import GenreDetector
    from api_config import DISCOGS_TOKEN
    genre_detector = GenreDetector(discogs_token=DISCOGS_TOKEN)
    GENRE_DETECTOR_ENABLED = True
    print(f"GenreDetector inicializado (Discogs: {'S' if DISCOGS_TOKEN else 'No'})")
except ImportError as e:
    print(f"genre_detection.py no encontrado: {e}")
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
    print("similar_tracks_endpoint.py no encontrado")
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
                print(f"  [Beatport] BPM match directo: {beatport_bpm}")
            elif abs(ratio - 2.0) <= 0.15 or abs(ratio - 0.5) <= 0.15:
                print(f"  [Beatport] BPM half/double corregido: local {local_bpm:.1f} -> Beatport {beatport_bpm}")
            else:
                print(f"  [Beatport] BPM override: local {local_bpm:.1f} -> Beatport {beatport_bpm} (Beatport tiene prioridad)")
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
            print(f"   BPM auto-corregido: {original_bpm:.1f} -> {best_bpm:.1f} (confianza baja: {bpm_confidence:.2f})")
        
        return best_bpm
    except Exception:
        return original_bpm

# ==================== APP ====================

app = FastAPI(title="DJ Analyzer Pro API", version="2.3.0", default_response_class=SafeJSONResponse)
app.include_router(sync_router)
app.include_router(admin_panel_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if not DEBUG else ["*"],
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
        print(f" Error buscando en BD colectiva: {e}")
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
            print(f"  [Beatport] HTTP {response.status_code}")
            return None
        
        # Extraer __NEXT_DATA__
        next_data_match = re.search(
            r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
            response.text, re.DOTALL
        )
        if not next_data_match:
            print(f"  [Beatport] No __NEXT_DATA__ encontrado")
            return None
        
        data = json.loads(next_data_match.group(1))
        
        # Navegar a la ruta de tracks
        # props.pageProps.dehydratedState.queries[0].state.data.tracks.data
        try:
            tracks = data["props"]["pageProps"]["dehydratedState"]["queries"][0]["state"]["data"]["tracks"]["data"]
        except (KeyError, IndexError, TypeError):
            print(f"  [Beatport] Estructura JSON no esperada")
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
        print(f"  [Beatport] Timeout")
        return None
    except Exception as e:
        print(f"  [Beatport] Error: {e}")
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

def get_camelot(key: str) -> str:
    """Convierte key musical a notacion Camelot"""
    return KEY_TO_CAMELOT.get(key, '?')

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

def classify_track_type(energy: float, segments: dict, duration: float) -> str:
    if energy < 0.5 and segments['has_intro']:
        return "warmup"
    if energy > 0.7 and segments['has_drop']:
        return "peak_time"
    if segments['has_outro'] and duration > 300:
        return "closing"
    return "peak_time" if energy > 0.6 else "warmup"

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
        print(f"Error detectando vocals: {e}")
        return False

def get_acousticbrainz_genre(fingerprint=None, artist=None, title=None):
    """AcousticBrainz cerró en 2022. Stub que retorna None para no romper llamadas."""
    return None


# ==================== PREVIEW SNIPPET ====================

import subprocess
from pathlib import Path as PathLib

# Asegurar directorio de previews
PathLib(PREVIEWS_DIR).mkdir(parents=True, exist_ok=True)

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
        print(f"  [Preview] Ya existe snippet para {fingerprint[:8]}...")
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
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(round(start, 2)),
            '-i', file_path,
            '-t', '6',
            '-ac', '1',           # Mono
            '-ab', '64k',         # 64kbps
            '-ar', '22050',       # Sample rate bajo (suficiente para preview)
            '-af', 'afade=t=in:st=0:d=0.3,afade=t=out:st=5.5:d=0.5',  # Fade in/out
            output_path
        ]
        
        proc_result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            check=True,
        )
        
        # Verificar que el archivo se generó y tiene tamaño razonable
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            if size > 1000:  # Mínimo 1KB
                print(f"  [Preview] Snippet generado: {fingerprint[:8]}... "
                      f"({size//1024}KB, start={start:.1f}s)")
                return output_path
            else:
                print(f"  [Preview] Snippet demasiado pequeño ({size}B), eliminando")
                os.unlink(output_path)
                return None
        
        return None
        
    except subprocess.TimeoutExpired:
        print(f"  [Preview] Timeout generando snippet para {fingerprint[:8]}...")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode('utf-8', errors='replace')[:200] if e.stderr else 'unknown'
        print(f"  [Preview] ffmpeg error: {stderr_msg}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None
    except Exception as e:
        print(f"  [Preview] Error generando snippet: {e}")
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
        print(f" Track largo ({duration/60:.1f} min) - Usando anlisis por chunks")
        return analyze_audio_chunked(file_path, fingerprint, duration)
    
    # Track corto: anlisis tradicional (carga todo en RAM)
    print(f" Track corto ({duration/60:.1f} min) - Usando anlisis tradicional")
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    
    # ==================== ID3 METADATA ====================
    id3_data = {}
    if ARTWORK_ENABLED:
        id3_data = extract_id3_metadata(file_path)
    
    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
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
        print(f"   Key: {key} ({camelot}) [modo ambiguo: margen={mode_margin:.3f}]")
    
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
    else:
        normalized = (energy_raw - 0.02) / (0.42 - 0.02)
        powered = normalized ** 0.55  # expande rango bajo-medio
        energy_dj = int(round(1 + powered * 9))
        energy_dj = max(1, min(10, energy_dj))
    
    energy_normalized = energy_dj / 10.0
    print(f"   Energia: raw={energy_raw:.4f} -> DJ level {energy_dj}")
    
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
    
    # Classification
    track_type = 'peak_time'  # default seguro
    try:
        track_type = classify_track_type(energy_normalized, segments, duration)
    except Exception as e:
        print(f"  [TrackType] Error clasificando: {e}")
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
        print(f" Buscando g(c)nero: {artist_name} - {title_name}")
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
                print(f" Discogs: {genre} | {label} ({year})")
            else:
                print(f" Discogs: No encontrado")
        except Exception as e:
            print(f" Error Discogs: {e}")
        
        # 2. Si no hay Discogs, intentar MusicBrainz
        if genre_source not in ["discogs"]:
            try:
                mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                if mb_result and mb_result.get('genre'):
                    genre = mb_result.get('genre')
                    genre_source = "musicbrainz"
                    print(f"   MusicBrainz: {genre}")
            except Exception as e:
                print(f" Error MusicBrainz: {e}")
    
    # 3. Si no hay Discogs ni MusicBrainz, usar ID3 (gen(c)rico pero mejor que nada)
    if genre_source == "spectral_analysis" and id3_genre:
        genre = id3_genre
        genre_source = "id3"
        print(f" ID3 (fallback): {genre}")
    
    # ==================== BEATPORT: FUENTE PRIMARIA BPM/KEY/GENRE ====================
    # Beatport tiene datos 100% precisos del sello discografico
    # Intentar SIEMPRE si tenemos artista y titulo
    
    if not artist_name:
        artist_name = id3_data.get('artist')
    if not title_name:
        title_name = id3_data.get('title')
    
    # Si no hay metadata ID3, intentar con filename parseado
    if not artist_name or not title_name:
        parsed = parse_filename(os.path.basename(file_path))
        if not artist_name:
            artist_name = parsed.get('artist')
        if not title_name:
            title_name = parsed.get('title')
    
    if artist_name and title_name:
        print(f"  BEATPORT: Buscando {artist_name} - {title_name}")
        try:
            beatport_data = search_beatport(artist_name, title_name)
            if beatport_data:
                # ===== BPM: Beatport siempre tiene prioridad =====
                bp_bpm = beatport_data.get('bpm')
                if bp_bpm:
                    corrected = smart_bpm_correction(bpm, bp_bpm)
                    bpm = corrected
                    bpm_source = 'beatport'
                    bpm_confidence = 0.99

                # Key
                if beatport_data.get('key'):
                    bp_key = beatport_data['key']
                    bp_camelot = KEY_TO_CAMELOT.get(bp_key, None)
                    if bp_camelot:
                        key = bp_key
                        camelot = bp_camelot
                        key_source = 'beatport'
                        key_confidence = 0.99
                        print(f"  [Beatport] Key: {key} ({camelot})")
                    else:
                        print(f"  [Beatport] Key '{bp_key}' no mapeada a Camelot")

                # Genero (proteger Discogs/MusicBrainz)
                if beatport_data.get('genre'):
                    bp_genre = beatport_data['genre']
                    generic_genres = ['Electronic', 'Dance', 'Unknown', 'electronic', 'dance']
                    if genre_source in ['discogs', 'musicbrainz']:
                        print(f"  [Beatport] Genre '{bp_genre}' no sobreescribe '{genre}' ({genre_source})")
                    elif genre in generic_genres or genre_source in ['spectral_analysis', 'chunked_analysis', 'id3']:
                        genre = bp_genre
                        genre_source = 'beatport'
                        print(f"  [Beatport] Genre: {genre}")
                    else:
                        print(f"  [Beatport] Genre '{bp_genre}' no sobreescribe '{genre}' ({genre_source})")

                # Track type hint
                if beatport_data.get('track_type_hint'):
                    tt_hint = beatport_data['track_type_hint']
                    old_type = track_type
                    track_type = tt_hint
                    track_type_source = 'beatport'
                    print(f"  [Beatport] Track type hint: {tt_hint}")
                    if old_type != tt_hint:
                        print(f"  [Beatport] Track type override: {old_type} -> {tt_hint}")
            else:
                print(f"  [Beatport] No encontrado")
        except Exception as e:
            print(f"  [Beatport] Error: {e}")
    
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
    
    if ARTWORK_ENABLED and fingerprint:
        artwork_info = extract_artwork_from_file(file_path)
        
        # Verificar que el artwork sea vlido (mnimo 10KB para evitar corruptos/placeholders)
        if artwork_info and artwork_info.get('size', 0) > 10000:
            artwork_embedded = True
            artwork_source = "id3"
            # Guardar en cach(c)
            save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            print(f"   Artwork ID3: {artwork_info.get('size', 0)} bytes")
        else:
            # Fallback: buscar online (iTunes/Deezer)
            if artwork_info:
                print(f" Artwork ID3 muy peque+/-o ({artwork_info.get('size', 0)} bytes), buscando online...")
            else:
                print(f" Sin artwork ID3, buscando online...")
            
            artist_name = id3_data.get('artist')
            title_name = id3_data.get('title')
            album_name = id3_data.get('album')
            
            try:
                from artwork_and_cuepoints import search_artwork_online
                online_artwork = search_artwork_online(artist_name, title_name, album_name)
                
                if online_artwork and online_artwork.get('data'):
                    artwork_embedded = False  # No est embebido, viene de online
                    artwork_source = online_artwork.get('source', 'online')
                    save_artwork_to_cache(fingerprint, online_artwork['data'], online_artwork['mime_type'])
                    artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                    print(f"   Artwork {artwork_source}: {online_artwork.get('size', 0)} bytes")
                else:
                    print(f" No se encontr artwork online")
            except Exception as e:
                print(f" Error buscando artwork online: {e}")
    
    # ==================== TRACK TYPE: BEATPORT OVERRIDE ====================
    # Si Beatport dio un track_type_hint, tiene prioridad sobre waveform
    track_type_source = 'waveform'
    # Guard: asegurar que track_type existe (por si classify_track_type falló)
    try:
        track_type
    except NameError:
        track_type = 'peak_time'
    if artist_name and title_name:
        try:
            if beatport_data and beatport_data.get('track_type_hint'):
                bp_type = beatport_data['track_type_hint']
                print(f"  [Beatport] Track type override: {track_type} -> {bp_type}")
                track_type = bp_type
                track_type_source = 'beatport'
        except NameError:
            pass  # beatport_data no existe si no se ejecuto el bloque
    
    return AnalysisResult(
        title=id3_data.get('title'),
        artist=id3_data.get('artist'),
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

def analyze_audio_chunked(file_path: str, fingerprint: str, duration: float) -> AnalysisResult:
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
        print(f"   Buscando g(c)nero: {artist_name} - {title_name}")
        try:
            discogs_result = genre_detector.get_discogs_genre(artist_name, title_name)
            if discogs_result and discogs_result.get('genre'):
                genre = discogs_result.get('genre')
                genre_source = "discogs"
                if not label and discogs_result.get('label'):
                    label = discogs_result['label']
                if not year and discogs_result.get('year'):
                    year = str(discogs_result['year'])
                print(f"   Discogs: {genre}")
        except Exception as e:
            print(f"   Error Discogs: {e}")
        
        if genre_source not in ["discogs"]:
            try:
                mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                if mb_result and mb_result.get('genre'):
                    genre = mb_result.get('genre')
                    genre_source = "musicbrainz"
                    print(f"   MusicBrainz: {genre}")
            except Exception as e:
                print(f"   Error MusicBrainz: {e}")
    
    if genre_source == "chunked_analysis" and id3_genre:
        genre = id3_genre
        genre_source = "id3"
    
    # ==================== BEATPORT: FUENTE PRIMARIA BPM/KEY/GENRE ====================
    # Beatport tiene datos 100% precisos del sello discografico
    # Intentar SIEMPRE si tenemos artista y titulo
    
    if not artist_name:
        artist_name = id3_data.get('artist')
    if not title_name:
        title_name = id3_data.get('title')
    
    # Si no hay metadata ID3, intentar con filename parseado
    if not artist_name or not title_name:
        parsed = parse_filename(os.path.basename(file_path))
        if not artist_name:
            artist_name = parsed.get('artist')
        if not title_name:
            title_name = parsed.get('title')
    
    if artist_name and title_name:
        print(f"  BEATPORT: Buscando {artist_name} - {title_name}")
        try:
            beatport_data = search_beatport(artist_name, title_name)
            if beatport_data:
                # ===== BPM: Beatport siempre tiene prioridad =====
                bp_bpm = beatport_data.get('bpm')
                if bp_bpm:
                    corrected = smart_bpm_correction(bpm, bp_bpm)
                    bpm = corrected
                    bpm_source = 'beatport'
                    bpm_confidence = 0.99

                # Key
                if beatport_data.get('key'):
                    bp_key = beatport_data['key']
                    bp_camelot = KEY_TO_CAMELOT.get(bp_key, None)
                    if bp_camelot:
                        key = bp_key
                        camelot = bp_camelot
                        key_source = 'beatport'
                        key_confidence = 0.99
                        print(f"  [Beatport] Key: {key} ({camelot})")
                    else:
                        print(f"  [Beatport] Key '{bp_key}' no mapeada a Camelot")

                # Genero (proteger Discogs/MusicBrainz)
                if beatport_data.get('genre'):
                    bp_genre = beatport_data['genre']
                    generic_genres = ['Electronic', 'Dance', 'Unknown', 'electronic', 'dance']
                    if genre_source in ['discogs', 'musicbrainz']:
                        print(f"  [Beatport] Genre '{bp_genre}' no sobreescribe '{genre}' ({genre_source})")
                    elif genre in generic_genres or genre_source in ['spectral_analysis', 'chunked_analysis', 'id3']:
                        genre = bp_genre
                        genre_source = 'beatport'
                        print(f"  [Beatport] Genre: {genre}")
                    else:
                        print(f"  [Beatport] Genre '{bp_genre}' no sobreescribe '{genre}' ({genre_source})")
            else:
                print(f"  [Beatport] No encontrado")
        except Exception as e:
            print(f"  [Beatport] Error: {e}")

    # ==================== ARTWORK ====================
    artwork_embedded = False
    artwork_url = None
    
    if ARTWORK_ENABLED and fingerprint:
        artwork_info = extract_artwork_from_file(file_path)
        
        if artwork_info and artwork_info.get('size', 0) > 10000:
            artwork_embedded = True
            save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            print(f"   Artwork ID3: {artwork_info.get('size', 0)} bytes")
        else:
            if artist_name and title_name:
                try:
                    online_artwork = search_artwork_online(artist_name, title_name, id3_data.get('album'))
                    if online_artwork and online_artwork.get('data'):
                        save_artwork_to_cache(fingerprint, online_artwork['data'], online_artwork['mime_type'])
                        artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                        print(f"   Artwork online: {online_artwork.get('size', 0)} bytes")
                except Exception as e:
                    print(f"   Error artwork online: {e}")
    
    # ==================== TRACK TYPE: BEATPORT OVERRIDE ====================
    track_type = result['track_type']
    track_type_source = 'waveform'
    if artist_name and title_name:
        try:
            if beatport_data and beatport_data.get('track_type_hint'):
                bp_type = beatport_data['track_type_hint']
                print(f"  [Beatport] Track type override: {track_type} -> {bp_type}")
                track_type = bp_type
                track_type_source = 'beatport'
        except NameError:
            pass
    
    # ==================== RESULTADO ====================
    return AnalysisResult(
        title=id3_data.get('title'),
        artist=id3_data.get('artist'),
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

# ==================== ENDPOINTS PRINCIPALES ====================

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_track(
    request: Request, 
    file: UploadFile = File(...),
    force: bool = Query(False, description="Forzar reanalisis ignorando cache")
):
    #  Rate limiting (opcional - descomenta si quieres)
    # check_rate_limit(get_client_ip(request))
    
    # Obtener path original del cliente (para generacion de previews)
    original_path = request.headers.get("X-Original-Path", "")

    #  Validacin mejorada de archivo
    if not file.filename:
        raise HTTPException(400, "No se proporcion archivo")
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg')):
        raise HTTPException(400, "Formato no soportado. Permitidos: mp3, wav, flac, m4a, aac, ogg")
    
    # Leer contenido y validar tama+/-o
    content = await file.read()
    max_size = 100 * 1024 * 1024  # 100 MB
    if len(content) > max_size:
        raise HTTPException(400, f"Archivo demasiado grande. Mximo: 100 MB")
    if len(content) < 1000:
        raise HTTPException(400, "Archivo demasiado peque+/-o o corrupto")
    
    # Si force=true, eliminar registro antiguo para reanalisis completo
    if force:
        db.delete_track_by_filename(file.filename)
    else:
        # Verificar si ya existe en BD por filename
        existing = db.get_track_by_filename(file.filename)
        if existing:
            analysis_json = json.loads(existing[11]) if len(existing) > 11 and existing[11] else {}

            # Si no tiene preview, intentar generarlo ahora
            fp = analysis_json.get('fingerprint') or (existing[13] if len(existing) > 13 else None)
            if fp:
                preview_file = os.path.join(PREVIEWS_DIR, f"{fp}.mp3")
                if not os.path.exists(preview_file) and original_path and os.path.exists(original_path):
                    logger.debug(f"[Preview] Cache hit pero sin snippet, generando para {fp[:8]}...")
                    try:
                        generate_preview_snippet(
                            file_path=original_path,
                            fingerprint=fp,
                            drop_timestamp=analysis_json.get('drop_timestamp', 30.0),
                            duration=analysis_json.get('duration', 180.0),
                        )
                    except (FileNotFoundError, IOError, OSError) as e:
                        logger.error(f"[Preview] Error generando desde cache: {e}")

                # Asegurar que el original_path se guarda para futuras peticiones
                if original_path and not analysis_json.get('original_file_path'):
                    analysis_json['original_file_path'] = original_path
                    existing_dict = db._row_to_dict(existing) or {}
                    existing_dict['analysis_json'] = json.dumps(analysis_json)
                    # Re-save no es trivial, pero al menos guardamos el path
            return AnalysisResult(**analysis_json)
    
    # Guardar archivo temporal para calcular fingerprint
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        # Calcular fingerprint del archivo
        fingerprint = calculate_fingerprint(tmp_path)
        
        # NUEVO: Buscar por fingerprint si no se encontro por filename
        # Esto recupera datos de AudD guardados previamente
        if not force:
            existing_by_fp = db.get_track_by_fingerprint(fingerprint)
            if existing_by_fp:
                print(f"[Cache] Track encontrado por fingerprint: {existing_by_fp.get('artist')} - {existing_by_fp.get('title')}")
                
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
                        print(f"[Cache] Error parseando analysis_json: {e}")
                
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
                    print(f"[Cache] Error construyendo resultado desde cache: {e}")
                    # Continuar con analisis normal si falla
        
        collective_genre = db.get_collective_genre(fingerprint)
        
        result = analyze_audio(tmp_path, fingerprint)
        
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
        
        # Prioridad: Memoria colectiva > Discogs > MusicBrainz > ID3 > AcousticBrainz > Anlisis
        if collective_genre:
            result.genre = collective_genre
            result.genre_source = "collective_memory"
        elif result.genre_source in ["discogs", "musicbrainz"]:
            # Ya tenemos buen g(c)nero, mantenerlo
            pass
        elif result.genre_source == "id3":
            # ID3 est bien, pero si hay AcousticBrainz especfico, usarlo
            if ab_genre and ab_genre.lower() not in ["electronic", "dance"]:
                result.subgenre = result.genre
                result.genre = ab_genre
                result.genre_source = "acousticbrainz"
        elif ab_genre:
            # Fallback a AcousticBrainz
            result.genre = ab_genre
            result.genre_source = "acousticbrainz"
        
        # Guardar en BD
        track_data = result.dict()
        track_data['id'] = fingerprint
        track_data['filename'] = file.filename
        track_data['fingerprint'] = fingerprint
        db.save_track(track_data)
        
        # Generar preview snippet (no bloquea si falla)
        try:
            preview_path = generate_preview_snippet(
                file_path=tmp_path,
                fingerprint=fingerprint,
                drop_timestamp=result.drop_timestamp,
                duration=result.duration,
            )
            if preview_path:
                result.preview_url = f"{BASE_URL}/preview/{fingerprint}"
        except Exception as preview_err:
            print(f"  [Preview] Error (no crítico): {preview_err}")
        
        result.fingerprint = fingerprint
        return result
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR en anlisis de audio:\n{error_detail}")
        
        # ==================== FALLBACK: Track corrupto ====================
        # Intentar crear resultado bsico con ID3 y/o filename
        print(f" Intentando fallback para: {file.filename}")
        
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
                except:
                    pass
            
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
                genre=id3_data.get('genre', 'Unknown'),
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
            track_data = result.dict()
            track_data['id'] = fingerprint
            track_data['filename'] = file.filename
            track_data['fingerprint'] = fingerprint
            track_data['analysis_status'] = 'failed'  # Marcador especial
            db.save_track(track_data)
            
            print(f"Fallback creado: {artist} - {title} (anlisis pendiente)")
            
            result.fingerprint = fingerprint
            return result
            
        except Exception as fallback_error:
            print(f"Fallback tambi(c)n fall: {fallback_error}")
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
    
    db.save_correction(track_id, field, old_value, new_value, request.fingerprint)
    return {"status": "ok", "message": "Correccin guardada"}

# ==================== IDENTIFICAR TRACK CON AUDD ====================

@app.post("/identify")
async def identify_track(file: UploadFile = File(...)):
    """
    Identifica un track usando AudD y hace RE-ANALISIS COMPLETO.
    
    Flujo:
    1. AudD identifica artista/ttulo
    2. Busca g(c)nero en Discogs con el nuevo nombre
    3. Intenta re-analizar audio (BPM, Key, Energy)
    4. Busca artwork online
    5. Actualiza todo en BD
    """
    try:
        from api_config import AUDD_API_TOKEN
    except ImportError:
        AUDD_API_TOKEN = None
    
    if not AUDD_API_TOKEN:
        raise HTTPException(500, "AudD API token no configurado")
    
    tmp_path = None
    fragment_path = None
    
    try:
        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Identificando track: {file.filename}")
        
        # Calcular fingerprint del CONTENIDO del archivo (igual que en /analyze)
        fingerprint = calculate_fingerprint(tmp_path)
        print(f"  Fingerprint (contenido): {fingerprint[:12]}...")
        
        # ==================== PASO 1: IDENTIFICAR CON AUDD ====================
        audio_to_send = tmp_path
        
        try:
            y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=20, offset=30)
            import soundfile as sf
            fragment_path = tmp_path + "_fragment.wav"
            sf.write(fragment_path, y, sr)
            audio_to_send = fragment_path
            print(f"Fragmento extrado: 20 seg desde 0:30")
        except Exception as e:
            print(f"No se pudo extraer fragmento, usando archivo completo: {e}")
        
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
        
        print(f"  AudD identifico: {artist} - {title}")
        
        # ==================== PASO 2: BUSCAR GENERO EN DISCOGS ====================
        genre = "Electronic"
        genre_source = "default"
        
        if GENRE_DETECTOR_ENABLED and genre_detector and artist and title:
            print(f"Buscando g(c)nero: {artist} - {title}")
            try:
                discogs_result = genre_detector.get_discogs_genre(artist, title)
                if discogs_result and discogs_result.get('genre'):
                    genre = discogs_result['genre']
                    genre_source = "discogs"
                    if not label and discogs_result.get('label'):
                        label = discogs_result['label']
                    if not year and discogs_result.get('year'):
                        year = str(discogs_result['year'])
                    print(f"Discogs: {genre} | {label} ({year})")
            except Exception as e:
                print(f"Error Discogs: {e}")
            
            if genre_source != "discogs":
                try:
                    mb_result = genre_detector.get_musicbrainz_info(artist, title)
                    if mb_result and mb_result.get('genre'):
                        genre = mb_result['genre']
                        genre_source = "musicbrainz"
                        print(f"MusicBrainz: {genre}")
                except Exception as e:
                    print(f"Error MusicBrainz: {e}")
        
        # ==================== PASO 3: RE-ANALIZAR AUDIO ====================
        bpm = None
        bpm_confidence = 0.0
        key = None
        camelot = None
        energy_dj = 5
        duration = 0.0
        bpm_source = 'pending'
        key_source = 'pending'
        beatport_data = None
        
        print(f"Re-analizando audio...")
        try:
            # Local: sr=44100 para maxima precision. Render: sr=22050 para ahorrar RAM.
            analysis_sr = 44100 if IS_LOCAL_ENGINE else 22050
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
            print(f"BPM: {bpm} (confianza: {bpm_confidence:.2f})")
            
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
            print(f"Key: {key} ({camelot})")
            
            # MEJORA 2: Energy con power curve
            rms = librosa.feature.rms(y=y_full)[0]
            avg_rms = float(np.mean(rms))
            if avg_rms <= 0.02:
                energy_dj = 1
            elif avg_rms >= 0.42:
                energy_dj = 10
            else:
                normalized = (avg_rms - 0.02) / (0.42 - 0.02)
                powered = normalized ** 0.55
                energy_dj = int(round(1 + powered * 9))
                energy_dj = max(1, min(10, energy_dj))
            print(f"Energy: {energy_dj} (raw: {avg_rms:.4f})")
            
        except Exception as e:
            print(f"Re-anlisis fall: {e}")
            # FALLBACK 1: Buscar en BD colectiva
            if artist and title:
                print(f"Buscando en BD colectiva...")
                collective_data = search_collective_db(artist, title)
                if collective_data:
                    if collective_data.get('bpm'):
                        bpm = collective_data['bpm']
                        bpm_confidence = 0.9
                        bpm_source = 'collective'
                        print(f"BD Colectiva BPM: {bpm}")
                    if collective_data.get('key'):
                        key = collective_data['key']
                        camelot = collective_data.get('camelot') or get_camelot(key)
                        key_source = 'collective'
                        print(f"BD Colectiva Key: {key} ({camelot})")
                    if collective_data.get('duration') and collective_data['duration'] > 0:
                        duration = collective_data['duration']
                else:
                    print(f"No encontrado en BD colectiva")
        
        # BEATPORT: SIEMPRE intentar (fuera del except)
        if artist and title:
            print(f"  BEATPORT: Buscando {artist} - {title}")
            try:
                beatport_data = search_beatport(artist, title)
                if beatport_data:
                    if beatport_data.get('bpm'):
                        bpm = beatport_data['bpm']
                        bpm_confidence = 0.99
                        bpm_source = 'beatport'
                        print(f"  [Beatport] BPM: {bpm}")
                    if beatport_data.get('key'):
                        bp_key = beatport_data['key']
                        bp_camelot = KEY_TO_CAMELOT.get(bp_key, None)
                        if bp_camelot:
                            key = bp_key
                            camelot = bp_camelot
                            key_source = 'beatport'
                            print(f"  [Beatport] Key: {key} ({camelot})")
                    if beatport_data.get('genre') and genre_source != 'corrections':
                        bp_genre = beatport_data['genre']
                        is_junk = beatport_data.get('is_junk_genre', False)
                        if not is_junk:
                            genre = bp_genre
                            genre_source = 'beatport'
                            print(f"  [Beatport] Genre: {genre}")
                        else:
                            print(f"  [Beatport] Genre '{beatport_data.get('genre_raw', bp_genre)}' descartado (categoria comercial)")
                    if beatport_data.get('track_type_hint'):
                        print(f"  [Beatport] Track type hint: {beatport_data['track_type_hint']}")
                    if beatport_data.get('duration') and (not duration or duration == 0):
                        duration = beatport_data['duration']
                else:
                    print(f"  [Beatport] No encontrado")
            except Exception as e:
                print(f"  [Beatport] Error: {e}")
        
        # ==================== PASO 4: BUSCAR ARTWORK ====================
        artwork_url = None
        artwork_source = None
        
        if artist and title and search_artwork_online:
            print(f"   Buscando artwork...")
            artwork_info = search_artwork_online(artist, title)
            if artwork_info:
                save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
                artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                artwork_source = artwork_info.get('source', 'online')
                print(f"Artwork: {artwork_source} ({artwork_info['size']} bytes)")
            else:
                print(f"No se encontr artwork")
        elif not search_artwork_online:
            print(f"search_artwork_online no disponible")
        
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
            'track_type': (beatport_data.get('track_type_hint') or 'peak_time') if beatport_data else 'peak_time',
            'track_type_source': 'beatport' if (beatport_data and beatport_data.get('track_type_hint')) else 'waveform',
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
        print(f"  Guardado en BD con fingerprint: {fingerprint[:12]}...")
        
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
        print(f"Error identificando: {traceback.format_exc()}")
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if fragment_path and os.path.exists(fragment_path):
            os.unlink(fragment_path)

# ==================== RECONOCIMIENTO DE AUDIO (SHAZAM-LIKE) ====================

@app.post("/recognize")
async def recognize_audio(file: UploadFile = File(...)):
    """
    Reconoce una cancin a partir de audio grabado usando AudD API.
    Similar a Shazam - graba audio ambiente y lo identifica.
    """
    try:
        from api_config import AUDD_API_TOKEN
    except ImportError:
        AUDD_API_TOKEN = None
    
    if not AUDD_API_TOKEN:
        raise HTTPException(500, "AudD API token no configurado en api_config.py")
    
    # Guardar archivo temporal
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.aac') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Reconociendo audio: {file.filename} ({len(content)} bytes)")
        
        # Enviar a AudD API
        with open(tmp_path, 'rb') as audio_file:
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
            print(f" AudD error: {audd_response.status_code}")
            raise HTTPException(500, f"Error AudD API: {audd_response.status_code}")
        
        result = audd_response.json()
        
        if result.get('status') != 'success':
            error_msg = result.get('error', {}).get('error_message', 'Unknown error')
            print(f"AudD error: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        track_data = result.get('result')
        
        if not track_data:
            print("No se reconoci ninguna cancin")
            return {"status": "not_found", "message": "No se pudo identificar la cancin"}
        
        # Extraer datos
        artist = track_data.get('artist', 'Unknown Artist')
        title = track_data.get('title', 'Unknown Title')
        album = track_data.get('album')
        release_date = track_data.get('release_date')
        label = track_data.get('label')
        isrc = track_data.get('isrc')
        
        # Datos de servicios
        spotify_data = track_data.get('spotify')
        deezer_data = track_data.get('deezer')
        apple_music_data = track_data.get('apple_music')
        
        print(f"Reconocido: {artist} - {title}")
        
        # Buscar si ya tenemos anlisis de este track en la BD
        backend_analysis = None
        existing_tracks = db.search_by_artist(artist, limit=50)
        for track in existing_tracks:
            if track.get('title', '').lower() == title.lower():
                backend_analysis = track
                print(f"Encontrado en biblioteca: {track.get('id')}")
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

        # Si tenemos anlisis previo, incluirlo
        if backend_analysis:
            response["backend_analysis"] = backend_analysis

        # Guardar reconocimiento en BD colectiva para enriquecer futuras consultas
        # (artwork, genero, label, etc. disponibles para todos los usuarios)
        if not backend_analysis and artist and title:
            try:
                # Buscar artwork del track detectado
                artwork_url = None
                if search_artwork_online:
                    artwork_info = search_artwork_online(artist, title)
                    if artwork_info:
                        # Generar un ID estable basado en artist+title
                        detect_id = hashlib.md5(f"{artist.lower().strip()}|{title.lower().strip()}".encode()).hexdigest()
                        save_artwork_to_cache(detect_id, artwork_info['data'], artwork_info['mime_type'])
                        artwork_url = f"{BASE_URL}/artwork/{detect_id}"
                        response["artwork_url"] = artwork_url
                        logger.info(f"Artwork guardado para deteccion: {detect_id[:12]}")

                # Guardar datos basicos en BD colectiva
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
                # Solo guardar si no existe ya (no sobreescribir datos mejores)
                existing = db.get_track(detect_id)
                if not existing:
                    db.save_track(detect_data)
                    logger.info(f"Deteccion guardada en BD colectiva: {detect_id[:12]}")
            except Exception as e:
                logger.error(f"Error guardando deteccion en BD: {e}")

        return response
        
    except requests.Timeout:
        print("AudD timeout")
        raise HTTPException(504, "Timeout conectando con AudD")
    except Exception as e:
        import traceback
        print(f"Error reconocimiento: {traceback.format_exc()}")
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

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
        except:
            pass
        
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

@app.get("/artwork/{track_id}")
async def get_artwork(track_id: str):
    """Devuelve el artwork de un track como imagen"""
    for ext in ['jpg', 'png', 'jpeg']:
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            media_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else "image/png"
            return FileResponse(cache_path, media_type=media_type)
    
    raise HTTPException(404, "Artwork no encontrado")

# ==================== ENDPOINTS DE BUSQUEDA ====================

@app.get("/search/artist/{artist}")
async def search_by_artist(artist: str, limit: int = Query(50, ge=1, le=200)):
    """Buscar tracks por artista"""
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
                except:
                    pass
            
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
        print(f"Error en search-analyzed: {e}")
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
    return {"status": "healthy", "version": "2.3.0"}

# ==================== ADMIN / RESET ====================

@app.delete("/admin/reset-database")
async def reset_database(confirm: str = Query(..., description="Escribe 'CONFIRMAR' para borrar")):
    """
    PELIGROSO: Borra TODA la base de datos.
    Requiere confirmar escribiendo 'CONFIRMAR' como parmetro.
    """
    if confirm != "CONFIRMAR":
        raise HTTPException(400, "Debes escribir 'CONFIRMAR' para borrar la base de datos")
    
    try:
        import shutil
        
        # Borrar artwork cache
        if os.path.exists(ARTWORK_CACHE_DIR):
            shutil.rmtree(ARTWORK_CACHE_DIR)
            os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)
        
        # Borrar y recrear BD
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM tracks")
        c.execute("DELETE FROM corrections")
        c.execute("DELETE FROM dj_notes")
        conn.commit()
        conn.close()
        
        return {
            "status": "ok",
            "message": "Base de datos reseteada completamente",
            "artwork_cache": "limpiado",
            "tracks": "eliminados",
            "corrections": "eliminadas"
        }
    except Exception as e:
        raise HTTPException(500, f"Error reseteando: {str(e)}")

@app.delete("/admin/clear-artwork-cache")
async def clear_artwork_cache():
    """Limpia solo el cach(c) de artwork"""
    import shutil
    try:
        if os.path.exists(ARTWORK_CACHE_DIR):
            shutil.rmtree(ARTWORK_CACHE_DIR)
            os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)
        return {"status": "ok", "message": "Cach(c) de artwork limpiado"}
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

# ==================== COMMUNITY BEAT GRID ====================

class BeatGridCorrectionRequest(BaseModel):
    fingerprint: str
    device_id: str
    bpm_adjust: float = 0.0
    beat_offset: float = 0.0
    original_bpm: float = 0.0

@app.post("/community/beat-grid")
async def submit_beat_grid_correction(request: BeatGridCorrectionRequest):
    """Recibe correccion de beat grid de un DJ"""
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
        print(f"[Community] Error saving beat grid: {e}")
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/community/beat-grid/{fingerprint}")
async def get_community_beat_grid(fingerprint: str):
    """Obtiene la correccion promedio de la comunidad"""
    try:
        result = db.get_community_beat_grid(fingerprint)
        return result
    except Exception as e:
        print(f"[Community] Error fetching beat grid: {e}")
        return {"bpm_adjust": 0.0, "beat_offset": 0.0, "contributors": 0, "validated": False}
