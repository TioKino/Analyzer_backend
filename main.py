"""
DJ Analyzer Pro API v2.5.0
==========================
Backend principal - Importa funcionalidad de modulos separados

Estructura:
- main.py          - FastAPI app, analysis engine, main endpoints
- models.py        - Pydantic models (AnalysisResult, CorrectionRequest)
- audio_helpers.py - Fingerprint, filename parsing, structure detection, vocals
- bpm_utils.py     - BPM validation and correction (half/double tempo)
- beatport.py      - Beatport search and genre intelligence
- preview_generator.py - 6-second MP3 preview snippet generation
- database.py      - SQLite management
- genre_detection.py - Genre detection with multiple sources
- artwork_and_cuepoints.py - Artwork extraction + cue points
- similar_tracks_endpoint.py - Similar tracks search
- spectral_genre_classifier.py - Spectral genre classification
- routes/          - Extracted route modules (search, library, admin, etc.)
- sync_endpoints.py - Cloud sync endpoints
- validation.py    - Input validation
- config.py        - Configuration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from sync_endpoints import sync_router, admin_sync_router
from routes import (
    search_router, library_router, admin_router, community_router,
    preview_router, media_router,
    init_search, init_library, init_admin, init_community,
    init_preview, init_media,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import logging
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
from spectral_genre_classifier import classify_genre_advanced

# Split modules
from models import AnalysisResult, CorrectionRequest, KEY_TO_CAMELOT, get_camelot
from audio_helpers import (
    sanitize_float, sanitize_for_json, SafeJSONResponse,
    calculate_fingerprint, parse_filename, detect_structure,
    find_drop_timestamp, classify_track_type, detect_vocals_improved,
    get_acousticbrainz_genre,
)
from bpm_utils import validate_beatport_bpm, smart_bpm_correction, try_bpm_double_half
from beatport import search_beatport, find_tracks_in_json, clean_beatport_genre, convert_beatport_key
from preview_generator import generate_preview_snippet as _generate_preview_snippet, init_previews_dir
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

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    print_config()  # Muestra configuracin al arrancar

try:
    from chunked_analyzer import ChunkedAudioAnalyzer, get_chunked_analyzer
    CHUNKED_ANALYZER_ENABLED = True
    logger.info("ChunkedAudioAnalyzer disponible para tracks largos")
except ImportError as e:
    CHUNKED_ANALYZER_ENABLED = False
    logger.warning(f"ChunkedAudioAnalyzer no disponible: {e}")

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
    logger.warning("artwork_and_cuepoints.py no encontrado - funciones deshabilitadas")
    ARTWORK_ENABLED = False
    ARTWORK_CACHE_DIR = "/data/artwork_cache"
    search_artwork_online = None

# Importar clasificador de géneros
try:
    from genre_detection import GenreDetector
    from api_config import DISCOGS_TOKEN
    genre_detector = GenreDetector(discogs_token=DISCOGS_TOKEN)
    GENRE_DETECTOR_ENABLED = True
    logger.info(f"GenreDetector inicializado (Discogs: {'S' if DISCOGS_TOKEN else 'No'})")
except ImportError as e:
    logger.warning(f"genre_detection.py no encontrado: {e}")
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
    logger.warning("similar_tracks_endpoint.py no encontrado")
    SIMILAR_TRACKS_ENABLED = False
    CAMELOT_COMPATIBLE = {}


# ==================== APP ====================

app = FastAPI(title="DJ Analyzer Pro API", version="2.3.0", default_response_class=SafeJSONResponse)
app.include_router(sync_router)
app.include_router(admin_sync_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if not DEBUG else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-Signature", "X-Device-Id", "X-Original-Path"],
)

#  Manejador de errores de validacin
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return SafeJSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "detail": exc.detail,
            "field": getattr(exc, 'field', None)
        }
    )

# Inicializar BD con path de config (no hardcoded)
db = AnalysisDB(db_path=DATABASE_PATH)

# Inicializar y montar route modules
init_search(db, camelot_compatible=CAMELOT_COMPATIBLE)
init_library(db)
init_admin(db, artwork_cache_dir=ARTWORK_CACHE_DIR, artwork_enabled=ARTWORK_ENABLED,
           genre_detector_enabled=GENRE_DETECTOR_ENABLED, similar_tracks_enabled=SIMILAR_TRACKS_ENABLED)
init_community(db)
init_media(db, ARTWORK_CACHE_DIR)

# Preview: init directory + wrapper for preview_generator module
init_previews_dir(PREVIEWS_DIR)

def generate_preview_snippet(file_path, fingerprint, drop_timestamp, duration):
    return _generate_preview_snippet(file_path, fingerprint, drop_timestamp, duration, PREVIEWS_DIR)

init_preview(db, PREVIEWS_DIR, generate_preview_snippet)

app.include_router(search_router)
app.include_router(library_router)
app.include_router(admin_router)
app.include_router(community_router)
app.include_router(preview_router)
app.include_router(media_router)

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
        
    except sqlite3.Error as e:
        logger.error(f"Error buscando en BD colectiva: {e}")
        return None


# ==================== ANALISIS PRINCIPAL ====================

def analyze_audio(file_path: str, fingerprint: str = None) -> AnalysisResult:
    import warnings
    warnings.filterwarnings('ignore')
    
    #  Obtener duracin SIN cargar audio completo
    duration = librosa.get_duration(path=file_path)
    
    #  Si el track es largo (>4 min), usar anlisis por chunks
    if CHUNKED_ANALYZER_ENABLED and duration > CHUNK_ANALYSIS_THRESHOLD:
        logger.info(f"Track largo ({duration/60:.1f} min) - Usando anlisis por chunks")
        return analyze_audio_chunked(file_path, fingerprint, duration)
    
    # Track corto: anlisis tradicional (carga todo en RAM)
    logger.info(f"Track corto ({duration/60:.1f} min) - Usando anlisis tradicional")
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    
    # ==================== ID3 METADATA ====================
    id3_data = {}
    if ARTWORK_ENABLED:
        id3_data = extract_id3_metadata(file_path)
    
    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
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
        logger.debug(f"Key: {key} ({camelot}) [modo ambiguo: margen={mode_margin:.3f}]")
    
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
    logger.debug(f"Energia: raw={energy_raw:.4f} -> DJ level {energy_dj}")
    
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
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"[TrackType] Error clasificando: {e}")
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
        logger.info(f"Buscando genero: {artist_name} - {title_name}")
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
                logger.info(f"Discogs: {genre} | {label} ({year})")
            else:
                logger.debug(f"Discogs: No encontrado")
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            logger.error(f"Error Discogs: {e}")

        # 2. Si no hay Discogs, intentar MusicBrainz
        if genre_source not in ["discogs"]:
            try:
                mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                if mb_result and mb_result.get('genre'):
                    genre = mb_result.get('genre')
                    genre_source = "musicbrainz"
                    logger.info(f"MusicBrainz: {genre}")
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                logger.error(f"Error MusicBrainz: {e}")

    # 3. Si no hay Discogs ni MusicBrainz, usar ID3 (gen(c)rico pero mejor que nada)
    if genre_source == "spectral_analysis" and id3_genre:
        genre = id3_genre
        genre_source = "id3"
        logger.info(f"ID3 (fallback): {genre}")
    
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
        logger.info(f"BEATPORT: Buscando {artist_name} - {title_name}")
        try:
            beatport_data = search_beatport(artist_name, title_name)
            if beatport_data:
                # ===== VALIDAR BPM con correccion inteligente =====
                bp_bpm = beatport_data.get('bpm')
                if bp_bpm:
                    corrected = smart_bpm_correction(bpm, bp_bpm)
                    if corrected is None:
                        logger.warning(f"[Beatport] MATCH RECHAZADO: BPM {bp_bpm} vs local {bpm:.1f} (sin match half/double)")
                        beatport_data = None
                    elif corrected != bp_bpm:
                        logger.info(f"[Beatport] BPM half/double: local {bpm:.1f} -> Beatport {bp_bpm}")

                if beatport_data:
                    # BPM validado
                    if beatport_data.get('bpm'):
                        logger.info(f"[Beatport] BPM: {beatport_data['bpm']} (local: {bpm:.1f})")
                        bpm = beatport_data['bpm']
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
                            logger.info(f"[Beatport] Key: {key} ({camelot})")
                        else:
                            logger.warning(f"[Beatport] Key '{bp_key}' no mapeada a Camelot")

                    # Genero (proteger Discogs/MusicBrainz)
                    if beatport_data.get('genre'):
                        bp_genre = beatport_data['genre']
                        generic_genres = ['Electronic', 'Dance', 'Unknown', 'electronic', 'dance']
                        if genre_source in ['discogs', 'musicbrainz']:
                            logger.debug(f"[Beatport] Genre '{bp_genre}' no sobreescribe '{genre}' ({genre_source})")
                        elif genre in generic_genres or genre_source in ['spectral_analysis', 'chunked_analysis', 'id3']:
                            genre = bp_genre
                            genre_source = 'beatport'
                            logger.info(f"[Beatport] Genre: {genre}")
                        else:
                            logger.debug(f"[Beatport] Genre '{bp_genre}' no sobreescribe '{genre}' ({genre_source})")

                    # Track type hint (si tienes clean_beatport_genre)
                    if beatport_data.get('track_type_hint'):
                        tt_hint = beatport_data['track_type_hint']
                        old_type = track_type
                        track_type = tt_hint
                        track_type_source = 'beatport'
                        logger.info(f"[Beatport] Track type hint: {tt_hint}")
                        if old_type != tt_hint:
                            logger.info(f"[Beatport] Track type override: {old_type} -> {tt_hint}")
            else:
                logger.debug(f"[Beatport] No encontrado")
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            logger.error(f"[Beatport] Error: {e}")

    drop_time = find_drop_timestamp(y, sr, segments)
    
    # ==================== CUE POINTS + BEAT GRID ====================
    cue_points = []

    if ARTWORK_ENABLED:
        cue_points = detect_cue_points(y, sr, duration, segments)

    # Beat grid: SIEMPRE detectar (no depende de ARTWORK_ENABLED)
    # El intervalo se calcula del BPM final (que puede venir de ID3/Beatport)
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    first_beat = 0.0
    try:
        beat_grid = detect_beat_grid(y, sr, bpm)
        first_beat = beat_grid.get('first_beat', 0.0)
        beat_interval = beat_grid.get('beat_interval', beat_interval)
        logger.info(f"[Beat Grid] fb={first_beat:.4f}s iv={beat_interval:.6f}s err={beat_grid.get('grid_error_ms', '?')}ms")
    except (ValueError, TypeError, KeyError, IndexError) as e:
        logger.warning(f"[Beat Grid] Error: {e}, usando defaults")
    
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
            logger.debug(f"Artwork ID3: {artwork_info.get('size', 0)} bytes")
        else:
            # Fallback: buscar online (iTunes/Deezer)
            if artwork_info:
                logger.info(f"Artwork ID3 muy pequeno ({artwork_info.get('size', 0)} bytes), buscando online...")
            else:
                logger.info(f"Sin artwork ID3, buscando online...")
            
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
                    # Usar URL directa de iTunes/Deezer para que funcione en todos los dispositivos
                    artwork_url = online_artwork.get('url') or f"{BASE_URL}/artwork/{fingerprint}"
                    logger.info(f"Artwork {artwork_source}: {online_artwork.get('size', 0)} bytes")
                else:
                    logger.debug(f"No se encontr artwork online")
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                logger.error(f"Error buscando artwork online: {e}")
    
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
                logger.info(f"[Beatport] Track type override: {track_type} -> {bp_type}")
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
        logger.info(f"Buscando genero: {artist_name} - {title_name}")
        try:
            discogs_result = genre_detector.get_discogs_genre(artist_name, title_name)
            if discogs_result and discogs_result.get('genre'):
                genre = discogs_result.get('genre')
                genre_source = "discogs"
                if not label and discogs_result.get('label'):
                    label = discogs_result['label']
                if not year and discogs_result.get('year'):
                    year = str(discogs_result['year'])
                logger.info(f"Discogs: {genre}")
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            logger.error(f"Error Discogs: {e}")

        if genre_source not in ["discogs"]:
            try:
                mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                if mb_result and mb_result.get('genre'):
                    genre = mb_result.get('genre')
                    genre_source = "musicbrainz"
                    logger.info(f"MusicBrainz: {genre}")
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                logger.error(f"Error MusicBrainz: {e}")
    
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
        logger.info(f"BEATPORT: Buscando {artist_name} - {title_name}")
        try:
            beatport_data = search_beatport(artist_name, title_name)
            if beatport_data:
                # ===== VALIDAR BPM con correccion inteligente =====
                bp_bpm = beatport_data.get('bpm')
                if bp_bpm:
                    corrected = smart_bpm_correction(bpm, bp_bpm)
                    if corrected is None:
                        logger.warning(f"[Beatport] MATCH RECHAZADO: BPM {bp_bpm} vs local {bpm:.1f} (sin match half/double)")
                        beatport_data = None
                    elif corrected != bp_bpm:
                        logger.info(f"[Beatport] BPM half/double: local {bpm:.1f} -> Beatport {bp_bpm}")

                if beatport_data:
                    # BPM validado
                    if beatport_data.get('bpm'):
                        logger.info(f"[Beatport] BPM: {beatport_data['bpm']} (local: {bpm:.1f})")
                        bpm = beatport_data['bpm']
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
                            logger.info(f"[Beatport] Key: {key} ({camelot})")
                        else:
                            logger.warning(f"[Beatport] Key '{bp_key}' no mapeada a Camelot")

                    # Genero (proteger Discogs/MusicBrainz)
                    if beatport_data.get('genre'):
                        bp_genre = beatport_data['genre']
                        generic_genres = ['Electronic', 'Dance', 'Unknown', 'electronic', 'dance']
                        if genre_source in ['discogs', 'musicbrainz']:
                            logger.debug(f"[Beatport] Genre '{bp_genre}' no sobreescribe '{genre}' ({genre_source})")
                        elif genre in generic_genres or genre_source in ['spectral_analysis', 'chunked_analysis', 'id3']:
                            genre = bp_genre
                            genre_source = 'beatport'
                            logger.info(f"[Beatport] Genre: {genre}")
                        else:
                            logger.debug(f"[Beatport] Genre '{bp_genre}' no sobreescribe '{genre}' ({genre_source})")

                    # Track type hint (si tienes clean_beatport_genre)
                    if beatport_data.get('track_type_hint'):
                        tt_hint = beatport_data['track_type_hint']
                        old_type = track_type
                        track_type = tt_hint
                        track_type_source = 'beatport'
                        logger.info(f"[Beatport] Track type hint: {tt_hint}")
                        if old_type != tt_hint:
                            logger.info(f"[Beatport] Track type override: {old_type} -> {tt_hint}")
            else:
                logger.debug(f"[Beatport] No encontrado")
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            logger.error(f"[Beatport] Error: {e}")

    # ==================== ARTWORK ====================
    artwork_embedded = False
    artwork_url = None
    
    if ARTWORK_ENABLED and fingerprint:
        artwork_info = extract_artwork_from_file(file_path)
        
        if artwork_info and artwork_info.get('size', 0) > 10000:
            artwork_embedded = True
            save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            logger.debug(f"Artwork ID3: {artwork_info.get('size', 0)} bytes")
        else:
            if artist_name and title_name:
                try:
                    online_artwork = search_artwork_online(artist_name, title_name, id3_data.get('album'))
                    if online_artwork and online_artwork.get('data'):
                        save_artwork_to_cache(fingerprint, online_artwork['data'], online_artwork['mime_type'])
                        artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                        logger.info(f"Artwork online: {online_artwork.get('size', 0)} bytes")
                except (requests.RequestException, ConnectionError, TimeoutError) as e:
                    logger.error(f"Error artwork online: {e}")
    
    # ==================== TRACK TYPE: BEATPORT OVERRIDE ====================
    track_type = result['track_type']
    track_type_source = 'waveform'
    if artist_name and title_name:
        try:
            if beatport_data and beatport_data.get('track_type_hint'):
                bp_type = beatport_data['track_type_hint']
                logger.info(f"[Beatport] Track type override: {track_type} -> {bp_type}")
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
    force: bool = Query(False, description="Forzar reanalisis ignorando cache"),
):
    # Leer path original del cliente (enviado como header)
    original_path = request.headers.get('x-original-path')
    # Rate limiting
    if RATE_LIMIT_ENABLED:
        check_rate_limit(get_client_ip(request))
    
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
        
        # Calcular fingerprint del archivo (Chromaprint basado en audio real)
        fingerprint, chromaprint_raw = calculate_fingerprint(tmp_path)
        
        # NUEVO: Buscar por fingerprint si no se encontro por filename
        # Esto recupera datos de AudD guardados previamente
        if not force:
            existing_by_fp = db.get_track_by_fingerprint(fingerprint)
            if existing_by_fp:
                logger.info(f"[Cache] Track encontrado por fingerprint: {existing_by_fp.get('artist')} - {existing_by_fp.get('title')}")
                
                # Actualizar el filename en la BD para futuras busquedas
                existing_by_fp['filename'] = file.filename
                db.save_track(existing_by_fp)
                
                # Intentar construir respuesta desde analysis_json
                if existing_by_fp.get('analysis_json'):
                    try:
                        analysis_data = json.loads(existing_by_fp['analysis_json'])
                        
                        # Si no tiene preview, generarlo con el archivo temporal
                        preview_file = os.path.join(PREVIEWS_DIR, f"{fingerprint}.mp3")
                        if not os.path.exists(preview_file):
                            src = original_path if (original_path and os.path.exists(original_path)) else tmp_path
                            logger.debug(f"[Preview] Cache FP hit sin snippet, generando para {fingerprint[:8]}...")
                            try:
                                generate_preview_snippet(
                                    file_path=src,
                                    fingerprint=fingerprint,
                                    drop_timestamp=analysis_data.get('drop_timestamp', 30.0),
                                    duration=analysis_data.get('duration', 180.0),
                                )
                            except (FileNotFoundError, IOError, OSError) as e:
                                logger.error(f"[Preview] Error: {e}")

                        # Limpiar archivo temporal antes de retornar
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        return AnalysisResult(**analysis_data)
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
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
                except (ValueError, TypeError, KeyError) as e:
                    logger.error(f"[Cache] Error construyendo resultado desde cache: {e}")
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
        if chromaprint_raw:
            track_data['chromaprint'] = chromaprint_raw
        # Guardar path original del cliente para preview on-the-fly
        if original_path:
            track_data['original_file_path'] = original_path
        db.save_track(track_data)
        
        # Generar preview snippet (no bloquea si falla)
        try:
            logger.debug(f"[Preview] Generando snippet para {fingerprint[:8]}... (tmp={tmp_path})")
            logger.debug(f"[Preview] PREVIEWS_DIR={PREVIEWS_DIR}, drop={result.drop_timestamp:.1f}s, dur={result.duration:.1f}s")
            preview_path = generate_preview_snippet(
                file_path=tmp_path,
                fingerprint=fingerprint,
                drop_timestamp=result.drop_timestamp,
                duration=result.duration,
            )
            if preview_path:
                result.preview_url = f"{BASE_URL}/preview/{fingerprint}"
                logger.info(f"[Preview] OK: {preview_path}")
            else:
                logger.warning(f"[Preview] generate_preview_snippet retorno None")
        except (FileNotFoundError, IOError, OSError) as preview_err:
            logger.error(f"[Preview] Error (no critico): {preview_err}", exc_info=True)
        
        result.fingerprint = fingerprint
        return result
    except (ValueError, TypeError, KeyError, IndexError, IOError, OSError,
            json.JSONDecodeError) as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"ERROR en anlisis de audio:\n{error_detail}")

        # ==================== FALLBACK: Track corrupto ====================
        # Intentar crear resultado bsico con ID3 y/o filename
        logger.warning(f"Intentando fallback para: {file.filename}")
        
        try:
            # Intentar fingerprint del contenido primero
            try:
                fingerprint, _ = calculate_fingerprint(tmp_path)
            except (FileNotFoundError, IOError, OSError, ValueError) as e:
                # Si falla (archivo muy corrupto), usar md5 del nombre
                fingerprint = hashlib.md5(file.filename.encode()).hexdigest()
            
            # Intentar leer metadatos ID3 aunque el audio est(c) corrupto
            id3_data = {}
            if ARTWORK_ENABLED:
                try:
                    id3_data = extract_id3_metadata(tmp_path)
                except (ValueError, TypeError, KeyError, IOError, OSError):
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
            
            logger.info(f"Fallback creado: {artist} - {title} (anlisis pendiente)")
            
            result.fingerprint = fingerprint
            return result
            
        except (ValueError, TypeError, KeyError) as fallback_error:
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
        
        logger.info(f"Identificando track: {file.filename}")
        
        # Calcular fingerprint del CONTENIDO del archivo (igual que en /analyze)
        fingerprint, _ = calculate_fingerprint(tmp_path)
        logger.debug(f"Fingerprint (contenido): {fingerprint[:12]}...")
        
        # ==================== PASO 1: IDENTIFICAR CON AUDD ====================
        audio_to_send = tmp_path
        
        try:
            y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=20, offset=30)
            import soundfile as sf
            fragment_path = tmp_path + "_fragment.wav"
            sf.write(fragment_path, y, sr)
            audio_to_send = fragment_path
            logger.debug(f"Fragmento extrado: 20 seg desde 0:30")
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning(f"No se pudo extraer fragmento, usando archivo completo: {e}")
        
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
        
        logger.info(f"AudD identifico: {artist} - {title}")
        
        # ==================== PASO 2: BUSCAR GENERO EN DISCOGS ====================
        genre = "Electronic"
        genre_source = "default"
        
        if GENRE_DETECTOR_ENABLED and genre_detector and artist and title:
            logger.info(f"Buscando genero: {artist} - {title}")
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
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                logger.error(f"Error Discogs: {e}")

            if genre_source != "discogs":
                try:
                    mb_result = genre_detector.get_musicbrainz_info(artist, title)
                    if mb_result and mb_result.get('genre'):
                        genre = mb_result['genre']
                        genre_source = "musicbrainz"
                        logger.info(f"MusicBrainz: {genre}")
                except (requests.RequestException, ConnectionError, TimeoutError) as e:
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
        beatport_data = None
        
        logger.info(f"Re-analizando audio...")
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
                    logger.debug(f"No encontrado en BD colectiva")
        
        # BEATPORT: SIEMPRE intentar (fuera del except)
        if artist and title:
            logger.info(f"BEATPORT: Buscando {artist} - {title}")
            try:
                beatport_data = search_beatport(artist, title)
                if beatport_data:
                    if beatport_data.get('bpm'):
                        bpm = beatport_data['bpm']
                        bpm_confidence = 0.99
                        bpm_source = 'beatport'
                        logger.info(f"[Beatport] BPM: {bpm}")
                    if beatport_data.get('key'):
                        bp_key = beatport_data['key']
                        bp_camelot = KEY_TO_CAMELOT.get(bp_key, None)
                        if bp_camelot:
                            key = bp_key
                            camelot = bp_camelot
                            key_source = 'beatport'
                            logger.info(f"[Beatport] Key: {key} ({camelot})")
                    if beatport_data.get('genre') and genre_source != 'corrections':
                        bp_genre = beatport_data['genre']
                        is_junk = beatport_data.get('is_junk_genre', False)
                        if not is_junk:
                            genre = bp_genre
                            genre_source = 'beatport'
                            logger.info(f"[Beatport] Genre: {genre}")
                        else:
                            logger.debug(f"[Beatport] Genre '{beatport_data.get('genre_raw', bp_genre)}' descartado (categoria comercial)")
                    if beatport_data.get('track_type_hint'):
                        logger.info(f"[Beatport] Track type hint: {beatport_data['track_type_hint']}")
                    if beatport_data.get('duration') and (not duration or duration == 0):
                        duration = beatport_data['duration']
                else:
                    logger.debug(f"[Beatport] No encontrado")
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                logger.error(f"[Beatport] Error: {e}")

        # ==================== PASO 4: BUSCAR ARTWORK ====================
        artwork_url = None
        artwork_source = None
        
        if artist and title and search_artwork_online:
            logger.info(f"Buscando artwork...")
            artwork_info = search_artwork_online(artist, title)
            if artwork_info:
                save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
                artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                artwork_source = artwork_info.get('source', 'online')
                logger.info(f"Artwork: {artwork_source} ({artwork_info['size']} bytes)")
            else:
                logger.debug(f"No se encontr artwork")
        elif not search_artwork_online:
            logger.warning(f"search_artwork_online no disponible")
        
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
        logger.info(f"Guardado en BD con fingerprint: {fingerprint[:12]}...")
        
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
        
    except (ValueError, TypeError, KeyError, IOError, OSError,
            json.JSONDecodeError, sqlite3.DatabaseError) as e:
        import traceback
        logger.error(f"Error identificando: {traceback.format_exc()}")
        raise HTTPException(500, f"Error: {str(e)}")
    except requests.RequestException as e:
        logger.error(f"Error de red identificando: {e}")
        raise HTTPException(502, f"Error de red: {str(e)}")
    except Exception as e:
        import traceback
        logger.error(f"Error inesperado identificando: {traceback.format_exc()}")
        raise HTTPException(500, f"Error inesperado: {str(e)}")
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
        
        logger.info(f"Reconociendo audio: {file.filename} ({len(content)} bytes)")
        
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
            logger.error(f"AudD error: {audd_response.status_code}")
            raise HTTPException(500, f"Error AudD API: {audd_response.status_code}")
        
        result = audd_response.json()
        
        if result.get('status') != 'success':
            error_msg = result.get('error', {}).get('error_message', 'Unknown error')
            logger.error(f"AudD error: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        track_data = result.get('result')
        
        if not track_data:
            logger.info("No se reconoci ninguna cancin")
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
        
        logger.info(f"Reconocido: {artist} - {title}")
        
        # Buscar si ya tenemos anlisis de este track en la BD
        backend_analysis = None
        existing_tracks = db.search_by_artist(artist, limit=50)
        for track in existing_tracks:
            if track.get('title', '').lower() == title.lower():
                backend_analysis = track
                logger.info(f"Encontrado en biblioteca: {track.get('id')}")
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
        
        return response
        
    except requests.Timeout:
        logger.warning("AudD timeout")
        raise HTTPException(504, "Timeout conectando con AudD")
    except requests.RequestException as e:
        logger.error(f"Error de red en reconocimiento: {e}")
        raise HTTPException(502, f"Error de red: {str(e)}")
    except (ValueError, TypeError, KeyError, IOError, OSError,
            json.JSONDecodeError) as e:
        import traceback
        logger.error(f"Error reconocimiento: {traceback.format_exc()}")
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ==================== EXTRACTED TO ROUTE MODULES ====================
# /check-analyzed, /analysis/{filename}, /artwork/{track_id} → routes/media.py
# /preview/{track_id}, /preview/generate, /previews/check → routes/preview.py
# /search/*, /library/*, /admin/*, /community/* → routes/search.py, library.py, admin.py, community.py

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

# Preview + media endpoints extracted to routes/preview.py and routes/media.py
