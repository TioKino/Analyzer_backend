"""
DJ Analyzer Pro API v2.3.0
==========================
Backend principal - Importa funcionalidad de módulos separados

Estructura:
- main.py (este archivo) - FastAPI app, endpoints principales
- database.py - Gestión de SQLite
- genre_detection.py - Detección de género con múltiples fuentes
- artwork_and_cuepoints.py - Extracción artwork + cue points
- similar_tracks_endpoint.py - Búsqueda de tracks similares
- essentia_analyzer.py - Análisis con Essentia (opcional)
- api_config.py / config.py - Configuración
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
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
)

# 🆕 Importar módulo de validación
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
    print_config()  # Muestra configuración al arrancar

try:
    from chunked_analyzer import ChunkedAudioAnalyzer, get_chunked_analyzer
    CHUNKED_ANALYZER_ENABLED = True
    print("✅ ChunkedAudioAnalyzer disponible para tracks largos")
except ImportError as e:
    CHUNKED_ANALYZER_ENABLED = False
    print(f"⚠️ ChunkedAudioAnalyzer no disponible: {e}")

# Umbral de duración para usar análisis por chunks (en segundos)
# Tracks > 4 minutos usarán el analizador por chunks
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
    print("âš ï¸ artwork_and_cuepoints.py no encontrado - funciones deshabilitadas")
    ARTWORK_ENABLED = False
    ARTWORK_CACHE_DIR = "/data/artwork_cache"
    search_artwork_online = None

# Importar clasificador espectral de géneros
from spectral_genre_classifier import classify_genre_advanced
try:
    from genre_detection import GenreDetector
    from api_config import DISCOGS_TOKEN
    genre_detector = GenreDetector(discogs_token=DISCOGS_TOKEN)
    GENRE_DETECTOR_ENABLED = True
    print(f"âœ… GenreDetector inicializado (Discogs: {'Sí' if DISCOGS_TOKEN else 'No'})")
except ImportError as e:
    print(f"âš ï¸ genre_detection.py no encontrado: {e}")
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
    print("âš ï¸ similar_tracks_endpoint.py no encontrado")
    SIMILAR_TRACKS_ENABLED = False

# ==================== APP ====================

app = FastAPI(title="DJ Analyzer Pro API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if not DEBUG else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# 🆕 Manejador de errores de validación
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

# Inicializar BD
db = AnalysisDB()

# Crear directorio de caché para artwork
os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)

# ==================== BÃšSQUEDA EN BD COLECTIVA ====================

def search_collective_db(artist: str, title: str) -> Optional[Dict]:
    """
    Busca BPM y Key en nuestra BD colectiva.
    Si otro usuario ya analizó este track con éxito, usamos esos datos.
    """
    try:
        # Normalizar para búsqueda
        clean_artist = artist.lower().strip()
        clean_title = re.sub(r'\s*\(?(Original Mix|Extended Mix|Radio Edit|Remix)\)?', '', title, flags=re.IGNORECASE).lower().strip()
        
        # Buscar en BD por artista y título similar
        conn = db.conn
        cursor = conn.cursor()
        
        # Búsqueda exacta primero
        cursor.execute("""
            SELECT bpm, key, camelot, duration, genre, label, bpm_source, key_source
            FROM tracks 
            WHERE LOWER(artist) = ? AND LOWER(title) LIKE ?
            AND bpm IS NOT NULL AND bpm > 0
            ORDER BY analyzed_at DESC
            LIMIT 1
        """, (clean_artist, f"%{clean_title}%"))
        
        row = cursor.fetchone()
        
        if not row:
            # Búsqueda más flexible
            cursor.execute("""
                SELECT bpm, key, camelot, duration, genre, label, bpm_source, key_source
                FROM tracks 
                WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
                AND bpm IS NOT NULL AND bpm > 0
                ORDER BY analyzed_at DESC
                LIMIT 1
            """, (f"%{clean_artist}%", f"%{clean_title}%"))
            row = cursor.fetchone()
        
        if row:
            result = {
                'bpm': row[0],
                'key': row[1],
                'camelot': row[2],
                'duration': row[3],
                'genre': row[4],
                'label': row[5],
                'bpm_source': row[6],
                'key_source': row[7],
                'source': 'collective_db'
            }
            return result
        
        return None
        
    except Exception as e:
        print(f"    âš ï¸ Error buscando en BD colectiva: {e}")
        return None


# ==================== BEATPORT SEARCH ====================

def search_beatport(artist: str, title: str) -> Optional[Dict]:
    """
    Busca BPM y Key de un track en Beatport.
    Beatport tiene datos muy precisos proporcionados por los sellos.
    """
    try:
        import urllib.parse
        
        # Limpiar título (quitar "Original Mix", "Remix", etc. para mejor búsqueda)
        clean_title = re.sub(r'\s*\(?(Original Mix|Extended Mix|Radio Edit)\)?', '', title, flags=re.IGNORECASE).strip()
        
        # Construir query de búsqueda
        query = f"{artist} {clean_title}"
        encoded_query = urllib.parse.quote(query)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Método 1: API de búsqueda de Beatport
        api_url = f"https://www.beatport.com/api/v4/catalog/search/?q={encoded_query}&type=tracks&per_page=10"
        
        try:
            api_response = requests.get(api_url, headers=headers, timeout=15)
            
            if api_response.status_code == 200:
                data = api_response.json()
                tracks = data.get('tracks', data.get('results', []))
                
                if isinstance(tracks, dict):
                    tracks = tracks.get('data', [])
                
                for track in tracks[:10]:
                    track_name = track.get('name', track.get('title', '')).lower()
                    track_artists = track.get('artists', [])
                    
                    # Verificar artista
                    artist_match = False
                    for a in track_artists:
                        a_name = a.get('name', '') if isinstance(a, dict) else str(a)
                        if artist.lower() in a_name.lower() or a_name.lower() in artist.lower():
                            artist_match = True
                            break
                    
                    # Verificar título
                    title_match = clean_title.lower() in track_name or track_name in clean_title.lower()
                    
                    if title_match and (artist_match or not track_artists):
                        result = {}
                        
                        # BPM
                        if track.get('bpm'):
                            result['bpm'] = float(track['bpm'])
                        
                        # Key - puede venir como objeto o string
                        key_data = track.get('key')
                        if key_data:
                            if isinstance(key_data, dict):
                                # Formato: {"id": 1, "name": "D Minor", "camelot": "7A"}
                                key_name = key_data.get('name', key_data.get('shortName', ''))
                                result['key'] = convert_beatport_key(key_name)
                            else:
                                result['key'] = convert_beatport_key(str(key_data))
                        
                        # Duración
                        if track.get('length_ms'):
                            result['duration'] = track['length_ms'] / 1000
                        elif track.get('length'):
                            # Formato "M:SS" o segundos
                            length = track['length']
                            if isinstance(length, str) and ':' in length:
                                parts = length.split(':')
                                result['duration'] = int(parts[0]) * 60 + int(parts[1])
                            else:
                                result['duration'] = float(length)
                        
                        # Género de Beatport (más específico)
                        if track.get('genre'):
                            genre_data = track['genre']
                            if isinstance(genre_data, dict):
                                result['genre'] = genre_data.get('name', '')
                            else:
                                result['genre'] = str(genre_data)
                        
                        if result.get('bpm') or result.get('key'):
                            return result
        except Exception as api_error:
            print(f"    âš ï¸ API Beatport falló: {api_error}")
        
        # Método 2: Scraping HTML como fallback
        try:
            from bs4 import BeautifulSoup
            
            search_url = f"https://www.beatport.com/search?q={encoded_query}"
            html_response = requests.get(search_url, headers={**headers, 'Accept': 'text/html'}, timeout=15)
            
            if html_response.status_code == 200:
                soup = BeautifulSoup(html_response.text, 'html.parser')
                
                # Buscar datos en scripts JSON embebidos
                scripts = soup.find_all('script', type='application/json')
                for script in scripts:
                    try:
                        json_data = json.loads(script.string)
                        # Navegar la estructura para encontrar tracks
                        tracks = find_tracks_in_json(json_data, [])
                        
                        for track in tracks:
                            if isinstance(track, dict):
                                track_name = track.get('name', track.get('title', '')).lower()
                                if clean_title.lower() in track_name:
                                    result = {}
                                    if track.get('bpm'):
                                        result['bpm'] = float(track['bpm'])
                                    if track.get('key'):
                                        key_data = track['key']
                                        if isinstance(key_data, dict):
                                            result['key'] = convert_beatport_key(key_data.get('name', ''))
                                        else:
                                            result['key'] = convert_beatport_key(str(key_data))
                                    if result:
                                        return result
                    except:
                        continue
        except Exception as scrape_error:
            print(f"    âš ï¸ Scraping Beatport falló: {scrape_error}")
        
        return None
        
    except Exception as e:
        print(f"    âš ï¸ Error buscando en Beatport: {e}")
        return None


def find_tracks_in_json(obj, results):
    """Busca recursivamente tracks en estructura JSON"""
    if isinstance(obj, dict):
        # Si tiene bpm y name, probablemente es un track
        if 'bpm' in obj and ('name' in obj or 'title' in obj):
            results.append(obj)
        # Buscar en valores
        for v in obj.values():
            find_tracks_in_json(v, results)
    elif isinstance(obj, list):
        for item in obj:
            find_tracks_in_json(item, results)
    return results


def convert_beatport_key(beatport_key: str) -> str:
    """
    Convierte key de formato Beatport a formato estándar.
    Beatport: "D Minor", "G Major", "F# Minor"
    Estándar: "Dm", "G", "F#m"
    """
    if not beatport_key:
        return None
    
    key = beatport_key.strip()
    
    # Ya está en formato corto
    if len(key) <= 3:
        return key
    
    # Convertir "D Minor" -> "Dm", "G Major" -> "G"
    key = key.replace(' Minor', 'm').replace(' minor', 'm')
    key = key.replace(' Major', '').replace(' major', '')
    key = key.replace(' min', 'm').replace(' maj', '')
    
    return key

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
    genre: str
    subgenre: Optional[str] = None
    genre_source: str = "spectral_analysis"
    has_vocals: bool
    has_heavy_bass: bool
    has_pads: bool
    percussion_density: float
    mix_energy_start: float
    mix_energy_end: float
    drop_timestamp: float
    # 🆕 Cue Points
    cue_points: List[Dict] = []
    # 🆕 Beat Grid
    first_beat: float = 0.0
    beat_interval: float = 0.5
    # 🆕 Artwork
    artwork_embedded: bool = False
    artwork_url: Optional[str] = None

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
        return "peak"
    if segments['has_outro'] and duration > 300:
        return "closing"
    return "peak" if energy > 0.6 else "warmup"

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
    if not artist or not title:
        return None
    
    try:
        search_url = "https://musicbrainz.org/ws/2/recording/"
        params = {
            'query': f'artist:"{artist}" AND recording:"{title}"',
            'fmt': 'json',
            'limit': 1
        }
        
        headers = {'User-Agent': 'DJAnalyzerPro/2.3'}
        response = requests.get(search_url, params=params, headers=headers, timeout=5)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if not data.get('recordings'):
            return None
        
        mbid = data['recordings'][0]['id']
        
        ab_url = f"https://acousticbrainz.org/api/v1/{mbid}/high-level"
        ab_response = requests.get(ab_url, timeout=5)
        
        if ab_response.status_code != 200:
            return None
        
        ab_data = ab_response.json()
        
        genre_data = ab_data.get('highlevel', {}).get('genre_rosamerica', {})
        if genre_data:
            genre = genre_data.get('value', '')
            probability = genre_data.get('probability', 0)
            
            if probability > 0.5:
                return genre
        
        return None
        
    except Exception as e:
        print(f"Error AcousticBrainz: {e}")
        return None

# ==================== ANÃLISIS PRINCIPAL ====================

def analyze_audio(file_path: str, fingerprint: str = None) -> AnalysisResult:
    import warnings
    warnings.filterwarnings('ignore')
    
    # 🆕 Obtener duración SIN cargar audio completo
    duration = librosa.get_duration(path=file_path)
    
    # 🆕 Si el track es largo (>4 min), usar análisis por chunks
    if CHUNKED_ANALYZER_ENABLED and duration > CHUNK_ANALYSIS_THRESHOLD:
        print(f"📦 Track largo ({duration/60:.1f} min) - Usando análisis por chunks")
        return analyze_audio_chunked(file_path, fingerprint, duration)
    
    # Track corto: análisis tradicional (carga todo en RAM)
    print(f"⚡ Track corto ({duration/60:.1f} min) - Usando análisis tradicional")
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
    
    if len(beat_intervals) > 1:
        groove_score = min(np.std(beat_intervals) * 10, 1.0)
        swing_factor = float(np.mean(beat_intervals[::2]) / np.mean(beat_intervals[1::2]) 
                           if len(beat_intervals) > 2 else 0.5)
    else:
        groove_score = 0.0
        swing_factor = 0.5
    
    # Key
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = np.argmax(np.sum(chroma, axis=1))
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    
    chroma_mean = np.mean(chroma, axis=1)
    major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, key_idx))[0, 1]
    minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, key_idx))[0, 1]
    
    is_minor = minor_corr > major_corr
    key = keys[key_idx] + ('m' if is_minor else '')
    key_source = "analysis"
    camelot = KEY_TO_CAMELOT.get(key, '?')
    key_confidence = float(max(major_corr, minor_corr)) if not np.isnan(major_corr) else 0.5
    
    # Usar Key de ID3 solo si existe y es válido
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
            # ID3 key no reconocida, mantener análisis
            key_source = "analysis"
    
    # Energy - Calibrado para música electrónica
    rms = librosa.feature.rms(y=y)[0]
    energy_raw = float(np.mean(rms))
    
    # RMS típico en música electrónica: 0.05 (ambient) a 0.35 (hardstyle)
    # Mapear a escala 1-10 con mejor distribución
    # Usamos una curva que da más rango en el medio
    
    # Valores de referencia calibrados para electrónica:
    # 0.05 - 0.10: baja (1-3)
    # 0.10 - 0.18: media (4-5)
    # 0.18 - 0.25: alta (6-7)
    # 0.25+: muy alta (8-10)
    
    if energy_raw < 0.05:
        energy_dj = 1
    elif energy_raw < 0.08:
        energy_dj = 2
    elif energy_raw < 0.10:
        energy_dj = 3
    elif energy_raw < 0.14:
        energy_dj = 4
    elif energy_raw < 0.18:
        energy_dj = 5
    elif energy_raw < 0.22:
        energy_dj = 6
    elif energy_raw < 0.25:
        energy_dj = 7
    elif energy_raw < 0.30:
        energy_dj = 8
    elif energy_raw < 0.35:
        energy_dj = 9
    else:
        energy_dj = 10
    
    energy_normalized = energy_dj / 10.0
    print(f"  ⚡ Energía: raw={energy_raw:.4f} -> DJ level {energy_dj}")
    
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
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    percussion_density = min(float(np.mean(onset_env)) / 10, 1.0)
    
    # Classification
    track_type = classify_track_type(energy_normalized, segments, duration)
    genre = classify_genre_advanced(
        bpm, energy_normalized, has_heavy_bass,
        y, sr, percussion_density,
        spectral_centroid, rolloff
    )
    genre_source = "spectral_analysis"
    label = id3_data.get('label')
    year = id3_data.get('year')
    
    # Guardar género ID3 como fallback (suele ser genérico: "House", "Techno")
    id3_genre = id3_data.get('genre')
    
    # ==================== PRIORIDAD DE GÃ‰NEROS ====================
    # Discogs > MusicBrainz > ID3 > Análisis espectral
    # Discogs/MusicBrainz dan géneros específicos (ej: "Minimal Techno" vs "Techno")
    
    artist_name = id3_data.get('artist')
    title_name = id3_data.get('title')
    
    if GENRE_DETECTOR_ENABLED and genre_detector and artist_name and title_name:
        print(f"  ðŸ”Â Buscando género: {artist_name} - {title_name}")
        # 1. Intentar Discogs primero (mejor para electrónica)
        try:
            discogs_result = genre_detector.get_discogs_genre(artist_name, title_name)
            if discogs_result and discogs_result.get('genre'):
                genre = discogs_result.get('genre')
                genre_source = "discogs"
                # También obtener label y year si no los tenemos
                if not label and discogs_result.get('label'):
                    label = discogs_result['label']
                if not year and discogs_result.get('year'):
                    year = str(discogs_result['year'])
                print(f"  ðŸŽµ Discogs: {genre} | {label} ({year})")
            else:
                print(f"  âš ï¸ Discogs: No encontrado")
        except Exception as e:
            print(f"  âš ï¸ Error Discogs: {e}")
        
        # 2. Si no hay Discogs, intentar MusicBrainz
        if genre_source not in ["discogs"]:
            try:
                mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                if mb_result and mb_result.get('genre'):
                    genre = mb_result.get('genre')
                    genre_source = "musicbrainz"
                    print(f"  ðŸŽµ MusicBrainz: {genre}")
            except Exception as e:
                print(f"  âš ï¸ Error MusicBrainz: {e}")
    
    # 3. Si no hay Discogs ni MusicBrainz, usar ID3 (genérico pero mejor que nada)
    if genre_source == "spectral_analysis" and id3_genre:
        genre = id3_genre
        genre_source = "id3"
        print(f"  ðŸŽµ ID3 (fallback): {genre}")
    
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
        
        # Verificar que el artwork sea válido (mínimo 10KB para evitar corruptos/placeholders)
        if artwork_info and artwork_info.get('size', 0) > 10000:
            artwork_embedded = True
            artwork_source = "id3"
            # Guardar en caché
            save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            print(f"  🖼️ Artwork ID3: {artwork_info.get('size', 0)} bytes")
        else:
            # Fallback: buscar online (iTunes/Deezer)
            if artwork_info:
                print(f"  âš ï¸ Artwork ID3 muy pequeño ({artwork_info.get('size', 0)} bytes), buscando online...")
            else:
                print(f"  ðŸ”Â Sin artwork ID3, buscando online...")
            
            artist_name = id3_data.get('artist')
            title_name = id3_data.get('title')
            album_name = id3_data.get('album')
            
            try:
                from artwork_and_cuepoints import search_artwork_online
                online_artwork = search_artwork_online(artist_name, title_name, album_name)
                
                if online_artwork and online_artwork.get('data'):
                    artwork_embedded = False  # No está embebido, viene de online
                    artwork_source = online_artwork.get('source', 'online')
                    save_artwork_to_cache(fingerprint, online_artwork['data'], online_artwork['mime_type'])
                    artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                    print(f"  🖼️ Artwork {artwork_source}: {online_artwork.get('size', 0)} bytes")
                else:
                    print(f"  âŒ No se encontró artwork online")
            except Exception as e:
                print(f"  âš ï¸ Error buscando artwork online: {e}")
    
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
    Usado automáticamente para tracks > 4 minutos.
    """
    import gc
    
    # Crear analizador por chunks
    analyzer = get_chunked_analyzer(chunk_duration=60)
    
    # Ejecutar análisis chunked
    result = analyzer.full_analysis(file_path)
    
    # Limpiar memoria
    del analyzer
    gc.collect()
    
    # ==================== ID3 METADATA ====================
    id3_data = {}
    if ARTWORK_ENABLED:
        id3_data = extract_id3_metadata(file_path)
    
    # Sobrescribir con ID3 si existe y es válido
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
    
    # ==================== GÉNERO ====================
    genre = "Electronic"
    genre_source = "chunked_analysis"
    label = id3_data.get('label')
    year = id3_data.get('year')
    id3_genre = id3_data.get('genre')
    
    artist_name = id3_data.get('artist')
    title_name = id3_data.get('title')
    
    # Intentar obtener género de Discogs/MusicBrainz
    if GENRE_DETECTOR_ENABLED and genre_detector and artist_name and title_name:
        print(f"  🔍 Buscando género: {artist_name} - {title_name}")
        try:
            discogs_result = genre_detector.get_discogs_genre(artist_name, title_name)
            if discogs_result and discogs_result.get('genre'):
                genre = discogs_result.get('genre')
                genre_source = "discogs"
                if not label and discogs_result.get('label'):
                    label = discogs_result['label']
                if not year and discogs_result.get('year'):
                    year = str(discogs_result['year'])
                print(f"  🎵 Discogs: {genre}")
        except Exception as e:
            print(f"  ⚠️ Error Discogs: {e}")
        
        if genre_source not in ["discogs"]:
            try:
                mb_result = genre_detector.get_musicbrainz_info(artist_name, title_name)
                if mb_result and mb_result.get('genre'):
                    genre = mb_result.get('genre')
                    genre_source = "musicbrainz"
                    print(f"  🎵 MusicBrainz: {genre}")
            except Exception as e:
                print(f"  ⚠️ Error MusicBrainz: {e}")
    
    if genre_source == "chunked_analysis" and id3_genre:
        genre = id3_genre
        genre_source = "id3"
    
    # ==================== ARTWORK ====================
    artwork_embedded = False
    artwork_url = None
    
    if ARTWORK_ENABLED and fingerprint:
        artwork_info = extract_artwork_from_file(file_path)
        
        if artwork_info and artwork_info.get('size', 0) > 10000:
            artwork_embedded = True
            save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
            artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
            print(f"  🖼️ Artwork ID3: {artwork_info.get('size', 0)} bytes")
        else:
            if artist_name and title_name:
                try:
                    online_artwork = search_artwork_online(artist_name, title_name, id3_data.get('album'))
                    if online_artwork and online_artwork.get('data'):
                        save_artwork_to_cache(fingerprint, online_artwork['data'], online_artwork['mime_type'])
                        artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                        print(f"  🖼️ Artwork online: {online_artwork.get('size', 0)} bytes")
                except Exception as e:
                    print(f"  ⚠️ Error artwork online: {e}")
    
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
        track_type=result['track_type'],
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
    # 🆕 Rate limiting (opcional - descomenta si quieres)
    # check_rate_limit(get_client_ip(request))
    
    # 🆕 Validación mejorada de archivo
    if not file.filename:
        raise HTTPException(400, "No se proporcionó archivo")
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg')):
        raise HTTPException(400, "Formato no soportado. Permitidos: mp3, wav, flac, m4a, aac, ogg")
    
    # Leer contenido y validar tamaño
    content = await file.read()
    max_size = 100 * 1024 * 1024  # 100 MB
    if len(content) > max_size:
        raise HTTPException(400, f"Archivo demasiado grande. Máximo: 100 MB")
    if len(content) < 1000:
        raise HTTPException(400, "Archivo demasiado pequeño o corrupto")
    
    # Verificar si ya existe en BD (solo si no es force)
    if not force:
        existing = db.get_track_by_filename(file.filename)
        if existing:
            analysis_json = json.loads(existing[11]) if len(existing) > 11 else {}
            return AnalysisResult(**analysis_json)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        fingerprint = calculate_fingerprint(tmp_path)
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
        
        # Prioridad: Memoria colectiva > Discogs > MusicBrainz > ID3 > AcousticBrainz > Análisis
        if collective_genre:
            result.genre = collective_genre
            result.genre_source = "collective_memory"
        elif result.genre_source in ["discogs", "musicbrainz"]:
            # Ya tenemos buen género, mantenerlo
            pass
        elif result.genre_source == "id3":
            # ID3 está bien, pero si hay AcousticBrainz específico, usarlo
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
        
        return result
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR en análisis de audio:\n{error_detail}")
        
        # ==================== FALLBACK: Track corrupto ====================
        # Intentar crear resultado básico con ID3 y/o filename
        print(f"âš ï¸ Intentando fallback para: {file.filename}")
        
        try:
            fingerprint = hashlib.md5(file.filename.encode()).hexdigest()
            
            # Intentar leer metadatos ID3 aunque el audio esté corrupto
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
            
            # Crear resultado mínimo marcado como "pendiente"
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
            
            # Guardar en BD (marcado como pendiente de análisis real)
            track_data = result.dict()
            track_data['id'] = fingerprint
            track_data['filename'] = file.filename
            track_data['fingerprint'] = fingerprint
            track_data['analysis_status'] = 'failed'  # Marcador especial
            db.save_track(track_data)
            
            print(f"âœ… Fallback creado: {artist} - {title} (análisis pendiente)")
            
            return result
            
        except Exception as fallback_error:
            print(f"âŒ Fallback también falló: {fallback_error}")
            raise HTTPException(500, f"Error analizando: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/correction")
async def save_correction(request: CorrectionRequest):
    # 🆕 Validar campos
    track_id = validate_track_id(request.track_id)
    field = sanitize_string(request.field, max_length=50, allow_empty=False, field_name="field")
    old_value = sanitize_string(request.old_value, max_length=200, field_name="old_value")
    new_value = sanitize_string(request.new_value, max_length=200, allow_empty=False, field_name="new_value")
    
    # Validar que el campo sea uno permitido
    allowed_fields = {'genre', 'bpm', 'key', 'camelot', 'energy', 'artist', 'title', 'label', 'track_type'}
    if field not in allowed_fields:
        raise HTTPException(400, f"Campo no permitido: {field}. Permitidos: {', '.join(allowed_fields)}")
    
    db.save_correction(track_id, field, old_value, new_value, request.fingerprint)
    return {"status": "ok", "message": "Corrección guardada"}

# ==================== IDENTIFICAR TRACK CON AUDD ====================

@app.post("/identify")
async def identify_track(file: UploadFile = File(...)):
    """
    Identifica un track usando AudD y hace RE-ANÃLISIS COMPLETO.
    
    Flujo:
    1. AudD identifica artista/título
    2. Busca género en Discogs con el nuevo nombre
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
        
        print(f"ðŸ”Â Identificando track: {file.filename}")
        fingerprint = hashlib.md5(file.filename.encode()).hexdigest()
        
        # ==================== PASO 1: IDENTIFICAR CON AUDD ====================
        audio_to_send = tmp_path
        
        try:
            y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=20, offset=30)
            import soundfile as sf
            fragment_path = tmp_path + "_fragment.wav"
            sf.write(fragment_path, y, sr)
            audio_to_send = fragment_path
            print(f"  ðŸ”Ž Fragmento extraído: 20 seg desde 0:30")
        except Exception as e:
            print(f"  âš ï¸ No se pudo extraer fragmento: {e}")
        
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
        
        print(f"âœ… AudD identificó: {artist} - {title}")
        
        # ==================== PASO 2: BUSCAR GÃ‰NERO EN DISCOGS ====================
        genre = "Electronic"
        genre_source = "default"
        
        if GENRE_DETECTOR_ENABLED and genre_detector and artist and title:
            print(f"  ðŸ”Â Buscando género: {artist} - {title}")
            try:
                discogs_result = genre_detector.get_discogs_genre(artist, title)
                if discogs_result and discogs_result.get('genre'):
                    genre = discogs_result['genre']
                    genre_source = "discogs"
                    if not label and discogs_result.get('label'):
                        label = discogs_result['label']
                    if not year and discogs_result.get('year'):
                        year = str(discogs_result['year'])
                    print(f"  ðŸŽµ Discogs: {genre} | {label} ({year})")
            except Exception as e:
                print(f"  âš ï¸ Error Discogs: {e}")
            
            if genre_source != "discogs":
                try:
                    mb_result = genre_detector.get_musicbrainz_info(artist, title)
                    if mb_result and mb_result.get('genre'):
                        genre = mb_result['genre']
                        genre_source = "musicbrainz"
                        print(f"  ðŸŽµ MusicBrainz: {genre}")
                except Exception as e:
                    print(f"  âš ï¸ Error MusicBrainz: {e}")
        
        # ==================== PASO 3: RE-ANALIZAR AUDIO ====================
        bpm = None
        bpm_confidence = 0.0
        key = None
        camelot = None
        energy_dj = 5
        duration = 0.0
        bpm_source = 'pending'
        key_source = 'pending'
        
        print(f"  ðŸ”Â¬ Re-analizando audio...")
        try:
            y_full, sr_full = librosa.load(tmp_path, sr=22050, mono=True)
            duration = librosa.get_duration(y=y_full, sr=sr_full)
            
            # BPM
            tempo, beat_frames = librosa.beat.beat_track(y=y_full, sr=sr_full)
            if hasattr(tempo, '__iter__'):
                tempo = float(tempo[0])
            bpm = round(tempo, 1)
            bpm_confidence = 0.7
            bpm_source = 'analysis'
            print(f"    âœ“ BPM: {bpm}")
            
            # Key
            chroma = librosa.feature.chroma_cqt(y=y_full, sr=sr_full)
            key_profiles = np.mean(chroma, axis=1)
            key_idx = int(np.argmax(key_profiles))
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = f"{key_names[key_idx]}m"  # Asumimos menor por defecto
            camelot = get_camelot(key)
            key_source = 'analysis'
            print(f"    âœ“ Key: {key} ({camelot})")
            
            # Energy
            rms = librosa.feature.rms(y=y_full)[0]
            avg_rms = float(np.mean(rms))
            if avg_rms < 0.05:
                energy_dj = 2
            elif avg_rms < 0.10:
                energy_dj = 4
            elif avg_rms < 0.18:
                energy_dj = 6
            elif avg_rms < 0.25:
                energy_dj = 8
            else:
                energy_dj = 9
            print(f"    âœ“ Energy: {energy_dj}")
            
        except Exception as e:
            print(f"  âš ï¸ Re-análisis falló: {e}")
            # 🆕 FALLBACK 1: Buscar en BD colectiva
            if artist and title:
                print(f"  ðŸ”Â Buscando en BD colectiva...")
                collective_data = search_collective_db(artist, title)
                if collective_data:
                    if collective_data.get('bpm'):
                        bpm = collective_data['bpm']
                        bpm_confidence = 0.9
                        bpm_source = 'collective'
                        print(f"    âœ“ BD Colectiva BPM: {bpm}")
                    if collective_data.get('key'):
                        key = collective_data['key']
                        camelot = collective_data.get('camelot') or get_camelot(key)
                        key_source = 'collective'
                        print(f"    âœ“ BD Colectiva Key: {key} ({camelot})")
                    if collective_data.get('duration') and collective_data['duration'] > 0:
                        duration = collective_data['duration']
                else:
                    print(f"    âœ— No encontrado en BD colectiva")
                    
                    # 🆕 FALLBACK 2: Buscar en Beatport
                    print(f"  ðŸ”Â Buscando en Beatport: {artist} - {title}")
                    beatport_data = search_beatport(artist, title)
                    if beatport_data:
                        if beatport_data.get('bpm'):
                            bpm = beatport_data['bpm']
                            bpm_confidence = 0.95
                            bpm_source = 'beatport'
                            print(f"    âœ“ Beatport BPM: {bpm}")
                        if beatport_data.get('key'):
                            key = beatport_data['key']
                            camelot = get_camelot(key)
                            key_source = 'beatport'
                            print(f"    âœ“ Beatport Key: {key} ({camelot})")
                        if beatport_data.get('duration'):
                            duration = beatport_data['duration']
                    else:
                        print(f"    âœ— No encontrado en Beatport")
        
        # ==================== PASO 4: BUSCAR ARTWORK ====================
        artwork_url = None
        artwork_source = None
        
        if artist and title and search_artwork_online:
            print(f"  🖼️ Buscando artwork...")
            artwork_info = search_artwork_online(artist, title)
            if artwork_info:
                save_artwork_to_cache(fingerprint, artwork_info['data'], artwork_info['mime_type'])
                artwork_url = f"{BASE_URL}/artwork/{fingerprint}"
                artwork_source = artwork_info.get('source', 'online')
                print(f"    âœ“ Artwork: {artwork_source} ({artwork_info['size']} bytes)")
            else:
                print(f"    âœ— No se encontró artwork")
        elif not search_artwork_online:
            print(f"  âš ï¸ search_artwork_online no disponible")
        
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
            'key_source': key_source,
            'energy_dj': energy_dj,
            'duration': duration,
            'genre': genre,
            'genre_source': genre_source,
            'label': label,
            'year': year,
            'artwork_url': artwork_url,
            'track_type': 'Main Floor',  # Default
        }
        
        db.save_track(track_db_data)
        print(f"  ðŸ’¾ Guardado en BD")
        
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
        print(f"âŒ Error identificando: {traceback.format_exc()}")
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
    Reconoce una canción a partir de audio grabado usando AudD API.
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
        
        print(f"ðŸŽ¤ Reconociendo audio: {file.filename} ({len(content)} bytes)")
        
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
            print(f"âŒ AudD error: {audd_response.status_code}")
            raise HTTPException(500, f"Error AudD API: {audd_response.status_code}")
        
        result = audd_response.json()
        
        if result.get('status') != 'success':
            error_msg = result.get('error', {}).get('error_message', 'Unknown error')
            print(f"âŒ AudD error: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        track_data = result.get('result')
        
        if not track_data:
            print("ðŸ”â€¡ No se reconoció ninguna canción")
            return {"status": "not_found", "message": "No se pudo identificar la canción"}
        
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
        
        print(f"âœ… Reconocido: {artist} - {title}")
        
        # Buscar si ya tenemos análisis de este track en la BD
        backend_analysis = None
        existing_tracks = db.search_by_artist(artist, limit=50)
        for track in existing_tracks:
            if track.get('title', '').lower() == title.lower():
                backend_analysis = track
                print(f"  ðŸ“Š Encontrado en biblioteca: {track.get('id')}")
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
        
        # Si tenemos análisis previo, incluirlo
        if backend_analysis:
            response["backend_analysis"] = backend_analysis
        
        return response
        
    except requests.Timeout:
        print("âŒ AudD timeout")
        raise HTTPException(504, "Timeout conectando con AudD")
    except Exception as e:
        import traceback
        print(f"âŒ Error reconocimiento: {traceback.format_exc()}")
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/check-analyzed")
async def check_analyzed(filenames: list[str]):
    """Verificar cuáles tracks ya están analizados"""
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
    """Obtener análisis guardado de un track por filename"""
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
            analysis_json = existing[11]  # índice de analysis_json
            if analysis_json:
                return json.loads(analysis_json)
        except:
            pass
        
        # Fallback: construir respuesta básica
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
    
    raise HTTPException(404, f"Análisis no encontrado para: {filename}")

@app.get("/artwork/{track_id}")
async def get_artwork(track_id: str):
    """Devuelve el artwork de un track como imagen"""
    for ext in ['jpg', 'png', 'jpeg']:
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            media_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else "image/png"
            return FileResponse(cache_path, media_type=media_type)
    
    raise HTTPException(404, "Artwork no encontrado")

# ==================== ENDPOINTS DE BÃšSQUEDA ====================

@app.get("/search/artist/{artist}")
async def search_by_artist(artist: str, limit: int = Query(50, ge=1, le=200)):
    """Buscar tracks por artista"""
    results = db.search_by_artist(artist, limit)
    return {"query": artist, "count": len(results), "tracks": results}

@app.get("/search/genre/{genre}")
async def search_by_genre(genre: str, limit: int = Query(100, ge=1, le=500)):
    # 🆕 Sanitizar género
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
    # 🆕 Validar rangos
    min_bpm, max_bpm = validate_bpm_range(min_bpm, max_bpm)
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_bpm(min_bpm, max_bpm, limit)}

@app.get("/search/energy")
async def search_by_energy(
    request: Request,
    min_energy: Optional[int] = None,
    max_energy: Optional[int] = None,
    limit: int = Query(100, ge=1, le=500)
):
    # 🆕 Validar rangos
    min_energy, max_energy = validate_energy_range(min_energy, max_energy)
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_energy(min_energy, max_energy, limit)}

@app.get("/search/key/{key}")
async def search_by_key(key: str, limit: int = Query(100, ge=1, le=500)):
    # 🆕 Validar tonalidad
    try:
        key = validate_key(key)
    except ValidationError:
        # Si no es válido como key, intentar como está
        key = sanitize_string(key, max_length=10)
    
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_key(key, limit)}

@app.get("/search/compatible/{camelot}")
async def search_compatible_keys(camelot: str, limit: int = Query(50, ge=1, le=200)):
    # 🆕 Validar Camelot
    camelot = validate_camelot(camelot)
    limit = validate_limit(limit, max_limit=200)
    
    # Obtener keys compatibles
    compatible = CAMELOT_COMPATIBLE.get(camelot, [camelot])
    
    return {
        "camelot": camelot,
        "compatible_keys": compatible,
        "tracks": db.search_by_compatible_keys(compatible, limit)
    }
@app.get("/search-analyzed")
async def search_analyzed_track(
    artist: str = Query(..., description="Nombre del artista"),
    title: str = Query(..., description="Título del track")
):
    """
    Busca si un track ya fue analizado por algún usuario.
    Devuelve TODA la información del análisis si existe.
    
    Returns:
        - found: bool - Si se encontró el track
        - track: dict - Toda la información del análisis (si existe)
        - in_collective: bool - Si está en la memoria colectiva
    """
    import re
    
    # Validar y sanitizar entrada
    artist_clean = sanitize_string(artist, max_length=200, allow_empty=False, field_name="artist")
    title_clean = sanitize_string(title, max_length=200, allow_empty=False, field_name="title")
    
    # Normalizar para búsqueda
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
        
        # Búsqueda exacta primero
        cursor.execute("""
            SELECT * FROM tracks 
            WHERE LOWER(artist) = ? AND LOWER(title) LIKE ?
            AND bpm IS NOT NULL AND bpm > 0
            ORDER BY analyzed_at DESC
            LIMIT 1
        """, (artist_normalized, f"%{title_normalized}%"))
        
        row = cursor.fetchone()
        
        if not row:
            # Búsqueda más flexible
            cursor.execute("""
                SELECT * FROM tracks 
                WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
                AND bpm IS NOT NULL AND bpm > 0
                ORDER BY analyzed_at DESC
                LIMIT 1
            """, (f"%{artist_normalized}%", f"%{title_normalized}%"))
            row = cursor.fetchone()
        
        if row:
            # Convertir a dict usando el método existente
            track_dict = db._row_to_dict(row)
            
            # Si hay analysis_json, parsear para obtener todos los campos
            if track_dict and track_dict.get('analysis_json'):
                try:
                    full_analysis = json.loads(track_dict['analysis_json'])
                    # Combinar con los campos básicos
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
    # 🆕 Validar tipo de track
    track_type = validate_track_type(track_type)
    limit = validate_limit(limit, max_limit=500)
    
    return {"tracks": db.search_by_track_type(track_type, limit)}

@app.post("/search/advanced")
async def search_advanced(search_request: SearchRequest):
    # 🆕 Validar y sanitizar todos los campos
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
    # 🆕 Validar límite
    limit = validate_limit(limit, max_limit=5000)
    
    return {"tracks": db.get_all_tracks(limit)}

@app.get("/library/artists")
async def get_unique_artists():
    """Obtener lista de artistas únicos"""
    artists = db.get_unique_artists()
    return {"count": len(artists), "artists": artists}

@app.get("/library/genres")
async def get_unique_genres():
    """Obtener lista de géneros únicos"""
    genres = db.get_unique_genres()
    return {"count": len(genres), "genres": genres}

@app.get("/library/stats")
async def get_library_stats():
    """Obtener estadísticas de la biblioteca"""
    return db.get_stats()

@app.get("/track/{track_id}")
async def get_track(track_id: str):
    """Obtener información de un track específico"""
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
        """Buscar tracks similares a uno específico"""
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
        ]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.3.0"}

# ==================== ADMIN / RESET ====================

@app.delete("/admin/reset-database")
async def reset_database(confirm: str = Query(..., description="Escribe 'CONFIRMAR' para borrar")):
    """
    âš ï¸ PELIGROSO: Borra TODA la base de datos.
    Requiere confirmar escribiendo 'CONFIRMAR' como parámetro.
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
    """Limpia solo el caché de artwork"""
    import shutil
    try:
        if os.path.exists(ARTWORK_CACHE_DIR):
            shutil.rmtree(ARTWORK_CACHE_DIR)
            os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)
        return {"status": "ok", "message": "Caché de artwork limpiado"}
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
