"""
DJ ANALYZER - Artwork & Cue Points Module v2.3.0
====================================================
Modulo para extraccion de artwork y deteccion de cue points.

Exporta:
- extract_artwork_from_file()
- extract_id3_metadata()
- detect_cue_points()
- detect_beat_grid()
- save_artwork_to_cache()
- search_artwork_online()
- ARTWORK_CACHE_DIR
"""
from config import LASTFM_API_KEY
import logging
import os
import base64
import requests
from typing import Optional, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Constante para el directorio de cache
ARTWORK_CACHE_DIR = "/data/artwork_cache"
os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)

# ==================== 0. BUSQUEDA DE ARTWORK ONLINE ====================

def search_artwork_online(artist: str, title: str, album: str = None) -> Optional[Dict]:
    """
    Busca artwork en servicios online cuando no hay ID3 o es invalido.
    Intenta: iTunes -> Deezer -> Last.fm
    
    Returns:
        Dict con 'url', 'data' (bytes), 'mime_type', 'size', 'source' o None
    """
    if not artist or not title:
        logger.warning(f"No se puede buscar artwork: artist={artist}, title={title}")
        return None
    
    # Limpiar query
    query = f"{artist} {title}".replace("(", "").replace(")", "").replace("-", " ")
    logger.info(f"Buscando artwork online: {artist} - {title}")
    
    # 1. Intentar iTunes (no requiere API key)
    logger.debug("Intentando iTunes...")
    artwork = _search_itunes(query)
    if artwork:
        logger.debug("iTunes encontro artwork")
        return artwork
    logger.debug("iTunes no encontro")
    
    # 2. Intentar Deezer (no requiere API key)
    logger.debug("Intentando Deezer...")
    artwork = _search_deezer(artist, title)
    if artwork:
        logger.debug("Deezer encontro artwork")
        return artwork
    logger.debug("Deezer no encontro")
    
    # 3. Intentar Last.fm
    logger.debug("Intentando Last.fm...")
    artwork = _search_lastfm(artist, title)
    if artwork:
        logger.debug("Last.fm encontro artwork")
        return artwork
    logger.debug("Last.fm no encontro")
    
    return None

def _search_itunes(query: str) -> Optional[Dict]:
    """Buscar artwork en iTunes API"""
    try:
        from urllib.parse import quote
        url = f"https://itunes.apple.com/search?term={quote(query)}&media=music&limit=5"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('resultCount', 0) > 0:
                # Buscar en los primeros resultados
                for result in data['results'][:5]:
                    # Obtener imagen grande (600x600)
                    artwork_url = result.get('artworkUrl100', '').replace('100x100', '600x600')
                    
                    if artwork_url:
                        # Descargar imagen
                        img_response = requests.get(artwork_url, timeout=5)
                        if img_response.status_code == 200 and len(img_response.content) > 10000:
                            return {
                                'url': artwork_url,
                                'data': img_response.content,
                                'mime_type': 'image/jpeg',
                                'size': len(img_response.content),
                                'source': 'itunes'
                            }
    except requests.RequestException as e:
        logger.error(f"Error iTunes: {e}")
    
    return None

def _search_deezer(artist: str, title: str) -> Optional[Dict]:
    """Buscar artwork en Deezer API"""
    try:
        from urllib.parse import quote
        # Intentar busqueda exacta primero
        url = f"https://api.deezer.com/search?q=artist:\"{quote(artist)}\" track:\"{quote(title)}\"&limit=5"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                for track in data['data'][:5]:
                    album = track.get('album', {})
                    # cover_xl es 1000x1000
                    artwork_url = album.get('cover_xl') or album.get('cover_big') or album.get('cover_medium')
                    
                    if artwork_url:
                        img_response = requests.get(artwork_url, timeout=5)
                        if img_response.status_code == 200 and len(img_response.content) > 10000:
                            return {
                                'url': artwork_url,
                                'data': img_response.content,
                                'mime_type': 'image/jpeg',
                                'size': len(img_response.content),
                                'source': 'deezer'
                            }
        
        # Si no encuentra con busqueda exacta, intentar busqueda general
        url = f"https://api.deezer.com/search?q={quote(artist + ' ' + title)}&limit=5"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                for track in data['data'][:5]:
                    album = track.get('album', {})
                    artwork_url = album.get('cover_xl') or album.get('cover_big')
                    if artwork_url:
                        img_response = requests.get(artwork_url, timeout=5)
                        if img_response.status_code == 200 and len(img_response.content) > 10000:
                            return {
                                'url': artwork_url,
                                'data': img_response.content,
                                'mime_type': 'image/jpeg',
                                'size': len(img_response.content),
                                'source': 'deezer'
                            }
    except requests.RequestException as e:
        logger.error(f"Error Deezer: {e}")
    
    return None

def _search_spotify(artist: str, title: str) -> Optional[Dict]:
    """Spotify requiere OAuth - no disponible sin autenticacion"""
    # La API de Spotify requiere token de acceso
    # No es posible buscar sin autenticacion
    return None

def _search_lastfm(artist: str, title: str) -> Optional[Dict]:
    """Buscar artwork en Last.fm"""
    try:
        from urllib.parse import quote
        
        # Intentar primero buscar el track
        url = f"https://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={LASTFM_API_KEY}&artist={quote(artist)}&track={quote(title)}&format=json"
        
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            data = response.json()
            
            # Verificar si hay error
            if 'error' in data:
                logger.debug(f"Last.fm: {data.get('message', 'Track no encontrado')}")
            else:
                track = data.get('track', {})
                album = track.get('album', {})
                images = album.get('image', [])
                
                # Buscar imagen mas grande (ultima en la lista)
                for img in reversed(images):
                    img_url = img.get('#text', '')
                    if img_url and len(img_url) > 10:
                        # Last.fm a veces devuelve URLs vacias o placeholder
                        # Intentar obtener version grande
                        if '/i/u/' in img_url:
                            # Formato: https://lastfm.freetls.fastly.net/i/u/300x300/xxx.jpg
                            img_url = img_url.replace('/64s/', '/300x300/').replace('/174s/', '/300x300/')
                        
                        try:
                            img_response = requests.get(img_url, timeout=5)
                            if img_response.status_code == 200 and len(img_response.content) > 5000:
                                logger.debug(f"Artwork Last.fm: {len(img_response.content)} bytes")
                                return {
                                    'url': img_url,
                                    'data': img_response.content,
                                    'mime_type': 'image/jpeg',
                                    'size': len(img_response.content),
                                    'source': 'lastfm'
                                }
                        except:
                            continue
        
        # Si no encontro por track, intentar por album/artista
        url = f"https://ws.audioscrobbler.com/2.0/?method=artist.getTopAlbums&api_key=57ee3318536b23ee81d6b27e36997cde&artist={quote(artist)}&limit=1&format=json"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            albums = data.get('topalbums', {}).get('album', [])
            if albums and len(albums) > 0:
                images = albums[0].get('image', [])
                for img in reversed(images):
                    img_url = img.get('#text', '')
                    if img_url and len(img_url) > 10:
                        try:
                            img_response = requests.get(img_url, timeout=5)
                            if img_response.status_code == 200 and len(img_response.content) > 5000:
                                logger.debug(f"Artwork Last.fm (album): {len(img_response.content)} bytes")
                                return {
                                    'url': img_url,
                                    'data': img_response.content,
                                    'mime_type': 'image/jpeg',
                                    'size': len(img_response.content),
                                    'source': 'lastfm'
                                }
                        except:
                            continue
                            
    except requests.RequestException as e:
        logger.error(f"Error Last.fm: {e}")
    
    return None

# ==================== 1. EXTRACCIN DE ARTWORK ====================

def extract_artwork_from_file(file_path: str) -> Optional[Dict]:
    """
    Extrae artwork embebido de archivos de audio (MP3, FLAC, M4A)
    
    Returns:
        Dict con 'data' (base64), 'mime_type', 'size' o None si no hay artwork
    """
    try:
        from mutagen import File as MutagenFile
        from mutagen.mp3 import MP3
        from mutagen.id3 import ID3, APIC
        from mutagen.flac import FLAC
        from mutagen.mp4 import MP4
        
        audio = MutagenFile(file_path)
        
        if audio is None:
            return None
        
        artwork_data = None
        mime_type = 'image/jpeg'
        
        # MP3 con ID3 tags
        if isinstance(audio, MP3) or file_path.lower().endswith('.mp3'):
            try:
                tags = ID3(file_path)
                for key in tags.keys():
                    if key.startswith('APIC'):
                        apic = tags[key]
                        artwork_data = apic.data
                        mime_type = apic.mime or 'image/jpeg'
                        break
            except:
                pass
        
        # FLAC
        elif isinstance(audio, FLAC):
            if audio.pictures:
                pic = audio.pictures[0]
                artwork_data = pic.data
                mime_type = pic.mime or 'image/jpeg'
        
        # M4A/MP4/AAC
        elif isinstance(audio, MP4) or file_path.lower().endswith(('.m4a', '.mp4', '.aac')):
            if 'covr' in audio.tags:
                covers = audio.tags['covr']
                if covers:
                    artwork_data = bytes(covers[0])
                    # MP4 cover format: 13=JPEG, 14=PNG
                    if hasattr(covers[0], 'imageformat'):
                        mime_type = 'image/png' if covers[0].imageformat == 14 else 'image/jpeg'
        
        # Fallback generico para otros formatos
        else:
            if hasattr(audio, 'pictures') and audio.pictures:
                pic = audio.pictures[0]
                artwork_data = pic.data
                mime_type = getattr(pic, 'mime', 'image/jpeg')
        
        if artwork_data:
            return {
                'data': artwork_data,  # Devuelve bytes directamente
                'mime_type': mime_type,
                'size': len(artwork_data),
            }
        
        return None
        
    except ImportError:
        logger.warning("mutagen no instalado. Ejecuta: pip install mutagen")
        return None
    except (FileNotFoundError, IOError, OSError) as e:
        logger.error(f"Error extrayendo artwork: {e}")
        return None


def save_artwork_to_cache(fingerprint: str, artwork_data: bytes, mime_type: str) -> str:
    """
    Guarda artwork en cache local y devuelve el nombre del archivo
    """
    ext = 'jpg' if 'jpeg' in mime_type else 'png' if 'png' in mime_type else 'jpg'
    filename = f"{fingerprint}.{ext}"
    cache_path = os.path.join(ARTWORK_CACHE_DIR, filename)
    
    with open(cache_path, 'wb') as f:
        f.write(artwork_data)
    
    return filename


# ==================== EXTRACCIN DE ID3 METADATA ====================

def extract_id3_metadata(file_path: str) -> Dict:
    """
    Extrae metadata completa de ID3 tags (MP3, FLAC, M4A)
    
    Returns:
        Dict con title, artist, album, label, year, genre, bpm, key, isrc
    """
    metadata = {
        'title': None,
        'artist': None,
        'album': None,
        'label': None,
        'year': None,
        'genre': None,
        'bpm': None,
        'key': None,
        'isrc': None,
    }
    
    try:
        from mutagen import File as MutagenFile
        from mutagen.mp3 import MP3
        from mutagen.id3 import ID3
        from mutagen.flac import FLAC
        from mutagen.mp4 import MP4
        
        if file_path.lower().endswith('.mp3'):
            try:
                tags = ID3(file_path)
                
                if 'TIT2' in tags:
                    metadata['title'] = str(tags['TIT2'])
                if 'TPE1' in tags:
                    metadata['artist'] = str(tags['TPE1'])
                if 'TALB' in tags:
                    metadata['album'] = str(tags['TALB'])
                if 'TPUB' in tags:
                    metadata['label'] = str(tags['TPUB'])
                if 'TDRC' in tags:
                    metadata['year'] = str(tags['TDRC'])[:4]
                elif 'TYER' in tags:
                    metadata['year'] = str(tags['TYER'])[:4]
                if 'TCON' in tags:
                    genre = str(tags['TCON'])
                    if not genre.isdigit():
                        metadata['genre'] = genre
                if 'TBPM' in tags:
                    try:
                        metadata['bpm'] = float(str(tags['TBPM']))
                    except:
                        pass
                if 'TKEY' in tags:
                    metadata['key'] = str(tags['TKEY'])
                if 'TSRC' in tags:
                    metadata['isrc'] = str(tags['TSRC'])
                    
            except (FileNotFoundError, IOError, OSError, ValueError, KeyError) as e:
                logger.error(f"Error leyendo ID3: {e}")
        
        elif file_path.lower().endswith('.flac'):
            try:
                audio = FLAC(file_path)
                if 'title' in audio:
                    metadata['title'] = audio['title'][0]
                if 'artist' in audio:
                    metadata['artist'] = audio['artist'][0]
                if 'album' in audio:
                    metadata['album'] = audio['album'][0]
                if 'label' in audio or 'organization' in audio:
                    metadata['label'] = audio.get('label', audio.get('organization', [None]))[0]
                if 'date' in audio:
                    metadata['year'] = audio['date'][0][:4]
                if 'genre' in audio:
                    metadata['genre'] = audio['genre'][0]
                if 'bpm' in audio:
                    metadata['bpm'] = float(audio['bpm'][0])
            except (FileNotFoundError, IOError, OSError, ValueError, KeyError) as e:
                logger.error(f"Error leyendo FLAC: {e}")
        
        elif file_path.lower().endswith(('.m4a', '.mp4')):
            try:
                audio = MP4(file_path)
                if '\xa9nam' in audio:
                    metadata['title'] = audio['\xa9nam'][0]
                if '\xa9ART' in audio:
                    metadata['artist'] = audio['\xa9ART'][0]
                if '\xa9alb' in audio:
                    metadata['album'] = audio['\xa9alb'][0]
                if '\xa9day' in audio:
                    metadata['year'] = str(audio['\xa9day'][0])[:4]
                if '\xa9gen' in audio:
                    metadata['genre'] = audio['\xa9gen'][0]
                if 'tmpo' in audio:
                    metadata['bpm'] = float(audio['tmpo'][0])
            except (FileNotFoundError, IOError, OSError, ValueError, KeyError) as e:
                logger.error(f"Error leyendo M4A: {e}")
    
    except ImportError:
        logger.warning("mutagen no instalado. Ejecuta: pip install mutagen")

    return metadata


# ==================== 2. CUE POINTS DETECTION ====================

class CuePoint:
    """Representa un cue point profesional"""
    def __init__(self, timestamp: float, cue_type: str, name: str, 
                 energy: float = 0.0, confidence: float = 0.0):
        self.timestamp = timestamp
        self.type = cue_type  # intro, buildup, drop, breakdown, outro, mix_in, mix_out
        self.name = name
        self.energy = energy
        self.confidence = confidence
    
    def to_dict(self):
        return {
            'timestamp': round(self.timestamp, 2),
            'type': self.type,
            'name': self.name,
            'energy': round(self.energy, 3),
            'confidence': round(self.confidence, 2),
        }


def detect_cue_points(y, sr, duration: float, segments: dict) -> List[Dict]:
    """
    Detecta cue points profesionales para DJs
    
    Tipos de cue points:
    - MIX_IN: Punto optimo para empezar a mezclar (inicio del track)
    - INTRO_END: Fin de la intro, empieza el cuerpo
    - BUILDUP: Inicio del buildup antes del drop
    - DROP: El drop principal
    - BREAKDOWN: Inicio del breakdown
    - BREAKDOWN_END: Fin del breakdown (antes del siguiente buildup)
    - MIX_OUT: Punto optimo para empezar a sacar el track
    - OUTRO_START: Inicio del outro
    
    Returns:
        Lista de cue points ordenados por timestamp
    """
    import librosa
    import numpy as np
    
    cue_points = []
    sections = segments.get('sections', [])
    
    if not sections:
        return []
    
    # Calcular energia por seccion mas detallada (segmentos de 4 segundos)
    segment_length = int(sr * 4)
    n_segments = len(y) // segment_length
    
    detailed_energies = []
    for i in range(n_segments):
        segment = y[i*segment_length:(i+1)*segment_length]
        rms = np.mean(librosa.feature.rms(y=segment))
        spectral_flux = np.mean(np.abs(np.diff(librosa.feature.spectral_centroid(y=segment))))
        detailed_energies.append({
            'time': i * 4.0,
            'rms': float(rms),
            'flux': float(spectral_flux),
        })
    
    if not detailed_energies:
        return []
    
    avg_rms = np.mean([e['rms'] for e in detailed_energies])
    max_rms = np.max([e['rms'] for e in detailed_energies])
    
    # ==================== MIX IN POINT ====================
    # Buscar el primer momento con energia suficiente para mezclar
    # Tipicamente despues de 4-8 segundos de intro
    mix_in_time = 0.0
    for e in detailed_energies[:10]:  # Primeros 40 segundos
        if e['rms'] > avg_rms * 0.4:
            mix_in_time = e['time']
            break
    
    # Si no hay nada con energia, usar el inicio
    if mix_in_time == 0.0 and detailed_energies:
        mix_in_time = 4.0  # Default a 4 segundos
    
    cue_points.append(CuePoint(
        timestamp=mix_in_time,
        cue_type='mix_in',
        name='Mix In',
        energy=detailed_energies[int(mix_in_time/4)]['rms'] if int(mix_in_time/4) < len(detailed_energies) else 0,
        confidence=0.8
    ))
    
    # ==================== INTRO END ====================
    # Buscar donde la energia sube significativamente
    intro_end = None
    for i, e in enumerate(detailed_energies[2:15], start=2):  # Entre 8 y 60 segundos
        if e['rms'] > avg_rms * 0.7:
            intro_end = e['time']
            cue_points.append(CuePoint(
                timestamp=intro_end,
                cue_type='intro_end',
                name='Intro End',
                energy=e['rms'],
                confidence=0.7
            ))
            break
    
    # ==================== DROPS ====================
    # Detectar todos los drops (picos de energia significativos)
    drop_indices = []
    drop_threshold = avg_rms * 1.3
    
    for i, e in enumerate(detailed_energies):
        if e['rms'] > drop_threshold:
            # Verificar que sea un pico real (no continuacion de otro drop)
            if not drop_indices or (e['time'] - detailed_energies[drop_indices[-1]]['time']) > 30:
                drop_indices.append(i)
    
    # Anadir cue points para cada drop detectado
    for idx, drop_idx in enumerate(drop_indices):
        drop_time = detailed_energies[drop_idx]['time']
        
        # Buscar el buildup antes del drop (caida de energia seguida de subida)
        buildup_time = drop_time - 16  # Default: 16 segundos antes
        for i in range(max(0, drop_idx - 8), drop_idx):
            if detailed_energies[i]['rms'] < avg_rms * 0.6:
                buildup_time = detailed_energies[i]['time']
                break
        
        if buildup_time > 0:
            cue_points.append(CuePoint(
                timestamp=buildup_time,
                cue_type='buildup',
                name=f'Buildup {idx+1}' if len(drop_indices) > 1 else 'Buildup',
                energy=detailed_energies[int(buildup_time/4)]['rms'] if int(buildup_time/4) < len(detailed_energies) else 0,
                confidence=0.75
            ))
        
        cue_points.append(CuePoint(
            timestamp=drop_time,
            cue_type='drop',
            name=f'Drop {idx+1}' if len(drop_indices) > 1 else 'Drop',
            energy=detailed_energies[drop_idx]['rms'],
            confidence=0.9
        ))
    
    # ==================== BREAKDOWNS ====================
    # Buscar valles de energia significativos (no al principio ni al final)
    breakdown_threshold = avg_rms * 0.5
    in_breakdown = False
    breakdown_start = None
    
    for i, e in enumerate(detailed_energies[5:-5], start=5):  # Evitar intro/outro
        if e['rms'] < breakdown_threshold and not in_breakdown:
            in_breakdown = True
            breakdown_start = e['time']
        elif e['rms'] > avg_rms * 0.7 and in_breakdown:
            in_breakdown = False
            if breakdown_start:
                cue_points.append(CuePoint(
                    timestamp=breakdown_start,
                    cue_type='breakdown',
                    name='Breakdown',
                    energy=e['rms'],
                    confidence=0.7
                ))
                cue_points.append(CuePoint(
                    timestamp=e['time'],
                    cue_type='breakdown_end',
                    name='Breakdown End',
                    energy=e['rms'],
                    confidence=0.7
                ))
            breakdown_start = None
    
    # ==================== MIX OUT / OUTRO ====================
    # Buscar donde empieza a bajar la energia para el outro
    outro_start = None
    mix_out = None
    
    # Buscar desde el final hacia atras
    for i in range(len(detailed_energies) - 1, max(len(detailed_energies) - 20, 0), -1):
        e = detailed_energies[i]
        if e['rms'] > avg_rms * 0.6 and outro_start is None:
            outro_start = e['time']
        if e['rms'] > avg_rms * 0.8 and mix_out is None:
            mix_out = e['time']
    
    if outro_start and outro_start < duration - 8:
        cue_points.append(CuePoint(
            timestamp=outro_start,
            cue_type='outro_start',
            name='Outro Start',
            energy=detailed_energies[int(outro_start/4)]['rms'] if int(outro_start/4) < len(detailed_energies) else 0,
            confidence=0.7
        ))
    
    # Mix out: punto optimo para empezar a mezclar el siguiente track
    if mix_out:
        cue_points.append(CuePoint(
            timestamp=mix_out,
            cue_type='mix_out',
            name='Mix Out',
            energy=detailed_energies[int(mix_out/4)]['rms'] if int(mix_out/4) < len(detailed_energies) else 0,
            confidence=0.8
        ))
    else:
        # Default: 32 segundos antes del final
        mix_out_default = max(duration - 32, duration * 0.8)
        cue_points.append(CuePoint(
            timestamp=mix_out_default,
            cue_type='mix_out',
            name='Mix Out',
            energy=0.5,
            confidence=0.6
        ))
    
    # Ordenar por timestamp y convertir a dict
    cue_points.sort(key=lambda x: x.timestamp)
    return [cp.to_dict() for cp in cue_points]


def detect_beat_grid(y, sr, bpm: float) -> Dict:
    """
    Detecta el beat grid preciso para sincronización.

    v3: Enfoque onset-first — usa onset strength para encontrar los transitorios
    reales del audio, filtra outliers, y ajusta la fase con búsqueda de
    máxima correlación contra el grid teórico.

    Returns:
        Dict con first_beat, beat_interval, beats (lista de timestamps)
    """
    import librosa
    import numpy as np

    if bpm <= 0:
        return {
            'first_beat': 0.0,
            'beat_interval': 0.5,
            'beats': [],
        }

    beat_interval = 60.0 / bpm
    duration = len(y) / sr

    # ===== PASO 1: Onset strength — detectar transitorios reales =====
    # Más fiable que librosa.beat.beat_track para música electrónica
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    # ===== PASO 2: Autocorrelación para verificar/corregir BPM =====
    # Verificar que el BPM es correcto usando la autocorrelación del onset
    ac = librosa.autocorrelate(onset_env, max_size=int(sr / hop_length * 2))
    # Buscar pico de autocorrelación cerca del BPM esperado
    expected_lag = int((60.0 / bpm) * sr / hop_length)
    search_range = max(int(expected_lag * 0.1), 3)  # ±10%
    lag_start = max(1, expected_lag - search_range)
    lag_end = min(len(ac) - 1, expected_lag + search_range)
    if lag_end > lag_start:
        best_lag = lag_start + np.argmax(ac[lag_start:lag_end + 1])
        refined_interval = best_lag * hop_length / sr
        # Solo usar si la diferencia es pequeña (< 3%)
        if abs(refined_interval - beat_interval) / beat_interval < 0.03:
            beat_interval = refined_interval

    # ===== PASO 3: Brute-force phase search =====
    # Probar 100 fases candidatas y elegir la que maximiza la
    # energía de onset acumulada en posiciones de beat
    n_candidates = 100
    best_phase = 0.0
    best_score = -1.0

    for i in range(n_candidates):
        phase = i * beat_interval / n_candidates
        score = 0.0
        n_beats = 0
        t = phase
        while t < duration:
            # Encontrar el frame de onset más cercano
            idx = int(t * sr / hop_length)
            if 0 <= idx < len(onset_env):
                # Sumar onset strength en ventana de ±15ms
                window = max(1, int(0.015 * sr / hop_length))
                start = max(0, idx - window)
                end = min(len(onset_env), idx + window + 1)
                score += float(np.max(onset_env[start:end]))
                n_beats += 1
            t += beat_interval

        if n_beats > 0:
            score /= n_beats
        if score > best_score:
            best_score = score
            best_phase = phase

    first_beat = best_phase

    # ===== PASO 4: Refinar fase con sub-frame precision =====
    # Fine-tune alrededor de la mejor fase con resolución de 1ms
    fine_range = beat_interval / n_candidates
    fine_steps = 20
    best_fine_phase = first_beat
    best_fine_score = best_score

    for i in range(fine_steps):
        phase = first_beat - fine_range + (2 * fine_range * i / fine_steps)
        if phase < 0:
            phase += beat_interval
        score = 0.0
        n_beats = 0
        t = phase
        while t < duration:
            idx = int(t * sr / hop_length)
            if 0 <= idx < len(onset_env):
                window = max(1, int(0.010 * sr / hop_length))
                start = max(0, idx - window)
                end = min(len(onset_env), idx + window + 1)
                score += float(np.max(onset_env[start:end]))
                n_beats += 1
            t += beat_interval
        if n_beats > 0:
            score /= n_beats
        if score > best_fine_score:
            best_fine_score = score
            best_fine_phase = phase

    first_beat = best_fine_phase

    # ===== PASO 5: Generar grid y calcular error =====
    grid_beats = []
    t = first_beat
    while t < duration:
        grid_beats.append(round(t, 4))
        t += beat_interval

    # Calcular error usando detected beats de librosa como referencia
    try:
        tempo, lib_beats = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm)
        lib_times = librosa.frames_to_time(lib_beats, sr=sr)
        errors = []
        for bt in lib_times:
            nearest = round((bt - first_beat) / beat_interval) * beat_interval + first_beat
            errors.append(abs(bt - nearest))
        avg_error = float(np.mean(errors)) if errors else 0.0
    except Exception:
        avg_error = 0.0

    max_beats = 500

    return {
        'first_beat': round(first_beat, 4),
        'beat_interval': round(beat_interval, 6),
        'beats': grid_beats[:max_beats],
        'total_beats': len(grid_beats),
        'grid_error_ms': round(avg_error * 1000, 1),
    }


# ==================== 3. MODELO ACTUALIZADO ====================

"""
Anadir estos campos al modelo AnalysisResult en main.py:

class AnalysisResult(BaseModel):
    # ... campos existentes ...
    
    #  Artwork
    artwork_embedded: bool = False  # Si tiene artwork embebido
    artwork_url: Optional[str] = None  # URL del artwork (si se sirve)
    
    #  Cue Points
    cue_points: List[Dict] = []  # Lista de cue points detectados
    
    #  Beat Grid
    first_beat: float = 0.0  # Timestamp del primer beat
    beat_interval: float = 0.5  # Intervalo entre beats en segundos
    
    #  Metadata adicional
    label: Optional[str] = None
    year: Optional[str] = None
    isrc: Optional[str] = None
    genre_source: str = "spectral_analysis"
"""


# ==================== 4. ENDPOINTS NUEVOS ====================

"""
Anadir estos endpoints a main.py:

@app.get("/artwork/{track_id}")
async def get_artwork(track_id: str):
    '''
    Devuelve el artwork de un track como imagen
    '''
    # Buscar en cache
    cache_path_jpg = f"artwork_cache/{track_id}.jpg"
    cache_path_png = f"artwork_cache/{track_id}.png"
    
    if os.path.exists(cache_path_jpg):
        return FileResponse(cache_path_jpg, media_type="image/jpeg")
    elif os.path.exists(cache_path_png):
        return FileResponse(cache_path_png, media_type="image/png")
    
    raise HTTPException(404, "Artwork no encontrado")


@app.get("/cue-points/{track_id}")
async def get_cue_points(track_id: str):
    '''
    Devuelve los cue points de un track
    '''
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track no encontrado")
    
    # Si ya tiene cue points guardados, devolverlos
    if track.get('cue_points'):
        return {"cue_points": track['cue_points']}
    
    raise HTTPException(404, "Cue points no disponibles")
"""


# ==================== 5. INTEGRACIN EN analyze_audio() ====================

"""
Modificar la funcion analyze_audio() en main.py para incluir las nuevas features:

def analyze_audio(file_path: str) -> AnalysisResult:
    # ... codigo existente ...
    
    #  Extraer artwork
    artwork_info = extract_artwork_from_file(file_path)
    artwork_embedded = artwork_info is not None
    
    #  Detectar cue points
    cue_points = detect_cue_points(y, sr, duration, segments)
    
    #  Detectar beat grid
    beat_grid = detect_beat_grid(y, sr, bpm)
    
    return AnalysisResult(
        # ... campos existentes ...
        
        #  Nuevos campos
        artwork_embedded=artwork_embedded,
        cue_points=cue_points,
        first_beat=beat_grid['first_beat'],
        beat_interval=beat_grid['beat_interval'],
    )
"""


# ==================== 6. FUNCIN DE TEST ====================

def test_cue_points():
    """Test de deteccion de cue points con datos simulados"""
    import numpy as np
    
    # Simular 5 minutos de audio con estructura tipica
    duration = 300  # 5 minutos
    sr = 44100
    
    # Crear senal simulada con estructura
    # Intro (0-30s): energia baja
    # Buildup (30-60s): energia subiendo
    # Drop 1 (60-120s): energia alta
    # Breakdown (120-150s): energia baja
    # Buildup 2 (150-180s): energia subiendo
    # Drop 2 (180-240s): energia alta
    # Outro (240-300s): energia bajando
    
    segments = {
        'has_intro': True,
        'has_buildup': True,
        'has_drop': True,
        'has_breakdown': True,
        'has_outro': True,
        'sections': [
            {'type': 'intro', 'start': 0, 'end': 32, 'energy': 0.3},
            {'type': 'body', 'start': 32, 'end': 64, 'energy': 0.6},
            {'type': 'drop', 'start': 64, 'end': 128, 'energy': 0.9},
            {'type': 'breakdown', 'start': 128, 'end': 160, 'energy': 0.4},
            {'type': 'drop', 'start': 160, 'end': 240, 'energy': 0.85},
            {'type': 'outro', 'start': 240, 'end': 300, 'energy': 0.35},
        ]
    }
    
    logger.info("Test de cue points:")
    logger.info(f"  Duracion: {duration}s")
    logger.info(f"  Secciones: {len(segments['sections'])}")
    logger.info("NOTA: Para test real, ejecutar con archivo de audio")


if __name__ == "__main__":
    test_cue_points()
