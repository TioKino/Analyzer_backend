"""
============================================================
DETECCIÓN DE GÉNEROS - DJ ANALYZER
============================================================

Sistema UNIVERSAL de detección y normalización de géneros.
Cubre todos los estilos musicales del mundo.
Integra Discogs, MusicBrainz y análisis espectral.

Total: ~400+ géneros mapeados
"""

import json
import logging
import os
import requests
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import discogs_client
    DISCOGS_AVAILABLE = True
    # Excepciones del SDK de Discogs, para capturarlas explicitamente en
    # get_discogs_genre y separarlas de errores genericos de red.
    try:
        from discogs_client.exceptions import DiscogsAPIError, HTTPError as DiscogsHTTPError
        _DISCOGS_EXCS = (DiscogsAPIError, DiscogsHTTPError)
    except ImportError:
        _DISCOGS_EXCS = ()
except ImportError:
    DISCOGS_AVAILABLE = False
    _DISCOGS_EXCS = ()
    logger.warning("discogs_client no instalado. pip install discogs_client")


# ============================================================
# MAPEO UNIVERSAL DE GÉNEROS (cargado desde data/genre_mappings.json)
# ============================================================

_GENRE_MAP_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'genre_mappings.json'

def _load_genre_map() -> dict:
    try:
        return json.loads(_GENRE_MAP_PATH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        logger.error(f"Genre mappings file not found: {_GENRE_MAP_PATH}")
        return {}

GENRE_MAP = _load_genre_map()

# Pre-computado al cargar el modulo: claves de GENRE_MAP ordenadas por
# longitud descendente. Usado en _normalize_genre para que las claves mas
# especificas ("industrial techno", 17 chars) ganen sobre las genericas
# ("techno", 6 chars) en el match parcial. Sin esto, el primer match en
# el dict (orden de insercion) ganaba arbitrariamente y se perdia
# especificidad de genero. Cierra A6 del audit 2026-05-08.
_GENRE_KEYS_BY_SPECIFICITY = sorted(GENRE_MAP.keys(), key=len, reverse=True)




class GenreDetector:
    def __init__(self, discogs_token=None):
        self.discogs_token = discogs_token
        self.discogs_client = None
        
        if discogs_token and DISCOGS_AVAILABLE:
            try:
                self.discogs_client = discogs_client.Client(
                    'DJAnalyzerPro/2.3',
                    user_token=discogs_token
                )
            except (ImportError, ValueError, ConnectionError) as e:
                logger.warning(f"Error inicializando Discogs: {e}")
    
    def get_discogs_genre(self, artist: str, title: str) -> Optional[dict]:
        """Obtener género de Discogs"""
        if not self.discogs_client or not artist or not title:
            return None
        
        try:
            # Buscar release
            query = f"{artist} {title}"
            results = self.discogs_client.search(query, type='release')
            
            # Iterar sobre resultados (evita el error de slicing)
            release = None
            for r in results:
                release = r
                break
            
            if not release:
                return None
            
            # Obtener géneros
            genres = getattr(release, 'genres', None)
            styles = getattr(release, 'styles', None)
            
            # Obtener label y year si están disponibles
            label = None
            year = None
            try:
                if hasattr(release, 'labels') and release.labels:
                    label = release.labels[0].name if hasattr(release.labels[0], 'name') else str(release.labels[0])
                if hasattr(release, 'year'):
                    year = release.year
            except (AttributeError, IndexError):
                # Pre-A6: era 'except: pass' (capturaba SystemExit/KeyboardInterrupt).
                # Ahora solo los errores plausibles del API de discogs_client.
                pass
            
            if styles:
                # Usar estilo (más específico)
                return {
                    'genre': self._normalize_genre(styles[0]),
                    'confidence': 0.85,
                    'source': 'discogs',
                    'parent_genre': self._normalize_genre(genres[0]) if genres else None,
                    'label': label,
                    'year': year
                }
            elif genres:
                return {
                    'genre': self._normalize_genre(genres[0]),
                    'confidence': 0.7,
                    'source': 'discogs',
                    'label': label,
                    'year': year
                }
            
            return None
            
        except (requests.RequestException, AttributeError, IndexError, KeyError, ValueError) + _DISCOGS_EXCS as e:
            # ValueError cubre json.JSONDecodeError de discogs_client cuando
            # la API devuelve cuerpo vacio (rate limit / 5xx transitorios).
            # _DISCOGS_EXCS cubre HTTPError/DiscogsAPIError del SDK.
            # Antes era logger.error: hacia ruido en alertas para algo que
            # no es bug nuestro y para el que ya tenemos fallback a MB.
            logger.warning(f"Discogs no disponible para '{artist} - {title}': {type(e).__name__}: {e}")
            return None
        except Exception as e:
            # Red de seguridad: cualquier excepcion no anticipada (cambios
            # internos del SDK, TimeoutError nativo, etc) tambien debe
            # caer aqui para que main.py:cascade_genres no la propague
            # como ERROR. Verificado contra discogs_client 2.3.0 donde
            # HTTPError esta en _DISCOGS_EXCS, pero la libreria podria
            # introducir clases nuevas en versiones futuras.
            logger.warning(f"Discogs error inesperado para '{artist} - {title}': {type(e).__name__}: {e}")
            return None

    def get_musicbrainz_info(self, artist: str, title: str) -> Optional[dict]:
        """Obtener info de MusicBrainz con timeout y retry."""
        if not artist or not title:
            return None

        # MB latency mediana ~2-5s pero p99 puede irse a 15-20s en horas pico.
        # Timeout=10s producia muchos "Read timed out" en los logs. Subimos
        # a 20s con 1 retry simple con backoff de 2s.
        search_url = "https://musicbrainz.org/ws/2/recording/"
        params = {
            'query': f'artist:"{artist}" AND recording:"{title}"',
            'fmt': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'DJAnalyzerPro/2.3 (contact@djanalyzer.pro)'}

        response = None
        for attempt in range(2):
            try:
                response = requests.get(search_url, params=params, headers=headers, timeout=20)
                break
            except requests.Timeout:
                if attempt == 0:
                    time.sleep(2)
                    continue
                logger.warning(
                    f"MusicBrainz timeout tras 2 intentos para '{artist} - {title}'"
                )
                return None
            except requests.RequestException as e:
                logger.warning(f"MusicBrainz request fallo: {type(e).__name__}: {e}")
                return None

        if response is None or response.status_code != 200:
            return None

        try:
            data = response.json()
        except ValueError as e:
            logger.warning(f"MusicBrainz JSON invalido: {e}")
            return None

        recordings = data.get('recordings', [])
        if not recordings:
            return None

        recording = recordings[0]

        # Extraer tags como género
        tags = recording.get('tags', [])
        genre = None
        if tags:
            sorted_tags = sorted(tags, key=lambda x: x.get('count', 0), reverse=True)
            genre = self._normalize_genre(sorted_tags[0].get('name', ''))

        # Rate limit para MusicBrainz (1 req/sec)
        time.sleep(1)

        return {
            'genre': genre,
            'mbid': recording.get('id'),
            'source': 'musicbrainz'
        }

    def _normalize_genre(self, genre: str) -> str:
        """Normalizar géneros a categorías consistentes usando el mapeo universal.

        Estrategia:
        1. Match exacto contra GENRE_MAP (O(1) dict lookup).
        2. Match parcial (substring) iterando claves ordenadas por longitud
           descendente: las claves mas especificas ("industrial techno",
           17 chars) ganan sobre las genericas ("techno", 6 chars).
           Antes el orden era el de insercion del dict y "techno" ganaba
           arbitrariamente, perdiendo especificidad.
        3. Sin match: capitalizar el input crudo.
        """
        if not genre:
            return 'Electronic'

        genre_lower = genre.lower().strip()

        # 1. Match exacto
        if genre_lower in GENRE_MAP:
            return GENRE_MAP[genre_lower]

        # 2. Match parcial — las claves mas especificas ganan
        for key in _GENRE_KEYS_BY_SPECIFICITY:
            if key in genre_lower:
                return GENRE_MAP[key]

        # 3. Sin match
        return genre.title()
    
    def detect_genre(self, artist: str, title: str, 
                    collective_genre: Optional[str] = None,
                    spectral_genre: str = "Electronic") -> dict:
        """
        Detectar género con sistema de prioridades
        
        Prioridad:
        1. Memoria colectiva (correcciones de usuarios)
        2. Discogs (más específico para electrónica)
        3. MusicBrainz
        4. Análisis espectral
        """
        
        # 1. Memoria colectiva (máxima prioridad)
        if collective_genre:
            return {
                'genre': collective_genre,
                'confidence': 1.0,
                'source': 'collective_memory'
            }
        
        # 2. Discogs (mejor para electrónica)
        discogs_result = self.get_discogs_genre(artist, title)
        if discogs_result:
            return discogs_result
        
        # 3. MusicBrainz
        mb_result = self.get_musicbrainz_info(artist, title)
        if mb_result and mb_result.get('genre'):
            return mb_result
        
        # 4. Fallback: Análisis espectral
        return {
            'genre': spectral_genre,
            'confidence': 0.5,
            'source': 'spectral_analysis'
        }


def get_all_genres() -> list:
    """Retorna lista de todos los géneros normalizados únicos"""
    return sorted(list(set(GENRE_MAP.values())))


def get_genre_count() -> int:
    """Retorna el número total de géneros únicos"""
    return len(set(GENRE_MAP.values()))


def search_genres(query: str) -> list:
    """Busca géneros que contengan el texto dado"""
    if not query:
        return []
    query_lower = query.lower()
    all_genres = set(GENRE_MAP.values())
    return sorted([g for g in all_genres if query_lower in g.lower()])
