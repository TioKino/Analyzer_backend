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
except ImportError:
    DISCOGS_AVAILABLE = False
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
            except:
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
            
        except (requests.RequestException, AttributeError, IndexError, KeyError) as e:
            logger.error(f"Error Discogs: {e}")
            return None

    def get_musicbrainz_info(self, artist: str, title: str) -> Optional[dict]:
        """Obtener info de MusicBrainz"""
        if not artist or not title:
            return None
        
        try:
            # Buscar en MusicBrainz
            search_url = "https://musicbrainz.org/ws/2/recording/"
            params = {
                'query': f'artist:"{artist}" AND recording:"{title}"',
                'fmt': 'json',
                'limit': 1
            }
            
            headers = {'User-Agent': 'DJAnalyzerPro/2.3 (contact@djanalyzer.pro)'}
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            recordings = data.get('recordings', [])
            
            if not recordings:
                return None
            
            recording = recordings[0]
            
            # Extraer tags como género
            tags = recording.get('tags', [])
            genre = None
            if tags:
                # Ordenar por count y tomar el primero
                sorted_tags = sorted(tags, key=lambda x: x.get('count', 0), reverse=True)
                genre = self._normalize_genre(sorted_tags[0].get('name', ''))
            
            # Rate limit para MusicBrainz (1 req/sec)
            time.sleep(1)
            
            return {
                'genre': genre,
                'mbid': recording.get('id'),
                'source': 'musicbrainz'
            }
            
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error MusicBrainz: {e}")
            return None

    def _normalize_genre(self, genre: str) -> str:
        """Normalizar géneros a categorías consistentes usando el mapeo universal"""
        if not genre:
            return 'Electronic'
            
        genre_lower = genre.lower().strip()
        
        # Buscar coincidencia exacta primero
        if genre_lower in GENRE_MAP:
            return GENRE_MAP[genre_lower]
        
        # Buscar coincidencias parciales
        for key, value in GENRE_MAP.items():
            if key in genre_lower:
                return value
        
        # Si no hay match, capitalizar
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
