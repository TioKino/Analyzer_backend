import requests
import time
from typing import Optional

try:
    import discogs_client
    DISCOGS_AVAILABLE = True
except ImportError:
    DISCOGS_AVAILABLE = False
    print("⚠️ discogs_client no instalado. pip install discogs_client")

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
            except Exception as e:
                print(f"⚠️ Error inicializando Discogs: {e}")
    
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
            
        except Exception as e:
            print(f"Error Discogs: {e}")
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
            
        except Exception as e:
            print(f"Error MusicBrainz: {e}")
            return None
    
    def get_acousticbrainz_genre(self, artist: str, title: str) -> Optional[dict]:
        """Obtener género de AcousticBrainz (DEPRECATED - cerró en 2022)"""
        # AcousticBrainz cerró, pero dejamos el método por compatibilidad
        return None
    
    def _normalize_genre(self, genre: str) -> str:
        """Normalizar géneros a categorías consistentes"""
        if not genre:
            return 'Electronic'
            
        genre_lower = genre.lower().strip()
        
        # Mapeo de géneros
        genre_map = {
            # Techno variants
            'techno': 'Techno',
            'tech': 'Techno',
            'detroit techno': 'Detroit Techno',
            'detroit': 'Detroit Techno',
            'minimal techno': 'Minimal Techno',
            'minimal': 'Minimal Techno',
            'acid techno': 'Acid Techno',
            'acid': 'Acid Techno',
            'industrial techno': 'Industrial Techno',
            'hard techno': 'Hard Techno',
            'peak time': 'Peak Time Techno',
            'peak time techno': 'Peak Time Techno',
            'melodic techno': 'Melodic Techno',
            'hypnotic': 'Hypnotic Techno',
            'dub techno': 'Dub Techno',
            'dark techno': 'Dark Techno',
            
            # House variants
            'house': 'House',
            'deep house': 'Deep House',
            'deep': 'Deep House',
            'tech house': 'Tech House',
            'tech-house': 'Tech House',
            'progressive house': 'Progressive House',
            'electro house': 'Electro House',
            'chicago house': 'Chicago House',
            'chicago': 'Chicago House',
            'afro house': 'Afro House',
            'afro': 'Afro House',
            'tribal house': 'Tribal House',
            'tribal': 'Tribal House',
            'funky house': 'Funky House',
            'funky': 'Funky House',
            'soulful house': 'Soulful House',
            'soulful': 'Soulful House',
            'minimal house': 'Minimal House',
            
            # Trance
            'trance': 'Trance',
            'uplifting trance': 'Uplifting Trance',
            'uplifting': 'Uplifting Trance',
            'progressive trance': 'Progressive Trance',
            'psytrance': 'Psytrance',
            'psy': 'Psytrance',
            'goa': 'Goa Trance',
            'goa trance': 'Goa Trance',
            'vocal trance': 'Vocal Trance',
            'tech trance': 'Tech Trance',
            'hard trance': 'Hard Trance',
            
            # Hard dance
            'hardcore': 'Hardcore',
            'hardstyle': 'Hardstyle',
            'hard dance': 'Hardstyle',
            'gabber': 'Gabber',
            'happy hardcore': 'Happy Hardcore',
            'frenchcore': 'Frenchcore',
            
            # Bass music
            'drum & bass': 'Drum & Bass',
            'drum and bass': 'Drum & Bass',
            'drum&bass': 'Drum & Bass',
            'dnb': 'Drum & Bass',
            'd&b': 'Drum & Bass',
            'jungle': 'Jungle',
            'dubstep': 'Dubstep',
            'bass': 'Bass Music',
            'bass music': 'Bass Music',
            'uk garage': 'UK Garage',
            'garage': 'UK Garage',
            'grime': 'Grime',
            'breakbeat': 'Breakbeat',
            'breaks': 'Breakbeat',
            'halftime': 'Halftime',
            
            # Disco & Funk
            'disco': 'Disco',
            'nu disco': 'Nu Disco',
            'nu-disco': 'Nu Disco',
            'italo disco': 'Italo Disco',
            'italo': 'Italo Disco',
            'funk': 'Funk',
            'boogie': 'Boogie',
            
            # Ambient & Chill
            'ambient': 'Ambient',
            'downtempo': 'Downtempo',
            'chillout': 'Chillout',
            'chill out': 'Chillout',
            'chill': 'Chillout',
            'idm': 'IDM',
            
            # Other
            'electro': 'Electro',
            'electronica': 'Electronica',
            'electronic': 'Electronic',
            'edm': 'EDM',
            'synthwave': 'Synthwave',
            'industrial': 'Industrial',
            'ebm': 'EBM',
            'experimental': 'Experimental',
        }
        
        # Buscar coincidencia exacta primero
        if genre_lower in genre_map:
            return genre_map[genre_lower]
        
        # Buscar coincidencias parciales
        for key, value in genre_map.items():
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
