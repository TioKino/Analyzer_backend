"""
DJ ANALYZER - Similar Tracks Module v2.3.0
==============================================
Módulo para búsqueda de tracks similares por compatibilidad DJ.

Exporta:
- SimilarTracksRequest (modelo Pydantic)
- SimilarTrackResult (modelo Pydantic)
- CAMELOT_COMPATIBLE (diccionario de compatibilidad armónica)
- is_key_compatible() 
- calculate_compatibility_score()

Los endpoints se definen en main.py usando estas funciones.
"""

from typing import Optional, List
from pydantic import BaseModel


# ==================== MODELOS ====================

class SimilarTracksRequest(BaseModel):
    """Request para buscar tracks similares"""
    bpm: Optional[float] = None
    key: Optional[str] = None
    camelot: Optional[str] = None
    energy_dj: Optional[int] = None
    genre: Optional[str] = None
    track_type: Optional[str] = None
    has_vocals: Optional[bool] = None
    has_drop: Optional[bool] = None
    limit: int = 20
    # Tolerancias personalizables
    bpm_tolerance: int = 4  # ±4 BPM por defecto
    energy_tolerance: int = 2  # ±2 niveles de energía


class SimilarTrackResult(BaseModel):
    """Resultado de búsqueda de tracks similares"""
    id: str
    filename: str
    title: Optional[str]
    artist: Optional[str]
    bpm: float
    key: Optional[str]
    camelot: Optional[str]
    energy_dj: int
    genre: str
    track_type: str
    # Puntuación de compatibilidad (0-100)
    compatibility_score: float
    # Detalles de compatibilidad
    bpm_match: bool
    key_compatible: bool
    energy_match: bool
    genre_match: bool


# ==================== COMPATIBILIDAD ARMÓNICA (Camelot Wheel) ====================

CAMELOT_COMPATIBLE = {
    # Cada key es compatible con: misma, +1, -1, y paralela (A<->B)
    '1A': ['1A', '12A', '2A', '1B'],
    '2A': ['2A', '1A', '3A', '2B'],
    '3A': ['3A', '2A', '4A', '3B'],
    '4A': ['4A', '3A', '5A', '4B'],
    '5A': ['5A', '4A', '6A', '5B'],
    '6A': ['6A', '5A', '7A', '6B'],
    '7A': ['7A', '6A', '8A', '7B'],
    '8A': ['8A', '7A', '9A', '8B'],
    '9A': ['9A', '8A', '10A', '9B'],
    '10A': ['10A', '9A', '11A', '10B'],
    '11A': ['11A', '10A', '12A', '11B'],
    '12A': ['12A', '11A', '1A', '12B'],
    '1B': ['1B', '12B', '2B', '1A'],
    '2B': ['2B', '1B', '3B', '2A'],
    '3B': ['3B', '2B', '4B', '3A'],
    '4B': ['4B', '3B', '5B', '4A'],
    '5B': ['5B', '4B', '6B', '5A'],
    '6B': ['6B', '5B', '7B', '6A'],
    '7B': ['7B', '6B', '8B', '7A'],
    '8B': ['8B', '7B', '9B', '8A'],
    '9B': ['9B', '8B', '10B', '9A'],
    '10B': ['10B', '9B', '11B', '10A'],
    '11B': ['11B', '10B', '12B', '11A'],
    '12B': ['12B', '11B', '1B', '12A'],
}


# ==================== FUNCIONES ====================

def is_key_compatible(camelot1: str, camelot2: str) -> bool:
    """
    Verifica si dos keys son armónicamente compatibles según Camelot Wheel
    
    Args:
        camelot1: Key de referencia (ej: "8A")
        camelot2: Key a comparar (ej: "9A")
    
    Returns:
        True si son compatibles para mezcla
    """
    if not camelot1 or not camelot2:
        return True  # Si no hay info, asumimos compatible
    return camelot2 in CAMELOT_COMPATIBLE.get(camelot1, [camelot1])


def calculate_compatibility_score(
    track: dict,
    target_bpm: float = None,
    target_camelot: str = None,
    target_energy: int = None,
    target_genre: str = None,
    bpm_tolerance: int = 4,
    energy_tolerance: int = 2
) -> tuple:
    """
    Calcula puntuación de compatibilidad DJ (0-100) y detalles
    
    Pesos:
    - BPM: 30 puntos (crítico para mezcla beatmatch)
    - Key: 30 puntos (armonía musical)
    - Energía: 25 puntos (flow del set)
    - Género: 15 puntos (coherencia estilística)
    
    Args:
        track: Dict con datos del track a evaluar
        target_bpm: BPM de referencia
        target_camelot: Key Camelot de referencia
        target_energy: Energía DJ (1-10) de referencia
        target_genre: Género de referencia
        bpm_tolerance: Tolerancia de BPM (default ±4)
        energy_tolerance: Tolerancia de energía (default ±2)
    
    Returns:
        Tuple de (score, bpm_match, key_compatible, energy_match, genre_match)
    """
    score = 0
    bpm_match = False
    key_compatible = False
    energy_match = False
    genre_match = False
    
    track_bpm = track.get('bpm', 0)
    track_camelot = track.get('camelot', '')
    track_energy = track.get('energy_dj', 5)
    track_genre = (track.get('genre', '') or '').lower()
    
    # ==================== BPM (30 puntos) ====================
    if target_bpm:
        bpm_diff = abs(track_bpm - target_bpm)
        if bpm_diff <= bpm_tolerance:
            score += 30
            bpm_match = True
        elif bpm_diff <= bpm_tolerance * 2:
            score += 15  # Medio punto
        # También considerar doble/mitad tempo
        elif abs(track_bpm - target_bpm * 2) <= bpm_tolerance or abs(track_bpm - target_bpm / 2) <= bpm_tolerance:
            score += 20
            bpm_match = True
    else:
        score += 30  # Sin filtro BPM = todos cumplen
        bpm_match = True
    
    # ==================== KEY (30 puntos) ====================
    if target_camelot:
        if is_key_compatible(target_camelot, track_camelot):
            score += 30
            key_compatible = True
        elif track_camelot:
            # Key no compatible pero existe
            score += 5
    else:
        score += 30
        key_compatible = True
    
    # ==================== ENERGÍA (25 puntos) ====================
    if target_energy:
        energy_diff = abs(track_energy - target_energy)
        if energy_diff <= energy_tolerance:
            score += 25
            energy_match = True
        elif energy_diff <= energy_tolerance * 2:
            score += 12
    else:
        score += 25
        energy_match = True
    
    # ==================== GÉNERO (15 puntos) ====================
    if target_genre:
        target_genre_lower = target_genre.lower()
        if track_genre == target_genre_lower:
            score += 15
            genre_match = True
        elif target_genre_lower in track_genre or track_genre in target_genre_lower:
            score += 10
            genre_match = True
        # Géneros relacionados
        elif _are_genres_related(track_genre, target_genre_lower):
            score += 8
            genre_match = True
    else:
        score += 15
        genre_match = True
    
    return score, bpm_match, key_compatible, energy_match, genre_match


def _are_genres_related(genre1: str, genre2: str) -> bool:
    """
    Verifica si dos géneros están relacionados (misma familia)
    
    Args:
        genre1: Primer género (lowercase)
        genre2: Segundo género (lowercase)
    
    Returns:
        True si pertenecen a la misma familia de géneros
    """
    related_groups = [
        ['techno', 'industrial techno', 'hard techno', 'peak time techno', 
         'melodic techno', 'hypnotic techno', 'acid techno', 'detroit techno',
         'minimal techno', 'dub techno'],
        ['house', 'tech house', 'deep house', 'progressive house',
         'minimal house', 'electro house', 'acid house', 'chicago house'],
        ['trance', 'progressive trance', 'uplifting trance', 'psytrance',
         'vocal trance', 'tech trance', 'goa trance'],
        ['drum & bass', 'dnb', 'liquid dnb', 'neurofunk', 'jungle', 'breakbeat'],
        ['dubstep', 'riddim', 'brostep', 'bass music'],
        ['hardcore', 'hardstyle', 'uptempo hardcore', 'gabber', 'frenchcore'],
    ]
    
    for group in related_groups:
        if genre1 in group and genre2 in group:
            return True
    return False
