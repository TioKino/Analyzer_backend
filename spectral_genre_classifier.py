"""
============================================================
CLASIFICACIÓN DE GÉNERO POR ANÁLISIS ESPECTRAL
============================================================

Sistema UNIVERSAL de clasificación de género basado en
características de audio: BPM, energía, espectro, etc.

Esta función se usa como FALLBACK cuando no hay información
de Discogs, MusicBrainz o memoria colectiva.
"""

import numpy as np


def classify_genre_advanced(bpm: float, energy: float, has_bass: bool, 
                           y, sr, percussion_density: float, 
                           spectral_centroid, rolloff) -> str:
    """
    Clasifica el género musical basándose en análisis espectral.
    
    Parámetros:
    - bpm: Tempo detectado
    - energy: Nivel de energía normalizado (0-1)
    - has_bass: Si tiene graves prominentes
    - y: Señal de audio
    - sr: Sample rate
    - percussion_density: Densidad de percusión (0-1)
    - spectral_centroid: Centroide espectral
    - rolloff: Rolloff espectral
    
    Retorna:
    - Género clasificado (string)
    """
    
    brightness = float(np.mean(spectral_centroid))
    rolloff_mean = float(np.mean(rolloff))
    
    # Calcular características adicionales
    spectral_std = float(np.std(spectral_centroid))
    is_melodic = spectral_std > 800  # Alta variación = más melódico
    is_dark = brightness < 2000
    is_bright = brightness > 3500
    
    # ============================================================
    # DETECCIÓN POR RANGO DE BPM
    # ============================================================
    
    # === HIP HOP / TRAP / R&B (60-100 BPM) ===
    if 60 <= bpm < 100:
        if percussion_density > 0.6 and has_bass:
            if brightness > 2500:
                return "Trap"
            return "Hip Hop"
        if is_melodic and energy < 0.5:
            return "R&B"
        if energy < 0.4:
            return "Lo-Fi Hip Hop"
        if has_bass and energy > 0.6:
            return "Phonk"
        return "Hip Hop"
    
    # === REGGAETON / DEMBOW / LATIN (90-110 BPM) ===
    if 90 <= bpm < 115:
        if percussion_density > 0.7:
            if has_bass and energy > 0.6:
                return "Reggaeton"
            return "Dembow"
        if is_melodic and energy < 0.5:
            return "Bachata"
        if energy > 0.7 and has_bass:
            return "Latin Urban"
        return "Latin"
    
    # === HOUSE (115-130 BPM) ===
    if 115 <= bpm < 130:
        if has_bass and energy > 0.7:
            if percussion_density > 0.7:
                return "Tech House"
            if is_bright:
                return "Bass House"
            return "Tech House"
        
        if brightness > 3000:
            if is_melodic:
                return "Progressive House"
            return "Electro House"
        
        if is_dark and energy < 0.5:
            return "Deep House"
        
        if is_melodic and energy < 0.6:
            if brightness < 2500:
                return "Organic House"
            return "Melodic House"
        
        if percussion_density < 0.4:
            return "Minimal House"
        
        if has_bass and energy > 0.5:
            return "Jackin House"
        
        return "House"
    
    # === TECHNO (120-140 BPM) ===
    if 120 <= bpm <= 140:
        # Hard Techno
        if energy > 0.75 and percussion_density > 0.7:
            if is_dark:
                return "Industrial Techno"
            return "Hard Techno"
        
        # Peak Time
        if energy > 0.65 and 0.5 < percussion_density < 0.8:
            return "Peak Time Techno"
        
        # Acid Techno
        if brightness > 3500 and energy > 0.55:
            return "Acid Techno"
        
        # Detroit Techno
        if 0.4 < percussion_density < 0.7 and brightness > 2500:
            return "Detroit Techno"
        
        # Melodic Techno
        if is_melodic and brightness > 2800 and 0.4 < energy < 0.7:
            return "Melodic Techno"
        
        # Dub Techno
        if is_dark and energy < 0.5 and percussion_density < 0.5:
            return "Dub Techno"
        
        # Hypnotic Techno
        if energy < 0.5 and percussion_density < 0.5:
            return "Hypnotic Techno"
        
        # Minimal Techno
        if percussion_density < 0.4 and not is_melodic:
            return "Minimal Techno"
        
        # Berlin/Warehouse
        if has_bass and is_dark and energy > 0.5:
            return "Berlin Techno"
        
        return "Techno"
    
    # === TRANCE (130-150 BPM) ===
    if 130 <= bpm <= 150:
        if is_bright and is_melodic:
            if energy > 0.7:
                return "Uplifting Trance"
            return "Vocal Trance"
        
        if energy > 0.65 and brightness > 2500:
            return "Progressive Trance"
        
        if is_dark and energy > 0.7:
            return "Tech Trance"
        
        if bpm > 145:
            return "Psytrance"
        
        return "Trance"
    
    # === HARD DANCE (145-180 BPM) ===
    if 145 <= bpm <= 180:
        if bpm > 170:
            if energy > 0.8:
                return "Gabber"
            return "Speedcore"
        
        if bpm > 160:
            if is_melodic and brightness > 3000:
                return "Happy Hardcore"
            return "Hardcore"
        
        if energy > 0.8:
            if is_melodic:
                return "Euphoric Hardstyle"
            return "Rawstyle"
        
        if is_dark and has_bass:
            return "Frenchcore"
        
        return "Hardstyle"
    
    # === DRUM & BASS (160-180 BPM) ===
    if 160 <= bpm <= 180:
        if brightness > 3000 and is_melodic:
            return "Liquid Drum & Bass"
        
        if is_dark and energy > 0.7:
            return "Neurofunk"
        
        if percussion_density > 0.8:
            return "Jump Up"
        
        if is_dark and energy < 0.5:
            return "Atmospheric DnB"
        
        return "Drum & Bass"
    
    # === JUNGLE (150-170 BPM con breakbeats) ===
    if 150 <= bpm <= 175:
        if percussion_density > 0.75 and spectral_std > 1000:
            return "Jungle"
    
    # === DUBSTEP (70 BPM o 140 BPM half-time) ===
    if (68 <= bpm <= 75) or (136 <= bpm <= 145):
        if has_bass and energy > 0.6:
            if brightness > 3000:
                return "Brostep"
            if is_dark:
                return "Deep Dubstep"
            if percussion_density > 0.7:
                return "Riddim"
            return "Dubstep"
        if is_melodic:
            return "Melodic Dubstep"
        if energy < 0.4:
            return "Chillstep"
    
    # === BREAKBEAT / UK SOUND (130-145 BPM) ===
    if 130 <= bpm <= 145:
        if spectral_std > 1000 and percussion_density > 0.6:
            if brightness > 2500:
                return "Breakbeat"
            return "UK Garage"
    
    # === AMBIENT / DOWNTEMPO (60-100 BPM bajo) ===
    if bpm < 100 and energy < 0.4:
        if percussion_density < 0.2:
            if is_dark:
                return "Dark Ambient"
            return "Ambient"
        if is_melodic:
            return "Downtempo"
        return "Chillout"
    
    # === POP / DANCE POP (100-130 BPM) ===
    if 100 <= bpm <= 130:
        if is_melodic and is_bright:
            if energy > 0.6:
                return "Dance Pop"
            return "Pop"
        if energy > 0.7 and brightness > 3000:
            return "EDM"
    
    # === ROCK / METAL (Variable) ===
    if 80 <= bpm <= 200:
        # Características de rock/metal: distorsión, sustain largo
        if brightness > 4000 and energy > 0.8:
            if bpm > 160:
                return "Thrash Metal"
            if bpm > 140:
                return "Heavy Metal"
            return "Hard Rock"
        
        if is_melodic and 100 <= bpm <= 140:
            if energy > 0.6:
                return "Rock"
            return "Alternative Rock"
    
    # === DISCO / FUNK (110-130 BPM) ===
    if 110 <= bpm <= 130:
        if is_melodic and brightness > 2500:
            if energy > 0.6:
                return "Disco"
            return "Nu Disco"
        if percussion_density > 0.5 and has_bass:
            return "Funk"
    
    # === JAZZ (Variable, típicamente 80-200 BPM) ===
    if spectral_std > 1200 and percussion_density < 0.5:
        if is_melodic and not has_bass:
            return "Jazz"
    
    # === REGGAE / DUB (60-90 BPM) ===
    if 60 <= bpm <= 90:
        if has_bass and energy < 0.6:
            if is_dark:
                return "Dub"
            return "Reggae"
    
    # === SYNTHWAVE (80-120 BPM) ===
    if 80 <= bpm <= 120:
        if brightness > 3000 and is_melodic:
            if is_dark:
                return "Darksynth"
            return "Synthwave"
    
    # === BIG ROOM / FESTIVAL (126-132 BPM) ===
    if 126 <= bpm <= 132:
        if energy > 0.8 and brightness > 3500:
            return "Big Room"
        if has_bass and energy > 0.7:
            return "Festival Progressive"
    
    # === MOOMBAHTON (100-112 BPM) ===
    if 100 <= bpm <= 112:
        if has_bass and percussion_density > 0.6:
            return "Moombahton"
    
    # === FUTURE BASS (140-160 BPM con características específicas) ===
    if 140 <= bpm <= 160:
        if is_bright and is_melodic and energy > 0.5:
            return "Future Bass"
    
    # === ELECTRO (125-135 BPM) ===
    if 125 <= bpm <= 135:
        if brightness > 3000 and percussion_density > 0.5:
            return "Electro"
    
    # === GRIME (140 BPM típico) ===
    if 138 <= bpm <= 142:
        if has_bass and energy > 0.6:
            if brightness < 2500:
                return "Grime"
    
    # === FALLBACK POR ENERGÍA ===
    if energy > 0.7:
        if bpm > 140:
            return "Hard Dance"
        return "EDM"
    
    if energy < 0.4:
        if is_melodic:
            return "Chillout"
        return "Ambient"
    
    # === FALLBACK FINAL ===
    return "Electronic"


def get_genre_characteristics(genre: str) -> dict:
    """
    Retorna las características típicas de un género.
    Útil para sugerencias y validación.
    """
    characteristics = {
        'Techno': {'bpm_range': (120, 140), 'energy': 'high', 'bass': True},
        'House': {'bpm_range': (118, 130), 'energy': 'medium-high', 'bass': True},
        'Trance': {'bpm_range': (130, 150), 'energy': 'high', 'melodic': True},
        'Drum & Bass': {'bpm_range': (160, 180), 'energy': 'high', 'bass': True},
        'Dubstep': {'bpm_range': (140, 150), 'energy': 'high', 'bass': True},
        'Hip Hop': {'bpm_range': (70, 100), 'energy': 'medium', 'bass': True},
        'Trap': {'bpm_range': (130, 170), 'energy': 'medium-high', 'bass': True},
        'Reggaeton': {'bpm_range': (90, 100), 'energy': 'medium-high', 'bass': True},
        'Pop': {'bpm_range': (100, 130), 'energy': 'medium', 'melodic': True},
        'Rock': {'bpm_range': (100, 140), 'energy': 'high', 'distortion': True},
        'Metal': {'bpm_range': (100, 200), 'energy': 'very_high', 'distortion': True},
        'Jazz': {'bpm_range': (80, 200), 'energy': 'variable', 'melodic': True},
        'Ambient': {'bpm_range': (60, 100), 'energy': 'low', 'melodic': True},
    }
    
    return characteristics.get(genre, {'bpm_range': (60, 200), 'energy': 'variable'})
