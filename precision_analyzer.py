"""
DJ Analyzer Pro - Precision Structure & Cue Point Analysis v3.0
================================================================

Reemplaza los algoritmos básicos de detect_structure() y detect_cue_points()
con detección real basada en:

1. NOVELTY CURVE: Detecta cambios espectrales reales (no promedios por ventana)
2. BEAT-ALIGNED: Todos los timestamps snap al beat/bar más cercano
3. ONSET STRENGTH: Usa peaks de onset para encontrar transiciones exactas
4. MFCC SEGMENTATION: Agrupa frames por similitud tímbrica real

Requiere: librosa, numpy, scipy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    from scipy.signal import find_peaks, medfilt
    from scipy.ndimage import uniform_filter1d
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False


# ============================================================================
# UTILIDADES DE BEAT ALIGNMENT
# ============================================================================

def get_beat_times(y: np.ndarray, sr: int, bpm: float = None) -> Tuple[np.ndarray, float, float]:
    """
    Obtiene beat grid preciso.
    
    Returns:
        (beat_times, first_beat, beat_interval)
    """
    if bpm and bpm > 0:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm)
    else:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    if len(beat_times) < 2:
        interval = 60.0 / (bpm if bpm and bpm > 0 else 120.0)
        return np.array([0.0]), 0.0, interval
    
    intervals = np.diff(beat_times)
    beat_interval = float(np.median(intervals))
    first_beat = float(beat_times[0])
    
    return beat_times, first_beat, beat_interval


def snap_to_beat(timestamp: float, beat_times: np.ndarray) -> float:
    """Snap un timestamp al beat más cercano."""
    if len(beat_times) == 0:
        return timestamp
    idx = np.argmin(np.abs(beat_times - timestamp))
    return float(beat_times[idx])


def snap_to_bar(timestamp: float, beat_times: np.ndarray, beats_per_bar: int = 4) -> float:
    """Snap un timestamp al inicio de compás (bar) más cercano."""
    if len(beat_times) < beats_per_bar:
        return snap_to_beat(timestamp, beat_times)
    
    # Bar boundaries = cada 4 beats
    bar_times = beat_times[::beats_per_bar]
    if len(bar_times) == 0:
        return timestamp
    
    idx = np.argmin(np.abs(bar_times - timestamp))
    return float(bar_times[idx])


def snap_to_phrase(timestamp: float, beat_times: np.ndarray, bars_per_phrase: int = 4) -> float:
    """Snap al inicio de frase más cercano (típicamente 16 beats = 4 compases)."""
    beats_per_phrase = 4 * bars_per_phrase  # 16 beats
    if len(beat_times) < beats_per_phrase:
        return snap_to_bar(timestamp, beat_times)
    
    phrase_times = beat_times[::beats_per_phrase]
    if len(phrase_times) == 0:
        return timestamp
    
    idx = np.argmin(np.abs(phrase_times - timestamp))
    return float(phrase_times[idx])


# ============================================================================
# NOVELTY / SEGMENTATION CURVE
# ============================================================================

def compute_novelty_curve(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computa curva de novelty espectral para detectar cambios tímbricos.
    Combina MFCC novelty + onset strength para máxima precisión.
    
    Returns:
        (novelty_curve, time_axis)
    """
    # 1. MFCC-based novelty (cambios tímbricos)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_delta = np.diff(mfcc, axis=1)
    mfcc_novelty = np.sum(np.abs(mfcc_delta), axis=0)
    
    # 2. Chroma-based novelty (cambios armónicos)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_delta = np.diff(chroma, axis=1)
    chroma_novelty = np.sum(np.abs(chroma_delta), axis=0)
    
    # 3. RMS energy curve (cambios de volumen)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_delta = np.abs(np.diff(rms))
    
    # Alinear longitudes
    min_len = min(len(mfcc_novelty), len(chroma_novelty), len(rms_delta))
    mfcc_novelty = mfcc_novelty[:min_len]
    chroma_novelty = chroma_novelty[:min_len]
    rms_delta = rms_delta[:min_len]
    
    # Normalizar cada componente
    if np.max(mfcc_novelty) > 0:
        mfcc_novelty = mfcc_novelty / np.max(mfcc_novelty)
    if np.max(chroma_novelty) > 0:
        chroma_novelty = chroma_novelty / np.max(chroma_novelty)
    if np.max(rms_delta) > 0:
        rms_delta = rms_delta / np.max(rms_delta)
    
    # Combinar (MFCC tiene más peso para estructura, RMS para energía)
    novelty = mfcc_novelty * 0.5 + chroma_novelty * 0.2 + rms_delta * 0.3
    
    # Suavizar para evitar falsos positivos
    novelty = uniform_filter1d(novelty, size=int(sr / hop_length * 1.0))  # ventana ~1s
    
    time_axis = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=hop_length)
    
    return novelty, time_axis


def compute_energy_curve(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Curva de energía RMS suavizada.
    
    Returns:
        (energy_curve, time_axis) ambos normalizados
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Suavizar con ventana de ~2 segundos
    window = max(1, int(sr / hop_length * 2.0))
    rms_smooth = uniform_filter1d(rms, size=window)
    
    # Normalizar
    if np.max(rms_smooth) > 0:
        rms_smooth = rms_smooth / np.max(rms_smooth)
    
    time_axis = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)
    
    return rms_smooth, time_axis


# ============================================================================
# DETECCIÓN DE SECCIONES PRECISA
# ============================================================================

def detect_structure_precise(y: np.ndarray, sr: int, bpm: float = None, 
                              beat_times: np.ndarray = None) -> Dict:
    """
    Detección de estructura musical precisa usando novelty curve.
    
    En lugar de dividir en ventanas fijas, encuentra los puntos reales
    donde la música cambia y alinea todo a frases musicales (16 beats).
    
    Returns:
        Dict con 'sections', 'has_intro', 'has_buildup', etc.
    """
    duration = len(y) / sr
    
    if beat_times is None or len(beat_times) < 4:
        beat_times, _, _ = get_beat_times(y, sr, bpm)
    
    # 1. Compute novelty curve
    novelty, novelty_times = compute_novelty_curve(y, sr)
    energy, energy_times = compute_energy_curve(y, sr)
    
    # 2. Find peaks en novelty = section boundaries
    # Distancia mínima entre secciones: ~8 segundos (2 compases a 120bpm)
    min_distance = int(8.0 * sr / 512)
    prominence = np.std(novelty) * 1.2  # Solo peaks significativos
    
    peaks, properties = find_peaks(
        novelty,
        distance=min_distance,
        prominence=prominence,
        height=np.mean(novelty) + np.std(novelty) * 0.5,
    )
    
    # 3. Convertir peaks a timestamps y snap a frase
    boundary_times = [0.0]  # Siempre empieza en 0
    
    for peak_idx in peaks:
        if peak_idx < len(novelty_times):
            raw_time = novelty_times[peak_idx]
            # Snap a frase musical (16 beats) para secciones
            snapped = snap_to_phrase(raw_time, beat_times, bars_per_phrase=4)
            # Evitar duplicados y boundaries demasiado cercanos
            if snapped > boundary_times[-1] + 4.0 and snapped < duration - 4.0:
                boundary_times.append(snapped)
    
    boundary_times.append(duration)
    
    # 4. Si hay muy pocas secciones, intentar con threshold más bajo
    if len(boundary_times) < 4 and duration > 60:
        peaks2, _ = find_peaks(
            novelty,
            distance=min_distance,
            prominence=prominence * 0.6,
        )
        for peak_idx in peaks2:
            if peak_idx < len(novelty_times):
                raw_time = novelty_times[peak_idx]
                snapped = snap_to_phrase(raw_time, beat_times, bars_per_phrase=4)
                if all(abs(snapped - bt) > 8.0 for bt in boundary_times) and 4.0 < snapped < duration - 4.0:
                    boundary_times.append(snapped)
        boundary_times.sort()
    
    # 5. Clasificar cada sección por su perfil de energía
    sections = []
    avg_energy_global = np.mean(energy)
    
    for i in range(len(boundary_times) - 1):
        start = boundary_times[i]
        end = boundary_times[i + 1]
        
        # Energía media de esta sección
        mask = (energy_times >= start) & (energy_times < end)
        section_energy = float(np.mean(energy[mask])) if np.any(mask) else 0.5
        
        # Tendencia de energía (subiendo, bajando, estable)
        if np.sum(mask) > 2:
            section_rms = energy[mask]
            first_quarter = np.mean(section_rms[:len(section_rms)//4+1])
            last_quarter = np.mean(section_rms[3*len(section_rms)//4:])
            trend = last_quarter - first_quarter
        else:
            trend = 0.0
        
        # Clasificar
        section_type = _classify_section(
            section_energy, avg_energy_global, trend,
            i, len(boundary_times) - 1, duration, start, end
        )
        
        sections.append({
            'type': section_type,
            'start': round(start, 2),
            'end': round(end, 2),
            'energy': round(section_energy, 3),
        })
    
    # 6. Post-proceso: fusionar secciones consecutivas iguales muy cortas
    sections = _merge_short_sections(sections, min_duration=6.0)
    
    # 7. Determinar flags
    types_present = set(s['type'] for s in sections)
    
    return {
        'has_intro': 'intro' in types_present,
        'has_buildup': 'buildup' in types_present,
        'has_drop': 'drop' in types_present,
        'has_breakdown': 'breakdown' in types_present,
        'has_outro': 'outro' in types_present,
        'sections': sections,
        'source': 'precision_v3',
    }


def _classify_section(energy: float, avg: float, trend: float,
                       index: int, total: int, duration: float,
                       start: float, end: float) -> str:
    """
    Clasifica una sección por su energía, posición y tendencia.
    """
    section_duration = end - start
    fraction_start = start / duration
    fraction_end = end / duration
    
    # INTRO: primeras secciones con energía baja
    if index == 0 and energy < avg * 0.7:
        return 'intro'
    if fraction_start < 0.15 and energy < avg * 0.65:
        return 'intro'
    
    # OUTRO: últimas secciones con energía baja
    if index == total - 1 and energy < avg * 0.7:
        return 'outro'
    if fraction_end > 0.85 and energy < avg * 0.65:
        return 'outro'
    
    # DROP: energía alta, estable o ligeramente bajando
    if energy > avg * 1.25:
        return 'drop'
    
    # BUILDUP: energía media-alta con tendencia creciente
    if trend > 0.08 and energy > avg * 0.6:
        return 'buildup'
    
    # BREAKDOWN: energía baja en el medio del track
    if energy < avg * 0.55 and 0.15 < fraction_start < 0.85:
        return 'breakdown'
    
    # BODY: todo lo demás
    return 'body'


def _merge_short_sections(sections: List[Dict], min_duration: float = 6.0) -> List[Dict]:
    """Fusiona secciones consecutivas del mismo tipo si son muy cortas."""
    if len(sections) <= 1:
        return sections
    
    merged = [sections[0]]
    for s in sections[1:]:
        prev = merged[-1]
        if s['type'] == prev['type'] or (s['end'] - s['start'] < min_duration):
            # Fusionar
            merged[-1] = {
                'type': prev['type'] if (prev['end'] - prev['start']) >= (s['end'] - s['start']) else s['type'],
                'start': prev['start'],
                'end': s['end'],
                'energy': round((prev['energy'] + s['energy']) / 2, 3),
            }
        else:
            merged.append(s)
    
    return merged


# ============================================================================
# DETECCIÓN DE CUE POINTS PRECISA
# ============================================================================

def detect_cue_points_precise(y: np.ndarray, sr: int, duration: float,
                                sections: List[Dict], 
                                beat_times: np.ndarray,
                                bpm: float = None) -> List[Dict]:
    """
    Detecta cue points profesionales con timestamps beat-aligned.
    
    Usa onset strength curve para encontrar el momento EXACTO de cada
    transición, luego snap al beat más cercano.
    
    Returns:
        Lista de cue points ordenados por timestamp
    """
    if not sections or len(beat_times) < 2:
        return []
    
    beat_interval = float(np.median(np.diff(beat_times)))
    
    # Onset strength para detección precisa
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    
    # Energy curve
    energy, energy_times = compute_energy_curve(y, sr)
    
    cue_points = []
    
    # ==================== MIX IN ====================
    # Primer beat con energía suficiente para mezclar
    mix_in = _find_mix_in(energy, energy_times, beat_times, duration)
    if mix_in is not None:
        cue_points.append({
            'timestamp': round(mix_in, 2),
            'type': 'mix_in',
            'name': 'Mix In',
            'energy': _energy_at_time(energy, energy_times, mix_in),
            'confidence': 0.85,
        })
    
    # ==================== SECTION-BASED CUE POINTS ====================
    for i, section in enumerate(sections):
        s_type = section['type']
        s_start = section['start']
        s_end = section['end']
        
        if s_type == 'intro' and i == 0:
            # INTRO END: transición exacta de intro a siguiente sección
            intro_end = snap_to_bar(s_end, beat_times)
            # Refinar: buscar el onset peak más cercano al boundary
            refined = _refine_with_onset(intro_end, onset_env, onset_times, beat_times, window=4.0)
            cue_points.append({
                'timestamp': round(refined, 2),
                'type': 'intro_end',
                'name': 'Intro End',
                'energy': _energy_at_time(energy, energy_times, refined),
                'confidence': 0.80,
            })
        
        elif s_type == 'buildup':
            # BUILDUP START: inicio preciso del buildup
            build_start = snap_to_bar(s_start, beat_times)
            cue_points.append({
                'timestamp': round(build_start, 2),
                'type': 'buildup',
                'name': f'Buildup{f" {i}" if _count_type(sections, "buildup") > 1 else ""}',
                'energy': _energy_at_time(energy, energy_times, build_start),
                'confidence': 0.80,
            })
        
        elif s_type == 'drop':
            # DROP: momento exacto del impacto
            # Buscar el onset peak más fuerte justo al inicio de la sección drop
            drop_onset = _find_drop_onset(s_start, onset_env, onset_times, beat_times, window=4.0)
            cue_points.append({
                'timestamp': round(drop_onset, 2),
                'type': 'drop',
                'name': f'Drop{f" {_count_type_before(sections, "drop", i) + 1}" if _count_type(sections, "drop") > 1 else ""}',
                'energy': _energy_at_time(energy, energy_times, drop_onset),
                'confidence': 0.90,
            })
        
        elif s_type == 'breakdown':
            # BREAKDOWN START
            brk_start = snap_to_bar(s_start, beat_times)
            cue_points.append({
                'timestamp': round(brk_start, 2),
                'type': 'breakdown',
                'name': 'Breakdown',
                'energy': _energy_at_time(energy, energy_times, brk_start),
                'confidence': 0.75,
            })
            
            # BREAKDOWN END (si la siguiente sección es buildup o drop)
            if i + 1 < len(sections) and sections[i + 1]['type'] in ('buildup', 'drop'):
                brk_end = snap_to_bar(s_end, beat_times)
                refined = _refine_with_onset(brk_end, onset_env, onset_times, beat_times, window=4.0)
                cue_points.append({
                    'timestamp': round(refined, 2),
                    'type': 'breakdown_end',
                    'name': 'Breakdown End',
                    'energy': _energy_at_time(energy, energy_times, refined),
                    'confidence': 0.75,
                })
        
        elif s_type == 'outro':
            # OUTRO START
            outro_start = snap_to_bar(s_start, beat_times)
            cue_points.append({
                'timestamp': round(outro_start, 2),
                'type': 'outro_start',
                'name': 'Outro Start',
                'energy': _energy_at_time(energy, energy_times, outro_start),
                'confidence': 0.75,
            })
    
    # ==================== MIX OUT ====================
    mix_out = _find_mix_out(energy, energy_times, beat_times, duration, sections)
    if mix_out is not None:
        cue_points.append({
            'timestamp': round(mix_out, 2),
            'type': 'mix_out',
            'name': 'Mix Out',
            'energy': _energy_at_time(energy, energy_times, mix_out),
            'confidence': 0.85,
        })
    
    # Ordenar y eliminar duplicados cercanos
    cue_points.sort(key=lambda cp: cp['timestamp'])
    cue_points = _deduplicate_cues(cue_points, min_gap=2.0)
    
    return cue_points


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def _find_mix_in(energy: np.ndarray, energy_times: np.ndarray,
                  beat_times: np.ndarray, duration: float) -> Optional[float]:
    """
    Encuentra el punto óptimo de Mix In: primer bar donde la energía
    es suficiente para empezar a mezclar.
    """
    threshold = np.mean(energy) * 0.35
    
    # Buscar en los primeros 60 segundos
    for i, t in enumerate(energy_times):
        if t > 60:
            break
        if energy[i] > threshold:
            return snap_to_bar(t, beat_times)
    
    # Default: primer bar
    return snap_to_bar(2.0, beat_times) if len(beat_times) > 0 else 0.0


def _find_mix_out(energy: np.ndarray, energy_times: np.ndarray,
                   beat_times: np.ndarray, duration: float,
                   sections: List[Dict]) -> Optional[float]:
    """
    Encuentra el punto óptimo de Mix Out: último bar con energía
    suficiente antes del outro.
    """
    # Si hay outro, el mix out es al inicio del outro
    outro_sections = [s for s in sections if s['type'] == 'outro']
    if outro_sections:
        outro_start = outro_sections[0]['start']
        return snap_to_bar(outro_start, beat_times)
    
    # Si no hay outro, buscar donde la energía empieza a caer definitivamente
    threshold = np.mean(energy) * 0.6
    mix_out_time = duration - 32  # Default: 32 segundos antes del final
    
    # Buscar desde el 70% del track hacia adelante
    start_search = int(len(energy) * 0.7)
    for i in range(len(energy) - 1, start_search, -1):
        if energy[i] > threshold:
            mix_out_time = energy_times[i]
            break
    
    return snap_to_bar(mix_out_time, beat_times)


def _find_drop_onset(section_start: float, onset_env: np.ndarray,
                      onset_times: np.ndarray, beat_times: np.ndarray,
                      window: float = 4.0) -> float:
    """
    Encuentra el onset más fuerte cerca del inicio de un drop.
    El drop real suele estar en el primer beat fuerte de la sección.
    """
    # Buscar onset peak en ventana alrededor del start
    mask = (onset_times >= section_start - window/2) & (onset_times <= section_start + window)
    
    if not np.any(mask):
        return snap_to_bar(section_start, beat_times)
    
    windowed_onset = onset_env[mask]
    windowed_times = onset_times[mask]
    
    # El onset más fuerte en esa ventana
    peak_idx = np.argmax(windowed_onset)
    raw_time = windowed_times[peak_idx]
    
    # Snap al beat más cercano
    return snap_to_beat(raw_time, beat_times)


def _refine_with_onset(timestamp: float, onset_env: np.ndarray,
                        onset_times: np.ndarray, beat_times: np.ndarray,
                        window: float = 4.0) -> float:
    """
    Refina un timestamp usando el onset peak más cercano, snap a beat.
    """
    mask = (onset_times >= timestamp - window) & (onset_times <= timestamp + window)
    
    if not np.any(mask):
        return snap_to_beat(timestamp, beat_times)
    
    windowed = onset_env[mask]
    windowed_times = onset_times[mask]
    
    # Buscar peaks en la ventana
    if len(windowed) > 3:
        peaks, _ = find_peaks(windowed, prominence=np.std(windowed) * 0.5)
        if len(peaks) > 0:
            # El peak más cercano al timestamp original
            peak_times = windowed_times[peaks]
            closest_idx = np.argmin(np.abs(peak_times - timestamp))
            return snap_to_beat(peak_times[closest_idx], beat_times)
    
    return snap_to_beat(timestamp, beat_times)


def _energy_at_time(energy: np.ndarray, energy_times: np.ndarray, t: float) -> float:
    """Obtiene energía en un timestamp dado."""
    if len(energy_times) == 0:
        return 0.5
    idx = np.argmin(np.abs(energy_times - t))
    return round(float(energy[idx]), 3)


def _count_type(sections: List[Dict], section_type: str) -> int:
    """Cuenta cuántas secciones de un tipo hay."""
    return sum(1 for s in sections if s['type'] == section_type)


def _count_type_before(sections: List[Dict], section_type: str, index: int) -> int:
    """Cuenta secciones de un tipo antes del índice dado."""
    return sum(1 for s in sections[:index] if s['type'] == section_type)


def _deduplicate_cues(cue_points: List[Dict], min_gap: float = 2.0) -> List[Dict]:
    """Elimina cue points demasiado cercanos, manteniendo el de mayor confianza."""
    if len(cue_points) <= 1:
        return cue_points
    
    result = [cue_points[0]]
    for cp in cue_points[1:]:
        prev = result[-1]
        if cp['timestamp'] - prev['timestamp'] < min_gap:
            # Mantener el de mayor confianza
            if cp.get('confidence', 0) > prev.get('confidence', 0):
                result[-1] = cp
        else:
            result.append(cp)
    
    return result


# ============================================================================
# FUNCIÓN PRINCIPAL (drop-in replacement)
# ============================================================================

def analyze_structure_and_cues(y: np.ndarray, sr: int, duration: float,
                                bpm: float = None) -> Dict:
    """
    Análisis completo de estructura + cue points con precisión profesional.
    
    Drop-in replacement para:
    - detect_structure() en main.py
    - detect_cue_points() en artwork_and_cuepoints.py
    - detect_beat_grid() en artwork_and_cuepoints.py
    
    Returns:
        {
            'structure': {sections, has_intro, ...},
            'cue_points': [...],
            'beat_grid': {first_beat, beat_interval, beats, total_beats},
        }
    """
    if not LIBS_AVAILABLE:
        return _fallback_result(duration)
    
    try:
        print(f"  🎯 Análisis preciso v3: {duration:.1f}s, BPM={bpm}")
        
        # 1. Beat grid
        beat_times, first_beat, beat_interval = get_beat_times(y, sr, bpm)
        print(f"    ✓ Beat grid: {len(beat_times)} beats, interval={beat_interval:.4f}s")
        
        # 2. Estructura precisa
        structure = detect_structure_precise(y, sr, bpm, beat_times)
        sections = structure['sections']
        print(f"    ✓ Estructura: {len(sections)} secciones")
        for s in sections:
            print(f"      {s['type']:>10} | {s['start']:6.1f}s - {s['end']:6.1f}s | E={s['energy']:.2f}")
        
        # 3. Cue points beat-aligned
        cue_points = detect_cue_points_precise(y, sr, duration, sections, beat_times, bpm)
        print(f"    ✓ Cue points: {len(cue_points)}")
        for cp in cue_points:
            print(f"      {cp['type']:>15} | {cp['timestamp']:6.1f}s | conf={cp['confidence']:.2f}")
        
        # 4. Beat grid para export
        max_beats = 500
        beat_grid = {
            'first_beat': round(first_beat, 4),
            'beat_interval': round(beat_interval, 4),
            'beats': [round(b, 4) for b in beat_times[:max_beats]],
            'total_beats': len(beat_times),
        }
        
        return {
            'structure': structure,
            'cue_points': cue_points,
            'beat_grid': beat_grid,
        }
    
    except Exception as e:
        print(f"  ⚠️ Error en análisis preciso: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_result(duration)


def _fallback_result(duration: float) -> Dict:
    """Resultado por defecto si el análisis falla."""
    return {
        'structure': {
            'has_intro': False, 'has_buildup': False, 'has_drop': False,
            'has_breakdown': False, 'has_outro': False, 'sections': [],
            'source': 'fallback',
        },
        'cue_points': [],
        'beat_grid': {
            'first_beat': 0.0,
            'beat_interval': 0.5,
            'beats': [],
            'total_beats': 0,
        },
    }
