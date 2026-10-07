"""
Chunked Audio Analyzer para DJ Analyzer Pro
============================================

Analiza tracks largos por segmentos para reducir uso de RAM.
En lugar de cargar 8 minutos de audio (~400MB RAM), procesa
chunks de 60 segundos (~50MB RAM) y fusiona los resultados.

Beneficios:
- RAM máxima por track: ~50-80 MB (vs ~400 MB)
- Permite 5-8 análisis simultáneos en 2GB RAM
- Resultados equivalentes al análisis completo
- Cue points y estructura precisos

v1.0.0 - Implementación inicial
"""

import logging
import math
import os
import subprocess
import numpy as np
from typing import Dict, List, Optional, Tuple

# La rejilla (fase + downbeat) vive aparte y la comparten los dos caminos de
# analisis. El cliente lleva el MISMO algoritmo en beat_grid_detector.dart: si
# tocas uno, toca el otro.
from beat_grid import onset_envelope as beat_grid_onset_envelope
# Lo que este camino tiene que calcular IGUAL que el corto (`main.analyze_audio`):
# el BPM y su doble/mitad, la rejilla, el groove, el tipo de track, los graves,
# los pads, la percusion y el genero espectral. Ver el docstring del modulo.
from tonalidad import croma_de_trozos
from rasgos_del_tema import (HOP, pulso_de_beats, rasgos_espectrales,
                             rejilla_y_bpm, tempo_y_beats, tempograma,
                             try_bpm_double_half, energia_del_tema,
                             nivel_de_energia)
import warnings
import gc

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa no disponible")

try:
    from audio_helpers import silence_native_stderr, beat_track_seguro
except ImportError:
    # Fallback: no-op si por lo que sea audio_helpers no esta disponible
    # (ej. cli scripts independientes).
    import contextlib
    @contextlib.contextmanager
    def silence_native_stderr():
        yield

    def beat_track_seguro(y, sr, **kwargs):
        # Sin `audio_helpers` no hay guarda, pero tampoco se cambia el
        # comportamiento: el `except` del llamador sigue siendo la red.
        return librosa.beat.beat_track(y=y, sr=sr, **kwargs)


# ==================== CONFIGURACIÓN ====================

# Tamaño de chunk en segundos (60s = ~50MB RAM a 44100Hz)
CHUNK_DURATION = 60

# Overlap entre chunks para no perder transiciones (en segundos)
CHUNK_OVERLAP = 5

# Sample rate estándar
SAMPLE_RATE = 44100

# Perfiles para detección de key
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Normalizar perfiles
MAJOR_PROFILE = MAJOR_PROFILE / np.sum(MAJOR_PROFILE)
MINOR_PROFILE = MINOR_PROFILE / np.sum(MINOR_PROFILE)

KEY_TO_CAMELOT = {
    'C': '8B', 'C#': '3B', 'Db': '3B', 'D': '10B', 'D#': '5B', 'Eb': '5B',
    'E': '12B', 'F': '7B', 'F#': '2B', 'Gb': '2B', 'G': '9B', 'G#': '4B', 
    'Ab': '4B', 'A': '11B', 'A#': '6B', 'Bb': '6B', 'B': '1B',
    'Cm': '5A', 'C#m': '12A', 'Dbm': '12A', 'Dm': '7A', 'D#m': '2A', 'Ebm': '2A',
    'Em': '9A', 'Fm': '4A', 'F#m': '11A', 'Gbm': '11A', 'Gm': '6A', 'G#m': '1A', 
    'Abm': '1A', 'Am': '8A', 'A#m': '3A', 'Bbm': '3A', 'Bm': '10A',
}


class ChunkedAudioAnalyzer:
    """
    Analizador de audio que procesa por chunks para reducir uso de RAM.
    """
    
    def __init__(self, chunk_duration: int = CHUNK_DURATION, 
                 chunk_overlap: int = CHUNK_OVERLAP,
                 sample_rate: int = SAMPLE_RATE):
        """
        Args:
            chunk_duration: Duración de cada chunk en segundos
            chunk_overlap: Overlap entre chunks en segundos
            sample_rate: Sample rate para análisis
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa no está instalado")
        
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.sr = sample_rate
        
        logger.info("ChunkedAudioAnalyzer inicializado")
        logger.info(f"  Chunk: {chunk_duration}s | Overlap: {chunk_overlap}s | SR: {sample_rate}Hz")
    
    def get_audio_duration(self, file_path: str) -> float:
        """Obtiene la duración sin cargar el audio completo."""
        with silence_native_stderr():
            return librosa.get_duration(path=file_path)

    def load_chunk(self, file_path: str, start_time: float, duration: float) -> Tuple[np.ndarray, int]:
        """
        Carga solo un segmento del audio.

        Args:
            file_path: Ruta al archivo
            start_time: Tiempo de inicio en segundos
            duration: Duración del chunk en segundos

        Returns:
            Tuple de (audio_array, sample_rate). Devuelve array vacio +
            self.sr si start_time supera el audio o duration<=0 — evita
            que soundfile.read pida frames negativos y explote con
            'negative dimensions are not allowed' (era el error #2 del
            panel admin).
        """
        # Guard contra dimensiones negativas: si start_time o duration
        # son raros (audio corrupto, chunking mal calculado, etc.) NO
        # delegamos a librosa porque acaba en np.empty(shape_neg).
        if duration is None or duration <= 0 or not math.isfinite(duration):
            logger.warning(
                f"[ChunkedAnalyzer] load_chunk skip — duration={duration} "
                f"start_time={start_time} file={file_path}"
            )
            return np.zeros(0, dtype=np.float32), self.sr
        if start_time is None or start_time < 0 or not math.isfinite(start_time):
            logger.warning(
                f"[ChunkedAnalyzer] load_chunk skip — start_time={start_time} "
                f"file={file_path}"
            )
            return np.zeros(0, dtype=np.float32), self.sr
        try:
            with silence_native_stderr():
                y, sr = librosa.load(
                    file_path,
                    sr=self.sr,
                    mono=True,
                    offset=start_time,
                    duration=duration
                )
            return y, sr
        except Exception as e:
            # librosa.load puede fallar por (a) 'negative dimensions' en headers
            # raros o (b) en Render, libsndfile sin mp3 + audioread sin backend
            # (LibsndfileError/NoBackendError). Intentamos un fallback via ffmpeg
            # extrayendo SOLO la ventana [start_time, start_time+duration]; si
            # tambien falla, devolvemos chunk vacio y el caller decide.
            logger.warning(
                f"[ChunkedAnalyzer] librosa.load fallo — {type(e).__name__}: {e} "
                f"start={start_time} dur={duration} file={file_path}; probando ffmpeg"
            )
            wav_path = f"{file_path}.chunk{int(start_time)}.wav"
            try:
                import soundfile as sf
                subprocess.run(
                    ['ffmpeg', '-v', 'error', '-y',
                     '-ss', str(start_time), '-t', str(duration),
                     '-i', file_path, '-ac', '1', '-ar', str(self.sr), wav_path],
                    capture_output=True, timeout=120, check=True,
                )
                y, sr = sf.read(wav_path, dtype='float32')
                if getattr(y, 'ndim', 1) > 1:
                    y = y.mean(axis=1)
                return y, sr
            except Exception as e2:
                logger.warning(
                    f"[ChunkedAnalyzer] ffmpeg fallback tambien fallo — "
                    f"{type(e2).__name__}: {e2} file={file_path}"
                )
                return np.zeros(0, dtype=np.float32), self.sr
            finally:
                if os.path.exists(wav_path):
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
    
    def analyze_chunk_key(self, y: np.ndarray, sr: int) -> Dict:
        """Analiza key/tonalidad de un chunk."""
        # Guard contra chunks vacios o demasiado cortos. chroma_cqt necesita
        # minimo ~2^7 muestras para 7 octavas; con <1 segundo suele fallar
        # con librosa.ParameterError. Pasa en tracks largos cuando el ultimo
        # chunk queda partido por el boundary.
        min_samples = sr  # 1 segundo como minimo razonable
        if y is None or len(y) < min_samples:
            logger.warning(
                f"Chunk key skip: len={len(y) if y is not None else 'None'} "
                f"< min={min_samples} (sr={sr})"
            )
            return {'key': 'C', 'scale': 'major', 'confidence': 0.0, 'chroma_vector': []}
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-10)
            
            best_key = None
            best_corr = -1
            best_scale = None
            
            for i, key_name in enumerate(KEY_NAMES):
                major_rot = np.roll(MAJOR_PROFILE, i)
                minor_rot = np.roll(MINOR_PROFILE, i)
                
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
            
            key_str = f"{best_key}m" if best_scale == 'minor' else best_key
            
            return {
                'key': key_str,
                'scale': best_scale,
                'confidence': max(0, min(1, best_corr)),
                'chroma_vector': chroma_mean.tolist()
            }
        except Exception as e:
            # Capturamos generico porque librosa levanta ParameterError
            # (subclase de Exception) y otras excepciones propias que no
            # estaban cubiertas por la tripleta original.
            logger.warning(f"Error Key chunk: {type(e).__name__}: {e}")
            return {'key': 'C', 'scale': 'major', 'confidence': 0.0, 'chroma_vector': []}
    
    def analyze_chunk_energy(self, y: np.ndarray, sr: int, chunk_start: float) -> Dict:
        """
        Analiza energía de un chunk y devuelve curva de energía.
        
        Args:
            y: Audio del chunk
            sr: Sample rate
            chunk_start: Tiempo de inicio del chunk (para timestamps absolutos)
        """
        try:
            # RMS en ventanas de ~1 segundo
            hop_length = sr  # 1 segundo
            frame_length = sr * 2  # 2 segundos de ventana
            
            rms = librosa.feature.rms(y=y, frame_length=min(frame_length, len(y)), 
                                       hop_length=min(hop_length, len(y)//4 + 1))[0]
            
            # Crear curva de energía con timestamps absolutos
            energy_curve = []
            time_per_frame = len(y) / sr / len(rms)
            
            for i, e in enumerate(rms):
                energy_curve.append({
                    'time': chunk_start + i * time_per_frame,
                    'energy': float(e)
                })
            
            return {
                'energy_mean': float(np.mean(rms)),
                'energy_max': float(np.max(rms)),
                'energy_min': float(np.min(rms)),
                'energy_std': float(np.std(rms)),
                'energy_curve': energy_curve
            }
        except Exception as e:  # generico: ver analyze_chunk_key (ParameterError)
            logger.warning(f"Error Energy chunk: {type(e).__name__}: {e}")
            return {'energy_mean': 0.1, 'energy_curve': []}
    
    def fuse_key_results(self, chunk_results: List[Dict], energy_weights: List[float] = None) -> Dict:
        """
        Fusiona resultados de key de múltiples chunks.
        Pondera más los chunks con mayor energía (suelen tener key más clara).
        """
        if not chunk_results:
            return {'key': 'C', 'camelot': '8B', 'confidence': 0.0}

        # Si no hay pesos de energía, usar confianza.
        #
        # `r.get('confidence') or 0.5`, NO `r.get('confidence', 0.5)`: el
        # segundo solo aplica el default si la CLAVE no existe, no si su valor
        # es None — y los chunks degenerados traen `'confidence': None` literal.
        # Con el default de dict, `sum(energy_weights)` unas lineas mas abajo
        # petaba con "TypeError: unsupported operand type(s) for +: 'int' and
        # 'NoneType'" y se llevaba por delante el analisis del track entero.
        #
        # Es EL MISMO fallo que ya se corrigio para 'key' en el bucle de votos
        # de mas abajo (panel admin 2026-05-20); alli se arreglo y aqui se
        # quedo a medias. Lo destapo test_chunked_analyzer.py.
        if energy_weights is None:
            energy_weights = [(r.get('confidence') or 0.5) for r in chunk_results]

        # Normalizar pesos
        total_weight = sum(energy_weights) + 1e-10
        weights = [w / total_weight for w in energy_weights]

        # Contar votos ponderados por key. Usar `r.get('key') or 'C'`
        # en vez de `r.get('key', 'C')`: el segundo solo aplica default
        # si la KEY no existe, NO si su valor es None — y en chunks
        # degenerados algunos r tienen literal `'key': None`, lo que
        # provocaba que best_key acabara siendo None y la linea
        # `best_key.endswith('m')` petara con AttributeError
        # (visto en panel admin 2026-05-20).
        key_votes = {}
        for r, w in zip(chunk_results, weights):
            key = r.get('key') or 'C'
            if key not in key_votes:
                key_votes[key] = 0
            key_votes[key] += w * (r.get('confidence') or 0.5)

        # Defensa contra el caso edge de 0 chunks (key_votes vacio
        # rompe max() con ValueError).
        if not key_votes:
            return {
                'key': 'C', 'camelot': '8B', 'scale': 'major',
                'confidence': 0.0, 'source': 'chunked_analysis',
            }

        # Key ganadora
        best_key = max(key_votes, key=key_votes.get)
        camelot = KEY_TO_CAMELOT.get(best_key, '8B')

        # Confianza = voto ganador / total votos
        total_votes = sum(key_votes.values()) + 1e-10
        confidence = key_votes[best_key] / total_votes

        # Defensive: best_key garantizado string aqui (vino del dict de
        # votos, que solo contiene strings tras el `or 'C'` de arriba),
        # pero protegemos el endswith con cast por si llegan claves no
        # string en el futuro.
        scale = 'minor' if str(best_key).endswith('m') else 'major'

        return {
            'key': best_key,
            'camelot': camelot,
            'scale': scale,
            'confidence': round(confidence, 3),
            'source': 'chunked_analysis'
        }
    
    def build_energy_curve(self, chunk_results: List[Dict]) -> List[Dict]:
        """Combina las curvas de energía de todos los chunks."""
        full_curve = []
        for r in chunk_results:
            # La curva viene directamente en 'energy_curve', no anidada
            curve = r.get('energy_curve', [])
            full_curve.extend(curve)
        
        # Ordenar por tiempo
        full_curve.sort(key=lambda x: x['time'])
        return full_curve
    
    def detect_structure_from_energy(self, energy_curve: List[Dict], duration: float) -> Dict:
        """
        Detecta estructura del track (intro, drop, breakdown, outro) 
        a partir de la curva de energia completa.
        
        Mejorado para detectar mejor los drops en tracks con energia constante.
        """
        if not energy_curve:
            return {
                'has_intro': False,
                'has_buildup': False,
                'has_drop': False,
                'has_breakdown': False,
                'has_outro': False,
                'sections': [],
                'drop_timestamp': duration / 3
            }
        
        # Extraer valores de energia
        energies = np.array([p['energy'] for p in energy_curve])
        times = np.array([p['time'] for p in energy_curve])
        
        # Filtrar NaN
        valid_mask = ~np.isnan(energies)
        if not np.any(valid_mask):
            return {
                'has_intro': False,
                'has_buildup': False,
                'has_drop': False,
                'has_breakdown': False,
                'has_outro': False,
                'sections': [],
                'drop_timestamp': duration / 3
            }
        
        energies = energies[valid_mask]
        times = times[valid_mask]
        
        avg_energy = np.mean(energies)
        max_energy = np.max(energies)
        min_energy = np.min(energies)
        energy_range = max_energy - min_energy
        
        # Umbrales adaptativos basados en el rango de energia del track
        # Para tracks con poca variacion, usamos umbrales mas sensibles
        if energy_range < avg_energy * 0.3:
            # Track con energia muy constante
            drop_threshold = avg_energy * 1.15
            breakdown_threshold = avg_energy * 0.85
            buildup_factor = 1.08
        elif energy_range < avg_energy * 0.5:
            # Track con variacion moderada
            drop_threshold = avg_energy * 1.25
            breakdown_threshold = avg_energy * 0.75
            buildup_factor = 1.12
        else:
            # Track con mucha variacion (tipico EDM)
            drop_threshold = avg_energy * 1.35
            breakdown_threshold = avg_energy * 0.65
            buildup_factor = 1.15
        
        # Dividir en secciones de ~8 segundos para analisis de estructura
        section_duration = 8.0
        num_sections = max(1, int(duration / section_duration))
        
        sections = []
        section_energies = []
        
        for i in range(num_sections):
            start_time = i * section_duration
            end_time = min((i + 1) * section_duration, duration)
            
            # Encontrar puntos de energia en este rango
            mask = (times >= start_time) & (times < end_time)
            section_e = energies[mask]
            
            if len(section_e) > 0:
                mean_e = float(np.mean(section_e))
            else:
                mean_e = avg_energy
            
            section_energies.append(mean_e)
        
        section_energies = np.array(section_energies)
        
        # Detectar caracteristicas estructurales
        has_intro = section_energies[0] < breakdown_threshold if len(section_energies) > 0 else False
        has_outro = section_energies[-1] < breakdown_threshold if len(section_energies) > 0 else False
        
        # Buscar el pico maximo de energia (drop principal)
        max_idx = np.argmax(section_energies)
        has_drop = section_energies[max_idx] > drop_threshold
        drop_time = max_idx * section_duration + 4.0 if has_drop else duration / 3
        
        has_buildup = max_idx > 1 and section_energies[max_idx-1] > section_energies[0] * buildup_factor
        has_breakdown = max_idx < len(section_energies) - 2 and np.min(section_energies[max_idx+1:]) < breakdown_threshold
        
        # Crear lista de secciones con mejor clasificacion
        for i, e in enumerate(section_energies):
            start = i * section_duration
            end = min((i + 1) * section_duration, duration)
            
            # Clasificar seccion
            if e > drop_threshold:
                section_type = 'drop'
            elif e < breakdown_threshold:
                if i < 2:
                    section_type = 'intro'
                elif i > len(section_energies) - 3:
                    section_type = 'outro'
                else:
                    section_type = 'breakdown'
            elif i > 0 and e > section_energies[i-1] * buildup_factor:
                section_type = 'buildup'
            else:
                section_type = 'main'
            
            sections.append({
                'type': section_type,
                'start': round(start, 2),
                'end': round(end, 2),
                'energy': round(e, 4) if not np.isnan(e) else 0.0
            })
        
        return {
            'has_intro': has_intro,
            'has_buildup': has_buildup,
            'has_drop': has_drop,
            'has_breakdown': has_breakdown,
            'has_outro': has_outro,
            'sections': sections,
            'drop_timestamp': round(drop_time, 2)
        }
    
    def detect_cue_points_from_structure(self, structure: Dict, duration: float, bpm: float) -> List[Dict]:
        """
        Genera cue points de ALTA CALIDAD basados en la estructura detectada.
        
        Principios:
        - Calidad sobre cantidad
        - Siempre mix_in y mix_out
        - Solo puntos realmente utiles para DJ
        - Tracks lineales = menos cue points
        """
        sections = structure.get('sections', [])
        
        # Calcular alineacion a barras
        beat_duration = 60.0 / bpm if bpm > 0 else 0.5
        bar_duration = beat_duration * 4
        
        def snap_to_bar(time_sec):
            """Alinea el tiempo al inicio de la barra mas cercana"""
            return round(time_sec / bar_duration) * bar_duration
        
        # Extraer energias de las secciones
        section_energies = [s.get('energy', 0) for s in sections]
        if not section_energies:
            # Track sin analisis - solo mix_in y mix_out basicos
            return [
                {'index': 0, 'time': round(snap_to_bar(duration * 0.05), 2), 'type': 'mix_in', 'label': 'Mix In'},
                {'index': 1, 'time': round(snap_to_bar(duration * 0.85), 2), 'type': 'mix_out', 'label': 'Mix Out'}
            ]
        
        # Filtrar NaN
        valid_energies = [e for e in section_energies if e is not None and not np.isnan(e)]
        if not valid_energies:
            return [
                {'index': 0, 'time': round(snap_to_bar(duration * 0.05), 2), 'type': 'mix_in', 'label': 'Mix In'},
                {'index': 1, 'time': round(snap_to_bar(duration * 0.85), 2), 'type': 'mix_out', 'label': 'Mix Out'}
            ]
        
        avg_energy = np.mean(valid_energies)
        max_energy = np.max(valid_energies)
        min_energy = np.min(valid_energies)
        energy_range = max_energy - min_energy
        
        # ==================== DETECTAR PUNTOS CLAVE ====================
        
        cue_points = []
        
        # 1. MIX IN - despues de la intro o al 5-10% del track
        intro_end = None
        for i, section in enumerate(sections):
            if section.get('type') == 'intro':
                intro_end = section.get('end', 0)
            elif intro_end is None and section.get('energy', 0) > avg_energy * 0.8:
                # Primera seccion con energia "normal" = fin de intro
                intro_end = section.get('start', 0)
                break
        
        if intro_end and intro_end > duration * 0.03:
            mix_in_time = snap_to_bar(intro_end)
        else:
            # Sin intro clara, usar ~32 barras desde el inicio
            mix_in_time = snap_to_bar(min(bar_duration * 16, duration * 0.08))
        
        cue_points.append({
            'time': round(mix_in_time, 2),
            'type': 'mix_in',
            'label': 'Mix In'
        })
        
        # 2. DROPS - buscar picos SIGNIFICATIVOS de energia
        # Un drop real es cuando la energia sube mucho respecto a la seccion anterior
        drops_found = []
        
        for i in range(1, len(sections)):
            current = sections[i]
            prev = sections[i-1]
            
            current_energy = current.get('energy', 0)
            prev_energy = prev.get('energy', 0)
            
            # Validar que no sean NaN
            if current_energy is None or prev_energy is None:
                continue
            if np.isnan(current_energy) or np.isnan(prev_energy):
                continue
            
            # Drop = energia alta + salto significativo desde seccion anterior
            is_high_energy = current_energy > avg_energy * 1.1
            is_significant_jump = current_energy > prev_energy * 1.3
            prev_was_low = prev_energy < avg_energy * 0.85
            
            if is_high_energy and (is_significant_jump or prev_was_low):
                drop_time = current.get('start', 0)
                # Evitar drops muy cercanos (minimo 20 segundos entre drops)
                if not drops_found or (drop_time - drops_found[-1]) > 20:
                    drops_found.append(drop_time)
        
        # Añadir solo los drops mas importantes (max 2-3)
        for i, drop_time in enumerate(drops_found[:3]):
            label = 'Drop' if i == 0 else f'Drop {i+1}'
            cue_points.append({
                'time': round(snap_to_bar(drop_time), 2),
                'type': 'drop',
                'label': label
            })
        
        # 3. BREAKDOWNS - buscar caidas SIGNIFICATIVAS de energia
        breakdowns_found = []
        
        for i in range(1, len(sections)):
            current = sections[i]
            prev = sections[i-1]
            
            current_energy = current.get('energy', 0)
            prev_energy = prev.get('energy', 0)
            
            if current_energy is None or prev_energy is None:
                continue
            if np.isnan(current_energy) or np.isnan(prev_energy):
                continue
            
            # Breakdown = energia baja + caida significativa
            is_low_energy = current_energy < avg_energy * 0.7
            is_significant_drop = current_energy < prev_energy * 0.6
            prev_was_high = prev_energy > avg_energy
            
            # No contar como breakdown si es intro u outro
            section_position = current.get('start', 0) / duration
            is_middle = 0.15 < section_position < 0.85
            
            if is_low_energy and is_significant_drop and prev_was_high and is_middle:
                bd_time = current.get('start', 0)
                # Evitar breakdowns muy cercanos
                if not breakdowns_found or (bd_time - breakdowns_found[-1]) > 30:
                    breakdowns_found.append(bd_time)
        
        # Añadir solo los breakdowns mas importantes (max 2)
        for i, bd_time in enumerate(breakdowns_found[:2]):
            label = 'Breakdown' if i == 0 else f'Breakdown {i+1}'
            cue_points.append({
                'time': round(snap_to_bar(bd_time), 2),
                'type': 'breakdown',
                'label': label
            })
        
        # 4. BUILDUP - solo si hay un drop claro despues
        # Buscar la subida ANTES del drop principal
        if drops_found:
            main_drop_time = drops_found[0]
            # Buscar seccion de buildup en los 30 segundos antes del drop
            for section in sections:
                section_start = section.get('start', 0)
                section_end = section.get('end', 0)
                
                # Esta en la zona pre-drop?
                if (main_drop_time - 30) < section_start < (main_drop_time - 4):
                    section_energy = section.get('energy', 0)
                    if section_energy and not np.isnan(section_energy):
                        # Energia media-alta = buildup
                        if avg_energy * 0.7 < section_energy < avg_energy * 1.2:
                            cue_points.append({
                                'time': round(snap_to_bar(section_start), 2),
                                'type': 'buildup',
                                'label': 'Buildup'
                            })
                            break  # Solo un buildup principal
        
        # 5. MIX OUT - siempre, en el ultimo 15% del track o inicio del outro
        outro_start = None
        for section in reversed(sections):
            if section.get('type') == 'outro':
                outro_start = section.get('start', 0)
                break
            # O buscar caida final de energia
            section_energy = section.get('energy', 0)
            if section_energy and not np.isnan(section_energy):
                if section_energy < avg_energy * 0.6 and section.get('start', 0) > duration * 0.8:
                    outro_start = section.get('start', 0)
                    break
        
        if outro_start and outro_start > duration * 0.7:
            mix_out_time = snap_to_bar(outro_start)
        else:
            # Sin outro claro, usar 85% del track
            mix_out_time = snap_to_bar(duration * 0.85)
        
        # Asegurar que mix_out no sea muy cercano al final
        mix_out_time = min(mix_out_time, duration - bar_duration * 8)
        
        cue_points.append({
            'time': round(mix_out_time, 2),
            'type': 'mix_out',
            'label': 'Mix Out'
        })
        
        # ==================== ORDENAR Y LIMPIAR ====================
        
        # Ordenar por tiempo
        cue_points.sort(key=lambda x: x['time'])
        
        # Eliminar duplicados cercanos (menos de 8 segundos)
        cleaned = []
        for cue in cue_points:
            if not cleaned or (cue['time'] - cleaned[-1]['time']) > 8:
                cleaned.append(cue)
            else:
                # Si hay conflicto, preferir mix_in/mix_out/drop sobre buildup
                priority = {'mix_in': 5, 'mix_out': 5, 'drop': 4, 'breakdown': 3, 'buildup': 2}
                if priority.get(cue['type'], 1) > priority.get(cleaned[-1]['type'], 1):
                    cleaned[-1] = cue
        
        # Asegurar que mix_in y mix_out estan presentes
        has_mix_in = any(c['type'] == 'mix_in' for c in cleaned)
        has_mix_out = any(c['type'] == 'mix_out' for c in cleaned)
        
        if not has_mix_in:
            cleaned.insert(0, {
                'time': round(snap_to_bar(duration * 0.05), 2),
                'type': 'mix_in',
                'label': 'Mix In'
            })
        
        if not has_mix_out:
            cleaned.append({
                'time': round(snap_to_bar(duration * 0.85), 2),
                'type': 'mix_out',
                'label': 'Mix Out'
            })
        
        # Reindexar
        for i, cue in enumerate(cleaned):
            cue['index'] = i
        
        logger.info(f"Cue points detectados: {len(cleaned)} (drops: {len(drops_found)}, breakdowns: {len(breakdowns_found)})")
        
        return cleaned
    
    def _rasgos_del_trozo(self, y: np.ndarray, sr: int) -> Dict:
        """Lo de este trozo que hace falta para el tema entero, frame a frame
        (hop 512): las MISMAS llamadas que hace el camino corto con el tema
        entero, para coserlas y sacar las mismas cuentas.

        - `onset`: la envolvente de siempre (media), para la rejilla, el
          doble/mitad y la percusion.
        - `onset_med` y `tg`: la envolvente con mediana y su tempograma, que
          es con lo que `librosa.beat.beat_track` decide el tempo.
        - `centroid`, `rolloff`: genero y pads.
        - `bass`, `mid`, `treble`: las bandas del clasificador espectral.
        - `rms`: la energía, con la ventana de 46 ms del camino corto
          (`energia_del_tema`). La de `analyze_chunk_energy`, de 2 s, sigue
          para la estructura y para pesar la tonalidad de cada trozo.
        """
        from spectral_classifier import bandas_de_audio
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        onset_med = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        bandas = bandas_de_audio(y, sr)
        vacio = np.zeros(0, dtype=np.float32)
        return {
            'onset': onset,
            'onset_med': onset_med,
            'tg': tempograma(onset_med, sr),
            'centroid': librosa.feature.spectral_centroid(y=y, sr=sr)[0],
            'rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
            'bass': bandas[0] if bandas else vacio,
            'mid': bandas[1] if bandas else vacio,
            'treble': bandas[2] if bandas else vacio,
            'rms': librosa.feature.rms(y=y)[0],
        }

    def full_analysis(self, file_path: str,
                      bpm_etiqueta: Optional[float] = None) -> Dict:
        """
        Análisis completo por chunks.

        `bpm_etiqueta`: el BPM de las etiquetas del fichero. Manda como en el
        camino corto (si está entre 60 y 200): sin doble/mitad y sin afinarlo
        con la rejilla.

        Returns:
            Dict con todos los resultados del análisis
        """
        logger.info(f"Analisis chunked: {file_path}")
        
        # Obtener duración sin cargar audio
        duration = self.get_audio_duration(file_path)
        logger.info(f"Duracion: {duration:.1f}s ({duration/60:.1f} min)")
        
        # Trozos que empiezan en un frame EXACTO (multiplo de 512 muestras).
        #
        # Con 55 s de paso, cada trozo empezaba en 4737,3 frames: al coser las
        # envolventes por frames enteros se perdian 3,4 ms por trozo, que se
        # ACUMULABAN — ~20 ms a lo largo de un tema de 6 minutos, una
        # compresion del eje de tiempo que la rejilla leia como otro tempo
        # (128,01 en vez de 128,00) y que movia su fase. Con el paso en frames
        # enteros, el frame j de cada trozo es exactamente el frame
        # `inicio/512 + j` del tema.
        paso_muestras = int(round(
            (self.chunk_duration - self.chunk_overlap) * self.sr / HOP)) * HOP
        paso_frames = paso_muestras // HOP
        paso_s = paso_muestras / self.sr
        # Cada trozo aporta desde la MITAD de su solape con el anterior hasta
        # la mitad del solape con el siguiente: sus bordes (donde librosa
        # rellena y la envolvente se inventa un golpe) se quedan fuera.
        medio_solape = int(round(self.chunk_overlap * self.sr / HOP)) // 2

        chunk_starts = []
        while len(chunk_starts) * paso_s < duration:
            chunk_starts.append(len(chunk_starts) * paso_s)

        num_chunks = len(chunk_starts)
        logger.info(f"Procesando {num_chunks} chunks de {self.chunk_duration}s")
        
        # Resultados por chunk
        key_results = []
        energy_results = []
        cosido = {k: [] for k in ('onset', 'onset_med', 'centroid', 'rolloff',
                                  'bass', 'mid', 'treble', 'rms')}
        suma_tg = None
        frames_tg = 0
        sr = self.sr
        
        # Procesar cada chunk
        for i, start_time in enumerate(chunk_starts):
            chunk_duration = min(self.chunk_duration, duration - start_time)
            
            logger.debug(f"Chunk {i+1}/{num_chunks}: {start_time:.0f}s - {start_time + chunk_duration:.0f}s")
            
            # Cargar chunk
            y, sr = self.load_chunk(file_path, start_time, chunk_duration)

            # Si load_chunk devolvio array vacio (caso edge: start_time
            # mas alla del audio, duration<=0, o soundfile fallo), saltar
            # el chunk en lugar de pasarlo a los analyzers que petarian
            # con array de longitud 0.
            if y is None or len(y) == 0:
                logger.warning(
                    f"Chunk {i+1}/{num_chunks} vacio (start={start_time:.1f}s) — skip"
                )
                continue

            # Analizar chunk
            key_results.append(self.analyze_chunk_key(y, sr))
            energy_results.append(self.analyze_chunk_energy(y, sr, start_time))

            # Lo que se cose: frame a frame, sin el solape repetido.
            try:
                r = self._rasgos_del_trozo(y, sr)
                desde = 0 if i == 0 else medio_solape
                hasta = None if i == num_chunks - 1 else paso_frames + medio_solape
                for k in cosido:
                    cosido[k].append(np.asarray(r[k])[desde:hasta])
                tg = r['tg'][:, desde:hasta]
                suma_tg = tg.sum(axis=1) if suma_tg is None else suma_tg + tg.sum(axis=1)
                frames_tg += tg.shape[1]
                del r, tg
            except Exception as e:  # noqa: BLE001 - un trozo malo no tumba el tema
                logger.warning(f"[Chunk {i+1}] rasgos descartados: {type(e).__name__}: {e}")
            
            # ⚡ CRÍTICO: Liberar memoria del chunk
            del y
            gc.collect()
        
        # ==================== FUSIÓN DE RESULTADOS ====================
        
        logger.info("Fusionando resultados...")
        f = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.float32))
             for k, v in cosido.items()}
        fps = sr / HOP

        # BPM: como el camino corto. El tempo del tema entero (la media del
        # tempograma de todos los trozos), los beats sobre la envolvente cosida,
        # la confianza y el groove con ellos, el doble/mitad si la confianza es
        # baja, y despues la rejilla. Hasta el 2026-10-06 era la media de los
        # BPM de cada trozo, sin doble/mitad y con groove y swing fijos a 0,5.
        tempo, beats = tempo_y_beats(f['onset_med'], suma_tg if suma_tg is not None
                                     else np.zeros(1), frames_tg, sr)
        bpm_confidence, groove_score, swing_factor = pulso_de_beats(beats, sr)
        bpm = float(tempo)
        bpm_source = 'chunked_analysis' if bpm > 0 else ''
        if bpm_etiqueta and 60 < bpm_etiqueta < 200:
            bpm = float(bpm_etiqueta)
            bpm_source = 'id3'
        if bpm_source == 'chunked_analysis':
            bpm = try_bpm_double_half(None, sr, bpm, bpm_confidence, onset_env=f['onset'])
        first_beat, beat_interval, bpm = rejilla_y_bpm(
            f['onset'], fps, bpm, bpm_del_dsp=(bpm_source == 'chunked_analysis'))
        
        # Key final (ponderado por energía de cada chunk)
        energy_weights = [r.get('energy_mean', 0.5) for r in energy_results]
        key_final = self.fuse_key_results(key_results, energy_weights)
        
        # Curva de energía completa
        energy_curve = self.build_energy_curve(energy_results)
        
        # Estructura del track
        structure = self.detect_structure_from_energy(energy_curve, duration)
        
        # Cue points automaticos deshabilitados - el usuario los pone a mano
        cue_points = []
        
        # Energía: la del camino corto, sobre el RMS cosido frame a frame
        # (`energia_del_tema`). Hasta el 2026-10-07 era la media de RMS en
        # ventanas de 2 s, que con el mismo audio sale más alta: los temas
        # largos quedaban ~1 nivel por encima de los cortos, y con ellos su
        # tipo y su género. Sin nada cosido (todos los trozos fallaron), la
        # de las ventanas de 2 s, que es lo único que queda.
        if len(f['rms']):
            energy_mean, energy_dj, mix_energy_start, mix_energy_end = \
                energia_del_tema(f['rms'], sr)
        else:
            energy_mean = float(np.mean([r.get('energy_mean', 0.1) for r in energy_results])
                                if energy_results else 0.1)
            energy_dj = nivel_de_energia(energy_mean)
            if energy_results:
                mix_energy_start = energy_results[0].get('energy_mean', 0.5)
                mix_energy_end = energy_results[-1].get('energy_mean', 0.5)
            else:
                mix_energy_start = mix_energy_end = 0.5

        # Tipo, graves, pads, percusion y genero: las cuentas del camino corto
        # sobre lo cosido (`rasgos_espectrales`).
        rasgos = rasgos_espectrales(
            bpm=bpm, energy_normalized=energy_dj / 10, segments=structure,
            duration=duration, onset_env=f['onset'] if len(f['onset']) else np.zeros(1),
            spectral_centroid=f['centroid'] if len(f['centroid']) else np.zeros(1),
            rolloff=f['rolloff'] if len(f['rolloff']) else np.zeros(1),
            bandas=(f['bass'], f['mid'], f['treble']) if len(f['bass']) else None)

        logger.info(f"BPM: {bpm} | Key: {key_final['key']}/{key_final['camelot']} | Energy: {energy_dj}/10")
        
        return {
            'duration': duration,
            'bpm': bpm,
            'bpm_confidence': bpm_confidence,
            'bpm_source': bpm_source,
            'key': key_final['key'],
            'camelot': key_final['camelot'],
            'key_confidence': key_final['confidence'],
            'key_source': 'chunked_analysis',
            # El croma del tema entero, para medir otros perfiles de
            # tonalidad (`tonalidad.py`): la media del de cada trozo.
            'croma': croma_de_trozos(r.get('chroma_vector') for r in key_results),
            'energy_raw': energy_mean,
            'energy_normalized': energy_dj / 10,
            'energy_dj': energy_dj,
            'mix_energy_start': mix_energy_start,
            'mix_energy_end': mix_energy_end,
            'groove_score': groove_score,
            'swing_factor': swing_factor,
            'has_intro': structure['has_intro'],
            'has_buildup': structure['has_buildup'],
            'has_drop': structure['has_drop'],
            'has_breakdown': structure['has_breakdown'],
            'has_outro': structure['has_outro'],
            'structure_sections': structure['sections'],
            'drop_timestamp': structure['drop_timestamp'],
            'track_type': rasgos['track_type'],
            'track_type_confidence': rasgos['track_type_confidence'],
            'track_type_alternatives': rasgos['track_type_alternatives'],
            'genre': rasgos['genre'],
            # Las voces no se miden en ningun camino: el detector que habia
            # decia que si a casi cualquier tema con agudos (ver `main`).
            'has_vocals': False,
            'has_heavy_bass': rasgos['has_heavy_bass'],
            'has_pads': rasgos['has_pads'],
            'percussion_density': rasgos['percussion_density'],
            'cue_points': cue_points,
            'first_beat': first_beat,
            'beat_interval': round(beat_interval, 6),
            'analyzer': 'chunked_librosa'
        }


def get_chunked_analyzer(chunk_duration: int = 60) -> ChunkedAudioAnalyzer:
    """Factory function para obtener el analizador."""
    return ChunkedAudioAnalyzer(chunk_duration=chunk_duration)


# ==================== TEST ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Chunked Audio Analyzer - Test")
    logger.info("=" * 60)

    analyzer = get_chunked_analyzer()
    logger.info("Analizador listo")
    logger.info(f"  Chunk duration: {analyzer.chunk_duration}s")
    logger.info(f"  RAM estimada por chunk: ~{analyzer.chunk_duration * 44100 * 4 / 1024 / 1024:.0f} MB")