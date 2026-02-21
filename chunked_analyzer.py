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

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import gc

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("❌ Librosa no disponible")


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
        
        print(f"✓ ChunkedAudioAnalyzer inicializado")
        print(f"  Chunk: {chunk_duration}s | Overlap: {chunk_overlap}s | SR: {sample_rate}Hz")
    
    def get_audio_duration(self, file_path: str) -> float:
        """Obtiene la duración sin cargar el audio completo."""
        return librosa.get_duration(path=file_path)
    
    def load_chunk(self, file_path: str, start_time: float, duration: float) -> Tuple[np.ndarray, int]:
        """
        Carga solo un segmento del audio.
        
        Args:
            file_path: Ruta al archivo
            start_time: Tiempo de inicio en segundos
            duration: Duración del chunk en segundos
            
        Returns:
            Tuple de (audio_array, sample_rate)
        """
        y, sr = librosa.load(
            file_path, 
            sr=self.sr, 
            mono=True,
            offset=start_time,
            duration=duration
        )
        return y, sr
    
    def analyze_chunk_bpm(self, y: np.ndarray, sr: int) -> Dict:
        """Analiza BPM de un chunk."""
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
            
            # Calcular confianza basada en regularidad de beats
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                intervals = np.diff(beat_times)
                confidence = max(0, min(1, 1.0 - np.std(intervals) * 2))
            else:
                confidence = 0.3
            
            return {
                'bpm': tempo,
                'confidence': confidence,
                'beat_count': len(beats)
            }
        except Exception as e:
            print(f"  ⚠️ Error BPM chunk: {e}")
            return {'bpm': 120.0, 'confidence': 0.0, 'beat_count': 0}
    
    def analyze_chunk_key(self, y: np.ndarray, sr: int) -> Dict:
        """Analiza key/tonalidad de un chunk."""
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
            print(f"  ⚠️ Error Key chunk: {e}")
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
        except Exception as e:
            print(f"  ⚠️ Error Energy chunk: {e}")
            return {'energy_mean': 0.1, 'energy_curve': []}
    
    def analyze_chunk_spectral(self, y: np.ndarray, sr: int) -> Dict:
        """Analiza características espectrales de un chunk."""
        try:
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Onset strength para densidad de percusión
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            return {
                'centroid_mean': float(np.mean(centroid)),
                'centroid_std': float(np.std(centroid)),
                'rolloff_mean': float(np.mean(rolloff)),
                'rolloff_std': float(np.std(rolloff)),
                'onset_strength_mean': float(np.mean(onset_env)),
                'has_heavy_bass': float(np.mean(centroid)) < 2500,
                'has_pads': float(np.std(rolloff)) < 1500
            }
        except Exception as e:
            print(f"  ⚠️ Error Spectral chunk: {e}")
            return {'centroid_mean': 3000, 'has_heavy_bass': False, 'has_pads': False}
    
    def fuse_bpm_results(self, chunk_results: List[Dict]) -> Dict:
        """
        Fusiona resultados de BPM de múltiples chunks usando votación ponderada.
        """
        if not chunk_results:
            return {'bpm': 120.0, 'confidence': 0.0}
        
        # Filtrar BPMs en rango válido para electrónica
        valid_results = [r for r in chunk_results if 60 <= r['bpm'] <= 200 and r['confidence'] > 0.1]
        
        if not valid_results:
            valid_results = chunk_results
        
        # Normalizar armónicos (doblar BPMs muy bajos, dividir muy altos)
        normalized = []
        for r in valid_results:
            bpm = r['bpm']
            if bpm < 70:
                bpm *= 2
            elif bpm > 170:
                bpm /= 2
            normalized.append({'bpm': bpm, 'confidence': r['confidence']})
        
        # Votación ponderada por confianza
        total_weight = sum(r['confidence'] for r in normalized)
        if total_weight > 0:
            weighted_bpm = sum(r['bpm'] * r['confidence'] for r in normalized) / total_weight
        else:
            weighted_bpm = np.median([r['bpm'] for r in normalized])
        
        # Confianza final = promedio de confianzas
        avg_confidence = np.mean([r['confidence'] for r in normalized])
        
        return {
            'bpm': round(weighted_bpm, 1),
            'confidence': round(avg_confidence, 3),
            'source': 'chunked_analysis'
        }
    
    def fuse_key_results(self, chunk_results: List[Dict], energy_weights: List[float] = None) -> Dict:
        """
        Fusiona resultados de key de múltiples chunks.
        Pondera más los chunks con mayor energía (suelen tener key más clara).
        """
        if not chunk_results:
            return {'key': 'C', 'camelot': '8B', 'confidence': 0.0}
        
        # Si no hay pesos de energía, usar confianza
        if energy_weights is None:
            energy_weights = [r.get('confidence', 0.5) for r in chunk_results]
        
        # Normalizar pesos
        total_weight = sum(energy_weights) + 1e-10
        weights = [w / total_weight for w in energy_weights]
        
        # Contar votos ponderados por key
        key_votes = {}
        for r, w in zip(chunk_results, weights):
            key = r.get('key', 'C')
            if key not in key_votes:
                key_votes[key] = 0
            key_votes[key] += w * r.get('confidence', 0.5)
        
        # Key ganadora
        best_key = max(key_votes, key=key_votes.get)
        camelot = KEY_TO_CAMELOT.get(best_key, '8B')
        
        # Confianza = voto ganador / total votos
        total_votes = sum(key_votes.values()) + 1e-10
        confidence = key_votes[best_key] / total_votes
        
        return {
            'key': best_key,
            'camelot': camelot,
            'scale': 'minor' if best_key.endswith('m') else 'major',
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
        
        print(f"    🎯 Cue points detectados: {len(cleaned)} (drops: {len(drops_found)}, breakdowns: {len(breakdowns_found)})")
        
        return cleaned
    
    def calculate_beat_grid(self, bpm: float, first_beat_offset: float = 0.0) -> Dict:
        """Calcula el beat grid basado en BPM."""
        beat_interval = 60.0 / bpm if bpm > 0 else 0.5
        return {
            'first_beat': first_beat_offset,
            'beat_interval': round(beat_interval, 6),
            'bpm': bpm
        }
    
    def full_analysis(self, file_path: str) -> Dict:
        """
        Análisis completo por chunks.
        
        Returns:
            Dict con todos los resultados del análisis
        """
        print(f"🔬 Análisis chunked: {file_path}")
        
        # Obtener duración sin cargar audio
        duration = self.get_audio_duration(file_path)
        print(f"  ⏱️ Duración: {duration:.1f}s ({duration/60:.1f} min)")
        
        # Calcular chunks necesarios
        chunk_starts = []
        current_start = 0
        
        while current_start < duration:
            chunk_starts.append(current_start)
            current_start += self.chunk_duration - self.chunk_overlap
        
        num_chunks = len(chunk_starts)
        print(f"  📦 Procesando {num_chunks} chunks de {self.chunk_duration}s")
        
        # Resultados por chunk
        bpm_results = []
        key_results = []
        energy_results = []
        spectral_results = []
        
        # Procesar cada chunk
        for i, start_time in enumerate(chunk_starts):
            chunk_duration = min(self.chunk_duration, duration - start_time)
            
            print(f"  📊 Chunk {i+1}/{num_chunks}: {start_time:.0f}s - {start_time + chunk_duration:.0f}s")
            
            # Cargar chunk
            y, sr = self.load_chunk(file_path, start_time, chunk_duration)
            
            # Analizar chunk
            bpm_result = self.analyze_chunk_bpm(y, sr)
            key_result = self.analyze_chunk_key(y, sr)
            energy_result = self.analyze_chunk_energy(y, sr, start_time)
            spectral_result = self.analyze_chunk_spectral(y, sr)
            
            bpm_results.append(bpm_result)
            key_results.append(key_result)
            energy_results.append(energy_result)
            spectral_results.append(spectral_result)
            
            # ⚡ CRÍTICO: Liberar memoria del chunk
            del y
            gc.collect()
        
        # ==================== FUSIÓN DE RESULTADOS ====================
        
        print("  🔗 Fusionando resultados...")
        
        # BPM final
        bpm_final = self.fuse_bpm_results(bpm_results)
        
        # Key final (ponderado por energía de cada chunk)
        energy_weights = [r.get('energy_mean', 0.5) for r in energy_results]
        key_final = self.fuse_key_results(key_results, energy_weights)
        
        # Curva de energía completa
        energy_curve = self.build_energy_curve(energy_results)
        
        # Estructura del track
        structure = self.detect_structure_from_energy(energy_curve, duration)
        
        # Cue points automaticos deshabilitados - el usuario los pone a mano
        cue_points = []
        
        # Beat grid
        beat_grid = self.calculate_beat_grid(bpm_final['bpm'])
        
        # Energía DJ (1-10)
        energy_mean = np.mean([r.get('energy_mean', 0.1) for r in energy_results])
        energy_dj = self._calculate_energy_dj(energy_mean)
        
        # Características espectrales agregadas
        has_heavy_bass = sum(1 for r in spectral_results if r.get('has_heavy_bass')) > len(spectral_results) / 2
        has_pads = sum(1 for r in spectral_results if r.get('has_pads')) > len(spectral_results) / 2
        percussion_density = np.mean([r.get('onset_strength_mean', 0.5) for r in spectral_results]) / 10
        
        # Energía inicio/fin para mix
        if energy_results:
            mix_energy_start = energy_results[0].get('energy_mean', 0.5)
            mix_energy_end = energy_results[-1].get('energy_mean', 0.5)
        else:
            mix_energy_start = mix_energy_end = 0.5
        
        # Track type
        track_type = self._classify_track_type(energy_dj / 10, structure, duration)
        
        print(f"  ✅ BPM: {bpm_final['bpm']} | Key: {key_final['key']}/{key_final['camelot']} | Energy: {energy_dj}/10")
        
        return {
            'duration': duration,
            'bpm': bpm_final['bpm'],
            'bpm_confidence': bpm_final['confidence'],
            'bpm_source': 'chunked_analysis',
            'key': key_final['key'],
            'camelot': key_final['camelot'],
            'key_confidence': key_final['confidence'],
            'key_source': 'chunked_analysis',
            'energy_raw': energy_mean,
            'energy_normalized': energy_dj / 10,
            'energy_dj': energy_dj,
            'mix_energy_start': mix_energy_start,
            'mix_energy_end': mix_energy_end,
            'groove_score': 0.5,  # TODO: Calcular desde beat intervals
            'swing_factor': 0.5,
            'has_intro': structure['has_intro'],
            'has_buildup': structure['has_buildup'],
            'has_drop': structure['has_drop'],
            'has_breakdown': structure['has_breakdown'],
            'has_outro': structure['has_outro'],
            'structure_sections': structure['sections'],
            'drop_timestamp': structure['drop_timestamp'],
            'track_type': track_type,
            'has_vocals': False,  # Desactivado (muchos falsos positivos)
            'has_heavy_bass': has_heavy_bass,
            'has_pads': has_pads,
            'percussion_density': min(percussion_density, 1.0),
            'cue_points': cue_points,
            'first_beat': beat_grid['first_beat'],
            'beat_interval': beat_grid['beat_interval'],
            'analyzer': 'chunked_librosa'
        }
    
    def _calculate_energy_dj(self, energy_raw: float) -> int:
        """Convierte energía raw a escala DJ 1-10."""
        if energy_raw < 0.05:
            return 1
        elif energy_raw < 0.08:
            return 2
        elif energy_raw < 0.10:
            return 3
        elif energy_raw < 0.14:
            return 4
        elif energy_raw < 0.18:
            return 5
        elif energy_raw < 0.22:
            return 6
        elif energy_raw < 0.25:
            return 7
        elif energy_raw < 0.30:
            return 8
        elif energy_raw < 0.35:
            return 9
        else:
            return 10
    
    def _classify_track_type(self, energy_normalized: float, structure: Dict, duration: float) -> str:
        """Clasifica el tipo de track para DJs."""
        if energy_normalized < 0.5 and structure['has_intro']:
            return "warmup"
        if energy_normalized > 0.7 and structure['has_drop']:
            return "peak"
        if structure['has_outro'] and duration > 300:
            return "closing"
        return "peak" if energy_normalized > 0.6 else "warmup"


def get_chunked_analyzer(chunk_duration: int = 60) -> ChunkedAudioAnalyzer:
    """Factory function para obtener el analizador."""
    return ChunkedAudioAnalyzer(chunk_duration=chunk_duration)


# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Chunked Audio Analyzer - Test")
    print("="*60)
    
    analyzer = get_chunked_analyzer()
    print(f"✓ Analizador listo")
    print(f"  Chunk duration: {analyzer.chunk_duration}s")
    print(f"  RAM estimada por chunk: ~{analyzer.chunk_duration * 44100 * 4 / 1024 / 1024:.0f} MB")