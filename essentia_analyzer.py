"""
Audio Analyzer Pro para DJ Analyzer Pro
========================================

v3.0.1 - Librosa optimizado (sin dependencias de Essentia)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings

# Importar Librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("Librosa cargado correctamente")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Librosa no disponible")

# Essentia no disponible en este sistema
ESSENTIA_AVAILABLE = False


# ==================== MAPEOS ====================

KEY_TO_CAMELOT = {
    'C': '8B', 'C#': '3B', 'Db': '3B', 'D': '10B', 'D#': '5B', 'Eb': '5B',
    'E': '12B', 'F': '7B', 'F#': '2B', 'Gb': '2B', 'G': '9B', 'G#': '4B', 
    'Ab': '4B', 'A': '11B', 'A#': '6B', 'Bb': '6B', 'B': '1B',
    'Cm': '5A', 'C#m': '12A', 'Dbm': '12A', 'Dm': '7A', 'D#m': '2A', 'Ebm': '2A',
    'Em': '9A', 'Fm': '4A', 'F#m': '11A', 'Gbm': '11A', 'Gm': '6A', 'G#m': '1A', 
    'Abm': '1A', 'Am': '8A', 'A#m': '3A', 'Bbm': '3A', 'Bm': '10A',
}

# Perfiles Krumhansl-Schmuckler (originales)
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Perfiles EDMA (Electronic Dance Music Analysis) - optimizados para electronica
EDMA_MAJOR = np.array([7.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 5.5, 2.0, 3.5, 2.0, 3.0])
EDMA_MINOR = np.array([7.0, 2.5, 3.5, 5.5, 2.0, 3.5, 2.5, 5.0, 4.0, 2.5, 3.5, 3.0])


class ImprovedLibrosaAnalyzer:
    """
    Analizador de audio mejorado usando Librosa con algoritmos optimizados.
    """
    
    def __init__(self, use_edma_profiles: bool = True):
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa no esta instalado")
        
        self.use_edma = use_edma_profiles
        self.major_profile = EDMA_MAJOR if use_edma_profiles else KRUMHANSL_MAJOR
        self.minor_profile = EDMA_MINOR if use_edma_profiles else KRUMHANSL_MINOR
        
        self.major_profile = self.major_profile / np.sum(self.major_profile)
        self.minor_profile = self.minor_profile / np.sum(self.minor_profile)
        
        print(f"ImprovedLibrosaAnalyzer inicializado (EDMA: {use_edma_profiles})")
    
    def load_audio(self, file_path: str, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        return y, sr
    
    def analyze_bpm_improved(self, y: np.ndarray, sr: int = 44100) -> Dict:
        """BPM con multiples tecnicas y voting."""
        try:
            # Metodo 1: Beat tracking estandar
            tempo_standard, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo_standard = float(tempo_standard)
            
            # Metodo 2: Onset + autocorrelacion
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_autocorr = librosa.feature.tempo(
                onset_envelope=onset_env, sr=sr, aggregate=None
            )
            
            # Metodo 3: PLP
            try:
                pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
                tempo_plp = float(60.0 * sr / (np.argmax(pulse) + 1)) if len(pulse) > 0 else tempo_standard
                if tempo_plp < 60 or tempo_plp > 200:
                    tempo_plp = tempo_standard
            except:
                tempo_plp = tempo_standard
            
            # Candidatos
            candidates = [tempo_standard]
            if isinstance(tempo_autocorr, np.ndarray) and len(tempo_autocorr) > 0:
                valid = [t for t in tempo_autocorr.flatten() if 60 <= t <= 200]
                candidates.extend(valid[:3])
            candidates.append(tempo_plp)
            
            # Filtrar y votar
            filtered = self._filter_harmonics(candidates)
            bpm = self._vote_tempo(filtered)
            
            # Confianza
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
            confidence = max(0.0, min(1.0, 1.0 - np.std(beat_intervals) * 2)) if len(beat_intervals) > 1 else 0.5
            
            # Groove
            groove_score = min(np.std(beat_intervals) * 10, 1.0) if len(beat_intervals) > 1 else 0.0
            swing_factor = 0.5
            if len(beat_intervals) > 2:
                even = beat_intervals[::2]
                odd = beat_intervals[1::2]
                if len(odd) > 0 and np.mean(odd) > 0:
                    swing_factor = float(np.mean(even) / np.mean(odd))
            
            return {
                'bpm': round(bpm, 1),
                'confidence': round(confidence, 3),
                'groove_score': round(groove_score, 3),
                'swing_factor': round(swing_factor, 3),
                'source': 'librosa_improved'
            }
        except Exception as e:
            print(f" Error BPM: {e}")
            return {'bpm': 120.0, 'confidence': 0.0, 'source': 'error'}
    
    def _filter_harmonics(self, candidates: List[float]) -> List[float]:
        filtered = []
        for tempo in candidates:
            if tempo < 60 or tempo > 200:
                continue
            adjusted = tempo
            if tempo < 70:
                adjusted = tempo * 2
            elif tempo > 170:
                adjusted = tempo / 2
            if 60 <= adjusted <= 200:
                filtered.append(adjusted)
        return filtered if filtered else [120.0]
    
    def _vote_tempo(self, candidates: List[float], tolerance: float = 2.0) -> float:
        if not candidates:
            return 120.0
        groups = []
        for tempo in sorted(candidates):
            added = False
            for group in groups:
                if abs(tempo - np.mean(group)) <= tolerance:
                    group.append(tempo)
                    added = True
                    break
            if not added:
                groups.append([tempo])
        largest = max(groups, key=len)
        return float(np.median(largest))
    
    def analyze_key_improved(self, y: np.ndarray, sr: int = 44100) -> Dict:
        """Key con Krumhansl-Schmuckler y perfiles EDMA."""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, n_octaves=7)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_mean = chroma_mean / np.sum(chroma_mean)
            
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            best_key, best_corr, best_scale = None, -1, None
            
            for i, key_name in enumerate(keys):
                major_rot = np.roll(self.major_profile, i)
                minor_rot = np.roll(self.minor_profile, i)
                
                major_corr = np.corrcoef(chroma_mean, major_rot)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_rot)[0, 1]
                
                if major_corr > best_corr:
                    best_corr, best_key, best_scale = major_corr, key_name, 'major'
                if minor_corr > best_corr:
                    best_corr, best_key, best_scale = minor_corr, key_name, 'minor'
            
            key_str = f"{best_key}m" if best_scale == 'minor' else best_key
            camelot = KEY_TO_CAMELOT.get(key_str, '?')
            
            return {
                'key': key_str,
                'camelot': camelot,
                'scale': best_scale,
                'confidence': round(max(0, min(1, best_corr)), 3),
                'source': 'librosa_improved'
            }
        except Exception as e:
            print(f" Error Key: {e}")
            return {'key': 'C', 'camelot': '8B', 'confidence': 0.0, 'source': 'error'}
    
    def analyze_energy(self, y: np.ndarray, sr: int = 44100) -> Dict:
        try:
            rms = librosa.feature.rms(y=y)[0]
            energy_raw = float(np.mean(rms))
            energy_normalized = min(energy_raw * 4, 1.0)
            energy_dj = int(np.clip(energy_normalized * 10, 1, 10))
            
            chunk = len(rms) // 3
            if chunk > 0:
                mix_start = min(float(np.mean(rms[:chunk])) * 4, 1.0)
                mix_end = min(float(np.mean(rms[2*chunk:])) * 4, 1.0)
            else:
                mix_start = mix_end = energy_normalized
            
            return {
                'energy_raw': round(energy_raw, 4),
                'energy_normalized': round(energy_normalized, 3),
                'energy_dj': energy_dj,
                'mix_energy_start': round(mix_start, 3),
                'mix_energy_end': round(mix_end, 3),
                'source': 'librosa_improved'
            }
        except Exception as e:
            print(f" Error Energy: {e}")
            return {'energy_dj': 5, 'source': 'error'}
    
    def analyze_spectral(self, y: np.ndarray, sr: int = 44100) -> Dict:
        try:
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff_mean = float(np.mean(rolloff))
            
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            percussion_density = min(float(np.mean(onset_env)) / 10, 1.0)
            
            has_heavy_bass = centroid < 2500
            has_pads = float(np.std(rolloff)) < 1500
            
            return {
                'spectral_centroid': round(centroid, 1),
                'has_heavy_bass': has_heavy_bass,
                'has_pads': has_pads,
                'percussion_density': round(percussion_density, 3),
                'source': 'librosa_improved'
            }
        except Exception as e:
            return {'source': 'error'}
    
    def detect_vocals(self, y: np.ndarray, sr: int = 44100) -> Dict:
        """
        Deteccion de vocals DESACTIVADA.
        La deteccion automatica genera muchos falsos positivos en musica electronica.
        """
        # Siempre devolver False - la deteccion no es fiable para electronica
        return {
            'has_vocals': False,
            'confidence': 0.0,
            'source': 'disabled'
        }
    
    def analyze_structure(self, y: np.ndarray, sr: int = 44100) -> Dict:
        try:
            duration = len(y) / sr
            rms = librosa.feature.rms(y=y)[0]
            
            num_sections = min(10, int(duration / 30) + 1)
            section_len = len(rms) // num_sections
            energies = [float(np.mean(rms[i*section_len:(i+1)*section_len])) for i in range(num_sections)]
            energies = np.array(energies)
            avg = np.mean(energies)
            
            has_intro = energies[0] < avg * 0.7
            has_outro = energies[-1] < avg * 0.7
            
            max_idx = np.argmax(energies)
            has_drop = energies[max_idx] > avg * 1.4
            drop_time = (max_idx / num_sections) * duration if has_drop else 0.0
            
            has_buildup = has_drop and max_idx > 1 and energies[max_idx-1] > energies[0] * 1.2
            has_breakdown = has_drop and max_idx < num_sections - 2 and np.min(energies[max_idx+1:]) < avg * 0.6
            
            sections = []
            sec_dur = duration / num_sections
            for i, e in enumerate(energies):
                if e > avg * 1.4:
                    t = 'drop'
                elif e < avg * 0.6:
                    t = 'intro' if i < 2 else ('outro' if i > num_sections - 3 else 'breakdown')
                elif i > 0 and energies[i] > energies[i-1] * 1.15:
                    t = 'buildup'
                else:
                    t = 'main'
                sections.append({'start': round(i*sec_dur, 2), 'end': round((i+1)*sec_dur, 2), 'type': t})
            
            return {
                'has_intro': has_intro,
                'has_buildup': has_buildup,
                'has_drop': has_drop,
                'has_breakdown': has_breakdown,
                'has_outro': has_outro,
                'drop_timestamp': round(drop_time, 2),
                'sections': sections,
                'source': 'librosa_improved'
            }
        except Exception as e:
            return {'has_drop': False, 'sections': [], 'source': 'error'}
    
    def full_analysis(self, file_path: str, sample_rate: int = 44100) -> Dict:
        print(f"ðŸ”¬ Analizando: {file_path}")
        
        y, sr = self.load_audio(file_path, sample_rate)
        duration = len(y) / sr
        
        bpm = self.analyze_bpm_improved(y, sr)
        key = self.analyze_key_improved(y, sr)
        energy = self.analyze_energy(y, sr)
        spectral = self.analyze_spectral(y, sr)
        vocals = self.detect_vocals(y, sr)
        structure = self.analyze_structure(y, sr)
        
        print(f"  â†’ BPM: {bpm.get('bpm')} | Key: {key.get('key')}/{key.get('camelot')} | Energy: {energy.get('energy_dj')}/10")
        
        return {
            'duration': duration,
            'bpm': bpm,
            'key': key,
            'energy': energy,
            'spectral': spectral,
            'vocals': vocals,
            'structure': structure,
            'analyzer': 'librosa_improved'
        }


def get_analyzer():
    if LIBROSA_AVAILABLE:
        return ImprovedLibrosaAnalyzer(use_edma_profiles=True)
    raise ImportError("Librosa no disponible")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Audio Analyzer Pro - Test")
    print("="*50)
    analyzer = get_analyzer()
    print(f"“ {type(analyzer).__name__} listo")
