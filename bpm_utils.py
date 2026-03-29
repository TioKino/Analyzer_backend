"""
BPM validation and correction utilities.
"""
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


def validate_beatport_bpm(local_bpm: float, beatport_bpm: float, tolerance: float = 0.12) -> bool:
    """Valida si el BPM de Beatport corresponde al track local."""
    if local_bpm <= 0 or beatport_bpm <= 0:
        return True
    ratio = beatport_bpm / local_bpm
    if abs(ratio - 1.0) <= tolerance:
        return True
    if abs(ratio - 2.0) <= tolerance and beatport_bpm >= 80:
        return True
    if abs(ratio - 0.5) <= tolerance and beatport_bpm >= 80:
        return True
    return False


def smart_bpm_correction(local_bpm: float, beatport_bpm: float) -> float:
    """
    Correccion inteligente de BPM half/double tempo.

    Si Beatport dice 140 y local dice 70 -> es double tempo -> corregir a 140.
    Si Beatport dice 70 y local dice 140 -> es half tempo -> corregir a 70.
    Si estan dentro del 12%, Beatport gana (datos del sello).
    Si no hay match, devolver None (rechazar Beatport).
    """
    if local_bpm <= 0 or beatport_bpm <= 0:
        return beatport_bpm or local_bpm

    ratio = beatport_bpm / local_bpm

    if abs(ratio - 1.0) <= 0.12:
        return beatport_bpm
    if abs(ratio - 2.0) <= 0.15:
        return beatport_bpm
    if abs(ratio - 0.5) <= 0.15:
        return beatport_bpm

    return None


def try_bpm_double_half(y, sr, original_bpm: float, bpm_confidence: float, onset_env=None) -> float:
    """
    Si la confianza del BPM es baja, probar con doble y mitad.
    """
    if bpm_confidence > 0.7:
        return original_bpm

    candidates = [original_bpm]

    double_bpm = original_bpm * 2
    half_bpm = original_bpm / 2

    if 60 <= double_bpm <= 200:
        candidates.append(double_bpm)
    if 60 <= half_bpm <= 200:
        candidates.append(half_bpm)

    if len(candidates) == 1:
        return original_bpm

    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    best_bpm = original_bpm
    best_score = -1

    for bpm in candidates:
        try:
            beat_interval = 60.0 / bpm
            beat_frames = int(beat_interval * sr / 512)

            if beat_frames <= 0 or beat_frames >= len(onset_env):
                continue

            autocorr = np.correlate(onset_env, onset_env, mode='full')
            center = len(autocorr) // 2

            idx = center + beat_frames
            if idx < len(autocorr):
                score = autocorr[idx]
                if score > best_score:
                    best_score = score
                    best_bpm = bpm
        except Exception:
            continue

    if best_bpm != original_bpm:
        logger.info(f"[BPM] Correccion double/half: {original_bpm:.1f} -> {best_bpm:.1f}")

    return best_bpm
