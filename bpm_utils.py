"""
BPM validation and correction utilities.

Beatport helpers (validate_beatport_bpm, smart_bpm_correction) se eliminaron
junto con el resto de la integracion Beatport en rama
claude/kill-beatport-fix-audd-logging. Beatport era inalcanzable desde
Render (Cloudflare WAF) y desde IPs residenciales sin browser real;
0 de 84 tracks tenian bpm_source='beatport'.
"""
import math
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)



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

    # La autocorrelacion del onset_env NO depende de `bpm` (es loop-invariante):
    # antes se recalculaba dentro del bucle hasta 3 veces. np.correlate(a, a,
    # 'full') es O(n^2) sobre miles de frames, asi que computarla una sola vez
    # ahorra 2/3 del coste en el path de correccion (tracks de baja confianza).
    autocorr = np.correlate(onset_env, onset_env, mode='full')
    center = len(autocorr) // 2

    for bpm in candidates:
        try:
            beat_interval = 60.0 / bpm
            beat_frames = int(beat_interval * sr / 512)

            if beat_frames <= 0 or beat_frames >= len(onset_env):
                continue

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


def normalize_bpm_to_canonical(bpm: float) -> float:
    """
    Normaliza BPM al rango canonico [60, 180] multiplicando o dividiendo por 2.

    Util para consensus comunitario: distintos DJs pueden reportar el mismo
    track como 64, 128 o 256 (halftime/doubletime). La normalizacion los
    colapsa al mismo valor canonico para que la mediana funcione.

    Ejemplos:
    - 32   -> 64  (32 -> 64; 64 ya esta en [60, 180])
    - 64   -> 64  (ya esta en rango)
    - 128  -> 128 (no cambia)
    - 180  -> 180 (limite superior incluido)
    - 256  -> 128 (256 -> 128)
    - 60   -> 60  (limite inferior incluido)
    - 59   -> 118 (59 -> 118; sale del rango por el limite inferior)

    Casos edge:
    - bpm <= 0: raise ValueError
    - bpm es NaN/Inf: raise ValueError
    - Resultado redondeado a 1 decimal.
    """
    try:
        value = float(bpm)
    except (TypeError, ValueError):
        raise ValueError(f"BPM debe ser numerico: {bpm!r}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"BPM no puede ser NaN/Inf: {bpm!r}")
    if value <= 0:
        raise ValueError(f"BPM debe ser positivo: {bpm}")

    while value < 60:
        value *= 2
    while value > 180:
        value /= 2

    return round(value, 1)
