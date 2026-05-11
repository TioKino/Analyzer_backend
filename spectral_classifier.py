"""
SPECTRAL TRACK CLASSIFIER (Fase 3 v2)
====================================

Port a Python del clasificador spectral que vivia en Flutter:
- lib/ui/desktop/waveform/waveform_fft.dart (extraccion de metricas)
- lib/ui/desktop/waveform/spectral_track_classifier.dart (scoring 7 tipos)

Mismas formulas y mismo scoring que el clasificador Dart original (calibrado
con 39 tracks reales — ver comentarios del Dart para rangos observados). Al
correr aqui en lugar de Flutter:

  - Mobile gana spectral classification (antes no la tenia: era desktop only).
  - El motor local PyInstaller hereda la misma logica que Render.
  - Backend = source of truth: una sola implementacion en lugar de duplicar
    el algoritmo en cada plataforma.

Pipeline:
  1. compute_spectral_metrics(y, sr) -> dict de 16 metricas (STFT + extraccion).
  2. classify_track_type_spectral(metrics, bpm, duration) -> dict con
     {type, confidence, alternatives, reason, source='spectral'}.
  3. detect_heavy_bass(metrics) -> bool (bassRatio >= 0.45).

El caller (main.py) hace ensemble con classify_track_type heuristic de Fase 1.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import librosa


# ============================================================================
# COMPUTO DE METRICAS (port de waveform_fft.dart _extractMetrics)
# ============================================================================

def compute_spectral_metrics(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Computa las 16 metricas spectral de un track (igual que la version Dart).

    Args:
        y: array mono float32/64. Si viene estereo se promedia. Mismo rango
           que librosa.load (~ -1.0..1.0).
        sr: sample rate.

    Returns:
        dict con: avgBass, avgMid, avgTreble, bassRatio, midRatio,
        trebleRatio, energyTrend, energyVariance, peakPosition, dropContrast,
        introPercent, outroPercent, coreEnergy, bassRegularity,
        transientDensity, midVariance.
    """
    n_fft = 2048
    hop = 512

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    if len(y) < n_fft:
        return _empty_metrics()

    # STFT magnitudes — mismas dimensiones que la version Dart (hop=512, nFft=2048).
    stft_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))  # (n_freq, n_frames)
    n_freq, n = stft_mag.shape
    if n == 0:
        return _empty_metrics()

    # Banding identico al Dart: bass 0-250Hz, mid 250-4000Hz, treble >4000Hz.
    freq_per_bin = sr / n_fft
    bass_max_bin = int(np.ceil(250.0 / freq_per_bin))
    mid_max_bin = int(np.ceil(4000.0 / freq_per_bin))
    half_n = n_fft // 2

    bass_bins = max(1, bass_max_bin)
    mid_bins = max(1, mid_max_bin - bass_max_bin)
    treble_bins = max(1, half_n - mid_max_bin)

    # Sumas por banda y por frame (excluyendo bin DC = 0).
    bass = stft_mag[1:bass_max_bin + 1, :].sum(axis=0) / bass_bins
    mid = stft_mag[bass_max_bin + 1:mid_max_bin + 1, :].sum(axis=0) / mid_bins
    treble = stft_mag[mid_max_bin + 1:half_n, :].sum(axis=0) / treble_bins

    # Paso 0: peaks per band.
    peak_bass = float(bass.max()) if len(bass) else 0.0
    peak_mid = float(mid.max()) if len(mid) else 0.0
    peak_treble = float(treble.max()) if len(treble) else 0.0

    # Paso 1: totales por frame + averages crudos.
    total_energy = bass + mid + treble
    avg_bass = float(bass.mean())
    avg_mid = float(mid.mean())
    avg_treble = float(treble.mean())
    peak_e = float(total_energy.max())
    peak_idx = int(np.argmax(total_energy))

    # Paso 2: ratios per-band normalized.
    norm_bass = bass / peak_bass if peak_bass > 0 else np.zeros_like(bass)
    norm_mid = mid / peak_mid if peak_mid > 0 else np.zeros_like(mid)
    norm_treble = treble / peak_treble if peak_treble > 0 else np.zeros_like(treble)
    sum_norm_bass = float(norm_bass.sum())
    sum_norm_mid = float(norm_mid.sum())
    sum_norm_treble = float(norm_treble.sum())
    total_norm = sum_norm_bass + sum_norm_mid + sum_norm_treble
    if total_norm > 0:
        bass_ratio = sum_norm_bass / total_norm
        mid_ratio = sum_norm_mid / total_norm
        treble_ratio = sum_norm_treble / total_norm
    else:
        bass_ratio = mid_ratio = treble_ratio = 1.0 / 3.0

    # Paso 3: energy normalizada al pico del propio track.
    normalized_energy = total_energy / peak_e if peak_e > 0 else np.zeros_like(total_energy)

    # Regresion lineal (slope * n -> energyTrend acotado).
    x = np.arange(n, dtype=np.float64)
    y_norm = normalized_energy.astype(np.float64)
    sum_x = x.sum()
    sum_y = y_norm.sum()
    sum_xy = float((x * y_norm).sum())
    sum_xx = float((x * x).sum())
    denom = n * sum_xx - sum_x * sum_x
    slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0
    energy_trend = float(np.clip(slope * n, -2.0, 2.0))

    # Paso 4: varianza (std) de energia normalizada.
    avg_norm_e = float(y_norm.mean())
    energy_variance = float(np.sqrt(((y_norm - avg_norm_e) ** 2).mean()))

    # Paso 5: peak position.
    peak_position = peak_idx / n if n > 0 else 0.5

    # Paso 6: drop contrast (p90/p10 cappeado).
    sorted_norm = np.sort(y_norm)
    p10 = float(sorted_norm[min(int(n * 0.10), n - 1)])
    p90 = float(sorted_norm[min(int(n * 0.90), n - 1)])
    if p10 > 0.01:
        drop_contrast = float(np.clip(p90 / p10, 1.0, 20.0))
    else:
        drop_contrast = 10.0 if p90 > 0 else 1.0

    # Paso 7: intro y outro (threshold 30% del pico).
    intro_threshold = 0.30
    intro_end = 0
    for i in range(n):
        if y_norm[i] >= intro_threshold:
            intro_end = i
            break
    intro_percent = intro_end / n if n > 0 else 0.0

    outro_start = n
    for i in range(n - 1, -1, -1):
        if y_norm[i] >= intro_threshold:
            outro_start = i + 1
            break
    outro_percent = (n - outro_start) / n if n > 0 else 0.0

    # Paso 8: core energy (avg normalized en el core).
    core_start = intro_end
    core_end = max(core_start + 1, min(outro_start, n))
    core_len = core_end - core_start
    if core_len > 0:
        core_energy = float(np.clip(y_norm[core_start:core_end].mean(), 0.0, 1.0))
    else:
        core_energy = 0.5

    # Paso 9: bass regularity (autocorrelacion bass con lag 15-25).
    best_corr = 0.0
    max_i_global = min(n // 2, n - 15)
    for lag in range(15, 26):
        if n - lag <= 0:
            continue
        a_start = core_start
        a_end = min(core_start + max_i_global, core_end - lag)
        if a_end <= a_start:
            continue
        a = bass[a_start:a_end]
        b = bass[a_start + lag:a_end + lag]
        if len(a) != len(b) or len(a) == 0:
            continue
        norm_a = float((a * a).sum())
        norm_b = float((b * b).sum())
        d = np.sqrt(norm_a * norm_b)
        if d > 0:
            corr = float((a * b).sum() / d)
            if corr > best_corr:
                best_corr = corr
    bass_regularity = float(np.clip(best_corr, 0.0, 1.0))

    # Paso 10: transient density.
    transient_count = 0
    for i in range(1, n):
        prev = y_norm[i - 1]
        if prev > 0.05 and y_norm[i] > prev * 1.5:
            transient_count += 1
    transient_density = float(np.clip(transient_count / (n - 1), 0.0, 1.0)) if n > 1 else 0.0

    # Paso 11: mid variance (vocal detection proxy).
    if core_len > 0:
        norm_mid_core = norm_mid[core_start:core_end]
        mid_norm_avg = float(norm_mid_core.mean())
        mid_variance = float(np.sqrt(((norm_mid_core - mid_norm_avg) ** 2).mean()))
    else:
        mid_variance = 0.0

    return {
        'avgBass': round(avg_bass, 4),
        'avgMid': round(avg_mid, 4),
        'avgTreble': round(avg_treble, 4),
        'bassRatio': round(bass_ratio, 4),
        'midRatio': round(mid_ratio, 4),
        'trebleRatio': round(treble_ratio, 4),
        'energyTrend': round(energy_trend, 4),
        'energyVariance': round(energy_variance, 4),
        'peakPosition': round(peak_position, 4),
        'dropContrast': round(drop_contrast, 2),
        'introPercent': round(intro_percent, 4),
        'outroPercent': round(outro_percent, 4),
        'coreEnergy': round(core_energy, 4),
        'bassRegularity': round(bass_regularity, 4),
        'transientDensity': round(transient_density, 4),
        'midVariance': round(mid_variance, 4),
    }


def _empty_metrics() -> Dict[str, float]:
    return {
        'avgBass': 0.0, 'avgMid': 0.0, 'avgTreble': 0.0,
        'bassRatio': 0.33, 'midRatio': 0.33, 'trebleRatio': 0.33,
        'energyTrend': 0.0, 'energyVariance': 0.0, 'peakPosition': 0.5,
        'dropContrast': 1.0, 'introPercent': 0.0, 'outroPercent': 0.0,
        'coreEnergy': 0.0, 'bassRegularity': 0.0, 'transientDensity': 0.0,
        'midVariance': 0.0,
    }


# ============================================================================
# CLASIFICADOR SPECTRAL (port de spectral_track_classifier.dart)
# ============================================================================

_ALL_TYPES_SPECTRAL = (
    'opener', 'warmup', 'builder', 'peak_time', 'anthem', 'cooldown', 'closing',
)


def classify_track_type_spectral(
    metrics: Dict[str, float],
    bpm: float,
    duration: float,
) -> Dict[str, Any]:
    """Scoring de 7 tipos basado en metricas spectral + BPM + duration.

    Mismos pesos que el Dart original (calibrado con 39 tracks). Devuelve dict
    con la misma shape que classify_track_type (heuristic de Fase 1) para que
    el ensemble en main.py pueda combinarlos directamente.
    """
    m = metrics
    scores = {t: 0.0 for t in _ALL_TYPES_SPECTRAL}

    core = m.get('coreEnergy', 0.0)
    bass = m.get('bassRatio', 0.33)
    contrast = m.get('dropContrast', 1.0)
    trend = m.get('energyTrend', 0.0)
    variance = m.get('energyVariance', 0.0)
    peak_pos = m.get('peakPosition', 0.5)
    intro_pct = m.get('introPercent', 0.0)
    outro_pct = m.get('outroPercent', 0.0)
    transients = m.get('transientDensity', 0.0)
    bass_reg = m.get('bassRegularity', 0.0)

    # OPENER: energia baja, poco percusivo, BPM moderado.
    if core < 0.28: scores['opener'] += 3.0
    if core < 0.25: scores['opener'] += 2.0
    if bass_reg < 0.70: scores['opener'] += 1.5
    if transients < 0.03: scores['opener'] += 1.0
    if variance < 0.18: scores['opener'] += 1.0
    if 0 < bpm < 118: scores['opener'] += 1.0

    # WARMUP: energia media-baja, dinamica moderada, plano.
    if 0.26 <= core <= 0.35: scores['warmup'] += 3.0
    if 0.15 <= variance <= 0.22: scores['warmup'] += 2.0
    if abs(trend) < 0.06: scores['warmup'] += 2.0  # plano
    if contrast < 8.0: scores['warmup'] += 1.0
    if bass < 0.42: scores['warmup'] += 1.0

    # BUILDER: tendencia ascendente o pico en la segunda mitad.
    if trend > 0.06: scores['builder'] += 3.0
    if trend > 0.10: scores['builder'] += 2.0
    if trend > 0.14: scores['builder'] += 2.0
    if peak_pos > 0.65: scores['builder'] += 2.0
    if peak_pos > 0.80: scores['builder'] += 1.5
    if core >= 0.28: scores['builder'] += 1.0
    if trend < 0: scores['builder'] -= 5.0  # penalizacion

    # PEAK TIME: energia alta, contraste alto, dinamica variada.
    if core > 0.35: scores['peak_time'] += 2.0
    if core > 0.40: scores['peak_time'] += 2.0
    if contrast > 8.0: scores['peak_time'] += 2.0
    if variance > 0.20: scores['peak_time'] += 2.0
    if 0.40 < bass < 0.55: scores['peak_time'] += 1.0
    if 0.35 < peak_pos < 0.75: scores['peak_time'] += 1.5
    if variance < 0.16: scores['peak_time'] -= 2.0  # penalizacion

    # ANTHEM: contraste MUY alto, energia alta, bass-heavy.
    if contrast > 12.0: scores['anthem'] += 3.0
    if contrast > 16.0: scores['anthem'] += 2.0
    if variance > 0.22: scores['anthem'] += 2.0
    if core > 0.30: scores['anthem'] += 1.5
    if bass > 0.45: scores['anthem'] += 1.5
    if contrast < 6.0: scores['anthem'] -= 4.0  # penalizacion

    # COOLDOWN: tendencia descendente, peak al principio.
    if trend < -0.06: scores['cooldown'] += 3.0
    if trend < -0.10: scores['cooldown'] += 2.0
    if trend < -0.14: scores['cooldown'] += 2.0
    if peak_pos < 0.35: scores['cooldown'] += 2.0
    if peak_pos < 0.20: scores['cooldown'] += 1.5
    if trend > 0: scores['cooldown'] -= 5.0  # penalizacion

    # CLOSING: outro largo, energia baja, tendencia descendente.
    if outro_pct > 0.08: scores['closing'] += 3.0
    if outro_pct > 0.15: scores['closing'] += 2.0
    if core < 0.30: scores['closing'] += 2.0
    if trend < -0.03: scores['closing'] += 1.5
    if intro_pct > 0.10: scores['closing'] += 1.0
    if duration > 420: scores['closing'] += 0.5  # tracks largos

    # BPM adjustments.
    if bpm > 0:
        if bpm < 120:
            scores['opener'] += 1.5
            scores['warmup'] += 1.0
        if bpm > 130:
            scores['peak_time'] += 1.0
            scores['anthem'] += 1.0
        if bpm > 138:
            scores['anthem'] += 1.5

    # Encontrar ganador + confidence.
    sorted_items = sorted(scores.items(), key=lambda kv: -kv[1])
    winner_type, winner_score = sorted_items[0]
    second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    margin = winner_score - second_score
    confidence = float(np.clip(margin / (abs(winner_score) + 1.0), 0.0, 1.0))

    return {
        'type': winner_type,
        'confidence': round(confidence, 2),
        'alternatives': [
            {'type': t, 'score': round(s, 2)} for t, s in sorted_items
        ],
        'reason': _build_reason(winner_type, m, bpm),
        'source': 'spectral',
    }


def _build_reason(track_type: str, m: Dict[str, float], bpm: float) -> str:
    """Razones cortas para logs / debug. Mismos formatos que el Dart."""
    core = m.get('coreEnergy', 0.0)
    bass = m.get('bassRatio', 0.0)
    contrast = m.get('dropContrast', 0.0)
    trend = m.get('energyTrend', 0.0)
    variance = m.get('energyVariance', 0.0)
    peak_pos = m.get('peakPosition', 0.0)
    outro_pct = m.get('outroPercent', 0.0)

    if track_type == 'builder':
        return f"Trend +{trend:.2f}, peak at {int(peak_pos * 100)}%"
    if track_type == 'cooldown':
        return f"Trend {trend:.2f}, peak at {int(peak_pos * 100)}%"
    if track_type == 'opener':
        return f"Core {core:.2f}, low dynamics"
    if track_type == 'closing':
        return f"Outro {int(outro_pct * 100)}%, core {core:.2f}"
    if track_type == 'peak_time':
        return f"Core {core:.2f}, contrast {contrast:.1f}x, var {variance:.2f}"
    if track_type == 'anthem':
        return f"Contrast {contrast:.1f}x, bass {int(bass * 100)}%, var {variance:.2f}"
    return f"Core {core:.2f}, bass {int(bass * 100)}%, bpm {bpm:.0f}"


# ============================================================================
# HEAVY BASS DETECTION (port de spectral_track_classifier.dart:detectHeavyBass)
# ============================================================================

def detect_heavy_bass(metrics: Dict[str, float]) -> bool:
    """Bass-dominant si bassRatio (normalizado per-band) >= 0.45."""
    return metrics.get('bassRatio', 0.0) >= 0.45


# ============================================================================
# ENSEMBLE: combina heuristic (Fase 1) + spectral (Fase 3)
# ============================================================================

# Pesos iniciales. El spectral pesa mas porque tiene 7 tipos discriminados
# vs los 3 del heuristic (warmup/peak_time/closing). Calibracion futura con
# dataset de tracks con ground truth.
ENSEMBLE_ALPHA = 1.0   # peso heuristic
ENSEMBLE_BETA = 1.5    # peso spectral

_ENSEMBLE_ALL_TYPES = (
    'warmup', 'peak_time', 'closing',
    'opener', 'builder', 'anthem', 'cooldown',
)


def ensemble_classify(
    heuristic: Dict[str, Any],
    spectral: Dict[str, Any],
) -> Dict[str, Any]:
    """Combina los 'alternatives' de heuristic + spectral con pesos α/β.

    Returns dict con el mismo shape que classify_track_type (Fase 1):
      {type, confidence, alternatives, reason, source='ensemble'}.

    Si una de las dos clasificaciones esta vacia o falla, devuelve la otra
    intacta.
    """
    h_alts = heuristic.get('alternatives', []) if heuristic else []
    s_alts = spectral.get('alternatives', []) if spectral else []
    if not h_alts and not s_alts:
        return {
            'type': 'peak_time',
            'confidence': 0.0,
            'alternatives': [],
            'reason': 'sin datos',
            'source': 'ensemble',
        }

    h_scores = {a['type']: float(a.get('score', 0.0)) for a in h_alts}
    s_scores = {a['type']: float(a.get('score', 0.0)) for a in s_alts}

    ensemble_scores: Dict[str, float] = {}
    for t in _ENSEMBLE_ALL_TYPES:
        ensemble_scores[t] = (
            ENSEMBLE_ALPHA * h_scores.get(t, 0.0)
            + ENSEMBLE_BETA * s_scores.get(t, 0.0)
        )

    sorted_items = sorted(ensemble_scores.items(), key=lambda kv: -kv[1])
    winner_type, winner_score = sorted_items[0]
    second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    if winner_score <= 0:
        # Caso degenerate: todos los scores son 0 o negativos. Caer al spectral
        # winner (que ya tiene un fallback razonable).
        if spectral and spectral.get('type'):
            return {**spectral, 'source': 'ensemble'}
        return {**heuristic, 'source': 'ensemble'}

    margin = winner_score - second_score
    confidence = float(np.clip(margin / (abs(winner_score) + 0.5), 0.0, 1.0))

    return {
        'type': winner_type,
        'confidence': round(confidence, 2),
        'alternatives': [
            {'type': t, 'score': round(s, 2)} for t, s in sorted_items
        ],
        'reason': (
            f"ensemble α={ENSEMBLE_ALPHA}/β={ENSEMBLE_BETA} "
            f"| h={heuristic.get('type') if heuristic else 'n/a'}"
            f"/{heuristic.get('confidence') if heuristic else 'n/a'} "
            f"| s={spectral.get('type') if spectral else 'n/a'}"
            f"/{spectral.get('confidence') if spectral else 'n/a'}"
        ),
        'source': 'ensemble',
    }
