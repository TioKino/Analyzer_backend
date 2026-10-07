"""Lo que los dos caminos de /analyze tienen que calcular IGUAL.

POR QUÉ EXISTE ESTE MÓDULO
--------------------------
`/analyze` tiene dos caminos: el CORTO (`main.analyze_audio`, el tema entero
en memoria) para lo que dura 4 minutos o menos —y para TODO en el motor
local—, y el POR TROZOS (`chunked_analyzer`, trozos de 60 s) para lo que dura
más en Render, que es casi cualquier tema de club. Hasta el 2026-10-06 cada
uno calculaba lo suyo, y el mismo tema salía distinto según durara 3:59 o
4:01. Medido con el mismo tema sintético de 5 minutos por los dos caminos:

    campo            corto              por trozos
    género           Minimal Techno     Electronic        (fijo)
    groove / swing   0,057 / 1,000      0,5 / 0,5         (fijos)
    graves pesados   no                 sí                (otra regla)
    tipo             opener             warmup            (sin el espectral)
    doble/mitad      sí                 no

Aquí vive lo que los dos tienen que hacer con las MISMAS cuentas: el corto lo
llama con el tema entero y el por trozos con lo de cada trozo cosido
(`chunked_analyzer.full_analysis`). Con un solo trozo, sale lo mismo que el
corto bit a bit (`test_el_chunked_calcula_lo_mismo.py`).

La ENERGÍA es igual desde el 2026-10-07 (`energia_del_tema`): el de trozos
la medía en ventanas de 2 s y el corto en las de 46 ms de `librosa`, y como
la media de RMS sube con la ventana, los temas largos salían ~1 nivel más
arriba con el mismo volumen (0,146 frente a 0,132 en el mismo tema
sintético: nivel 6 frente a 5), y con ella cambiaban el tipo y el género.

Lo que todavía NO es igual, a propósito: la TONALIDAD (voto por trozos
frente a croma del tema entero) y la ESTRUCTURA (dos algoritmos). Cambiarlas
mueve valores que la gente ya tiene guardados, y se deciden con medidas en
sus propios puntos (`PENDING.md`).
"""

import logging
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import librosa

logger = logging.getLogger(__name__)

HOP = 512


def try_bpm_double_half(y, sr, original_bpm: float, bpm_confidence: float, onset_env=None) -> float:
    """
    Si la confianza del BPM es baja, probar con doble y mitad.
    
    Logica: si librosa dice 131 con confianza 0.4, probar 262 y 65.5.
    Si alguno de esos tiene sentido musical (60-200 BPM range) Y tiene
    mejor alineacion con los beats, usarlo.
    
    Args:
        onset_env: Si ya se calculó onset_strength, pasarlo para no duplicar CPU.
    """
    if bpm_confidence >= 0.7:
        return original_bpm  # Alta confianza, no tocar
    
    candidates = [original_bpm]
    
    # Probar doble
    double = original_bpm * 2
    if 60 <= double <= 200:
        candidates.append(double)
    
    # Probar mitad
    half = original_bpm / 2
    if 60 <= half <= 200:
        candidates.append(half)
    
    if len(candidates) == 1:
        return original_bpm
    
    # Evaluar cual se alinea mejor con onset strength
    try:
        if onset_env is None:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        best_bpm = original_bpm
        best_score = 0
        
        for candidate in candidates:
            # Crear pulso teorico para este BPM
            beat_interval = 60.0 / candidate
            sr_onset = sr / 512  # hop_length default
            
            # Autocorrelacion con el BPM candidato
            period = int(round(sr_onset * beat_interval))
            if period > 0 and period < len(onset_env) // 2:
                corr = np.correlate(onset_env[:len(onset_env)//2], 
                                     onset_env[period:period + len(onset_env)//2])
                score = float(np.max(corr)) if len(corr) > 0 else 0
                if score > best_score:
                    best_score = score
                    best_bpm = candidate
        
        if best_bpm != original_bpm:
            logger.info(f"   BPM auto-corregido: {original_bpm:.1f} -> {best_bpm:.1f} (confianza baja: {bpm_confidence:.2f})")
        
        return best_bpm
    except Exception:
        return original_bpm


def classify_track_type(energy: float, segments: dict, duration: float) -> dict:
    # Fase 1 Track Type v2: pasamos de cascada de returns simples a scoring
    # con margin top-1 vs top-2 -> confidence (0..1). Permite a la UI mostrar
    # honestidad sobre tracks ambiguos en lugar de mentir con un tipo forzado.
    # Plan completo en Analyzer/PENDING_NEXT_SESSION_TRACKTYPE_V2.md.
    #
    # Mismas señales que la cascada original (has_intro/has_drop + energy),
    # pero acumulamos en lugar de decidir inmediato.
    #
    # *Closing* es el tema con el que se CIERRA una sesión (owner,
    # 2026-10-07), y la heurística no tiene con qué verlo: hasta ese día
    # sumaba 1,0 a closing con un outro y más de 5 minutos, o sea a cualquier
    # extended mix con su outro de batería para mezclar (8 de 13 temas largos
    # en un log de Render, con confianza 1,0 y el espectral descartado). Un
    # outro para mezclar no dice que el tema cierre nada. Closing se queda en
    # el reparto con 0 y lo decide el espectral (energía baja que va bajando).
    scores = {'warmup': 0.0, 'peak_time': 0.0, 'closing': 0.0}

    if energy < 0.5 and segments['has_intro']:
        scores['warmup'] += 1.0
    if energy < 0.4 and segments['has_intro']:
        scores['warmup'] += 0.5
    if energy > 0.7 and segments['has_drop']:
        scores['peak_time'] += 1.0
    if energy > 0.8 and segments['has_drop']:
        scores['peak_time'] += 0.5
    # Soft signals para desempates: cualquier track con energia alta
    # tira hacia peak_time, cualquiera con energia baja hacia warmup.
    if energy > 0.6:
        scores['peak_time'] += 0.2
    elif energy < 0.5:
        scores['warmup'] += 0.2

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    winner_type, winner_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    if winner_score == 0.0:
        # Track sin señales claras: caer al fallback de la cascada original
        # (energy>0.6 -> peak_time, sino warmup) y reportar confidence 0
        # para que la UI muestre el badge como "incierto".
        winner_type = 'peak_time' if energy > 0.6 else 'warmup'
        confidence = 0.0
    else:
        margin = winner_score - second_score
        confidence = min(1.0, margin / max(winner_score, 0.5))

    return {
        'type': winner_type,
        'confidence': round(confidence, 2),
        'alternatives': [
            {'type': t, 'score': round(s, 2)} for t, s in sorted_scores
        ],
        'reason': (
            f"energy={energy:.2f} duration={duration:.0f} "
            f"intro={segments['has_intro']} drop={segments['has_drop']} "
            f"outro={segments['has_outro']}"
        ),
        'source': 'waveform',
    }


# La escala DJ de la energía: el RMS medio del tema, de 0,02 (ambient) a 0,42
# (hardstyle), con una curva 0,55 que abre el rango medio. El cliente puede
# además repartirla por percentiles de su biblioteca (por eso se guarda
# `energy_raw`).
ENERGIA_RMS_MIN = 0.02
ENERGIA_RMS_MAX = 0.42
# Lo que se mide para mezclar: el principio y el final del tema.
SEGUNDOS_DE_MEZCLA = 30


def nivel_de_energia(energy_raw: float) -> int:
    """El RMS medio → nivel DJ de 1 a 10.

    Un RMS que no es un número (audio muy corto, silencio total) da 5: sin
    esa guarda `int(NaN)` reventaba el análisis entero (era el error nº 1 del
    panel admin, 112 veces).
    """
    if not math.isfinite(energy_raw):
        logger.warning(f"   Energia: energy_raw={energy_raw} NaN/Inf, fallback a 5")
        return 5
    if energy_raw <= ENERGIA_RMS_MIN:
        return 1
    if energy_raw >= ENERGIA_RMS_MAX:
        return 10
    normalizada = (energy_raw - ENERGIA_RMS_MIN) / (ENERGIA_RMS_MAX - ENERGIA_RMS_MIN)
    return max(1, min(10, int(round(1 + normalizada ** 0.55 * 9))))


def energia_del_tema(rms: np.ndarray, sr: int,
                     hop: int = HOP) -> Tuple[float, int, float, float]:
    """(energy_raw, energy_dj, mix_energy_start, mix_energy_end) a partir del
    RMS frame a frame de TODO el tema: `librosa.feature.rms` con sus valores
    por defecto (ventana de 2048 muestras, hop 512). El corto lo calcula con
    el tema entero y el de trozos lo cose trozo a trozo, igual que la
    envolvente de onset.

    No cambies la ventana en uno solo: la media de RMS sube con ella (lo que
    se promedia dentro de cada ventana va bajo la raíz), y con ventanas de
    2 s el de trozos salía un nivel más arriba que el corto con el mismo
    audio.
    """
    rms = np.asarray(rms, dtype=float)
    energy_raw = float(np.mean(rms)) if len(rms) else float('nan')
    frames = int(sr * SEGUNDOS_DE_MEZCLA) // hop
    if len(rms):
        inicio = float(np.mean(rms[:min(frames, len(rms))]))
        final = float(np.mean(rms[max(0, len(rms) - frames):]))
    else:
        inicio = final = 0.5
    return energy_raw, nivel_de_energia(energy_raw), inicio, final


def pulso_de_beats(beats, sr: int, hop: int = HOP) -> Tuple[float, float, float]:
    """(confianza del BPM, groove, swing) a partir de los beats de librosa.

    Las cuentas del camino corto de siempre; el de trozos daba 0,5 y 0,5 fijos.
    """
    intervalos = np.diff(librosa.frames_to_time(beats, sr=sr, hop_length=hop))
    confianza = (1.0 - min(np.std(intervalos) * 2, 0.5)
                 if len(intervalos) > 0 else 0.5)
    if len(intervalos) > 1:
        groove = min(np.std(intervalos) * 10, 1.0)
        swing = float(np.mean(intervalos[::2]) / np.mean(intervalos[1::2])
                      if len(intervalos) > 2 else 0.5)
    else:
        groove = 0.0
        swing = 0.5
    return float(confianza), float(groove), float(swing)


def tempograma(onset_mediana: np.ndarray, sr: int, hop: int = HOP) -> np.ndarray:
    """El tempograma con el que `librosa.beat.beat_track` decide el tempo:
    autocorrelación de 8 s sobre la envolvente de onset con MEDIANA."""
    ventana = librosa.time_to_frames(8.0, sr=sr, hop_length=hop).item()
    return librosa.feature.tempogram(onset_envelope=onset_mediana, sr=sr,
                                     hop_length=hop, win_length=ventana)


def tempo_y_beats(onset_mediana: np.ndarray, suma_tempograma: np.ndarray,
                  frames: int, sr: int, hop: int = HOP):
    """(tempo, beats) del tema entero a partir de lo cosido trozo a trozo.

    `beat_track` de librosa, con el audio, hace tres cosas: la envolvente de onset con
    mediana, el tempo como el máximo de la MEDIA del tempograma (con su
    prior), y los beats por programación dinámica sobre la envolvente. La
    media del tempograma de todo el tema es la media de las medias de cada
    trozo pesada por sus frames —eso es `suma_tempograma / frames`—, así que
    da el mismo tempo sin tener el tempograma entero en memoria (en un tema
    de 12 minutos son ~350 MB). Los beats salen de la envolvente cosida.
    """
    from audio_helpers import beat_track_seguro
    if frames <= 0 or not np.asarray(onset_mediana).any():
        return 0.0, np.array([], dtype=int)
    tg = (np.asarray(suma_tempograma, dtype=float) / frames)[:, None]
    tempo = float(librosa.feature.tempo(tg=tg, sr=sr, hop_length=hop)[0])
    _, beats = beat_track_seguro(None, sr, onset_envelope=onset_mediana,
                                 hop_length=hop, bpm=tempo)
    return tempo, beats


def rasgos_espectrales(*, bpm: float, energy_normalized: float,
                       segments: dict, duration: float,
                       onset_env, spectral_centroid, rolloff,
                       bandas: Optional[Sequence[np.ndarray]],
                       graves_si_falla: bool = False) -> Dict:
    """Tipo de track, graves, pads, percusión y género espectral.

    Las cuentas del camino corto: la clasificación heurística más la espectral
    en conjunto (`ensemble_classify`), los graves por la proporción de la banda
    baja, los pads por la variación del rolloff, la percusión por la media del
    onset y el género por perfiles (`classify_genre_advanced`). `bandas` son
    las tres bandas por frame (`spectral_classifier.bandas_de_audio`).
    `graves_si_falla`: lo que vale «graves pesados» si el espectral falla.
    """
    from spectral_genre_classifier import classify_genre_advanced

    percussion_density = min(float(np.mean(onset_env)) / 10, 1.0)
    has_pads = float(np.std(rolloff)) < 1000
    has_heavy_bass = bool(graves_si_falla)

    track_type = 'peak_time'  # default seguro
    track_type_confidence = 0.5  # neutral si la clasificacion falla
    track_type_alternatives = []
    try:
        classification = classify_track_type(energy_normalized, segments, duration)
        track_type = classification['type']
        track_type_confidence = classification['confidence']
        track_type_alternatives = classification['alternatives']
    except Exception as e:  # noqa: BLE001
        logger.error(f"  [TrackType] Error clasificando: {e}")
        classification = None

    # Spectral + ensemble (Fase 3 v2): metrics FFT + scoring 7 tipos
    # (vs 3 del heuristic). El spectral pesa β=1.5 vs α=1.0 del heuristic.
    # Refina tambien has_heavy_bass con bassRatio normalizado per-band.
    try:
        from spectral_classifier import (
            metricas_de_bandas, _empty_metrics,
            classify_track_type_spectral,
            detect_heavy_bass as _spectral_detect_heavy_bass,
            ensemble_classify,
        )
        spectral_metrics = (metricas_de_bandas(*bandas) if bandas is not None
                            else _empty_metrics())
        spectral_classification = classify_track_type_spectral(
            spectral_metrics, bpm, duration
        )
        ensemble = ensemble_classify(classification, spectral_classification)
        track_type = ensemble['type']
        track_type_confidence = ensemble['confidence']
        track_type_alternatives = ensemble['alternatives']
        # Heavy bass refinado (per-band ratio > heuristic crudo de low_freq_energy)
        has_heavy_bass = _spectral_detect_heavy_bass(spectral_metrics)
        logger.info(
            f"  [Spectral+Ensemble] {ensemble['type']} "
            f"conf={ensemble['confidence']:.2f} | {ensemble['reason']}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  [Spectral] Failed, usando solo heuristic: {e}")

    genre = classify_genre_advanced(
        bpm, energy_normalized, has_heavy_bass,
        None, None, percussion_density,
        spectral_centroid, rolloff
    )
    return {
        'track_type': track_type,
        'track_type_confidence': track_type_confidence,
        'track_type_alternatives': track_type_alternatives,
        'has_heavy_bass': bool(has_heavy_bass),
        'has_pads': bool(has_pads),
        'percussion_density': percussion_density,
        'genre': genre,
    }


def rejilla_y_bpm(onset_env, fps: float, bpm: float,
                  bpm_del_dsp: bool) -> Tuple[float, float, float]:
    """(primer beat, intervalo, BPM) a partir de la envolvente de onset.

    `fit_beat_grid` da fase, downbeat e intervalo afinado; None = no hay pulso
    que medir, y entonces la rejilla se queda sin fase (una inventada mueve
    la rejilla a un sitio que no es). Si el BPM lo midió el DSP
    (`bpm_del_dsp`), el que se guarda es el del intervalo afinado
    (`bpm_de_la_rejilla`); el de las etiquetas manda su número.

    Va ANTES del tipo de track y del género, que dependen del BPM: con el del
    bin de librosa (129,20 para un 128) un tema podía caer al otro lado del
    rango de BPM de un género.
    """
    from beat_grid import bpm_de_la_rejilla, fit_beat_grid
    first_beat = 0.0
    # Lo peor que puede salir es 60/bpm, no 0.5: ese 0.5 son 120 BPM clavados,
    # un intervalo que no tiene nada que ver con el track y que ademas se
    # exporta tal cual al XML de Rekordbox.
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    try:
        fit = fit_beat_grid(onset_env, fps, bpm)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  [BeatGrid] fallo la fase ({type(e).__name__}): {e}")
        return first_beat, beat_interval, bpm
    if not fit:
        logger.info("  [BeatGrid] sin pulso claro; rejilla sin fase")
        return first_beat, beat_interval, bpm
    first_beat = fit['first_beat']
    beat_interval = fit['beat_interval']
    logger.info(
        f"  [BeatGrid] first_beat={first_beat:.3f}s "
        f"iv={beat_interval:.5f}s downbeat={fit['downbeat_index']} "
        f"conf={fit['confidence']:.2f}"
    )
    if bpm_del_dsp:
        afinado = bpm_de_la_rejilla(bpm, beat_interval)
        if afinado != bpm:
            logger.info(f"  [BPM] de la rejilla: {bpm:.2f} -> {afinado:.2f}")
        bpm = afinado
    return first_beat, beat_interval, bpm
