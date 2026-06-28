"""
Audio analysis helper functions: fingerprint, filename parsing, structure detection, vocals.
"""
import contextlib
import math
import json
import hashlib
import os
import re
import logging
import sys
import numpy as np
import librosa

from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def silence_native_stderr():
    """Silencia el FD 2 (stderr) durante la operacion envuelta.

    libmpg123 / libsndfile / libavcodec escriben sus warnings de MP3
    corruptos directamente al descriptor de stderr (no via logging de
    Python), inundando los logs de Render con lineas tipo
    "[src/libmpg123/parse.c:skip_junk():1276] error: Giving up...".
    Nuestro codigo Python ya maneja los fallos via try/except, asi que
    esos mensajes son ruido. Aqui redirigimos FD 2 a /dev/null solo
    durante la llamada (deberia durar ~1-2s para una carga de audio).

    Si por lo que sea no podemos redirigir, dejamos pasar la operacion
    sin silenciar — perderemos limpieza de logs pero no funcionalidad.
    """
    if os.name == 'nt' or not hasattr(os, 'dup2'):
        yield
        return
    try:
        sys.stderr.flush()
    except Exception:
        pass
    devnull_fd = None
    saved_fd = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
    except OSError:
        if saved_fd is not None:
            try:
                os.close(saved_fd)
            except OSError:
                pass
            saved_fd = None
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass
            devnull_fd = None
        yield
        return
    try:
        yield
    finally:
        try:
            os.dup2(saved_fd, 2)
        except OSError:
            pass
        try:
            os.close(saved_fd)
        except OSError:
            pass
        try:
            os.close(devnull_fd)
        except OSError:
            pass


# ==================== FLOAT SANITIZER ====================

def sanitize_float(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return v


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        return sanitize_float(obj)
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        content = sanitize_for_json(content)
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


# ==================== HELPERS ====================

def calculate_fingerprint(file_path):
    """
    Calcular fingerprint de audio usando Chromaprint (basado en el sonido real).
    Mismo audio = mismo fingerprint, sin importar nombre, formato o bitrate.
    Fallback a MD5 del contenido si Chromaprint no está disponible.

    Returns: (short_id, chromaprint_raw) donde:
      - short_id: MD5 del chromaprint hash (32 chars, compatible con IDs existentes)
      - chromaprint_raw: fingerprint completo de Chromaprint (para matching futuro) o None
    """
    try:
        import acoustid
        duration, fp = acoustid.fingerprint_file(file_path)
        fp_bytes = fp if isinstance(fp, bytes) else fp.encode()
        short_id = hashlib.md5(fp_bytes).hexdigest()
        logger.info(f"[Fingerprint] Chromaprint OK: {short_id} ({duration:.0f}s)")
        return short_id, fp.decode() if isinstance(fp, bytes) else fp
    except Exception as e:
        logger.warning(f"[Fingerprint] Chromaprint falló ({e}), usando MD5 del archivo")
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash, None


def parse_filename(filename: str) -> dict:
    # Quitar la extension. Antes solo cubria mp3|wav|flac|m4a -> para los otros
    # formatos soportados (aac, ogg, aiff/aif, opus, wma) la extension quedaba
    # pegada al titulo ("Domino.aiff"), ensuciando lo que se muestra y empeorando
    # el match en AudD/Discogs. Lista alineada con config.SUPPORTED_FORMATS.
    name = re.sub(
        r'\.(mp3|wav|flac|m4a|aac|ogg|aiff|aif|opus|wma)$',
        '', filename, flags=re.IGNORECASE,
    )
    name = re.sub(r'^\d+[\s\-_.]+', '', name)
    name = re.sub(r'[\[\(][A-Z0-9]+[\]\)]', '', name)
    name = re.sub(r'(?i)\(original mix\)|\[original mix\]', '', name)
    name = re.sub(r'(?i)\(extended\)|\[extended\]', '', name)

    artist = None
    title = None

    if ' - ' in name:
        parts = name.split(' - ', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    elif '-' in name and len(name.split('-')) == 2:
        parts = name.split('-', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    else:
        title = name.strip()

    if artist:
        artist = ' '.join(artist.split())
    if title:
        title = ' '.join(title.split())

    return {'artist': artist, 'title': title if title else name.strip()}


def detect_structure(y, sr, duration):
    segment_length = int(sr * 8)
    n_segments = len(y) // segment_length

    if n_segments == 0:
        return {
            'has_intro': False, 'has_buildup': False, 'has_drop': False,
            'has_breakdown': False, 'has_outro': False, 'sections': []
        }

    energies = []
    for i in range(n_segments):
        segment = y[i*segment_length:(i+1)*segment_length]
        energies.append(np.mean(librosa.feature.rms(y=segment)))

    energies = np.array(energies)
    avg_energy = np.mean(energies)

    has_intro = energies[0] < avg_energy * 0.6 if len(energies) > 0 else False
    has_outro = energies[-1] < avg_energy * 0.6 if len(energies) > 0 else False
    has_drop = np.max(energies) > avg_energy * 1.4 if len(energies) > 0 else False
    has_breakdown = np.min(energies[1:-1]) < avg_energy * 0.5 if len(energies) > 2 else False
    has_buildup = has_drop

    sections = []
    for i, e in enumerate(energies):
        section_type = "body"
        if i == 0 and has_intro:
            section_type = "intro"
        elif i == len(energies)-1 and has_outro:
            section_type = "outro"
        elif e > avg_energy * 1.4:
            section_type = "drop"
        elif e < avg_energy * 0.5:
            section_type = "breakdown"

        sections.append({
            "type": section_type,
            "start": i * 8.0,
            "end": (i + 1) * 8.0,
            "energy": float(e)
        })

    return {
        'has_intro': has_intro, 'has_buildup': has_buildup, 'has_drop': has_drop,
        'has_breakdown': has_breakdown, 'has_outro': has_outro, 'sections': sections
    }


def find_drop_timestamp(y, sr, segments: dict) -> float:
    if not segments['sections']:
        return 30.0

    for section in segments['sections']:
        if section['type'] == 'drop':
            return section['start'] + 4.0

    max_energy = max(s['energy'] for s in segments['sections'])
    for section in segments['sections']:
        if section['energy'] == max_energy:
            return section['start'] + 4.0

    duration = segments['sections'][-1]['end'] if segments['sections'] else 180
    return duration / 3


# NOTA: `classify_track_type` (version string vieja) se borro aqui — estaba
# muerta (no la importaba nadie). La version viva es `main.classify_track_type`
# (Fase 1 v2: devuelve dict con confidence + alternativas), espejada en
# `ChunkedAudioAnalyzer._classify_track_type`. No la recrees en audio_helpers.


def detect_vocals_improved(y, sr, spectral_centroid):
    try:
        centroid_mean = float(np.mean(spectral_centroid))
        has_high_centroid = centroid_mean > 3500

        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = float(np.mean(flatness))
        is_tonal = flatness_mean < 0.15

        centroid_std = float(np.std(spectral_centroid))
        has_variation = centroid_std > 500

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        zcr_in_voice_range = 0.05 < zcr_mean < 0.15

        criteria_met = sum([has_high_centroid, is_tonal, has_variation, zcr_in_voice_range])
        return criteria_met >= 3

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Error detectando vocals: {e}")
        return False


def get_acousticbrainz_genre(fingerprint=None, artist=None, title=None):
    """AcousticBrainz cerro en 2022. Stub que retorna None."""
    return None
