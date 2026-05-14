"""
DJ ANALYZER - AudD Auto-Trigger
================================
Logica para invocar AudD automaticamente desde /analyze cuando los
metadatos ID3 son insuficientes y la cascada externa
(Beatport/Discogs/MusicBrainz/iTunes) no puede arrancar por falta de
artist+title.

Disparador (proactive, antes de la cascada externa):
    1. artist o title vacios tras ID3 + filename parse
    2. metadatos basura ("Unknown", "Track 01", etc.)

Presupuesto:
    - Hard cap diario configurable (AUDD_DAILY_CAP, default 50)
    - Cooldown por fingerprint (AUDD_COOLDOWN_DAYS, default 7)
    - Skip tracks fuera de [AUDD_MIN_DURATION, AUDD_MAX_DURATION]

El modulo no toca el endpoint /identify ni /recognize. Esos siguen
exponiendo AudD de forma manual y no consumen del mismo presupuesto.
"""
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

import requests

logger = logging.getLogger(__name__)


GARBAGE_ARTIST_PATTERNS = (
    r'^unknown(\s+artist)?$',
    r'^various(\s+artists)?$',
    r'^v\.?\s*a\.?$',
    r'^artist$',
    r'^n\.?\s*a\.?$',
    r'^\d+$',
)

GARBAGE_TITLE_PATTERNS = (
    r'^track\s*\d+$',
    r'^untitled.*$',
    r'^audio\s*\d+$',
    r'^pista\s*\d+$',
    r'^\d+$',
    r'^.{0,2}$',
)


def _matches_any(value: str, patterns) -> bool:
    val = (value or '').strip().lower()
    if not val:
        return True
    return any(re.match(p, val, re.IGNORECASE) for p in patterns)


def is_garbage_metadata(artist: Optional[str], title: Optional[str]) -> bool:
    """True si artist o title estan ausentes o son evidentemente inservibles."""
    if not artist or not title:
        return True
    if _matches_any(artist, GARBAGE_ARTIST_PATTERNS):
        return True
    if _matches_any(title, GARBAGE_TITLE_PATTERNS):
        return True
    return False


def should_trigger_audd(
    artist: Optional[str],
    title: Optional[str],
    duration: float,
    fingerprint: Optional[str],
    db,
    *,
    min_duration: float = 30.0,
    max_duration: float = 720.0,
    daily_cap: int = 50,
    cooldown_days: int = 7,
    force: bool = False,
) -> Tuple[bool, str]:
    """Decide si AudD debe dispararse para este track.

    Con force=True (usuario pidio explicitamente "limpiar con AudD" desde la
    UI) se saltea el check de metadata-utilizable y el cooldown por
    fingerprint. El daily cap y los limites de duracion SE RESPETAN para no
    quemar cuota ni mandar fragmentos invalidos.

    Returns: (should_fire, reason)
    """
    if duration and duration < min_duration:
        return False, f"duracion<{min_duration}s"
    if duration and duration > max_duration:
        return False, f"duracion>{max_duration}s"

    if not force and not is_garbage_metadata(artist, title):
        return False, "metadata utilizable"

    if not force and fingerprint and db is not None:
        try:
            last_call = db.get_last_audd_call(fingerprint)
            if last_call is not None:
                elapsed_days = (datetime.now(timezone.utc).timestamp() - last_call) / 86400
                if elapsed_days < cooldown_days:
                    return False, f"cooldown ({elapsed_days:.1f}d/{cooldown_days}d)"
        except Exception as e:
            logger.warning(f"[AudD-auto] cooldown check fallo: {e}")

    if db is not None:
        try:
            today_count = db.count_audd_calls_today()
            if today_count >= daily_cap:
                return False, f"daily cap ({today_count}/{daily_cap})"
        except Exception as e:
            logger.warning(f"[AudD-auto] cap check fallo: {e}")

    return True, "force manual" if force else "metadata insuficiente"


def call_audd(file_path: str, api_token: str, timeout: int = 30) -> Optional[Dict]:
    """Llama a AudD con un fragmento de 20s desde 0:30 del track.

    Devuelve track_data crudo o None si no se identifica.
    """
    if not api_token:
        # Antes silencioso — el operador podia tener AUDD_AUTO_ENABLED=true
        # y este module saltando sin que apareciera nada en logs. WARNING
        # visible para diagnosticar configuracion incompleta en Render.
        logger.warning(
            "[AudD-auto] disabled — AUDD_API_TOKEN missing. "
            "Configurar en env vars o desactivar AUDD_AUTO_ENABLED."
        )
        return None

    fragment_path = None
    try:
        import librosa
        import soundfile as sf

        # Fragmento de 20s desde 0:30 (mismo patron que /identify)
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=20, offset=30)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            fragment_path = tmp.name
        sf.write(fragment_path, y, sr)

        with open(fragment_path, 'rb') as audio_file:
            response = requests.post(
                'https://api.audd.io/',
                data={
                    'api_token': api_token,
                    'return': 'spotify,deezer,apple_music,musicbrainz',
                },
                files={'file': audio_file},
                timeout=timeout,
            )

        if response.status_code != 200:
            logger.warning(f"[AudD-auto] HTTP {response.status_code}")
            return None

        data = response.json()
        if data.get('status') != 'success':
            err = data.get('error', {}).get('error_message', 'unknown')
            logger.info(f"[AudD-auto] API: {err}")
            return None

        track = data.get('result')
        if not track:
            logger.info("[AudD-auto] no match")
        return track

    except (ImportError, OSError, requests.RequestException, ValueError) as e:
        logger.error(f"[AudD-auto] error: {e}")
        return None
    finally:
        if fragment_path and os.path.exists(fragment_path):
            try:
                os.unlink(fragment_path)
            except OSError:
                pass


def enrich_with_audd_if_needed(
    file_path: str,
    fingerprint: Optional[str],
    duration: float,
    artist: Optional[str],
    title: Optional[str],
    api_token: str,
    db,
    *,
    min_duration: float = 30.0,
    max_duration: float = 720.0,
    daily_cap: int = 50,
    cooldown_days: int = 7,
    force: bool = False,
) -> Optional[Dict]:
    """Si el trigger lo permite, llama a AudD y devuelve track_data crudo.

    Tambien registra cada intento (exito o fallo) en la BD para honrar el
    cooldown y el daily cap.

    force=True salta el cooldown por fingerprint y el check de garbage
    metadata. Lo usa el endpoint `/analyze?force_audd=true` cuando el usuario
    pide "limpiar con AudD" desde la UI con un track ya analizado. El daily
    cap y los limites de duracion siguen respetandose.

    Returns: track_data dict (con artist/title/label/release_date/...) o None.
    """
    if not api_token:
        return None

    should, reason = should_trigger_audd(
        artist, title, duration, fingerprint, db,
        min_duration=min_duration, max_duration=max_duration,
        daily_cap=daily_cap, cooldown_days=cooldown_days,
        force=force,
    )
    if not should:
        logger.debug(f"[AudD-auto] skip: {reason}")
        return None

    logger.info(f"[AudD-auto] disparando: {reason}")
    track_data = call_audd(file_path, api_token)

    if db is not None and fingerprint:
        success = bool(track_data and track_data.get('artist') and track_data.get('title'))
        try:
            db.log_audd_call(
                fingerprint, success,
                artist=(track_data or {}).get('artist'),
                title=(track_data or {}).get('title'),
            )
        except Exception as e:
            logger.warning(f"[AudD-auto] log_audd_call fallo: {e}")

    if not track_data:
        return None
    if not (track_data.get('artist') and track_data.get('title')):
        return None

    logger.info(f"[AudD-auto] identificado: {track_data['artist']} - {track_data['title']}")
    return track_data
