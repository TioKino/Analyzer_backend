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
            # FAIL-CLOSED: ambos checks (cooldown y cap) son guards de GASTO —
            # AudD cobra por llamada. Si no podemos VERIFICAR que estamos dentro
            # de limites, NO disparamos. Antes el except solo logueaba y caia al
            # `return True` final, de modo que un fallo transitorio de la BD (lock
            # / timeout bajo carga) saltaba el control de coste y disparaba AudD
            # sin tope. El path auto reintenta este track en el siguiente
            # /analyze; force (peticion manual del usuario) ni pasa por aqui.
            logger.warning(f"[AudD-auto] cooldown check fallo, fail-closed: {e}")
            return False, "cooldown check error"

    if db is not None:
        try:
            today_count = db.count_audd_calls_today()
            if today_count >= daily_cap:
                return False, f"daily cap ({today_count}/{daily_cap})"
        except Exception as e:
            # FAIL-CLOSED (ver nota arriba): un fallo al leer el contador NO debe
            # conceder permiso para gastar. Skipear una identificacion es barato;
            # reventar el presupuesto de AudD no.
            logger.warning(f"[AudD-auto] cap check fallo, fail-closed: {e}")
            return False, "cap check error"

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
        try:
            from audio_helpers import silence_native_stderr
        except ImportError:
            import contextlib
            @contextlib.contextmanager
            def silence_native_stderr():
                yield

        # Fragmento de 20s desde 0:30 (mismo patron que /identify).
        # Offset adaptativo: en tracks cortos (<33s) un offset=30 hace que
        # librosa intente leer un rango que empieza pasado el EOF y revienta
        # con "ValueError: negative dimensions are not allowed". Calculamos la
        # duracion primero (barato, lee metadata) y caemos a offset=0 cuando el
        # track no llega — asi los tracks cortos tambien se identifican via AudD
        # en vez de saltarse silenciosamente.
        offset = 30.0
        try:
            total_dur = librosa.get_duration(path=file_path)
        except Exception:
            total_dur = 0.0
        # Necesitamos al menos 3s tras el offset (mismo umbral que el check de
        # abajo). Si no llega, empezamos desde el principio.
        if total_dur and total_dur < offset + 3.0:
            offset = 0.0

        with silence_native_stderr():
            y, sr = librosa.load(
                file_path, sr=22050, mono=True, duration=20, offset=offset
            )

        # Validar que el fragmento sirve antes de gastar cuota de AudD.
        # AudD devuelve "Recognition failed: there's been a problem with
        # creating an audio fingerprint" cuando el audio es muy corto,
        # silencioso o esta corrupto — facil de detectar localmente y
        # ahorrarnos la llamada (era ~30% de las llamadas en una semana).
        try:
            import numpy as _np
            duration_loaded = len(y) / sr if sr else 0
            rms = float(_np.sqrt(_np.mean(_np.square(y)))) if len(y) else 0.0
            if duration_loaded < 3.0:
                logger.info(f"[AudD-auto] skip: fragmento muy corto ({duration_loaded:.1f}s)")
                return None
            if rms < 0.002:
                logger.info(f"[AudD-auto] skip: fragmento silencioso (rms={rms:.4f})")
                return None
        except Exception:
            # Si el check falla por algun motivo raro, seguimos a AudD
            # — no queremos bloquear identificacion por un bug aqui.
            pass

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

        try:
            data = response.json()
        except ValueError as e:
            logger.warning(f"[AudD-auto] respuesta no es JSON: {e}")
            return None
        if data.get('status') != 'success':
            err = data.get('error', {}).get('error_message', 'unknown')
            # Truncamos: AudD devuelve mensajes de ~500 chars con links.
            logger.info(f"[AudD-auto] API rechazo: {(err or '').split('.')[0][:120]}")
            return None

        track = data.get('result')
        if not track:
            logger.info("[AudD-auto] no match")
        return track

    except (ImportError, OSError, requests.RequestException, ValueError) as e:
        # Timeouts y errores de red contra api.audd.io son esperados (servicio
        # externo, no bug nuestro). Mantenemos el log para diagnostico pero
        # como warning para no disparar alertas en el panel admin.
        logger.warning(f"[AudD-auto] {type(e).__name__}: {e}")
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


def _extract_artwork_url_from_audd(track_data: Dict) -> Optional[Tuple[str, str]]:
    """Extrae la URL de portada del payload AudD (apple_music/deezer/spotify).

    AudD ya identifico el track EXACTO, asi que esta portada es la oficial del
    release — mas fiable que volver a buscar por texto en iTunes/Deezer. No
    cuesta cuota extra: la respuesta ya venia con `return=apple_music,deezer,...`.

    Prioridad por resolucion: Apple Music (template hasta 1000x1000) > Deezer
    cover_xl (1000x1000) > Spotify (640x640).

    Returns: (url, source) o None.
    """
    if not track_data:
        return None

    # 1. Apple Music — la URL es un template con {w}x{h}; pedimos 1000x1000.
    apple = track_data.get('apple_music') or {}
    artwork = apple.get('artwork') or {}
    apple_url = artwork.get('url')
    if apple_url and '{w}' in apple_url and '{h}' in apple_url:
        return apple_url.replace('{w}', '1000').replace('{h}', '1000'), 'apple_music'
    if apple_url:
        return apple_url, 'apple_music'

    # 2. Deezer — cover_xl es 1000x1000.
    deezer = track_data.get('deezer') or {}
    deezer_album = deezer.get('album') or {}
    deezer_url = deezer_album.get('cover_xl') or deezer_album.get('cover_big')
    if deezer_url:
        return deezer_url, 'deezer'

    # 3. Spotify — la primera imagen es la mas grande (640x640).
    spotify = track_data.get('spotify') or {}
    spotify_album = spotify.get('album') or {}
    images = spotify_album.get('images') or []
    if images and isinstance(images, list):
        img_url = images[0].get('url') if isinstance(images[0], dict) else None
        if img_url:
            return img_url, 'spotify'

    return None


def download_artwork_from_audd(track_data: Dict, timeout: int = 8) -> Optional[Dict]:
    """Descarga la portada exacta que AudD devolvio para el track identificado.

    Devuelve el mismo formato que `search_artwork_online`:
        {'url', 'data' (bytes), 'mime_type', 'size', 'source'} o None.

    El `source` lleva sufijo `_audd` (ej. 'apple_music_audd') para distinguir
    en logs/panel admin que vino del match exacto de AudD, no de una busqueda
    por texto.
    """
    extracted = _extract_artwork_url_from_audd(track_data)
    if not extracted:
        return None
    url, source = extracted

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 10000:
            mime = resp.headers.get('Content-Type', 'image/jpeg').split(';')[0].strip()
            if not mime.startswith('image/'):
                mime = 'image/jpeg'
            logger.info(
                f"[AudD-auto] artwork {source}: {len(resp.content)} bytes (match exacto)"
            )
            return {
                'url': url,
                'data': resp.content,
                'mime_type': mime,
                'size': len(resp.content),
                'source': f'{source}_audd',
            }
        logger.debug(
            f"[AudD-auto] artwork {source} descarga insuficiente "
            f"(HTTP {resp.status_code}, {len(resp.content)}b)"
        )
    except requests.RequestException as e:
        logger.warning(f"[AudD-auto] artwork download {type(e).__name__}: {e}")

    return None
