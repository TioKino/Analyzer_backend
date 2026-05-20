"""
Preview snippet generator — creates 6-second MP3 previews from tracks.
"""
import os
import sys
import subprocess
import logging
from typing import Optional
from pathlib import Path as PathLib

logger = logging.getLogger(__name__)


def _preview_timeout_seconds() -> int:
    """Lee PREVIEW_TIMEOUT_SECONDS del env, default 15s.

    En Render Standard con CPU compartida los tracks de 200+ MB pueden
    sobrepasar 15s al pinchar el ffmpeg sin que sea un fallo real —
    permite a operaciones subir el límite via env var sin tocar código.
    """
    raw = os.environ.get('PREVIEW_TIMEOUT_SECONDS')
    if not raw:
        return 15
    try:
        v = int(raw)
        if v <= 0:
            return 15
        return v
    except (TypeError, ValueError):
        return 15


def init_previews_dir(previews_dir: str):
    """Ensure previews directory exists."""
    PathLib(previews_dir).mkdir(parents=True, exist_ok=True)


def generate_preview_snippet(
    file_path: str,
    fingerprint: str,
    drop_timestamp: float,
    duration: float,
    previews_dir: str,
) -> Optional[str]:
    """
    Genera snippet MP3 de 6s desde el punto mas interesante del track.

    Formato: MP3 mono, 64kbps, 22050Hz ~ 48KB por track.
    Incluye fade in (0.3s) y fade out (0.5s) via ffmpeg.
    """
    output_path = os.path.join(previews_dir, f"{fingerprint}.mp3")

    if os.path.exists(output_path):
        logger.debug(f"[Preview] Ya existe snippet para {fingerprint[:8]}...")
        return output_path

    start = max(0, drop_timestamp - 2.0)
    if start + 6 > duration:
        start = max(0, duration - 6)
    if duration < 6:
        start = 0

    try:
        # Usar FFMPEG_BIN absoluto si esta seteado (lo pone local_engine.py).
        # En Windows 11 24H2+ pasar solo 'ffmpeg' hace que CreateProcessW
        # recorra el PATH y dispare WinError 448 si hay reparse points
        # (OneDrive/junctions/symlinks) en alguno de los dirs del PATH.
        ffmpeg_bin = os.environ.get('FFMPEG_BIN', 'ffmpeg')
        cmd = [
            ffmpeg_bin, '-y',
            '-loglevel', 'error',
            '-ss', str(round(start, 2)),
            '-i', file_path,
            # Descartar streams no-audio: artwork embebido (APIC corrupto en
            # MP3s ripeados de fuentes raras), data streams, subtitulos. Sin
            # esto ffmpeg intenta decodificar el artwork y aborta toda la
            # conversion con "Error while decoding stream #0:1: Invalid data
            # found... | Conversion failed!". Visto repetidamente en logs
            # Render: el preview NUNCA se generaba para esos tracks.
            '-vn', '-sn', '-dn',
            '-t', '6',
            '-ac', '1',
            '-ab', '64k',
            '-ar', '22050',
            '-af', 'afade=t=in:st=0:d=0.3,afade=t=out:st=5.5:d=0.5',
            output_path
        ]

        kwargs = {
            'capture_output': True,
            'timeout': _preview_timeout_seconds(),
            'check': True,
        }
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        subprocess.run(cmd, **kwargs)

        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            if size > 1000:
                logger.info(f"[Preview] Snippet generado: {fingerprint[:8]}... "
                      f"({size//1024}KB, start={start:.1f}s)")
                return output_path
            else:
                logger.warning(f"[Preview] Snippet demasiado pequeno ({size}B), eliminando")
                os.unlink(output_path)
                return None

        return None

    except subprocess.TimeoutExpired:
        logger.warning(f"[Preview] Timeout generando snippet para {fingerprint[:8]}...")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None
    except subprocess.CalledProcessError as e:
        # ffmpeg vuelca su banner de version y headers de input al stderr
        # antes del mensaje real de error. Antes haciamos stderr[:200],
        # lo que solo recogia el banner ("ffmpeg version 5.1.8...") y
        # disparaba alertas falsas. Filtramos por lineas que parecen
        # mensajes de error reales y caemos al tail si no hay match.
        stderr_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else ''
        error_lines = [
            ln for ln in stderr_msg.splitlines()
            if ln and (
                ln.lower().startswith('error')
                or 'invalid' in ln.lower()
                or 'no such file' in ln.lower()
                or 'permission denied' in ln.lower()
                or 'failed' in ln.lower()
            )
        ]
        if error_lines:
            real_err = ' | '.join(error_lines[-3:])[:400]
        else:
            real_err = stderr_msg[-400:].strip() or 'unknown'
        logger.error(f"[Preview] ffmpeg exit {e.returncode}: {real_err}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None
    except (FileNotFoundError, IOError, OSError) as e:
        logger.error(f"[Preview] Error generando snippet: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None
