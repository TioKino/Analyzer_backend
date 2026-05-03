"""Chromaprint fingerprinting helper.

Calcula fingerprints acusticos via libchromaprint (binario `fpcalc`).
Idempotente al filename, tags ID3 y re-codec ligero (mp3 320 vs flac
producen el MISMO fingerprint cuando es la misma grabacion).

Decision arquitectura PLAN_CHROMAPRINT.md (validada 2026-05-03):
- Chromaprint LOCAL only — sin lookup AcoustID externo (no necesitamos
  enrichment, basta dedup interno).
- Fingerprint exportado = MD5 hex del array de int32 que devuelve
  Chromaprint. 32 chars, mismo formato que la columna `fingerprint`
  legacy (MD5 contenido) → migracion drop-in.
- El array completo se serializa aparte en base64 por si en el futuro
  queremos matching tolerante (>95% similar) en vez de exact match.

Fallback: si fpcalc no esta instalado o el archivo no es decodificable,
los callers deben capturar `ChromaprintUnavailable` / `ChromaprintFailed`
y caer al MD5 legacy del contenido. El track se marca con
`fingerprint_source='md5_legacy'` para reprocesarlo cuando fpcalc este
disponible.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import shutil
from typing import Tuple

logger = logging.getLogger(__name__)


class ChromaprintUnavailable(RuntimeError):
    """fpcalc binario no disponible en el sistema (Aptfile no instalo libchromaprint-tools, etc)."""


class ChromaprintFailed(RuntimeError):
    """fpcalc se ejecuto pero fallo (archivo corrupto, formato no soportado, <30s, etc)."""


_FPCALC_AVAILABLE: bool | None = None


def is_fpcalc_available() -> bool:
    """Cachea si fpcalc esta accesible. La verificacion se hace solo una vez."""
    global _FPCALC_AVAILABLE
    if _FPCALC_AVAILABLE is None:
        _FPCALC_AVAILABLE = shutil.which("fpcalc") is not None
        if not _FPCALC_AVAILABLE:
            logger.warning(
                "fpcalc no encontrado en PATH. Chromaprint deshabilitado, "
                "se usara MD5 legacy para todos los tracks."
            )
    return _FPCALC_AVAILABLE


def calculate_chromaprint_fingerprint(file_path: str) -> Tuple[str, str, int]:
    """Calcula fingerprint Chromaprint de un audio file.

    Args:
        file_path: ruta al archivo de audio (formato cualquiera que ffmpeg
            pueda decodificar — mp3, flac, wav, m4a, opus, etc).

    Returns:
        (fp_md5, fp_array_b64, duration_ms): tupla con
        - fp_md5: MD5 hex del array Chromaprint (32 chars). Es nuestro
          identificador estable cross-device.
        - fp_array_b64: array de int32 serializado en base64 (matching
          tolerante futuro, exact-match-only por ahora).
        - duration_ms: duracion del audio en milisegundos segun fpcalc.

    Raises:
        ChromaprintUnavailable: fpcalc binario no disponible.
        ChromaprintFailed: el archivo no es decodificable o demasiado corto
            para Chromaprint (<30s tipicamente).
    """
    if not is_fpcalc_available():
        raise ChromaprintUnavailable("fpcalc binario no encontrado en PATH")

    try:
        import acoustid
    except ImportError as exc:
        raise ChromaprintUnavailable(
            f"pyacoustid no instalado correctamente: {exc}"
        ) from exc

    try:
        # acoustid.fingerprint_file llama a fpcalc internamente.
        # Devuelve (duration_seconds, fingerprint_compressed_str).
        # Usamos chromaprint.decode_fingerprint para obtener el array crudo.
        duration_s, fp_compressed = acoustid.fingerprint_file(file_path)
    except acoustid.FingerprintGenerationError as exc:
        raise ChromaprintFailed(
            f"fpcalc fallo procesando {file_path}: {exc}"
        ) from exc
    except Exception as exc:  # pyacoustid puede lanzar otros tipos
        raise ChromaprintFailed(
            f"Error inesperado en fpcalc para {file_path}: {exc}"
        ) from exc

    if not fp_compressed:
        raise ChromaprintFailed(
            f"fpcalc devolvio fingerprint vacio para {file_path} "
            "(audio probablemente <30s o corrupto)"
        )

    # `fp_compressed` es un str con la representacion compacta de Chromaprint.
    # Para tener el array de int32 hace falta `chromaprint.decode_fingerprint`.
    try:
        import chromaprint as cp
        # decode_fingerprint devuelve (raw_array, version)
        raw_array, _version = cp.decode_fingerprint(fp_compressed.encode("ascii"))
    except Exception as exc:
        # Fallback: si el modulo chromaprint no expone decode_fingerprint
        # (varia por version), usamos el str compactado directamente.
        # MD5 sobre la cadena compactada sigue siendo determinista cross-device.
        logger.debug(
            "chromaprint.decode_fingerprint no disponible (%s), "
            "usando fingerprint compactado como fuente del MD5",
            exc,
        )
        fp_bytes = fp_compressed.encode("ascii") if isinstance(fp_compressed, str) else fp_compressed
        fp_md5 = hashlib.md5(fp_bytes).hexdigest()
        fp_array_b64 = base64.b64encode(fp_bytes).decode("ascii")
        return fp_md5, fp_array_b64, int(duration_s * 1000)

    # Serializar el array de int32 a bytes (little-endian, signed).
    fp_bytes = b"".join(int(x).to_bytes(4, "little", signed=True) for x in raw_array)
    fp_md5 = hashlib.md5(fp_bytes).hexdigest()
    fp_array_b64 = base64.b64encode(fp_bytes).decode("ascii")
    return fp_md5, fp_array_b64, int(duration_s * 1000)
