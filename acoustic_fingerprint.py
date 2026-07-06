"""
acoustic_fingerprint.py — Huella acustica Chromaprint + matching tolerante.

CORAZON de la memoria colectiva cross-version.

Hoy la memoria colectiva (cues, beat-grid, correcciones, ratings, popularidad)
se agrupa por MD5 del CONTENIDO del archivo (bit-exact). Dos copias del mismo
track con distinto codec / bitrate / tag ID3 caen en claves distintas y NO
comparten memoria. Eso rompe la promesa central del producto: la memoria
colectiva ENTRE usuarios, que rara vez tienen el archivo byte-identico.

Este modulo agrupa por SONIDO. El fingerprint Chromaprint (`fpcalc -raw`) es un
array de subfingerprints uint32:
  - idempotente a filename / tags / container (mismo audio, mismo codec, otro
    tag ID3 -> fingerprint IDENTICO), y
  - ~99.8% identico cross-codec (mismo audio mp3 vs flac -> Hamming ~0.0001,
    verificado empiricamente 2026-05-04; ver CLAUDE.md).

Dos tracks cuyo fingerprint esta dentro de un umbral Hamming pertenecen al
mismo "cluster acustico" y comparten memoria colectiva.

TESTABILIDAD: todo aqui (salvo `compute_raw_chromaprint`) opera sobre arrays de
int, asi que se valida sin fpcalc ni audio real. `compute_raw_chromaprint` es
I/O sobre el binario `fpcalc` (libchromaprint-tools, ya en el Aptfile de Render;
el cliente desktop usa el mismo binario) y se valida en Render/staging.
"""
import base64
import hashlib
import json
import logging
import os
import struct
import subprocess

logger = logging.getLogger(__name__)

# Umbral de distancia Hamming normalizada (0..1) bajo el cual dos fingerprints
# se consideran el MISMO audio. El mismo audio re-encoded a otro codec mide
# ~0.0001 (CLAUDE.md 2026-05-04); audio DISTINTO mide ~0.4-0.5 (aleatorio =
# 0.5). 0.15 deja un margen amplio y seguro entre ambos regimenes.
MATCH_THRESHOLD = 0.15

# Posiciones de desplazamiento a probar al alinear dos fingerprints. Compensa
# el encoder-delay / padding (LAME, boundaries del decoder) que corre el array
# unas pocas posiciones sin cambiar el audio. Cada shift es O(n); barrido barato.
_MAX_ALIGN_SHIFT = 3

# Minimo de subfingerprints solapados para que una comparacion sea fiable. Por
# debajo (tracks muy cortos, fingerprints truncados) no arriesgamos un match
# espurio: se devuelve distancia maxima.
_MIN_OVERLAP = 8


def compute_raw_chromaprint(file_path, timeout=30):
    """Extrae el fingerprint Chromaprint crudo con `fpcalc -raw -json`.

    Devuelve una lista de int (subfingerprints uint32) o None si fpcalc no esta
    disponible / falla. Mismo binario que usa el cliente desktop, asi que el
    array es consistente entre backend y cliente.

    Resuelve el binario via FPCALC_BIN (ruta absoluta que setea local_engine.py
    desde el bundle PyInstaller, igual que FFMPEG_BIN) y cae a 'fpcalc' en PATH
    (Render lo trae en el Aptfile: libchromaprint-tools).
    """
    fpcalc_bin = os.environ.get('FPCALC_BIN', 'fpcalc')
    try:
        out = subprocess.run(
            [fpcalc_bin, '-raw', '-json', file_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if out.returncode != 0:
            logger.warning(
                f"[Acoustic] fpcalc exit {out.returncode}: {(out.stderr or '')[:200]}"
            )
            return None
        data = json.loads(out.stdout)
        fp = data.get('fingerprint')
        if isinstance(fp, list) and fp:
            return [int(x) & 0xFFFFFFFF for x in fp]
        return None
    except FileNotFoundError:
        logger.warning("[Acoustic] fpcalc no instalado; sin huella acustica")
        return None
    except Exception as e:  # noqa: BLE001 - best-effort, nunca romper /analyze
        logger.warning(f"[Acoustic] fpcalc fallo: {e}")
        return None


def encode_raw(ints):
    """Serializa el array de subfingerprints a string compacto (base64 de
    uint32 big-endian) para persistir en `tracks.chromaprint`. None si vacio."""
    if not ints:
        return None
    packed = struct.pack(f'>{len(ints)}I', *[i & 0xFFFFFFFF for i in ints])
    return base64.b64encode(packed).decode('ascii')


def decode_raw(s):
    """Inverso de encode_raw. Devuelve lista de int, o [] si es None/invalido."""
    if not s:
        return []
    try:
        packed = base64.b64decode(s)
        n = len(packed) // 4
        if n == 0:
            return []
        return list(struct.unpack(f'>{n}I', packed[:n * 4]))
    except Exception:  # noqa: BLE001 - dato corrupto en BD no debe romper nada
        return []


def acoustic_key(ints):
    """Clave EXACTA del fingerprint: MD5 hex del array empaquetado.

    Dos fingerprints identicos (mismo audio + mismo codec, tags/filename
    distintos) dan la MISMA key -> dedup O(1) por indice, sin barrido Hamming.
    Es el 'nuevo cluster id' cuando un audio se ve por primera vez.
    """
    if not ints:
        return None
    packed = struct.pack(f'>{len(ints)}I', *[i & 0xFFFFFFFF for i in ints])
    return hashlib.md5(packed).hexdigest()


def hamming_distance(fp_a, fp_b):
    """Distancia Hamming NORMALIZADA (0..1) entre dos fingerprints Chromaprint.

    Alinea por la longitud minima y prueba pequenos desplazamientos (encoder
    delay), quedandose con el minimo. Devuelve 1.0 si no hay solape suficiente
    (`_MIN_OVERLAP`) para comparar con fiabilidad.
    """
    if not fp_a or not fp_b:
        return 1.0
    best = 1.0
    for shift in range(-_MAX_ALIGN_SHIFT, _MAX_ALIGN_SHIFT + 1):
        if shift >= 0:
            a, b = fp_a[shift:], fp_b
        else:
            a, b = fp_a, fp_b[-shift:]
        n = min(len(a), len(b))
        if n < _MIN_OVERLAP:
            continue
        diff_bits = 0
        for i in range(n):
            diff_bits += bin((a[i] ^ b[i]) & 0xFFFFFFFF).count('1')
        dist = diff_bits / (32.0 * n)
        if dist < best:
            best = dist
    return best


def fingerprints_match(fp_a, fp_b, threshold=MATCH_THRESHOLD):
    """True si dos fingerprints son el MISMO audio (Hamming < threshold)."""
    return hamming_distance(fp_a, fp_b) < threshold
