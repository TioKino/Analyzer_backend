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
I/O sobre el binario `fpcalc`. OJO (2026-07-07): Debian de Render NO ofrece
`libchromaprint-tools` en apt, asi que el binario se AUTO-DESCARGA (estatico
oficial de acoustid) a `/data` al primer uso — ver `ensure_fpcalc`. El motor
local (Mac/PC) sigue usando el bundle PyInstaller via FPCALC_BIN.
"""
import base64
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import tarfile

logger = logging.getLogger(__name__)

# ============================================================================
# Auto-provisión de fpcalc (Render no tiene libchromaprint-tools en apt)
# ============================================================================
# Debian de Render NO ofrece el paquete `libchromaprint-tools` (verificado
# 2026-07-07: apt "Unable to locate package"), así que `fpcalc` no llega por el
# Aptfile aunque `libchromaprint1` sí (dependencia de ffmpeg). Sin `fpcalc` NO
# hay huella acústica → la memoria colectiva por sonido queda muerta.
#
# Solución sin tocar el dashboard: al primer uso, si `fpcalc` no está en
# PATH/FPCALC_BIN, descargamos el binario ESTÁTICO oficial de acoustid (bundlea
# ffmpeg, corre en cualquier x86_64 linux) a un directorio persistente. En
# Render `/data` sobrevive a deploys → se descarga UNA vez en la vida del disco.
# Best-effort y memoizado: si la descarga falla, la huella simplemente se salta
# (como antes) y se reintenta en el próximo arranque.
_FPCALC_VERSION = os.environ.get("FPCALC_VERSION", "1.5.1")
_FPCALC_URL = os.environ.get(
    "FPCALC_URL",
    f"https://github.com/acoustid/chromaprint/releases/download/"
    f"v{_FPCALC_VERSION}/chromaprint-fpcalc-{_FPCALC_VERSION}-linux-x86_64.tar.gz",
)
# /data = disco persistente de Render; overridable para tests / otros entornos.
_FPCALC_CACHE_DIR = os.environ.get("FPCALC_CACHE_DIR", "/data/bin")
_FPCALC_AUTODOWNLOAD = os.environ.get("FPCALC_AUTODOWNLOAD", "1") != "0"

_fpcalc_path = None       # ruta resuelta (cache)
_fpcalc_resolved = False  # ya intentamos resolver en este proceso


def _resolve_existing_fpcalc():
    """Busca fpcalc SIN descargar: FPCALC_BIN explícito, PATH, o cache previa."""
    env = os.environ.get("FPCALC_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    found = shutil.which("fpcalc")
    if found:
        return found
    cached = os.path.join(_FPCALC_CACHE_DIR, "fpcalc")
    if os.path.isfile(cached) and os.access(cached, os.X_OK):
        return cached
    return None


def _download_fpcalc():
    """Descarga el fpcalc estático a _FPCALC_CACHE_DIR (fallback /tmp si /data no
    es escribible). Devuelve la ruta o None. Best-effort, nunca lanza."""
    global _FPCALC_CACHE_DIR
    import requests  # import local: solo se necesita en el arranque en Render

    cache_dir = _FPCALC_CACHE_DIR
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        cache_dir = "/tmp/dja_fpcalc"
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Acoustic] no pude crear cache de fpcalc: {e}")
            return None
    _FPCALC_CACHE_DIR = cache_dir
    dest = os.path.join(cache_dir, "fpcalc")
    tmp_tar = os.path.join(cache_dir, "fpcalc.tar.gz")
    try:
        logger.info(f"[Acoustic] fpcalc ausente; descargando estático de {_FPCALC_URL}")
        resp = requests.get(_FPCALC_URL, timeout=60, stream=True)
        resp.raise_for_status()
        with open(tmp_tar, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        with tarfile.open(tmp_tar, "r:gz") as tf:
            member = next(
                (m for m in tf.getmembers()
                 if m.name == "fpcalc" or m.name.endswith("/fpcalc")),
                None,
            )
            if member is None:
                logger.warning("[Acoustic] el tar de fpcalc no contiene el binario")
                return None
            member.name = "fpcalc"  # aplanar (sin subdirectorio)
            tf.extract(member, cache_dir)
        os.chmod(dest, 0o755)
        try:
            os.remove(tmp_tar)
        except Exception:  # noqa: BLE001
            pass
        smoke = subprocess.run(
            [dest, "-version"], capture_output=True, text=True, timeout=10,
        )
        if smoke.returncode == 0:
            logger.info(
                f"[Acoustic] fpcalc descargado OK "
                f"({(smoke.stdout or smoke.stderr).strip()[:60]})"
            )
            return dest
        logger.warning(f"[Acoustic] fpcalc descargado pero no ejecuta (exit {smoke.returncode})")
        return None
    except Exception as e:  # noqa: BLE001 - nunca romper /analyze por esto
        logger.warning(f"[Acoustic] descarga de fpcalc falló: {e}")
        return None


def ensure_fpcalc():
    """Ruta al binario fpcalc, auto-descargándolo si hace falta. Memoizado por
    proceso. Devuelve None si no hay forma de tenerlo (la huella se salta)."""
    global _fpcalc_path, _fpcalc_resolved
    if _fpcalc_resolved:
        return _fpcalc_path
    path = _resolve_existing_fpcalc()
    if path is None and _FPCALC_AUTODOWNLOAD:
        path = _download_fpcalc()
    _fpcalc_path = path
    _fpcalc_resolved = True
    if path:
        logger.info(f"[Acoustic] fpcalc en uso: {path}")
    return path

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

    Resuelve el binario via ensure_fpcalc(): FPCALC_BIN explícito (motor local,
    bundle PyInstaller), 'fpcalc' en PATH, o binario estático auto-descargado a
    disco persistente (Render no ofrece libchromaprint-tools en apt — ver
    ensure_fpcalc). None si no hay forma de tenerlo.
    """
    fpcalc_bin = ensure_fpcalc()
    if not fpcalc_bin:
        logger.warning("[Acoustic] fpcalc no disponible; sin huella acustica")
        return None
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
