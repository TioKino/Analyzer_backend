"""Un análisis FALLIDO de `/analyze` y cuándo se vuelve a intentar (2026-10-14).

Cuando librosa (y ffmpeg detrás) no puede con un fichero, `/analyze` guarda un
fallback: BPM 0, sin tonalidad y `analysis_status: 'failed'`. Hasta el
2026-10-14 ese fallback era para siempre: el pre-check lo daba por analizado
(solo el motor local exige datos, `con_datos`), así que la app ni volvía a
subir el fichero; y si lo subía, el cache-hit de `/analyze` le devolvía el
fallo. Un tema que falló por un error NUESTRO (memoria, un bug del DSP) se
quedaba a BPM 0 aunque el error se arreglara.

Reintentar NO gasta AudD: en los dos caminos de `/analyze` el audio se lee
antes que AudD, y un fichero ilegible falla ahí. Lo que cuesta es volver a
subir el fichero entero en cada reimportación. Por eso:

- **`ilegible`** (ffmpeg no lo abre, o decodifica a una señal vacía): el
  fichero está roto y subirlo otra vez no lo arregla. No se reintenta.
- **`error`** (cualquier otro fallo): se reintenta como mucho
  `MAX_REINTENTOS` veces, una por cada vez que se vuelva a subir.
- **Sin motivo** (los fallbacks de antes de este cambio): uno, para saber por
  qué fallaron. Si vuelven a fallar, ya guardan el motivo.

Lo usan el pre-check (`routes/analysis_artwork.py`) y los dos atajos de caché
de `/analyze` (por nombre y por huella), y el fallback apunta el motivo y los
intentos. Lo ata `test_el_analisis_fallido_se_reintenta.py`.
"""

import json
from typing import Optional

ILEGIBLE = 'ilegible'
ERROR = 'error'

# Reintentos de un fallo que no es del fichero. Cada uno es una subida entera.
MAX_REINTENTOS = 3


def _analisis(fila) -> dict:
    """El `analysis_json` de una fila, como dict (o el dict mismo)."""
    raw = fila.get('analysis_json') if isinstance(fila, dict) else None
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        datos = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return datos if isinstance(datos, dict) else {}


def fallo_de(fila) -> Optional[dict]:
    """`{'motivo': 'ilegible' | 'error' | None, 'intentos': n}` si la fila es
    el fallback de un análisis fallido; `None` si no lo es. `intentos` cuenta
    los intentos que fallaron (un fallback de antes de esto cuenta uno)."""
    if not isinstance(fila, dict) or not fila:
        return None
    datos = _analisis(fila)

    def campo(nombre):
        valor = fila.get(nombre)
        return valor if valor is not None else datos.get(nombre)

    if campo('analysis_status') != 'failed':
        return None
    motivo = campo('fallo')
    try:
        intentos = int(campo('intentos_fallidos') or 1)
    except (TypeError, ValueError):
        intentos = 1
    return {
        'motivo': motivo if motivo in (ILEGIBLE, ERROR) else None,
        'intentos': max(1, intentos),
    }


def se_reintenta(fila) -> bool:
    """¿Este fallback se vuelve a intentar la próxima vez que llegue el
    fichero? `False` también para lo que no es un fallback."""
    fallo = fallo_de(fila)
    if fallo is None or fallo['motivo'] == ILEGIBLE:
        return False
    if fallo['motivo'] is None:
        return fallo['intentos'] <= 1
    return fallo['intentos'] <= MAX_REINTENTOS


def intentos_tras(fila_anterior) -> int:
    """Los intentos fallidos que llevará un fallback nuevo: uno más que el de
    la fila que sustituye, si esa también era un fallo."""
    fallo = fallo_de(fila_anterior)
    return fallo['intentos'] + 1 if fallo else 1
