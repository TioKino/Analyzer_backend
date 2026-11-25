"""Las portadas, a un tamaño razonable (2026-11-19).

El diagnóstico de la limpieza (`limpieza.py`) dio el disco de Render al 82 %:
84,5 GB, con 49 GB en portadas (230.298 ficheros, ~213 KB de media: se
guardaban a resolución completa, tal como venían del fichero o de internet).
La app las enseña pequeñas —una miniatura en la tabla, la ficha de un tema— y
las copias semanales de `mantenimiento.py` piden el doble de las bases de
datos libre, así que el disco iba a dejar sin copias al servidor.

Desde este día una portada se guarda como mucho a `LADO_MAX` px de lado, en
JPEG de calidad `CALIDAD` (`reducir`), en las dos puertas que las escriben
(`save_artwork_to_cache` y `POST /artwork/upload`); y las que ya había las
reduce una pasada de fondo (`Reduccion`), que lanza el admin. Reglas:

- Solo se cambia una portada si sale MÁS PEQUEÑA; si no, se queda como está.
- Un GIF no se toca (puede ser animado), ni lo que Pillow no sabe abrir.
- Una con transparencia se aplana sobre negro (el fondo de la app).
- Se escribe con temporal + renombrar, y la de otra extensión de la misma
  huella se borra DESPUÉS (`GET /artwork` mira `jpg` el primero).

Lo ata `test_las_portadas_se_reducen.py`.
"""

import io
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:  # Pillow solo hace falta para reducir; sin él, todo sigue como antes.
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

LADO_MAX = 600
CALIDAD = 85
# Por debajo de esto no vale la pena abrirla (ya es pequeña).
BYTES_MIN = 60 * 1024
_EXTS = ('jpg', 'jpeg', 'png', 'webp', 'gif')
# Lo único que Pillow puede abrir aquí (ver `reducir`).
_FORMATOS_PIL = ('JPEG', 'PNG', 'WEBP', 'GIF')


def reducir(datos: bytes) -> Optional[bytes]:
    """La portada en JPEG de como mucho `LADO_MAX` px, o None si no hay que
    tocarla (pequeña, GIF, ilegible, o no saldría más pequeña)."""
    if Image is None or not datos or len(datos) < BYTES_MIN:
        return None
    if datos[:4] == b'GIF8':
        return None
    try:
        # Solo los formatos de una portada. Sin `formats`, Pillow abre lo que
        # reconozca, y un EPS/PostScript (que puede venir dentro de las
        # etiquetas de cualquier audio subido) lo pasa a Ghostscript: la vía
        # de Upload2Shell (CVE-2026-125396, aviso de Render del 2026-11-21).
        with Image.open(io.BytesIO(datos), formats=_FORMATOS_PIL) as img:
            # Un JPEG que ya cabe no se recomprime: perdería calidad cada vez
            # que pasara la pasada, y no hay nada que ganar.
            if img.format == 'JPEG' and max(img.size) <= LADO_MAX:
                return None
            img.load()
            if img.mode in ('RGBA', 'LA') or (
                    img.mode == 'P' and 'transparency' in img.info):
                rgba = img.convert('RGBA')
                fondo = Image.new('RGB', rgba.size, (0, 0, 0))
                fondo.paste(rgba, mask=rgba.split()[-1])
                img2 = fondo
            else:
                img2 = img.convert('RGB')
            img2.thumbnail((LADO_MAX, LADO_MAX), Image.LANCZOS)
            salida = io.BytesIO()
            img2.save(salida, 'JPEG', quality=CALIDAD, optimize=True,
                      progressive=True)
    except Exception:
        return None
    nuevo = salida.getvalue()
    return nuevo if len(nuevo) < len(datos) else None


def guardar(carpeta: str, huella: str, datos: bytes,
            ext_original: str,
            reducida: Optional[bytes] = None) -> Tuple[str, int]:
    """Escribe la portada de [huella], reducida si toca ([reducida] si ya
    se calculó). Devuelve el nombre del fichero y los bytes escritos. Borra
    las de otras extensiones."""
    if reducida is None:
        reducida = reducir(datos)
    if reducida is not None:
        datos, ext = reducida, 'jpg'
    else:
        ext = ext_original
    nombre = f"{huella}.{ext}"
    ruta = os.path.join(carpeta, nombre)
    tmp = ruta + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(datos)
    os.replace(tmp, ruta)
    for otra in _EXTS:
        if otra != ext:
            vieja = os.path.join(carpeta, f"{huella}.{otra}")
            if os.path.exists(vieja):
                try:
                    os.unlink(vieja)
                except OSError:
                    pass
    return nombre, len(datos)


def _ficheros(carpeta: str) -> List[os.DirEntry]:
    if not carpeta or not os.path.isdir(carpeta):
        return []
    with os.scandir(carpeta) as it:
        return [e for e in it if e.is_file()
                and e.name.rpartition('.')[2].lower() in _EXTS]


def medir(carpeta: str, muestra: int = 400,
          semilla: Optional[int] = None) -> Dict[str, Any]:
    """Cuánto se ahorraría: reduce en memoria una muestra al azar (no escribe
    nada) y lo extrapola a la carpeta entera."""
    todos = _ficheros(carpeta)
    total_bytes = 0
    for e in todos:
        try:
            total_bytes += e.stat().st_size
        except OSError:
            pass
    rnd = random.Random(semilla)
    elegidos = rnd.sample(todos, min(muestra, len(todos)))
    antes = despues = cambiarian = 0
    for e in elegidos:
        try:
            with open(e.path, 'rb') as fh:
                datos = fh.read()
        except OSError:
            continue
        nuevo = reducir(datos)
        antes += len(datos)
        despues += len(nuevo) if nuevo is not None else len(datos)
        if nuevo is not None:
            cambiarian += 1
    proporcion = (despues / antes) if antes else 1.0
    return {
        'ficheros': len(todos),
        'bytes': total_bytes,
        'muestra': len(elegidos),
        'muestra_bytes_antes': antes,
        'muestra_bytes_despues': despues,
        'muestra_cambiarian': cambiarian,
        'bytes_estimados_despues': int(total_bytes * proporcion),
        'ahorro_estimado': int(total_bytes * (1 - proporcion)),
        'lado_max': LADO_MAX,
        'calidad': CALIDAD,
    }


class Reduccion:
    """La pasada sobre las portadas que ya había: en un hilo, una a una, con
    una pausa entre cada una para no quitarle la CPU al resto de Render. Se
    puede parar (`parar`) y volver a lanzar: lo ya reducido sale enseguida
    (pesa menos de `BYTES_MIN` o no se queda más pequeño)."""

    def __init__(self, carpeta: str, pausa_s: float = 0.02):
        self.carpeta = carpeta
        self.pausa_s = pausa_s
        self._hilo: Optional[threading.Thread] = None
        self._parar = threading.Event()
        self.estado: Dict[str, Any] = {'corriendo': False}

    def en_marcha(self) -> bool:
        return self._hilo is not None and self._hilo.is_alive()

    def empezar(self) -> bool:
        if self.en_marcha():
            return False
        self._parar.clear()
        self._hilo = threading.Thread(target=self.correr, daemon=True,
                                      name='reduccion-portadas')
        self._hilo.start()
        return True

    def parar(self):
        self._parar.set()

    def correr(self):
        e0 = time.time()
        todos = _ficheros(self.carpeta)
        st = {
            'corriendo': True, 'empezo': e0, 'total': len(todos),
            'vistos': 0, 'reducidos': 0, 'bytes_antes': 0,
            'bytes_despues': 0, 'errores': 0, 'parada': False,
        }
        self.estado = st
        for e in todos:
            if self._parar.is_set():
                st['parada'] = True
                break
            st['vistos'] += 1
            try:
                tam = e.stat().st_size
                if tam < BYTES_MIN:
                    continue
                with open(e.path, 'rb') as fh:
                    datos = fh.read()
                nuevo = reducir(datos)
                if nuevo is None:
                    continue
                huella, _, ext = e.name.rpartition('.')
                guardar(self.carpeta, huella, datos, ext.lower(),
                        reducida=nuevo)
                st['reducidos'] += 1
                st['bytes_antes'] += tam
                st['bytes_despues'] += len(nuevo)
            except Exception:
                st['errores'] += 1
            if self.pausa_s:
                time.sleep(self.pausa_s)
        st['corriendo'] = False
        st['segundos'] = round(time.time() - e0, 1)
        try:  # pragma: no cover - solo log
            import logging
            logging.getLogger(__name__).info(
                f"[Portadas] reducidas {st['reducidos']} de {st['vistos']} "
                f"({st['bytes_antes'] // 1_000_000} MB -> "
                f"{st['bytes_despues'] // 1_000_000} MB), "
                f"{st['errores']} errores, {st['segundos']} s"
                f"{' (parada)' if st['parada'] else ''}")
        except Exception:
            pass
