"""El mantenimiento diario de Render: copias, cuentas abandonadas y errores
viejos (auditoría S8, 2026-11-13).

Hasta ese día:
  * no había copia propia de `sync.db` ni de `analysis.db` (solo el snapshot
    del disco de Render, que no se ve desde aquí), y la memoria colectiva es lo
    único del producto que no se puede rehacer;
  * la nube personal de una cuenta que no volvió se quedaba para siempre (lo
    que aportó a la memoria colectiva SÍ se queda para siempre, a propósito:
    ver `sync_endpoints.caducar_cuenta`);
  * `prune_old_errors` solo borraba los errores RESUELTOS.

Una pasada al arrancar y luego una al día, en un hilo. Cada parte es
best-effort por separado y deja SIEMPRE su línea, también con 0:
`[Copias] …`, `[Caducidad] …`, `[Errores] …`.

Las copias van en el MISMO disco (`/data/copias`): protegen de una base de
datos estropeada (una migración mala, un borrado de más), no de perder el
disco. Sacarlas fuera de Render pide un almacenamiento del owner
(`PENDING.md`).
"""
import gzip
import logging
import os
import shutil
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cada cuántos días se hace una copia, y cuántas se guardan.
COPIA_CADA_DIAS = int(os.environ.get("COPIA_CADA_DIAS", "7"))
COPIAS_QUE_SE_GUARDAN = int(os.environ.get("COPIAS_QUE_SE_GUARDAN", "2"))
# Una cuenta sin ningún aparato visto en este tiempo se borra. Lo dice la
# política de privacidad.
MESES_SIN_USO = int(os.environ.get("CUENTA_CADUCA_MESES", "24"))
# Cuántas cuentas caducan como mucho en una pasada (las demás, mañana).
CUENTAS_POR_PASADA = 200
# Los errores SIN resolver también se van a este tiempo.
DIAS_DE_UN_ERROR = 180


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


def copias_hechas(dir_copias: str) -> List[str]:
    """Las carpetas de copia, la más vieja primero (se llaman por la fecha)."""
    if not os.path.isdir(dir_copias):
        return []
    return sorted(d for d in os.listdir(dir_copias)
                  if os.path.isdir(os.path.join(dir_copias, d))
                  and not d.endswith(".tmp"))


def toca_copia(dir_copias: str, ahora: Optional[datetime] = None) -> bool:
    hechas = copias_hechas(dir_copias)
    if not hechas:
        return True
    try:
        ultima = datetime.strptime(hechas[-1][:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return True
    return (ahora or _ahora()) - ultima >= timedelta(days=COPIA_CADA_DIAS)


def hacer_copia(bases: Dict[str, str], dir_copias: str,
                ahora: Optional[datetime] = None) -> Dict[str, int]:
    """Copia cada base con `VACUUM INTO` (coherente aunque otros escriban),
    la comprime y la deja en `dir_copias/<fecha>/`. Se escribe en una carpeta
    `.tmp` y se renombra al final: una copia a medias no cuenta como hecha.
    Sin sitio en el disco (el doble de lo que ocupan), no se hace.
    Devuelve {nombre: bytes comprimidos}."""
    fecha = (ahora or _ahora()).strftime("%Y-%m-%d")
    final = os.path.join(dir_copias, fecha)
    tmp = final + ".tmp"
    os.makedirs(dir_copias, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    necesario = 2 * sum(os.path.getsize(p) for p in bases.values()
                        if os.path.exists(p))
    libre = shutil.disk_usage(dir_copias).free
    if libre < necesario:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"sin sitio: hacen falta {necesario // 2**20} MB "
                           f"y hay {libre // 2**20} MB")
    hecho: Dict[str, int] = {}
    try:
        for nombre, ruta in bases.items():
            if not os.path.exists(ruta):
                continue
            plano = os.path.join(tmp, f"{nombre}.db")
            conn = sqlite3.connect(ruta, timeout=60.0)
            try:
                conn.execute("VACUUM INTO ?", (plano,))
            finally:
                conn.close()
            with open(plano, "rb") as src, gzip.open(plano + ".gz", "wb",
                                                     compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.remove(plano)
            hecho[nombre] = os.path.getsize(plano + ".gz")
        shutil.rmtree(final, ignore_errors=True)
        os.rename(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    for vieja in copias_hechas(dir_copias)[:-COPIAS_QUE_SE_GUARDAN]:
        shutil.rmtree(os.path.join(dir_copias, vieja), ignore_errors=True)
    return hecho


def caducar_cuentas(
        sin_ver_desde: Callable[[str, int], List[Dict]],
        caducar: Callable[[str, List[str]], Dict[str, int]],
        ahora: Optional[datetime] = None) -> int:
    """Vacía la nube personal de las cuentas sin ningún aparato visto en
    `MESES_SIN_USO` meses (la memoria colectiva no se toca). Devuelve cuántas."""
    corte = ((ahora or _ahora()) - timedelta(days=30 * MESES_SIN_USO)
             ).isoformat()
    n = 0
    for cuenta in sin_ver_desde(corte, CUENTAS_POR_PASADA):
        try:
            caducar(cuenta["user_id"], cuenta["aparatos"])
            n += 1
        except Exception as e:  # noqa: BLE001 - la siguiente, mañana
            logger.warning(f"[Caducidad] {cuenta['user_id']}: {e}")
    return n


def pasada(bases: Dict[str, str], dir_copias: str, db,
           sin_ver_desde, caducar) -> Dict[str, object]:
    """Una pasada entera. Cada parte por su lado: si una falla, las demás
    siguen. Devuelve lo que hizo, para los tests."""
    r: Dict[str, object] = {}
    try:
        if toca_copia(dir_copias):
            hecho = hacer_copia(bases, dir_copias)
            r["copia"] = hecho
            logger.info("[Copias] hecha: " + ", ".join(
                f"{k} {v // 2**20} MB" for k, v in hecho.items())
                + f" (se guardan {COPIAS_QUE_SE_GUARDAN})")
        else:
            r["copia"] = None
            logger.info(f"[Copias] no toca: la última es "
                        f"{copias_hechas(dir_copias)[-1]}")
    except Exception as e:  # noqa: BLE001
        r["copia_error"] = str(e)
        logger.error(f"[Copias] falló: {e}")
    try:
        r["caducadas"] = caducar_cuentas(sin_ver_desde, caducar)
        logger.info(f"[Caducidad] {r['caducadas']} cuentas sin usar en "
                    f"{MESES_SIN_USO} meses: nube personal vaciada, memoria "
                    f"colectiva intacta")
    except Exception as e:  # noqa: BLE001
        r["caducidad_error"] = str(e)
        logger.error(f"[Caducidad] falló: {e}")
    try:
        r["errores"] = db.prune_old_errors(days=DIAS_DE_UN_ERROR,
                                           only_resolved=False)
        logger.info(f"[Errores] {r['errores']} errores de más de "
                    f"{DIAS_DE_UN_ERROR} días borrados")
    except Exception as e:  # noqa: BLE001
        r["errores_error"] = str(e)
        logger.error(f"[Errores] falló la poda: {e}")
    return r


def arrancar(bases: Dict[str, str], dir_copias: str, db,
             sin_ver_desde, caducar, primera_tras_s: int = 300) -> None:
    """La pasada a los 5 minutos del arranque (que no compita con él) y luego
    una al día, en un hilo que no frena el cierre."""
    def bucle():
        espera = primera_tras_s
        evento = threading.Event()
        while True:
            evento.wait(espera)
            pasada(bases, dir_copias, db, sin_ver_desde, caducar)
            espera = 24 * 3600

    threading.Thread(target=bucle, name="mantenimiento", daemon=True).start()
