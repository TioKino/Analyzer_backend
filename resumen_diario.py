"""La foto diaria del Resumen del panel admin (2026-10-24).

Owner: el panel «es un caos». Al abrirse lanzaba siete peticiones pesadas a la
vez y lo pintaba todo en un scroll; la pregunta «¿vamos bien?» solo la
contestaba `scripts/panel.py`, en la terminal y con lo que se hubiera guardado
a mano en `funnel_data/`.

El Resumen son ~12 números con su flecha contra la foto anterior. Calcularlos
es caro (recorren `sync.db` y `events`), así que se calculan UNA vez al día y
se guardan aquí: una fila por día en `admin_resumen_diario` de `analysis.db`.
`GET /admin/resumen` lee de esta tabla (milisegundos); si falta la de hoy, la
pide en segundo plano y contesta con la última que haya.

Cada número es best-effort: uno que falla sale `None` («falló»), no 0 ni
ausente — «no está» y «falló» no pueden decir lo mismo (CLAUDE.md, el panel).
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

TABLA = "admin_resumen_diario"

# Qué se guarda, en el orden en que se enseña. La dirección buena vive en el
# cliente (es presentación); aquí solo los nombres, para que el test vea que
# el cálculo los da todos.
METRICAS = (
    "activos_7d",
    "activos_30d",
    "altas_30d",
    "d7_rolling_pct",
    "volvio_alguna_vez_pct",
    "con_biblioteca",
    "mediana_temas",
    "gte_500",
    "cobertura_huella_pct",
    "audd_30d_render",
    "audd_30d_motor_local",
    "audd_30d_total",
    "errores_sin_resolver",
    "version_actual",
    "adopcion_escritorio_pct",
    "adopcion_movil_pct",
)


def hoy() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def asegurar_tabla(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLA} ("
        " dia TEXT PRIMARY KEY,"
        " datos TEXT NOT NULL,"
        " calculado_at TEXT NOT NULL,"
        " segundos REAL)"
    )


def guardar(conn: sqlite3.Connection, dia: str, datos: dict,
            segundos: Optional[float] = None) -> None:
    asegurar_tabla(conn)
    conn.execute(
        f"INSERT OR REPLACE INTO {TABLA} (dia, datos, calculado_at, segundos) "
        f"VALUES (?, ?, ?, ?)",
        (dia, json.dumps(datos), datetime.now(timezone.utc).isoformat(), segundos),
    )
    conn.commit()


def ultimas(conn: sqlite3.Connection, n: int = 14) -> List[dict]:
    """Las últimas `n` fotos, de la más nueva a la más vieja."""
    asegurar_tabla(conn)
    out = []
    for dia, datos, calculado_at, segundos in conn.execute(
            f"SELECT dia, datos, calculado_at, segundos FROM {TABLA} "
            f"ORDER BY dia DESC LIMIT ?", (n,)):
        try:
            d = json.loads(datos)
        except (TypeError, ValueError):
            d = {}
        out.append({"dia": dia, "datos": d, "calculado_at": calculado_at,
                    "segundos": segundos})
    return out


def calcular(medidas: Dict[str, Callable[[], dict]]) -> dict:
    """Junta las métricas a partir de `medidas` (nombre → función que devuelve
    un dict parcial). Lo que falla deja SUS claves en `None` y sigue: una
    medida rota no tumba el resumen entero."""
    datos: Dict[str, object] = {m: None for m in METRICAS}
    fallos = []
    for nombre, f in medidas.items():
        try:
            parte = f() or {}
        except Exception as e:  # noqa: BLE001 - best-effort por diseño
            fallos.append(f"{nombre}: {type(e).__name__}: {e}")
            continue
        for k, v in parte.items():
            if k in datos:
                datos[k] = v
    if datos.get("audd_30d_render") is not None or datos.get("audd_30d_motor_local") is not None:
        datos["audd_30d_total"] = ((datos.get("audd_30d_render") or 0)
                                   + (datos.get("audd_30d_motor_local") or 0))
    datos["_fallos"] = fallos
    return datos


def orden_de_version(v) -> list:
    partes = []
    for trozo in str(v or "").replace("+", ".").split("."):
        try:
            partes.append(int(trozo))
        except ValueError:
            partes.append(-1)
    return partes


def adopcion_pct(por_version: Dict[str, int], version: str) -> Optional[float]:
    """% de los aparatos (con versión conocida) que están en `version` o en una
    posterior: una build de prueba más nueva cuenta como al día."""
    total = sum(por_version.values())
    if not total or not version:
        return None
    objetivo = orden_de_version(version)
    al_dia = sum(n for v, n in por_version.items()
                 if orden_de_version(v) >= objetivo)
    return round(100.0 * al_dia / total, 1)
