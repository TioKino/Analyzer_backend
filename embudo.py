"""El embudo, con UNA definición (v3, 2026-11-25).

Owner, tras la auditoría de esa fecha: «es frecuente que surjan errores. No
quiero más. Haz un embudo definitivo». Hasta ese día había cuatro sitios que
calculaban «el embudo» o «la retención» cada uno a su manera (`/admin/funnel`,
`/admin/retention`, la foto del Resumen y los paneles de Técnico), y el embudo
en sí tenía cinco fallos de definición que hacían que cada lectura invitara a
una conclusión distinta (`HISTORIAL.md` → «2026-11-25 — El embudo definitivo»):

  1. COHORTE SIN MADURAR. Contaba las altas de los últimos 30 días, incluidas
     las de esta mañana, que aún no han tenido tiempo de dar ningún paso: la
     conversión bajaba cuando llegaba más gente y subía cuando llegaba menos.
     Ahora: las altas de hace 8 a 37 días, y cada paso cuenta si se dio en los
     7 días siguientes al alta. Toda alta tiene el mismo tiempo para convertir.
  2. PASOS QUE UNA PLATAFORMA NO EMITE. El escritorio no mandaba
     `onboarding_shown`, así que su paso «Vio el onboarding» salía a 0 y el
     siguiente era mayor que él.
  3. PASOS SUELTOS, NO UN EMBUDO. Cada paso contaba sus aparatos por su cuenta:
     «vio su 1er track» superaba a «importó» (llegan por sync o por un XML) y
     la caída salía 0 o negativa. Ahora cuenta el PASO MÁS LEJANO al que llegó
     cada aparato: quien vio su música empezó a traerla, aunque su evento se
     perdiera o llegara por un camino sin evento. Nunca sube de un paso a otro.
  4. «IMPORTÓ» SIN NADA DENTRO. Un import con 0 temas (todo fallido o
     repetido) contaba como música importada. Ahora el valor es VER su música.
  5. EL TOTAL MEZCLABA LAS DOS APPS: aplicaba los pasos del escritorio al
     móvil, donde importar no es el camino. Ahora cada aparato se mide con los
     pasos de SU plataforma (la del alta), y el total suma los dos.

Y quedan FUERA de toda cohorte los aparatos de prueba (una build de
desarrollo: `flutter run`, `run_device.*`; lo dice la app con `prueba`) y los
que estrenaron id al «Borrar mis datos de la nube» (`reidentificado`): no son
altas, son el mismo aparato con otro nombre, y entraban como recién llegados
que no daban ningún paso.

**Una medida nueva del embudo va AQUÍ**, no en una consulta suelta del panel.
"""

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set

from validation import is_desktop_platform, is_mobile_platform

DEFINICION = "v3"
DESCRIPCION = (
    "altas de hace 8 a 37 dias (sin aparatos de prueba ni reidentificados), "
    "cada paso dado en los 7 dias siguientes al alta, contado por el paso mas "
    "lejano al que llego cada aparato; cada aparato con los pasos de su "
    "plataforma"
)

# Lo que tiene cada alta para dar los pasos, y cuántos días de altas entran.
DIAS_PARA_MADURAR = 7
VENTANA_DE_ALTAS = 30

GRUPOS = ("escritorio", "movil")

# Los pasos, en orden. Cada aparato «llega» a un paso si dio ese paso o
# cualquiera de los siguientes (ver `_llego`).
PASOS = ("alta", "abrio", "bienvenida_vista", "bienvenida_pasada", "empezo",
         "valor")

ETIQUETAS = {
    "alta": "Se dio de alta",
    "abrio": "Abrio la app",
    "bienvenida_vista": "Vio la bienvenida",
    "bienvenida_pasada": "Paso la bienvenida",
}
ETIQUETAS_DEL_GRUPO = {
    "escritorio": {
        "empezo": "Empezo a traer su musica",
        "valor": "Vio su musica",
    },
    "movil": {
        "empezo": "Probo algo (escuchar, importar o vincular)",
        "valor": "Identifico un tema o vio su musica",
    },
    "todos": {
        "empezo": "Empezo a usarla",
        "valor": "Llego a lo que vale",
    },
}

# Los eventos que lee el embudo. Cualquier otro no cambia nada.
EVENTOS = (
    "app_opened", "session_start",
    "onboarding_shown", "onboarding_completed", "onboarding_skipped",
    "import_started", "import_completed", "import_biblioteca",
    "link_sheet_opened", "device_linked", "library_synced",
    "listen_result", "first_track_viewed",
)


def grupo_de(plataforma: Optional[str]) -> Optional[str]:
    """`escritorio`, `movil` o None («no lo sé»: no cae en ningún grupo)."""
    if is_mobile_platform(plataforma):
        return "movil"
    if is_desktop_platform(plataforma):
        return "escritorio"
    return None


def ventana(hoy: Optional[date] = None,
            dias: int = DIAS_PARA_MADURAR) -> (str, str):
    """Las fechas (UTC, inclusive) de los `VENTANA_DE_ALTAS` días de altas
    que ya han tenido `dias` días ENTEROS después del alta (hoy no cuenta:
    no ha terminado). Con los 7 del embudo: de hace 37 a hace 8 días.

    La retención reciente usa esta misma regla con 1, 7 y 28 días, así que su
    D7 es exactamente la «vuelta en 7 días» del embudo: la misma cohorte."""
    if hoy is None:
        hoy = datetime.now(timezone.utc).date()
    hasta = hoy - timedelta(days=dias + 1)
    desde = hasta - timedelta(days=VENTANA_DE_ALTAS - 1)
    return desde.isoformat(), hasta.isoformat()


def _columnas(conn: sqlite3.Connection, tabla: str) -> Set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({tabla})")}
    except sqlite3.Error:
        return set()


def _props(raw) -> dict:
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _entero(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def hechos_del_evento(nombre: str, props: dict, dia: str, d0: str) -> Set[str]:
    """Lo que un evento dice que el aparato hizo (pura, para los tests)."""
    h: Set[str] = set()
    if nombre in ("app_opened", "session_start"):
        h.add("abrio")
        if dia > d0:
            h.add("volvio")
    elif nombre in ("onboarding_shown", "onboarding_completed",
                    "onboarding_skipped", "import_started",
                    "link_sheet_opened", "device_linked", "library_synced",
                    "first_track_viewed", "import_biblioteca"):
        h.add(nombre)
    elif nombre == "import_completed":
        h.add("import_completed")
        # El escritorio manda `ok` (los que entraron) y el móvil `total` (los
        # ficheros del import, que solo se emite con alguno).
        if _entero(props.get("ok")) > 0 or _entero(props.get("total")) > 0:
            h.add("import_con_temas")
    elif nombre == "listen_result":
        h.add("escucho")
        if props.get("outcome") == "found":
            h.add("identifico")
    return h


def pasos_directos(grupo: str, hechos: Set[str]) -> Dict[str, bool]:
    """Qué pasos dio un aparato con su PROPIO evento (pura)."""
    movil = grupo == "movil"
    return {
        "alta": True,
        "abrio": "abrio" in hechos,
        "bienvenida_vista": "onboarding_shown" in hechos,
        "bienvenida_pasada": ("onboarding_completed" in hechos
                              or (movil and "onboarding_skipped" in hechos)),
        "empezo": bool(hechos & (
            {"escucho", "import_started", "import_completed",
             "import_biblioteca", "link_sheet_opened", "device_linked",
             "library_synced"} if movil else
            {"import_started", "import_completed", "import_biblioteca"})),
        "valor": ("first_track_viewed" in hechos
                  or (movil and "identifico" in hechos)),
    }


def paso_mas_lejano(directos: Dict[str, bool]) -> int:
    """El índice en `PASOS` del último paso dado (pura)."""
    lejos = 0
    for i, paso in enumerate(PASOS):
        if directos.get(paso):
            lejos = i
    return lejos


def _tasa(n: int, de: int) -> Optional[float]:
    return round(100.0 * n / de, 1) if de else None


def _resumen_de(aparatos: List[dict], etiquetas: Dict[str, str]) -> dict:
    """El embudo de un grupo de aparatos ya clasificados."""
    n = len(aparatos)
    llegaron = [0] * len(PASOS)
    directos = [0] * len(PASOS)
    for a in aparatos:
        for i in range(a["lejos"] + 1):
            llegaron[i] += 1
        for i, paso in enumerate(PASOS):
            if a["directos"][paso]:
                directos[i] += 1
    pasos = []
    for i, paso in enumerate(PASOS):
        antes = llegaron[i - 1] if i else None
        pasos.append({
            "paso": paso,
            "etiqueta": ETIQUETAS.get(paso) or etiquetas[paso],
            "llegaron": llegaron[i],
            "pct": _tasa(llegaron[i], n),
            # Los que lo dieron con su propio evento. Menos que `llegaron` =
            # llegaron por un camino sin evento (o se perdió): no es una fuga.
            "directos": directos[i],
            "caen_pct": (round(100.0 * (antes - llegaron[i]) / antes, 1)
                         if antes else None),
        })
    con = [a for a in aparatos if a["lejos"] == len(PASOS) - 1]
    sin = [a for a in aparatos if a["lejos"] < len(PASOS) - 1]

    def _vuelta(grupo):
        v = sum(1 for a in grupo if "volvio" in a["hechos"])
        return {"cohorte": len(grupo), "volvieron": v,
                "pct": _tasa(v, len(grupo))}

    def _cuantos(hecho):
        return sum(1 for a in aparatos if hecho in a["hechos"])

    return {
        "cohorte": n,
        "pasos": pasos,
        # Volver en los 7 días siguientes al alta (abrir otro día). Con y sin
        # haber llegado al valor: si los que llegan vuelven y los demás no, el
        # trabajo está en llevarlos al valor, no en traerlos de vuelta.
        "volvio_7d": {
            "todos": _vuelta(aparatos),
            "con_valor": _vuelta(con),
            "sin_valor": _vuelta(sin),
        },
        # La bienvenida, en eventos de los mismos aparatos (sin cierre):
        # saltar es una decisión, abandonar es una fuga.
        "bienvenida": {
            "vista": _cuantos("onboarding_shown"),
            "completada": _cuantos("onboarding_completed"),
            "saltada": _cuantos("onboarding_skipped"),
        },
        # Por dónde llegan (eventos propios, un aparato puede estar en varios).
        "caminos": {k: _cuantos(k) for k in (
            "import_started", "import_con_temas", "import_biblioteca",
            "escucho", "identifico", "link_sheet_opened", "device_linked",
            "library_synced", "first_track_viewed")},
    }


def calcular(conn: sqlite3.Connection, hoy: Optional[date] = None,
             plataforma: Optional[str] = None) -> dict:
    """El embudo de la cohorte madura, por grupo y en total.

    `plataforma` restringe a una plataforma concreta (`macos-mas`, `ios`…),
    que se mide con los pasos de su grupo.
    """
    desde, hasta = ventana(hoy)
    cols_dfs = _columnas(conn, "device_first_seen")
    cols_ev = _columnas(conn, "events")
    marca_prueba = "prueba" in cols_dfs
    marca_reid = "reidentificado" in cols_dfs

    sel = ["device_id", "first_day", "first_platform"]
    sel += ["prueba" if marca_prueba else "0",
            "reidentificado" if marca_reid else "0"]
    cohorte: Dict[str, dict] = {}
    excluidos = {"prueba": 0, "reidentificado": 0}
    sin_plataforma: List[str] = []
    for dev, d0, plat, prueba, reid in conn.execute(
            f"SELECT {', '.join(sel)} FROM device_first_seen "
            "WHERE device_id IS NOT NULL AND first_day >= ? AND first_day <= ?",
            (desde, hasta)):
        if prueba:
            excluidos["prueba"] += 1
            continue
        if reid:
            excluidos["reidentificado"] += 1
            continue
        cohorte[dev] = {"d0": d0, "plataforma": (plat or "").strip().lower(),
                        "hechos": set()}

    props_col = "e.props" if "props" in cols_ev else "NULL"
    plat_col = "e.platform" if "platform" in cols_ev else "NULL"
    marcas = ",".join("?" * len(EVENTOS))
    for dev, nombre, dia, raw, plat in conn.execute(
            f"SELECT e.device_id, e.event_name, substr(e.timestamp, 1, 10), "
            f"{props_col}, {plat_col} FROM events e "
            "JOIN device_first_seen f ON f.device_id = e.device_id "
            "WHERE f.first_day >= ? AND f.first_day <= ? "
            f"AND e.event_name IN ({marcas}) "
            "AND substr(e.timestamp, 1, 10) >= f.first_day "
            "AND substr(e.timestamp, 1, 10) <= date(f.first_day, ?)",
            (desde, hasta) + EVENTOS + (f"+{DIAS_PARA_MADURAR} days",)):
        a = cohorte.get(dev)
        if a is None:
            continue
        if not a["plataforma"] and plat:
            # Sin plataforma en el alta (filas sembradas en agosto): la del
            # primer evento que la diga.
            a["plataforma"] = plat.strip().lower()
        props = _props(raw) if nombre in (
            "import_completed", "listen_result") else {}
        a["hechos"] |= hechos_del_evento(nombre, props, dia or "", a["d0"])

    por_grupo: Dict[str, List[dict]] = {g: [] for g in GRUPOS}
    for dev, a in cohorte.items():
        if plataforma and a["plataforma"] != plataforma.strip().lower():
            continue
        g = grupo_de(a["plataforma"])
        if g is None:
            sin_plataforma.append(dev)
            continue
        directos = pasos_directos(g, a["hechos"])
        a["directos"] = directos
        a["lejos"] = paso_mas_lejano(directos)
        por_grupo[g].append(a)

    return _salida(desde, hasta, plataforma, excluidos, len(sin_plataforma),
                   por_grupo)


def _salida(desde: str, hasta: str, plataforma: Optional[str],
            excluidos: Dict[str, int], sin_plataforma: int,
            por_grupo: Dict[str, List[dict]]) -> dict:
    salida = {
        "definicion": DEFINICION,
        "descripcion": DESCRIPCION,
        "altas_desde": desde,
        "altas_hasta": hasta,
        "dias_para_madurar": DIAS_PARA_MADURAR,
        "plataforma": plataforma or None,
        "excluidos": excluidos,
        "sin_plataforma": sin_plataforma,
        "grupos": {
            g: _resumen_de(por_grupo[g], ETIQUETAS_DEL_GRUPO[g])
            for g in GRUPOS
        },
    }
    salida["todos"] = _resumen_de(
        por_grupo["escritorio"] + por_grupo["movil"], ETIQUETAS_DEL_GRUPO["todos"])
    return salida


def vacio(hoy: Optional[date] = None) -> dict:
    """El embudo sin nadie: los mismos pasos, todos a cero.

    Para cuando no hay base de datos (o el calculo fallo): un paso sin datos
    sale a 0, no desaparece. Sin pasos, «no hay datos» se leeria como «el
    embudo no tiene pasos», y los clientes que pintan `steps` se quedan en
    blanco (paso en el CI el 2026-11-25: una ruta de base borrada por otro
    test dejaba el embudo sin pasos).
    """
    desde, hasta = ventana(hoy)
    return _salida(desde, hasta, None, {"prueba": 0, "reidentificado": 0}, 0,
                   {g: [] for g in GRUPOS})


def vista(embudo: dict, grupo: Optional[str]) -> dict:
    """El bloque de `grupo` (`escritorio`, `movil`) o el total."""
    if grupo in GRUPOS:
        return embudo["grupos"][grupo]
    return embudo["todos"]


def pasos_para_el_panel(bloque: dict) -> List[dict]:
    """Los pasos con la forma de siempre de `/admin/funnel` (`steps`), para
    los clientes que la leen: `event`, `label`, `devices`, `pct_of_top`,
    `drop_from_prev`."""
    return [{
        "event": p["paso"],
        "label": p["etiqueta"],
        "devices": p["llegaron"],
        "pct_of_top": p["pct"] if p["pct"] is not None else 0.0,
        "drop_from_prev": p["caen_pct"] if p["caen_pct"] is not None else 0.0,
    } for p in bloque["pasos"]]


def excluir_marcados(cols: Iterable[str], alias: str = "") -> str:
    """El trozo de WHERE que deja fuera los aparatos de prueba y los
    reidentificados de una consulta sobre `device_first_seen` (vacío si la
    tabla aún no tiene esas columnas)."""
    cols = set(cols)
    p = f"{alias}." if alias else ""
    trozo = ""
    if "prueba" in cols:
        trozo += f" AND COALESCE({p}prueba, 0) = 0"
    if "reidentificado" in cols:
        trozo += f" AND COALESCE({p}reidentificado, 0) = 0"
    return trozo
