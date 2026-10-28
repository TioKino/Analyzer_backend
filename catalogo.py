"""El catálogo del panel admin: «la biblioteca de DJ Analyzer» (2026-10-24).

Owner: «un catálogo global de todos los tracks analizados». Sale de la tabla
`tracks` de `analysis.db`, que es la memoria colectiva (por huella, sin dueño),
y NO de `sync.db` (las bibliotecas personales): por eso no lleva nombre de
fichero ni aparato. Lo que sí lleva: portada y preview (por huella), BPM y de
dónde salió, tonalidad, género, energía, tipo, cuántos DJs lo tienen, si tiene
huella acústica, cuántas versiones del mismo audio hay (su cluster), la
plataforma que lo analizó y la fecha.

Paginado en el servidor, y cada consulta entra por un índice: `tracks` tiene
más de 120.000 filas con su `analysis_json` dentro, y un recorrido de la tabla
son segundos en caliente y 35 s en frío (CLAUDE.md, la ficha de Escuchar):

  - **Sin filtros** (lo de por defecto): `idx_tracks_analyzed_at` hacia atrás
    y el `LIMIT` corta.
  - **Con filtros** (los de la biblioteca —género, BPM, tonalidad, energía,
    tipo— y los del panel —plataforma, motor, huella, ISRC, fecha—, y también
    buscando o dentro de un cluster): cada filtro es el conjunto de `rowid` que
    saca SU índice (`_conjuntos`), y se recorre el índice de la fecha mirando
    si cada `rowid` está en todos. Solo índices: de la tabla se leen los temas
    de la página. Medido con 120.000 filas (878 MB): todas por debajo de
    125 ms, también la que no casa con nada (recorre el índice entero).
  - **Por nº de DJs**: desde `track_popularity` (filas pequeñas), y cada clave
    a su tema por `idx_acoustic` o `idx_tracks_fingerprint`. Una fila por TEMA
    (el cluster), no por fichero. Con este orden solo se busca (o se abre un
    cluster): cruzar filtros obligaría a recorrer la tabla.

El total sale siempre de un índice, salvo por nº de DJs buscando (las que
casan, como mucho `MAX_COINCIDENCIAS`). Buscando se miran como mucho esas:
`al_tope` dice que el total es «o más».
"""

import json
import sqlite3
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

POR_PAGINA = 50
MAX_POR_PAGINA = 100
# Buscando y por nº de DJs: como mucho cuántos temas se ordenan en memoria.
MAX_COINCIDENCIAS = 2000

_COLUMNAS = (
    "t.fingerprint, t.artist, t.title, t.bpm, t.key, t.camelot, t.genre, "
    "t.energy_dj, t.track_type, t.duration, t.analyzed_at, t.platform, "
    "t.engine_source, t.acoustic_id, t.isrc, "
    "(t.chromaprint IS NOT NULL AND t.chromaprint != '') AS con_huella, "
    "t.analysis_json"
)


def _fuentes(analysis_json: Optional[str]) -> Dict[str, Optional[str]]:
    """De dónde salió cada dato (solo las de primer nivel: `save_track` aplana
    el JSON al guardar)."""
    try:
        d = json.loads(analysis_json or "{}")
    except (TypeError, ValueError):
        d = {}
    if not isinstance(d, dict):
        d = {}
    anio = d.get("year")
    return {
        "bpm_source": d.get("bpm_source"),
        "key_source": d.get("key_source"),
        "genre_source": d.get("genre_source"),
        "year": str(anio) if anio not in (None, "", 0) else None,
    }


def _fila(r: sqlite3.Row, djs: Dict[str, int], versiones: Dict[str, int],
          con_preview: Callable[[str], bool], base_url: str) -> dict:
    fp = r["fingerprint"] or ""
    cluster = r["acoustic_id"] or ""
    fila = {
        "fingerprint": fp,
        "artist": r["artist"] or "",
        "title": r["title"] or "",
        "bpm": r["bpm"],
        "key": r["key"],
        "camelot": r["camelot"],
        "genre": r["genre"],
        "energy_dj": r["energy_dj"],
        "track_type": r["track_type"],
        "duration": r["duration"],
        "analyzed_at": r["analyzed_at"],
        "platform": r["platform"],
        "engine_source": r["engine_source"],
        "isrc": r["isrc"],
        "con_huella": bool(r["con_huella"]),
        "cluster": cluster or None,
        "versiones": versiones.get(cluster, 1) if cluster else 1,
        "djs": djs.get(cluster or fp, 0),
        "con_preview": bool(fp) and con_preview(fp),
        # A Render y sin buscar en internet: la del catálogo es la que hay.
        "artwork_url": f"{base_url}/artwork/{fp}?online=0" if fp and base_url else "",
    }
    fila["preview_url"] = (f"{base_url}/preview/{fp}"
                           if fila["con_preview"] and base_url else "")
    fila.update(_fuentes(r["analysis_json"]))
    return fila


def _djs_de(conn, claves: List[str]) -> Dict[str, int]:
    claves = [c for c in claves if c]
    if not claves:
        return {}
    marcas = ",".join("?" for _ in claves)
    try:
        return {
            r[0]: int(r[1] or 0)
            for r in conn.execute(
                f"SELECT fingerprint, dj_count FROM track_popularity "
                f"WHERE fingerprint IN ({marcas})", claves)
        }
    except sqlite3.OperationalError:  # BD sin la tabla (tests, motor local)
        return {}


def _versiones_de(conn, clusters: List[str]) -> Dict[str, int]:
    clusters = sorted({c for c in clusters if c})
    if not clusters:
        return {}
    marcas = ",".join("?" for _ in clusters)
    return {
        r[0]: int(r[1])
        for r in conn.execute(
            f"SELECT acoustic_id, COUNT(*) FROM tracks INDEXED BY idx_acoustic "
            f"WHERE acoustic_id IN ({marcas}) GROUP BY acoustic_id", clusters)
    }


def _como_like(q: str) -> str:
    q = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{q}%"


# Los valores que guarda el clasificador (`spectral_classifier.py`); el cliente
# los enseña con su nombre (`trackTypes` de `genre_constants.dart`).
TIPOS = ("opener", "warmup", "builder", "peak_time", "anthem", "cooldown",
         "closing")
MOTORES = ("render", "local_engine", "sin_sellar")


def _buscando(q: str):
    return ("SELECT rowid FROM tracks INDEXED BY idx_tracks_ficha "
            "WHERE (artist LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\') "
            "AND bpm > 0 LIMIT ?", [_como_like(q), _como_like(q), MAX_COINCIDENCIAS])


def _conjuntos(*, q, cluster, genero, bpm_min, bpm_max, camelot, energia_min,
               energia_max, tipo, plataforma, motor, huella, isrc):
    """Cada filtro, como el conjunto de `rowid` que saca SU índice (sin leer
    la tabla). `dentro`: el tema tiene que estar; `fuera`: no puede."""
    dentro, fuera = [], []
    if q:
        dentro.append(_buscando(q))
    if cluster:
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_acoustic "
                       "WHERE acoustic_id = ?", [cluster]))
    if genero:
        # Sin mayúsculas: el índice entero (solo el índice, sin la tabla).
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_genre "
                       "WHERE genre = ? COLLATE NOCASE", [genero]))
    if bpm_min is not None or bpm_max is not None:
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_bpm "
                       "WHERE bpm > 0 AND bpm >= ? AND bpm <= ?",
                       [float(bpm_min or 0), float(bpm_max if bpm_max is not None else 1e9)]))
    else:
        # Solo lo ANALIZADO: fuera las filas a 0 (un fallback, lo que siembra
        # Escuchar), que son pocas.
        fuera.append(("SELECT rowid FROM tracks INDEXED BY idx_bpm "
                      "WHERE bpm IS NULL OR bpm <= 0", []))
    if camelot:
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_camelot "
                       "WHERE camelot = ?", [camelot.upper()]))
    if energia_min is not None or energia_max is not None:
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_energy "
                       "WHERE energy_dj >= ? AND energy_dj <= ?",
                       [int(energia_min or 0), int(energia_max if energia_max is not None else 99)]))
    if tipo:
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_tracks_track_type "
                       "WHERE track_type = ?",
                       [tipo.strip().lower().replace(" ", "_")]))
    if plataforma:
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_tracks_platform "
                       "WHERE platform = ?", [plataforma]))
    if motor == "sin_sellar":
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_tracks_engine "
                       "WHERE engine_source IS NULL OR engine_source = ''", []))
    elif motor:
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_tracks_engine "
                       "WHERE engine_source = ?", [motor]))
    if huella in ("con", "sin"):
        # Las huellas pesan KB: del índice solo se leen las que NO hay.
        sin = ("SELECT rowid FROM tracks INDEXED BY idx_chromaprint "
               "WHERE chromaprint IS NULL OR chromaprint = ''", [])
        (dentro if huella == "sin" else fuera).append(sin)
    if isrc == "con":
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_isrc "
                       "WHERE isrc > ''", []))
    elif isrc == "sin":
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_isrc "
                       "WHERE isrc IS NULL OR isrc = ''", []))
    return dentro, fuera


def _dia_siguiente(dia: str) -> str:
    return (date.fromisoformat(dia) + timedelta(days=1)).isoformat()


def pagina_del_catalogo(
    conn: sqlite3.Connection,
    *,
    q: str = "",
    orden: str = "fecha",
    genero: Optional[str] = None,
    bpm_min: Optional[float] = None,
    bpm_max: Optional[float] = None,
    camelot: Optional[str] = None,
    energia_min: Optional[int] = None,
    energia_max: Optional[int] = None,
    tipo: Optional[str] = None,
    plataforma: Optional[str] = None,
    motor: Optional[str] = None,
    huella: Optional[str] = None,
    isrc: Optional[str] = None,
    desde: Optional[str] = None,
    hasta: Optional[str] = None,
    cluster: Optional[str] = None,
    dir: str = "desc",
    pagina: int = 0,
    por_pagina: int = POR_PAGINA,
    con_preview: Callable[[str], bool] = lambda _fp: False,
    base_url: str = "",
) -> dict:
    """Una página del catálogo. Solo temas ANALIZADOS (`bpm > 0`): las filas a
    0 son el fallback de un análisis fallido o lo que siembra Escuchar."""
    conn.row_factory = sqlite3.Row
    por_pagina = max(1, min(int(por_pagina or POR_PAGINA), MAX_POR_PAGINA))
    pagina = max(0, int(pagina or 0))
    q = (q or "").strip()
    orden = orden if orden == "djs" or orden in ORDENES else "fecha"
    dir = "asc" if dir == "asc" else "desc"
    filtros = dict(
        genero=genero, bpm_min=bpm_min, bpm_max=bpm_max, camelot=camelot,
        energia_min=energia_min, energia_max=energia_max, tipo=tipo,
        plataforma=plataforma, motor=motor if motor in MOTORES else None,
        huella=huella, isrc=isrc)
    hay_filtros = any(v is not None and v != "" for v in filtros.values()) \
        or bool(desde or hasta)

    al_tope = False
    if orden == "djs" and not q and not cluster:
        # Con este orden no se cruzan filtros: obligaría a recorrer la tabla.
        filas, total, hay_mas = _por_djs(conn, pagina, por_pagina)
    elif orden == "fecha" and dir == "desc" and pagina == 0 and not q \
            and not cluster and not hay_filtros:
        # La primera página de lo de siempre: el índice de la fecha y el LIMIT
        # corta. Las demás, por el índice solo (no lee la tabla para saltar).
        filas, total, hay_mas = _lo_mas_nuevo(conn, pagina, por_pagina)
    else:
        if desde or hasta:
            try:
                desde = date.fromisoformat(desde).isoformat() if desde else None
                hasta = _dia_siguiente(hasta) if hasta else None
            except ValueError:
                desde = hasta = None
        dentro, fuera = _conjuntos(q=q, cluster=cluster, **filtros)
        if orden == "djs":
            filas, total, hay_mas = _por_conjuntos(
                conn, dentro, fuera, desde=desde, hasta=hasta, orden=orden,
                pagina=pagina, por_pagina=por_pagina)
        else:
            filas, total, hay_mas = _por_columna(
                conn, dentro, fuera, desde=desde, hasta=hasta, orden=orden,
                dir=dir, pagina=pagina, por_pagina=por_pagina)
        if q:
            sql, params = _buscando(q)
            al_tope = conn.execute(
                f"SELECT COUNT(*) FROM ({sql})", params).fetchone()[0] >= MAX_COINCIDENCIAS

    djs = _djs_de(conn, [r["acoustic_id"] or r["fingerprint"] for r in filas])
    versiones = _versiones_de(conn, [r["acoustic_id"] for r in filas])
    return {
        "temas": [_fila(r, djs, versiones, con_preview, base_url) for r in filas],
        "pagina": pagina,
        "por_pagina": por_pagina,
        "total": total,
        "hay_mas": hay_mas,
        "orden": orden,
        "dir": dir,
        # Buscando se miran como mucho `tope` temas: con `al_tope`, el total
        # es «o más».
        "tope": MAX_COINCIDENCIAS,
        "al_tope": al_tope,
    }


# Ordenar por una columna, como en la biblioteca (owner, 2026-10-28: «al
# ordenar por columnas ya hace función de filtro»). Cada columna se recorre por
# SU índice —solo el índice—, con el desempate por `rowid` en el mismo sentido
# (el índice ya lo lleva detrás). Lo que no tiene valor va al final en los dos
# sentidos, como en la biblioteca.
#   orden: (índice, términos del ORDER BY, condición de «tiene valor»)
ORDENES = {
    "fecha": ("idx_tracks_analyzed_at", ("analyzed_at",),
              "analyzed_at IS NOT NULL AND analyzed_at != ''"),
    "titulo": ("idx_tracks_titulo", ("title COLLATE NOCASE",),
               "title IS NOT NULL AND title != ''"),
    # El índice de la ficha entero: artista, y dentro su título.
    "artista": ("idx_tracks_ficha", ("artist COLLATE NOCASE", "title COLLATE NOCASE",
                                     "bpm", "analyzed_at"),
                "artist IS NOT NULL AND artist != ''"),
    "genero": ("idx_genre", ("genre",), "genre IS NOT NULL AND genre != ''"),
    "bpm": ("idx_bpm", ("bpm",), "bpm > 0"),
    # 1A, 1B, 2A… 12B: el número delante (CAST lee el prefijo numérico).
    "tonalidad": ("idx_tracks_camelot_orden",
                  ("CAST(camelot AS INTEGER)", "camelot"),
                  "camelot IS NOT NULL AND camelot != ''"),
    "energia": ("idx_energy", ("energy_dj",), "energy_dj IS NOT NULL"),
    "tipo": ("idx_tracks_track_type", ("track_type",),
             "track_type IS NOT NULL AND track_type != ''"),
    "duracion": ("idx_tracks_duration", ("duration",), "duration > 0"),
}


def _por_columna(conn, dentro, fuera, *, desde, hasta, orden, dir, pagina,
                 por_pagina):
    indice, terminos, con_valor = ORDENES[orden]
    dentro = list(dentro)
    if desde or hasta:
        conds, ps = [], []
        if desde:
            conds.append("analyzed_at >= ?")
            ps.append(desde)
        if hasta:
            conds.append("analyzed_at < ?")
            ps.append(hasta)
        dentro.append(("SELECT rowid FROM tracks INDEXED BY idx_tracks_analyzed_at "
                       f"WHERE {' AND '.join(conds)}", ps))
    donde, params = [], []
    for sql, ps in dentro:
        donde.append(f"rowid IN ({sql})")
        params += ps
    for sql, ps in fuera:
        donde.append(f"rowid NOT IN ({sql})")
        params += ps
    where = " AND ".join(donde) or "1"
    orden_sql = ", ".join(f"{t} {dir.upper()}" for t in terminos) + \
        f", rowid {dir.upper()}"
    con = (f"SELECT rowid FROM tracks INDEXED BY {indice} "
           f"WHERE ({con_valor}) AND {where} ORDER BY {orden_sql}")
    # `IS NOT 1` y no `NOT (…)`: con un NULL la condición da NULL, y su NOT
    # también, así que esa fila no salía ni en un lado ni en el otro.
    sin = (f"SELECT rowid FROM tracks INDEXED BY {indice} "
           f"WHERE ({con_valor}) IS NOT 1 AND {where} ORDER BY rowid {dir.upper()}")
    desde_n = pagina * por_pagina
    ids = [r[0] for r in conn.execute(f"{con} LIMIT ? OFFSET ?",
                                      params + [por_pagina + 1, desde_n])]
    if len(ids) <= por_pagina:
        # Se acabaron los que tienen valor: siguen los que no.
        cuantos_con = conn.execute(
            f"SELECT COUNT(*) FROM tracks INDEXED BY {indice} "
            f"WHERE ({con_valor}) AND {where}", params).fetchone()[0]
        falta = por_pagina + 1 - len(ids)
        ids += [r[0] for r in conn.execute(
            f"{sin} LIMIT ? OFFSET ?",
            params + [falta, max(0, desde_n - cuantos_con)])]
    hay_mas = len(ids) > por_pagina
    total = conn.execute(
        f"SELECT COUNT(*) FROM tracks INDEXED BY idx_tracks_analyzed_at WHERE {where}",
        params).fetchone()[0]
    return _filas_en_orden(conn, ids[:por_pagina]), total, hay_mas


def _filas_en_orden(conn, rowids: List[int]) -> List[sqlite3.Row]:
    if not rowids:
        return []
    marcas = ",".join("?" for _ in rowids)
    por_id = {r["rid"]: r for r in conn.execute(
        f"SELECT t.rowid AS rid, {_COLUMNAS} FROM tracks t "
        f"WHERE t.rowid IN ({marcas})", rowids)}
    return [por_id[i] for i in rowids if i in por_id]


def _por_conjuntos(conn, dentro, fuera, *, desde, hasta, orden, pagina,
                   por_pagina):
    """Con cualquier filtro: se recorre el índice de la fecha (solo el
    índice) mirando si cada `rowid` está en los conjuntos, y de la tabla solo
    se leen los temas de la página."""
    donde, params = [], []
    for sql, ps in dentro:
        donde.append(f"rowid IN ({sql})")
        params += ps
    for sql, ps in fuera:
        donde.append(f"rowid NOT IN ({sql})")
        params += ps
    if desde:
        donde.append("analyzed_at >= ?")
        params.append(desde)
    if hasta:
        donde.append("analyzed_at < ?")
        params.append(hasta)
    where = " AND ".join(donde) or "1"
    recorrido = (f"SELECT rowid FROM tracks INDEXED BY idx_tracks_analyzed_at "
                 f"WHERE {where} ORDER BY analyzed_at DESC")

    if orden == "djs":
        # Buscando (o dentro de un cluster): pocas filas, se ordenan aquí.
        ids = [r[0] for r in conn.execute(f"{recorrido} LIMIT ?",
                                          params + [MAX_COINCIDENCIAS])]
        claves = {}
        for i in range(0, len(ids), 500):
            trozo = ids[i:i + 500]
            marcas = ",".join("?" for _ in trozo)
            for r in conn.execute(
                    f"SELECT rowid, acoustic_id, fingerprint, analyzed_at FROM tracks "
                    f"WHERE rowid IN ({marcas})", trozo):
                claves[r[0]] = (r[1] or r[2], r[3] or "")
        djs = _djs_de(conn, [c for c, _ in claves.values()])
        ids.sort(key=lambda i: (djs.get(claves.get(i, ("", ""))[0], 0),
                                claves.get(i, ("", ""))[1]), reverse=True)
        total = len(ids)
        trozo = ids[pagina * por_pagina:(pagina + 1) * por_pagina]
        return _filas_en_orden(conn, trozo), total, (pagina + 1) * por_pagina < total

    ids = [r[0] for r in conn.execute(f"{recorrido} LIMIT ? OFFSET ?",
                                      params + [por_pagina + 1, pagina * por_pagina])]
    hay_mas = len(ids) > por_pagina
    # El total, del mismo recorrido: solo índices.
    total = conn.execute(
        f"SELECT COUNT(*) FROM tracks INDEXED BY idx_tracks_analyzed_at WHERE {where}",
        params).fetchone()[0]
    return _filas_en_orden(conn, ids[:por_pagina]), total, hay_mas


def _lo_mas_nuevo(conn, pagina, por_pagina):
    """Sin filtros: el índice de la fecha hacia atrás y el `LIMIT` corta."""
    filas = conn.execute(
        f"SELECT {_COLUMNAS} FROM tracks t INDEXED BY idx_tracks_analyzed_at "
        f"WHERE +t.bpm > 0 ORDER BY t.analyzed_at DESC LIMIT ? OFFSET ?",
        [por_pagina + 1, pagina * por_pagina]).fetchall()
    hay_mas = len(filas) > por_pagina
    # Solo el índice del BPM, sin leer la tabla.
    total = conn.execute(
        "SELECT COUNT(*) FROM tracks INDEXED BY idx_bpm WHERE bpm > 0").fetchone()[0]
    return filas[:por_pagina], total, hay_mas


def opciones_del_catalogo(conn: sqlite3.Connection) -> dict:
    """Lo que hay para elegir en los filtros, de los índices (sin la tabla)."""
    def grupos(columna, indice):
        return [
            {"valor": r[0], "temas": int(r[1])}
            for r in conn.execute(
                f"SELECT {columna}, COUNT(*) FROM tracks INDEXED BY {indice} "
                f"WHERE {columna} IS NOT NULL AND {columna} != '' "
                f"GROUP BY {columna} ORDER BY COUNT(*) DESC LIMIT 300")
        ]
    return {
        "generos": grupos("genre", "idx_genre"),
        "plataformas": grupos("platform", "idx_tracks_platform"),
        "tipos": list(TIPOS),
        "motores": list(MOTORES),
    }


def _por_djs(conn, pagina, por_pagina):
    """Una fila por tema, de los que más DJs tienen a los que menos."""
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM track_popularity").fetchone()[0]
        claves = conn.execute(
            "SELECT fingerprint FROM track_popularity "
            "ORDER BY dj_count DESC, last_analyzed DESC LIMIT ? OFFSET ?",
            [por_pagina + 1, pagina * por_pagina]).fetchall()
    except sqlite3.OperationalError:
        return [], 0, False
    hay_mas = len(claves) > por_pagina
    filas = []
    for (clave,) in claves[:por_pagina]:
        # La clave es el cluster si el tema lo tiene, y si no su huella.
        r = conn.execute(
            f"SELECT {_COLUMNAS} FROM tracks t WHERE t.acoustic_id = ? AND +t.bpm > 0 "
            f"ORDER BY t.analyzed_at DESC LIMIT 1", [clave]).fetchone()
        if r is None:
            r = conn.execute(
                f"SELECT {_COLUMNAS} FROM tracks t WHERE t.fingerprint = ? AND +t.bpm > 0 "
                f"LIMIT 1", [clave]).fetchone()
        if r is not None:
            filas.append(r)
    return filas, total, hay_mas
