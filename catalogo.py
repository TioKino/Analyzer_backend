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

  - **Por fecha** (lo de por defecto): `idx_tracks_analyzed_at` hacia atrás y
    el `LIMIT` corta. Con filtros de género, BPM o tonalidad SQLite sigue por
    ese índice y descarta: caro solo si el filtro casa con muy pocos.
  - **Buscando** (`q`): las filas que casan salen de `idx_tracks_ficha` (que
    cubre artista, título y BPM), sin tocar la tabla.
  - **Por nº de DJs**: desde `track_popularity` (filas pequeñas), y cada clave
    a su tema por `idx_acoustic` o `idx_tracks_fingerprint`. Una fila por TEMA
    (el cluster), no por fichero. Con este orden no hay filtros: cruzarlos
    obligaría a recorrer la tabla.

El total solo se da cuando sale de un índice (sin filtros, o buscando);
con filtros es `None` y vale `hay_mas`.
"""

import json
import sqlite3
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
    return {
        "bpm_source": d.get("bpm_source"),
        "key_source": d.get("key_source"),
        "genre_source": d.get("genre_source"),
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


def pagina_del_catalogo(
    conn: sqlite3.Connection,
    *,
    q: str = "",
    orden: str = "fecha",
    genero: Optional[str] = None,
    bpm_min: Optional[float] = None,
    bpm_max: Optional[float] = None,
    camelot: Optional[str] = None,
    cluster: Optional[str] = None,
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
    orden = "djs" if orden == "djs" else "fecha"
    filtros = bool(genero or camelot or bpm_min is not None or bpm_max is not None)

    if orden == "djs" and not q and not cluster:
        filas, total, hay_mas = _por_djs(conn, pagina, por_pagina)
        filtros = False
    else:
        filas, total, hay_mas = _por_fecha_o_busqueda(
            conn, q=q, orden=orden, genero=genero, bpm_min=bpm_min,
            bpm_max=bpm_max, camelot=camelot, cluster=cluster,
            pagina=pagina, por_pagina=por_pagina, filtros=filtros)

    djs = _djs_de(conn, [r["acoustic_id"] or r["fingerprint"] for r in filas])
    versiones = _versiones_de(conn, [r["acoustic_id"] for r in filas])
    return {
        "temas": [_fila(r, djs, versiones, con_preview, base_url) for r in filas],
        "pagina": pagina,
        "por_pagina": por_pagina,
        "total": total,
        "hay_mas": hay_mas,
        "orden": orden,
        # Buscando, como mucho se cuentan estas: un total igual es «o más».
        "tope": MAX_COINCIDENCIAS,
    }


def _por_fecha_o_busqueda(conn, *, q, orden, genero, bpm_min, bpm_max, camelot,
                          cluster, pagina, por_pagina, filtros):
    donde = ["+t.bpm > 0"]
    params: list = []
    if q:
        # Las que casan, del índice de la ficha (cubre artista, título y BPM).
        donde.append(
            "t.rowid IN (SELECT rowid FROM tracks INDEXED BY idx_tracks_ficha "
            "WHERE (artist LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\') "
            "AND bpm > 0 LIMIT ?)")
        like = _como_like(q)
        params += [like, like, MAX_COINCIDENCIAS]
    if cluster:
        donde.append("t.acoustic_id = ?")
        params.append(cluster)
    if genero:
        donde.append("t.genre = ? COLLATE NOCASE")
        params.append(genero)
    if camelot:
        donde.append("t.camelot = ?")
        params.append(camelot.upper())
    if bpm_min is not None:
        donde.append("t.bpm >= ?")
        params.append(float(bpm_min))
    if bpm_max is not None:
        donde.append("t.bpm <= ?")
        params.append(float(bpm_max))
    where = " AND ".join(donde)

    if orden == "djs":
        # Buscando (o dentro de un cluster): pocas filas, se ordenan aquí.
        filas = conn.execute(
            f"SELECT {_COLUMNAS} FROM tracks t WHERE {where} LIMIT ?",
            params + [MAX_COINCIDENCIAS]).fetchall()
        djs = _djs_de(conn, [r["acoustic_id"] or r["fingerprint"] for r in filas])
        filas.sort(key=lambda r: (djs.get(r["acoustic_id"] or r["fingerprint"], 0),
                                  r["analyzed_at"] or ""), reverse=True)
        total = len(filas)
        trozo = filas[pagina * por_pagina:(pagina + 1) * por_pagina]
        return trozo, total, (pagina + 1) * por_pagina < total

    filas = conn.execute(
        f"SELECT {_COLUMNAS} FROM tracks t WHERE {where} "
        f"ORDER BY t.analyzed_at DESC LIMIT ? OFFSET ?",
        params + [por_pagina + 1, pagina * por_pagina]).fetchall()
    hay_mas = len(filas) > por_pagina
    filas = filas[:por_pagina]

    total = None
    if q and not filtros and not cluster:
        total = conn.execute(
            "SELECT COUNT(*) FROM (SELECT rowid FROM tracks INDEXED BY idx_tracks_ficha "
            "WHERE (artist LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\') "
            "AND bpm > 0 LIMIT ?)", [_como_like(q), _como_like(q), MAX_COINCIDENCIAS],
        ).fetchone()[0]
    elif cluster and not q and not filtros:
        total = conn.execute(
            "SELECT COUNT(*) FROM tracks WHERE acoustic_id = ? AND +bpm > 0",
            [cluster]).fetchone()[0]
    elif not q and not filtros and not cluster:
        # Solo el índice del BPM, sin leer la tabla.
        total = conn.execute(
            "SELECT COUNT(*) FROM tracks INDEXED BY idx_bpm WHERE bpm > 0"
        ).fetchone()[0]
    return filas, total, hay_mas


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
