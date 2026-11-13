"""
Search, library and single-track endpoints for DJ Analyzer Pro API.

PASO 2 del troceo de main.py (review 2026-06-29). Estos endpoints
(/search-analyzed, /library/all, /track/{id}) eran inline en
main.py; aqui se mueven VERBATIM (mismo comportamiento, sin cambio de
logica) a un router montado con `include_router`. Las dependencias que
antes eran globales de main.py se inyectan via `init(database, camelot_compatible)`:

  - `db`: la instancia AnalysisDB (analysis.db).
  - `CAMELOT_COMPATIBLE`: el dict de keys compatibles (vive en
    similar_tracks_endpoint.py, opcional — si ese modulo no carga,
    main.py inyecta {}).

A diferencia de los 5 modulos muertos que se borraron en el paso 1
(#28), ESTE router SI se monta: main.py hace `init(...)` +
`app.include_router(search_router)` y BORRA los endpoints inline. No hay
duplicacion stale (la causa del doble incidente de /admin/reset-database).
"""

import json
import logging
import re
import unicodedata
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from validation import sanitize_string, validate_limit

logger = logging.getLogger(__name__)


def _require_admin(request: Request) -> None:
    """Auth Bearer contra ADMIN_TOKEN para los endpoints destructivos de aquí.

    Mismo esquema que `main._verify_admin_bearer` y `sync_endpoints._verify_admin`,
    reimplementado en local para no crear un import circular
    (main importa este módulo). En dev sin ADMIN_TOKEN se deja pasar; en
    Render/Railway sin token se falla rápido con 500, igual que el resto.
    """
    import hmac as _hmac
    import os as _os

    token = _os.environ.get('ADMIN_TOKEN', '')
    if not token:
        if _os.getenv('RENDER') or _os.getenv('RAILWAY_ENVIRONMENT'):
            raise HTTPException(500, "ADMIN_TOKEN required in production")
        return
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer ') or not _hmac.compare_digest(auth[7:], token):
        raise HTTPException(401, "Admin token requerido")


# ── Dependencias inyectadas desde main.py ────────────────────
db = None
CAMELOT_COMPATIBLE = {}


def init(database, camelot_compatible=None):
    """Inyecta la instancia de BD y el mapa Camelot desde main.py.

    Debe llamarse ANTES de `app.include_router(search_router)`.
    """
    global db, CAMELOT_COMPATIBLE
    db = database
    if camelot_compatible is not None:
        CAMELOT_COMPATIBLE = camelot_compatible


# ── Router ───────────────────────────────────────────────────

search_router = APIRouter(tags=["search"])


# ==================== ENDPOINTS DE BUSQUEDA ====================
#
# Aquí vivían doce rutas PÚBLICAS más, quitadas el 2026-11-12 (auditoría, S6):
# `/search/artist|genre|bpm|energy|key|compatible|track-type`,
# `/search/advanced`, `/library/artists|genres|stats` y `GET /track/{id}`.
# Ninguna versión del cliente las llamó nunca (comprobado en el historial
# entero de `lib/`, desde el primer commit), ni la web, ni los scripts, ni el
# motor local. Lo que sí hacían: dar a cualquiera, sin credencial, las huellas,
# los artistas y los nombres de fichero de la memoria colectiva (lo mismo por
# lo que se cerró `/library/all` en agosto, por trozos), y hacerlo con SQLite
# dentro del event loop del único proceso. Con esas huellas se escribía en lo
# que aún acepta escrituras sin acreditar (notas, previews).
#
# **No las resucites para una función nueva sin credencial y sin threadpool.**
# Lo que el cliente usa de verdad para buscar es `/search-analyzed` (uno) y
# `/search-analyzed/batch` (lote), abajo. Lo ata
# `test_las_rutas_sin_uso_no_vuelven.py`.


# ── La ficha por nombre: el mismo TEMA y la misma VERSION ──
# Cada fuente escribe el titulo a su manera: Shazam «The Age Of Love (Jam &
# Spoon Watch Out For Stella Mix)», un tag «The Age of Love - Jam and Spoon
# Watch Out for Stella Mix», Beatport «Rave (Original Mix)». Hasta el
# 2026-09-30 se buscaba con `LIKE '%titulo%'` sobre el titulo casi entero, y
# un parentesis frente a un guion bastaba para no encontrar nada (Age of Love
# salio dos veces sin ficha en el iPhone del owner, con el tema analizado).
# Y al reves: el titulo sin version casaba con cualquier remix, y la ficha
# enseñaba el BPM y la tonalidad de OTRA version.
#
# Hoy el titulo se parte en TEMA y VERSION, los dos normalizados (sin acentos,
# `&` = `and`, sin puntuacion), y casa lo que tiene el mismo tema y la misma
# version. «Original Mix», «Extended Mix», «Radio Edit»… son el original.

# Versiones que son el original: el mismo corte, mas largo o mas corto.
_VERSIONES_ORIGINALES = frozenset({
    'original', 'original mix', 'original version', 'extended',
    'extended mix', 'extended version', 'radio edit', 'radio version',
    'radio mix', 'edit', 'mix',
    # Lo que Shazam y las tiendas ponen al tema de un recopilatorio mezclado
    # o a su corte corto: la misma grabacion. Con «(Mixed)» como version,
    # «Knights of the Jaguar (Mixed)» no casaba con el del DJ (2026-10-02).
    'mixed', 'mixed version', 'mix cut', 'radio cut',
})

# Palabras que no distinguen una version de otra: «X Remix» y «X Mix» son la
# misma.
_PALABRAS_DE_RELLENO = frozenset({
    'remix', 'mix', 'rmx', 'version', 'edit', 'and', 'the',
})

# Lo que va tras « - » es la version solo si lo parece.
_PARECE_VERSION = re.compile(
    r'\b(mix|remix|rmx|edit|version|dub|rework|vip|bootleg|remaster(ed)?)\b',
    re.IGNORECASE)

# Palabras que dicen QUE CLASE de version es, no DE QUIEN: «(Dub)» no puede
# hacerse pasar por «(Adam Beyer Dub)» por estar dentro (`misma_version`).
_TIPOS_DE_VERSION = frozenset({
    'dub', 'vip', 'rework', 'bootleg', 'remaster', 'remastered', 'club',
    'vocal', 'instrumental', 'acapella', 'extended', 'original', 'radio',
})

_ENTRE_PARENTESIS = re.compile(r'[(\[]([^)\]]*)[)\]]')
_FEAT = re.compile(r'\s(?:feat|ft|featuring)\.?\s.*$', re.IGNORECASE)
_SEPARA_ARTISTAS = re.compile(
    r'\s*(?:,|;|/|&|\bfeat\b\.?|\bft\b\.?|\bfeaturing\b|\bvs\b\.?|'
    r'\bx\b|\band\b|\bwith\b)\s*', re.IGNORECASE)


def _plano(texto: Optional[str]) -> str:
    """Minusculas, sin acentos, `&` = `and`, sin puntuacion ni espacios de
    mas."""
    t = unicodedata.normalize('NFKD', texto or '')
    t = ''.join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace('&', ' and ')
    t = re.sub(r'[\W_]+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def tema_y_version(titulo: Optional[str]):
    """(tema, version) de un titulo, normalizados. La version es un conjunto
    de palabras, vacio si es el original."""
    t = titulo or ''
    # Entre parentesis siempre es la version («(base)» tambien). Entre
    # corchetes solo si lo parece: ahi van el sello («[Drumcode]») y la
    # basura de las webs de descarga («[www.x.tk]»), que no son una version.
    partes = [m.group(1) for m in _ENTRE_PARENTESIS.finditer(t)
              if m.group(0).startswith('(') or _PARECE_VERSION.search(m.group(1))]
    tema = _ENTRE_PARENTESIS.sub(' ', t)
    guion = re.match(r'^(.*?)\s+[-\u2013\u2014]\s+(.+)$', tema)
    if guion and _PARECE_VERSION.search(guion.group(2)):
        tema = guion.group(1)
        partes.append(guion.group(2))
    tema = _FEAT.sub('', f' {tema} ').strip()

    version = set()
    for parte in partes:
        p = _plano(parte)
        if not p or re.match(r'^(feat|ft|featuring|with)\b', p):
            continue  # un «(feat. X)» no es una version
        if p in _VERSIONES_ORIGINALES:
            continue
        palabras = set(p.split()) - _PALABRAS_DE_RELLENO
        # «(Remix)» a secas sigue siendo una version, no el original.
        version |= palabras or {f'<{p}>'}
    return _plano(tema), frozenset(version)


def _artistas(artista: Optional[str]) -> frozenset:
    return frozenset(
        a for a in (_plano(x) for x in _SEPARA_ARTISTAS.split(artista or ''))
        if a)


def _contiene(largo: str, corto: str) -> bool:
    """`corto` dentro de `largo` por palabras enteras, y con algo de cuerpo:
    «dj» no casa con cualquier cosa."""
    return len(corto) >= 4 and f' {corto} ' in f' {largo} '


def _mismo_artista(buscado: str, guardado: str) -> bool:
    b, g = _plano(buscado), _plano(guardado)
    if not b or not g:
        return False
    if b == g or _contiene(g, b) or _contiene(b, g):
        return True
    return bool(_artistas(buscado) & _artistas(guardado))


def misma_version(a: frozenset, b: frozenset) -> bool:
    """La misma version, aunque una lleve el nombre mas completo que la otra:
    «(Jam & Spoon Mix)» y «(Jam & Spoon Watch Out For Stella Mix)» son la
    misma; «(Charlotte de Witte Remix)» y la que añade a Enrico Sangiuliano,
    tambien. El original (vacio) solo es el original, y la corta tiene que
    llevar un nombre, no solo la clase de version. Espejo de
    `mismaVersion` en el movil (`track_match.dart`)."""
    if a == b:
        return True
    if not a or not b:
        return False
    corta, larga = (a, b) if len(a) <= len(b) else (b, a)
    # La corta tiene que nombrar a alguien: «(Dub)» dentro de «(Adam Beyer
    # Dub)» no dice que sea la de Adam Beyer.
    return corta <= larga and bool(corta - _TIPOS_DE_VERSION)


def _parecido_de_titulo(buscado, guardado) -> int:
    """2 = el mismo tema y version; 1 = el tema dentro del otro (un numero de
    pista delante, un subtitulo) y la misma version; 0 = no."""
    tema_b, version_b = buscado
    tema_g, version_g = tema_y_version(guardado)
    if not tema_b or not tema_g or not misma_version(version_b, version_g):
        return 0
    if tema_b == tema_g:
        return 2
    if _contiene(tema_g, tema_b) or _contiene(tema_b, tema_g):
        return 1
    return 0


def _palabra_para_like(texto: str, dentro_de: Optional[str] = None
                       ) -> Optional[str]:
    """La palabra mas larga que se puede pedir con LIKE, tal cual esta en el
    texto de entrada y sin acentos (LIKE no los pliega: «tiesto» no encuentra
    «Tiësto», ni «cafe» a «Café»). Con [dentro_de], solo palabras del tema
    (no de la version)."""
    del_tema = set(dentro_de.split()) if dentro_de is not None else None
    candidatas = [
        w for w in re.split(r'[\W_]+', (texto or '').lower())
        if len(w) >= 3 and w.isascii() and w not in _PALABRAS_DE_RELLENO
        and (del_tema is None or w in del_tema)]
    return max(candidatas, key=len) if candidatas else None


def _aplanar(track_dict: Optional[dict]) -> Optional[dict]:
    """El `analysis_json` al nivel de arriba y fuera el crudo: la ficha lista
    para el cliente, sin que tenga que re-parsear un blob."""
    if not track_dict:
        return None
    crudo = track_dict.pop('analysis_json', None)
    if crudo:
        try:
            detalle = json.loads(crudo) if isinstance(crudo, str) else crudo
            if isinstance(detalle, dict):
                track_dict.update(detalle)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("analysis_json corrupto en track %s: %s",
                           track_dict.get('id'), e)
    return track_dict


# Las dos busquedas de `buscar_analizado` van por el indice
# `idx_tracks_ficha` y eligen el tema DENTRO de el (`rowid`), sin leer la
# tabla con su `analysis_json` salvo la fila elegida. Sin mayusculas con
# `COLLATE NOCASE` y `LIKE`: con `LOWER(artist)` se recorria la tabla entera
# (35,5 s en frio, 2026-09-29). Ver el comentario del indice en database.py.
#
# El `INDEXED BY` NO sobra: sin el, con los `LIKE` SQLite prefiere otro indice
# que tambien sirve —el de `analyzed_at`, para ahorrarse ordenar, o el de
# `bpm`, por el `bpm > 0`— y cualquiera de los dos lee la tabla fila a fila:
# medido con 120.000 filas, 200 ms frente a 8 ms.
FICHA_EXACTA = (
    "SELECT * FROM tracks WHERE rowid = ("
    "SELECT rowid FROM tracks INDEXED BY idx_tracks_ficha "
    "WHERE artist = ? COLLATE NOCASE AND title = ? COLLATE NOCASE "
    "AND bpm IS NOT NULL AND bpm > 0 "
    "ORDER BY analyzed_at DESC LIMIT 1)"
)
# Candidatos por una palabra del artista y otra del titulo; el tema y la
# version se comparan despues, en Python (`_parecido_de_titulo`).
FICHA_CANDIDATOS = (
    "SELECT rowid, artist, title FROM tracks INDEXED BY idx_tracks_ficha "
    "WHERE artist LIKE ? AND title LIKE ? "
    "AND bpm IS NOT NULL AND bpm > 0 "
    "ORDER BY analyzed_at DESC LIMIT 500"
)


def buscar_analizado(artist: str, title: str,
                     isrc: Optional[str] = None) -> Optional[dict]:
    """La ficha de un tema que ALGUIEN ya analizo, o None.

    Una sola regla para `/search-analyzed` (Escuchar y el relleno de ghosts) y
    `/recognize` (AudD y Shazam).

    Orden: ISRC (identidad exacta de la grabacion) → artista y titulo exactos
    → el mismo tema y la misma VERSION por nombre (`tema_y_version`), con el
    artista tolerante a colaboraciones. Nunca otra version: la ficha de un
    remix no es la del original. Solo filas con bpm > 0: una deteccion de
    Escuchar no es un analisis. Entre iguales gana el analisis mas reciente.
    """
    if isrc:
        t = db.get_analyzed_track_by_isrc(isrc)
        if t:
            return _aplanar(t)

    if not (artist or '').strip() or not (title or '').strip():
        return None
    cursor = db.conn.cursor()
    cursor.execute(FICHA_EXACTA, (artist.strip(), title.strip()))
    row = cursor.fetchone()
    if row:
        return _aplanar(db._row_to_dict(row))

    buscado = tema_y_version(title)
    if not buscado[0]:
        return None
    palabra_artista = _palabra_para_like(artist)
    palabra_tema = _palabra_para_like(title, buscado[0])
    # Primero por las dos palabras (lo normal, pocos candidatos); si no sale,
    # por una sola: la otra puede estar guardada con acento («Tiësto» no casa
    # con `LIKE '%tiesto%'`). Cada pasada recorre el indice, no la tabla.
    pasadas = []
    for par in ((palabra_artista, palabra_tema), (None, palabra_tema),
                (palabra_artista, None)):
        if any(par) and par not in pasadas:
            pasadas.append(par)
    mejor_rowid = None
    for p_artista, p_tema in pasadas:
        cursor.execute(FICHA_CANDIDATOS, (
            f"%{p_artista}%" if p_artista else '%',
            f"%{p_tema}%" if p_tema else '%'))
        mejor = 0
        for rowid, art, tit in cursor.fetchall():
            if not _mismo_artista(artist, art):
                continue
            parecido = _parecido_de_titulo(buscado, tit)
            if parecido > mejor:  # a igualdad, el mas reciente (en orden)
                mejor, mejor_rowid = parecido, rowid
                if parecido == 2:
                    break
        if mejor_rowid is not None:
            break
    if mejor_rowid is None:
        return None
    cursor.execute("SELECT * FROM tracks WHERE rowid = ?", (mejor_rowid,))
    row = cursor.fetchone()
    return _aplanar(db._row_to_dict(row)) if row else None


@search_router.get("/search-analyzed")
async def search_analyzed_track(
    artist: str = Query(..., description="Nombre del artista"),
    title: str = Query(..., description="Titulo del track"),
    isrc: Optional[str] = Query(None, description="ISRC (identidad exacta, opcional)")
):
    """
    Busca si un track ya fue analizado por algun usuario.
    Devuelve TODA la informacion del analisis si existe (ver `buscar_analizado`).

    Returns:
        - found: bool - Si se encontro el track
        - track: dict - Toda la informacion del analisis (si existe)
        - in_collective: bool - Si esta en la memoria colectiva
    """
    artist_clean = sanitize_string(artist, max_length=200, allow_empty=False, field_name="artist")
    title_clean = sanitize_string(title, max_length=200, allow_empty=False, field_name="title")

    try:
        ficha = buscar_analizado(artist_clean, title_clean, isrc)
        if ficha:
            return {"found": True, "in_collective": True, "track": ficha}
        return {"found": False, "in_collective": False, "track": None}
    except Exception as e:
        logger.error(f"Error en search-analyzed: {e}")
        return {
            "found": False,
            "in_collective": False,
            "track": None,
            "error": str(e)
        }

# ── POST /search-analyzed/batch — las fichas de un set de una vez ─────────
# El modo set de Escuchar guarda un tracklist (artista, título, ISRC) sin la
# ficha de cada tema: pedirla por /search-analyzed era una llamada por tema
# (PENDING, 13-oct). Desde el 2026-10-24 el set la pide al abrirse, en UN lote,
# para enseñar BPM y tonalidad, abrir la ficha de un tema y «Rehazlo con lo
# tuyo» (el tema más parecido de tu biblioteca para los que no tienes). La
# regla es la de siempre (`buscar_analizado`), y la ficha va CORTA: un set
# entero con el análisis completo de cada tema serían megas para enseñar dos
# números.

MAX_FICHAS_POR_LOTE = 100

CAMPOS_DE_LA_FICHA_CORTA = (
    'artist', 'title', 'bpm', 'key', 'camelot', 'energy_dj', 'genre',
    'subgenre', 'track_type', 'duration', 'isrc', 'fingerprint', 'label',
    'has_vocals', 'has_drop', 'has_intro', 'has_outro', 'has_breakdown',
    'has_heavy_bass', 'bpm_source', 'key_source', 'genre_source',
)


def ficha_corta(ficha: Optional[dict]) -> Optional[dict]:
    """Lo que hace falta para enseñar y mezclar un tema, sin el análisis
    entero."""
    if not ficha:
        return None
    return {k: ficha[k] for k in CAMPOS_DE_LA_FICHA_CORTA if k in ficha}


class TemaParaFicha(BaseModel):
    artist: str = ''
    title: str = ''
    isrc: Optional[str] = None


class LoteDeFichas(BaseModel):
    temas: List[TemaParaFicha]


def fichas_de(temas: List[TemaParaFicha]) -> List[Optional[dict]]:
    """Una ficha corta (o None) por tema, en el MISMO orden."""
    fichas: List[Optional[dict]] = []
    for t in temas:
        artista = sanitize_string(t.artist or '', max_length=200)
        titulo = sanitize_string(t.title or '', max_length=200)
        isrc = (t.isrc or '').strip() or None
        try:
            fichas.append(ficha_corta(buscar_analizado(artista, titulo, isrc)))
        except Exception as e:  # un tema raro no tumba el lote
            logger.warning("search-analyzed/batch: %s – %s: %s",
                           artista, titulo, e)
            fichas.append(None)
    return fichas


@search_router.post("/search-analyzed/batch")
async def search_analyzed_batch(lote: LoteDeFichas):
    """Las fichas de hasta `MAX_FICHAS_POR_LOTE` temas, en el mismo orden que
    llegan (`null` donde nadie lo ha analizado). Fuera del event loop: son
    hasta cien búsquedas en la BD."""
    if len(lote.temas) > MAX_FICHAS_POR_LOTE:
        raise HTTPException(413, f"Como mucho {MAX_FICHAS_POR_LOTE} temas por lote")
    fichas = await run_in_threadpool(fichas_de, lote.temas)
    logger.info("[Fichas] lote: %d temas, %d con ficha", len(fichas),
                sum(1 for f in fichas if f))
    return {"fichas": fichas}



# ==================== ENDPOINTS DE BIBLIOTECA ====================

@search_router.get("/library/all")
async def get_all_tracks(request: Request, limit: int = Query(1000, ge=1, le=5000)):
    """Volcado completo de la base colectiva. SOLO admin.

    Devolvía hasta 5000 tracks con sus fingerprints a cualquiera que llamara.
    La memoria colectiva es el activo diferencial del producto y se podía
    clonar con un curl; además los fingerprints obtenidos aquí servían para
    cebar el fallback online de /artwork (auditoría 2026-08-09, SEC-13).
    Ningún cliente lo consume —el catálogo del panel admin es /admin/catalogo— así que
    cerrarlo no rompe nada. Las agregaciones (/library/artists, /genres,
    /stats) se quitaron el 2026-11-12: tampoco las llamaba nadie.
    """
    _require_admin(request)
    limit = validate_limit(limit, max_limit=5000)

    return {"tracks": db.get_all_tracks(limit)}


@search_router.delete("/track/{track_id}")
async def delete_track(track_id: str, request: Request):
    """Eliminar un track de la base de datos. SOLO admin.

    `tracks` es la memoria colectiva COMPARTIDA por todos los usuarios, no la
    biblioteca de quien llama. Sin auth, cualquiera podía borrar el análisis de
    cualquier track para todo el mundo (verificado en la auditoría 2026-08-09:
    DELETE anónimo -> 200 y la fila desaparecía). Ningún cliente consume este
    endpoint; se mantiene como herramienta de mantenimiento del owner.
    """
    _require_admin(request)
    deleted = db.delete_track(track_id)
    if not deleted:
        raise HTTPException(404, "Track no encontrado")
    return {"status": "ok", "message": "Track eliminado"}
