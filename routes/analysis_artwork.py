"""
Cache-lookup + artwork endpoints for DJ Analyzer Pro API.

PASO 5 del troceo de main.py (review 2026-06-29). Bloque movido VERBATIM
desde main.py (mismo comportamiento, sin cambio de logica):

  - POST /check-analyzed                       por NOMBRE: todo «no analizado» (2026-10-02)
  - POST /check-analyzed-by-fingerprint        que huellas ya estan analizadas (dedup)
  - GET  /analysis/by-fingerprint/{fingerprint}  hidrata cache local sin re-subir
  - GET  /analysis/{filename:path}             por NOMBRE: 410 (2026-10-02)
  - HEAD /artwork/{track_id}                   pre-check de existencia (sin fallback online)
  - GET  /artwork/{track_id}                   sirve artwork (cache -> fallback online)
  - POST /artwork/upload/{fingerprint}         sube artwork desde el motor local

Dependencias antes globales de main.py se inyectan con init(...). Se inyectan
(no se re-importan) para usar EXACTAMENTE lo que main resolvio:
  - db: instancia AnalysisDB.
  - is_analysis_current: helper de frescura (main.py:_is_analysis_current),
    usado tambien por /analyze -> se queda en main, aqui se inyecta.
  - artwork_cache_dir: ARTWORK_CACHE_DIR resuelto por main (artwork module o
    fallback de config).
  - search_artwork_online / save_artwork_to_cache: funcs de artwork_and_cuepoints
    (None si ARTWORK deshabilitado, igual que main). save_* solo se invoca tras
    `if ... search_artwork_online:`, asi que None nunca se llama.

_artwork_media_type y CheckAnalyzedByFingerprintRequest se mueven con el bloque
(grep confirmo que solo se usaban aqui).

Como pasos 2-4: el router SI se monta (init + include_router) y los endpoints
inline se BORRAN -> sin duplicacion stale.
"""

import json
import logging
import os
import re
from typing import List, Optional

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, Response

from validation import artwork_online_allowed, get_client_ip
from pydantic import BaseModel
from sync_endpoints import dispositivo_del_token

logger = logging.getLogger(__name__)

# ── Dependencias inyectadas desde main.py ────────────────────
db = None
_is_analysis_current = None
ARTWORK_CACHE_DIR = None
search_artwork_online = None
save_artwork_to_cache = None
# `buscar_portada` (artwork_and_cuepoints): como search_artwork_online, pero
# dice si el «no» es DEFINITIVO. None = no se recuerda nada.
buscar_portada = None
# Solo en el motor local: consulta a Render el analisis de un fingerprint.
# None en Render (no se consulta a si mismo).
fetch_render_cache = None
# Solo en el motor local: de un LOTE de huellas, cuales tiene Render con un
# analisis que sirva (`main._precheck_en_render`). None en Render.
precheck_en_render = None
# Solo en el motor local: las fichas de un LOTE de huellas que tiene Render y
# valen aquí (`main._fichas_en_render`). None en Render.
fichas_en_render = None
# Solo en Render: apunta al aparato como un DJ más que tiene esos temas
# (`db.registrar_analistas`). None en el motor local: su BD no es la de la
# comunidad, y lo que él sabe llega a Render reenviando el pre-check.
registrar_analistas = None


def init(database, is_analysis_current, artwork_cache_dir,
         search_online=None, save_to_cache=None, render_cache_lookup=None,
         buscar=None, lo_mejor=None, render_precheck=None,
         registrar=None, render_fichas=None):
    """Inyecta deps desde main.py. Llamar ANTES de include_router(router)."""
    global db, _is_analysis_current, ARTWORK_CACHE_DIR, buscar_portada
    global search_artwork_online, save_artwork_to_cache, fetch_render_cache
    global lo_mejor_para, precheck_en_render, registrar_analistas
    global fichas_en_render
    precheck_en_render = render_precheck
    fichas_en_render = render_fichas
    registrar_analistas = registrar
    buscar_portada = buscar
    lo_mejor_para = lo_mejor
    db = database
    _is_analysis_current = is_analysis_current
    ARTWORK_CACHE_DIR = artwork_cache_dir
    search_artwork_online = search_online
    save_artwork_to_cache = save_to_cache
    fetch_render_cache = render_cache_lookup


# ── Router ───────────────────────────────────────────────────
router = APIRouter(tags=["analysis-artwork"])


# `main._lo_mejor_para`: el mejor análisis del cluster MÁS lo que los programas
# de DJ de cualquiera dicen del fichero (y en el motor local, preguntado a
# Render). None en los tests que montan el router sin main.
lo_mejor_para = None


def _merge_cluster_best_into(result, fingerprint, fila):
    """Sobre un dict de analisis, adopta lo MAS FIABLE que sabe la memoria
    colectiva de este fichero: otra version del mismo audio con fuente
    superior, o lo que dice de él el Rekordbox/Traktor/VirtualDJ de alguien.
    Solo sube de fiabilidad (compara analysis_ranking); best-effort, nunca
    lanza."""
    if not isinstance(result, dict):
        return
    try:
        from analysis_ranking import get_source_priority
        if lo_mejor_para is not None:
            best = lo_mejor_para(fingerprint, fila.get('acoustic_id'),
                                 fila.get('chromaprint'), fila.get('duration'))
        elif fila.get('acoustic_id'):
            best = db.best_cluster_analysis(fila['acoustic_id'])
        else:
            best = None
        if not best:
            return
        if ('bpm' in best and get_source_priority(best.get('bpm_source'))
                > get_source_priority(result.get('bpm_source'))):
            result['bpm'] = best['bpm']
            result['bpm_source'] = best['bpm_source']
        # La rejilla del programa va con SU BPM (ver `_adopt_better_metadata`).
        if (best.get('first_beat') and best.get('bpm_source')
                and result.get('bpm_source') == best['bpm_source']):
            result['first_beat'] = best['first_beat']
            result['grid_source'] = best['bpm_source']
        if ('key' in best and get_source_priority(best.get('key_source'))
                > get_source_priority(result.get('key_source'))):
            result['key'] = best['key']
            result['key_source'] = best['key_source']
            if best.get('camelot'):
                result['camelot'] = best['camelot']
        if ('genre' in best and get_source_priority(best.get('genre_source'))
                > get_source_priority(result.get('genre_source'))):
            result['genre'] = best['genre']
            result['genre_source'] = best['genre_source']
    except Exception as e:  # noqa: BLE001 - best-effort
        logger.warning(f"[Cluster] merge best (lookup) fallo: {e}")


@router.post("/check-analyzed")
async def check_analyzed(filenames: list[str]):
    """POR NOMBRE NO SE CONTESTA: todo sale como «no analizado».

    Lo usaba el import del móvil hasta la auditoría del 2026-10-02, y con lo
    que contestaba pedía `/analysis/{nombre}` y se quedaba con el análisis de
    cualquier usuario cuyo fichero se llamara igual — BPM, tonalidad, nombre y
    la huella de otro audio. Es lo que SEC-01 cerró en `/analyze`, por otra
    puerta. Los móviles de antes siguen llamándolo: así suben el fichero y
    `/analyze` lo resuelve por huella, que es la identidad de verdad. El
    cliente nuevo pregunta por `/check-analyzed-by-fingerprint`.

    Y de paso deja de recorrer la tabla: `filename` no tenía índice y eran
    las ~122.000 filas UNA vez POR FICHERO, en el único proceso de Render.
    """
    return {
        "analyzed": [],
        "not_analyzed": list(filenames),
        "total": len(filenames),
        "analyzed_count": 0,
        "not_analyzed_count": len(filenames),
    }


class AcousticClustersRequest(BaseModel):
    fingerprints: List[str]


@router.post("/acoustic-clusters")
async def acoustic_clusters(request: AcousticClustersRequest):
    """Cluster acustico de un lote de huellas — para detectar DUPLICADOS.

    El caso que resuelve: el mismo tema en dos ficheros distintos (otro codec,
    otro bitrate, otro tag, rippeado de otro sitio). El MD5 del contenido no
    los junta porque son bytes distintos, y el nombre tampoco. El chromaprint
    si: agrupa por SONIDO, con Hamming tolerante y acotado por duracion.

    El cliente manda SUS huellas y agrupa la respuesta por `acoustic_id`: dos o
    mas huellas suyas en el mismo cluster = el mismo tema repetido en su
    biblioteca. No se devuelve nada de otros usuarios — el cluster es un id
    opaco y el agrupado lo hace el cliente sobre lo suyo.

    `clusters` NO incluye las huellas sin cluster. Eso es a proposito: «no
    tiene huella acustica todavia» y «no tiene duplicados» son cosas distintas,
    y devolver `null` para las primeras las haria indistinguibles de las
    segundas. `without_cluster` las nombra aparte para que el cliente pueda
    decir «de estos N no lo se» en vez de afirmar que estan limpios.

    Maximo 500 por peticion, igual que /check-analyzed-by-fingerprint.
    """
    fps = [f for f in (request.fingerprints or []) if f]
    if len(fps) > 500:
        raise HTTPException(400, "Máximo 500 fingerprints por petición")

    clusters = db.acoustic_ids_for(fps)
    sin_cluster = [f for f in fps if f not in clusters]
    return {
        "clusters": clusters,
        "without_cluster": sin_cluster,
        "total": len(fps),
        "with_cluster_count": len(clusters),
        "without_cluster_count": len(sin_cluster),
    }


class AcousticPendingRequest(BaseModel):
    fingerprints: List[str]


@router.post("/acoustic/pending")
async def acoustic_pending(request: AcousticPendingRequest):
    """De este lote de huellas, cuales se pueden CURAR subiendo el audio.

    Es el paso previo del backfill que si funciona en el Mac App Store y en
    movil. El backfill de siempre (`/backfill-fingerprint`) necesita `fpcalc`
    local: en MAS el sandbox no le deja abrir ficheros y en movil no hay
    binario, asi que esas dos plataformas no pueden rehacer la huella de su
    legado. Pero el audio si lo tienen — y `/backfill-audio` calcula la huella
    en el servidor, sin reanalizar y sin AudD.

    La respuesta parte en TRES, no en dos, porque piden acciones opuestas:

      `curable`         analizada y SIN chromaprint -> subir el audio.
      `with_chromaprint` ya esta en la memoria colectiva -> no subir nada.
      `not_analyzed`    no esta en la tabla. Subirla NO es un backfill: es un
                        analisis completo con su CPU y su posible AudD. Que el
                        cliente no la confunda con `curable` es justo el punto:
                        un solo cubo «no tiene huella» haria que el backfill se
                        comiera la biblioteca entera de un movil recien
                        instalado creyendo que rellena huecos.

    Maximo 500 por peticion, igual que sus vecinas.
    """
    fps = [f for f in (request.fingerprints or []) if f]
    if len(fps) > 500:
        raise HTTPException(400, "Máximo 500 fingerprints por petición")

    estado = db.chromaprint_status_for(fps)
    curable = [f for f in fps if estado.get(f) is False]
    con_huella = [f for f in fps if estado.get(f) is True]
    sin_analizar = [f for f in fps if f not in estado]
    return {
        "curable": curable,
        "with_chromaprint": con_huella,
        "not_analyzed": sin_analizar,
        "total": len(fps),
        "curable_count": len(curable),
        "with_chromaprint_count": len(con_huella),
        "not_analyzed_count": len(sin_analizar),
    }


class CheckAnalyzedByFingerprintRequest(BaseModel):
    fingerprints: List[str]
    # Los dos los manda SOLO el motor local cuando pregunta a Render por lo que
    # el no tiene (`main._precheck_en_render`). Para el, «analizado» es que
    # Render le pueda dar un analisis que valga AQUI: de SU version de
    # analisis (no la de Render) y con BPM y tonalidad. Una fila del fallback
    # (`analysis_status='failed'`, bpm 0) no le sirve: el motor local tiene
    # librosa y lo hace mejor. Es la misma vara que `_fetch_render_cache`.
    version: Optional[str] = None
    con_datos: bool = False


def _vale(fila: dict, version: Optional[str], con_datos: bool) -> bool:
    if version is not None:
        if (fila.get('analysis_version') or '1') != version:
            return False
    elif not _is_analysis_current(fila):
        return False
    if con_datos:
        try:
            bpm = float(fila.get('bpm') or 0)
        except (TypeError, ValueError):
            bpm = 0
        if bpm <= 0 or not (fila.get('key') or '').strip():
            return False
    return True


@router.post("/check-analyzed-by-fingerprint")
async def check_analyzed_by_fingerprint(request: CheckAnalyzedByFingerprintRequest,
                                        peticion: Request):
    """
    Dedup multi-dispositivo: dado un lote de fingerprints (MD5 del contenido
    del archivo) devuelve cuáles ya están analizados en Render. Esto
    permite que el cliente (especialmente móvil) evite subir y re-analizar
    tracks que ya fueron procesados desde otro dispositivo aunque el nombre
    del fichero sea distinto.

    Máximo 500 IDs por petición.

    El lote se resuelve en DOS consultas (`filas_por_huella`) y fuera del
    event loop. Hasta el 2026-10-06 era una consulta por huella —500 seguidas
    en el único worker por cada lote del móvil— y, en el motor local, un GET
    a Render POR HUELLA que no tenía, bloqueando y con 5 s de timeout cada
    uno: con Render dormido, una ventana de 25 temas eran dos minutos sin
    atender nada. Ahora lo que falta va a Render en UNA petición.

    Y cuenta en la popularidad: con el `X-Device-Token` de un aparato
    registrado, ese aparato queda como un DJ más que tiene los temas que se
    contestan como analizados (`registrar_analistas`). Sin token no se cuenta:
    la huella basta para preguntar, y sin aparato de verdad cualquiera
    inflaría los DJs. El motor local reenvía el token con lo que pregunta a
    Render.
    """
    fps = request.fingerprints or []
    if len(fps) > 500:
        raise HTTPException(400, "Máximo 500 fingerprints por petición")
    validas = [fp for fp in fps if fp]

    def _de_esta_bd():
        filas = db.filas_por_huella(
            validas, 'id, fingerprint, analysis_version, bpm, key')
        return {fp for fp, fila in filas.items()
                if _vale(fila, request.version, request.con_datos)}

    ya = await run_in_threadpool(_de_esta_bd)

    # Motor local: su BD es local a ESTA maquina, asi que tras un formateo
    # (o en un Mac nuevo) esta vacia y el pre-check del cliente fallaba
    # SIEMPRE -> subida + analisis completo de toda la biblioteca aunque
    # Render ya tuviera cada track. Preguntamos a Render antes de decir
    # "no analizado": es un SELECT, y el cliente se ahorra subir el fichero.
    token = (peticion.headers.get('X-Device-Token') or '').strip()
    faltan = [fp for fp in dict.fromkeys(validas) if fp not in ya]
    if faltan and precheck_en_render is not None:
        try:
            ya |= set(await run_in_threadpool(precheck_en_render, faltan,
                                              token or None))
        except Exception as e:  # nunca romper el pre-check por esto
            logger.info("[Dedup] Render no contesto al pre-check (%d huellas): %s",
                        len(faltan), e)

    analyzed = [fp for fp in validas if fp in ya]
    not_analyzed = [fp for fp in validas if fp not in ya]

    if analyzed and token and registrar_analistas is not None:
        try:
            aparato = await run_in_threadpool(dispositivo_del_token, token)
            if aparato:
                await run_in_threadpool(registrar_analistas, analyzed, aparato)
        except Exception as e:  # contar nunca tumba el pre-check
            logger.warning("[Popularidad] pre-check: %s", e)

    return {
        "analyzed": analyzed,
        "not_analyzed": not_analyzed,
        "total": len(fps),
        "analyzed_count": len(analyzed),
        "not_analyzed_count": len(not_analyzed),
    }


def _ficha_desde_fila(safe_fp, existing):
    """La ficha que se le da al cliente a partir de la fila de `tracks` de esa
    huella: su `analysis_json` (o las columnas, si no lo hay) con lo mejor que
    sabe la memoria colectiva encima. La comparten el GET de una huella y el
    lote (`/analysis/by-fingerprint/batch`)."""
    raw = existing.get('analysis_json')
    result = None
    if raw:
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            result = None
    if result is None:
        # Fallback: construir desde columnas. Incluir *_source para que el
        # cliente sepa si vale la pena sobreescribir su analisis local (ranking
        # "mejor gana" en sync comunitario, item 8).
        result = {
            "id": existing.get('id'),
            "filename": existing.get('filename'),
            "artist": existing.get('artist'),
            "title": existing.get('title'),
            "duration": existing.get('duration') or 0,
            "bpm": existing.get('bpm') or 0,
            "key": existing.get('key'),
            "camelot": existing.get('camelot'),
            "energy_dj": existing.get('energy_dj') or 5,
            "genre": existing.get('genre'),
            "track_type": existing.get('track_type'),
            "fingerprint": existing.get('fingerprint'),
            "bpm_source": existing.get('bpm_source'),
            "key_source": existing.get('key_source'),
            "genre_source": existing.get('genre_source'),
            "engine_source": existing.get('engine_source'),
            "analysis_version": existing.get('analysis_version') or '1',
        }
    # Corregir con la MEJOR metadata del cluster acustico (RETROACTIVO): si otra
    # version del mismo audio (otro usuario) aporto una fuente superior despues
    # de que este track se analizara, el cliente la recibe al re-consultar.
    _merge_cluster_best_into(result, safe_fp, existing)
    return result


@router.get("/analysis/by-fingerprint/{fingerprint}")
async def get_analysis_by_fingerprint(fingerprint: str):
    """Devuelve el análisis cacheado de un track por su fingerprint
    (MD5 del contenido). El cliente puede usar este endpoint tras
    `/check-analyzed-by-fingerprint` para hidratar su cache local sin
    subir el archivo otra vez."""
    safe_fp = re.sub(r'[^a-fA-F0-9]', '', fingerprint or '')
    if not safe_fp:
        raise HTTPException(400, "fingerprint inválido")
    existing = db.get_track_by_fingerprint(safe_fp)
    if not existing:
        # Motor local sin el track en su BD: si Render lo tiene, se lo damos al
        # cliente tal cual. Sin esto el pre-check de dedup decia "ya analizado"
        # (gracias al mismo fallback en /check-analyzed-by-fingerprint) y aqui
        # respondia 404, asi que el cliente acababa subiendo el fichero igual.
        if fetch_render_cache is not None:
            try:
                remote = await run_in_threadpool(fetch_render_cache, safe_fp)
            except Exception:
                remote = None
            if remote:
                return remote
        raise HTTPException(404, "fingerprint no encontrado")
    return _ficha_desde_fila(safe_fp, existing)


class FichasPorHuellaRequest(BaseModel):
    fingerprints: List[str]


# Una ficha son unos KB (el `analysis_json` entero): 100 por petición.
MAX_FICHAS_POR_PETICION = 100


@router.post("/analysis/by-fingerprint/batch")
async def fichas_por_huella(req: FichasPorHuellaRequest):
    """Las fichas de varias huellas en UNA petición: `{"fichas": {huella:
    ficha}}`, la misma ficha que da el GET de una (`_ficha_desde_fila`). Las
    que no hay no salen.

    Tras el pre-check, el import pedía la ficha de cada acierto con un GET:
    en un Mac recién formateado con 5.000 temas que Render ya tiene, 5.000
    viajes seguidos, y con motor local, cada uno pasando por él (2026-10-06).
    El motor local completa con Render lo que no tiene, en otra sola petición
    y con la misma vara que el GET (`_fetch_render_cache`).
    """
    fps = [re.sub(r'[^a-fA-F0-9]', '', f or '') for f in (req.fingerprints or [])]
    fps = [f for f in dict.fromkeys(fps) if f]
    if len(fps) > MAX_FICHAS_POR_PETICION:
        raise HTTPException(
            400, f"Máximo {MAX_FICHAS_POR_PETICION} fingerprints por petición")

    def _de_esta_bd():
        filas = db.fichas_por_huella(fps)
        return {fp: _ficha_desde_fila(fp, fila) for fp, fila in filas.items()}

    fichas = await run_in_threadpool(_de_esta_bd)
    faltan = [fp for fp in fps if fp not in fichas]
    if faltan and fichas_en_render is not None:
        try:
            fichas.update(await run_in_threadpool(fichas_en_render, faltan))
        except Exception as e:  # nunca romper el lote por esto
            logger.info("[Fichas] Render no contesto (%d huellas): %s",
                        len(faltan), e)
    return {"fichas": fichas}


# ==================== ENDPOINTS DE ARTWORK ====================

@router.get("/analysis/{filename:path}")
async def get_analysis(filename: str):
    """Ya no se da un análisis por NOMBRE de fichero: 410.

    La tabla es la de todos los usuarios, y por nombre salía el de cualquiera
    que se llamara igual (ver `/check-analyzed`). Lo de este audio se pide por
    su huella: `/analysis/by-fingerprint/{huella}`. Un móvil de antes, con el
    410, analiza el fichero (`AudioAnalysisApi.getAnalysis` devolvía null en
    cualquier respuesta que no fuera 200).
    """
    raise HTTPException(
        410,
        "Por nombre de fichero no: pídelo por su huella "
        "(/analysis/by-fingerprint/{huella})",
    )

# Ids que se aceptan en /artwork: huellas (hex) y los `imp_…`/detecciones que
# usa el propio backend. Nada con puntos ni barras llega a `os.path.join`.
_ID_VALIDO = re.compile(r'^[A-Za-z0-9_\-]{1,80}$')

# «No hay portada», por huella, con la hora en que se supo. Solo lo SEGURO:
# todas las fuentes contestaron y ninguna la tenia (`buscar_portada`). Sin
# esto, cada peticion de una huella sin portada salia otra vez a internet: hasta
# ocho peticiones desde la IP de Render, que comparten todos los usuarios. En
# memoria a proposito: un deploy lo vacia y se vuelve a mirar, que es barato.
_SIN_PORTADA = {}
_SIN_PORTADA_TTL = 7 * 24 * 3600


def _sin_portada_reciente(clave: str) -> bool:
    import time
    t = _SIN_PORTADA.get(clave)
    if t is None:
        return False
    if time.time() - t > _SIN_PORTADA_TTL:
        _SIN_PORTADA.pop(clave, None)
        return False
    return True


def _cacheada(clave: str):
    for ext in ('jpg', 'png', 'jpeg', 'webp', 'gif'):
        ruta = os.path.join(ARTWORK_CACHE_DIR, f"{clave}.{ext}")
        if os.path.exists(ruta):
            return ruta, ext
    return None, None


@router.head("/artwork/{track_id}")
async def head_artwork(track_id: str):
    """HEAD para /artwork/{track_id} - el cliente desktop pre-comprueba
    existencia antes de subir su propio artwork (evita re-upload). Solo
    mira el cache local del disco; NO dispara el fallback online del GET
    (search_artwork_online tiene side effects: red + escritura a cache).
    Devuelve 200 con Content-Type/Content-Length, o 404 sin body.
    """
    if not _ID_VALIDO.match(track_id or ''):
        raise HTTPException(404, "Artwork no encontrado")
    for ext in ('jpg', 'png', 'jpeg', 'webp', 'gif'):
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            return Response(
                status_code=200,
                headers={
                    "Content-Type": _artwork_media_type(ext),
                    "Content-Length": str(os.path.getsize(cache_path)),
                },
            )
    raise HTTPException(404, "Artwork no encontrado")


def _artwork_media_type(ext: str) -> str:
    """Mimetype para servir un artwork cacheado según su extensión."""
    return {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'webp': 'image/webp',
        'gif': 'image/gif',
    }.get(ext, 'image/jpeg')


@router.get("/artwork/{track_id}")
async def get_artwork(track_id: str, request: Request = None, online: int = 1):
    """Devuelve el artwork de un track como imagen.

    Cascade:
      1. Cache local (`{ARTWORK_CACHE_DIR}/{track_id}.{ext}`).
      2. Si la BD tiene el track pero falta el archivo (típicamente
         tracks analizados con motor local cuyo PUSH a Render falló o
         no se hizo), buscamos artwork online (iTunes/Deezer) usando
         artist+title de la BD y lo cacheamos para futuras peticiones.
      3. 404 si nada de lo anterior funciona.

    `online=0`: no salir a internet. Lo pide el escritorio, que busca por su
    cuenta desde la IP del usuario y sube lo que encuentra.
    """
    if not _ID_VALIDO.match(track_id or ''):
        raise HTTPException(404, "Artwork no encontrado")
    for ext in ['jpg', 'png', 'jpeg', 'webp', 'gif']:
        cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{track_id}.{ext}")
        if os.path.exists(cache_path):
            # Leer los bytes AQUÍ (no FileResponse): FileResponse abre el
            # fichero en la fase de envío ASGI, así que si el .jpg desaparece
            # entre el os.path.exists y el envío (race con un wipe / borrado
            # concurrente) lanza FileNotFoundError SIN capturar → 500 feo. Con
            # la lectura en el handler, un fichero que se esfumó cae a la
            # cascada (online / 404) en vez de reventar.
            try:
                with open(cache_path, 'rb') as f:
                    data = f.read()
                return Response(content=data, media_type=_artwork_media_type(ext))
            except OSError:
                continue

    # Fallback: buscar online por artist+title si tenemos el track en BD.
    if not online or _sin_portada_reciente(track_id):
        raise HTTPException(404, "Artwork no encontrado")
    try:
        existing = db.get_track_by_fingerprint(track_id) or db.get_track_by_id(track_id)
        if existing:
            artist = existing.get('artist')
            title = existing.get('title')
            if artist and title and search_artwork_online:
                # SEC-13: el camino caro lleva cupo APARTE por IP. Servir una
                # caratula cacheada (bloque de arriba) NO se capa —la app pide
                # cientos al pintar la biblioteca y capar eso romperia el uso
                # normal— pero salir a internet si: son hasta 8 peticiones a
                # iTunes/Deezer/Last.fm desde la IP de Render, y un bucle sobre
                # fingerprints sin caratula puede hacer que esas APIs nos
                # limiten o baneen, lo que degradaria el servicio para TODOS.
                #
                # Al agotarse el cupo NO se lanza 429: se salta la busqueda y se
                # cae al 404 del final, que es la respuesta normal de "no hay
                # caratula" y que el cliente ya pinta como placeholder. Meter un
                # estado de error nuevo romperia UI que hoy funciona.
                #
                # OJO: aqui NO se puede `raise` — este bloque vive dentro de un
                # `try/except Exception` que se lo tragaria y loguearia
                # "Fallback online error", que es un mensaje falso.
                if request is not None and not artwork_online_allowed(
                        get_client_ip(request)):
                    logger.warning(
                        f"[Artwork] Cupo de busqueda online agotado para esta IP; "
                        f"no se sale a internet ({track_id[:8]})"
                    )
                else:
                    logger.info(f"[Artwork] Cache MISS para {track_id[:8]}, buscando online...")
                    # search_artwork_online encadena hasta 8 peticiones HTTP
                    # SINCRONAS (iTunes busqueda + descarga, Deezer x2, Last.fm x2)
                    # con timeouts de 5-8 s. Llamarla directa desde este handler
                    # async congelaba el event loop del unico worker hasta ~45 s.
                    if buscar_portada is not None:
                        encontrada, definitivo = await run_in_threadpool(
                            buscar_portada, artist, title)
                    else:
                        encontrada = await run_in_threadpool(
                            search_artwork_online, artist, title)
                        definitivo = False
                    if encontrada and encontrada.get('data'):
                        save_artwork_to_cache(
                            track_id, encontrada['data'], encontrada['mime_type'])
                        # Devolver los bytes que ya tenemos en memoria (no re-leer
                        # el fichero recién guardado → no puede fallar por race).
                        return Response(
                            content=encontrada['data'],
                            media_type=encontrada['mime_type'])
                    if definitivo:
                        import time
                        _SIN_PORTADA[track_id] = time.time()
    except Exception as e:
        logger.warning(f"[Artwork] Fallback online error: {e}")

    raise HTTPException(404, "Artwork no encontrado")


@router.post("/artwork/upload/{fingerprint}")
async def upload_artwork(fingerprint: str, request: Request,
                         file: UploadFile = File(...),
                         solo_si_falta: int = 0):
    """Recibe artwork desde el local engine para que Render lo sirva
    también a otros devices vía `/artwork/{fingerprint}`. Sin esto,
    cuando el local engine analiza un track el artwork se queda en
    disco PC y los móviles ven placeholder.

    Sanitiza el fingerprint (solo hex 32 chars). Acepta JPEG/PNG/WEBP/GIF.

    `solo_si_falta=1`: si ya hay portada para esa huella, no se toca y se
    contesta `exists`. Lo manda el escritorio con todo lo que NO sale del
    propio fichero (una busqueda o una identificacion pueden equivocarse, y la
    portada de una huella la ven todos los que tienen ese fichero). La que
    viene DENTRO del fichero si pisa: es parte de su contenido.

    Pisar exige ser un aparato registrado (`X-Device-Token` del sync). Sin
    credencial la subida vale igual, pero como `solo_si_falta`: la portada de
    una huella la ven todos los que tienen ese fichero, y hasta el 2026-09-26
    un curl sin nada podía cambiársela a todos. No lo hace imposible (el
    token lo saca cualquiera que se registre con el secreto del binario),
    pero lo sube de «un curl» a «registrarse», y cada pisada queda en el log
    con el aparato que la hizo.
    """
    safe_fp = re.sub(r'[^a-fA-F0-9]', '', fingerprint or '')
    if not safe_fp or len(safe_fp) > 64:
        raise HTTPException(400, "fingerprint inválido")
    quien = None
    if not solo_si_falta:
        quien = await run_in_threadpool(
            dispositivo_del_token, request.headers.get("X-Device-Token", ""))
        if not quien:
            solo_si_falta = 1
    habia = _cacheada(safe_fp)[0]
    if solo_si_falta and habia:
        return {"status": "exists", "fingerprint": safe_fp}

    content = await file.read()
    if not content or len(content) < 100:
        raise HTTPException(400, "archivo vacío o demasiado pequeño")
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(400, "artwork demasiado grande (max 5MB)")

    # Detectar tipo por bytes mágicos. Defaults a jpg si no clarifica.
    # WEBP/GIF se aceptan porque algunas carátulas embebidas vienen en esos
    # formatos; el motor local las escribe como `.jpg` aunque el contenido
    # sea webp/gif, así que sin esto el upload daba 400 en bucle.
    if content[:3] == b'\xff\xd8\xff':
        ext = 'jpg'
    elif content[:8] == b'\x89PNG\r\n\x1a\n':
        ext = 'png'
    elif content[:4] == b'RIFF' and content[8:12] == b'WEBP':
        ext = 'webp'
    elif content[:6] in (b'GIF87a', b'GIF89a'):
        ext = 'gif'
    else:
        # No reconocido — rechazar para no llenar el disco con basura.
        raise HTTPException(400, "formato no soportado (sólo JPEG/PNG/WEBP/GIF)")

    # Eliminar versiones previas con otra extensión para evitar dos
    # archivos del mismo fingerprint en el cache.
    for prev_ext in ('jpg', 'jpeg', 'png', 'webp', 'gif'):
        prev = os.path.join(ARTWORK_CACHE_DIR, f"{safe_fp}.{prev_ext}")
        if os.path.exists(prev):
            try:
                os.unlink(prev)
            except OSError:
                pass

    cache_path = os.path.join(ARTWORK_CACHE_DIR, f"{safe_fp}.{ext}")
    with open(cache_path, 'wb') as f:
        f.write(content)
    _SIN_PORTADA.pop(safe_fp, None)
    if habia:
        logger.info(f"[Artwork] {safe_fp} pisada por {quien}")

    return {"status": "ok", "fingerprint": safe_fp, "size": len(content), "ext": ext}
