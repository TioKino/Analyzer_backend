# ============================================================================
# COMMUNITY CUES ENDPOINT - Zonas de cue validadas por la comunidad
# ============================================================================
# Aniadir a main.py despues de los endpoints existentes.
# La tabla `community_cues` se crea en `database.py:init_db()`.
#
# Flujo:
#   1. DJ analiza track -> coloca cues manuales en CueFlow
#   2. Al exportar o sincronizar, envia sus cues al backend (POST /community-cues)
#   3. Backend agrega cues de todos los DJs por fingerprint
#   4. Cuando otro DJ analiza el mismo track, recibe zonas comunitarias (GET)
#
# ---------------------------------------------------------------------------
# LA CLAVE: por que existe `legacy_key` / `?also=`
# ---------------------------------------------------------------------------
# Durante anios el cliente subio y leyo estos cues con su `track.id`, que es
# MD5("nombre|tamanio") y NO es la huella. Aqui `tracks.id = tracks.fingerprint`
# (MD5 del CONTENIDO), asi que ese `track.id` no existe en ninguna tabla:
# `fingerprints_in_cluster()` no encontraba fila, no podia resolver el cluster
# acustico y devolvia la clave pelada. Resultado: cada usuario escribia en su
# propio compartimento y los cues comunitarios no cruzaron entre DJs ni una vez
# (para cruzar harian falta dos ficheros de identico nombre Y tamanio; en
# produccion, `collision_groups: 0` entre 317 dispositivos).
#
# El cliente ya manda la huella. Estos dos parametros son el puente para que el
# cambio no se note, y son OPCIONALES: un cliente viejo se comporta igual que
# antes.
#
#   GET  ?also=<clave vieja>   suma esa clave al conjunto que se consulta, asi
#                              lo que subiste antes se sigue viendo aunque no
#                              hayas vuelto a tocar el track.
#   POST legacy_key            despues de guardar bajo la huella, borra las
#                              filas de ESE MISMO device bajo la clave vieja.
#                              Sin esto, mover un cue deja la posicion antigua
#                              colgada y la zona sale por duplicado.
#
# Los dos estan acotados al `device_id` que llama o a una lectura de solo
# lectura: no hay forma de tocar datos de otro usuario con ellos.
# ============================================================================

import re
import sqlite3
from pydantic import BaseModel
from typing import List, Optional
from fastapi import APIRouter, Request
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/community-cues", tags=["community"])


# ==================== MODELS ====================

class CueSubmission(BaseModel):
    """Un cue enviado por un DJ"""
    type: str  # mixIn, mixOut, drop, breakdown, vocal, buildup, loop, custom
    position_ms: int
    end_position_ms: Optional[int] = None  # para loops/regiones
    note: Optional[str] = None


class CueUpload(BaseModel):
    """Envio completo de cues de un DJ para un track"""
    fingerprint: str
    device_id: str  # identificador anonimo del dispositivo
    cues: List[CueSubmission]
    # Clave con la que ESTE cliente subia antes este track (su `track.id`,
    # MD5(nombre|tamanio)). Ver el bloque LA CLAVE, abajo. Solo se usa para
    # limpiar las filas viejas DE ESTE MISMO device; nunca para escribir.
    legacy_key: Optional[str] = None


class CueBatchItem(BaseModel):
    """Los cues de UN tema dentro de un envío por lotes."""
    fingerprint: str
    cues: List[CueSubmission]
    legacy_key: Optional[str] = None


class CueBatchUpload(BaseModel):
    """Los cues de muchos temas de un aparato, de una vez. El aparato NO va en
    el cuerpo: sale del `X-Device-Token` (ver `/community/cues/batch`)."""
    items: List[CueBatchItem]


class PuntoDeZona(BaseModel):
    """Un sitio de la zona donde al menos dos DJs pusieron un cue (ver
    `puntos_de_la_zona`). Es lo que el DJ puede ADOPTAR como cue suyo."""
    pos: float  # segundos
    djs: int
    fin: Optional[float] = None  # segundos; solo en loops, si coinciden


class CommunityZoneResponse(BaseModel):
    type: str
    start: float  # segundos
    end: float  # segundos
    dj_count: int
    confidence: float
    top_note: Optional[str] = None
    puntos: List[PuntoDeZona] = []


class CommunityResponse(BaseModel):
    fingerprint: str
    zones: List[CommunityZoneResponse]
    total_contributors: int
    last_updated: Optional[str] = None


# ==================== INPUT HARDENING ====================

# Limites defensivos para el upload de cues (input NO confiable). Un track real
# rara vez pasa de ~50 cues; 200 es holgado. Evita que un cliente con el secreto
# meta miles de filas, posiciones negativas o notas gigantes en la tabla
# comunitaria. La SQL ya es parametrizada (sin inyeccion); esto acota volumen.
MAX_CUES_PER_UPLOAD = 200
MAX_CUE_NOTE_LEN = 500
# Temas por petición en `/community/cues/batch`. Una biblioteca importada de
# 3.000 temas son 15 peticiones, no 3.000.
MAX_TEMAS_POR_LOTE = 200

_HUELLA = re.compile(r'[0-9a-f]{32}')


def sanitize_cue_submissions(cues) -> list:
    """Filtra/acota cues no confiables antes de insertarlos.

    - Descarta cues con position_ms ausente o < 0 (no existe tiempo negativo).
    - Anula end_position_ms si es < position_ms (region invalida -> cue puntual).
    - Trunca la nota a MAX_CUE_NOTE_LEN.
    - Corta la lista a MAX_CUES_PER_UPLOAD.

    Devuelve lista de tuplas (type, position_ms, end_position_ms, note) lista
    para el INSERT. Funcion pura (sin DB) para poder testearla aislada.
    """
    cleaned = []
    for cue in cues:
        pos = getattr(cue, 'position_ms', None)
        if pos is None or pos < 0:
            continue
        end = getattr(cue, 'end_position_ms', None)
        if end is not None and end < pos:
            end = None
        note = getattr(cue, 'note', None)
        if isinstance(note, str) and len(note) > MAX_CUE_NOTE_LEN:
            note = note[:MAX_CUE_NOTE_LEN]
        cleaned.append((cue.type, pos, end, note))
        if len(cleaned) >= MAX_CUES_PER_UPLOAD:
            break
    return cleaned


# ==================== AGGREGATION LOGIC ====================

# SELECT explicito que se reusa en endpoints. Fija el orden de columnas
# y elimina el riesgo de drift que dejaba B-M3 (queries SELECT * con
# accesos `r[2]`, `r[3]` etc. hardcodeados). Cierra B-M3 del AUDIT 2026-04-20.
_CUES_COLUMNS = (
    'id, fingerprint, device_id, cue_type, position_ms, '
    'end_position_ms, note, created_at'
)


def _dj(device_id, cuentas: Optional[dict]) -> str:
    """Quién es el DJ detrás de un aparato: su cuenta si la tiene."""
    if cuentas and device_id in cuentas:
        return 'u:' + cuentas[device_id]
    return device_id


# ---------------------------------------------------------------------------
# LOS PUNTOS DE UNA ZONA: lo que se puede ADOPTAR (2026-11-11)
# ---------------------------------------------------------------------------
# Una zona junta los cues de un tipo encadenados a 5 s unos de otros, con 2 s
# de margen a cada lado: dice «aquí pasa algo», no DÓNDE. Su centro puede no
# ser de nadie (un DJ en el compás 33 y otro en el 35 dan una zona cuyo medio
# no marcó ninguno). Para que un DJ convierta lo de la comunidad en cues suyos
# con un clic (`COMPETENCIA.md`, punto 4) hace falta otra cosa: los sitios
# donde al menos DOS DJs pusieron un cue EN EL MISMO SITIO.
#
# «El mismo sitio» son [TOLERANCIA_DE_UN_PUNTO_MS]: la zona junta todas las
# versiones del sonido (el cluster), y la misma muestra no cae en el mismo
# milisegundo en un MP3 y en un FLAC, ni en Rekordbox y en Traktor (medido: el
# retardo del codificador y la trama de cabecera mueven hasta ~51 ms). Y dos
# beats seguidos están a más de 300 ms incluso a 200 BPM, así que con 120 ms no
# se juntan dos golpes distintos.
#
# Cada cuenta cuenta UNA vez (su posición mediana en ese sitio), y el punto es
# la mediana de las cuentas: tres cues del mismo DJ en un sitio no lo mueven.
# Lo usa el cliente (`cues_de_la_comunidad.dart`); el cue que se adopta lleva
# una marca y no se vuelve a subir como un voto más mientras siga en su sitio.
TOLERANCIA_DE_UN_PUNTO_MS = 120
MIN_DJS_PUNTO = 2


def _mediana(valores) -> float:
    v = sorted(valores)
    n = len(v)
    if n == 0:
        return 0.0
    return float(v[n // 2]) if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def puntos_de_la_zona(cues, cuentas: Optional[dict] = None) -> List[dict]:
    """Los sitios de una zona donde al menos [MIN_DJS_PUNTO] cuentas pusieron
    un cue a menos de [TOLERANCIA_DE_UN_PUNTO_MS] unos de otros.

    [cues]: los de UNA zona (mismo tipo), dicts con `device_id`,
    `position_ms` y `end_position_ms`. Devuelve `{pos, djs[, fin]}` en
    segundos, por orden. Avaro: se elige primero el sitio con más cuentas
    (y, a igualdad, el más apretado), se quitan sus cues y se repite.
    `fin` solo sale en un loop cuyas cuentas coinciden también en el final.
    """
    quedan = sorted(cues, key=lambda c: c['position_ms'])
    puntos = []
    while quedan:
        mejor = None
        for i, c in enumerate(quedan):
            ventana = []
            for d in quedan[i:]:
                if d['position_ms'] - c['position_ms'] > TOLERANCIA_DE_UN_PUNTO_MS:
                    break
                ventana.append(d)
            djs = {_dj(d['device_id'], cuentas) for d in ventana}
            if len(djs) < MIN_DJS_PUNTO:
                continue
            apretado = ventana[-1]['position_ms'] - ventana[0]['position_ms']
            clave = (len(djs), -apretado)
            if mejor is None or clave > mejor[0]:
                mejor = (clave, ventana)
        if mejor is None:
            break
        ventana = mejor[1]
        por_dj = {}
        for d in ventana:
            por_dj.setdefault(_dj(d['device_id'], cuentas), []).append(d)
        pos = _mediana([_mediana([d['position_ms'] for d in ds])
                        for ds in por_dj.values()])
        punto = {'pos': round(pos / 1000.0, 3), 'djs': len(por_dj)}
        # El final de un loop: el de cada cuenta que lo tiene, y solo si todas
        # dicen el mismo (4 y 8 beats darían de mediana 6, que no es de nadie).
        finales = []
        for ds in por_dj.values():
            fs = [d['end_position_ms'] for d in ds
                  if d.get('end_position_ms') and d['end_position_ms'] > d['position_ms']]
            if fs:
                finales.append(_mediana(fs))
        if (len(finales) >= MIN_DJS_PUNTO
                and max(finales) - min(finales) <= TOLERANCIA_DE_UN_PUNTO_MS):
            punto['fin'] = round(_mediana(finales) / 1000.0, 3)
        puntos.append(punto)
        quitar = {id(d) for d in ventana}
        quedan = [d for d in quedan if id(d) not in quitar]
    puntos.sort(key=lambda p: p['pos'])
    return puntos


def contribuyentes(rows, cuentas: Optional[dict] = None) -> int:
    """Cuántos DJs distintos (cuentas) han subido cues."""
    return len({_dj(r['device_id'], cuentas) for r in rows})


def aggregate_cues_into_zones(rows, duration_seconds: float = 0,
                              cuentas: Optional[dict] = None) -> List[dict]:
    """
    Agrega cues individuales de multiples DJs en zonas comunitarias.

    Algoritmo:
    1. Agrupa cues por tipo
    2. Para cada tipo, clustering por proximidad (ventana de 5 segundos)
    3. Para cada cluster: calcular centro, rango, conteo de DJs unicos, confianza
    4. Filtrar: minimo 2 DJs deben haber marcado la zona

    `rows` debe ser una lista de `sqlite3.Row` (acceso por nombre) o de
    dicts con claves: cue_type, device_id, position_ms, end_position_ms, note.
    """
    from collections import defaultdict

    # Agrupar por tipo. Usamos acceso por nombre en lugar de indices
    # hardcodeados para que el orden de SELECT no afecte. Cierra B-M3.
    by_type = defaultdict(list)
    for row in rows:
        by_type[row['cue_type']].append({
            'device_id': row['device_id'],
            'position_ms': row['position_ms'],
            'end_position_ms': row['end_position_ms'],
            'note': row['note'],
        })

    zones = []
    cluster_window_ms = 5000  # 5 segundos de tolerancia

    for cue_type, cues in by_type.items():
        # Ordenar por posicion
        cues.sort(key=lambda c: c['position_ms'])

        # Clustering simple por proximidad
        clusters = []
        current_cluster = [cues[0]]

        for i in range(1, len(cues)):
            if cues[i]['position_ms'] - current_cluster[-1]['position_ms'] <= cluster_window_ms:
                current_cluster.append(cues[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [cues[i]]
        clusters.append(current_cluster)

        # Convertir clusters en zonas
        for cluster in clusters:
            # DJs = CUENTAS, no aparatos: los cues viajan por sync entre el
            # escritorio y el móvil de la misma persona, y si los dos los
            # suben, contando aparatos una sola persona fabricaba una zona
            # «de 2 DJs» consigo misma.
            unique_djs = set(_dj(c['device_id'], cuentas) for c in cluster)
            dj_count = len(unique_djs)

            # Minimo 2 DJs para crear una zona comunitaria
            if dj_count < 2:
                continue

            positions = [c['position_ms'] for c in cluster]
            min_pos = min(positions)
            max_pos = max(positions)

            # Zona = rango de cues + margen de 2 segundos
            start_sec = max(0, (min_pos - 2000)) / 1000.0
            end_sec = (max_pos + 2000) / 1000.0
            if duration_seconds > 0:
                end_sec = min(end_sec, duration_seconds)

            # Confianza basada en acuerdo entre DJs
            spread_ms = max_pos - min_pos
            if spread_ms < 1000:
                confidence = 0.95
            elif spread_ms < 3000:
                confidence = 0.80
            elif spread_ms < 5000:
                confidence = 0.65
            else:
                confidence = 0.50

            # Boost confianza por cantidad de DJs
            confidence = min(1.0, confidence + (dj_count - 2) * 0.03)

            # Nota mas comun
            notes = [c['note'] for c in cluster if c.get('note')]
            top_note = max(set(notes), key=notes.count) if notes else None

            zones.append({
                'type': cue_type,
                'start': round(start_sec, 2),
                'end': round(end_sec, 2),
                'dj_count': dj_count,
                'confidence': round(confidence, 2),
                'top_note': top_note,
                # Dónde coinciden de verdad: lo que el DJ puede adoptar.
                'puntos': puntos_de_la_zona(cluster, cuentas),
            })

    # Ordenar por posicion
    zones.sort(key=lambda z: z['start'])
    return zones


# ==================== ENDPOINTS ====================

def _guardar_cues(c, fingerprint: str, device_id: str, cues, legacy_key,
                  now: str) -> list:
    """Sustituye los cues de ESTE aparato para este tema por [cues].

    Lo comparten el envío de un tema (`/community/cues`) y el de muchos
    (`/community/cues/batch`). Devuelve los cues guardados, ya saneados. No
    hace commit: eso le toca a quien llama, una vez por petición.
    """
    # Borrar cues anteriores de este device para este track
    c.execute(
        'DELETE FROM community_cues WHERE fingerprint = ? AND device_id = ?',
        (fingerprint, device_id)
    )

    # Insertar nuevos cues (acotados/saneados — input no confiable)
    sanitized = sanitize_cue_submissions(cues)
    for cue_type, position_ms, end_position_ms, note in sanitized:
        # OR REPLACE: si el mismo upload trae dos cues con igual
        # (cue_type, position_ms) — o el DELETE previo no cubrió una
        # colisión — no reventamos por el UNIQUE; nos quedamos con el
        # último. (Bug visto en prod: "UNIQUE constraint failed:
        # community_cues.fingerprint, device_id, cue_type, position_ms".)
        c.execute('''
            INSERT OR REPLACE INTO community_cues
            (fingerprint, device_id, cue_type, position_ms, end_position_ms, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            fingerprint,
            device_id,
            cue_type,
            position_ms,
            end_position_ms,
            note,
            now,
        ))

    # Migracion silenciosa: las filas que este MISMO device dejo bajo
    # su clave vieja ya estan representadas por las que acabamos de
    # insertar, asi que sobran. Y no solo sobran: si el DJ movio un
    # cue, la posicion antigua sigue ahi y la zona sale DOS VECES, en
    # dos sitios distintos de la onda.
    #
    # Acotado a (clave vieja, este device_id): no puede tocar datos de
    # nadie mas ni la clave nueva. Y solo si de verdad es OTRA clave —
    # un cliente sin huella manda las dos iguales y ahi borrar seria
    # borrarse lo que se acaba de escribir.
    if legacy_key and legacy_key != fingerprint:
        c.execute(
            'DELETE FROM community_cues '
            'WHERE fingerprint = ? AND device_id = ?',
            (legacy_key, device_id),
        )
        if c.rowcount:
            logger.info(
                "community_cues: migradas %s filas de %s -> %s (device %s)",
                c.rowcount, legacy_key[:8],
                fingerprint[:8], device_id[:12],
            )
    return sanitized


def register_community_endpoints(app, db):
    """Registra los endpoints de community cues en la app FastAPI"""

    @app.post("/community-cues", response_model=dict)
    @app.post("/community/cues", response_model=dict)  # alias: el cliente llama con barra
    async def upload_community_cues(upload: CueUpload, http: Request = None):
        """
        Un DJ sube sus cues para un track (por fingerprint).
        Se borran cues anteriores del mismo device_id para ese fingerprint
        y se insertan los nuevos.

        Pasa por el guard SEC-12 como los demas votos: las zonas que se
        devuelven llevan `dj_count` y `confidence`, o sea que esto TAMBIEN es
        consenso agregado, y con ids inventados desde una IP se inventaban
        zonas "validadas por N DJs".
        """
        from datetime import datetime

        # Sin cues = el DJ los borró todos: se retira lo que este aparato
        # aportaba a las zonas. Hasta el 2026-09-26 una lista vacía era un
        # error y el cliente ni la mandaba, así que borrar los cues de un tema
        # dejaba los viejos contando en las zonas para siempre.
        if not upload.fingerprint:
            return {"status": "error", "message": "fingerprint requerido"}

        if http is not None:
            from routes.community import _guard_vote_source

            _guard_vote_source(http, upload.fingerprint, upload.device_id)

        conn = db.conn
        c = conn.cursor()

        try:
            sanitized = _guardar_cues(
                c, upload.fingerprint, upload.device_id, upload.cues,
                upload.legacy_key, datetime.now().isoformat(),
            )
            conn.commit()

            # Devolver zonas actualizadas agregando TODO el cluster acustico
            # (todas las versiones del mismo audio), no solo este fingerprint.
            cluster = db.fingerprints_in_cluster(upload.fingerprint)
            ph = ','.join('?' * len(cluster))
            c.execute(
                f'SELECT {_CUES_COLUMNS} FROM community_cues WHERE fingerprint IN ({ph})',
                cluster,
            )
            all_rows = c.fetchall()
            cuentas = db.cuentas_de_votantes(r['device_id'] for r in all_rows)

            # Obtener duracion del track si existe (cualquier version del cluster)
            c.execute(
                f'SELECT duration FROM tracks WHERE fingerprint IN ({ph}) '
                'AND duration IS NOT NULL LIMIT 1',
                cluster,
            )
            dur_row = c.fetchone()
            duration = dur_row['duration'] if dur_row else 0

            zones = aggregate_cues_into_zones(all_rows, duration, cuentas)

            return {
                "status": "ok",
                "cues_saved": len(sanitized),
                "total_contributors": contribuyentes(all_rows, cuentas),
                "zones": zones,
            }

        except sqlite3.Error as e:
            logger.error(f"Error saving community cues: {e}")
            return {"status": "error", "message": str(e)}

    @app.post("/community/cues/batch", response_model=dict)
    async def upload_community_cues_batch(upload: CueBatchUpload,
                                          http: Request):
        """Los cues de muchos temas de un aparato, de una vez.

        Es la puerta de lo IMPORTADO: los cues que trae un XML de Rekordbox,
        Traktor o VirtualDJ. Hasta el 2026-09-26 se guardaban en la biblioteca
        del DJ y no salían de ahí: solo subían los cues que se tocaban a mano
        o se exportaban, así que una biblioteca entera de hot cues no aportaba
        nada a las zonas de nadie. Un tema por petición serían 3.000
        peticiones al importar una biblioteca: por eso va por lotes.

        Por tema hace lo mismo que `/community/cues` (sustituye los cues de
        ESTE aparato para esa huella), pero no devuelve zonas: recalcularlas
        tema a tema es justo el coste que el lote viene a ahorrar.

        Exige un aparato REGISTRADO: el `device_id` sale del `X-Device-Token`,
        no del cuerpo, así que no se puede escribir en nombre de otro. Y cada
        tema pasa por el mismo guard de procedencia que un voto suelto: el
        que lo rechaza se cuenta y el resto sigue.
        """
        from datetime import datetime

        from fastapi import HTTPException
        from routes.community import _guard_vote_source
        from sync_endpoints import dispositivo_del_token

        device = dispositivo_del_token(http.headers.get("X-Device-Token", ""))
        if not device:
            raise HTTPException(401, "Device token required")
        if len(upload.items) > MAX_TEMAS_POR_LOTE:
            raise HTTPException(
                400, f"Máximo {MAX_TEMAS_POR_LOTE} temas por petición")

        # Primero se decide qué entra y DESPUÉS se escribe. El guard apunta
        # la procedencia con su propia conexión: si se intercalara con las
        # escrituras de `db.conn`, esperaría al cerrojo de la transacción
        # abierta en cada tema.
        aceptados = []
        descartados = 0
        for item in upload.items:
            # Solo la HUELLA: una clave que no lo es (el `track.id` de un
            # cliente viejo, un id de ghost) escribiría en un compartimento
            # que no ve nadie.
            # Sin cues también entra: retira lo de este aparato (ver el
            # envío suelto).
            if not _HUELLA.fullmatch(item.fingerprint or ''):
                descartados += 1
                continue
            try:
                _guard_vote_source(http, item.fingerprint, device)
            except HTTPException:
                descartados += 1
                continue
            legacy = item.legacy_key
            aceptados.append(
                (item, legacy if legacy and _HUELLA.fullmatch(legacy) else None))

        conn = db.conn
        c = conn.cursor()
        now = datetime.now().isoformat()
        cues = 0
        try:
            for item, legacy in aceptados:
                cues += len(_guardar_cues(
                    c, item.fingerprint, device, item.cues, legacy, now))
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error saving community cues batch: {e}")
            raise HTTPException(503, "No se pudieron guardar los cues")

        return {"status": "ok", "temas": len(aceptados), "cues": cues,
                "descartados": descartados}

    @app.get("/community-cues/{fingerprint}", response_model=CommunityResponse)
    @app.get("/community/cues/{fingerprint}", response_model=CommunityResponse)  # alias barra
    async def get_community_cues(fingerprint: str, also: Optional[str] = None):
        """
        Obtiene las zonas comunitarias agregadas para un track.
        Solo devuelve zonas con >= 2 DJs de acuerdo.

        `also` es la clave con la que el cliente subia ANTES este track (ver el
        bloque LA CLAVE de la cabecera). Se suma al conjunto consultado para
        que lo ya subido no desaparezca durante la transicion. Es solo lectura.
        """
        conn = db.conn
        c = conn.cursor()

        # Memoria colectiva por SONIDO: agregar los cues de TODAS las versiones
        # del mismo audio (cluster acustico), no solo las de este fingerprint.
        # Asi los cues que otro DJ marco en el flac aparecen al abrir el mp3.
        cluster = db.fingerprints_in_cluster(fingerprint)
        if also and also not in cluster:
            cluster = cluster + [also]
        ph = ','.join('?' * len(cluster))
        c.execute(
            f'SELECT {_CUES_COLUMNS} FROM community_cues WHERE fingerprint IN ({ph})',
            cluster,
        )
        rows = c.fetchall()

        if not rows:
            return CommunityResponse(
                fingerprint=fingerprint,
                zones=[],
                total_contributors=0,
            )

        cuentas = db.cuentas_de_votantes(r['device_id'] for r in rows)

        # Obtener duracion (de cualquier version del cluster).
        c.execute(
            f'SELECT duration FROM tracks WHERE fingerprint IN ({ph}) '
            'AND duration IS NOT NULL LIMIT 1',
            cluster,
        )
        dur_row = c.fetchone()
        duration = dur_row['duration'] if dur_row else 0

        zones_raw = aggregate_cues_into_zones(rows, duration, cuentas)
        zones = [CommunityZoneResponse(**z) for z in zones_raw]

        # Ultima actualizacion
        last = max(r['created_at'] for r in rows) if rows else None

        return CommunityResponse(
            fingerprint=fingerprint,
            zones=zones,
            total_contributors=contribuyentes(rows, cuentas),
            last_updated=last,
        )

    # Aquí vivían tres rutas más, quitadas el 2026-11-12 (auditoría, S6):
    # `/community-cues/{huella}/raw` («para debug»: cada cue de cada aparato,
    # con su nota) y `/user-cues/{device_id}[/{huella}]` (los cues y las notas
    # de UN aparato, a quien supiera su id). Sin credencial las tres, y ningún
    # cliente las llamó nunca: los cues del DJ viajan entre sus aparatos por
    # `/sync` (`all_cues`), no por aquí. No las resucites sin el
    # `X-Device-Token` del aparato.

    logger.info("Community cues endpoints registered")
