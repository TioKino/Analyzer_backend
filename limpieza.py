"""Basura en el servidor: el DIAGNÓSTICO, que no borra nada (2026-11-19).

El owner, antes de la 2.11.0: «desde los inicios hemos pasado muchísimas fases
de prueba y fijo que hay basura» en la base de datos y en el disco de Render.
Borrar a ciegas no: la memoria colectiva es para siempre (decisión del
2026-11-13) y una fila que parece basura puede ser lo que evita que un
aparato vuelva a subir un fichero roto. Así que primero se MIRA:
`GET /admin/limpieza` (y `scripts/limpieza.sh` del cliente) cuenta cada clase
de sospechoso, con ejemplos, y el borrado se decide con los números delante.

Las clases, de más segura a más dudosa:

  ficheros_rotos        previews y portadas vacíos, diminutos o que no son lo
                        que dicen (ni MP3 ni JPEG/PNG/WebP/GIF), y temporales
                        `.tmp` que se quedaron a medias.
  ficheros_huerfanos    previews y portadas cuya huella no tiene ninguna fila
                        en `tracks`: nadie los pide por nada que exista.
  fallidos_ilegibles    `/analyze` no pudo leer el fichero (ffmpeg no lo abre):
                        BPM 0, sin tonalidad. OJO: la fila es la que hace que
                        el aparato NO vuelva a subirlo (`analisis_fallido.py`).
  fallidos_otros        fallos que no son del fichero; se reintentan solos.
  sin_datos             ni artista, ni título, ni BPM, y no es un fallo ni una
                        detección de Escuchar: una fila que no dice nada.
  muy_cortos            menos de 30 s: un trozo, un preview, un sample.
  muy_largos            más de 20 min: una sesión analizada como tema (hoy el
                        escritorio las aparta como mezclas).
  huella_compartida     un cluster acústico con temas de nombres distintos:
                        la huella del SILENCIO o de un fichero roto, que junta
                        cosas que no se parecen (ver «Duplicados» del cliente).
  nombres_de_prueba     el nombre dice prueba, test, tono, sintético…: las
                        pistas de las mediciones (desfase, tonalidad).
  partes_de_sesion      una SESIÓN partida en pistas que se subió tema a
                        tema: el mismo artista y títulos iguales salvo el
                        número («Pista 1», «Pista 2»…), tres o más, y algo
                        dice sesión (session, set, mix, live, @…). Es la
                        regla de `sesion_en_partes.dart` del cliente, que
                        desde la 2.11.0 las aparta sin subirlas; lo de antes
                        sigue aquí. Sin la palabra de sesión van aparte
                        (`partes_posibles`): pueden ser un álbum sin nombres.

No se cuentan aparte las detecciones que siembra Escuchar
(`analysis_status: recognize_only`): son a propósito.
Lo ata `test_la_limpieza_mira_antes_de_borrar.py`.
"""

import json
import os
import re
import shutil
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Por debajo, un preview o una portada no son un fichero de verdad.
MIN_BYTES_PREVIEW = 4 * 1024
MIN_BYTES_PORTADA = 512

CORTO_S = 30.0
LARGO_S = 20 * 60.0

# Un cluster con tantas filas y tantos nombres distintos es sospechoso.
CLUSTER_MIN_FILAS = 4
CLUSTER_MIN_NOMBRES = 3

EJEMPLOS = 10

# Las partes de una sesión (la regla del cliente, `sesion_en_partes.dart`).
MIN_PARTES = 3
_SESION = re.compile(
    r'\b(sesion(?:es)?|sessions?|(?<!box )sets?|dj set|live|directo|en vivo'
    r'|mix(?:es|ed)?|megamix|minimix|mixtape|podcast|radio ?show|spinn?ing'
    r'|b2b)\b|(@)')
_DIGITOS = re.compile(r'\d+')
_NO_LETRAS = re.compile(r'[^a-z0-9#]+')
_ARTISTA_VACIO = {
    '', 'unknown', 'unknown artist', 'artista desconocido', 'desconocido',
    'various', 'various artists', 'varios', 'varios artistas', 'va',
}


def _sin_acentos(s: Optional[str]) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


def dice_que_es_una_sesion(*textos: Optional[str]) -> bool:
    return any(t and _SESION.search(_sin_acentos(t)) for t in textos)


def patron_numerado(titulo: Optional[str]) -> Optional[str]:
    """El título con los números como «#», o None si no lleva número."""
    t = _sin_acentos(titulo)
    if not _DIGITOS.search(t):
        return None
    p = re.sub(r'\s+', ' ', _NO_LETRAS.sub(' ', _DIGITOS.sub('#', t))).strip()
    return p or None

# Solo lo que dice ser una prueba técnica: la primera versión pillaba «A Test
# of Patience», «Big Dummy», «Beep Me 911» y «Il Seno» (lectura del 19-nov).
_PRUEBA = re.compile(
    r'(pista[ _-]?de[ _-]?prueba|tono[ _-]?de[ _-]?prueba|test[ _-]?tone|'
    r'sine[ _-]?(wave|sweep)|pink[ _-]?noise|white[ _-]?noise|'
    r'sint[eé]tic[ao]|\bdja[ _-]?test)', re.IGNORECASE)


def _normal(s: Optional[str]) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


def es_nombre_de_prueba(*textos: Optional[str]) -> bool:
    return any(t and _PRUEBA.search(t) for t in textos)


def tipo_de_fichero(cabeza: bytes) -> Optional[str]:
    """Qué es por sus primeros bytes: mp3, jpg, png, webp, gif o None."""
    if cabeza.startswith(b'ID3') or (
            len(cabeza) > 1 and cabeza[0] == 0xFF and (cabeza[1] & 0xE0) == 0xE0):
        return 'mp3'
    if cabeza.startswith(b'\xFF\xD8\xFF'):
        return 'jpg'
    if cabeza.startswith(b'\x89PNG'):
        return 'png'
    if cabeza.startswith(b'RIFF') and cabeza[8:12] == b'WEBP':
        return 'webp'
    if cabeza.startswith(b'GIF8'):
        return 'gif'
    return None


def _estado(analysis_json: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(analysis_status, fallo) de una fila."""
    if not analysis_json:
        return None, None
    try:
        d = json.loads(analysis_json)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(d, dict):
        return None, None
    return d.get('analysis_status'), d.get('fallo')


def _ejemplo(f: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'huella': f.get('fingerprint') or f.get('id'),
        'fichero': f.get('filename'),
        'artista': f.get('artist'),
        'titulo': f.get('title'),
        'bpm': f.get('bpm'),
        'duracion': f.get('duration'),
        'analizado': (f.get('analyzed_at') or '')[:10],
    }


class _Clase:
    def __init__(self):
        self.n = 0
        self.bytes = 0
        self.ejemplos: List[Dict[str, Any]] = []

    def suma(self, ejemplo: Dict[str, Any], bytes_: int = 0):
        self.n += 1
        self.bytes += bytes_
        if len(self.ejemplos) < EJEMPLOS:
            self.ejemplos.append(ejemplo)

    def dict(self, con_bytes: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {'n': self.n, 'ejemplos': self.ejemplos}
        if con_bytes:
            d['bytes'] = self.bytes
        return d


def diagnostico_de_tracks(filas: Iterable[Dict[str, Any]],
                          ids: Optional[Dict[str, List[str]]] = None
                          ) -> Tuple[Dict[str, Any], set]:
    """Las clases de `tracks`. Devuelve también las claves (huellas e ids)
    que existen, para saber qué ficheros son huérfanos."""
    clases = {k: _Clase() for k in (
        'fallidos_ilegibles', 'fallidos_otros', 'sin_datos', 'muy_cortos',
        'muy_largos', 'nombres_de_prueba')}
    claves = set()
    clusters: Dict[str, Dict[str, Any]] = {}
    numeradas: Dict[Tuple[str, str], Dict[str, Any]] = {}
    total = 0
    detecciones = 0
    for f in filas:
        total += 1
        for k in ('fingerprint', 'id'):
            if f.get(k):
                claves.add(str(f[k]))
        if 'estado' in f:
            estado, fallo = f.get('estado'), f.get('fallo')
        else:
            estado, fallo = _estado(f.get('analysis_json'))
        if estado == 'recognize_only':
            detecciones += 1
            continue
        ej = _ejemplo(f)
        if estado == 'failed':
            clases['fallidos_ilegibles' if fallo == 'ilegible'
                   else 'fallidos_otros'].suma(ej)
            continue
        artista = (f.get('artist') or '').strip()
        titulo = (f.get('title') or '').strip()
        bpm = f.get('bpm') or 0
        dur = f.get('duration') or 0
        if not artista and not titulo and not bpm:
            clases['sin_datos'].suma(ej)
            if ids is not None:
                ids.setdefault('sin_datos', []).append(f['id'])
        if 0 < dur < CORTO_S:
            clases['muy_cortos'].suma(ej)
        elif dur > LARGO_S:
            clases['muy_largos'].suma(ej)
        if es_nombre_de_prueba(artista, titulo, f.get('filename')):
            clases['nombres_de_prueba'].suma(ej)
        patron = patron_numerado(titulo)
        artista_n = _normal(artista)
        if patron and artista_n not in _ARTISTA_VACIO:
            g = numeradas.setdefault((artista_n, patron), {
                'filas': 0, 'numeros': set(), 'sesion': False, 'ejemplos': [],
                'ids': []})
            g['ids'].append(f['id'])
            g['filas'] += 1
            g['numeros'].add(tuple(int(x) for x in _DIGITOS.findall(
                _sin_acentos(titulo))))
            if dice_que_es_una_sesion(titulo, f.get('filename'), artista):
                g['sesion'] = True
            if len(g['ejemplos']) < 5:
                g['ejemplos'].append(ej)
        cid = f.get('acoustic_id')
        if cid:
            g = clusters.setdefault(cid, {'filas': 0, 'nombres': set(),
                                          'ejemplos': []})
            g['filas'] += 1
            nombre = _normal(f'{artista} {titulo}') or _normal(f.get('filename'))
            if nombre:
                g['nombres'].add(nombre)
            if len(g['ejemplos']) < 5:
                g['ejemplos'].append(ej)

    sospechosos = [
        (cid, g) for cid, g in clusters.items()
        if g['filas'] >= CLUSTER_MIN_FILAS
        and len(g['nombres']) >= CLUSTER_MIN_NOMBRES
    ]
    sospechosos.sort(key=lambda x: -x[1]['filas'])
    salida = {k: c.dict() for k, c in clases.items()}
    salida['huella_compartida'] = {
        'n': sum(g['filas'] for _, g in sospechosos),
        'clusters': len(sospechosos),
        'ejemplos': [
            {'cluster': cid, 'filas': g['filas'],
             'nombres_distintos': len(g['nombres']), 'temas': g['ejemplos']}
            for cid, g in sospechosos[:EJEMPLOS]
        ],
    }
    if ids is not None:
        ids['partes_de_sesion'] = [
            tid for g in numeradas.values()
            if g['sesion'] and len(g['numeros']) >= MIN_PARTES
            for tid in g['ids']
        ]
    for clase, segura in (('partes_de_sesion', True),
                          ('partes_posibles', False)):
        grupos = [
            (k, g) for k, g in numeradas.items()
            if g['sesion'] == segura and len(g['numeros']) >= MIN_PARTES
        ]
        grupos.sort(key=lambda x: -x[1]['filas'])
        salida[clase] = {
            'n': sum(g['filas'] for _, g in grupos),
            'sesiones': len(grupos),
            'ejemplos': [
                {'artista': g['ejemplos'][0].get('artista'),
                 'patron': k[1], 'pistas': g['filas'], 'temas': g['ejemplos']}
                for k, g in grupos[:EJEMPLOS]
            ],
        }
    salida['total_tracks'] = total
    salida['detecciones_de_escuchar'] = detecciones
    return salida, claves


def diagnostico_de_ficheros(carpeta: Optional[str], claves: set,
                            min_bytes: int, tipos: Tuple[str, ...]) -> Dict[str, Any]:
    """Rotos y huérfanos de una carpeta de assets (previews o portadas)."""
    rotos, huerfanos = _Clase(), _Clase()
    total = total_bytes = 0
    if not carpeta or not os.path.isdir(carpeta):
        return {'carpeta': carpeta, 'existe': False}
    with os.scandir(carpeta) as it:
        for e in it:
            if not e.is_file():
                continue
            try:
                tam = e.stat().st_size
            except OSError:
                continue
            total += 1
            total_bytes += tam
            nombre = e.name
            base, _, ext = nombre.rpartition('.')
            ej = {'fichero': nombre, 'bytes': tam}
            if nombre.endswith('.tmp') or not base:
                rotos.suma(ej, tam)
                continue
            if tam < min_bytes:
                rotos.suma(ej, tam)
                continue
            try:
                with open(e.path, 'rb') as fh:
                    cabeza = fh.read(16)
            except OSError:
                rotos.suma(ej, tam)
                continue
            if tipo_de_fichero(cabeza) not in tipos:
                rotos.suma(ej, tam)
                continue
            if base not in claves:
                huerfanos.suma(ej, tam)
    return {
        'carpeta': carpeta,
        'existe': True,
        'ficheros': total,
        'bytes': total_bytes,
        'rotos': rotos.dict(con_bytes=True),
        'huerfanos': huerfanos.dict(con_bytes=True),
    }


def _bytes_de(carpeta: str) -> int:
    total = 0
    for raiz, _, ficheros in os.walk(carpeta):
        for f in ficheros:
            try:
                total += os.path.getsize(os.path.join(raiz, f))
            except OSError:
                pass
    return total


def diagnostico(conn, previews_dir: Optional[str], artwork_dir: Optional[str],
                database_path: Optional[str] = None) -> Dict[str, Any]:
    """Todo el diagnóstico. Solo LEE: ni una escritura en la base ni en el
    disco."""
    conn.row_factory = _como_dict
    # Sin traer el `analysis_json` a Python (el panel ya se cayó una vez por
    # memoria): las marcas se buscan como texto en SQLite, como el resto del
    # panel, con y sin espacio tras los dos puntos.
    def marca(clave, valor):
        return (f"(instr(analysis_json, '\"{clave}\": \"{valor}\"') > 0"
                f" OR instr(analysis_json, '\"{clave}\":\"{valor}\"') > 0)")
    cur = conn.execute(
        'SELECT id, fingerprint, filename, artist, title, bpm, duration,'
        ' analyzed_at, acoustic_id,'
        f" CASE WHEN {marca('analysis_status', 'failed')} THEN 'failed'"
        f"      WHEN {marca('analysis_status', 'recognize_only')}"
        "            THEN 'recognize_only' END AS estado,"
        f" CASE WHEN {marca('fallo', 'ilegible')} THEN 'ilegible' END AS fallo"
        ' FROM tracks')
    tracks, claves = diagnostico_de_tracks(cur)
    salida: Dict[str, Any] = {
        'tracks': tracks,
        'previews': diagnostico_de_ficheros(
            previews_dir, claves, MIN_BYTES_PREVIEW, ('mp3',)),
        'portadas': diagnostico_de_ficheros(
            artwork_dir, claves, MIN_BYTES_PORTADA,
            ('jpg', 'png', 'webp', 'gif')),
    }
    if database_path:
        datos = os.path.dirname(os.path.abspath(database_path))
        disco: Dict[str, Any] = {'carpeta': datos}
        try:
            u = shutil.disk_usage(datos)
            disco.update(total=u.total, usado=u.used, libre=u.free)
        except OSError:
            pass
        bases = {}
        for nombre in sorted(os.listdir(datos)) if os.path.isdir(datos) else []:
            ruta = os.path.join(datos, nombre)
            if os.path.isfile(ruta) and nombre.endswith(('.db', '.db-wal')):
                bases[nombre] = os.path.getsize(ruta)
        disco['bases'] = bases
        copias = os.path.join(datos, 'copias')
        if os.path.isdir(copias):
            disco['copias'] = _bytes_de(copias)
        salida['disco'] = disco
    return salida


def _como_dict(cursor, fila):
    return {d[0]: fila[i] for i, d in enumerate(cursor.description)}


# ── Separar las huellas compartidas por temas distintos (2026-11-19) ─────────
# Un grupo acústico (`acoustic_id`) con temas de nombres distintos no es el
# mismo audio: es la huella de un sample corto, del silencio o de un fichero
# truncado (ver `acoustic_fingerprint.huella_que_no_dice_nada`, que desde
# ese día impide que vuelva a pasar). Se separa por NOMBRE: los que se llaman
# igual siguen juntos (son copias), el nombre con más temas se queda con el
# grupo de siempre —y con lo que la comunidad guardó bajo él— y cada otro
# nombre pasa a un grupo propio, con la huella de su primer tema como id.
# No se borra ningún tema.

def plan_de_separacion(conn) -> Dict[str, Any]:
    """Qué cambiaría: los grupos sospechosos y, por tema, su grupo nuevo.
    Solo LEE."""
    grupos: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    cur = conn.cursor()
    cur.row_factory = _como_dict
    for f in cur.execute(
            'SELECT id, fingerprint, artist, title, filename, acoustic_id'
            ' FROM tracks WHERE acoustic_id IS NOT NULL'):
        nombre = (_normal(f'{f.get("artist") or ""} {f.get("title") or ""}')
                  or _normal(f.get('filename')) or f['id'])
        grupos.setdefault(f['acoustic_id'], {}).setdefault(
            nombre, []).append((f['id'], f.get('fingerprint') or f['id']))
    cambios: List[Tuple[str, str]] = []  # (id del tema, acoustic_id nuevo)
    resumen = []
    for cid, por_nombre in grupos.items():
        filas = sum(len(v) for v in por_nombre.values())
        if filas < CLUSTER_MIN_FILAS or len(por_nombre) < CLUSTER_MIN_NOMBRES:
            continue
        nombres = sorted(por_nombre.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for nombre, temas in nombres[1:]:
            nuevo = temas[0][1]
            for tid, _ in temas:
                cambios.append((tid, nuevo))
        resumen.append({'cluster': cid, 'temas': filas,
                        'nombres': len(por_nombre),
                        'se_queda': nombres[0][0],
                        'separados': filas - len(nombres[0][1])})
    resumen.sort(key=lambda r: -r['temas'])
    return {'grupos': len(resumen), 'temas_que_cambian': len(cambios),
            'ejemplos': resumen[:EJEMPLOS], '_cambios': cambios}


def separar(conn) -> Dict[str, Any]:
    """Aplica `plan_de_separacion`, en una transacción."""
    plan = plan_de_separacion(conn)
    cambios = plan.pop('_cambios')
    with conn:
        conn.executemany('UPDATE tracks SET acoustic_id = ? WHERE id = ?',
                         [(nuevo, tid) for tid, nuevo in cambios])
    plan['hecho'] = True
    return plan


# ── Borrar lo poco que es basura segura (2026-11-19) ────────────────────────
# Decidido con la primera lectura delante (owner): la fila sin datos, las
# sesiones subidas pista a pista (`partes_de_sesion`: el cliente ya no las
# sube) y los ficheros de previews y portadas VACÍOS o temporales a medias de
# hace más de una hora (uno reciente puede estar escribiéndose). Nada más: ni
# los fallidos (evitan que se vuelva a subir un fichero roto), ni los
# huérfanos (las apps los siguen pidiendo por la huella), ni los dudosos.

TMP_VIEJO_S = 3600


def _ficheros_a_borrar(carpeta: Optional[str], ahora: float) -> List[str]:
    if not carpeta or not os.path.isdir(carpeta):
        return []
    rutas = []
    with os.scandir(carpeta) as it:
        for e in it:
            if not e.is_file():
                continue
            try:
                st = e.stat()
            except OSError:
                continue
            if st.st_size == 0 or (e.name.endswith('.tmp')
                                   and ahora - st.st_mtime > TMP_VIEJO_S):
                rutas.append(e.path)
    return rutas


def plan_de_borrado(conn, previews_dir: Optional[str],
                    artwork_dir: Optional[str],
                    ahora: Optional[float] = None) -> Dict[str, Any]:
    """Qué se borraría. Solo LEE."""
    import time as _t
    ahora = _t.time() if ahora is None else ahora
    cur = conn.cursor()
    cur.row_factory = _como_dict
    ids: Dict[str, List[str]] = {}
    diagnostico_de_tracks(cur.execute(
        'SELECT id, fingerprint, filename, artist, title, bpm, duration,'
        ' analyzed_at, acoustic_id, NULL AS estado, NULL AS fallo'
        ' FROM tracks WHERE analysis_json IS NULL OR ('
        "instr(analysis_json, '\"analysis_status\": \"failed\"') = 0 AND"
        " instr(analysis_json, '\"analysis_status\":\"failed\"') = 0 AND"
        " instr(analysis_json, '\"analysis_status\": \"recognize_only\"') = 0"
        " AND instr(analysis_json, '\"analysis_status\":\"recognize_only\"') = 0)"),
        ids)
    previews = _ficheros_a_borrar(previews_dir, ahora)
    portadas_ = _ficheros_a_borrar(artwork_dir, ahora)
    return {
        'temas_sin_datos': len(ids.get('sin_datos', [])),
        'temas_de_sesiones': len(ids.get('partes_de_sesion', [])),
        'previews_vacias': len(previews),
        'portadas_vacias': len(portadas_),
        '_ids': ids.get('sin_datos', []) + ids.get('partes_de_sesion', []),
        '_ficheros': previews + portadas_,
    }


def borrar(conn, previews_dir: Optional[str],
           artwork_dir: Optional[str]) -> Dict[str, Any]:
    """Aplica `plan_de_borrado`."""
    plan = plan_de_borrado(conn, previews_dir, artwork_dir)
    ids = plan.pop('_ids')
    ficheros = plan.pop('_ficheros')
    with conn:
        conn.executemany('DELETE FROM tracks WHERE id = ?', [(i,) for i in ids])
    borrados = 0
    for ruta in ficheros:
        try:
            os.unlink(ruta)
            borrados += 1
        except OSError:
            pass
    plan.update(hecho=True, filas_borradas=len(ids), ficheros_borrados=borrados)
    return plan
