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

_PRUEBA = re.compile(
    r'(pista[ _-]?de[ _-]?prueba|\bprueba\b|\btest\b|\btesting\b|test[ _-]?tone|'
    r'\bsine\b|\bseno\b|sint[eé]tic|\bbeep\b|\bdummy\b|\bplaceholder\b|'
    r'\bclick[ _-]?track\b|\bdja[ _-]?test)', re.IGNORECASE)


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


def diagnostico_de_tracks(filas: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Any], set]:
    """Las clases de `tracks`. Devuelve también las claves (huellas e ids)
    que existen, para saber qué ficheros son huérfanos."""
    clases = {k: _Clase() for k in (
        'fallidos_ilegibles', 'fallidos_otros', 'sin_datos', 'muy_cortos',
        'muy_largos', 'nombres_de_prueba')}
    claves = set()
    clusters: Dict[str, Dict[str, Any]] = {}
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
        if 0 < dur < CORTO_S:
            clases['muy_cortos'].suma(ej)
        elif dur > LARGO_S:
            clases['muy_largos'].suma(ej)
        if es_nombre_de_prueba(artista, titulo, f.get('filename')):
            clases['nombres_de_prueba'].suma(ej)
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
