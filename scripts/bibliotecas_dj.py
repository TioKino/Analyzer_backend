"""Lo que dicen de cada fichero los programas de DJ del owner: Rekordbox (XML),
Traktor (collection.nml) y VirtualDJ (database.xml).

Lo usan las medidas que corren en el ordenador del DJ, sobre su disco
(`medir_tonalidad.py`, `medir_desfase_mp3.py`; estudio de la competencia del
2026-11-05, `COMPETENCIA.md` del cliente). Lee lo mismo que el import de la app
(`lib/services/library_import_service.dart`) y con las mismas unidades:

| Programa  | Rejilla                                  | Cues                     |
|-----------|------------------------------------------|--------------------------|
| Rekordbox | `<TEMPO Inizio>`, segundos               | `<POSITION_MARK Start>`, s |
| Traktor   | `<CUE_V2 TYPE="4" START>`, MILISEGUNDOS  | `<CUE_V2 START>`, ms     |
| VirtualDJ | `<Scan Phase>` (un beat cualquiera), s   | `<Poi Pos>`, s           |

Solo lee: no escribe nada en ningún sitio. Sin dependencias más allá de la
biblioteca estándar (mutagen solo para la etiqueta de tonalidad del fichero, y
si no está, se sigue sin ella).
"""

import os
import re
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

PROGRAMAS = ('rekordbox', 'traktor', 'virtualdj')

# Las 12 notas, de Do a Si, en sostenidos.
_NOTAS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_BEMOLES = {'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#',
            'E#': 'F', 'B#': 'C', 'FB': 'E', 'CB': 'B'}


@dataclass
class Cue:
    segundos: float
    tipo: str            # 'cue' o 'loop'
    pad: int = -1        # 0-7, o -1 si no está en un pad


@dataclass
class TemaDeUnPrograma:
    programa: str
    ruta: str
    tonalidad: Optional[str] = None   # en Camelot («8A»)
    bpm: Optional[float] = None
    ancla: Optional[float] = None     # segundos de un beat de su rejilla
    cues: List[Cue] = field(default_factory=list)


# --------------------------------------------------------------- tonalidades

def camelot_de(tonica: int, mayor: bool) -> str:
    """(9, False) → «8A». La misma cuenta que `tonalidad.camelot_de`."""
    if mayor:
        return f'{(tonica * 7 % 12 + 7) % 12 + 1}B'
    return f'{((tonica - 9) * 7 % 12 + 7) % 12 + 1}A'


def camelot_de_texto(raw: Optional[str]) -> Optional[str]:
    """La tonalidad que escribe CUALQUIER programa, en Camelot: «Am», «Bbm»,
    «A♭m», «F# min», «C major», «8A», «08A» y el Open Key de Traktor («1m»,
    «1d»). None si no se entiende. Dice lo mismo que `main.normalize_musical_key`
    (lo ata el test), sin cargar el servidor."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().replace('\u266f', '#').replace('\u266d', 'b').upper()
    s = s.replace('\u00a0', ' ')
    if not s or s in ('?', 'UNKNOWN', 'NONE', '-'):
        return None
    compacto = s.replace(' ', '')
    m = re.match(r'^(\d{1,2})([AB])$', compacto)
    if m:
        n = int(m.group(1))
        return f'{n}{m.group(2)}' if 1 <= n <= 12 else None
    m = re.match(r'^(\d{1,2})([DM])$', compacto)
    if m:
        n = int(m.group(1))
        if not 1 <= n <= 12:
            return None
        return f'{((n + 6) % 12) + 1}{"B" if m.group(2) == "D" else "A"}'
    s = re.sub(r'[\s_\-]*(MINOR|MIN|MOLL)$', 'M', s)
    s = re.sub(r'[\s_\-]*(MAJOR|MAJ|DUR)$', '', s)
    s = s.replace(' ', '')
    menor = s.endswith('M')
    nota = s[:-1] if menor else s
    nota = _BEMOLES.get(nota, nota)
    if nota not in _NOTAS:
        return None
    return camelot_de(_NOTAS.index(nota), not menor)


def camelot_de_traktor(valor) -> Optional[str]:
    """`<MUSICAL_KEY VALUE>` de Traktor: 0-11 mayores de Do a Si y 12-23
    menores en el mismo orden (medido contra un Traktor 4.5.1, ver
    `traktor_keys.dart` del cliente)."""
    try:
        v = int(valor)
    except (TypeError, ValueError):
        return None
    if not 0 <= v <= 23:
        return None
    return camelot_de(v % 12, v < 12)


def tonalidad_de_la_etiqueta(ruta: str) -> Optional[str]:
    """La tonalidad que va DENTRO del fichero (TKEY en ID3, INITIALKEY en
    FLAC/OGG, `initialkey` en M4A): la escriben Beatport, Mixed In Key y otros.
    None si no hay o no se puede leer (o si falta mutagen)."""
    try:
        import mutagen
        f = mutagen.File(ruta)
    except Exception:
        return None
    if f is None or not getattr(f, 'tags', None):
        return None
    tags = f.tags
    candidatos = []
    try:
        if 'TKEY' in tags:
            candidatos.append(str(tags['TKEY'].text[0]))
    except Exception:
        pass
    for nombre in ('initialkey', 'INITIALKEY', 'key', 'KEY',
                   '----:com.apple.iTunes:initialkey',
                   '----:com.apple.iTunes:INITIALKEY'):
        try:
            v = tags.get(nombre)
        except Exception:
            v = None
        if not v:
            continue
        v = v[0] if isinstance(v, (list, tuple)) else v
        if isinstance(v, bytes):
            v = v.decode('utf-8', 'replace')
        candidatos.append(str(v))
    for c in candidatos:
        k = camelot_de_texto(c)
        if k:
            return k
    return None


# --------------------------------------------------------------- rutas

def clave_de_ruta(ruta: str) -> str:
    """Para casar el mismo fichero entre programas: NFC (macOS guarda los
    acentos descompuestos y los XML compuestos), barras de Unix y minúsculas."""
    r = unicodedata.normalize('NFC', ruta or '').replace('\\', '/')
    return r.lower()


def ruta_de_rekordbox(location: str) -> str:
    """`file://localhost/Users/x/a%20b.mp3` → `/Users/x/a b.mp3`; en Windows,
    `file://localhost/C:/…` → `C:/…`. Como `decodeRekordboxLocation`."""
    if not location:
        return ''
    d = urllib.parse.unquote(location)
    if d.startswith('file://localhost/'):
        d = d[len('file://localhost/'):]
    elif d.startswith('file:///'):
        d = d[len('file:///'):]
    else:
        return d
    if not re.match(r'^[A-Za-z]:', d) and not d.startswith('/'):
        d = '/' + d
    return d


def _apunta_a_la_raiz(montaje: str) -> bool:
    try:
        return os.path.realpath(montaje) == '/'
    except OSError:
        return False


def ruta_de_traktor(volumen: str, carpeta: str, fichero: str,
                    existe=os.path.exists) -> str:
    """`VOLUME` + `DIR` (carpetas separadas por `/:`) + `FILE`. En el Mac el
    disco de arranque es un enlace a `/` dentro de `/Volumes`, y un disco
    externo cuelga de `/Volumes` aunque esté desenchufado. Como
    `_reconstructTraktorPath` + `_traktorVolumePrefix`."""
    if not carpeta and not fichero:
        return ''
    d = (carpeta or '').replace('/:', '/')
    if not d.startswith('/'):
        d = '/' + d
    if re.match(r'^[A-Za-z]:$', volumen or ''):
        return f'{volumen}{d}{fichero}'
    if not volumen:
        return f'{d}{fichero}'
    montaje = f'/Volumes/{volumen}'
    if _apunta_a_la_raiz(montaje):
        return f'{d}{fichero}'
    for base in ('/Volumes', '/media', '/run/media'):
        if existe(f'{base}/{volumen}'):
            return f'{base}/{volumen}{d}{fichero}'
    if existe(f'{d}{fichero}'):
        return f'{d}{fichero}'
    return f'{d}{fichero}' if volumen == 'Macintosh HD' else f'{montaje}{d}{fichero}'


def _float(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def _entero(x, defecto: int = -1) -> int:
    """Un entero de un atributo, o `defecto` si no lo es. Ojo: el pad 0 es
    un pad (el A), no «nada»."""
    v = _float(x)
    return int(v) if v is not None else defecto


# --------------------------------------------------------------- los tres

def leer_rekordbox(ruta_xml: str) -> List[TemaDeUnPrograma]:
    """El `rekordbox.xml` (Archivo → Exportar colección en formato xml)."""
    temas = []
    for _, el in ET.iterparse(ruta_xml, events=('end',)):
        if el.tag != 'TRACK':
            continue
        loc = el.get('Location')
        if not loc:
            continue          # las entradas de una lista solo llevan Key
        t = TemaDeUnPrograma('rekordbox', ruta_de_rekordbox(loc),
                             tonalidad=camelot_de_texto(el.get('Tonality')),
                             bpm=_float(el.get('AverageBpm')))
        tempo = el.find('TEMPO')
        if tempo is not None:
            t.ancla = _float(tempo.get('Inizio'))
            t.bpm = _float(tempo.get('Bpm')) or t.bpm
        for pm in el.findall('POSITION_MARK'):
            s = _float(pm.get('Start'))
            if s is None:
                continue
            num = _entero(pm.get('Num'))
            tipo = 'loop' if pm.get('Type') == '4' else 'cue'
            t.cues.append(Cue(s, tipo, num if 0 <= num < 8 else -1))
        temas.append(t)
        el.clear()
    return temas


def leer_traktor(ruta_nml: str, existe=os.path.exists) -> List[TemaDeUnPrograma]:
    """El `collection.nml` de Traktor (también una lista exportada en NML)."""
    temas = []
    for _, el in ET.iterparse(ruta_nml, events=('end',)):
        if el.tag != 'ENTRY':
            continue
        loc = el.find('LOCATION')
        if loc is None:
            el.clear()
            continue
        ruta = ruta_de_traktor(loc.get('VOLUME', ''), loc.get('DIR', ''),
                               loc.get('FILE', ''), existe=existe)
        if not ruta:
            el.clear()
            continue
        t = TemaDeUnPrograma('traktor', ruta)
        mk = el.find('MUSICAL_KEY')
        if mk is not None:
            t.tonalidad = camelot_de_traktor(mk.get('VALUE'))
        info = el.find('INFO')
        if t.tonalidad is None and info is not None:
            t.tonalidad = camelot_de_texto(info.get('KEY'))
        tempo = el.find('TEMPO')
        if tempo is not None:
            t.bpm = _float(tempo.get('BPM'))
        for cue in el.findall('CUE_V2'):
            ms = _float(cue.get('START'))
            if ms is None:
                continue
            tipo = cue.get('TYPE')
            if tipo == '4':            # el marcador de la rejilla
                if t.ancla is None:
                    t.ancla = ms / 1000.0
                continue
            if tipo not in ('0', '5'):  # fade in/out, load: no son del DJ
                continue
            hot = _entero(cue.get('HOTCUE'))
            t.cues.append(Cue(ms / 1000.0, 'loop' if tipo == '5' else 'cue',
                              hot if 0 <= hot < 8 else -1))
        temas.append(t)
        el.clear()
    return temas


def leer_virtualdj(ruta_xml: str) -> List[TemaDeUnPrograma]:
    """El `database.xml` de VirtualDJ. Se lee con los bytes que no son UTF-8
    sustituidos (hay nombres de cue así en bases de datos de verdad, y el
    decodificador estricto tumbaba el import entero)."""
    with open(ruta_xml, 'rb') as f:
        texto = f.read().decode('utf-8', 'replace')
    raiz = ET.fromstring(texto)
    temas = []
    for song in raiz.iter('Song'):
        crudo = song.get('FilePath') or ''
        if '://' in crudo and not crudo.startswith('file:'):
            continue                   # streaming (netsearch://, tidal://…)
        ruta = ruta_de_rekordbox(crudo) if crudo.startswith('file:') else crudo
        if not ruta:
            continue
        t = TemaDeUnPrograma('virtualdj', ruta)
        scan, tags = song.find('Scan'), song.find('Tags')
        clave = (scan.get('Key') if scan is not None else None) or \
                (tags.get('Key') if tags is not None else None)
        t.tonalidad = camelot_de_texto(clave)
        bpm = _float(scan.get('Bpm')) if scan is not None else None
        if bpm and bpm > 0:
            t.bpm = 60.0 / bpm if bpm < 10 else bpm
        fase = _float(scan.get('Phase')) if scan is not None else None
        if fase and fase > 0:
            t.ancla = fase
        for poi in song.findall('Poi'):
            pos = _float(poi.get('Pos'))
            tipo = poi.get('Type') or 'cue'
            if pos is None:
                continue
            if tipo == 'beatgrid':
                if t.ancla is None and pos > 0:
                    t.ancla = pos
                continue
            if tipo not in ('cue', 'loop'):
                continue               # automix, remix: los pone VirtualDJ
            num = _entero(poi.get('Num'), 0)
            t.cues.append(Cue(pos, tipo, num - 1 if 1 <= num <= 8 else -1))
        temas.append(t)
    return temas


#: Dónde está cada biblioteca, para decírselo al DJ cuando la ruta no sirve.
COMO_CONSEGUIRLA = {
    'rekordbox': 'Rekordbox no guarda ese fichero solo: en Rekordbox, '
                 'Archivo → Exportar colección en formato xml, y guárdalo ahí',
    'traktor': 'es la colección de Traktor: '
               'Documents/Native Instruments/Traktor <versión>/collection.nml',
    'virtualdj': 'es la base de datos de VirtualDJ: en el Mac, '
                 '~/Library/Application Support/VirtualDJ/database.xml',
}


def leer_bibliotecas(rekordbox: Optional[str] = None, traktor: Optional[str] = None,
                     virtualdj: Optional[str] = None,
                     leidas: Optional[Dict[str, int]] = None,
                     avisar=print,
                     ) -> Dict[str, Dict[str, TemaDeUnPrograma]]:
    """Lo de cada programa por fichero: {clave de la ruta: {programa: tema}}.

    Una biblioteca que no está o no se puede leer NO corta la medida: se dice
    por qué y cómo conseguirla (`avisar`) y se sigue con las demás. En
    [leidas] queda cuántos temas salieron de cada programa que sí se leyó.
    """
    por_fichero: Dict[str, Dict[str, TemaDeUnPrograma]] = {}
    fuentes = (('rekordbox', rekordbox, leer_rekordbox),
               ('traktor', traktor, leer_traktor),
               ('virtualdj', virtualdj, leer_virtualdj))
    for programa, ruta, lector in fuentes:
        if not ruta:
            continue
        ruta = os.path.expanduser(ruta)
        if not os.path.isfile(ruta):
            avisar(f'[{programa}] no está {ruta}: {COMO_CONSEGUIRLA[programa]}. '
                   'Sigo con lo demás.')
            continue
        try:
            temas = lector(ruta)
        except (ET.ParseError, OSError) as e:
            avisar(f'[{programa}] {ruta} no se puede leer ({e}). Sigo con lo demás.')
            continue
        if leidas is not None:
            leidas[programa] = len(temas)
        for t in temas:
            por_fichero.setdefault(clave_de_ruta(t.ruta), {})[programa] = t
    return por_fichero


def ruta_que_existe(temas: Iterable[TemaDeUnPrograma],
                    existe=os.path.exists) -> Optional[str]:
    """La ruta de un fichero que de verdad está en el disco, de cualquiera de
    los programas que lo tienen."""
    for t in temas:
        for r in (t.ruta, unicodedata.normalize('NFD', t.ruta)):
            if r and existe(r):
                return r
    return None


def diferencia_de_fase(a: float, b: float, beat: float) -> float:
    """Cuánto se separan dos líneas de rejilla, módulo el beat, en segundos y
    con signo (de −beat/2 a +beat/2). Un beat justo es la misma rejilla."""
    if beat <= 0:
        return a - b
    d = (a - b) % beat
    return d - beat if d > beat / 2 else d


def emparejar_cues(a: List[Cue], b: List[Cue], tope: float = 0.15
                   ) -> List[Tuple[Cue, Cue, float]]:
    """Los cues del mismo tema en dos programas, uno a uno por cercanía (cada
    uno como mucho con otro, a menos de `tope` segundos): (a, b, a − b)."""
    usados = set()
    pares = []
    for ca in sorted(a, key=lambda c: c.segundos):
        mejor, cual = None, None
        for j, cb in enumerate(b):
            if j in usados:
                continue
            d = ca.segundos - cb.segundos
            if abs(d) <= tope and (mejor is None or abs(d) < abs(mejor)):
                mejor, cual = d, j
        if cual is not None:
            usados.add(cual)
            pares.append((ca, b[cual], mejor))
    return pares
