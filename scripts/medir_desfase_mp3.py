#!/usr/bin/env python3
"""¿Se corren 26 o 50 ms los cues y la rejilla de un MP3 al pasar de un
programa de DJ a otro? Lo mide con TUS bibliotecas y tus ficheros.

Por qué (estudio de la competencia, 2026-11-05): Lexicon corrige solo, en cada
import y en cada sincronización, que «en un pequeño porcentaje de MP3 unos
programas leen 26 o 50 ms de silencio al principio y otros no», y los cues y la
rejilla salen corridos. DJ Analyzer pasa cues de Rekordbox a Traktor y a
VirtualDJ tal cual, y junta en las zonas de la comunidad cues que vienen de
programas distintos: si nos pasa, hay que saberlo antes de tocar nada, y antes
de añadir Serato.

Qué hace: para cada fichero que está en dos programas, compara
- el ANCLA de la rejilla, módulo el beat (si cada programa analizó por su
  cuenta, se separan unos pocos ms; un fichero de los raros se separa ~26 o
  ~50 ms, siempre hacia el mismo lado), y
- los CUES, emparejados uno a uno por cercanía (si los pasó un conversor, en
  los ficheros normales salen idénticos),
y lo reparte por la CABECERA del MP3, que es donde se esconde la causa: si
lleva trama Xing/Info (una trama de audio vacía al principio que unos
programas cuentan y otros no) y si lleva la etiqueta LAME con el retardo del
codificador (los ~25 ms que unos programas descuentan y otros no).

No escribe nada en ningún sitio. Deja un CSV con el detalle por fichero.

Uso (al menos dos programas):

    python3 scripts/medir_desfase_mp3.py \\
        --rekordbox ~/rekordbox.xml \\
        --traktor "~/Documents/Native Instruments/Traktor 4.5.1/collection.nml" \\
        --virtualdj "~/Library/Application Support/VirtualDJ/database.xml"

Sin dependencias más allá de la biblioteca estándar.
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import bibliotecas_dj as bib  # noqa: E402

# --------------------------------------------------------------- la cabecera

_KBPS = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
         2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]}
_HZ = {1: (44100, 48000, 32000), 2: (22050, 24000, 16000), 25: (11025, 12000, 8000)}


def trama(b: bytes) -> Optional[Dict]:
    """La cabecera de una trama MPEG de capa III, o None si esos 4 bytes no lo
    son."""
    if len(b) < 4 or b[0] != 0xFF or (b[1] & 0xE0) != 0xE0:
        return None
    ver, capa = (b[1] >> 3) & 3, (b[1] >> 1) & 3
    if ver == 1 or capa != 1:
        return None
    version = {3: 1, 2: 2, 0: 25}[ver]
    ib, ih, relleno = b[2] >> 4, (b[2] >> 2) & 3, (b[2] >> 1) & 1
    if ib in (0, 15) or ih == 3:
        return None
    kbps = _KBPS[1 if version == 1 else 2][ib]
    hz = _HZ[version][ih]
    canales = 1 if (b[3] >> 6) == 3 else 2
    largo = (144 if version == 1 else 72) * kbps * 1000 // hz + relleno
    return {'version': version, 'hz': hz, 'canales': canales, 'largo': largo,
            'muestras': 1152 if version == 1 else 576}


def cabecera_mp3(ruta: str) -> Dict:
    """Lo que dice el principio de un MP3: cuántos bytes de ID3v2 lleva, si
    hay basura antes de la primera trama, si la primera trama es una Xing/Info
    (vacía de audio) o una VBRI, y si trae la etiqueta LAME con el retardo y el
    relleno del codificador. `tipo` resume lo que importa para el desfase."""
    salida: Dict = {'tipo': 'no es MP3', 'id3': 0, 'basura': 0, 'retardo': None,
                    'relleno': None, 'codificador': '', 'hz': None}
    if not ruta.lower().endswith('.mp3'):
        return salida
    try:
        with open(ruta, 'rb') as f:
            ini = f.read(10)
            id3 = 0
            if len(ini) == 10 and ini[:3] == b'ID3':
                tam = ((ini[6] & 0x7F) << 21) | ((ini[7] & 0x7F) << 14) | \
                      ((ini[8] & 0x7F) << 7) | (ini[9] & 0x7F)
                id3 = 10 + tam + (10 if ini[5] & 0x10 else 0)
            f.seek(id3)
            datos = f.read(65536)
    except OSError:
        salida['tipo'] = 'sin fichero'
        return salida
    salida['id3'] = id3
    i, primera = 0, None
    while i < len(datos) - 4:
        h = trama(datos[i:i + 4])
        if h and i + h['largo'] + 4 <= len(datos):
            sig = trama(datos[i + h['largo']:i + h['largo'] + 4])
            if sig and sig['version'] == h['version'] and sig['hz'] == h['hz']:
                primera = (i, h)
                break
        i += 1
    if primera is None:
        return salida
    i, h = primera
    salida['basura'], salida['hz'] = i, h['hz']
    lado = (32 if h['canales'] == 2 else 17) if h['version'] == 1 else \
           (17 if h['canales'] == 2 else 9)
    x = i + 4 + lado
    marca = datos[x:x + 4]
    if datos[i + 36:i + 40] == b'VBRI':
        salida['tipo'] = 'VBRI'
        return salida
    if marca not in (b'Xing', b'Info'):
        salida['tipo'] = 'sin cabecera'
        return salida
    nombre = marca.decode()
    banderas = int.from_bytes(datos[x + 4:x + 8], 'big')
    p = x + 8
    for bit, n in ((1, 4), (2, 4), (4, 100), (8, 4)):
        if banderas & bit:
            p += n
    cod = datos[p:p + 9]
    if cod[:4] in (b'LAME', b'Lavf', b'Lavc', b'GOGO') or cod[:2] == b'L3':
        salida['codificador'] = cod.decode('latin-1').strip('\x00 ')
        if p + 24 <= len(datos):
            salida['retardo'] = (datos[p + 21] << 4) | (datos[p + 22] >> 4)
            salida['relleno'] = ((datos[p + 22] & 0x0F) << 8) | datos[p + 23]
        salida['tipo'] = f'{nombre}+LAME'
    else:
        salida['tipo'] = nombre
    return salida


# --------------------------------------------------------------- las diferencias

def cajon(ms: float) -> str:
    """En qué grupo cae una diferencia (en ms, con signo)."""
    a = abs(ms)
    if a < 6:
        return '≈0'
    if 18 <= a <= 34:
        return '≈26'
    if 40 <= a <= 60:
        return '≈50'
    return 'otro'


CAJONES = ('≈0', '≈26', '≈50', 'otro')


def beat_comun(a: bib.TemaDeUnPrograma, b: bib.TemaDeUnPrograma) -> Optional[float]:
    """El beat (s) con el que comparar las dos rejillas: los dos BPM tienen que
    decir lo mismo (o uno el doble del otro) a menos de un 0,5 %."""
    if not a.bpm or not b.bpm or a.bpm <= 0 or b.bpm <= 0:
        return None
    alto, bajo = max(a.bpm, b.bpm), min(a.bpm, b.bpm)
    for k in (1.0, 2.0):
        if abs(alto / (bajo * k) - 1) <= 0.005:
            return 60.0 / alto
    return None


def comparar(por_fichero: Dict[str, Dict[str, bib.TemaDeUnPrograma]],
             cabeceras: Dict[str, Dict]) -> Dict:
    """Por pareja de programas: las diferencias de ancla y de cues (ms, A − B),
    con la cabecera de cada fichero."""
    salida: Dict = defaultdict(lambda: {'anclas': [], 'cues': []})
    for clave, temas in por_fichero.items():
        tipo = cabeceras.get(clave, {}).get('tipo', 'sin fichero')
        presentes = [p for p in bib.PROGRAMAS if p in temas]
        for i, pa in enumerate(presentes):
            for pb in presentes[i + 1:]:
                a, b = temas[pa], temas[pb]
                par = salida[f'{pa} ~ {pb}']
                beat = beat_comun(a, b)
                if beat and a.ancla is not None and b.ancla is not None:
                    d = bib.diferencia_de_fase(a.ancla, b.ancla, beat) * 1000
                    par['anclas'].append((clave, tipo, d))
                for _, _, d in bib.emparejar_cues(a.cues, b.cues):
                    par['cues'].append((clave, tipo, d * 1000))
    return salida


def resumen(filas: List[Tuple[str, str, float]]) -> Dict[str, Dict]:
    """Por tipo de cabecera: cuántos ficheros, cuántas diferencias en cada
    cajón y la mediana con signo."""
    por_tipo: Dict[str, Dict] = defaultdict(lambda: {'ficheros': set(), 'ms': []})
    for clave, tipo, d in filas:
        por_tipo[tipo]['ficheros'].add(clave)
        por_tipo[tipo]['ms'].append(d)
        por_tipo['TODOS']['ficheros'].add(clave)
        por_tipo['TODOS']['ms'].append(d)
    salida = {}
    for tipo, v in por_tipo.items():
        c = {k: 0 for k in CAJONES}
        for d in v['ms']:
            c[cajon(d)] += 1
        salida[tipo] = {'ficheros': len(v['ficheros']), 'n': len(v['ms']),
                        'cajones': c, 'mediana': statistics.median(v['ms'])}
    return salida


def imprimir(titulo: str, filas: List[Tuple[str, str, float]], unidad: str) -> None:
    if not filas:
        return
    r = resumen(filas)
    print(f'\n{titulo}: {r["TODOS"]["n"]} {unidad} de {r["TODOS"]["ficheros"]} ficheros')
    print(f'  {"cabecera":<16} {"ficheros":>8} ' + ' '.join(f'{c:>6}' for c in CAJONES)
          + f' {"mediana":>9}')
    orden = sorted((t for t in r if t != 'TODOS'), key=lambda t: -r[t]['n']) + ['TODOS']
    for tipo in orden:
        v = r[tipo]
        pcts = ' '.join(f'{100 * v["cajones"][c] / v["n"]:5.0f}%' for c in CAJONES)
        print(f'  {tipo:<16} {v["ficheros"]:>8} {pcts} {v["mediana"]:+8.1f} ms')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--rekordbox')
    ap.add_argument('--traktor')
    ap.add_argument('--virtualdj')
    ap.add_argument('--csv', default=os.path.expanduser('~/medir_desfase_mp3.csv'))
    args = ap.parse_args(argv)
    if sum(1 for x in (args.rekordbox, args.traktor, args.virtualdj) if x) < 2:
        ap.error('hacen falta al menos dos programas para comparar')

    por_fichero = bib.leer_bibliotecas(args.rekordbox, args.traktor, args.virtualdj)
    en_varios = {k: v for k, v in por_fichero.items() if len(v) >= 2}
    print(f'[Desfase] {len(por_fichero)} ficheros en tus bibliotecas, '
          f'{len(en_varios)} en dos programas o más')
    cabeceras = {}
    for clave, temas in en_varios.items():
        ruta = bib.ruta_que_existe(temas.values())
        cabeceras[clave] = cabecera_mp3(ruta) if ruta else {'tipo': 'sin fichero'}
        cabeceras[clave]['ruta'] = ruta or next(iter(temas.values())).ruta

    print('\n=== Desfase de los MP3 entre programas ===')
    tipos = defaultdict(int)
    for c in cabeceras.values():
        tipos[c['tipo']] += 1
    print('Cabeceras de los ficheros que están en dos programas: '
          + ' · '.join(f'{t} {n}' for t, n in sorted(tipos.items(), key=lambda x: -x[1])))
    comparado = comparar(en_varios, cabeceras)
    for par, datos in sorted(comparado.items()):
        imprimir(f'{par} · ancla de la rejilla (módulo el beat)', datos['anclas'], 'temas')
        imprimir(f'{par} · cues emparejados', datos['cues'], 'cues')
    print('\nmediana: con signo, el primero menos el segundo (positivo: en el primero cae más tarde)')
    print('≈0: menos de 6 ms · ≈26: de 18 a 34 · ≈50: de 40 a 60 · otro: lo demás')

    try:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            pares = sorted(comparado)
            w.writerow(['ruta', 'cabecera', 'codificador', 'retardo', 'relleno', 'id3', 'basura',
                        *(f'ancla {p} (ms)' for p in pares), *(f'cues {p} (mediana ms)' for p in pares)])
            por_clave = {p: (defaultdict(list), defaultdict(list)) for p in pares}
            for p in pares:
                for k, _, d in comparado[p]['anclas']:
                    por_clave[p][0][k].append(d)
                for k, _, d in comparado[p]['cues']:
                    por_clave[p][1][k].append(d)
            for clave, c in sorted(cabeceras.items()):
                anclas, cues = [], []
                for p in pares:
                    a, q = por_clave[p][0].get(clave), por_clave[p][1].get(clave)
                    anclas.append(f'{a[0]:.1f}' if a else '')
                    cues.append(f'{statistics.median(q):.1f}' if q else '')
                w.writerow([c.get('ruta', ''), c.get('tipo', ''), c.get('codificador', ''),
                            c.get('retardo', ''), c.get('relleno', ''), c.get('id3', ''),
                            c.get('basura', ''), *anclas, *cues])
        print(f'\nDetalle por fichero en: {args.csv}')
    except OSError as e:
        print(f'\n(no se pudo escribir el CSV: {e})')
    print('Pega desde «=== Desfase» hasta aquí.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
