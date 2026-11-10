#!/usr/bin/env python3
"""La PISTA DE PRUEBA: dónde pone cada programa de DJ el golpe de un MISMO tema
según la cabecera del MP3, sin el sesgo de ningún detector.

Por qué (2026-11-06): `medir_desfase_mp3.py --audio` midió en el disco del
owner que Rekordbox pone la rejilla de los MP3 con etiqueta LAME ~55 ms más
tarde que la de los MP3 sin cabecera, y que los decodificadores de DJ Analyzer
no se ponen de acuerdo entre ellos (Core Audio va 12 ms por delante de ffmpeg
en los MP3 sin cabecera). Lo que no se puede separar con música de verdad es
el sesgo del detector (cuánto antes o después del ataque pone cada uno la
línea), y de eso depende si a Rekordbox hay que corregirle 51 o 39 ms.

Aquí el tema es sintético (un bombo seco a 125 BPM) y va en cinco ficheros: el
WAV, que lee igual todo el mundo, y cuatro MP3 de las mismas tramas con
distinta cabecera. Lo que cada programa ponga en el WAV es su sesgo; lo que
ponga de más en cada MP3 es lo que cuenta de más.

    1. python3 scripts/pista_de_prueba.py --crear ~/Music/dja_prueba \\
           --rekordbox ~/Documents/rekordbox/rekordbox.xml
    2. Meter la carpeta en Rekordbox, Traktor y VirtualDJ y dejar que analicen
       la rejilla. Exportar la colección de Rekordbox en xml y CERRAR Traktor y
       VirtualDJ (escriben su base de datos al salir).
    3. python3 scripts/pista_de_prueba.py --leer ~/Music/dja_prueba \\
           --rekordbox ~/Documents/rekordbox/rekordbox.xml \\
           --traktor "~/Documents/Native Instruments/Traktor 4.5.1/collection.nml" \\
           --virtualdj "~/Library/Application Support/VirtualDJ/database.xml"

Segunda vuelta (2026-11-10). La primera lectura dio que el Rekordbox de hoy
pone el golpe de los cuatro MP3 en el MISMO sitio (no cuenta la trama Info),
y las rejillas que el owner ya tenía en Rekordbox dicen lo contrario (la
cuentan: +51 ms en los MP3 con etiqueta LAME). Para saber cuál de las dos cosas
es verdad con ficheros de verdad, `--crear` con una biblioteca delante añade:

- los mismos tres MP3 hechos por LAME DE VERDAD (si `lame` está instalado),
  por si la etiqueta que escribe ffmpeg no es lo bastante auténtica, y
- COPIAS de unos pocos MP3 de la biblioteca (`--reales N` de cada cabecera),
  con la rejilla que cada programa les tenía apuntada en un JSON. Al
  analizarlas hoy, cada programa las trata como temas nuevos: lo que su
  rejilla se mueva respecto a la que tenía es lo que ha cambiado su forma de
  leer el MP3, sin detector ni audio de por medio.

Necesita numpy y ffmpeg con libmp3lame (el de Homebrew lo trae); para leer,
también afconvert (el Core Audio del Mac) y soundfile, si están.
"""

import argparse
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import wave
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import bibliotecas_dj as bib  # noqa: E402
import medir_desfase_mp3 as md  # noqa: E402

HZ = 44100
BPM = 125.0
BEAT = 60.0 / BPM
T0 = 0.600                    # el primer golpe, en el WAV
SEGUNDOS = 40.0
PREFIJO = 'dja_prueba_'
VARIANTES = {                 # fichero → cabecera que tiene que salir
    'wav': None,
    'sin_cabecera': 'sin cabecera',
    'info_lame': 'Info+LAME',
    'xing_lame': 'Xing+LAME',
    'xing': 'Xing',
}
ORDEN = ['wav', 'sin_cabecera', 'info_lame', 'xing_lame', 'xing']
# Los mismos, codificados por LAME de verdad (si está instalado).
DE_LAME = {
    'lame_sin': 'sin cabecera',
    'lame_cbr': 'Info+LAME',
    'lame_vbr': 'Xing+LAME',
}
ETIQUETA = {'wav': 'WAV', 'sin_cabecera': 'sin cabecera', 'info_lame': 'Info+LAME',
            'xing_lame': 'Xing+LAME', 'xing': 'Xing', 'lame_sin': 'sin cab. · LAME',
            'lame_cbr': 'Info · LAME', 'lame_vbr': 'Xing · LAME'}
REALES_JSON = PREFIJO + 'reales.json'
TIPOS_REALES = ('sin cabecera', 'Info+LAME', 'Xing+LAME', 'Xing')


def nombre(variante: str) -> str:
    return PREFIJO + variante + ('.wav' if variante == 'wav' else '.mp3')


def el_tema(segundos: float = SEGUNDOS, hz: int = HZ):
    """Un bombo seco (un barrido de 130 a 50 Hz que se apaga) en cada beat desde
    `T0`, y un charles en cada contratiempo: sin nada más que lo tape."""
    import numpy as np
    rng = np.random.RandomState(11)
    n = int(segundos * hz)
    y = np.zeros(n)
    k = np.arange(int(0.12 * hz)) / hz
    bombo = np.sin(2 * np.pi * (50 + 80 * np.exp(-k * 40)) * k) * np.exp(-k * 18)
    largo = int(0.03 * hz)
    hat = rng.randn(largo) * np.exp(-np.arange(largo) / hz * 120) * 0.15
    j = 0
    while T0 + (j + 1) * BEAT < segundos - 0.3:
        a = int(round((T0 + j * BEAT) * hz))
        y[a:a + len(bombo)] += 0.8 * bombo
        h = int(round((T0 + (j + 0.5) * BEAT) * hz))
        y[h:h + largo] += hat
        j += 1
    return (y / np.abs(y).max() * 0.9).astype(np.float32)


def _escribir_wav(ruta: str, y, hz: int = HZ) -> None:
    import numpy as np
    datos = (np.clip(y, -1, 1) * 32767).astype('<i2')
    with wave.open(ruta, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(hz)
        w.writeframes(np.repeat(datos[:, None], 2, axis=1).tobytes())


def crc_de_la_etiqueta(trama: bytes, pos_crc: int = 190) -> int:
    """El CRC-16 (ARC) con que LAME (y ffmpeg) firman la etiqueta: el de los 190
    primeros bytes de la trama, con el sitio del CRC a cero. En estéreo el CRC
    cae justo en el 190 y queda fuera; en mono va antes (la información de
    lado es más corta) y cuenta como ceros (medido con lo que escribe ffmpeg)."""
    datos = bytearray(trama[:190])
    if pos_crc < 190:
        datos[pos_crc:pos_crc + 2] = b'\0\0'
    crc = 0
    for b in datos:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def con_etiqueta_de_lame(ruta: str) -> None:
    """ffmpeg firma la etiqueta como «Lavc…»; un MP3 de verdad dice «LAME…». Se
    pone «LAME3.100» y se rehace el CRC, para que cada programa la lea como la
    de un fichero de tienda."""
    c = md.cabecera_mp3(ruta)
    if c.get('pos_codificador') is None:
        return
    pos_crc = c['pos_codificador'] - c['trama'] + 34     # los 2 últimos de los 36
    with open(ruta, 'r+b') as f:
        f.seek(c['pos_codificador'])
        f.write(b'LAME3.100')
        f.seek(c['trama'])
        trama = f.read(192)
        f.seek(c['trama'] + pos_crc)
        f.write(crc_de_la_etiqueta(trama, pos_crc).to_bytes(2, 'big'))


def sin_etiqueta_de_lame(origen: str, destino: str) -> None:
    """Un MP3 con trama Xing y SIN etiqueta LAME (como los de los codificadores
    de antes): la misma trama, con los 36 bytes de la etiqueta a cero."""
    shutil.copyfile(origen, destino)
    c = md.cabecera_mp3(destino)
    if c.get('pos_codificador') is None:
        return
    with open(destino, 'r+b') as f:
        f.seek(c['pos_codificador'])
        f.write(bytes(36))


def _binario(nombre_: str, dado: Optional[str] = None) -> Optional[str]:
    """Un programa de la línea de comandos: el que se pasa, el del PATH o el de
    Homebrew (que no siempre está en el PATH)."""
    for f in (dado, shutil.which(nombre_), f'/opt/homebrew/bin/{nombre_}', f'/usr/local/bin/{nombre_}'):
        if f and (os.path.exists(f) or shutil.which(f)):
            return f
    return None


def _ya_esta(ruta: str, tipo: str) -> bool:
    """Un fichero de la pista que ya está y es lo que tiene que ser no se vuelve
    a escribir: los programas ya lo tienen analizado, y tocarlo es invitarlos a
    dudar."""
    return os.path.exists(ruta) and md.cabecera_mp3(ruta)['tipo'] == tipo


def crear_los_de_lame(carpeta: str, lame: str) -> List[str]:
    """Los tres MP3 de la pista hechos por LAME de verdad (trama Info del tamaño
    de las demás, su etiqueta y su CRC, sin nada rehecho a mano)."""
    wav = os.path.join(carpeta, nombre('wav'))
    hechos = []
    for variante, opciones in (('lame_sin', ['-b', '320', '-t']),
                               ('lame_cbr', ['-b', '320']),
                               ('lame_vbr', ['-V', '2'])):
        ruta = os.path.join(carpeta, nombre(variante))
        if not _ya_esta(ruta, DE_LAME[variante]):
            r = subprocess.run([lame, '--quiet', '--noreplaygain', *opciones, wav, ruta],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f'  (lame no pudo hacer {nombre(variante)}: {r.stderr.strip()[:120]})')
                continue
        hechos.append(variante)
    return hechos


def elegir_reales(por_fichero: Dict[str, Dict[str, bib.TemaDeUnPrograma]], carpeta: str,
                  por_cabecera: int, semilla: int = 7) -> List[Dict]:
    """Los MP3 de la biblioteca que se copian a la pista: `por_cabecera` de cada
    tipo, con rejilla en algún programa, primero los que la tienen en más
    programas (los pocos de Traktor entran así) y luego al azar."""
    dentro = bib.clave_de_ruta(os.path.join(carpeta, ''))
    grupos: Dict[str, List[Dict]] = defaultdict(list)
    for temas in por_fichero.values():
        rejillas = {p: {'ancla': t.ancla, 'bpm': t.bpm} for p, t in temas.items()
                    if t.ancla is not None and t.bpm and t.bpm > 0}
        if not rejillas:
            continue
        ruta = bib.ruta_que_existe(temas.values())
        if not ruta or not ruta.lower().endswith('.mp3') or bib.clave_de_ruta(ruta).startswith(dentro):
            continue
        cab = md.cabecera_mp3(ruta)
        if cab['tipo'] not in TIPOS_REALES:
            continue
        grupos[cab['tipo']].append({'original': ruta, 'cabecera': cab['tipo'],
                                    'codificador': cab.get('codificador', ''),
                                    'retardo': cab.get('retardo'), 'programas': rejillas})
    rnd = random.Random(semilla)
    elegidos: List[Dict] = []
    for tipo in TIPOS_REALES:
        lista = sorted(grupos.get(tipo, []), key=lambda d: d['original'])
        rnd.shuffle(lista)
        lista.sort(key=lambda d: -len(d['programas']))      # estable: el azar dentro de cada grupo
        elegidos += lista[:por_cabecera]
    return elegidos


def copiar_reales(carpeta: str, por_fichero, por_cabecera: int) -> List[Dict]:
    """Copia los elegidos a la pista con nombre nuevo y apunta en un JSON la
    rejilla que cada programa les tenía. Si el JSON ya está, son esos y no se
    eligen otros (los programas ya los tienen analizados)."""
    ruta_json = os.path.join(carpeta, REALES_JSON)
    if os.path.exists(ruta_json):
        with open(ruta_json, encoding='utf-8') as f:
            reales = json.load(f)
    else:
        reales = elegir_reales(por_fichero, carpeta, por_cabecera)
        for i, r in enumerate(reales, 1):
            r['copia'] = f'{PREFIJO}real_{i:02d}_{r["cabecera"].replace("+", "_").replace(" ", "_").lower()}.mp3'
    listos = []
    for r in reales:
        destino = os.path.join(carpeta, r['copia'])
        if not os.path.exists(destino):
            try:
                shutil.copyfile(r['original'], destino)
            except OSError as e:
                print(f'  (no se pudo copiar {r["original"]}: {e})')
                continue
        listos.append(r)
    if listos:
        with open(ruta_json, 'w', encoding='utf-8') as f:
            json.dump(listos, f, ensure_ascii=False, indent=1)
    return listos


def crear(carpeta: str, ffmpeg: Optional[str] = None, lame: Optional[str] = None,
          rekordbox=None, traktor=None, virtualdj=None, reales: int = 5) -> int:
    carpeta = os.path.expanduser(carpeta)
    ffmpeg = _binario('ffmpeg', ffmpeg)
    if not ffmpeg:
        print('Hace falta ffmpeg (brew install ffmpeg).')
        return 1
    os.makedirs(carpeta, exist_ok=True)
    wav = os.path.join(carpeta, nombre('wav'))
    if not os.path.exists(wav):
        _escribir_wav(wav, el_tema())

    def mp3(variante: str, *extra: str) -> Optional[str]:
        """La ruta si hay que rematarlo (recién hecho), None si ya estaba."""
        ruta = os.path.join(carpeta, nombre(variante))
        if _ya_esta(ruta, VARIANTES[variante]):
            return None
        r = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', wav,
                            '-c:a', 'libmp3lame', *extra,
                            '-metadata', 'artist=DJ Analyzer',
                            '-metadata', f'title=Pista de prueba · {VARIANTES[variante]}', ruta],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or 'ffmpeg falló')
        return ruta

    try:
        mp3('sin_cabecera', '-b:a', '320k', '-write_xing', '0')
        for variante, extra in (('info_lame', ('-b:a', '320k')), ('xing_lame', ('-q:a', '2'))):
            hecho = mp3(variante, *extra)
            if hecho:
                con_etiqueta_de_lame(hecho)
    except RuntimeError as e:
        print(f'ffmpeg no pudo hacer los MP3: {e}')
        return 1
    xing = os.path.join(carpeta, nombre('xing'))
    if not _ya_esta(xing, VARIANTES['xing']):
        sin_etiqueta_de_lame(os.path.join(carpeta, nombre('xing_lame')), xing)
    print(f'Pista de prueba en {carpeta}:')
    bien = True
    for v in ORDEN:
        ruta = os.path.join(carpeta, nombre(v))
        tipo = 'WAV' if v == 'wav' else md.cabecera_mp3(ruta)['tipo']
        esperado = VARIANTES[v] or 'WAV'
        bien &= tipo == esperado
        print(f'  {nombre(v):<36} {tipo}' + ('' if tipo == esperado else f'  (¡se esperaba {esperado}!)'))

    lame = _binario('lame', lame)
    if lame:
        for v in crear_los_de_lame(carpeta, lame):
            c = md.cabecera_mp3(os.path.join(carpeta, nombre(v)))
            bien &= c['tipo'] == DE_LAME[v]
            print(f'  {nombre(v):<36} {c["tipo"]} · {c.get("codificador") or "sin etiqueta"}'
                  + ('' if c['tipo'] == DE_LAME[v] else f'  (¡se esperaba {DE_LAME[v]}!)'))
    else:
        print('  (sin `lame`: con `brew install lame` la pista lleva también MP3 hechos por LAME de verdad)')

    if rekordbox or traktor or virtualdj:
        por_fichero = bib.leer_bibliotecas(rekordbox, traktor, virtualdj)
        copias = copiar_reales(carpeta, por_fichero, reales)
        print(f'  y {len(copias)} copias de MP3 de tu biblioteca, con la rejilla que les tenía cada programa:')
        for r in copias:
            quien = ' · '.join(f'{p} {d["ancla"]:.3f}' for p, d in sorted(r['programas'].items()))
            print(f'  {r["copia"]:<36} {r["cabecera"]} · {r.get("codificador") or "sin etiqueta"} · {quien}')
    print('\nAhora: métela en Rekordbox, Traktor y VirtualDJ, deja que analicen la rejilla, '
          'exporta la colección de Rekordbox en xml y cierra Traktor y VirtualDJ.')
    return 0 if bien else 1


# --------------------------------------------------------------- la lectura

def _presentes(carpeta: str) -> List[str]:
    """Las variantes de la pista que están en la carpeta, en su orden."""
    return ORDEN + [v for v in DE_LAME if os.path.exists(os.path.join(carpeta, nombre(v)))]


def donde_cae(anclas: Dict[str, Optional[float]], beat: float = BEAT,
              variantes: Optional[List[str]] = None) -> Dict[str, Optional[float]]:
    """Por variante, cuánto más tarde que en el WAV pone el golpe un programa
    (ms, de −beat/2 a +beat/2): el sesgo de su detector se va con la resta.
    None si no tiene rejilla en esa variante o en el WAV."""
    ref = anclas.get('wav')
    salida: Dict[str, Optional[float]] = {}
    for v in (variantes or ORDEN):
        a = anclas.get(v)
        if v == 'wav':
            salida[v] = 0.0 if ref is not None else None
        elif a is None or ref is None:
            salida[v] = None
        else:
            salida[v] = bib.diferencia_de_fase(a, ref, beat) * 1000
    return salida


def anclas_de_los_programas(por_fichero: Dict[str, Dict[str, bib.TemaDeUnPrograma]]
                            ) -> Dict[str, Dict[str, Optional[float]]]:
    """{programa: {variante: ancla en s}} de los ficheros de la prueba."""
    salida: Dict[str, Dict[str, Optional[float]]] = {}
    for temas in por_fichero.values():
        for programa, t in temas.items():
            base = os.path.basename(t.ruta.replace('\\', '/'))
            if not base.lower().startswith(PREFIJO):
                continue
            variante = os.path.splitext(base)[0][len(PREFIJO):].lower()
            if variante in VARIANTES or variante in DE_LAME:
                salida.setdefault(programa, {})[variante] = t.ancla
    return salida


def donde_cae_en_la_app(carpeta: str, ffmpeg: Optional[str] = None
                        ) -> Dict[str, Dict[str, Optional[float]]]:
    """{decodificador: {variante: ms}}: cuánto más tarde que en el WAV suena el
    golpe con cada decodificador de DJ Analyzer, por correlación (exacto)."""
    decs = md.decodificadores_disponibles(ffmpeg)
    wav_ruta = os.path.join(carpeta, nombre('wav'))
    salida: Dict[str, Dict[str, Optional[float]]] = {}
    for dec, binario in decs.items():
        ref = md.decodificar(wav_ruta, dec, binario, HZ, segundos=20)
        fila: Dict[str, Optional[float]] = {'wav': 0.0 if ref is not None else None}
        for v in _presentes(carpeta)[1:]:
            y = md.decodificar(os.path.join(carpeta, nombre(v)), dec, binario, HZ, segundos=20)
            s = md.retraso(ref, y, HZ) if ref is not None and y is not None else None
            fila[v] = s * 1000 if s is not None else None
        salida[dec] = fila
    return salida


def version_de(programa: str, ruta: Optional[str]) -> Optional[str]:
    """La versión del programa que escribió esa biblioteca, si la dice."""
    if not ruta:
        return None
    try:
        with open(os.path.expanduser(ruta), 'rb') as f:
            cabeza = f.read(4096).decode('utf-8', 'replace')
    except OSError:
        return None
    if programa == 'rekordbox':
        m = re.search(r'<PRODUCT\b[^>]*\bVersion="([^"]*)"', cabeza)
        return m.group(1) if m else None
    if programa == 'traktor':
        m = re.search(r'\bPROGRAM="([^"]*)"', cabeza)
        n = re.search(r'<NML\b[^>]*\bVERSION="([^"]*)"', cabeza)
        partes = [m.group(1) if m else None, f'NML {n.group(1)}' if n else None]
        return ' · '.join(x for x in partes if x) or None
    m = re.search(r'<VirtualDJ_Database\b[^>]*\bVersion="([^"]*)"', cabeza)
    return m.group(1) if m else None


def _beat_de(bpm_a: Optional[float], bpm_b: Optional[float]) -> Optional[float]:
    """El beat con el que comparar dos rejillas del mismo tema (el mismo BPM, o
    uno el doble del otro, a menos de un 0,5 %)."""
    if not bpm_a or not bpm_b or bpm_a <= 0 or bpm_b <= 0:
        return None
    alto, bajo = max(bpm_a, bpm_b), min(bpm_a, bpm_b)
    for k in (1.0, 2.0):
        if abs(alto / (bajo * k) - 1) <= 0.005:
            return 60.0 / alto
    return None


def lo_que_se_mueven(reales: List[Dict], por_fichero, carpeta: str
                     ) -> Dict[str, List[Tuple[str, str, float]]]:
    """{programa: [(copia, cabecera, ms)]}: la rejilla que el programa le pone
    HOY a cada copia menos la que le tenía al original (módulo el beat).
    Negativo: hoy la pone antes."""
    salida: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    for r in reales:
        hoy = por_fichero.get(bib.clave_de_ruta(os.path.join(carpeta, r['copia'])), {})
        for programa, antes in r.get('programas', {}).items():
            t = hoy.get(programa)
            if t is None or t.ancla is None:
                continue
            beat = _beat_de(antes.get('bpm'), t.bpm)
            if beat is None or antes.get('ancla') is None:
                continue
            ms = bib.diferencia_de_fase(t.ancla, antes['ancla'], beat) * 1000
            salida[programa].append((r['copia'], r['cabecera'], ms))
    return salida


def _imprimir_movidos(programa: str, filas: List[Tuple[str, str, float]]) -> None:
    por_tipo: Dict[str, List[float]] = defaultdict(list)
    for _, tipo, ms in filas:
        por_tipo[tipo].append(ms)
    print(f'\n  {programa} · hoy − lo que tenía: {len(filas)} copias')
    print(f'    {"cabecera":<14} {"copias":>6} ' + ' '.join(f'{c:>5}' for c in md.CAJONES)
          + f' {"mediana":>10}   una a una')
    for tipo in TIPOS_REALES:
        v = por_tipo.get(tipo)
        if not v:
            continue
        c = {k: 0 for k in md.CAJONES}
        for ms in v:
            c[md.cajon(ms)] += 1
        cajones = ' '.join(f'{c[k]:>5}' for k in md.CAJONES)
        una_a_una = ' '.join(f'{ms:+.0f}' for ms in sorted(v))
        print(f'    {tipo:<14} {len(v):>6} {cajones} {statistics.median(v):+8.1f} ms   {una_a_una}')


def _ms(x: Optional[float]) -> str:
    return f'{x:+7.1f}' if x is not None else '      -'


def leer(carpeta: str, rekordbox=None, traktor=None, virtualdj=None, ffmpeg=None) -> int:
    carpeta = os.path.expanduser(carpeta)
    print('\n=== Pista de prueba: dónde suena el golpe con cada uno ===')
    pedidos = {'rekordbox': rekordbox, 'traktor': traktor, 'virtualdj': virtualdj}
    versiones = [f'{p} {v}' for p in bib.PROGRAMAS for v in [version_de(p, pedidos[p])] if v]
    if versiones:
        print('Versiones: ' + ' · '.join(versiones))
    variantes = _presentes(carpeta)
    app = donde_cae_en_la_app(carpeta, ffmpeg) if os.path.isdir(carpeta) else {}
    por_fichero = bib.leer_bibliotecas(rekordbox, traktor, virtualdj)
    anclas = anclas_de_los_programas(por_fichero)
    programas = {p: donde_cae(a, variantes=variantes) for p, a in anclas.items()}
    columnas = [(d, app[d]) for d in md.DECODIFICADORES if d in app] + \
               [(p, programas[p]) for p in bib.PROGRAMAS if p in programas]
    ruta_json = os.path.join(carpeta, REALES_JSON)
    if not columnas and not os.path.exists(ruta_json):
        print('No hay nada que leer: ¿está la carpeta y han analizado los programas?')
        return 1
    if columnas:
        print('ms respecto al WAV (positivo: el golpe cae más tarde que en el WAV)\n')
        print(f'  {"fichero":<16}' + ''.join(f'{n:>11}' for n, _ in columnas))
        for v in variantes:
            print(f'  {ETIQUETA[v]:<16}' + ''.join(f'{_ms(c.get(v)):>11}' for _, c in columnas))
    en_el_wav = [f'{p} {bib.diferencia_de_fase(anclas[p]["wav"], T0, BEAT) * 1000:+.1f} ms'
                 for p in bib.PROGRAMAS if anclas.get(p, {}).get('wav') is not None]
    if en_el_wav:
        print('\nEn el WAV, la línea de cada programa respecto al golpe (su sesgo): '
              + ' · '.join(en_el_wav))
    faltan = [f'{p}: {", ".join(ETIQUETA[v] for v in variantes if programas[p].get(v) is None)}'
              for p in programas if any(programas[p].get(v) is None for v in variantes)]
    faltan += [f'{p}: ninguno' for p in bib.PROGRAMAS if pedidos[p] and p not in programas]
    if faltan:
        print('\nSin rejilla (¿no los analizó?): ' + ' · '.join(faltan))
    if app and programas:
        print('\nLo que hay que restar a un cue de cada programa para que suene en su sitio '
              'en DJ Analyzer (programa − decodificador):')
        decs = [d for d in md.DECODIFICADORES if d in app]
        for p in bib.PROGRAMAS:
            if p not in programas:
                continue
            print(f'  {p}')
            for v in variantes[1:]:
                pv = programas[p].get(v)
                partes = [f'{d} {_ms(pv - app[d][v]) if pv is not None and app[d].get(v) is not None else "      -"}'
                          for d in decs]
                print(f'    {ETIQUETA[v]:<16}' + '   '.join(partes))

    if os.path.exists(ruta_json):
        with open(ruta_json, encoding='utf-8') as f:
            reales = json.load(f)
        print(f'\n=== Las {len(reales)} copias de tu biblioteca: la rejilla de hoy frente a la que tenías ===')
        movidos = lo_que_se_mueven(reales, por_fichero, carpeta)
        if not movidos:
            print('Ningún programa les ha puesto rejilla todavía (¿las analizó? ¿exportaste '
                  'el xml y cerraste Traktor y VirtualDJ?).')
        for p in bib.PROGRAMAS:
            if movidos.get(p):
                _imprimir_movidos(p, movidos[p])
        if movidos:
            print('\n  negativo: hoy el programa pone la rejilla ANTES de donde la tenía. '
                  '≈0: menos de 6 ms · ≈26: de 18 a 34 · ≈50: de 40 a 60')
        # Y entre programas, con lo analizado hoy.
        claves = {bib.clave_de_ruta(os.path.join(carpeta, r['copia'])): r for r in reales}
        hoy = {k: v for k, v in por_fichero.items() if k in claves and len(v) >= 2}
        if hoy:
            cab = {k: {'tipo': claves[k]['cabecera']} for k in hoy}
            print('\n  Y entre programas, las copias analizadas hoy:')
            for par, datos in sorted(md.comparar(hoy, cab).items()):
                md.imprimir(f'  {par} · ancla de la rejilla (módulo el beat)', datos['anclas'], 'copias')
    print('\nPega desde «=== Pista de prueba» hasta aquí.')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--crear', metavar='CARPETA')
    ap.add_argument('--leer', metavar='CARPETA')
    ap.add_argument('--rekordbox')
    ap.add_argument('--traktor')
    ap.add_argument('--virtualdj')
    ap.add_argument('--ffmpeg')
    ap.add_argument('--lame')
    ap.add_argument('--reales', type=int, default=5,
                    help='con una biblioteca en --crear: cuántos MP3 suyos de cada cabecera se copian')
    args = ap.parse_args(argv)
    if args.crear:
        return crear(args.crear, args.ffmpeg, args.lame, args.rekordbox, args.traktor,
                     args.virtualdj, args.reales)
    if args.leer:
        return leer(args.leer, args.rekordbox, args.traktor, args.virtualdj, args.ffmpeg)
    ap.error('--crear CARPETA o --leer CARPETA')
    return 2


if __name__ == '__main__':
    sys.exit(main())
