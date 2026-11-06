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

Con `--audio N` mide además lo que decide qué hay que corregir: dónde cae la
rejilla de CADA programa sobre el audio tal como lo leen los decodificadores
de DJ Analyzer (ffmpeg, el del escritorio; afconvert, el Core Audio del Mac y
del iPhone; y soundfile, el que lee el análisis del servidor), con N ficheros
de cada cabecera. Para esto basta un programa.

No escribe nada en ningún sitio. Deja un CSV con el detalle por fichero.

Uso (al menos dos programas, o uno con `--audio`):

    python3 scripts/medir_desfase_mp3.py \\
        --rekordbox ~/rekordbox.xml \\
        --traktor "~/Documents/Native Instruments/Traktor 4.5.1/collection.nml" \\
        --virtualdj "~/Library/Application Support/VirtualDJ/database.xml" \\
        --audio 60

Sin dependencias más allá de la biblioteca estándar; `--audio` necesita numpy
y al menos un decodificador (ffmpeg en el PATH o en `--ffmpeg`, afconvert en
el Mac, o soundfile).
"""

import argparse
import concurrent.futures
import csv
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
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


# --------------------------------------------------------------- sobre el audio
#
# Lo de arriba compara un programa con otro. Lo que decide qué hay que
# corregir es dónde caen la rejilla y los cues de cada programa sobre el audio
# tal como lo leen los decodificadores de DJ Analyzer. Medido con un clic
# sintético (2026-11-06): ffmpeg y mpg123 se saltan la trama Info/Xing y el
# retardo que dice la etiqueta LAME (el clic sale en su muestra); sin cabecera
# salen los dos 25,1 ms tarde, porque no hay de dónde saber el retardo. Un
# programa que no se salta nada pone los de cabecera LAME 51,2 ms más tarde
# (la trama vacía, 1.152 muestras, más el retardo, 1.105) y los demás igual.

SEGUNDOS_DE_AUDIO = 120
PASO = 32                      # muestras por tramo del ataque (0,73 ms a 44,1 kHz)
NITIDEZ_MINIMA = 2.0           # cuánto tiene que destacar el pico del peine
DECODIFICADORES = ('ffmpeg', 'afconvert', 'soundfile')
QUE_ES = {
    'ffmpeg': 'el de DJ Analyzer en el escritorio (la onda; y lo que suena en Windows)',
    'afconvert': 'Core Audio: lo que suena en el Mac y en el iPhone',
    'soundfile': 'mpg123: lo que lee el análisis del servidor',
}


def decodificadores_disponibles(ffmpeg: Optional[str] = None) -> Dict[str, str]:
    """{nombre: binario} de los decodificadores que hay en esta máquina."""
    salida: Dict[str, str] = {}
    # Los mismos sitios en que lo busca la onda de la app en el Mac
    # (`waveform_fft.dart`): Homebrew no siempre está en el PATH.
    for f in (ffmpeg, os.environ.get('FFMPEG'), shutil.which('ffmpeg'),
              '/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg'):
        if f and (os.path.exists(f) or shutil.which(f)):
            salida['ffmpeg'] = f
            break
    a = shutil.which('afconvert')
    if a:
        salida['afconvert'] = a
    try:
        import soundfile  # noqa: F401
        salida['soundfile'] = 'soundfile'
    except Exception:
        pass
    return salida


def _mono(y):
    import numpy as np
    y = np.asarray(y, dtype=np.float32)
    return y.mean(axis=1) if y.ndim == 2 else y


def _leer_wav(ruta: str, segundos: float):
    """Un WAV de PCM de 16 bits (el que deja afconvert), en mono."""
    import numpy as np
    try:
        import soundfile as sf
        info = sf.info(ruta)
        y, _ = sf.read(ruta, frames=int(segundos * info.samplerate), dtype='float32',
                       always_2d=True)
        return _mono(y)
    except ImportError:
        import wave
        with wave.open(ruta, 'rb') as w:
            n = min(w.getnframes(), int(segundos * w.getframerate()))
            datos = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float32) / 32768.0
            return _mono(datos.reshape(-1, w.getnchannels()))


def decodificar(ruta: str, decodificador: str, binario: str, hz: int,
                segundos: float = SEGUNDOS_DE_AUDIO):
    """Los primeros `segundos` del fichero, en mono y a su frecuencia, tal como
    los da ese decodificador; None si no puede."""
    import numpy as np
    try:
        if decodificador == 'ffmpeg':
            r = subprocess.run([binario, '-v', 'error', '-nostdin', '-i', ruta, '-t', str(segundos),
                                '-map', '0:a:0', '-f', 'f32le', '-ac', '1', '-ar', str(hz), '-'],
                               capture_output=True, timeout=180)
            if r.returncode != 0 or not r.stdout:
                return None
            return np.frombuffer(r.stdout, dtype='<f4').copy()
        if decodificador == 'afconvert':
            with tempfile.TemporaryDirectory() as d:
                salida = os.path.join(d, 'a.wav')
                r = subprocess.run([binario, '-f', 'WAVE', '-d', 'LEI16', ruta, salida],
                                   capture_output=True, timeout=300)
                if r.returncode != 0 or not os.path.exists(salida):
                    return None
                return _leer_wav(salida, segundos)
        if decodificador == 'soundfile':
            import soundfile as sf
            y, sr = sf.read(ruta, frames=int(segundos * hz), dtype='float32', always_2d=True)
            return _mono(y) if sr == hz else None
    except Exception:
        return None
    return None


def golpes(y, hz: int, paso: int = PASO):
    """La fuerza del ataque de los GRAVES (el bombo, de 20 a 250 Hz) en tramos
    de `paso` muestras: lo que sube su envolvente de un tramo al siguiente."""
    import numpy as np
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    f = np.fft.rfftfreq(n, 1.0 / hz)
    # Bordes suaves: un corte en seco suena (anillos) y emborrona el ataque.
    peso = np.clip((f - 20.0) / 20.0, 0, 1) * np.clip((250.0 - f) / 70.0, 0, 1)
    # La envolvente es la de la señal ANALÍTICA, no el valor absoluto: con el
    # valor absoluto un bombo de 50 Hz sale con un rizo de 10 ms, el peine
    # tiene un pico en cada rizo y salta de uno a otro con mover el audio
    # siete muestras (4 ms de diferencia en el tema sintético del test).
    espectro = np.zeros(n, dtype=np.complex128)
    espectro[:len(f)] = 2.0 * np.fft.rfft(y) * peso
    env = np.abs(np.fft.ifft(espectro))
    m = (n // paso) * paso
    env = env[:m].reshape(-1, paso).mean(axis=1)
    ataque = np.maximum(np.diff(env, prepend=env[:1]), 0.0)
    return np.convolve(ataque, np.array([1, 2, 3, 2, 1]) / 9.0, mode='same')


def desfase_sobre_el_audio(ataque, paso_s: float, bpm: float, ancla: float
                           ) -> Optional[Tuple[float, float]]:
    """Cuánto cae la rejilla (`bpm`, `ancla`) DESPUÉS de los golpes del audio,
    en segundos y de −beat/2 a +beat/2, y la nitidez del pico (cuánto destaca
    sobre la mediana): el peine de la rejilla se corre sobre los ataques y se
    queda donde más golpes recoge. None si no hay pulso que medir."""
    import numpy as np
    if not bpm or bpm <= 0 or ancla is None or len(ataque) < 10:
        return None
    beat = 60.0 / bpm
    total = len(ataque) * paso_s
    k = np.arange(int(np.ceil(-ancla / beat)), int(np.floor((total - ancla) / beat)) + 1)
    lineas = ancla + k * beat
    if len(lineas) < 16:
        return None
    mitad = int(beat / paso_s / 2)
    deltas = np.arange(-mitad, mitad + 1) * paso_s
    idx = np.round((lineas[None, :] + deltas[:, None]) / paso_s).astype(int)
    dentro = (idx >= 0) & (idx < len(ataque))
    puntos = np.where(dentro, ataque[np.clip(idx, 0, len(ataque) - 1)], 0.0).sum(axis=1)
    i = int(np.argmax(puntos))
    base = float(np.median(puntos))
    if puntos[i] <= 0:
        return None
    afinado = 0.0
    if 0 < i < len(puntos) - 1:
        a, b, c = puntos[i - 1], puntos[i], puntos[i + 1]
        den = a - 2 * b + c
        if den < 0:
            afinado = 0.5 * (a - c) / den
    delta = (i - mitad + afinado) * paso_s
    return -delta, float(puntos[i] / base) if base > 0 else float('inf')


def retraso(a, b, hz: int, maximo: float = 0.1, segundos: float = 30) -> Optional[float]:
    """Cuántos segundos va `b` por detrás de `a` (positivo: más tarde), por
    correlación de los primeros `segundos`, entre −`maximo` y +`maximo`."""
    import numpy as np
    n = min(len(a), len(b), int(segundos * hz))
    if n < hz:
        return None
    x = np.asarray(a[:n], dtype=np.float64)
    y = np.asarray(b[:n], dtype=np.float64)
    tam = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(y, tam) * np.conj(np.fft.rfft(x, tam)), tam)
    m = int(maximo * hz)
    cand = np.concatenate([c[-m:], c[:m + 1]])
    i = int(np.argmax(cand))
    if cand[i] <= 0:
        return None
    return (i - m) / float(hz)


def medir_un_fichero(ruta: str, hz: int, rejillas: Dict[str, Tuple[float, float]],
                     decs: Dict[str, str]) -> Dict:
    """Un fichero: con cada decodificador, dónde cae sobre su audio la rejilla
    de cada programa, y cuánto se separa cada decodificador de ffmpeg. Corre
    en otro proceso."""
    salida: Dict = {'ruta': ruta, 'desfases': {}, 'retrasos': {}}
    audios = {}
    for nombre, binario in decs.items():
        y = decodificar(ruta, nombre, binario, hz)
        if y is not None and len(y) >= 20 * hz:
            audios[nombre] = y
    for nombre, y in audios.items():
        ataque = golpes(y, hz)
        salida['desfases'][nombre] = {
            p: desfase_sobre_el_audio(ataque, PASO / float(hz), bpm, ancla)
            for p, (bpm, ancla) in rejillas.items()}
    if 'ffmpeg' in audios:
        for nombre, y in audios.items():
            if nombre != 'ffmpeg':
                salida['retrasos'][nombre] = retraso(audios['ffmpeg'], y, hz)
    return salida


def _orden_de_tipos(tipos) -> List[str]:
    preferido = ['sin cabecera', 'Info+LAME', 'Xing+LAME', 'Info', 'Xing', 'VBRI']
    return [t for t in preferido if t in tipos] + sorted(t for t in tipos if t not in preferido)


def imprimir_sobre_el_audio(filas: List[Tuple[str, float]], titulo: str) -> Optional[float]:
    """Una tabla por cabecera de (tipo, ms): la mediana de cada una y, respecto a
    la de los ficheros SIN cabecera (en los que todos los programas y todos los
    decodificadores cuentan igual), en qué cajón cae cada fichero. Devuelve la
    mediana sin cabecera."""
    if not filas:
        return None
    por_tipo: Dict[str, List[float]] = defaultdict(list)
    for tipo, ms in filas:
        por_tipo[tipo].append(ms)
    base = statistics.median(por_tipo['sin cabecera']) if por_tipo.get('sin cabecera') else 0.0
    print(f'\n{titulo}: {len(filas)} temas')
    print(f'  {"cabecera":<16} {"temas":>6} {"mediana":>9} {"respecto a sin cabecera":>24}  '
          + ' '.join(f'{c:>5}' for c in CAJONES))
    for tipo in _orden_de_tipos(por_tipo):
        v = por_tipo[tipo]
        c = {k: 0 for k in CAJONES}
        for ms in v:
            c[cajon(ms - base)] += 1
        pcts = ' '.join(f'{100 * c[k] / len(v):4.0f}%' for k in CAJONES)
        med = statistics.median(v)
        print(f'  {tipo:<16} {len(v):>6} {med:+8.1f} ms {med - base:+20.1f} ms  {pcts}')
    return base


def medir_sobre_el_audio(por_fichero: Dict[str, Dict[str, bib.TemaDeUnPrograma]],
                         por_cabecera: int, procesos: int, semilla: int,
                         ffmpeg: Optional[str], ruta_csv: str) -> int:
    decs = decodificadores_disponibles(ffmpeg)
    print('\n=== Desfase: la rejilla de cada programa sobre el audio ===')
    if not decs:
        print('No hay ningún decodificador: instala ffmpeg (o pásalo con --ffmpeg) o soundfile.')
        return 1
    for nombre in DECODIFICADORES:
        print(f'  {nombre:<10} {"sí" if nombre in decs else "no":<3} {QUE_ES[nombre]}')
    grupos: Dict[str, List] = defaultdict(list)
    for temas in por_fichero.values():
        rejillas = {p: (t.bpm, t.ancla) for p, t in temas.items()
                    if t.ancla is not None and t.bpm and t.bpm > 0}
        if not rejillas:
            continue
        ruta = bib.ruta_que_existe(temas.values())
        if not ruta or not ruta.lower().endswith('.mp3'):
            continue
        cab = cabecera_mp3(ruta)
        if not cab.get('hz'):
            continue
        grupos[cab['tipo']].append((ruta, cab, rejillas))
    rnd = random.Random(semilla)
    elegidos = []
    for tipo in sorted(grupos):
        lista = grupos[tipo]
        rnd.shuffle(lista)
        elegidos += lista[:por_cabecera]
    print('MP3 con rejilla en algún programa, por cabecera: '
          + ' · '.join(f'{t} {len(grupos[t])}' for t in _orden_de_tipos(grupos))
          + f' · se miden hasta {por_cabecera} de cada una ({len(elegidos)})')
    if not elegidos:
        return 1
    resultados = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, procesos)) as pool:
        futuros = {pool.submit(medir_un_fichero, ruta, cab['hz'], rejillas, decs): cab
                   for ruta, cab, rejillas in elegidos}
        for i, fut in enumerate(concurrent.futures.as_completed(futuros), 1):
            try:
                resultados.append((futuros[fut], fut.result()))
            except Exception as e:  # un fichero raro no tumba la medida
                print(f'  (un fichero falló: {type(e).__name__})')
            if i % 20 == 0 or i == len(futuros):
                print(f'[Desfase] {i} de {len(futuros)} ficheros', flush=True)

    sin_pulso = defaultdict(int)
    filas: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    retrasos: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for cab, r in resultados:
        for dec, por_programa in r['desfases'].items():
            for programa, medida in por_programa.items():
                if medida is None or medida[1] < NITIDEZ_MINIMA:
                    sin_pulso[programa] += 1
                    continue
                filas[(programa, dec)].append((cab['tipo'], medida[0] * 1000))
        for dec, s in r['retrasos'].items():
            if s is not None:
                retrasos[dec].append((cab['tipo'], s * 1000))
    for programa in bib.PROGRAMAS:
        for dec in DECODIFICADORES:
            imprimir_sobre_el_audio(filas.get((programa, dec), []),
                                    f'{programa} sobre el audio de {dec}')
    for dec in DECODIFICADORES:
        if retrasos.get(dec):
            imprimir_sobre_el_audio(retrasos[dec], f'{dec} respecto a ffmpeg (correlación)')
    if sin_pulso:
        print('\nSin pulso que medir (descartados): '
              + ' · '.join(f'{p} {n}' for p, n in sorted(sin_pulso.items())))
    print('\nmediana: cuánto cae la línea del programa DESPUÉS del golpe que se oye con ese '
          'decodificador (lleva el sesgo del detector, igual en todos los ficheros).')
    print('«respecto a sin cabecera» es lo que cuenta: en esos ficheros todos los programas '
          'y decodificadores dicen lo mismo. +51 ms = ese programa cuenta la trama vacía y '
          'el retardo del codificador, y el decodificador no.')

    try:
        with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['ruta', 'cabecera', 'codificador', 'retardo', 'programa',
                        'decodificador', 'desfase (ms)', 'nitidez', 'respecto a ffmpeg (ms)'])
            for cab, r in resultados:
                for dec, por_programa in r['desfases'].items():
                    s = r['retrasos'].get(dec)
                    for programa, medida in por_programa.items():
                        w.writerow([r['ruta'], cab['tipo'], cab.get('codificador', ''),
                                    cab.get('retardo', ''), programa, dec,
                                    f'{medida[0] * 1000:.1f}' if medida else '',
                                    f'{medida[1]:.1f}' if medida else '',
                                    f'{s * 1000:.2f}' if s is not None else ''])
        print(f'\nDetalle por fichero en: {ruta_csv}')
    except OSError as e:
        print(f'\n(no se pudo escribir el CSV: {e})')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--rekordbox')
    ap.add_argument('--traktor')
    ap.add_argument('--virtualdj')
    ap.add_argument('--csv', default=os.path.expanduser('~/medir_desfase_mp3.csv'))
    ap.add_argument('--audio', type=int, default=0,
                    help='mide también la rejilla sobre el audio, con N ficheros de cada cabecera')
    ap.add_argument('--ffmpeg', help='el ffmpeg que usar (por defecto, el del PATH)')
    ap.add_argument('--procesos', type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument('--semilla', type=int, default=7)
    ap.add_argument('--csv-audio', default=os.path.expanduser('~/medir_desfase_audio.csv'))
    args = ap.parse_args(argv)
    programas = sum(1 for x in (args.rekordbox, args.traktor, args.virtualdj) if x)
    if programas < (1 if args.audio else 2):
        ap.error('hacen falta al menos dos programas para comparar (o uno con --audio)')

    leidas: Dict[str, int] = {}
    por_fichero = bib.leer_bibliotecas(args.rekordbox, args.traktor, args.virtualdj,
                                       leidas=leidas)
    if len(leidas) < 2:
        if args.audio and leidas:
            salida = medir_sobre_el_audio(por_fichero, args.audio, args.procesos,
                                          args.semilla, args.ffmpeg, args.csv_audio)
            print('Pega desde «=== Desfase» hasta aquí.')
            return salida
        print('Hacen falta dos bibliotecas que se puedan leer para comparar '
              f'(leídas: {", ".join(leidas) or "ninguna"}).')
        return 1
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
    if args.audio:
        medir_sobre_el_audio(por_fichero, args.audio, args.procesos, args.semilla,
                             args.ffmpeg, args.csv_audio)
    print('Pega desde «=== Desfase» hasta aquí.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
