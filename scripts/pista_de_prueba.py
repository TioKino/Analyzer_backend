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

    1. python3 scripts/pista_de_prueba.py --crear ~/Music/dja_prueba
    2. Meter la carpeta en Rekordbox, Traktor y VirtualDJ y dejar que analicen
       la rejilla. Exportar la colección de Rekordbox en xml y CERRAR Traktor y
       VirtualDJ (escriben su base de datos al salir).
    3. python3 scripts/pista_de_prueba.py --leer ~/Music/dja_prueba \\
           --rekordbox ~/Documents/rekordbox/rekordbox.xml \\
           --traktor "~/Documents/Native Instruments/Traktor 4.5.1/collection.nml" \\
           --virtualdj "~/Library/Application Support/VirtualDJ/database.xml"

Necesita numpy y ffmpeg con libmp3lame (el de Homebrew lo trae); para leer,
también afconvert (el Core Audio del Mac) y soundfile, si están.
"""

import argparse
import os
import shutil
import subprocess
import sys
import wave
from typing import Dict, List, Optional

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


def crear(carpeta: str, ffmpeg: Optional[str] = None) -> int:
    carpeta = os.path.expanduser(carpeta)
    ffmpeg = ffmpeg or shutil.which('ffmpeg') or next(
        (f for f in ('/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg') if os.path.exists(f)),
        None)
    if not ffmpeg:
        print('Hace falta ffmpeg (brew install ffmpeg).')
        return 1
    os.makedirs(carpeta, exist_ok=True)
    wav = os.path.join(carpeta, nombre('wav'))
    _escribir_wav(wav, el_tema())

    def mp3(variante: str, *extra: str) -> str:
        ruta = os.path.join(carpeta, nombre(variante))
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
        con_etiqueta_de_lame(mp3('info_lame', '-b:a', '320k'))
        con_etiqueta_de_lame(mp3('xing_lame', '-q:a', '2'))
    except RuntimeError as e:
        print(f'ffmpeg no pudo hacer los MP3: {e}')
        return 1
    sin_etiqueta_de_lame(os.path.join(carpeta, nombre('xing_lame')),
                         os.path.join(carpeta, nombre('xing')))
    print(f'Pista de prueba en {carpeta}:')
    bien = True
    for v in ORDEN:
        ruta = os.path.join(carpeta, nombre(v))
        tipo = 'WAV' if v == 'wav' else md.cabecera_mp3(ruta)['tipo']
        esperado = VARIANTES[v] or 'WAV'
        bien &= tipo == esperado
        print(f'  {nombre(v):<28} {tipo}' + ('' if tipo == esperado else f'  (¡se esperaba {esperado}!)'))
    print('\nAhora: métela en Rekordbox, Traktor y VirtualDJ, deja que analicen la rejilla, '
          'exporta la colección de Rekordbox en xml y cierra Traktor y VirtualDJ.')
    return 0 if bien else 1


# --------------------------------------------------------------- la lectura

def donde_cae(anclas: Dict[str, Optional[float]], beat: float = BEAT) -> Dict[str, Optional[float]]:
    """Por variante, cuánto más tarde que en el WAV pone el golpe un programa
    (ms, de −beat/2 a +beat/2): el sesgo de su detector se va con la resta.
    None si no tiene rejilla en esa variante o en el WAV."""
    ref = anclas.get('wav')
    salida: Dict[str, Optional[float]] = {}
    for v in ORDEN:
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
            if variante in VARIANTES:
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
        for v in ORDEN[1:]:
            y = md.decodificar(os.path.join(carpeta, nombre(v)), dec, binario, HZ, segundos=20)
            s = md.retraso(ref, y, HZ) if ref is not None and y is not None else None
            fila[v] = s * 1000 if s is not None else None
        salida[dec] = fila
    return salida


def _ms(x: Optional[float]) -> str:
    return f'{x:+7.1f}' if x is not None else '      -'


def leer(carpeta: str, rekordbox=None, traktor=None, virtualdj=None, ffmpeg=None) -> int:
    carpeta = os.path.expanduser(carpeta)
    print('\n=== Pista de prueba: dónde suena el golpe con cada uno ===')
    app = donde_cae_en_la_app(carpeta, ffmpeg) if os.path.isdir(carpeta) else {}
    por_fichero = bib.leer_bibliotecas(rekordbox, traktor, virtualdj)
    programas = {p: donde_cae(a) for p, a in anclas_de_los_programas(por_fichero).items()}
    columnas = [(d, app[d]) for d in md.DECODIFICADORES if d in app] + \
               [(p, programas[p]) for p in bib.PROGRAMAS if p in programas]
    if not columnas:
        print('No hay nada que leer: ¿está la carpeta y han analizado los programas?')
        return 1
    print('ms respecto al WAV (positivo: el golpe cae más tarde que en el WAV)\n')
    print(f'  {"fichero":<14}' + ''.join(f'{n:>11}' for n, _ in columnas))
    for v in ORDEN:
        print(f'  {VARIANTES[v] or "WAV":<14}' + ''.join(f'{_ms(c.get(v)):>11}' for _, c in columnas))
    pedidos = {'rekordbox': rekordbox, 'traktor': traktor, 'virtualdj': virtualdj}
    faltan = [f'{p}: {", ".join(VARIANTES[v] or "WAV" for v in ORDEN if programas[p].get(v) is None)}'
              for p in programas if any(programas[p].get(v) is None for v in ORDEN)]
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
            for v in ORDEN[1:]:
                pv = programas[p].get(v)
                partes = [f'{d} {_ms(pv - app[d][v]) if pv is not None and app[d].get(v) is not None else "      -"}'
                          for d in decs]
                print(f'    {VARIANTES[v]:<14}' + '   '.join(partes))
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
    args = ap.parse_args(argv)
    if args.crear:
        return crear(args.crear, args.ffmpeg)
    if args.leer:
        return leer(args.leer, args.rekordbox, args.traktor, args.virtualdj, args.ffmpeg)
    ap.error('--crear CARPETA o --leer CARPETA')
    return 2


if __name__ == '__main__':
    sys.exit(main())
