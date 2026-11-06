#!/usr/bin/env python3
"""Mide la TONALIDAD de DJ Analyzer contra la que ya tienes en tus programas de
DJ, con la música de TU disco, y prueba otras formas de calcularla.

Por qué (estudio de la competencia, 2026-11-05): la tonalidad de /analyze
coincide con la de Rekordbox/VirtualDJ del mismo fichero en un 41-49 %
(`tonalidad_en_sombra` del panel), y la rueda de Camelot, Suggest y Flow se
apoyan en ella. El panel solo puede probar perfiles sobre el croma que ya
guardó; esto, en cambio, tiene el AUDIO delante y prueba también otros cromas:

- `todo`       el croma del tema entero: lo que hace hoy /analyze.
- `centro`     solo la mitad central del tema (sin intro ni outro de batería).
- `sin_sub`    sin la primera octava (32-65 Hz: el bombo y el subgrave).
- `sin_graves` sin las dos primeras (hasta 130 Hz).
- `umbral`     cada instante sin las notas por debajo del 20 % de la más fuerte.
- `log`        con la energía comprimida (que una nota fuerte no lo tape todo).
- `armonico`   el centro del tema sin la percusión (separación armónica).
- `voto`       un voto por cada minuto, pesado por su energía: como el camino
               de los temas largos.

Cada croma se cruza con todos los perfiles de `tonalidad.PERFILES` (los de hoy
y los de música electrónica) y con uno APRENDIDO de tu propia música, medido
con validación cruzada (se aprende de una mitad y se prueba en la otra).

La verdad con la que se compara es la que ya tienes: la de Rekordbox, Traktor y
VirtualDJ, y la que va dentro del fichero (la etiqueta TKEY: la ponen Beatport,
Mixed In Key…). Ninguna es perfecta, así que también sale cuánto coinciden
entre ellas, y una tabla solo con los temas en que dos fuentes dicen lo mismo
(el «consenso»), que es la verdad más de fiar que hay.

No escribe nada en ningún sitio: lee tus bibliotecas y tus ficheros y deja un
CSV con el detalle.

Uso (con lo que tengas; basta una fuente):

    python3 scripts/medir_tonalidad.py --rekordbox ~/rekordbox.xml
    python3 scripts/medir_tonalidad.py \\
        --rekordbox ~/rekordbox.xml \\
        --traktor "~/Documents/Native Instruments/Traktor 4.5.1/collection.nml" \\
        --virtualdj "~/Library/Application Support/VirtualDJ/database.xml" \\
        --max 300

    --carpeta DIR   suma los ficheros de esa carpeta que traen la tonalidad
                    en la etiqueta (sin programa de DJ)
    --max N         cuántos temas analizar (por defecto 200, al azar)
    --sr 44100      la frecuencia de /analyze en el camino corto (tarda el
                    doble; por defecto 22050, que da el mismo croma)

Requisitos: librosa, numpy, scipy, soundfile y mutagen
(`pip install -r requirements.txt` del backend, o los cinco a mano).
"""

import argparse
import concurrent.futures
import csv
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))

import numpy as np  # noqa: E402

import bibliotecas_dj as bib  # noqa: E402
import tonalidad as ton  # noqa: E402

HOP = 512
EXTENSIONES = ('.mp3', '.flac', '.wav', '.aif', '.aiff', '.m4a', '.mp4', '.ogg', '.opus')
CROMAS = ('todo', 'centro', 'sin_sub', 'sin_graves', 'umbral', 'log', 'armonico')
FUENTES = ('rekordbox', 'traktor', 'virtualdj', 'etiqueta')
LO_DE_HOY = 'todo + kk'
COMPATIBLES = ('igual', 'relativa', 'quinta')


# --------------------------------------------------------------- el audio

def _media(ch) -> Optional[List[float]]:
    v = np.asarray(ch, dtype=float)
    if v.ndim != 2 or v.shape[1] == 0:
        return None
    v = v.mean(axis=1)
    s = float(v.sum())
    if not np.isfinite(s) or s <= 0:
        return None
    return [float(x) for x in v / s]


def cromas_del_audio(y, sr: int) -> Dict:
    """Los cromas medios de todas las variantes, de una señal mono. `todo` es
    exactamente `librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12,
    n_octaves=7)` promediado (lo ata el test): la CQT se calcula una vez y las
    variantes la recortan."""
    import librosa

    # `tuning=None`: que estime la afinación, como hace `chroma_cqt` por dentro.
    # `librosa.cqt` a secas NO la estima (su defecto es 0.0), y sin esto el
    # croma ya no sería el de /analyze (lo cazó el test).
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, n_bins=7 * 36, bins_per_octave=36,
                           tuning=None))
    ch = librosa.feature.chroma_cqt(C=C, n_chroma=12)
    n = ch.shape[1]
    out: Dict = {'todo': _media(ch), 'centro': _media(ch[:, n // 4: max(n // 4 + 1, 3 * n // 4)])}
    for nombre, octavas in (('sin_sub', 1), ('sin_graves', 2)):
        c = C.copy()
        c[:36 * octavas] = 0
        out[nombre] = _media(librosa.feature.chroma_cqt(C=c, n_chroma=12))
    u = ch.copy()
    u[u < 0.2] = 0
    out['umbral'] = _media(u)
    out['log'] = _media(librosa.feature.chroma_cqt(
        C=np.log1p(1000.0 * C / (C.max() + 1e-12)), n_chroma=12))
    a = len(y) // 4
    b = min(len(y) - len(y) // 4, a + int(150 * sr))
    if b - a > sr * 5:
        yh = librosa.effects.harmonic(np.ascontiguousarray(y[a:b]))
        Ch = np.abs(librosa.cqt(yh, sr=sr, hop_length=HOP, n_bins=7 * 36, bins_per_octave=36,
                                tuning=None))
        out['armonico'] = _media(librosa.feature.chroma_cqt(C=Ch, n_chroma=12))
    # Un trozo por minuto, con su energía, para el voto de los temas largos.
    paso = int(round(60 * sr / HOP))
    trozos = []
    for i in range(0, n, paso):
        t = ch[:, i:i + paso]
        if t.shape[1] < paso // 6:
            continue
        seg = y[i * HOP:(i + t.shape[1]) * HOP]
        energia = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        m = _media(t)
        if m:
            trozos.append((m, energia))
    out['trozos'] = trozos
    return out


def _duracion_de_la_cabecera(ruta: str) -> Optional[float]:
    """Los segundos que dice la cabecera (mutagen), sin decodificar nada."""
    try:
        import mutagen
        f = mutagen.File(ruta)
        return float(f.info.length) if f is not None and f.info else None
    except Exception:
        return None


def analizar(ruta: str, sr: int = 22050) -> Dict:
    """Un tema: sus cromas y la tonalidad de su etiqueta. Corre en otro proceso
    (por eso importa aquí dentro)."""
    import warnings
    warnings.filterwarnings('ignore')   # los avisos de librosa al leer un M4A
    largo = _duracion_de_la_cabecera(ruta)
    if largo and largo > 20 * 60:
        # Una mezcla de dos horas son 600 MB al cargarla: no se carga.
        return {'ruta': ruta, 'error': 'más de 20 min (una mezcla)'}
    try:
        import librosa
        y, sr_ = librosa.load(ruta, sr=sr, mono=True)
    except Exception as e:  # el fichero que no se puede leer no tumba la medida
        return {'ruta': ruta, 'error': f'no se puede leer ({type(e).__name__})'}
    dur = len(y) / float(sr_)
    if dur < 30:
        return {'ruta': ruta, 'error': 'menos de 30 s'}
    if dur > 20 * 60:
        return {'ruta': ruta, 'error': 'más de 20 min (una mezcla)'}
    try:
        cromas = cromas_del_audio(y, sr_)
    except Exception as e:
        return {'ruta': ruta, 'error': f'el análisis falló ({type(e).__name__}: {e})'}
    return {'ruta': ruta, 'duracion': dur, 'cromas': cromas,
            'etiqueta': bib.tonalidad_de_la_etiqueta(ruta)}


# --------------------------------------------------------------- las tonalidades

def tonalidad_y_correlacion(croma: Sequence[float], perfil: Sequence[Sequence[float]]
                            ) -> Tuple[Optional[str], float]:
    """Como `tonalidad.tonalidad_desde_croma`, y además lo que correla."""
    v = np.asarray(croma, dtype=float)
    if v.shape != (12,) or not np.all(np.isfinite(v)):
        return None, 0.0
    v = v - v.mean()
    if np.linalg.norm(v) < 1e-12:
        return None, 0.0
    v = v / np.linalg.norm(v)
    r = ton._plantillas(*perfil[:2]) @ v
    i = int(np.argmax(r))
    return ton.camelot_de(i % 12, 'mayor' if i < 12 else 'menor'), float(r[i])


def por_voto(trozos: Sequence[Tuple[Sequence[float], float]],
             perfil: Sequence[Sequence[float]]) -> Optional[str]:
    """Un voto por minuto, pesado por su energía y por lo que correla: lo que
    hace `ChunkedAudioAnalyzer.fuse_key_results` con los temas largos."""
    votos: Dict[str, float] = defaultdict(float)
    total = sum(e for _, e in trozos) + 1e-10
    for croma, energia in trozos:
        k, r = tonalidad_y_correlacion(croma, perfil)
        if k:
            votos[k] += (energia / total) * max(r, 0.0)
    return max(votos, key=votos.get) if votos else None


def consenso(refs: Dict[str, Optional[str]]) -> Optional[str]:
    """La tonalidad en que coinciden al menos dos fuentes (la de más fuentes;
    con un empate, ninguna)."""
    c = Counter(v for v in refs.values() if v)
    if not c:
        return None
    (k, n), *resto = c.most_common()
    if n < 2 or (resto and resto[0][1] == n):
        return None
    return k


def referencias(refs: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Contra qué se mide cada tema: cada fuente, la del programa (Rekordbox,
    si no Traktor, si no VirtualDJ: lo que el DJ ve) y el consenso."""
    salida = {f: refs.get(f) for f in FUENTES}
    salida['programa'] = next((refs.get(p) for p in bib.PROGRAMAS if refs.get(p)), None)
    salida['consenso'] = consenso({f: refs.get(f) for f in FUENTES})
    return salida


def _mitad(ruta: str) -> int:
    return zlib.crc32(ruta.encode('utf-8', 'replace')) % 2


def predicciones(temas: List[Dict]) -> Dict[str, List[Optional[str]]]:
    """Para cada variante (croma + perfil), la tonalidad de cada tema. Los
    `aprendido` se calculan aparte, por referencia (`aprendidas`)."""
    pred: Dict[str, List[Optional[str]]] = {}
    for croma in CROMAS:
        for nombre, perfil in ton.PERFILES.items():
            pred[f'{croma} + {nombre}'] = [
                tonalidad_y_correlacion(t['cromas'][croma], perfil)[0]
                if t['cromas'].get(croma) else None for t in temas]
    for nombre, perfil in ton.PERFILES.items():
        pred[f'voto + {nombre}'] = [por_voto(t['cromas'].get('trozos') or [], perfil)
                                    for t in temas]
    return pred


def aprendidas(temas: List[Dict], verdad: List[Optional[str]]
               ) -> Dict[str, List[Optional[str]]]:
    """El perfil aprendido de cada croma, con validación cruzada en dos mitades
    fijas por fichero: un tema nunca se mide con un perfil que lo ha visto."""
    pred: Dict[str, List[Optional[str]]] = {}
    for croma in CROMAS:
        salida: List[Optional[str]] = [None] * len(temas)
        for mitad in (0, 1):
            pares = [(t['cromas'][croma], v) for t, v in zip(temas, verdad)
                     if v and t['cromas'].get(croma) and _mitad(t['ruta']) != mitad]
            perfiles = ton.aprender_perfiles(pares)
            if perfiles is None:
                continue
            for i, t in enumerate(temas):
                if _mitad(t['ruta']) == mitad and t['cromas'].get(croma):
                    salida[i] = ton.tonalidad_desde_croma(t['cromas'][croma], *perfiles)
        pred[f'{croma} + aprendido'] = salida
    return pred


def contar(pred: List[Optional[str]], verdad: List[Optional[str]]) -> Dict[str, int]:
    d: Dict[str, int] = {'comparados': 0}
    for p, v in zip(pred, verdad):
        clase = ton.clase_de_tonalidad(p, v)
        if clase is None:
            continue
        d['comparados'] += 1
        d[clase] = d.get(clase, 0) + 1
    return d


def tabla(pred: Dict[str, List[Optional[str]]], verdad: List[Optional[str]],
          mejores: int = 12) -> List[Tuple[str, Dict[str, int]]]:
    """Las variantes ordenadas por cuántas aciertan exacto (y por cuántas
    mezclan bien, a igualdad), con lo de hoy siempre dentro."""
    filas = [(nombre, contar(p, verdad)) for nombre, p in pred.items()]
    filas = [f for f in filas if f[1]['comparados']]

    def orden(f):
        d = f[1]
        return (d.get('igual', 0) / d['comparados'],
                sum(d.get(c, 0) for c in COMPATIBLES) / d['comparados'])

    filas.sort(key=orden, reverse=True)
    arriba = filas[:mejores]
    if not any(n == LO_DE_HOY for n, _ in arriba):
        arriba += [f for f in filas if f[0] == LO_DE_HOY]
    return arriba


def _pct(d: Dict[str, int], *clases: str) -> str:
    n = d.get('comparados') or 0
    return f'{100 * sum(d.get(c, 0) for c in clases) / n:3.0f} %' if n else '   -'


def imprimir_tabla(titulo: str, filas: List[Tuple[str, Dict[str, int]]]) -> None:
    if not filas:
        return
    n = filas[0][1]['comparados']
    print(f'\n{titulo}: {n} temas')
    print(f'  {"variante":<24} {"igual":>6} {"mezcla":>7} {"paralela":>9} '
          f'{"semitono":>9} {"otra":>6}')
    for nombre, d in filas:
        marca = '  ← hoy' if nombre == LO_DE_HOY else ''
        print(f'  {nombre:<24} {_pct(d, "igual"):>6} {_pct(d, *COMPATIBLES):>7} '
              f'{_pct(d, "paralela"):>9} {_pct(d, "semitono"):>9} {_pct(d, "otra"):>6}{marca}')


# --------------------------------------------------------------- el disco

def ficheros_con_etiqueta(carpeta: str) -> Dict[str, Dict[str, str]]:
    """Los ficheros de audio de una carpeta que traen la tonalidad dentro."""
    salida = {}
    for raiz, _, nombres in os.walk(os.path.expanduser(carpeta)):
        for nombre in nombres:
            if not nombre.lower().endswith(EXTENSIONES) or nombre.startswith('._'):
                continue
            ruta = os.path.join(raiz, nombre)
            k = bib.tonalidad_de_la_etiqueta(ruta)
            if k:
                salida[bib.clave_de_ruta(ruta)] = {'ruta': ruta, 'etiqueta': k}
    return salida


def candidatos(args) -> List[Tuple[str, Dict[str, Optional[str]]]]:
    """(ruta que existe, {fuente: tonalidad}) de cada tema con alguna fuente."""
    por_fichero = bib.leer_bibliotecas(args.rekordbox, args.traktor, args.virtualdj)
    salida: Dict[str, Tuple[str, Dict[str, Optional[str]]]] = {}
    for clave, temas in por_fichero.items():
        ruta = bib.ruta_que_existe(temas.values())
        if not ruta:
            continue
        refs = {p: t.tonalidad for p, t in temas.items()}
        if any(refs.values()):
            salida[clave] = (ruta, refs)
    if args.carpeta:
        for clave, d in ficheros_con_etiqueta(args.carpeta).items():
            if clave not in salida:
                salida[clave] = (d['ruta'], {'etiqueta': d['etiqueta']})
    return list(salida.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--rekordbox')
    ap.add_argument('--traktor')
    ap.add_argument('--virtualdj')
    ap.add_argument('--carpeta')
    ap.add_argument('--max', type=int, default=200)
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--procesos', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--semilla', type=int, default=7)
    ap.add_argument('--csv', default=os.path.expanduser('~/medir_tonalidad.csv'))
    args = ap.parse_args(argv)
    if not (args.rekordbox or args.traktor or args.virtualdj or args.carpeta):
        ap.error('pasa al menos una fuente: --rekordbox, --traktor, --virtualdj o --carpeta')

    t0 = time.time()
    todos = candidatos(args)
    print(f'[Tonalidad] {len(todos)} temas con fichero y alguna tonalidad que comparar')
    if not todos:
        print('Ninguno: mira las rutas de arriba (¿el disco externo está enchufado?).')
        return 1
    random.Random(args.semilla).shuffle(todos)
    elegidos = todos[:args.max]

    temas: List[Dict] = []
    fallos: Counter = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.procesos) as pool:
        futuros = {pool.submit(analizar, ruta, args.sr): refs for ruta, refs in elegidos}
        for i, fut in enumerate(concurrent.futures.as_completed(futuros), 1):
            r = fut.result()
            if r.get('error'):
                fallos[r['error'].split(' (')[0]] += 1
            else:
                refs = dict(futuros[fut])
                refs['etiqueta'] = refs.get('etiqueta') or r.get('etiqueta')
                r['refs'] = refs
                temas.append(r)
            if i % 10 == 0 or i == len(futuros):
                pasado = time.time() - t0
                queda = pasado / i * (len(futuros) - i)
                print(f'[Tonalidad] {i} de {len(futuros)} · quedan ~{queda / 60:.0f} min',
                      flush=True)

    print('\n=== Tonalidad: DJ Analyzer frente a lo que ya tienes ===')
    print(f'Temas medidos: {len(temas)} de {len(elegidos)}'
          + (f' (sin medir: {dict(fallos)})' if fallos else '')
          + f' · {(time.time() - t0) / 60:.0f} min · sr {args.sr}')
    if not temas:
        return 1

    refs = [referencias(t['refs']) for t in temas]
    print('\nCuánto coinciden tus fuentes entre ellas (lo que se puede esperar):')
    parejas = 0
    for i, a in enumerate(FUENTES):
        for b in FUENTES[i + 1:]:
            d = contar([r[a] for r in refs], [r[b] for r in refs])
            if d['comparados'] >= 5:
                parejas += 1
                print(f'  {a + " ~ " + b:<24} {d["comparados"]:>4} temas · igual '
                      f'{_pct(d, "igual")} · mezcla {_pct(d, *COMPATIBLES)}')
    if not parejas:
        print('  (ninguna pareja de fuentes con 5 temas o más en común)')

    pred = predicciones(temas)
    for clave, titulo in (('consenso', 'Contra el CONSENSO (dos fuentes dicen lo mismo)'),
                          ('programa', 'Contra tu PROGRAMA de DJ'),
                          ('etiqueta', 'Contra la ETIQUETA del fichero'),
                          ('rekordbox', 'Contra REKORDBOX'),
                          ('traktor', 'Contra TRAKTOR'),
                          ('virtualdj', 'Contra VIRTUALDJ')):
        verdad = [r[clave] for r in refs]
        if sum(1 for v in verdad if v) < 20 and clave not in ('consenso', 'programa'):
            continue
        todas = dict(pred)
        todas.update(aprendidas(temas, verdad))
        imprimir_tabla(titulo, tabla(todas, verdad))

    print('\n«igual»: la misma tonalidad · «mezcla»: igual, relativa o quinta (mezclan bien)')
    print('«paralela»: la nota bien y el modo mal · «semitono»: medio tono (casi siempre la afinación)')

    try:
        columnas = ['ruta', 'duracion', *FUENTES, 'consenso', *sorted(pred)]
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(columnas)
            for i, (t, r) in enumerate(zip(temas, refs)):
                w.writerow([t['ruta'], f'{t["duracion"]:.0f}', *(r[x] or '' for x in FUENTES),
                            r['consenso'] or '', *(pred[k][i] or '' for k in sorted(pred))])
        print(f'\nDetalle por tema en: {args.csv}')
    except OSError as e:
        print(f'\n(no se pudo escribir el CSV: {e})')
    print('Pega desde «=== Tonalidad» hasta aquí.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
