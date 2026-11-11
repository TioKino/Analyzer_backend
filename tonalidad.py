"""La tonalidad a partir del CROMA medio del tema, con varios perfiles, para
medirlos contra lo que dicen los programas de DJ (auditoría del flujo de
análisis, #4 de `PENDING.md`, 2026-10-07).

La primera lectura de la medida (snapshot 54 de `funnel_data/`) dio que el DSP
acierta la tonalidad de Rekordbox en el 45 % de los temas, y que el corto y el
de trozos aciertan lo mismo. Los dos sacan la tonalidad igual: el croma medio
correlado con los perfiles de Krumhansl-Kessler, que salen de oyentes de música
clásica. Cambiar el algoritmo sin medir sería a ciegas, y desde el contenedor
de desarrollo no hay audio de electrónica con la tonalidad anotada (GiantSteps
y Beatport dan 403).

Así que se mide EN SOMBRA y con la música de verdad: `/analyze` guarda el croma
medio del tema (`croma`, 12 números, `AnalysisResult`), y el panel prueba sobre
él cada perfil contra lo importado de la misma huella. Uno de los perfiles se
APRENDE de esos mismos temas (el croma de cada uno girado a la tónica que dice
su programa, promediado por modo) y se mide con validación cruzada: se aprende
de una mitad y se prueba en la otra. Nadie ve nada distinto hasta que se decida
con los números delante.

Sin dependencias más allá de numpy.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Los perfiles, de Do (índice 0) a Si. Mayor y menor.
PERFILES: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {
    # Krumhansl-Kessler: el que usan hoy los dos caminos de /analyze.
    'kk': ([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
           [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    # Temperley (1999): pesa menos la sensible y más la escala.
    'temperley': ([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
                  [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
    # La escala a secas (menor natural): la referencia más tonta.
    'diatonico': ([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                  [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]),
    # Los de la música ELECTRÓNICA (estudio de la competencia, 2026-11-05:
    # Lexicon acaba de pasarse a una red neuronal para la tonalidad y la
    # nuestra coincide con Rekordbox/VirtualDJ en un 41-49 %). Son los perfiles
    # publicados en los artículos, con los valores con que los trae la
    # biblioteca Essentia (`src/algorithms/tonal/key.cpp`); aquí solo se MIDEN.
    # ⚠️ Si alguno pasa a /analyze, mirar antes la licencia de esos números.
    # Sha'ath (2011, KeyFinder): Krumhansl ajustado a pop y electrónica.
    'shaath': ([6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4],
               [6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2]),
    # Faraldo, Gómez, Jordà y Herrera (ECIR 2016), «Key Estimation in
    # Electronic Dance Music»: sacados de un corpus de EDM.
    'edma': ([1.00, 0.29, 0.50, 0.40, 0.60, 0.56, 0.32, 0.80, 0.31, 0.45, 0.42, 0.39],
             [1.00, 0.31, 0.44, 0.58, 0.33, 0.49, 0.29, 0.78, 0.43, 0.29, 0.53, 0.32]),
    # El mismo, retocado a mano: el mayor es plano, así que casi todo sale en
    # menor (en EDM el mayor es raro; dice el artículo que acierta más igual).
    'edmm': ([0.083] * 12,
             [0.17235348, 0.04, 0.0761009, 0.12, 0.05621498, 0.08527853,
              0.0497915, 0.13451001, 0.07458916, 0.05003023, 0.09187879, 0.05545106]),
    # Faraldo, Jordà y Herrera (AES Semantic Audio 2017), «A multi-profile
    # method for key estimation in EDM»: la mediana de cada modo en temas de
    # Beatport (`braw`) y la misma con los cuatro valores menos útiles a cero
    # (`bgate`, el que Essentia usa por defecto).
    'braw': ([1.0000, 0.1573, 0.4200, 0.1570, 0.5296, 0.3669, 0.1632, 0.7711, 0.1676, 0.3827, 0.2113, 0.2965],
             [1.0000, 0.2330, 0.3615, 0.3905, 0.2925, 0.3777, 0.1961, 0.7425, 0.2701, 0.2161, 0.4228, 0.2272]),
    'bgate': ([1.00, 0.00, 0.42, 0.00, 0.53, 0.37, 0.00, 0.77, 0.00, 0.38, 0.21, 0.30],
              [1.00, 0.00, 0.36, 0.39, 0.00, 0.38, 0.00, 0.74, 0.27, 0.00, 0.42, 0.23]),
}

NOMBRES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CLASES = ('igual', 'relativa', 'quinta', 'paralela', 'semitono', 'otra')


# --------------------------------------------------------------- Camelot

def partir_camelot(c: Optional[str]) -> Optional[Tuple[int, str]]:
    """«8A» → (8, 'A'). None si no es un Camelot."""
    import re
    m = re.match(r'^\s*(\d{1,2})\s*([ABab])\s*$', c or '')
    if not m or not 1 <= int(m.group(1)) <= 12:
        return None
    return int(m.group(1)), m.group(2).upper()


def tonica_y_modo(c: Optional[str]) -> Optional[Tuple[int, str]]:
    """«8A» → (9, 'menor'): la tónica en semitonos desde Do y el modo. En
    Camelot, cada número más es una quinta más (7 semitonos); 8B es Do mayor y
    8A La menor."""
    p = partir_camelot(c)
    if not p:
        return None
    n, letra = p
    if letra == 'B':
        return (n - 8) * 7 % 12, 'mayor'
    return (9 + (n - 8) * 7) % 12, 'menor'


def camelot_de(tonica: int, modo: str) -> str:
    """(9, 'menor') → «8A». El inverso de `tonica_y_modo` (7 es su propio
    inverso módulo 12)."""
    if modo == 'mayor':
        return f'{(tonica * 7 % 12 + 7) % 12 + 1}B'
    return f'{((tonica - 9) * 7 % 12 + 7) % 12 + 1}A'


def clase_de_tonalidad(dsp: Optional[str], programa: Optional[str]) -> Optional[str]:
    """Cómo se parece la tonalidad medida a la del programa, en Camelot.

    - `relativa` (8A↔8B) y `quinta` (8A↔7A/9A): los errores típicos de un
      detector, y mezclan bien.
    - `paralela` (La menor ↔ La mayor, 8A↔11B): la tónica bien y el modo mal.
      Pide otro arreglo que todo lo demás: decidir el modo, no la nota.
    - `semitono` (La menor ↔ La# menor, 8A↔3A): el mismo modo medio tono
      arriba o abajo. Suele ser la afinación (un vinilo pasado con pitch, un
      tema que no está a 440 Hz).
    - `otra`: lo demás.

    Hasta el 2026-10-07 `paralela` y `semitono` iban dentro de `otra`.
    """
    a, b = partir_camelot(dsp), partir_camelot(programa)
    if not a or not b:
        return None
    if a == b:
        return 'igual'
    if a[0] == b[0]:
        return 'relativa'
    if a[1] == b[1] and (a[0] - b[0]) % 12 in (1, 11):
        return 'quinta'
    ta, tb = tonica_y_modo(dsp), tonica_y_modo(programa)
    if ta[0] == tb[0]:
        return 'paralela'
    if ta[1] == tb[1] and (ta[0] - tb[0]) % 12 in (1, 11):
        return 'semitono'
    return 'otra'


# --------------------------------------------------------------- el croma

def croma_para_guardar(croma: Optional[Iterable[float]]) -> Optional[List[float]]:
    """El croma medio como se guarda: 12 números que suman 1, a 4 decimales.
    None si no hay croma que guardar (silencio, un fallo)."""
    if croma is None:
        return None
    v = np.asarray(list(croma), dtype=float)
    if v.shape != (12,) or not np.all(np.isfinite(v)) or v.sum() <= 0:
        return None
    v = v / v.sum()
    return [round(float(x), 4) for x in v]


def _plantillas(mayor: Sequence[float], menor: Sequence[float]) -> np.ndarray:
    """Las 24 plantillas (12 mayores y 12 menores) centradas y de norma 1,
    para que el producto con un croma centrado y normalizado sea su
    correlación de Pearson, la de `np.corrcoef`."""
    filas = [np.roll(np.asarray(p, dtype=float), i)
             for p in (mayor, menor) for i in range(12)]
    t = np.asarray(filas)
    t = t - t.mean(axis=1, keepdims=True)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)


def tonalidad_desde_croma(croma: Sequence[float], mayor: Sequence[float],
                          menor: Sequence[float]) -> Optional[str]:
    """El Camelot con más correlación entre el croma y las 24 plantillas.
    Con `PERFILES['kk']` es lo que hacía /analyze hasta el 2026-11-10; con
    `PERFIL`, la elección de hoy (`estimar_tonalidad`)."""
    v = np.asarray(croma, dtype=float)
    if v.shape != (12,) or not np.all(np.isfinite(v)):
        return None
    v = v - v.mean()
    if np.linalg.norm(v) < 1e-12:
        return None
    v = v / np.linalg.norm(v)
    i = int(np.argmax(_plantillas(mayor, menor) @ v))
    return camelot_de(i % 12, 'mayor' if i < 12 else 'menor')


# --------------------------------------------------------------- la de /analyze

# Lo que hace /analyze desde el 2026-11-10, en los DOS caminos (el corto y el
# de trozos), en el motor local y en /identify: el croma del tema entero con
# COMPRESIÓN LOGARÍTMICA y el perfil `braw`. Medido en el disco del owner con
# `scripts/medir_tonalidad.py` (287 temas contra su Rekordbox, 2026-11-06):
# coincide en un 63 %, frente al 44-45 % de Krumhansl sobre el croma lineal, y
# acierta igual en mayor (61 %) que en menor (63 %). `edmm` daba más (65 %)
# diciendo «menor» a todo: un perfil que tira a menor gana contra una
# biblioteca de electrónica sin ser mejor. Hasta ese día el corto usaba el
# croma lineal con Krumhansl y el de trozos votaba trozo a trozo, y los dos
# coincidían con Rekordbox en un 44 %.
#
# Los números de `braw` son los publicados por Faraldo, Jordà y Herrera (AES
# 2017), tal como los trae Essentia: 24 medianas medidas sobre temas de
# Beatport. Son el resultado de una medida publicada, no código. Si alguna vez
# hiciera falta quitarlos, el perfil aprendido de la biblioteca del owner daba
# un 62 % con validación cruzada (`evaluar_perfiles`) y es nuestro.
#
# Cada análisis lleva `tonalidad_metodo` = METODO: la sombra
# (`tonalidad_en_sombra`) y la puesta al día lo necesitan para no mezclar el
# croma de antes (lineal, otro algoritmo) con el de ahora.
METODO = 'log+braw'
PERFIL = 'braw'
# La confianza de un tema sin tonalidad medible (silencio, croma plano): 0 es
# «no medido» para el cliente (`KeyAnalysis.keyMeasured`).
CONFIANZA_SIN_TONO = 0.0
COMPRESION = 1000.0
HOP = 512
BINS_POR_OCTAVA = 36
OCTAVAS = 7


def cqt_para_tonalidad(y, sr) -> np.ndarray:
    """Las magnitudes de la CQT que `chroma_cqt` calcula por dentro: 7 octavas
    desde C1, 36 bins por octava, hop 512 y la afinación ESTIMADA
    (`tuning=None`: el defecto de `librosa.cqt` es 0,0, y con él el croma ya
    no es el de `chroma_cqt`). En float32: el camino de trozos la guarda
    entera."""
    import librosa
    return np.abs(librosa.cqt(y, sr=sr, hop_length=HOP,
                              n_bins=OCTAVAS * BINS_POR_OCTAVA,
                              bins_per_octave=BINS_POR_OCTAVA,
                              tuning=None)).astype(np.float32)


def croma_log_de_piezas(piezas) -> Optional[np.ndarray]:
    """El croma por frame (12 × frames) de una CQT de magnitudes partida en
    [piezas] (frames seguidos; el camino de trozos la cose así), con la
    compresión logarítmica sobre el máximo del TEMA ENTERO, `log1p(1000 ·
    C / máx)`: lo suave (el bajo, los pads, la armonía) pesa frente al bombo y
    los picos, que en electrónica la tapan. Cada frame va por su cuenta una vez
    se sabe el máximo, así que da lo mismo que `croma_log` de la CQT entera sin
    tener que juntarla (lo ata el test). None con silencio."""
    import librosa
    validas = []
    for p in piezas or ():
        if p is None:
            continue
        p = np.asarray(p, dtype=np.float32)
        if p.ndim == 2 and p.shape[1] > 0:
            validas.append(p)
    if not validas:
        return None
    m = max(float(p.max()) for p in validas)
    if not np.isfinite(m) or m <= 0:
        return None
    return np.concatenate(
        [librosa.feature.chroma_cqt(C=np.log1p(COMPRESION * p / m), n_chroma=12,
                                    bins_per_octave=BINS_POR_OCTAVA)
         for p in validas], axis=1)


def croma_log(C) -> Optional[np.ndarray]:
    """`croma_log_de_piezas` de una CQT entera (el camino corto)."""
    return croma_log_de_piezas([C])


def media_del_croma(ch) -> Optional[np.ndarray]:
    """La media por frame de un croma (12 × frames), sumando 1. Sin redondear:
    es lo que se compara con los perfiles (lo guardado se redondea aparte,
    `croma_para_guardar`)."""
    if ch is None:
        return None
    v = np.asarray(ch, dtype=float)
    if v.ndim != 2 or v.shape[0] != 12 or v.shape[1] == 0:
        return None
    v = v.mean(axis=1)
    s = float(v.sum())
    if not np.isfinite(s) or s <= 0:
        return None
    return v / s


def estimar_tonalidad(croma, perfil: str = PERFIL) -> Optional[Dict]:
    """La tonalidad del croma medio [croma] con [perfil]: la plantilla (de 24)
    con más correlación de Pearson, la misma elección que
    `tonalidad_desde_croma` (lo ata el test). Devuelve `key` ('Am', 'C#', con
    sostenidos, como `KEY_TO_CAMELOT`), `camelot`, `confianza` (la correlación
    de la elegida) y `margen_modo` (la mejor mayor frente a la mejor menor:
    por debajo de 0,05 el modo es dudoso). None si no hay croma."""
    if croma is None:
        return None
    v = np.asarray(croma, dtype=float)
    if v.shape != (12,) or not np.all(np.isfinite(v)):
        return None
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    corr = _plantillas(*PERFILES[perfil]) @ (v / n)
    i = int(np.argmax(corr))
    tonica, modo = i % 12, ('mayor' if i < 12 else 'menor')
    return {
        'key': NOMBRES[tonica] + ('m' if modo == 'menor' else ''),
        'camelot': camelot_de(tonica, modo),
        'confianza': float(corr[i]),
        'margen_modo': float(abs(corr[:12].max() - corr[12:].max())),
    }


def tonalidad_del_croma_log(ch) -> Optional[Dict]:
    """`estimar_tonalidad` del croma log por frame [ch], con el croma medio
    para guardar (`croma`) y el método (`metodo`)."""
    media = media_del_croma(ch)
    t = estimar_tonalidad(media)
    if t is None:
        return None
    t['croma'] = croma_para_guardar(media)
    t['metodo'] = METODO
    return t


def tonalidad_del_audio(y, sr) -> Optional[Dict]:
    """La tonalidad de /analyze sobre una señal entera (el camino corto, el
    motor local y /identify). None con silencio o una señal sin tono."""
    return tonalidad_del_croma_log(croma_log(cqt_para_tonalidad(y, sr)))


# --------------------------------------------------------------- el perfil aprendido

def aprender_perfiles(pares: Sequence[Tuple[Sequence[float], str]]
                      ) -> Optional[Tuple[List[float], List[float]]]:
    """El perfil mayor y el menor de ESTA música: el croma de cada tema girado
    para que su tónica (la que dice su programa) quede en Do, promediado por
    modo. None si falta alguno de los dos modos."""
    suma = {'mayor': np.zeros(12), 'menor': np.zeros(12)}
    n = Counter()
    for croma, verdad in pares:
        tm = tonica_y_modo(verdad)
        if tm is None or croma is None or len(croma) != 12:
            continue
        suma[tm[1]] += np.roll(np.asarray(croma, dtype=float), -tm[0])
        n[tm[1]] += 1
    if not n['mayor'] or not n['menor']:
        return None
    return ([round(float(x), 5) for x in suma['mayor'] / n['mayor']],
            [round(float(x), 5) for x in suma['menor'] / n['menor']])


def _mitad(huella: str) -> int:
    """La mitad de la validación cruzada: fija por huella, para que dos
    lecturas del panel partan igual."""
    try:
        return int(str(huella)[:8], 16) % 2
    except ValueError:
        return sum(map(ord, str(huella))) % 2


def _vacio() -> Dict[str, int]:
    return {'comparados': 0, 'igual': 0}


def _apuntar(d: Dict[str, int], clase: Optional[str]) -> None:
    if clase is None:
        return
    d['comparados'] += 1
    d[clase] = d.get(clase, 0) + 1


def evaluar_perfiles(temas: Sequence[Tuple[str, Sequence[float], str]]) -> Dict:
    """Cada perfil, medido sobre `temas` = [(huella, croma, Camelot del
    programa)]. Los fijos (`PERFILES`) sobre todos; el `aprendido`, con
    validación cruzada en dos mitades: se aprende de una y se prueba en la
    otra, así que no se mide sobre lo que ha visto. Devuelve también el
    perfil aprendido de todos, para poder llevarlo a /analyze si gana."""
    salida: Dict = {'temas': len(temas), 'perfiles': {}}
    for nombre, (mayor, menor) in PERFILES.items():
        d = _vacio()
        for _, croma, verdad in temas:
            _apuntar(d, clase_de_tonalidad(tonalidad_desde_croma(croma, mayor, menor), verdad))
        salida['perfiles'][nombre] = d
    d = _vacio()
    for mitad in (0, 1):
        perfiles = aprender_perfiles([(c, v) for h, c, v in temas if _mitad(h) != mitad])
        if perfiles is None:
            continue
        for h, croma, verdad in temas:
            if _mitad(h) == mitad:
                _apuntar(d, clase_de_tonalidad(
                    tonalidad_desde_croma(croma, *perfiles), verdad))
    salida['perfiles']['aprendido'] = d
    todos = aprender_perfiles([(c, v) for _, c, v in temas])
    if todos:
        salida['perfil_aprendido'] = {'mayor': todos[0], 'menor': todos[1]}
    return salida


def guardada_frente_a_la_de_hoy(temas: Sequence[Tuple[Sequence[float], str, str, Optional[str]]]
                                 ) -> Dict[str, Dict]:
    """Por camino, la tonalidad GUARDADA y la que da hoy /analyze (`PERFIL`)
    sobre el croma del mismo tema, las dos contra el programa, y en cuántos
    coinciden. `temas` = [(croma, Camelot del programa, camino, Camelot
    guardado)], solo de temas con el croma de hoy (`METODO`).

    Desde el 2026-11-10 el corto y el de trozos sacan la tonalidad con la misma
    cuenta, así que en los dos tiene que coincidir el 100 %: si no, algo la
    cambia entre que se mide y se guarda. (Hasta ese día era Krumhansl, y en
    los trozos dos algoritmos distintos: lectura 55.) En `etiquetas` no
    coincide, y está bien: manda la tonalidad del fichero."""
    salida: Dict[str, Dict] = {}
    for croma, verdad, camino, guardada in temas:
        d = salida.setdefault(camino, {'temas': 0, 'guardada': _vacio(),
                                       'hoy': _vacio(), 'hoy_igual_que_guardada': 0})
        d['temas'] += 1
        hoy = tonalidad_desde_croma(croma, *PERFILES[PERFIL])
        _apuntar(d['guardada'], clase_de_tonalidad(guardada, verdad))
        _apuntar(d['hoy'], clase_de_tonalidad(hoy, verdad))
        if hoy and partir_camelot(hoy) == partir_camelot(guardada):
            d['hoy_igual_que_guardada'] += 1
    return salida
