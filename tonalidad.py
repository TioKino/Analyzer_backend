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


def croma_de_trozos(cromas: Iterable[Sequence[float]]) -> Optional[List[float]]:
    """El croma del tema entero a partir del de cada trozo (cada uno ya
    normalizado): la media de los que hay. Los trozos son todos de la misma
    duración salvo el último, así que pesarlos igual es el croma del tema."""
    validos = [np.asarray(c, dtype=float) for c in cromas
               if c is not None and len(c) == 12]
    if not validos:
        return None
    return croma_para_guardar(np.mean(validos, axis=0))


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
    Es lo que hace hoy /analyze con Krumhansl-Kessler (`main.analyze_audio`,
    `ChunkedAudioAnalyzer.analyze_chunk_key`)."""
    v = np.asarray(croma, dtype=float)
    if v.shape != (12,) or not np.all(np.isfinite(v)):
        return None
    v = v - v.mean()
    if np.linalg.norm(v) < 1e-12:
        return None
    v = v / np.linalg.norm(v)
    i = int(np.argmax(_plantillas(mayor, menor) @ v))
    return camelot_de(i % 12, 'mayor' if i < 12 else 'menor')


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
