"""
Dónde cae el primer beat, cuál de los cuatro es el «1», y cada cuánto van.

POR QUÉ EXISTE ESTE MÓDULO
--------------------------
Una rejilla son TRES cosas y solo una es el tempo:

    1. el intervalo  (60 / BPM)
    2. la FASE       (dónde cae el primer beat)
    3. el DOWNBEAT   (cuál de cada cuatro es el «1» del compás)

Fallaban las tres a la vez en el camino que usa Render para todo lo que pasa
de 4 minutos —o sea casi cualquier tema de club, y el 86,7 % de los análisis
van a Render—: `ChunkedAudioAnalyzer.calculate_beat_grid` devolvía

    {'first_beat': first_beat_offset}   # y nadie le pasaba nunca el offset

es decir, `0.0` sin buscar nada. La rejilla arrancaba donde arranca el
FICHERO, que es donde acaba el silencio de cabecera del MP3 y no donde entra
el bombo. Reportado como «los BPM no caen de primeras en el sitio adecuado,
hay que usar el tap para regularlo», y era literalmente eso.

El camino no-chunked sí buscaba la fase (`artwork_and_cuepoints.detect_beat_grid`)
pero con dos problemas propios: probaba 100 fases sueltas en bucles de Python
—caro y con la resolución de una centésima de beat— y no miraba el downbeat,
así que el BPM y la fase podían estar perfectos y la línea gorda caer en el 3.

QUÉ HACE DISTINTO
-----------------
* **Histograma de fase** en vez de probar fases sueltas: se dobla la envolvente
  sobre un solo beat y cada frame suma su fuerza en el milisegundo que le toca.
  Una pasada, vectorizado, y da la fase exacta en vez de la mejor de cien.
* **Centroide** del pico en vez del bin ganador: la envolvente va a ~86 frames
  por segundo, así que un beat cae repartido entre varios bins y quedarse con
  el más alto deja ~12 ms de error. Con el centroide, ~2 ms.
* **Afinado del intervalo por NITIDEZ del pliegue**: se prueban intervalos
  alrededor de 60/BPM y gana el que deja el histograma más picudo, porque con
  el intervalo bueno todos los beats caen en el mismo sitio al doblar y con uno
  torcido se reparten. Un tema masterizado a 127,96 etiquetado «128» acumula
  ~113 ms en seis minutos —un cuarto de beat: entra clavada y sale con el beat
  cambiado— y con esto baja a 0,1 ms. **Lo que había antes restaba la fase de
  las dos mitades del track, y sobre un tema BIEN etiquetado inventaba 24 ms de
  deriva que no existía**: los detalles y las medidas, en `_afinar_intervalo`.
  Todo sobre señal sintética, ver `tests/test_beat_grid.py`.
* **Downbeat**: se suma la fuerza del onset en los beats de cada residuo mod 4
  y gana el más fuerte.

El cliente lleva el MISMO algoritmo en `lib/services/beat_grid_detector.dart`,
sobre las bandas del espectro que ya tiene cacheadas. Es a propósito: así un
track legado se arregla al abrirlo, sin reanalizar nada. **Si tocas uno, toca
el otro** — dos rejillas distintas para el mismo track es peor que una mala.
"""

from typing import Dict, Optional, Sequence

import numpy as np

# Resolución de la búsqueda de fase. Más fino que 1 ms no lo sostiene una
# envolvente de ~86 frames/s y solo añade ruido.
MS_POR_BIN = 1

# Ventana de suavizado circular del histograma, en ms. Un bombo real no cae dos
# veces en el mismo milisegundo: sin suavizar, el máximo lo gana el bin donde
# casualmente pegó más fuerte una vez.
SUAVIZADO_MS = 25

# Cuánto se deja estirar el intervalo respecto a 60/BPM. Con más margen, en vez
# de afinar el tempo se salta al de al lado.
TOL_INTERVALO = 0.015

# Candidatos del barrido de intervalo. El pico de nitidez mide ~3e-4 de ancho
# relativo, así que con este paso (1,25e-4) caen cinco puntos dentro y la
# parábola afina por debajo del paso. Subirlo no mejora la medida y el coste es
# lineal: son ~0,2 s sobre un tema de seis minutos.
PASOS_INTERVALO = 241

# Cuántas veces tiene que superar el candidato ganador al candidato TÍPICO para
# que nos creamos que el barrido ha encontrado un tempo y no la fluctuación más
# afortunada. Un barrido es un optimizador: sobre ruido puro también devuelve un
# máximo, y sin esta puerta pasaba de MIN_CONFIANZA y dibujaba rejilla donde no
# hay pulso. Medido sobre 360 s: ruido puro da 7,1-7,4 y una pista de verdad
# 30-61, incluida la más sucia. Fallar la puerta NO tira la rejilla: devuelve
# 60/BPM sin afinar, que es la dirección segura.
MIN_PICO_INTERVALO = 12.0

# Por debajo de esto no se devuelve rejilla. Un track sin pulso claro daría una
# fase cualquiera, y una rejilla inventada es peor que ninguna.
MIN_CONFIANZA = 0.12


def _histograma_de_fase(onset: np.ndarray, fps: float, intervalo: float,
                        desde: int = 0, hasta: Optional[int] = None) -> np.ndarray:
    """Dobla la envolvente sobre un solo beat: cada frame suma su fuerza en el
    milisegundo del beat en el que cae."""
    if hasta is None:
        hasta = len(onset)
    tramo = onset[desde:hasta]
    if tramo.size == 0:
        return np.zeros(1, dtype=float)

    bins = max(1, int(round(intervalo * 1000 / MS_POR_BIN)))
    t = np.arange(desde, hasta, dtype=float) / fps
    idx = np.floor((np.mod(t, intervalo) * 1000 / MS_POR_BIN)).astype(int) % bins
    hist = np.bincount(idx, weights=np.maximum(tramo, 0.0), minlength=bins)
    return hist.astype(float)


def _suavizar_circular(hist: np.ndarray, radio: int) -> np.ndarray:
    """Media móvil CIRCULAR: el histograma es un beat, así que el final pega con
    el principio. Sin el envoltorio, una fase cercana a 0 se suaviza solo por un
    lado y pierde contra el resto."""
    n = len(hist)
    if radio <= 0 or n < 3:
        return hist
    r = min(radio, n // 2)
    ventana = np.ones(2 * r + 1, dtype=float) / (2 * r + 1)
    extendido = np.concatenate([hist[-r:], hist, hist[:r]])
    return np.convolve(extendido, ventana, mode="valid")[:n]


def _fase_de(onset: np.ndarray, fps: float, intervalo: float,
             desde: int = 0, hasta: Optional[int] = None):
    """(fase en segundos, confianza 0..1) sobre el tramo [desde, hasta)."""
    h = _suavizar_circular(
        _histograma_de_fase(onset, fps, intervalo, desde, hasta),
        SUAVIZADO_MS // MS_POR_BIN,
    )
    n = len(h)
    pico = int(np.argmax(h))
    maximo = float(h[pico])
    media = float(h.mean())
    confianza = 0.0 if maximo <= 0 else max(0.0, min(1.0, (maximo - media) / maximo))

    # Centroide alrededor del pico, restando el suelo.
    r = min(SUAVIZADO_MS // MS_POR_BIN, n // 2)
    desplaz = np.arange(-r, r + 1)
    valores = h[(pico + desplaz) % n] - media
    valores = np.maximum(valores, 0.0)
    total = float(valores.sum())
    centro = pico + (float((valores * desplaz).sum()) / total if total > 0 else 0.0)
    fase = (centro % n) * MS_POR_BIN / 1000.0
    return fase, confianza


def _nitidez(onset: np.ndarray, fps: float, intervalo: float) -> float:
    """Cuánto PICO tiene el histograma al doblar el track con este intervalo.

    Es el criterio entero del afinado: con el intervalo bueno todos los beats
    caen en el mismo sitio del pliegue y el histograma sale picudo; con uno
    ligeramente corto o largo la fase deriva a lo largo del tema, los golpes se
    reparten por todo el beat y el pico se aplana.
    """
    h = _suavizar_circular(
        _histograma_de_fase(onset, fps, intervalo),
        SUAVIZADO_MS // MS_POR_BIN,
    )
    media = float(h.mean())
    if media <= 0:
        return 0.0
    return (float(h.max()) - media) / media


def _afinar_intervalo(onset: np.ndarray, fps: float, base: float) -> float:
    """El intervalo que deja el histograma más picudo, dentro de TOL_INTERVALO.

    POR QUÉ UN BARRIDO Y NO LA DERIVA ENTRE LAS DOS MITADES, que es como estaba
    hasta el 2026-09-23 y parece más barato y más directo:

    * **Sobre un track BIEN etiquetado inventaba deriva.** Las dos mitades daban
      fases que diferían ~12 ms por puro ruido de estimación, y eso se convertía
      en tempo: un tema exactamente a 128 salía con el intervalo 31 µs corto, o
      sea 24 ms de deriva al final de seis minutos METIDOS por el afinado. Y
      después «convergía», porque con el intervalo ya torcido las dos mitades
      vuelven a concordar. Ese es el caso COMÚN —la mayoría de los tracks van al
      tempo que dice su BPM—, así que el afinado empeoraba el caso mayoritario
      para arreglar el raro.
    * **Con deriva grande salía del revés.** La diferencia de fases es circular:
      en cuanto la deriva entre las dos mitades pasa de medio beat, el envoltorio
      la lee por el otro lado. Un 128,35 etiquetado «128» acababa con casi un
      segundo de error al final del tema.
    * **Iterar lo empeoraba.** Sobre el 127,96 la primera pasada acertaba y las
      siguientes oscilaban alrededor.

    La nitidez del pliegue no tiene ninguno de los tres problemas: no resta dos
    medidas ruidosas, no envuelve, y no hace falta iterar porque el máximo se
    busca de una vez. Medido sobre señal sintética (`tests/test_beat_grid.py`),
    deriva al final de un tema de seis minutos:

        tempo real   antes      ahora
        128,00       24,2 ms     2,5 ms   ← el caso común, lo ROMPÍA
        127,96       34,1 ms     0,1 ms
        128,35      928,2 ms    14,5 ms
        127,50     1078,7 ms     5,6 ms
    """
    cands = base * (1.0 + np.linspace(-TOL_INTERVALO, TOL_INTERVALO,
                                      PASOS_INTERVALO))
    puntuacion = np.array([_nitidez(onset, fps, c) for c in cands])

    i = int(np.argmax(puntuacion))
    if i <= 0 or i >= len(cands) - 1:
        # El máximo cae en un borde: no hay pico, hay una rampa. Fiarse de él
        # sería estirar el tempo hasta el tope de la tolerancia por nada.
        return base

    # ¿Ha encontrado algo, o ha elegido la fluctuación con más suerte? Con un
    # pulso de verdad el intervalo bueno saca MUCHO al candidato típico; sobre
    # ruido, todos los intervalos son igual de malos y el ganador apenas
    # destaca. Ver MIN_PICO_INTERVALO.
    tipico = float(np.median(puntuacion))
    if tipico <= 0 or float(puntuacion[i]) / tipico < MIN_PICO_INTERVALO:
        return base

    # Parábola por los tres puntos de alrededor: el paso del barrido es más
    # grueso que la precisión que da el pico, y esto la recupera.
    y0, y1, y2 = puntuacion[i - 1], puntuacion[i], puntuacion[i + 1]
    den = y0 - 2 * y1 + y2
    # den < 0 es la condición de máximo de verdad (cóncavo). Con den >= 0 los
    # tres puntos no dibujan un pico y el vértice saldría disparado.
    desplaz = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    desplaz = max(-1.0, min(1.0, float(desplaz)))
    paso = float(cands[1] - cands[0])
    return float(cands[i] + desplaz * paso)


def fit_beat_grid(onset: Sequence[float], fps: float, bpm: float,
                  afinar_intervalo: bool = True,
                  t0: float = 0.0) -> Optional[Dict]:
    """Fase, downbeat e intervalo a partir de una envolvente de onset.

    Args:
        onset: envolvente (energía que SUBE), muestreada uniformemente.
        fps: frames por segundo de esa envolvente.
        bpm: tempo de partida.
        afinar_intervalo: False cuando el BPM lo ha fijado una persona — manda
            su número, no el nuestro.
        t0: instante absoluto del primer frame. Sirve para medir sobre un chunk
            del medio del track y devolver la rejilla en tiempos del track
            entero.

    Returns:
        dict con first_beat / beat_interval / confidence / downbeat_index, o
        None si no hay pulso que medir. **None significa "no lo sé"**: el
        llamante debe dejar la rejilla como estaba, no poner ceros.
    """
    onset = np.asarray(onset, dtype=float)
    if onset.size < 8 or fps <= 0 or bpm <= 0:
        return None

    base = 60.0 / bpm
    duracion = onset.size / fps
    if duracion < base * 8:
        return None

    intervalo = base

    if afinar_intervalo and duracion > base * 32:
        intervalo = _afinar_intervalo(onset, fps, base)

    fase, confianza = _fase_de(onset, fps, intervalo)
    if confianza < MIN_CONFIANZA:
        return None

    # Downbeat: cuál de cada cuatro es el «1».
    ventana = max(1, int(round(0.030 * fps)))
    por_residuo = np.zeros(4, dtype=float)
    k = 0
    while True:
        idx = int(round((fase + k * intervalo) * fps))
        if idx >= onset.size:
            break
        a = max(0, idx - ventana)
        b = min(onset.size, idx + ventana + 1)
        por_residuo[k % 4] += float(onset[a:b].max()) if b > a else 0.0
        k += 1
    downbeat = int(np.argmax(por_residuo))

    # A tiempos del track entero. Se reduce módulo el COMPÁS (4 beats) para que
    # `first_beat` siga siendo un downbeat: reducirlo módulo un beat perdería
    # justo lo que acabamos de averiguar.
    compas = 4 * intervalo
    first_beat = (t0 + fase + downbeat * intervalo) % compas

    return {
        "first_beat": round(float(first_beat), 4),
        "beat_interval": round(float(intervalo), 6),
        "confidence": round(float(confianza), 3),
        "downbeat_index": downbeat,
    }


def onset_envelope(y, sr, hop_length: int = 512):
    """Envolvente de onset con librosa. Aparte del resto a propósito: así
    `fit_beat_grid` se puede testear con señal sintética sin audio ni librosa.
    """
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    return np.asarray(env, dtype=float), sr / hop_length
