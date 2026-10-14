"""La rejilla cae en el BOMBO, no a contratiempo (owner, 2026-10-14).

*The Age of Love* (Jam & Spoon Watch Out For Stella Mix, 132,5 BPM) salía con
las líneas entre los golpes y había que moverla a mano medio beat. La fase sale
de la envolvente de onset, que cuenta lo que SUBE; en el trance lo que más sube
es el contratiempo: el bajo sostenido hasta el bombo siguiente le come la
subida al bombo y el charles abierto suma la suya. Hoy deciden los graves
(`beat_grid._fase_del_bombo`), en los tres caminos que ponen la rejilla.

Señal sintética con la respuesta conocida, por la MISMA tubería que /analyze:
la envolvente de librosa (`beat_grid.onset_envelope`) y los graves de
`bandas_de_audio`. Cada caso de trance comprueba también que sin los graves
cae a contratiempo, para que el test no pase por casualidad.

Espejo de `test/unit/la_rejilla_no_cae_a_contratiempo_test.dart` del cliente.
"""

from pathlib import Path

import numpy as np
import pytest

import beat_grid
from spectral_classifier import bandas_de_audio

SR = 22050
FASE = 0.31


def sintetiza(bpm, compas, dura=60.0, fase=FASE):
    """`compas` (beat, qué, fuerza, duración en beats) repetido desde `fase`."""
    n = int(round(dura * SR))
    x = np.zeros(n)
    iv = 60.0 / bpm
    ruido = np.random.default_rng(3).uniform(-1, 1, n)
    agudo = np.zeros(n)
    agudo[2:] = (ruido[2:] - 2 * ruido[1:-1] + ruido[:-2]) / 4
    medio = (np.convolve(ruido, np.ones(4) / 4)[:n]
             - np.convolve(ruido, np.ones(8) / 8)[:n])

    def suena(t0, hasta, f):
        a = max(0, int(round(t0 * SR)))
        b = min(n, int(round((t0 + hasta) * SR)))
        if b > a:
            k = np.arange(a, b)
            x[a:b] += f(k / SR - t0, k)

    t = fase
    while t < dura:
        for beat, que, fuerza, d in compas:
            tt = t + beat * iv
            if que == 'bombo':
                suena(tt, .4, lambda s, k: fuerza * np.sin(2 * np.pi * (60 - 15 * s) * s)
                      * np.exp(-s / .12))
                suena(tt, .02, lambda s, k: fuerza * .3 * ruido[k] * np.exp(-s / .004))
            elif que == 'bajo':
                dd = iv * d

                def bajo(s, k, dd=dd):
                    env = (np.minimum(1, s / .005)
                           * np.where(s < dd - .01, 1.0, np.maximum(0, (dd - s) / .01)))
                    return fuerza * .6 * env * sum(
                        np.sin(2 * np.pi * 110 * h * s) / h for h in (1, 2, 3))
                suena(tt, dd, bajo)
            elif que == 'charles':
                suena(tt, .3, lambda s, k: fuerza * agudo[k] * np.exp(-s / .08))
            elif que == 'cerrado':
                suena(tt, .08, lambda s, k: fuerza * agudo[k] * np.exp(-s / .015))
            elif que == 'caja':
                suena(tt, .25, lambda s, k: fuerza * (2.5 * medio[k] + .5 * agudo[k])
                      * np.exp(-s / .06))
        t += 4 * iv
    return x.astype(np.float32)


def cuatro(que, fuerza, desde=0.0, dura=.25):
    return [(b + desde, que, fuerza, dura) for b in range(4)]


def rejilla(x, bpm, con_graves=True):
    onset, fps = beat_grid.onset_envelope(x, SR)
    graves = bandas_de_audio(x, SR)[0] if con_graves else None
    return beat_grid.fit_beat_grid(onset, fps, bpm, graves=graves)


def error(fit, referencia):
    """Distancia de la rejilla a `referencia`, módulo el beat, con signo."""
    iv = fit['beat_interval']
    d = (fit['first_beat'] - referencia) % iv
    return d - iv if d > iv / 2 else d


@pytest.fixture(scope='module')
def donde_cae_el_bombo():
    """Dónde pone la rejilla la envolvente de librosa con SOLO bombo: unos
    20-30 ms después del ataque, porque su pico va un frame tarde. Los casos se
    miden contra eso, no contra el ataque: aquí se prueba el medio beat."""
    fit = rejilla(sintetiza(128, cuatro('bombo', 1)), 128)
    assert not fit['al_bombo']
    d = error(fit, FASE)
    assert 0 <= d < 0.045, d
    return FASE + d


TRANCE = {
    # El de Age of Love: bombo, bajo a contratiempo hasta el siguiente bombo y
    # charles abierto a contratiempo.
    'bajo al 80 % y charles abierto': (132.5, cuatro('bombo', 1)
                                      + cuatro('bajo', .8, .5, .5)
                                      + cuatro('charles', 2, .5)),
    'bajo tan fuerte como el bombo': (132.5, cuatro('bombo', 1)
                                     + cuatro('bajo', 1, .5, .5)
                                     + cuatro('charles', 1.5, .5)),
    'a 140 BPM': (140, cuatro('bombo', 1) + cuatro('bajo', .8, .5, .5)
                  + cuatro('charles', 2, .5)),
    'bajo flojo y charles fuerte': (138, cuatro('bombo', 1)
                                    + cuatro('bajo', .6, .5, .5)
                                    + cuatro('charles', 3, .5)),
}


@pytest.mark.parametrize('nombre', list(TRANCE))
def test_trance_la_rejilla_va_al_bombo(nombre, donde_cae_el_bombo):
    bpm, compas = TRANCE[nombre]
    x = sintetiza(bpm, compas)
    antes = rejilla(x, bpm, con_graves=False)
    iv = antes['beat_interval']
    assert abs(error(antes, donde_cae_el_bombo)) > iv / 2 - 0.06, (
        'sin los graves tiene que caer a contratiempo: si no, el caso no '
        'prueba nada')

    hoy = rejilla(x, bpm)
    assert hoy['al_bombo']
    assert abs(error(hoy, donde_cae_el_bombo)) < 0.020, (
        'la rejilla sigue a contratiempo, medio beat corrida')


BIEN = {
    'solo bombo': (128, cuatro('bombo', 1)),
    # Bajo corto a contratiempo MÁS fuerte que el bombo: no llega al 20 %.
    'house: bajo corto al 120 %': (124, cuatro('bombo', 1) + cuatro('bajo', 1.2, .5, .3)),
    # El bombo en contratiempos SUELTOS: sumando el tema, el contratiempo
    # tiene más grave que el beat; tiempo a tiempo, no.
    'breakbeat': (130, [(0, 'bombo', 1, .25), (1.5, 'bombo', 1, .25),
                        (2.5, 'bombo', 1, .25), (1, 'caja', 2, .25),
                        (3, 'caja', 2, .25)]
                  + [(i / 2, 'cerrado', .8, .25) for i in range(8)]),
    'drum & bass': (174, [(0, 'bombo', 1, .25), (2.5, 'bombo', 1, .25),
                          (1, 'caja', 2, .25), (3, 'caja', 2, .25),
                          (0, 'bajo', .8, 1.5), (2.5, 'bajo', .8, 1.0)]),
    'garage': (132, [(0, 'bombo', 1, .25), (2.75, 'bombo', .9, .25),
                     (1, 'caja', 2, .25), (3, 'caja', 2, .25)]
               + [(i / 2 + .25, 'cerrado', .8, .25) for i in range(8)]),
}


@pytest.mark.parametrize('nombre', list(BIEN))
def test_lo_que_ya_caia_bien_no_se_mueve(nombre):
    bpm, compas = BIEN[nombre]
    x = sintetiza(bpm, compas)
    antes = rejilla(x, bpm, con_graves=False)
    hoy = rejilla(x, bpm)
    assert not hoy['al_bombo'], 'se movió una rejilla que estaba bien'
    assert hoy['first_beat'] == antes['first_beat']


def test_sin_graves_o_sin_resolucion_lo_de_siempre():
    x = sintetiza(132.5, TRANCE['bajo al 80 % y charles abierto'][1])
    onset, fps = beat_grid.onset_envelope(x, SR)
    graves = bandas_de_audio(x, SR)[0]
    assert not beat_grid.fit_beat_grid(onset, fps, 132.5)['al_bombo']
    lenta = beat_grid.fit_beat_grid(onset, 15.0, 132.5 * 15 / fps, graves=graves)
    assert not (lenta or {}).get('al_bombo', False)
    # Y si los graves no casan en frames con la envolvente, no se usan.
    from rasgos_del_tema import rejilla_y_bpm
    fb_sin, _, _ = rejilla_y_bpm(onset, fps, 132.5, bpm_del_dsp=False)
    fb_mal, _, _ = rejilla_y_bpm(onset, fps, 132.5, bpm_del_dsp=False,
                                 graves=graves[:-5])
    assert fb_mal == fb_sin


def test_los_tres_caminos_le_pasan_los_graves():
    main_src = Path('main.py').read_text(encoding='utf-8')
    assert 'graves=bandas[0] if bandas else None' in main_src
    assert 'graves=bandas_re[0]' in main_src
    chunked = Path('chunked_analyzer.py').read_text(encoding='utf-8')
    assert "graves=f['bass']" in chunked
