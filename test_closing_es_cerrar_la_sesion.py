"""*Closing* es el tema con el que se CIERRA una sesión, no uno con outro
(#7 de PENDING, decisión owner 2026-10-07).

Los dos clasificadores premiaban el outro: la heurística sumaba 1,0 a
*closing* con un outro y más de 5 minutos (y 0,3 más pasados 7), y el
espectral +3 con un outro del 8 % del tema y +2 más pasado el 15 %, más la
intro y la duración. El outro de batería de cualquier extended mix, que está
para mezclar, salía *closing* con confianza 1,0 y descartaba al espectral:
8 de 13 temas largos en un log de Render del 7-oct. Hoy *closing* solo lo da
el espectral, por energía baja que va bajando.

    pytest test_closing_es_cerrar_la_sesion.py -v
"""

import os
import tempfile
import uuid

import numpy as np
import pytest
import soundfile

from rasgos_del_tema import classify_track_type
from spectral_classifier import classify_track_type_spectral, ensemble_classify

SR = 44100


def _secciones(intro=False, outro=False, drop=False):
    return {'has_intro': intro, 'has_outro': outro, 'has_drop': drop,
            'has_buildup': drop, 'has_breakdown': False}


def _puntos(r, tipo):
    return {a['type']: a['score'] for a in r['alternatives']}[tipo]


@pytest.mark.parametrize('duracion', [310.0, 400.0, 480.0])
@pytest.mark.parametrize('energia', [0.5, 0.55, 0.65])
def test_la_heuristica_no_da_closing_por_un_outro(duracion, energia):
    r = classify_track_type(energia, _secciones(intro=True, outro=True), duracion)
    assert r['type'] != 'closing'
    assert _puntos(r, 'closing') == 0


def test_el_caso_de_render_ya_no_ciega_al_espectral():
    """Energía 0,5-0,6, outro y más de 5 minutos: antes closing con
    confianza 1,0, y por encima de 0,70 el ensemble ni miraba al espectral."""
    r = classify_track_type(0.55, _secciones(outro=True), 420.0)
    assert r['confidence'] < 0.70


def _metricas(**kw):
    m = {'coreEnergy': 0.33, 'bassRatio': 0.4, 'dropContrast': 7.0,
         'energyTrend': 0.0, 'energyVariance': 0.19, 'peakPosition': 0.5,
         'introPercent': 0.0, 'outroPercent': 0.0, 'transientDensity': 0.05,
         'bassRegularity': 0.8}
    m.update(kw)
    return m


def test_el_espectral_no_da_puntos_a_closing_por_outro_intro_ni_duracion():
    sin = classify_track_type_spectral(_metricas(), 128.0, 250.0)
    con = classify_track_type_spectral(
        _metricas(outroPercent=0.2, introPercent=0.15), 128.0, 500.0)
    assert _puntos(con, 'closing') == _puntos(sin, 'closing') == 0
    assert con['type'] != 'closing'


def test_el_espectral_da_closing_a_lo_que_baja():
    """Lo que le queda a closing: energía baja que va bajando."""
    r = classify_track_type_spectral(
        _metricas(coreEnergy=0.29, energyTrend=-0.04), 126.0, 400.0)
    assert _puntos(r, 'closing') == 3.5


def test_el_ensemble_sigue_pudiendo_dar_closing():
    h = classify_track_type(0.55, _secciones(), 400.0)
    s = {'type': 'closing', 'confidence': 0.5, 'source': 'spectral',
         'alternatives': [{'type': 'closing', 'score': 6.0},
                          {'type': 'warmup', 'score': 1.0}]}
    assert ensemble_classify(h, s)['type'] == 'closing'


def _extended_mix(dur=330.0, bpm=128.0):
    """Bombo, charles, bajo y acorde, con 32 s de intro y de outro solo de
    batería: el outro para mezclar de cualquier extended mix."""
    rnd = np.random.RandomState(5)
    n = int(SR * dur)
    iv = 60.0 / bpm
    tt = np.arange(n) / SR
    t = np.arange(int(0.12 * SR)) / SR
    bombo = (np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t)
             * np.exp(-t * 18)).astype(np.float32)
    bat = np.zeros(n, dtype=np.float32)
    k = 0
    while 0.2 + k * iv < dur - 0.2:
        s = 0.2 + k * iv
        i = int(s * SR)
        bat[i:i + len(bombo)] += 0.9 * bombo
        j = int((s + iv / 2) * SR)
        m = max(0, min(n - j, int(0.03 * SR)))
        bat[j:j + m] += (0.12 * rnd.randn(m)
                         * np.exp(-np.arange(m) / SR * 150)).astype(np.float32)
        k += 1
    tonal = (0.18 * np.sin(2 * np.pi * 55 * tt)
             + 0.05 * (np.sin(2 * np.pi * 220 * tt)
                       + np.sin(2 * np.pi * 277.2 * tt))).astype(np.float32)
    cuerpo = ((tt >= 32) & (tt <= dur - 32)).astype(np.float32)
    y = bat + tonal * cuerpo
    return y / (np.max(np.abs(y)) + 1e-9) * 0.8


def test_EL_CASO_un_extended_mix_no_sale_closing():
    """De punta a punta por el camino corto (el del motor local, que analiza
    los temas largos entero): el outro se detecta, y el tema no sale
    closing. Con la regla de antes salía closing con confianza 1,0."""
    import main
    ruta = tempfile.mktemp(suffix='.wav')
    soundfile.write(ruta, _extended_mix(), SR)
    viejos = (main.ARTWORK_ENABLED, main.AUDD_AUTO_ENABLED,
              main.GENRE_DETECTOR_ENABLED, main.CHUNKED_ANALYZER_ENABLED)
    main.ARTWORK_ENABLED = main.AUDD_AUTO_ENABLED = False
    main.GENRE_DETECTOR_ENABLED = main.CHUNKED_ANALYZER_ENABLED = False
    try:
        r = main.analyze_audio(ruta, fingerprint=uuid.uuid4().hex,
                               original_filename='x.wav')
    finally:
        (main.ARTWORK_ENABLED, main.AUDD_AUTO_ENABLED,
         main.GENRE_DETECTOR_ENABLED, main.CHUNKED_ANALYZER_ENABLED) = viejos
        os.remove(ruta)
    assert r.has_outro, 'el outro se sigue viendo: lo que cambia es lo que significa'
    assert r.track_type != 'closing'
