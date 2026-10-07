"""El BPM que se guarda es el del intervalo AFINADO de la rejilla (2026-10-06).

librosa no mide el tempo: lo elige de los bins del tempograma, y con
`sr=44100` y `hop=512` esos bins van separados más de un BPM en la zona de
club. Un tema a 128 salía 129,20 —en la ficha, en la mezcla armónica y en el
XML de Rekordbox—, mientras `fit_beat_grid` ya recuperaba el 128 con la misma
señal para dibujar la rejilla y nadie lo usaba para el número.

Se prueba sobre audio sintético con la respuesta conocida, pasando por el
mismo librosa que usa el análisis: así se ve la cuantización Y su arreglo.

    pytest test_el_bpm_sale_de_la_rejilla.py -v
"""

import numpy as np
import pytest

from audio_helpers import beat_track_seguro
from beat_grid import (TOL_INTERVALO, bpm_de_la_rejilla, fit_beat_grid,
                       onset_envelope)

SR = 44100


def _sintetico(bpm, dur=60.0, fase=0.21, semilla=3):
    """Bombo en cada beat (acento en el 1) y un charles a contratiempo."""
    y = np.zeros(int(SR * dur), dtype=np.float32)
    iv = 60.0 / bpm
    t = np.arange(int(0.08 * SR)) / SR
    bombo = (np.sin(2 * np.pi * 55 * t) * np.exp(-t * 35)).astype(np.float32)
    rnd = np.random.RandomState(semilla)
    k = 0
    while fase + k * iv < dur - 0.1:
        i = int((fase + k * iv) * SR)
        y[i:i + len(bombo)] += (1.0 if k % 4 == 0 else 0.7) * bombo
        j = int((fase + (k + 0.5) * iv) * SR)
        n = max(0, min(len(y) - j, int(0.02 * SR)))
        caida = np.exp(-np.arange(n) / SR * 200).astype(np.float32)
        y[j:j + n] += 0.15 * rnd.randn(n).astype(np.float32) * caida
        k += 1
    return y


def _medir(real, dur=60.0):
    y = _sintetico(real, dur=dur)
    tempo, _ = beat_track_seguro(y, SR)
    librosa_bpm = float(np.atleast_1d(tempo)[0])
    env, fps = onset_envelope(y, SR)
    fit = fit_beat_grid(env, fps, librosa_bpm)
    assert fit is not None
    return librosa_bpm, bpm_de_la_rejilla(librosa_bpm, fit['beat_interval'])


@pytest.mark.parametrize('real', [127.0, 128.0, 174.0])
def test_EL_CASO_librosa_se_pasa_y_la_rejilla_lo_recupera(real):
    librosa_bpm, de_la_rejilla = _medir(real)
    # La premisa: el bin de librosa cae a más de medio BPM (129,20 para 128).
    assert abs(librosa_bpm - real) > 0.5
    # Con un minuto de audio, a menos de 0,05; con un tema entero, a 0,01.
    assert abs(de_la_rejilla - real) < 0.05


def test_un_tempo_que_no_es_entero_se_queda_como_es():
    """127,96 no se redondea a 128: esa diferencia son 113 ms al final de seis
    minutos, que es justo lo que la rejilla viene a quitar."""
    librosa_bpm, de_la_rejilla = _medir(127.96, dur=360.0)
    assert abs(librosa_bpm - 127.96) > 0.5
    assert de_la_rejilla == pytest.approx(127.96, abs=0.015)


def test_sin_intervalo_manda_el_de_entrada():
    assert bpm_de_la_rejilla(129.2, None) == 129.2
    assert bpm_de_la_rejilla(129.2, 0) == 129.2
    assert bpm_de_la_rejilla(0, 0.46875) == 0


def test_el_intervalo_de_otro_tempo_no_cuela():
    """Fuera de lo que el afinado puede mover no es el mismo tempo afinado."""
    lejos = 60.0 / (129.2 * (1 + 2 * TOL_INTERVALO))
    assert bpm_de_la_rejilla(129.2, lejos) == 129.2
    assert bpm_de_la_rejilla(129.2, 60.0 / 128.0) == 128.0


def test_sin_afinar_el_bpm_no_cambia():
    """Cuando el afinado no se fía devuelve 60/BPM redondeado: el mismo BPM."""
    for bpm in (123.05, 126.05, 129.2, 136.0, 172.27):
        assert bpm_de_la_rejilla(bpm, round(60.0 / bpm, 6)) == round(bpm, 2)


def test_los_dos_caminos_del_dsp_lo_usan():
    """Los dos pasan por `rasgos_del_tema.rejilla_y_bpm`, y solo afinan el BPM
    si lo midió el DSP (el de las etiquetas manda su número)."""
    rej = open('rasgos_del_tema.py', encoding='utf-8').read()
    i = rej.index('def rejilla_y_bpm(')
    assert 'bpm_de_la_rejilla(bpm, beat_interval)' in rej[i:]
    src = open('main.py', encoding='utf-8').read()
    i = src.index('def analyze_audio(')
    corto = src[i:src.index('\ndef ', i + 10)]
    assert 'bpm_del_dsp=(bpm_source == "analysis")' in corto
    largo = open('chunked_analyzer.py', encoding='utf-8').read()
    assert "bpm_del_dsp=(bpm_source == 'chunked_analysis')" in largo
    # El BPM de las etiquetas llega al análisis por trozos ANTES de afinar.
    i = src.index('def analyze_audio_chunked(')
    envoltorio = src[i:src.index('\ndef ', i + 10)]
    assert "full_analysis(file_path, bpm_etiqueta=id3_data.get('bpm'))" in envoltorio
