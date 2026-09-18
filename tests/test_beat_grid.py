"""La rejilla cae donde esta el bombo, no en el segundo 0.

Reportado por el owner: «los BPM han mejorado pero no caen de primeras en el
sitio adecuado, hay que usar el tap para regularlo». Era literal:
`ChunkedAudioAnalyzer.calculate_beat_grid` devolvia `first_beat: 0.0` sin mirar
el audio, y por ahi pasa TODO lo que dura mas de 4 minutos — casi cualquier
tema de club, y el 86,7 % de los analisis van a Render.

Se prueba sobre senal sintetica con la respuesta conocida. Sobre un MP3 real no
hay contra que comparar mas que el ojo, que es justo como se colo esto.
"""

import numpy as np
import pytest

from beat_grid import MIN_CONFIANZA, fit_beat_grid

FPS = 86.13  # hop 512 a 44.1 kHz, que es lo que da el analisis


def pista(bpm, fase, duracion=360.0, residuo_downbeat=0,
          acento=2.2, ruido=0.05, semilla=1):
    """Envolvente de onset con un golpe por beat y acento uno de cada cuatro."""
    rnd = np.random.RandomState(semilla)
    n = int(duracion * FPS)
    o = rnd.rand(n) * ruido
    iv = 60.0 / bpm
    k = 0
    while fase + k * iv < duracion:
        i = int(round((fase + k * iv) * FPS))
        if 0 <= i < n:
            o[i] += acento if k % 4 == residuo_downbeat else 1.0
        k += 1
    return o


def distancia_a_la_rejilla(t, first_beat, iv):
    """A que distancia pasa la rejilla de `t`. Da igual que beat concreto sea."""
    d = (t - first_beat) % iv
    return min(d, iv - d)


def test_la_fase_cae_sobre_el_bombo():
    fase = 0.137
    fit = fit_beat_grid(pista(128, fase), FPS, 128)
    assert fit is not None
    # Menos de un frame (11,6 ms). Con la rejilla anclada en 0 el error seria
    # de 137 ms: un tercio de beat, que es lo que obligaba a dar al tap.
    assert distancia_a_la_rejilla(fase, fit["first_beat"], fit["beat_interval"]) < 0.012
    assert fit["confidence"] > 0.5


def test_el_downbeat_es_el_uno_del_compas():
    # El acento cae en el tercer beat de cada cuatro. Sin buscar downbeat, la
    # linea gorda —la que mira un DJ para entrar— caeria a contratiempo.
    fase, bpm = 0.891, 124.0
    iv = 60.0 / bpm
    fit = fit_beat_grid(
        pista(bpm, fase, residuo_downbeat=2, semilla=7), FPS, bpm)
    assert fit is not None
    compas = 4 * iv
    d = (fit["first_beat"] - (fase + 2 * iv)) % compas
    assert min(d, compas - d) < 0.015, "la linea de compas no cae en el «1»"


def test_first_beat_sigue_siendo_un_downbeat_al_reducir():
    # Se reduce modulo el COMPAS, no modulo un beat: reducir modulo un beat
    # perderia justo el downbeat que se acaba de averiguar.
    bpm = 128.0
    fit = fit_beat_grid(pista(bpm, 0.2, residuo_downbeat=1), FPS, bpm)
    assert fit is not None
    assert 0 <= fit["first_beat"] < 4 * (60.0 / bpm) + 1e-9


def test_un_tempo_mal_etiquetado_deja_de_derivar():
    # Caso real: tema masterizado a 127,96 con el ID3 diciendo «128».
    real, etiquetado, duracion = 127.96, 128.0, 360.0
    fit = fit_beat_grid(pista(real, 0.5, duracion), FPS, etiquetado)
    assert fit is not None

    iv_real = 60.0 / real
    beats = duracion / iv_real
    deriva_sin = abs(60.0 / etiquetado - iv_real) * beats
    deriva_con = abs(fit["beat_interval"] - iv_real) * beats

    # Sin afinar son ~113 ms al final: un cuarto de beat. La rejilla entra
    # clavada y sale con el beat cambiado.
    assert deriva_sin > 0.08
    assert deriva_con < 0.030


def test_con_bpm_manual_el_intervalo_se_respeta_exacto():
    onset = pista(127.96, 0.5)
    exacto = fit_beat_grid(onset, FPS, 128, afinar_intervalo=False)
    afinado = fit_beat_grid(onset, FPS, 128)
    assert exacto["beat_interval"] == pytest.approx(60.0 / 128, abs=1e-6)
    assert afinado["beat_interval"] != pytest.approx(60.0 / 128, abs=1e-7)


def test_sin_pulso_no_devuelve_rejilla():
    # Ruido puro: cualquier fase vale igual. Devolver una inventada moveria la
    # rejilla sola delante del usuario, que es peor que dejarla como estaba.
    rnd = np.random.RandomState(3)
    assert fit_beat_grid(rnd.rand(int(360 * FPS)), FPS, 128) is None


def test_no_se_inventa_el_tempo():
    # Si se le pide a media velocidad, responde a media velocidad. No le toca a
    # esta funcion decidir si el track va a 64 o a 128.
    fit = fit_beat_grid(pista(128, 0.2), FPS, 64)
    assert fit is not None
    assert abs(fit["beat_interval"] - 60.0 / 64) < 0.02


@pytest.mark.parametrize("onset,fps,bpm", [
    ([], FPS, 128),
    ([0.0] * 4, FPS, 128),
    (None, FPS, 128),
])
def test_entradas_vacias(onset, fps, bpm):
    if onset is None:
        pytest.skip("cubierto por los guards de tipo del llamante")
    assert fit_beat_grid(onset, fps, bpm) is None


def test_parametros_absurdos():
    onset = pista(128, 0.0, duracion=60)
    assert fit_beat_grid(onset, 0, 128) is None
    assert fit_beat_grid(onset, FPS, 0) is None
    assert fit_beat_grid(onset, FPS, -128) is None
    # Dos segundos: no hay ocho beats que medir.
    assert fit_beat_grid(pista(128, 0.0, duracion=2), FPS, 128) is None


def test_el_offset_absoluto_se_respeta():
    # Medir sobre un trozo del medio del track y devolver la rejilla en tiempos
    # del track entero: es lo que necesita el analizador por chunks.
    bpm, fase = 128.0, 0.2
    iv = 60.0 / bpm
    onset = pista(bpm, fase)
    sin_t0 = fit_beat_grid(onset, FPS, bpm)
    con_t0 = fit_beat_grid(onset, FPS, bpm, t0=30.0)
    compas = 4 * iv
    esperado = (sin_t0["first_beat"] + 30.0) % compas
    d = (con_t0["first_beat"] - esperado) % compas
    # 1 ms de margen: `first_beat` se redondea a 4 decimales en cada llamada y
    # los dos redondeos son independientes.
    assert min(d, compas - d) < 1e-3


def test_la_confianza_sube_con_el_pulso():
    limpio = fit_beat_grid(pista(128, 0.3, ruido=0.02), FPS, 128)
    sucio = fit_beat_grid(pista(128, 0.3, ruido=0.6, semilla=11), FPS, 128)
    assert limpio is not None
    assert limpio["confidence"] > MIN_CONFIANZA
    if sucio is not None:
        assert limpio["confidence"] >= sucio["confidence"]
