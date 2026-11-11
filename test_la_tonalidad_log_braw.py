"""La tonalidad de /analyze es el croma LOG con el perfil `braw` (2026-11-10).

Medido en el disco del owner (`scripts/medir_tonalidad.py`, 287 temas contra
su Rekordbox): Krumhansl sobre el croma lineal, lo que hacían los dos caminos,
coincidía en un 44-45 %; el croma con compresión logarítmica y el perfil
`braw` de Faraldo et al. coincide en un 63 %, y acierta igual en mayor (61 %)
que en menor (63 %). Este test ata que lo que corre es eso, en los TRES sitios
que sacan una tonalidad (el corto, el de trozos y /identify), con una sola
función, y que cada análisis dice con qué método salió.

    pytest test_la_tonalidad_log_braw.py -v
"""

import os
import tempfile

import numpy as np
import soundfile

import tonalidad as t

SR = 44100


def _acorde(notas, dur=30.0, bpm=126.0):
    """Notas sostenidas con sus armónicos y un bombo encima: lo que tapa la
    armonía en electrónica."""
    n = int(SR * dur)
    tt = np.arange(n) / SR
    y = np.zeros(n, dtype=np.float32)
    for f, a in notas:
        for h, ah in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2)):
            y += (a * ah * np.sin(2 * np.pi * f * h * tt)).astype(np.float32)
    iv = 60.0 / bpm
    tb = np.arange(int(0.12 * SR)) / SR
    bombo = (np.sin(2 * np.pi * (50 + 80 * np.exp(-tb * 30)) * tb)
             * np.exp(-tb * 18)).astype(np.float32)
    k = 0
    while k * iv < dur - 0.2:
        i = int(k * iv * SR)
        y[i:i + len(bombo)] += 0.9 * bombo
        k += 1
    return y / (np.max(np.abs(y)) + 1e-9) * 0.8


LA_MENOR = [(55.0, 0.5), (220.0, 0.25), (261.63, 0.2), (329.63, 0.2), (440.0, 0.15)]
DO_MAYOR = [(65.41, 0.5), (261.63, 0.25), (329.63, 0.2), (392.0, 0.2), (523.25, 0.15)]


def test_un_acorde_de_la_menor_y_uno_de_do_mayor():
    am = t.tonalidad_del_audio(_acorde(LA_MENOR), SR)
    assert (am['key'], am['camelot']) == ('Am', '8A')
    assert am['metodo'] == t.METODO == 'log+braw'
    assert len(am['croma']) == 12 and abs(sum(am['croma']) - 1) < 1e-3
    do = t.tonalidad_del_audio(_acorde(DO_MAYOR), SR)
    assert (do['key'], do['camelot']) == ('C', '8B')
    for r in (am, do):
        assert 0 < r['confianza'] <= 1
        assert r['margen_modo'] >= 0


def test_el_silencio_no_tiene_tonalidad():
    assert t.tonalidad_del_audio(np.zeros(SR * 5, dtype=np.float32), SR) is None
    assert t.croma_log_de_piezas([]) is None
    assert t.croma_log_de_piezas([None, np.zeros((252, 0))]) is None


def test_la_cqt_por_piezas_da_lo_mismo_que_entera():
    """El de trozos no junta la CQT del tema (sería el doble de memoria): la
    comprime pieza a pieza con el máximo de todas. Cada frame va por su
    cuenta, así que tiene que dar lo mismo que la CQT entera."""
    rnd = np.random.RandomState(5)
    C = (rnd.rand(252, 300) ** 4).astype(np.float32)
    entera = t.croma_log(C)
    por_piezas = t.croma_log_de_piezas([C[:, :70], C[:, 70:210], C[:, 210:]])
    assert entera.shape == por_piezas.shape == (12, 300)
    assert np.allclose(entera, por_piezas, atol=1e-6)


def test_el_maximo_es_el_del_tema_entero():
    """Con el máximo de cada trozo, un trozo flojo (una intro) se comprimiría
    contra sí mismo y pesaría como el drop."""
    rnd = np.random.RandomState(6)
    flojo = (rnd.rand(252, 50) * 0.01).astype(np.float32)
    fuerte = (rnd.rand(252, 50)).astype(np.float32)
    juntas = t.croma_log_de_piezas([flojo, fuerte])
    solo_flojo = t.croma_log_de_piezas([flojo])
    # El trozo flojo sale distinto comprimido con el máximo del tema que con
    # el suyo.
    assert not np.allclose(juntas[:, :50], solo_flojo, atol=1e-3)


def test_los_dos_caminos_dan_la_misma_tonalidad_y_el_mismo_croma():
    import main  # noqa: F401  (el de trozos importa de main)
    from chunked_analyzer import ChunkedAudioAnalyzer
    y = _acorde(LA_MENOR, dur=70.0)
    ruta = tempfile.mktemp(suffix='.wav')
    soundfile.write(ruta, y, SR)
    try:
        trozos = ChunkedAudioAnalyzer(chunk_duration=20, chunk_overlap=5,
                                      sample_rate=SR).full_analysis(ruta)
    finally:
        os.remove(ruta)
    corto = t.tonalidad_del_audio(y, SR)
    assert (trozos['key'], trozos['camelot']) == (corto['key'], corto['camelot']) == ('Am', '8A')
    assert trozos['tonalidad_metodo'] == corto['metodo'] == t.METODO
    a, b = np.array(trozos['croma']), np.array(corto['croma'])
    assert np.abs(a - b).max() < 0.005


def test_una_sola_cuenta_en_los_tres_sitios():
    """El corto (`analyze_audio`), /identify y el de trozos llaman a la misma
    función, y no queda ningún Krumhansl suelto calculando una tonalidad (los
    números de sus perfiles solo viven en `tonalidad.PERFILES`, para la
    sombra)."""
    with open('main.py', encoding='utf-8') as fh:
        main_src = fh.read()
    with open('chunked_analyzer.py', encoding='utf-8') as fh:
        trozos_src = fh.read()
    assert main_src.count('tonalidad_del_audio(') == 2      # corto e /identify
    assert 'tonalidad_del_croma_log(croma_log_de_piezas(' in trozos_src
    for src in (main_src, trozos_src):
        assert '6.35, 2.23, 3.48' not in src
        assert 'chroma_cqt(' not in src


def test_el_analisis_dice_con_que_metodo_salio():
    """La sombra y la puesta al día separan el croma de antes (lineal) del de
    hoy por este campo. Va en los dos `AnalysisResult` (CLAUDE.md: el que
    devuelve /analyze es el de main.py)."""
    import main
    import models
    assert 'tonalidad_metodo' in main.AnalysisResult.model_fields
    assert 'tonalidad_metodo' in models.AnalysisResult.model_fields
    with open('main.py', encoding='utf-8') as fh:
        src = fh.read()
    assert 'tonalidad_metodo=tonalidad_metodo,' in src            # el corto
    assert "tonalidad_metodo=result.get('tonalidad_metodo')," in src  # trozos
