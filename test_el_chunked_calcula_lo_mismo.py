"""El análisis por trozos calcula lo mismo que el corto (2026-10-06).

`/analyze` analiza en trozos de 60 s todo lo que pasa de 4 minutos en Render,
o sea casi cualquier tema de club, y entero lo demás (y TODO en el motor
local). Hasta ese día cada camino hacía sus cuentas, y el mismo tema salía
distinto según durara 3:59 o 4:01. Medido con un tema sintético de 5 minutos:

    campo            corto              por trozos (antes)
    género           Minimal Techno     Electronic        (fijo)
    groove / swing   0,057 / 1,000      0,5 / 0,5         (fijos)
    graves pesados   no                 sí                (otra regla)
    tipo             opener             warmup            (sin el espectral)
    doble/mitad      sí                 no
    BPM              128,00             128,01            (deriva al coser)

Hoy los dos pasan por `rasgos_del_tema`, y desde el 2026-10-07 también la
energía (antes, ventanas de 2 s en trozos: un nivel más arriba). Lo que NO
es igual todavía, a propósito, es la tonalidad y la estructura: ver el
docstring de ese módulo.

    pytest test_el_chunked_calcula_lo_mismo.py -v
"""

import os
import tempfile
import uuid

import librosa
import numpy as np
import pytest
import soundfile

import main
from audio_helpers import beat_track_seguro
from chunked_analyzer import ChunkedAudioAnalyzer
from rasgos_del_tema import tempo_y_beats, tempograma

SR = 44100


def _tema(bpm=128.0, dur=75.0, semilla=3):
    """Bombo con caída de tono, charles a contratiempo, bajo y un acorde."""
    rnd = np.random.RandomState(semilla)
    n = int(SR * dur)
    y = np.zeros(n, dtype=np.float32)
    iv = 60.0 / bpm
    t = np.arange(int(0.12 * SR)) / SR
    bombo = (np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t)
             * np.exp(-t * 18)).astype(np.float32)
    k, fase = 0, 0.2
    while fase + k * iv < dur - 0.2:
        s = fase + k * iv
        i = int(s * SR)
        y[i:i + len(bombo)] += 0.9 * bombo
        j = int((s + iv / 2) * SR)
        m = max(0, min(n - j, int(0.03 * SR)))
        y[j:j + m] += (0.12 * rnd.randn(m)
                       * np.exp(-np.arange(m) / SR * 150)).astype(np.float32)
        k += 1
    tt = np.arange(n) / SR
    y += (0.18 * np.sin(2 * np.pi * 55 * tt)
          + 0.05 * (np.sin(2 * np.pi * 220 * tt)
                    + np.sin(2 * np.pi * 277.2 * tt))).astype(np.float32)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.8


@pytest.fixture(scope='module')
def los_dos():
    """El MISMO fichero por el camino corto y por trozos (de 20 s, para que
    haya varios en un minuto y cuarto)."""
    ruta = tempfile.mktemp(suffix='.wav')
    soundfile.write(ruta, _tema(), SR)
    viejos = (main.ARTWORK_ENABLED, main.AUDD_AUTO_ENABLED,
              main.GENRE_DETECTOR_ENABLED, main.CHUNKED_ANALYZER_ENABLED)
    main.ARTWORK_ENABLED = main.AUDD_AUTO_ENABLED = False
    main.GENRE_DETECTOR_ENABLED = main.CHUNKED_ANALYZER_ENABLED = False
    try:
        corto = main.analyze_audio(ruta, fingerprint=uuid.uuid4().hex,
                                   original_filename='x.wav')
        trozos = ChunkedAudioAnalyzer(chunk_duration=20, chunk_overlap=5,
                                      sample_rate=SR).full_analysis(ruta)
    finally:
        (main.ARTWORK_ENABLED, main.AUDD_AUTO_ENABLED,
         main.GENRE_DETECTOR_ENABLED, main.CHUNKED_ANALYZER_ENABLED) = viejos
        os.remove(ruta)
    return corto, trozos


def test_EL_CASO_bpm_rejilla_y_groove_iguales(los_dos):
    corto, trozos = los_dos
    assert trozos['bpm'] == corto.bpm == 128.0
    assert trozos['bpm_confidence'] == pytest.approx(corto.bpm_confidence, abs=1e-3)
    assert trozos['groove_score'] == pytest.approx(corto.groove_score, abs=1e-3)
    assert trozos['swing_factor'] == pytest.approx(corto.swing_factor, abs=1e-3)
    assert trozos['beat_interval'] == pytest.approx(corto.beat_interval, abs=1e-5)
    # Con la deriva de antes al coser (3,4 ms por trozo) eran varios ms.
    assert trozos['first_beat'] == pytest.approx(corto.first_beat, abs=0.002)


def test_lo_espectral_igual(los_dos):
    corto, trozos = los_dos
    assert trozos['has_heavy_bass'] == corto.has_heavy_bass
    assert trozos['has_pads'] == corto.has_pads
    assert trozos['percussion_density'] == pytest.approx(
        corto.percussion_density, abs=0.005)
    assert trozos['genre'] != 'Electronic' or corto.genre == 'Electronic', \
        'el género espectral, no el «Electronic» fijo de antes'


def test_las_voces_no_se_miden_en_ninguno(los_dos):
    """El detector que había decía «con voces» a cualquier tema con agudos:
    dos de cinco instrumentales sintéticos, sin voz, salían con voces."""
    corto, trozos = los_dos
    assert corto.has_vocals is False and trozos['has_vocals'] is False
    for f in ('main.py', 'audio_helpers.py', 'chunked_analyzer.py'):
        assert 'def detect_vocals_improved' not in open(f, encoding='utf-8').read()


def test_el_tempo_cosido_es_el_de_beat_track():
    """Con un solo trozo, `tempo_y_beats` da lo mismo que `beat_track(y)`:
    es la forma de tener el tempo del tema entero sin su tempograma entero
    en memoria."""
    y = _tema(dur=30.0)
    esperado_tempo, esperado_beats = beat_track_seguro(y, SR)
    med = librosa.onset.onset_strength(y=y, sr=SR, aggregate=np.median)
    tg = tempograma(med, SR)
    tempo, beats = tempo_y_beats(med, tg.sum(axis=1), tg.shape[1], SR)
    assert tempo == pytest.approx(float(np.atleast_1d(esperado_tempo)[0]))
    assert np.array_equal(beats, esperado_beats)


def test_el_bpm_de_las_etiquetas_manda_tambien_en_trozos():
    ruta = tempfile.mktemp(suffix='.wav')
    soundfile.write(ruta, _tema(dur=50.0), SR)
    try:
        r = ChunkedAudioAnalyzer(chunk_duration=20, chunk_overlap=5,
                                 sample_rate=SR).full_analysis(ruta, bpm_etiqueta=127.0)
    finally:
        os.remove(ruta)
    assert (r['bpm'], r['bpm_source']) == (127.0, 'id3'), \
        'ni doble/mitad ni afinado con la rejilla'


def test_los_trozos_empiezan_en_un_frame_exacto():
    """Con 55 s de paso cada trozo empezaba en 4737,3 frames y se perdían
    3,4 ms por trozo al coser. El paso ahora es un número entero de frames."""
    src = open('chunked_analyzer.py', encoding='utf-8').read()
    assert 'paso_muestras = int(round(' in src and ') * HOP' in src
    for fuera in ('def fuse_bpm_results', 'def analyze_chunk_bpm',
                  'def analyze_chunk_spectral', 'def _classify_track_type',
                  'def calculate_beat_grid'):
        assert fuera not in src, f'{fuera}: los dos caminos usan rasgos_del_tema'


def test_el_croma_se_guarda_en_los_dos_y_es_el_mismo(los_dos):
    """El croma medio del tema va en el análisis por los dos caminos, para
    medir otros perfiles de tonalidad sin el audio (`tonalidad.py`). El tema
    sintético lleva La de bajo y La-Do# de acorde: el pico, en La."""
    corto, trozos = los_dos
    for croma in (corto.croma, trozos['croma']):
        assert len(croma) == 12
        assert abs(sum(croma) - 1) < 1e-3
        assert int(np.argmax(croma)) == 9, croma
    a, b = np.array(corto.croma), np.array(trozos['croma'])
    assert a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.95


def test_la_energia_es_la_misma_en_los_dos(los_dos):
    """El de trozos medía la energía en ventanas de 2 s y el corto en las de
    46 ms de librosa. La media de RMS sube con la ventana, así que el mismo
    audio salía más alto por trozos: en este tema, 0,144 frente a 0,132, y en
    uno de 5 minutos nivel 6 frente a 5. Hoy los dos cosen el mismo RMS y
    pasan por `energia_del_tema`."""
    corto, trozos = los_dos
    assert trozos['energy_raw'] == pytest.approx(corto.energy_raw, rel=1e-4)
    assert trozos['energy_dj'] == corto.energy_dj
    assert trozos['energy_normalized'] == pytest.approx(corto.energy_normalized)
    assert trozos['mix_energy_start'] == pytest.approx(corto.mix_energy_start, rel=1e-4)
    assert trozos['mix_energy_end'] == pytest.approx(corto.mix_energy_end, rel=1e-4)


def test_la_escala_de_energia_esta_en_un_solo_sitio():
    """Había tres copias de la escala (el corto, el de trozos y el
    reanálisis), y la del corto y la de trozos ya se habían separado una vez
    en el trato de un NaN."""
    for f in ('main.py', 'chunked_analyzer.py'):
        src = open(f, encoding='utf-8').read()
        assert '0.42 - 0.02' not in src and '** 0.55' not in src, f
    src = open('chunked_analyzer.py', encoding='utf-8').read()
    assert 'energia_del_tema(f[\'rms\'], sr)' in src
