"""La guarda del `beat_track` que revienta estaba en UN sitio de cuatro.

`librosa.beat.beat_track` con `trim=True` (su defecto) recorta los beats
debiles de los bordes, y cuando NINGUNO pasa el umbral hace
`beats[valid.min():valid.max()]` sobre un array vacio y lanza

    ValueError: zero-size array to reduction operation minimum
                which has no identity

Pasa con audio sin percusion, casi silencio o muy corto — o sea con intros
ambientales, con outros y con cualquier chunk mudo de un track largo.

El 2026-08-19 dos usuarios lo comieron en `/analyze` y se puso el reintento sin
trim… solo alli. Los otros tres sitios que llaman a `beat_track` seguian
igual, y en ellos el fallo NO se veia, que es peor que un error:

  · `/identify`: el `except Exception` del re-analisis se tragaba BPM, key y
    energia ENTEROS por no poder recortar unos beats.
  · `ChunkedAudioAnalyzer.analyze_chunk_bpm`: devolvia **120 BPM de default**.
    Un chunk sin percusion metia un numero inventado en el BPM agregado del
    track largo, sin dejar rastro en ningun log.

Hoy la regla vive en `audio_helpers.beat_track_seguro` y estos tests atan que
siga usandose en los tres.

    pytest test_beat_track_seguro.py -v
"""

import os

import numpy as np
import pytest

from audio_helpers import beat_track_seguro

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _src(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


# ============================================================================
# EL COMPORTAMIENTO
# ============================================================================

def test_silencio_puro_no_revienta():
    """El caso exacto que tumbaba el analisis: audio sin un solo beat."""
    sr = 22050
    y = np.zeros(sr * 3, dtype=np.float32)
    tempo, beats = beat_track_seguro(y, sr)
    assert beats is not None


def test_ruido_sin_percusion_no_revienta():
    sr = 22050
    rng = np.random.default_rng(7)
    y = (rng.standard_normal(sr * 3) * 0.001).astype(np.float32)
    tempo, beats = beat_track_seguro(y, sr)
    assert beats is not None


def test_audio_con_pulso_sigue_midiendo_el_tempo():
    """Reintentar sin trim NO puede cambiar el resultado del caso normal: el
    recorte solo quita beats de los BORDES."""
    sr = 22050
    dur = 8
    y = np.zeros(sr * dur, dtype=np.float32)
    # Un click cada 0.5 s = 120 BPM.
    for i in range(0, dur * 2):
        pos = int(i * 0.5 * sr)
        y[pos:pos + 400] = 1.0
    tempo, beats = beat_track_seguro(y, sr)
    t = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    assert 100 < t < 140, f'tempo fuera de rango: {t}'
    assert len(beats) > 4


def test_si_le_pasas_trim_False_no_reintenta_en_bucle():
    """Con `trim=False` ya no hay nada que reintentar: si aun asi falla, el
    error sube. Un reintento infinito seria peor que el fallo."""
    sr = 22050
    y = np.zeros(sr * 2, dtype=np.float32)
    # No debe colgarse ni recursar; el resultado da igual.
    beat_track_seguro(y, sr, trim=False)


def test_acepta_los_kwargs_de_librosa():
    """`artwork_and_cuepoints` llama con `bpm=`. Si el helper no los pasara,
    cambiar a el degradaria la rejilla en silencio."""
    sr = 22050
    dur = 6
    y = np.zeros(sr * dur, dtype=np.float32)
    for i in range(0, dur * 2):
        y[int(i * 0.5 * sr):int(i * 0.5 * sr) + 400] = 1.0
    tempo, beats = beat_track_seguro(y, sr, bpm=120)
    assert beats is not None


# ============================================================================
# EL CABLEADO: QUE NO SE QUEDE OTRA VEZ EN UN SOLO SITIO
# ============================================================================

@pytest.mark.parametrize('fichero', ['main.py', 'chunked_analyzer.py'])
def test_nadie_llama_a_beat_track_a_pelo(fichero):
    src = _src(fichero)
    # El fallback de `chunked_analyzer` para cuando no hay `audio_helpers` SI
    # llama a librosa directamente, y es correcto: define el propio helper.
    lineas = [
        ln for ln in src.split('\n')
        if 'librosa.beat.beat_track(' in ln
        and 'def beat_track_seguro' not in ln
    ]
    permitidas = [ln for ln in lineas if 'return librosa.beat.beat_track' in ln]
    sospechosas = [ln for ln in lineas if ln not in permitidas]
    assert sospechosas == [], (
        f'{fichero} llama a beat_track sin la guarda: {sospechosas}')


def test_los_tres_sitios_usan_el_helper():
    assert 'beat_track_seguro(y, sr)' in _src('main.py')
    assert 'beat_track_seguro(y_full, sr_full)' in _src('main.py')
    assert 'beat_track_seguro(y, sr)' in _src('chunked_analyzer.py')


def test_el_default_de_120_del_chunk_sigue_documentado_como_ultimo_recurso():
    """El 120 no se quita —un chunk puede fallar por otras razones— pero deja
    de ser lo que pasa cuando el audio simplemente no tiene percusion."""
    src = _src('chunked_analyzer.py')
    assert "'bpm': 120.0" in src
    i = src.index('beat_track_seguro(y, sr)')
    assert '120 BPM' in src[max(0, i - 600):i]
